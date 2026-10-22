"""A Boat sandbox behind Pydantic AI's `WorkspaceBackend` protocol, on Boat's official Python SDK (`boat-sdk`).

The SDK is synchronous and holds no event loop state, so every call runs in a worker thread through
`anyio.to_thread.run_sync`, and one `boat_sdk.BoatApi` per API key serves every backend in the
process, on asyncio or Trio. Work whose outcome must be recorded, such as creating a sandbox or
starting a command, records it from its worker thread, so a caller cancelled meanwhile loses nothing.

Every `run()` command runs as a detached Boat command: it has a process ID that a timeout or a
cancellation can stop with the SDK's `stop_command`, no server-side time limit, and its output is
followed with the SDK's `follow_command` until its own process exits. Internal helpers, such as
`pwd -P` and the shell file operations, run as synchronous commands: one round trip each.

Boat facts this module pins, verified 2026-10-21 against https://docs.boat.dev/api/v1 and the live
API. The `boat_live` tier (`tests/harness/boat_sandbox/test_boat_live.py`) exercises each one not
marked otherwise:

- A synchronous command returns at most the last 8 MiB of each stream.
- The file API moves at most 5 MiB in one call; a larger file needs a signed transfer URL.
- The file API reaches only paths that resolve under `/home/user` and `/tmp`.
- `noEnv` keeps the account's GitHub and Boat tokens out of commands.
- `ttlSeconds: 0` means Boat's default auto-stop of an hour, and `null` disables it (not exercised).
- Command activity does not move the auto-stop deadline; an update with `ttlSeconds` sets it to that
  many seconds from now.
- Deleting a sandbox needs its ID repeated as the delete confirmation.
- A stopped command gets about 5 seconds after SIGTERM before SIGKILL (not exercised).
- Commands run under `bash -lc`, whose `~/.bash_logout` prints to the output unless the command `exec`s.
"""

from __future__ import annotations

import base64
import errno
import logging
import os
import posixpath
import shlex
import threading
import uuid
from collections.abc import Awaitable, Callable, Generator, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cache, partial
from time import monotonic, sleep
from typing import TYPE_CHECKING, Literal, NoReturn, TypeVar

import anyio
import anyio.lowlevel
import anyio.to_thread
from pydantic import TypeAdapter, ValidationError

from pydantic_ai.workspaces import (
    CommandResult,
    FileEntry,
    SupportsCommands,
    SupportsFilesystem,
    SupportsRealpath,
    WorkspaceBackend,
    WorkspaceError,
    WorkspaceOutputLimitError,
    WorkspaceRef,
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
)
from pydantic_ai.workspaces.workspace import _ShellFilesystem  # pyright: ignore[reportPrivateUsage]
from pydantic_ai_harness._workspace_provider import (
    absolute_path,
    check_env,
    check_timeout,
    command_argv,
    command_deadline,
    safe_credential_reason,
)

if TYPE_CHECKING:
    from pydantic_ai.workspaces import WorkspaceCommand

__all__ = ('BoatSandboxBackend',)

try:
    import boat_sdk as boat
    import urllib3
    from boat_sdk import boat_helpers
except ImportError as error:
    raise ImportError('Install `pydantic-ai-harness[boat]` to use BoatSandbox.') from error

MachineType = Literal['small', 'default', 'large', 'xlarge']
"""Boat's machine sizes; see https://docs.boat.dev/machines."""

_AUTH_MESSAGE = 'Boat rejected the credentials. Set BOAT_API_KEY, or pass a configured `boat_sdk.BoatApi` as `client=`.'
_MISSING_CREDENTIALS_MESSAGE = (
    'No Boat credentials found. Set BOAT_API_KEY, or pass a configured `boat_sdk.BoatApi` as `client=`.'
)
_OUTPUT_LIMIT_MESSAGE = (
    'Boat command output exceeded the 8 MiB limit per stream; redirect it to a file and read part of it instead'
)
# Seconds to connect, and to wait for each read of an answer: the SDK sets no timeout of its own.
_REQUEST_TIMEOUT = (30.0, 30.0)
# Bound sandbox creation and attachment so a wedged control plane cannot hang acquisition.
_CREATE_TIMEOUT = 120.0
# How long a stop waits for Boat's answer: Boat sends SIGTERM, then SIGKILL about 5 seconds later, and
# answers once the command exits, about 5.4 seconds after the stop for a command that ignores SIGTERM.
_KILL_TIMEOUT = 15.0
# Internal commands (`pwd -P`, the shell file operations) run synchronously; Boat stops one after this.
_HELPER_TIMEOUT = 30
# Between attempts to create a sandbox whose earlier request may have been lost.
_RETRY_DELAY = 1.0
# Between checks of a sandbox that is starting or changing state.
_READY_POLL = 0.5
# The most partial output a timeout reports per stream.
_PARTIAL_OUTPUT_LIMIT = 4096
# The most of each stream a command may print: its whole output is held in memory until it exits.
_MAX_OUTPUT_BYTES = 8 * 1024 * 1024
_OUTPUT_PREVIEW_BYTES = 64 * 1024
# Boat's file API moves at most 5 MiB in one call; a larger file goes through a signed transfer URL.
_MAX_INLINE_FILE_BYTES = 5 * 1024 * 1024
# The most time between extensions of a sandbox's auto-stop while operations run; half the auto-stop when shorter.
_KEEP_ALIVE_INTERVAL = 60

_STOPPED_STATES = frozenset({'archiving', 'archived'})
_FAILED_STATES = frozenset({'error', 'cancelled'})
# Boat's 409 codes for a resume refused only while the sandbox is between states.
_TRANSITIONAL_CODES = frozenset({'sandbox_resuming', 'sandbox_stopping', 'sandbox_starting'})
# The POSIX error a file call failed with, from the start of Boat's message; a write below a file
# fails creating the file as a directory.
_PATH_ERRNOS = {
    'ENOENT': errno.ENOENT,
    'ENOTDIR': errno.ENOTDIR,
    'EISDIR': errno.EISDIR,
    'EEXIST': errno.ENOTDIR,
    'EACCES': errno.EACCES,
    'ELOOP': errno.ELOOP,
}
_ANSWER = TypeAdapter(dict[str, object])

_T = TypeVar('_T')
_Refusal = Literal['outside', 'not_a_file']

logger = logging.getLogger(__name__)


async def _in_thread(call: Callable[[], _T], *, abandon: bool = False) -> _T:
    """Run a blocking SDK call in a worker thread.

    With `abandon`, a cancelled caller returns at once and leaves the call to end on its own; without
    it, a cancellation waits for the call. Either way the call records its own outcome.
    """
    return await anyio.to_thread.run_sync(call, abandon_on_cancel=abandon)


def _default_client() -> boat.BoatApi:
    """The SDK client for `BOAT_API_KEY`."""
    key = os.environ.get('BOAT_API_KEY', '').strip()
    if not key:
        raise WorkspaceUnavailableError(_MISSING_CREDENTIALS_MESSAGE)
    return _client_for(key)


@cache
def _client_for(key: str) -> boat.BoatApi:
    # One per key for the process: the SDK holds no event loop state, and its connection pool is thread-safe.
    return boat.BoatApi(boat.ApiClient(boat.Configuration(access_token=key)))


@dataclass(frozen=True)
class _Sandbox:
    """A Boat sandbox reached through one SDK client."""

    client: boat.BoatApi
    id: str


def _details(error: boat.ApiException) -> tuple[int, str, str]:
    """Boat's HTTP status, error code and message, read from the answer the SDK keeps unparsed."""
    # The SDK leaves both untyped.
    status: object = getattr(error, 'status', None)
    body: object = getattr(error, 'body', None)
    try:
        answer = _ANSWER.validate_json(body) if isinstance(body, str) and body else {}
    except ValidationError:
        answer = {}
    # Boat's API names the error `code`; a signed file transfer names it `error`.
    code, message = answer.get('code', answer.get('error')), answer.get('message')
    return (
        status if isinstance(status, int) else 0,
        code if isinstance(code, str) else '',
        message if isinstance(message, str) else '',
    )


def _describe(error: Exception) -> str:
    if isinstance(error, boat.ApiException):
        status, code, message = _details(error)
        return f'Boat answered HTTP {status} ({code}): {message}'
    return repr(error)


async def destroy_sandbox(client: boat.BoatApi | None, sandbox_id: str) -> None:
    """Delete a sandbox by ID without resuming it, including one Boat has stopped.

    A sandbox that no longer exists returns quietly; rejected credentials raise
    `WorkspaceUnavailableError`, like any other operation. Without `client`, the one for
    `BOAT_API_KEY` is used.
    """
    if not sandbox_id.strip():
        raise ValueError('Boat workspace ref id cannot be empty')

    def delete() -> None:
        api = client or _default_client()
        with _mapped_errors(f'Could not delete Boat sandbox {sandbox_id!r}'):
            try:
                api.delete_sandbox(
                    x_ascii_confirm_delete=sandbox_id, sandbox_id=sandbox_id, _request_timeout=_REQUEST_TIMEOUT
                )
            except boat.ApiException as error:
                if _details(error)[0] != 404:
                    raise

    await _in_thread(delete)


def _not_found_message(sandbox_id: str) -> str:
    return (
        f'The Boat sandbox {sandbox_id!r} was not found (it was deleted, or never existed in this Boat '
        "account). Pass `workspace='new'` to start a fresh sandbox."
    )


def _unavailable_message(sandbox_id: str) -> str:
    return (
        f'The Boat sandbox {sandbox_id!r} no longer exists (it was deleted). '
        "Pass `workspace='new'` to start a fresh sandbox."
    )


def _stopped_message(sandbox_id: str) -> str:
    return (
        f'The Boat sandbox {sandbox_id!r} was stopped; a later run that attaches to it resumes it. '
        "Pass `workspace='new'` to start a fresh sandbox."
    )


def _failed_message(sandbox_id: str, state: str) -> str:
    return (
        f'The Boat sandbox {sandbox_id!r} is in the {state!r} state and cannot be used. '
        "Pass `workspace='new'` to start a fresh sandbox."
    )


def _missing_cwd(cwd: str | None, sandbox_id: str) -> WorkspaceError:
    return WorkspaceError(
        f'working_dir {cwd!r} does not exist in Boat sandbox {sandbox_id}. '
        'Create it there, or pass a working_dir that exists.'
    )


def _translate(error: Exception, context: str, *, sandbox_id: str | None = None, path: str | None = None) -> Exception:
    """Map a Boat API error onto the failures the workspace protocol promises.

    Returns `error` itself for what must propagate unchanged: network failures, rate limits, server
    errors, and anything that is not a `boat_sdk.ApiException`, which durable engines retry as transient.
    """
    if not isinstance(error, boat.ApiException):
        return error
    status, code, message = _details(error)
    if status == 401 or (status == 403 and code == 'forbidden'):
        return WorkspaceUnavailableError(f'{safe_credential_reason(error)}. {_AUTH_MESSAGE}')
    if status in (402, 403) or (status == 429 and code == 'trial_compute_limit_reached'):
        # Billing and plan refusals, which no retry fixes.
        return WorkspaceUnavailableError(f'{context}: {message}')
    if sandbox_id is not None and status == 404 and code == 'not_found':
        return WorkspaceUnavailableError(_unavailable_message(sandbox_id))
    if sandbox_id is not None and status == 409 and code in ('machine_not_running', 'sandbox_not_ready'):
        return WorkspaceUnavailableError(_stopped_message(sandbox_id))
    if path is not None and status == 400:
        number = _PATH_ERRNOS.get(message.partition(':')[0])
        if number is not None:
            return OSError(number, f'{os.strerror(number)} in the Boat sandbox', path)
    if 400 <= status < 500 and status != 429:
        return WorkspaceError(f'{context}: {_describe(error)}')
    return error


def _state_error(sandbox_id: str, state: str) -> WorkspaceUnavailableError | None:
    if state in _STOPPED_STATES:
        return WorkspaceUnavailableError(_stopped_message(sandbox_id))
    if state in _FAILED_STATES:
        return WorkspaceUnavailableError(_failed_message(sandbox_id, state))
    return None


def _probe(sandbox: _Sandbox) -> WorkspaceUnavailableError | None:
    """`WorkspaceUnavailableError` if Boat deleted or stopped `sandbox`; an inconclusive lookup counts as running."""
    try:
        state = sandbox.client.get(sandbox.id, _request_timeout=_REQUEST_TIMEOUT).sandbox.state
    except boat.ApiException as error:
        return WorkspaceUnavailableError(_unavailable_message(sandbox.id)) if _details(error)[0] == 404 else None
    except Exception:
        return None
    return _state_error(sandbox.id, state)


@contextmanager
def _mapped_errors(context: str, *, sandbox: _Sandbox | None = None, path: str | None = None) -> Generator[None]:
    """Raise the protocol's typed failure for a Boat API error, and let the backend's own failures through.

    With `sandbox`, a failure that would propagate unchanged is followed by one lookup, so a sandbox
    that was deleted or stopped meanwhile ends the run instead of being retried.
    """
    try:
        yield
    except (WorkspaceError, OSError):
        raise
    except Exception as error:
        translated = _translate(error, context, sandbox_id=None if sandbox is None else sandbox.id, path=path)
        if translated is not error:
            raise translated from error
        if sandbox is not None and (gone := _probe(sandbox)) is not None:
            raise gone from error
        raise


def _never_sent(error: Exception) -> bool:
    """Whether the SDK gave up connecting, so the request never reached Boat."""
    return isinstance(error, urllib3.exceptions.MaxRetryError) and isinstance(
        error.reason, urllib3.exceptions.ConnectTimeoutError
    )


def _marker() -> str:
    return f'pydantic-ai-{uuid.uuid4().hex}'


def _command_line(argv: Sequence[str], *, marker: str, cwd: str | None, env: Mapping[str, str]) -> str:
    """The Boat command string that runs `argv` in `cwd` with stdin at EOF.

    Boat runs the string with `bash -lc`. `exec` keeps that login shell from running
    `~/.bash_logout`, which prints to the output. The `sh` it becomes exits with the program's status,
    127 for a missing program, and 125 after printing `<marker>-cwd` when `cwd` is missing.
    """
    if env:
        # Inside the `sh`, since a `sh` such as dash drops names that are not shell identifiers.
        # `--` ends `env`'s options, so a name starting with `-` is not read as one.
        argv = ['env', '--', *(f'{name}={value}' for name, value in env.items()), *argv]
    guard = f'cd -- {shlex.quote(cwd)} || {{ printf %s {marker}-cwd; exit 125; }}; ' if cwd is not None else ''
    # `sh` as `$0`, which the shell's own messages, such as a missing program's, start with.
    return shlex.join(['exec', 'sh', '-c', f'{guard}"$@" </dev/null; exit $?', 'sh', *argv])


def _exec(sandbox: _Sandbox, argv: Sequence[str], *, cwd: str | None) -> CommandResult:
    """Run a short internal command synchronously: one round trip, bounded by Boat. Runs in a worker thread."""
    marker = _marker()
    line = _command_line(argv, marker=marker, cwd=cwd, env={})
    with _mapped_errors('Could not run a command in the sandbox', sandbox=sandbox):
        answer = sandbox.client.command(
            sandbox.id,
            boat.CommandRequest(command=line, timeoutSeconds=_HELPER_TIMEOUT),
            # Boat answers within 10 seconds of its own timeout, even when a child keeps the output open.
            _request_timeout=(_REQUEST_TIMEOUT[0], _HELPER_TIMEOUT + 15.0),
        ).actual_instance
    if not isinstance(answer, boat.CommandResponse) or (exit_code := answer.exit_code) is None:
        raise WorkspaceError(f'Boat did not report how an internal command exited in Boat sandbox {sandbox.id}.')
    if answer.timed_out:
        raise WorkspaceError(
            f'An internal command did not finish within {_HELPER_TIMEOUT}s in Boat sandbox {sandbox.id}.'
        )
    if answer.stdout_truncated or answer.stderr_truncated:
        raise WorkspaceOutputLimitError(_OUTPUT_LIMIT_MESSAGE, limit=_MAX_OUTPUT_BYTES)
    if exit_code == 125 and answer.stdout == f'{marker}-cwd':
        raise _missing_cwd(cwd, sandbox.id)
    return CommandResult(exit_code=exit_code, stdout=answer.stdout, stderr=answer.stderr)


def _frames(sandbox: _Sandbox, process_id: int) -> Iterator[Mapping[str, object]]:
    """A detached command's output frames, then its `exit` or `error` frame; the SDK reconnects a dropped stream."""
    # The SDK types the frames as a bare `dict`.
    return boat_helpers.follow_command(sandbox.client, sandbox.id, process_id)  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]


def _tail(chunks: list[str]) -> str:
    # Cut by bytes; a character split at the cut is dropped.
    return ''.join(chunks).encode()[-_PARTIAL_OUTPUT_LIMIT:].decode(errors='ignore')


def _head(chunks: list[str]) -> str:
    return ''.join(chunks).encode()[:_OUTPUT_PREVIEW_BYTES].decode(errors='ignore')


class _SyncShell:
    """Synchronous Boat commands as the runner core's shell filesystem drives."""

    def __init__(self, sandbox: _Sandbox, working_dir: Callable[[], Awaitable[str]]) -> None:
        self._sandbox = sandbox
        self._working_dir = working_dir

    async def working_dir(self) -> str:
        return await self._working_dir()

    async def run(
        self,
        command: WorkspaceCommand,
        *,
        shell: bool = False,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        # The shell filesystem passes absolute paths and no `env` or `timeout`.
        return await _in_thread(partial(_exec, self._sandbox, command_argv(command, shell), cwd=None), abandon=True)


class _BoatProcess:
    """A detached Boat command, followed until its own process exits. Its methods run in worker threads."""

    def __init__(self, sandbox: _Sandbox, *, cwd: str | None, keep_alive: Callable[[threading.Event], None]) -> None:
        self._sandbox = sandbox
        self._cwd = cwd
        # Extends Boat's auto-stop while the command runs, until the event is set.
        self._keep_alive = keep_alive
        self.marker = _marker()
        self._pid: int | None = None
        # Set once the start was answered or failed, so a stop knows what there is to stop.
        self._started = threading.Event()
        # Nothing is left to stop: the command exited, or Boat refused to start it.
        self._finished = False
        # Set once no one waits for the command any more, which ends the extensions of the auto-stop.
        self._done = threading.Event()
        self._output: dict[str, list[str]] = {'stdout': [], 'stderr': []}
        self._sizes = {'stdout': 0, 'stderr': 0}
        # Set when the stop finds the sandbox deleted or stopped, which a timeout then reports instead.
        self.gone: WorkspaceUnavailableError | None = None

    def start(self, line: str) -> int:
        """Start the command detached, and record its process ID even if the caller was cancelled meanwhile."""
        try:
            self._pid = self._submit(line)
            return self._pid
        finally:
            self._started.set()

    def _submit(self, line: str) -> int:
        sandbox = self._sandbox
        try:
            with _mapped_errors('Could not start the command', sandbox=sandbox):
                answer = sandbox.client.command(
                    sandbox.id, boat.CommandRequest(command=line, detached=True), _request_timeout=_REQUEST_TIMEOUT
                ).actual_instance
        except WorkspaceError:
            # A refusal, or a sandbox found stopped or deleted: there is no command to look for.
            self._finished = True
            raise
        except Exception as error:
            if _never_sent(error) or (isinstance(error, boat.ApiException) and _details(error)[0] == 429):
                # Never sent, or refused unrun: a durable engine may retry it.
                self._finished = True
                raise
            # Not transient: a retry could run the command a second time.
            raise WorkspaceError(
                f'Boat did not confirm that the command started, so it may be running: {_describe(error)}'
            ) from error
        if not isinstance(answer, boat.CommandStartedResponse):
            self._finished = True
            raise WorkspaceError('Boat ran the command without detaching it.')
        return answer.process_id

    def follow(self, pid: int) -> CommandResult | None:
        """The command's result once its own process exits, or `None` once a stream passes the output limit."""
        # Boat does not count a running command as activity, so its auto-stop is extended meanwhile.
        threading.Thread(target=self._keep_alive, args=(self._done,), daemon=True).start()
        try:
            last = self._read_output(pid)
        finally:
            self._done.set()
        if last is None:
            return None
        exit_code = last.get('exitCode')
        if last.get('type') != 'exit' or not isinstance(exit_code, int):
            # Boat forgets a command's exit when the sandbox's agent restarts.
            raise _probe(self._sandbox) or WorkspaceError('Boat lost track of the command; its result is unknown.')
        self._finished = True
        stdout, stderr = ''.join(self._output['stdout']), ''.join(self._output['stderr'])
        if exit_code == 125 and stdout == f'{self.marker}-cwd':
            raise _missing_cwd(self._cwd, self._sandbox.id)
        return CommandResult(exit_code=exit_code, stdout=stdout, stderr=stderr)

    def _read_output(self, pid: int) -> Mapping[str, object] | None:
        """The frame that ended the command's output, or `None` once a stream passes the output limit."""
        try:
            with _mapped_errors('Could not read the command result', sandbox=self._sandbox):
                for frame in _frames(self._sandbox, pid):
                    stream, data = frame.get('type'), frame.get('data')
                    if not isinstance(stream, str) or stream not in self._output:
                        return frame
                    text = data if isinstance(data, str) else ''
                    self._output[stream].append(text)
                    self._sizes[stream] += len(text.encode())
                    if self._sizes[stream] > _MAX_OUTPUT_BYTES:
                        return None
        except WorkspaceError:
            raise
        except Exception as error:
            raise WorkspaceError(
                f'Could not read the result of a command that may still be running: {_describe(error)}'
            ) from error
        raise WorkspaceError('Boat ended the output of the command without reporting its exit.')

    def partial_output(self) -> tuple[str, str]:
        """The end of each stream so far, which a timeout reports."""
        return _tail(self._output['stdout']), _tail(self._output['stderr'])

    def output_limit_error(self) -> WorkspaceOutputLimitError:
        return WorkspaceOutputLimitError(
            _OUTPUT_LIMIT_MESSAGE,
            limit=_MAX_OUTPUT_BYTES,
            stdout=_head(self._output['stdout']),
            stderr=_head(self._output['stderr']),
        )

    def kill(self) -> None:
        """Stop the command and every process it started, unless it already exited."""
        # Even if the stop fails and the output stream keeps going, the sandbox is no longer kept up for it.
        self._done.set()
        # A start the caller stopped waiting for still records the process ID first.
        if self._started.wait(_KILL_TIMEOUT) and self._finished:
            return
        stopped = False
        try:
            if self._pid is not None:
                try:
                    self._sandbox.client.stop_command(self._sandbox.id, self._pid, _request_timeout=_KILL_TIMEOUT)
                except boat.ApiException as error:
                    status = _details(error)[0]
                    gone = _translate(error, 'Could not stop the command', sandbox_id=self._sandbox.id)
                    if isinstance(gone, WorkspaceUnavailableError) and status in (404, 409):
                        # Deleted or stopped, with every process on it; a deleted sandbox's output stream
                        # can hang until the timeout instead of saying so.
                        self.gone = gone
                    elif status != 400:
                        # 400: Boat no longer knows the command.
                        raise
                stopped = True
        finally:
            if not stopped:
                # A failure, an unconfirmed start, or a start still unanswered.
                logger.warning(
                    'Could not stop a command in Boat sandbox %s; it may still be running.', self._sandbox.id
                )


class BoatSandboxBackend(WorkspaceBackend, SupportsCommands, SupportsFilesystem, SupportsRealpath):
    """A [Boat](https://boat.dev) sandbox as a Pydantic AI [`WorkspaceBackend`][pydantic_ai.workspaces.WorkspaceBackend].

    Commands and file operations run inside a Boat sandbox, a cloud VM, so the host is never exposed.

    Building one does no I/O. The first operation creates or attaches to a sandbox. A new sandbox
    gets none of the Boat account's own credentials. The backend does not stop or delete the
    sandbox; that is the application's job. Boat's SDK is synchronous, so its calls run in worker
    threads: the backend works on asyncio and Trio, and holds nothing that needs closing.

    A command's complete output is returned once it exits. A shell string runs under `/bin/sh -c`;
    a timeout or cancellation stops the command and every process it started.

    A deleted sandbox, one Boat stopped (attaching again resumes it), rejected credentials, and an
    unknown snapshot raise `WorkspaceUnavailableError`. Path failures raise the built-in errors,
    such as `FileNotFoundError`.

    Args:
        client: A `boat_sdk.BoatApi` to create or attach with, for example one whose
            `boat_sdk.Configuration` routes through a proxy. Without one, the backend uses one built
            from `BOAT_API_KEY` on first use, shared by every backend in the process with that key.
        ref: Identity of an existing sandbox to attach to on first use.
        machine_type: Boat machine size for a newly created sandbox: `'small'`, `'default'`,
            `'large'` or `'xlarge'`; Boat's default when `None`.
        snapshot: Named Boat snapshot a newly created sandbox starts from; Boat's base image when `None`.
        auto_stop: Seconds a newly created sandbox runs without the backend extending it before
            Boat stops it (its disk is kept); `0` disables it, and `None` keeps Boat's default of an
            hour. An operation, and a running command, set it again from now at most once a minute
            (once per half of it when it is under two minutes), so a run that works in the sandbox
            more often than `auto_stop` minus that interval keeps it up, and an idle one stops at
            most `auto_stop` seconds after its last operation.
        working_dir: Absolute directory commands start in and relative paths resolve against; the
            sandbox's own default when `None`, discovered with `pwd -P` on first use.
        env: Environment variables every command gets; a command's own `env` is layered on top.
            Values travel in each command's command line, so the sandbox's process list and Boat's
            command history show them.
    """

    def __init__(
        self,
        *,
        client: boat.BoatApi | None = None,
        ref: WorkspaceRef | None = None,
        machine_type: MachineType | None = None,
        snapshot: str | None = None,
        auto_stop: int | None = None,
        working_dir: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        if ref is not None and ref.provider != 'boat':
            raise ValueError(f"unsupported workspace provider {ref.provider!r}; expected 'boat'")
        if ref is not None and not ref.id.strip():
            raise ValueError('Boat workspace ref id cannot be empty')
        check_env(env)
        self._ref = ref
        self._client = client
        self._machine_type = machine_type
        self._snapshot = snapshot
        self._auto_stop = auto_stop
        self._env = dict(env or {})
        self._working_dir = absolute_path('working_dir', working_dir)
        # Sent with every creation attempt, so a retry after a lost reply gets the sandbox it created.
        self._idempotency_key = _marker()
        # Held in the worker thread that creates or attaches, so concurrent first uses share one sandbox.
        self._lock = threading.Lock()
        self._sandbox: _Sandbox | None = None
        # `working_dir()` must return a canonical absolute path, resolved once with `pwd -P`.
        self._resolved_working_dir: str | None = None
        # A sandbox this backend created still needs its `working_dir` made.
        self._working_dir_pending = False
        # The sandbox's auto-stop in seconds, read when it is acquired (`None` when it is off), and
        # when this backend last set its deadline; an attached sandbox's is set on first use.
        self._ttl: int | None = None
        self._ttl_set_at: float | None = None
        self._keep_alive_failing = False

    @property
    def ref(self) -> WorkspaceRef | None:
        """Identity of the sandbox, or `None` before one has been created.

        The `id` is Boat's sandbox ID, such as `bx_h8mygsw2`.
        """
        return self._ref

    async def _connect(self) -> _Sandbox:
        """Create or attach to the sandbox on first use, and return it once it is ready.

        Attaching by `ref` to a sandbox that no longer exists raises `WorkspaceUnavailableError`; it
        does not create a replacement.
        """
        sandbox = await _in_thread(self._prepare)
        # A caller cancelled while the sandbox was acquired stops here, before a command it can no
        # longer wait for starts.
        await anyio.lowlevel.checkpoint_if_cancelled()
        return sandbox

    def _prepare(self) -> _Sandbox:
        """Runs in a worker thread, and records what it creates on the backend, so a cancelled caller loses nothing."""
        with self._lock:
            sandbox = self._sandbox
            if sandbox is None:
                sandbox = self._sandbox = self._acquire(self._client or _default_client())
            if self._working_dir_pending:
                self._make_working_dir(sandbox)
        self._extend(sandbox)
        return sandbox

    def _acquire(self, client: boat.BoatApi) -> _Sandbox:
        if self._ref is None:
            sandbox = _Sandbox(client, self._create(client))
        else:
            sandbox = _Sandbox(client, self._ref.id)
            self._resume_if_stopped(sandbox)
        with _mapped_errors(f'Could not attach to Boat sandbox {sandbox.id!r}', sandbox=sandbox):
            try:
                # Commands sent to a starting sandbox wait for it, but file calls do not: right after a
                # resume, Boat's file API can report an existing file as missing.
                ready = boat_helpers.wait_until_ready(
                    client, sandbox.id, timeout_seconds=_CREATE_TIMEOUT, poll_interval_seconds=_READY_POLL
                )
            except RuntimeError as error:
                raise WorkspaceUnavailableError(
                    f'The Boat sandbox {sandbox.id!r} stopped or failed while starting ({error}). '
                    "Pass `workspace='new'` to start a fresh sandbox."
                ) from error
        self._ttl = ready.ttl_seconds
        return sandbox

    def _create(self, client: boat.BoatApi) -> str:
        """Create a sandbox under an idempotency key kept across retries, so a lost reply is found rather than duplicated."""
        # Without `noEnv`, Boat gives every command the account owner's GitHub token and other logins,
        # which code an agent runs could read. Explicit `None`s keep the SDK from sending its own
        # defaults, such as a `type` over a snapshot's or an `environment` over the account's.
        body: dict[str, object] = {'noEnv': True, 'type': self._machine_type, 'from': self._snapshot}
        body.update(dict.fromkeys(('environment', 'snapshots', 'monitoring', 'failFast', 'count')))
        if self._auto_stop is not None:
            # Boat reads `0` as its default; `null` turns the auto-stop off.
            body['ttlSeconds'] = self._auto_stop or None
        try:
            request = boat.CreateSandboxRequest.model_validate(body)
        except ValidationError as error:
            # The SDK refuses a setting Boat would, such as an unknown machine size, before sending it.
            raise WorkspaceUnavailableError(f'Could not start Boat sandbox: {error}') from error
        deadline = monotonic() + _CREATE_TIMEOUT
        while True:
            requested = monotonic()
            try:
                created = client.create(
                    idempotency_key=self._idempotency_key,
                    create_sandbox_request=request,
                    _request_timeout=_REQUEST_TIMEOUT,
                )
                break
            except urllib3.exceptions.HTTPError:
                pass  # The sandbox may have been created before the reply was lost; the key finds it.
            except boat.ApiException as error:
                status, code, _ = _details(error)
                if status != 409 or code != 'idempotency_in_progress':
                    translated = _translate(error, 'Could not start Boat sandbox')
                    if translated is error:
                        raise
                    if type(translated) is WorkspaceError:
                        # Boat refused the request (an unknown snapshot, an invalid setting): no retry can fix it.
                        translated = WorkspaceUnavailableError(str(translated))
                    raise translated from error
            if monotonic() + _RETRY_DELAY >= deadline:
                # A stalled control plane is transient, not a command deadline.
                raise TimeoutError(
                    f'Boat sandbox creation did not complete within {_CREATE_TIMEOUT:g}s; '
                    'the Boat control plane may be unreachable.'
                )
            sleep(_RETRY_DELAY)
        sandbox_id = created.sandbox.id
        self._ttl_set_at = requested
        self._ref = WorkspaceRef(provider='boat', id=sandbox_id)
        self._working_dir_pending = self._working_dir is not None
        # Logged at creation, so the id can be found whatever ends the run before it is stored.
        logger.info('Created Boat sandbox %s', sandbox_id)
        return sandbox_id

    def _resume_if_stopped(self, sandbox: _Sandbox) -> None:
        """Resume a sandbox Boat stopped; a deleted or failed one raises `WorkspaceUnavailableError`."""
        deadline = monotonic() + _CREATE_TIMEOUT
        with _mapped_errors(f'Could not attach to Boat sandbox {sandbox.id!r}'):
            while True:
                try:
                    state = sandbox.client.get(sandbox.id, _request_timeout=_REQUEST_TIMEOUT).sandbox.state
                    if state in _FAILED_STATES:
                        raise WorkspaceUnavailableError(_failed_message(sandbox.id, state))
                    if state not in _STOPPED_STATES:
                        return
                    requested = monotonic()
                    sandbox.client.resume(sandbox.id, _request_timeout=_REQUEST_TIMEOUT)
                    # A resume sets the auto-stop deadline again.
                    self._ttl_set_at = requested
                    return
                except boat.ApiException as error:
                    status, code, message = _details(error)
                    if status == 404:
                        raise WorkspaceUnavailableError(_not_found_message(sandbox.id)) from error
                    if status != 409:
                        raise
                    if code not in _TRANSITIONAL_CODES:
                        # Boat cannot resume it, for example once its snapshot has expired.
                        raise WorkspaceUnavailableError(
                            f'Could not resume Boat sandbox {sandbox.id!r}: {message}'
                        ) from error
                if monotonic() + _READY_POLL >= deadline:
                    raise TimeoutError(
                        f'Resuming Boat sandbox {sandbox.id!r} did not complete within {_CREATE_TIMEOUT:g}s.'
                    )
                sleep(_READY_POLL)

    def _make_working_dir(self, sandbox: _Sandbox) -> None:
        directory = self._working_dir
        assert directory is not None
        result = _exec(sandbox, ['mkdir', '-p', '--', directory], cwd=None)
        if result.exit_code != 0:
            raise WorkspaceError(
                f'Could not create working_dir {directory!r} in Boat sandbox {sandbox.id}: {result.stderr.strip()}'
            )
        self._working_dir_pending = False

    def _extend(self, sandbox: _Sandbox) -> None:
        """Set the sandbox's auto-stop deadline again from now, at most once a minute (or per half of it, if shorter).

        Boat counts the auto-stop from creation or resume, and commands do not move it. Best effort:
        only a sandbox found deleted or stopped raises.
        """
        ttl, set_at, now = self._ttl, self._ttl_set_at, monotonic()
        if ttl is None or (set_at is not None and now - set_at < min(ttl / 2, _KEEP_ALIVE_INTERVAL)):
            return
        # Claimed before the call, so operations running meanwhile do not send one too.
        self._ttl_set_at = now
        try:
            sandbox.client.update(
                sandbox.id, boat.UpdateSandboxRequest(ttlSeconds=ttl), _request_timeout=_REQUEST_TIMEOUT
            )
        except Exception as error:
            status = _details(error)[0] if isinstance(error, boat.ApiException) else 0
            gone = _translate(error, 'Could not extend the auto-stop', sandbox_id=sandbox.id)
            if isinstance(gone, WorkspaceUnavailableError) and status in (404, 409):
                # Deleted or stopped: the operation would find that too.
                raise gone from error
            if status in (402, 403):
                # Refused for good, as for a key without `sandbox.update`: stop asking.
                self._ttl = None
            # Tried again once the interval has passed, not on every operation; a failure that repeats
            # is logged at `WARNING` once.
            level = logging.DEBUG if self._keep_alive_failing else logging.WARNING
            self._keep_alive_failing = True
            logger.log(level, 'Could not extend the auto-stop of Boat sandbox %s: %s', sandbox.id, _describe(error))
            return
        self._keep_alive_failing = False

    def _keep_alive_until(self, sandbox: _Sandbox, done: threading.Event) -> None:
        """Extend the auto-stop while a command runs, until `done` is set. Runs in a thread of its own."""
        ttl = self._ttl
        if ttl is None:
            return
        while not done.wait(min(ttl / 2, _KEEP_ALIVE_INTERVAL)):
            try:
                self._extend(sandbox)
            except WorkspaceUnavailableError:
                # Deleted or stopped: following the command finds that too.
                return

    def _shell(self, sandbox: _Sandbox) -> _ShellFilesystem:
        return _ShellFilesystem(_SyncShell(sandbox, self.working_dir))

    async def working_dir(self) -> str:
        """Return the canonical directory commands start in, probed once."""
        if self._resolved_working_dir is None:
            sandbox = await self._connect()
            result = await _in_thread(partial(_exec, sandbox, ['pwd', '-P'], cwd=self._working_dir), abandon=True)
            printed = result.stdout.removesuffix('\n')
            if result.exit_code != 0 or not posixpath.isabs(printed):
                raise WorkspaceError(
                    f'Could not determine the working directory of Boat sandbox {sandbox.id}: {result.stderr}'
                )
            self._resolved_working_dir = printed
        return self._resolved_working_dir

    async def run(
        self,
        command: WorkspaceCommand,
        *,
        shell: bool = False,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        check_timeout(timeout)
        argv = command_argv(command, shell)
        check_env(env)
        # Acquiring the sandbox has its own bound; the timeout is the command's alone.
        sandbox = await self._connect()
        process = _BoatProcess(sandbox, cwd=self._working_dir, keep_alive=partial(self._keep_alive_until, sandbox))
        line = _command_line(argv, marker=process.marker, cwd=self._working_dir, env={**self._env, **(env or {})})

        async def stop() -> None:
            try:
                await _in_thread(process.kill, abandon=True)
            except anyio.get_cancelled_exc_class():
                # The stop's grace ran out first; the stop goes on in its thread.
                logger.warning(
                    'Boat did not confirm the stop of a command in Boat sandbox %s within %gs; it may still be running.',
                    sandbox.id,
                    _KILL_TIMEOUT,
                )
                raise

        try:
            async with command_deadline(timeout, stop=stop, grace=_KILL_TIMEOUT, output=process.partial_output):
                pid = await _in_thread(partial(process.start, line))
                result = await _in_thread(partial(process.follow, pid), abandon=True)
                if result is None:
                    raise process.output_limit_error()
        except WorkspaceTimeoutError as error:
            if process.gone is not None:
                raise process.gone from error
            raise
        return result

    async def realpath(self, path: str) -> str:
        # Native, so it is one synchronous round trip instead of a detached start and its output stream.
        _check_path(path)
        sandbox = await self._connect()
        return await self._shell(sandbox).realpath(posixpath.join(await self.working_dir(), path))

    async def read_bytes(self, path: str) -> bytes:
        _check_path(path)
        sandbox = await self._connect()
        outcome = await _in_thread(partial(_read_file, sandbox, path), abandon=True)
        if isinstance(outcome, bytes):
            return outcome
        if outcome == 'outside':
            return await self._shell(sandbox).read_bytes(path)
        await self._not_a_file(sandbox, path)

    async def write_bytes(self, path: str, data: bytes) -> None:
        _check_path(path)
        sandbox = await self._connect()
        refusal = await _in_thread(partial(_write_file, sandbox, path, data), abandon=True)
        if refusal == 'not_a_file':
            await self._not_a_file(sandbox, path)
        if refusal == 'outside':
            await self._shell(sandbox).write_bytes(path, data)

    async def _not_a_file(self, sandbox: _Sandbox, path: str) -> NoReturn:
        """Raise the built-in error for a path Boat's file API refused as not a regular file."""
        # A directory, or a special file such as a FIFO, which the shell's `stat` refuses itself.
        if (await self._shell(sandbox).stat(path)).is_dir:
            raise IsADirectoryError(errno.EISDIR, 'Is a directory in the Boat sandbox', path)
        raise OSError(f'Not a regular file in the Boat sandbox: {path!r}')

    # The other file operations are the ones Pydantic AI derives from commands, run synchronously.
    async def stat(self, path: str) -> FileEntry:
        _check_path(path)
        return await self._shell(await self._connect()).stat(path)

    async def list_dir(self, path: str) -> Sequence[FileEntry]:
        _check_path(path)
        return await self._shell(await self._connect()).list_dir(path)

    async def make_dir(self, path: str) -> None:
        _check_path(path)
        await self._shell(await self._connect()).make_dir(path)

    async def remove(self, path: str) -> None:
        _check_path(path)
        await self._shell(await self._connect()).remove(path)

    async def exists(self, path: str) -> bool:
        _check_path(path)
        return await self._shell(await self._connect()).exists(path)


def _check_path(path: str) -> None:
    # Refused as the Daytona backend refuses them, so a path behaves the same in both cloud sandboxes.
    if '\n' in path or '\r' in path:
        raise ValueError('Boat file paths cannot contain a newline or carriage return')


def _file_refusal(error: boat.ApiException) -> Literal['outside', 'not_a_file', 'too_large'] | None:
    """Why Boat's file API refused a path, when the refusal is not a POSIX error."""
    status, code, message = _details(error)
    if status != 400:
        return None
    # Boat's file API reaches only paths that resolve under /home/user or /tmp.
    if code == 'invalid_path' or 'Path must resolve under' in message:
        return 'outside'
    # The file API's answer, and a signed transfer's.
    if message == 'Path is not a file.' or code == 'not_a_file':
        return 'not_a_file'
    if message.startswith('File is too large'):
        return 'too_large'
    return None


def _read_file(sandbox: _Sandbox, path: str) -> bytes | _Refusal:
    """A file's bytes, or why Boat's file API left it to the shell. Runs in a worker thread."""
    with _mapped_errors(f'Could not access {path!r} in the sandbox', sandbox=sandbox, path=path):
        try:
            file = sandbox.client.read_file(sandbox.id, path, encoding='base64', _request_timeout=_REQUEST_TIMEOUT)
        except boat.ApiException as error:
            refusal = _file_refusal(error)
            if refusal == 'too_large':
                return _download(sandbox, path)
            if refusal is None:
                raise
            return refusal
        else:
            return base64.b64decode(file.content)


def _write_file(sandbox: _Sandbox, path: str, data: bytes) -> _Refusal | None:
    """Write a file, or say why Boat's file API left it to the shell. Runs in a worker thread."""
    # Boat creates missing parents, follows a symlink, and keeps an existing file's mode.
    with _mapped_errors(f'Could not access {path!r} in the sandbox', sandbox=sandbox, path=path):
        try:
            if len(data) <= _MAX_INLINE_FILE_BYTES:
                _write_inline(sandbox, path, data)
            else:
                _upload(sandbox, path, data)
        except boat.ApiException as error:
            refusal = _file_refusal(error)
            if refusal is None or refusal == 'too_large':
                raise
            return refusal
    return None


def _write_inline(sandbox: _Sandbox, path: str, data: bytes) -> None:
    request = boat.FileWriteRequest(path=path, content=base64.b64encode(data).decode(), encoding='base64')
    try:
        sandbox.client.write_file(sandbox.id, request, _request_timeout=_REQUEST_TIMEOUT)
    except urllib3.exceptions.MaxRetryError as error:
        if not isinstance(error.reason, urllib3.exceptions.ReadTimeoutError):
            raise
        # Boat answers a write to a FIFO only once something opens it for reading, so the SDK sends it
        # again until it gives up. Checked then rather than before every write, so a write stays one round trip.
        special = _exec(sandbox, ['sh', '-c', 'test -e "$1" && ! test -f "$1" && ! test -d "$1"', 'sh', path], cwd=None)
        if special.exit_code == 0:
            raise OSError(f'Not a regular file in the Boat sandbox: {path!r}') from error
        raise


def _pool(client: boat.BoatApi) -> urllib3.PoolManager:
    """The SDK client's own connection pool, so its proxy and TLS settings apply; the SDK leaves the path untyped."""
    return client.api_client.rest_client.pool_manager  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]


def _transfer(sandbox: _Sandbox, method: Literal['GET', 'PUT'], url: str, body: bytes | None = None) -> bytes:
    """Send a request to a signed file-transfer URL and return the answer's body."""
    # The pool adds no `Authorization` header: the URL carries its own signature and points at the
    # sandbox's machine.
    response = _pool(sandbox.client).request(
        method,
        url,
        body=body,
        headers=None if body is None else {'Content-Type': 'application/octet-stream'},
        timeout=urllib3.Timeout(connect=_REQUEST_TIMEOUT[0], read=_REQUEST_TIMEOUT[1]),
    )
    if 200 <= response.status < 300:
        return response.data
    raise boat.ApiException(http_resp=response)


def _download(sandbox: _Sandbox, path: str) -> bytes:
    url = sandbox.client.create_download_url(
        sandbox.id, boat.FileUrlRequest(path=path), _request_timeout=_REQUEST_TIMEOUT
    ).url
    try:
        return _transfer(sandbox, 'GET', url)
    except boat.ApiException as error:
        # Boat issues the URL without checking the path, so the machine answers for a missing file.
        if _details(error)[0] == 404:
            raise FileNotFoundError(errno.ENOENT, 'No such file or directory in the Boat sandbox', path) from error
        raise


def _upload(sandbox: _Sandbox, path: str, data: bytes) -> None:
    url = sandbox.client.create_upload_url(
        sandbox.id, boat.FileUrlRequest(path=path), _request_timeout=_REQUEST_TIMEOUT
    ).url
    _transfer(sandbox, 'PUT', url, data)
