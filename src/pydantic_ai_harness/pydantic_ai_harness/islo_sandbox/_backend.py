"""An Islo sandbox behind Pydantic AI's `WorkspaceBackend` protocol.

`run()` starts each command in a process group of its own with its output sent to files, and the
result is read once the command's shell exits. Islo has no request that stops a command, so a
timeout or a cancellation runs a second command that kills that group. Internal helpers, such as
`pwd -P` and the shell file operations, run as plain Islo commands.

Islo facts this module pins, verified 2026-10-22 against `islo` 0.3.20, https://docs.islo.dev and the
live API. The `islo_live` tier (`tests/harness/islo_sandbox/`) exercises each one not marked otherwise:

- Sandbox calls take the sandbox's name; `get_sandbox_by_id` finds a sandbox by Islo's ID, a deleted
  one too, in the `deleted` state.
- A command runs as the image's default user (`root` on Islo's default image), with its argv words
  kept literal. An exec `workdir` that does not exist runs the command in `/` (not exercised: the
  backend enters `working_dir` itself).
- Islo has no request that stops one command and no deadline of its own: `timeout_secs` is accepted
  and ignored.
- A command's result keeps the last 1 MiB of each stream, and arrives once every process holding its
  output pipes has closed them (not exercised: `run()` sends output to files). Undecodable bytes
  arrive as U+FFFD.
- The file API creates missing parents and writes through a symlink. It refuses a write under a file
  or onto a directory with a bare `500 INTERNAL_ERROR`. It takes absolute paths only, reads a directory
  as a missing file (`404 FILE_NOT_FOUND`), and never answers a read of a FIFO (not exercised: the
  backend sends absolute paths and checks a path's type before reading it).
- A paused sandbox refuses commands (`409 SANDBOX_INVALID_STATE`) and file calls (`503
  RESOURCE_UNAVAILABLE`); resuming it, also on activity, needs account credit (not exercised).
- After a delete, `get_sandbox_by_id` reports the sandbox `deleted` at once, while its own calls can
  still succeed for some seconds before they answer `404 SANDBOX_NOT_FOUND`.
- A create Islo refuses for lack of credit (`402 BILLING_NOT_ALLOWED`) can still make the sandbox,
  answering up to 20 seconds later, and a delete sent soon after can leave it running (seen while
  running the live tier, not exercised by it).
"""

from __future__ import annotations

import errno
import logging
import os
import posixpath
import re
import shlex
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import anyio
import anyio.lowlevel
import httpx
from pydantic import ValidationError
from typing_extensions import TypedDict

from pydantic_ai.exceptions import UserError
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
    WorkspaceUnavailableError,
)
from pydantic_ai.workspaces.workspace import _ShellFilesystem  # pyright: ignore[reportPrivateUsage]
from pydantic_ai_harness._workspace_provider import (
    absolute_path,
    check_env,
    check_timeout,
    command_argv,
    command_deadline,
    run_to_completion,
    running_on_asyncio,
    safe_credential_reason,
)

if TYPE_CHECKING:
    from islo import AsyncIslo
    from islo.core.request_options import RequestOptions
    from islo.types import ExecResultResponse

    from pydantic_ai.workspaces import WorkspaceCommand

__all__ = ('IsloSandboxBackend',)

try:
    import islo
    from islo.core.api_error import ApiError
    from islo.types import AutoResumePolicy, ErrorCode, ErrorResponse, LifecyclePolicy, SandboxResponse
except ImportError as error:
    raise ImportError('Install `pydantic-ai-harness[islo]` to use IsloSandbox.') from error

_AUTH_MESSAGE = 'Islo rejected the credentials. Set ISLO_API_KEY, or pass a configured `islo.AsyncIslo` as `client=`.'
_MISSING_CREDENTIALS_MESSAGE = (
    'No Islo credentials found. Set ISLO_API_KEY, or pass a configured `islo.AsyncIslo` as `client=`.'
)
_OUTPUT_LIMIT_MESSAGE = (
    'Islo command output exceeded the 1 MiB limit per stream; redirect it to a file and read part of it instead'
)
_REQUEST_TIMEOUT = 30
# Bound sandbox creation and attachment so a wedged control plane cannot hang acquisition.
_CREATE_TIMEOUT = 120
_TEARDOWN_TIMEOUT = 30.0
# How long the backend keeps deleting a sandbox that a create Islo refused left behind.
_DISCARD_TIMEOUT = 60.0
# Internal commands (`pwd -P`, the shell file operations, a stop) get this long before the backend stops waiting.
_HELPER_TIMEOUT = 60
# Between attempts to create a sandbox whose earlier request may have been lost.
_RETRY_DELAY = 1.0
# Between checks of a sandbox that is starting, and of a running command, which back off to `_MAX_POLL_INTERVAL`.
_READY_POLL = 0.5
_POLL_INTERVAL = 0.05
_MAX_POLL_INTERVAL = 1.0
# The most partial output a timeout reports per stream.
_PARTIAL_OUTPUT_LIMIT = 4096
# The most of each stream Islo keeps of a command's output.
_MAX_OUTPUT_BYTES = 1024 * 1024
_OUTPUT_PREVIEW_BYTES = 64 * 1024
# Where `run()` keeps a command's output and process group ID while it runs.
_SCRATCH_DIR = '/tmp'

_RUNNING = 'running'
# Islo still finds a deleted sandbox by its ID, in this state.
_DELETED = 'deleted'
_PAUSED_STATES = frozenset({'paused', 'stopped'})
_FAILED_STATES = frozenset({'failed', 'error'})
# Exec states Islo reports once a command has ended; `running` and `started` mean it has not.
_FINISHED_EXEC_STATES = frozenset({'completed', 'failed', 'timeout', 'killed', 'cancelled'})
# A request that changes something is sent once: the SDK's own retries could repeat it.
_ONCE: RequestOptions = {'max_retries': 0, 'timeout_in_seconds': _REQUEST_TIMEOUT}
_LOOKUP: RequestOptions = {'timeout_in_seconds': _REQUEST_TIMEOUT}
# Environment names a POSIX `sh` passes on to the programs it runs.
_SHELL_NAME = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')
# Exit codes of the script that tells why Islo's file API refused a write (100 plus the errno, as core's).
_NOT_A_DIRECTORY = 120
_IS_A_DIRECTORY = 121
_NOT_REGULAR = 122
_WRITE_REFUSAL = (
    'p=$1; if [ -d "$p" ]; then exit 121; fi; if [ -e "$p" ] && [ ! -f "$p" ]; then exit 122; fi; '
    'd=${p%/*}; while [ -n "$d" ]; do if [ -e "$d" ]; then [ -d "$d" ] || exit 120; exit 0; fi; d=${d%/*}; done'
)

logger = logging.getLogger(__name__)


def _require_asyncio() -> None:
    if not running_on_asyncio():
        raise UserError('Islo needs the asyncio event loop: the Islo SDK runs its calls on asyncio.')


def _new_client() -> tuple[AsyncIslo, httpx.AsyncClient]:
    """An `AsyncIslo` that authenticates with `ISLO_API_KEY`, and the HTTP client to close when done with it."""
    if not os.environ.get('ISLO_API_KEY', '').strip():
        raise WorkspaceUnavailableError(_MISSING_CREDENTIALS_MESSAGE)
    http = httpx.AsyncClient(timeout=_REQUEST_TIMEOUT, follow_redirects=True)
    return islo.AsyncIslo(httpx_client=http), http


async def _close_quietly(http: httpx.AsyncClient) -> None:
    """Close an HTTP client, bounded and shielded; a failure is logged, not raised."""
    with anyio.move_on_after(_TEARDOWN_TIMEOUT, shield=True) as deadline:
        try:
            await http.aclose()
        except Exception as error:
            logger.warning('Could not close the Islo API client: %r', error)
    if deadline.cancelled_caught:
        logger.warning('Closing the Islo API client did not complete within %ss.', _TEARDOWN_TIMEOUT)


@dataclass(frozen=True)
class _Sandbox:
    """An Islo sandbox reached through one API client. Islo's sandbox calls take its name."""

    client: AsyncIslo
    name: str
    id: str


def _is_id(key: str) -> bool:
    """Whether `key` has the shape of an Islo sandbox ID (a UUID) rather than a sandbox name."""
    try:
        uuid.UUID(key)
    except ValueError:
        return False
    return True


async def _lookup(client: AsyncIslo, key: str) -> SandboxResponse:
    """Islo's record of the sandbox with ID or name `key`."""
    if _is_id(key):
        try:
            return await client.sandboxes.get_sandbox_by_id(key, request_options=_LOOKUP)
        except ApiError as error:
            # A sandbox can also be named like an ID.
            if error.status_code != 404:
                raise
    return await client.sandboxes.get_sandbox(key, request_options=_LOOKUP)


async def _find(client: AsyncIslo, name: str) -> SandboxResponse | None:
    """Islo's record of the sandbox named `name`, or `None` if Islo has none or the lookup fails."""
    try:
        return await client.sandboxes.get_sandbox(name, request_options=_ONCE)
    except (ApiError, httpx.TransportError):
        return None


async def _discard(client: AsyncIslo, name: str) -> None:
    """Delete the sandbox named `name` if there is one; best effort, bounded and shielded.

    Islo can still be making the sandbox when it refuses the create, and then refuse to delete it,
    so the delete is sent again until a lookup no longer finds the sandbox.
    """
    with anyio.move_on_after(_DISCARD_TIMEOUT, shield=True) as deadline:
        while await _find(client, name) is not None:
            try:
                await client.sandboxes.delete_sandbox(name, request_options=_ONCE)
            except Exception:
                await anyio.sleep(_RETRY_DELAY)
            else:
                logger.info('Deleted Islo sandbox %s, which a refused create left behind', name)
    if deadline.cancelled_caught:
        logger.warning('Could not delete Islo sandbox %s, which a refused create left behind', name)


async def destroy_sandbox(client: AsyncIslo | None, sandbox_id: str) -> None:
    """Delete a sandbox by ID or name without resuming it, including one Islo has paused.

    A sandbox that no longer exists returns quietly; rejected credentials raise
    `WorkspaceUnavailableError`, like any other operation. Without `client`, one is opened from
    the environment and closed afterwards.

    Raises:
        UserError: The event loop is not asyncio.
    """
    _require_asyncio()
    if not sandbox_id.strip():
        raise ValueError('Islo workspace ref id cannot be empty')
    owned: httpx.AsyncClient | None = None
    if client is None:
        client, owned = _new_client()
    try:
        async with _mapped_errors(f'Could not delete Islo sandbox {sandbox_id!r}'):
            try:
                info = await _lookup(client, sandbox_id)
                # A deleted sandbox's name can belong to a newer one.
                if info.status != _DELETED:
                    await client.sandboxes.delete_sandbox(info.name, request_options=_ONCE)
            except ApiError as error:
                if error.status_code != 404:
                    raise
    finally:
        if owned is not None:
            await _close_quietly(owned)


def _not_found_message(key: str) -> str:
    return (
        f'The Islo sandbox {key!r} was not found (it was deleted, or never existed in this Islo '
        "organization). Pass `workspace='new'` to start a fresh sandbox."
    )


def _unavailable_message(sandbox_id: str) -> str:
    return f"The Islo sandbox {sandbox_id!r} no longer exists (it was deleted). Pass `workspace='new'` to start a fresh sandbox."


def _paused_message(sandbox_id: str, state: str) -> str:
    return (
        f'The Islo sandbox {sandbox_id!r} was {state}; a later run that attaches to it resumes it. '
        "Pass `workspace='new'` to start a fresh sandbox."
    )


def _failed_message(sandbox_id: str, state: str) -> str:
    return (
        f'The Islo sandbox {sandbox_id!r} is in the {state!r} state and cannot be used. '
        "Pass `workspace='new'` to start a fresh sandbox."
    )


def _missing_cwd(cwd: str | None, sandbox_id: str) -> WorkspaceError:
    return WorkspaceError(
        f'working_dir {cwd!r} does not exist in Islo sandbox {sandbox_id}. '
        'Create it there, or pass a working_dir that exists.'
    )


def _details(error: ApiError) -> ErrorResponse | None:
    """Islo's error code and message: the SDK parses an error body into `ErrorResponse` for some answers only."""
    try:
        return ErrorResponse.model_validate(error.body)
    except ValidationError:
        return None


def _already_exists(error: Exception) -> bool:
    details = _details(error) if isinstance(error, ApiError) else None
    return details is not None and details.code == ErrorCode.SANDBOX_ALREADY_EXISTS


def _translate(
    error: Exception, context: str, *, sandbox: _Sandbox | None = None, path: str | None = None
) -> Exception:
    """Map an Islo SDK error onto the failures the workspace protocol promises.

    Returns `error` itself for what must propagate unchanged: transport failures, timeouts, rate
    limits, conflicts, server errors, and anything that is not an Islo answer, which durable engines
    retry as transient.
    """
    if isinstance(error, httpx.HTTPStatusError):
        # The SDK trades ISLO_API_KEY for a session token before its first call; a refused key fails there.
        if error.request.url.path.endswith('/auth/token') and 400 <= error.response.status_code < 500:
            return WorkspaceUnavailableError(f'{safe_credential_reason(error)}. {_AUTH_MESSAGE}')
        return error
    if not isinstance(error, ApiError) or error.status_code is None:
        return error
    status = error.status_code
    details = _details(error)
    code = None if details is None else details.code
    message = str(error.body) if details is None else details.message
    if status == 401:
        return WorkspaceUnavailableError(f'{safe_credential_reason(error)}. {_AUTH_MESSAGE}')
    if status in (402, 403) or code in (ErrorCode.BILLING_NOT_ALLOWED, ErrorCode.TENANT_SUSPENDED):
        # Billing, plan and permission refusals, which no retry fixes.
        return WorkspaceUnavailableError(f'{context}: {message}')
    if sandbox is not None and code == ErrorCode.SANDBOX_NOT_FOUND:
        return WorkspaceUnavailableError(_unavailable_message(sandbox.id))
    if path is not None and code == ErrorCode.FILE_NOT_FOUND:
        return FileNotFoundError(errno.ENOENT, 'No such file or directory in the Islo sandbox', path)
    if 400 <= status < 500 and status not in (408, 409, 429):
        return WorkspaceError(f'{context}: {message}')
    return error


def _state_error(sandbox_id: str, state: str) -> WorkspaceUnavailableError | None:
    if state == _DELETED:
        return WorkspaceUnavailableError(_unavailable_message(sandbox_id))
    if state in _PAUSED_STATES:
        return WorkspaceUnavailableError(_paused_message(sandbox_id, state))
    if state in _FAILED_STATES:
        return WorkspaceUnavailableError(_failed_message(sandbox_id, state))
    return None


async def _probe(sandbox: _Sandbox) -> WorkspaceUnavailableError | None:
    """`WorkspaceUnavailableError` if Islo deleted or paused `sandbox`; an inconclusive lookup counts as running."""
    try:
        with anyio.fail_after(_REQUEST_TIMEOUT):
            info = await sandbox.client.sandboxes.get_sandbox(sandbox.name, request_options=_ONCE)
    except ApiError as error:
        return WorkspaceUnavailableError(_unavailable_message(sandbox.id)) if error.status_code == 404 else None
    except Exception:
        return None
    return _state_error(sandbox.id, info.status)


@asynccontextmanager
async def _mapped_errors(
    context: str, *, sandbox: _Sandbox | None = None, path: str | None = None
) -> AsyncGenerator[None]:
    """Raise the protocol's typed failure for an Islo SDK error, and let the backend's own failures through.

    With `sandbox`, a failure that would propagate unchanged is followed by one lookup, so a sandbox
    that was deleted or paused meanwhile ends the run instead of being retried.
    """
    try:
        yield
    except (WorkspaceError, OSError):
        raise
    except Exception as error:
        translated = _translate(error, context, sandbox=sandbox, path=path)
        if translated is not error:
            raise translated from error
        if sandbox is not None and (gone := await _probe(sandbox)) is not None:
            raise gone from error
        raise


def _marker() -> str:
    return f'pydantic-ai-{uuid.uuid4().hex}'


def _cwd_guard(marker: str, cwd: str | None) -> str:
    """Shell that enters `cwd`, or prints `<marker>-cwd` and exits 125 when it is missing.

    Islo itself would run the command in `/` instead.
    """
    if cwd is None:
        return ''
    return f'cd -- {shlex.quote(cwd)} 2>/dev/null || {{ printf %s {marker}-cwd; exit 125; }}; '


def _run_script(marker: str, cwd: str | None) -> str:
    """The `sh` script `run()` sends, with the command's argv as its positional parameters.

    The command gets a process group of its own from `setsid` where the image has it, so a stop ends
    its children too, and writes its output to files, so a background child it leaves holding them
    does not hold up the result. The script records the group's ID, waits for the command, and prints
    its output: in full, or after a `<marker>-limit` line just the first 64 KiB of each stream, when
    one passes Islo's 1 MiB. A stop that came first, which leaves `<file>.stop`, keeps it from starting.
    """
    limit, preview = _MAX_OUTPUT_BYTES, _OUTPUT_PREVIEW_BYTES
    return (
        f'f={_SCRATCH_DIR}/{marker}; if [ -e "$f.stop" ]; then exit 137; fi; {_cwd_guard(marker, cwd)}'
        'if command -v setsid >/dev/null 2>&1; then setsid "$@" </dev/null >"$f.out" 2>"$f.err" & '
        'else "$@" </dev/null >"$f.out" 2>"$f.err" & fi; '
        'p=$!; printf %s "$p" >"$f.pid"; wait "$p"; s=$?; if [ -e "$f.stop" ]; then exit "$s"; fi; '
        'o=$(($(wc -c <"$f.out"))); e=$(($(wc -c <"$f.err"))); '
        f'if [ "$o" -gt {limit} ] || [ "$e" -gt {limit} ]; then printf "%s\\n" {marker}-limit; '
        f'head -c {preview} "$f.out"; head -c {preview} "$f.err" >&2; '
        'else cat "$f.out"; cat "$f.err" >&2; fi; rm -f "$f.out" "$f.err" "$f.pid"; exit "$s"'
    )


def _stop_script(marker: str) -> str:
    """The `sh` script that stops `run()`'s command and prints the tail of what it wrote so far.

    It waits up to half a second for a start still in flight to record its process group, and leaves
    `<file>.stop` behind when none did, so a start that arrives later does not run.
    """
    tail = _PARTIAL_OUTPUT_LIMIT
    return (
        f'f={_SCRATCH_DIR}/{marker}; : >"$f.stop"; i=0; '
        'while [ ! -s "$f.pid" ] && [ "$i" -lt 5 ]; do sleep 0.1; i=$((i + 1)); done; '
        'if [ -s "$f.pid" ]; then p=$(cat "$f.pid"); kill -KILL -"$p" 2>/dev/null || kill -KILL "$p" 2>/dev/null; '
        f'tail -c {tail} "$f.out" 2>/dev/null; tail -c {tail} "$f.err" >&2 2>/dev/null; '
        'rm -f "$f.out" "$f.err" "$f.pid" "$f.stop"; fi'
    )


def _split_env(env: Mapping[str, str], argv: list[str]) -> tuple[dict[str, str | None], list[str]]:
    """Islo's `env` for names a `sh` passes on, and `argv` behind `env --` for the rest.

    A `sh` such as dash drops names that are not shell identifiers, so those travel in the command
    line instead. `--` ends `env`'s options, so a name starting with `-` is not read as one.
    """
    native: dict[str, str | None] = {name: value for name, value in env.items() if _SHELL_NAME.fullmatch(name)}
    other = [f'{name}={value}' for name, value in env.items() if name not in native]
    return native, ['env', '--', *other, *argv] if other else argv


async def _poll(sandbox: _Sandbox, exec_id: str) -> ExecResultResponse:
    """Check a command until Islo reports it ended."""
    interval = _POLL_INTERVAL
    while True:
        async with _mapped_errors('Could not read the command result', sandbox=sandbox):
            result = await sandbox.client.sandboxes.get_exec_result(sandbox.name, exec_id, request_options=_LOOKUP)
        if result.exit_code is not None or result.status in _FINISHED_EXEC_STATES:
            if result.exit_code is None:
                raise WorkspaceError(
                    f'Islo reported the command {result.status!r} without an exit code: {result.stderr}'
                )
            return result
        await anyio.sleep(interval)
        interval = min(interval * 2, _MAX_POLL_INTERVAL)


async def _exec(sandbox: _Sandbox, argv: Sequence[str], *, cwd: str | None) -> CommandResult:
    """Run a short internal command and wait for it, bounded by `_HELPER_TIMEOUT`."""
    marker = _marker()
    command = ['sh', '-c', f'{_cwd_guard(marker, cwd)}exec "$@"', 'sh', *argv] if cwd is not None else list(argv)
    with anyio.move_on_after(_HELPER_TIMEOUT):
        async with _mapped_errors('Could not run a command in the sandbox', sandbox=sandbox):
            started = await sandbox.client.sandboxes.exec_in_sandbox(
                sandbox.name, command=command, env={}, request_options=_ONCE
            )
        result = await _poll(sandbox, started.exec_id)
        if result.truncated:
            raise WorkspaceOutputLimitError(_OUTPUT_LIMIT_MESSAGE, limit=_MAX_OUTPUT_BYTES)
        if result.exit_code == 125 and result.stdout == f'{marker}-cwd':
            raise _missing_cwd(cwd, sandbox.id)
        assert result.exit_code is not None
        return CommandResult(exit_code=result.exit_code, stdout=result.stdout, stderr=result.stderr)
    raise WorkspaceError(
        f'An internal command did not finish within {_HELPER_TIMEOUT}s in Islo sandbox {sandbox.id}; '
        'it may still be running.'
    )


class _SyncShell:
    """Internal Islo commands, as the runner core's shell filesystem drives."""

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
        return await _exec(self._sandbox, command_argv(command, shell), cwd=None)


class _IsloProcess:
    """A command `run()` started in an Islo sandbox, checked until its shell exits."""

    def __init__(self, sandbox: _Sandbox) -> None:
        self._sandbox = sandbox
        # In the command's file names, so a stop finds the command even when Islo never confirmed its start.
        self.marker = _marker()
        self._exec_id: str | None = None
        # Nothing is left to stop: the command ended, or Islo refused to start it.
        self._finished = False
        self.missing_cwd = False
        # The partial output a timeout reports, which the stop prints.
        self.stdout = ''
        self.stderr = ''

    async def start(self, argv: list[str], *, cwd: str | None, env: dict[str, str | None]) -> None:
        command = ['sh', '-c', _run_script(self.marker, cwd), 'sh', *argv]
        try:
            async with _mapped_errors('Could not start the command', sandbox=self._sandbox):
                started = await self._sandbox.client.sandboxes.exec_in_sandbox(
                    self._sandbox.name, command=command, env=env, request_options=_ONCE
                )
        except WorkspaceError:
            # A refusal, or a sandbox found paused or deleted: there is no command to stop.
            self._finished = True
            raise
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout, httpx.UnsupportedProtocol):
            # Never sent.
            self._finished = True
            raise
        self._exec_id = started.exec_id

    async def wait(self) -> CommandResult:
        assert self._exec_id is not None
        try:
            result = await _poll(self._sandbox, self._exec_id)
        except WorkspaceUnavailableError:
            # Deleted or paused with the sandbox: a stop would only fail.
            self._finished = True
            raise
        self._finished = True
        stdout, stderr, exit_code = result.stdout, result.stderr, result.exit_code
        assert exit_code is not None
        if exit_code == 125 and stdout == f'{self.marker}-cwd':
            self.missing_cwd = True
            return CommandResult(exit_code=exit_code, stdout='', stderr='')
        limited = f'{self.marker}-limit\n'
        if stdout.startswith(limited):
            raise WorkspaceOutputLimitError(
                _OUTPUT_LIMIT_MESSAGE, limit=_MAX_OUTPUT_BYTES, stdout=stdout.removeprefix(limited), stderr=stderr
            )
        if result.truncated:
            # Output a background child wrote after the sizes were read.
            raise WorkspaceOutputLimitError(_OUTPUT_LIMIT_MESSAGE, limit=_MAX_OUTPUT_BYTES)
        return CommandResult(exit_code=exit_code, stdout=stdout, stderr=stderr)

    async def kill(self) -> None:
        """Kill the command's process group, unless it already ended, and keep the tail of its output."""
        if self._finished:
            return
        stopped = False
        try:
            result = await _exec(self._sandbox, ['sh', '-c', _stop_script(self.marker)], cwd=None)
            self.stdout, self.stderr = result.stdout, result.stderr
            stopped = True
        finally:
            if not stopped:
                # A failure, or the grace running out before Islo answered.
                logger.warning(
                    'Could not stop a command in Islo sandbox %s; it may still be running.', self._sandbox.id
                )


class _CreateOptions(TypedDict, total=False):
    """Creation settings, with unset ones left out: Islo refuses `null` for some of them."""

    image: str
    vcpus: int
    memory_mb: int
    gateway_profile: str
    lifecycle: LifecyclePolicy


class IsloSandboxBackend(WorkspaceBackend, SupportsCommands, SupportsFilesystem, SupportsRealpath):
    """An [Islo](https://islo.dev) sandbox as a Pydantic AI [`WorkspaceBackend`][pydantic_ai.workspaces.WorkspaceBackend].

    Commands and file operations run inside an Islo sandbox, a cloud VM, so the host is never exposed.

    Building one does no I/O. The first operation creates or attaches to a sandbox, and Islo's record
    of it is available through `get_sandbox()`. The backend does not pause or delete the sandbox;
    that is the application's job.

    A command's complete output is returned once it exits. A shell string runs under `/bin/sh -c`;
    a timeout or cancellation stops the command and the processes it started.

    A deleted sandbox, one Islo paused (attaching again resumes it), rejected credentials, a
    refusal for lack of account credit, and an unknown image raise `WorkspaceUnavailableError`.
    Path failures raise the built-in errors, such as `FileNotFoundError`.

    Args:
        client: An `islo.AsyncIslo` API client to create or attach with. The caller owns it.
            Without one, the backend opens its own from `ISLO_API_KEY` on first use and closes it
            in `aclose()`.
        ref: Identity of an existing sandbox to attach to on first use: its Islo ID, or its name.
        image: Container image a newly created sandbox runs; Islo's default image when `None`.
        vcpus: Virtual CPUs of a newly created sandbox; Islo's default when `None`.
        memory_mb: Memory of a newly created sandbox in MiB; Islo's default when `None`.
        gateway_profile: Islo gateway profile of a newly created sandbox, which sets what it can
            reach and which credentials Islo injects into its requests; Islo's default when `None`.
        pause_after_idle: Seconds without commands or file operations before Islo pauses a newly
            created sandbox, keeping its disk; a later command resumes it. `None` keeps Islo's
            default, which never pauses it, so it runs and is billed until you delete it.
        working_dir: Absolute directory commands start in and relative paths resolve against; the
            sandbox's own default when `None`, discovered with `pwd -P` on first use.
        env: Environment variables every command gets; a command's own `env` is layered on top.
    """

    def __init__(
        self,
        *,
        client: AsyncIslo | None = None,
        ref: WorkspaceRef | None = None,
        image: str | None = None,
        vcpus: int | None = None,
        memory_mb: int | None = None,
        gateway_profile: str | None = None,
        pause_after_idle: int | None = 900,
        working_dir: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        if ref is not None and ref.provider != 'islo':
            raise ValueError(f"unsupported workspace provider {ref.provider!r}; expected 'islo'")
        if ref is not None and not ref.id.strip():
            raise ValueError('Islo workspace ref id cannot be empty')
        check_env(env)
        self._ref = ref
        self._client = client
        # The HTTP client under a client this backend opened, closed by `aclose()`.
        self._http: httpx.AsyncClient | None = None
        options = _CreateOptions()
        if image is not None:
            options['image'] = image
        if vcpus is not None:
            options['vcpus'] = vcpus
        if memory_mb is not None:
            options['memory_mb'] = memory_mb
        if gateway_profile is not None:
            options['gateway_profile'] = gateway_profile
        if pause_after_idle is not None:
            # Resumed on the next command, so a run that pauses for longer than this carries on.
            options['lifecycle'] = LifecyclePolicy(
                pause_after_idle=pause_after_idle, auto_resume=AutoResumePolicy.ON_ACTIVITY
            )
        self._create_options = options
        # Kept across retries, so a lost reply finds the sandbox it created rather than making another.
        self._create_name = f'pydantic-ai-{uuid.uuid4().hex[:16]}'
        self._env = dict(env or {})
        self._working_dir = absolute_path('working_dir', working_dir)
        self._sandbox: _Sandbox | None = None
        # `working_dir()` must return a canonical absolute path, resolved once with `pwd -P`.
        self._resolved_working_dir: str | None = None
        # A sandbox this backend created still needs its `working_dir` made.
        self._working_dir_pending = False
        self._lock = anyio.Lock()
        # Set while nothing will call `aclose()`: the owned client then closes when the last
        # operation in flight ends.
        self._close_after_operation = False
        self._operations = 0

    @property
    def ref(self) -> WorkspaceRef | None:
        """Identity of the sandbox, or `None` before one has been created.

        The `id` is Islo's sandbox ID, a UUID, also after attaching by a `ref` that names the sandbox.
        """
        return self._ref

    async def get_sandbox(self) -> SandboxResponse:
        """Return Islo's current record of the sandbox, creating or attaching to it on first use.

        Its `name` is what the Islo SDK's `client.sandboxes` calls take. Attaching by `ref` to a
        sandbox that no longer exists raises `WorkspaceUnavailableError`; it does not create a
        replacement.
        """
        async with self._operation():
            sandbox = await self._connect()
            async with _mapped_errors(f'Could not look up Islo sandbox {sandbox.id!r}', sandbox=sandbox):
                return await sandbox.client.sandboxes.get_sandbox(sandbox.name, request_options=_LOOKUP)

    async def _connect(self) -> _Sandbox:
        """Create or attach to the sandbox on first use, and return it once it is running."""
        _require_asyncio()
        async with self._lock:
            if (sandbox := self._sandbox) is None:
                sandbox = await self._acquire_locked()
            if self._working_dir_pending:
                # A caller cancelled during creation stops here; the next operation makes the directory.
                await anyio.lowlevel.checkpoint_if_cancelled()
                await self._make_working_dir(sandbox)
        await anyio.lowlevel.checkpoint_if_cancelled()
        return sandbox

    async def _acquire_locked(self) -> _Sandbox:
        """Create or attach to the sandbox; the caller holds `_lock`."""
        if self._client is None:
            self._client, self._http = _new_client()
        client = self._client
        if self._ref is not None:
            info = await self._attach(client, self._ref.id)
        else:

            async def create() -> SandboxResponse:
                created = await self._create(client)
                self._ref = WorkspaceRef(provider='islo', id=created.id)
                self._working_dir_pending = self._working_dir is not None
                # Logged at creation, so the sandbox can be found whatever ends the run before it is stored.
                logger.info('Created Islo sandbox %s (%s)', created.id, created.name)
                return created

            # Creation finishes even if the caller is cancelled, so a billed sandbox is never left unrecorded.
            info = await run_to_completion(create)
        sandbox = _Sandbox(client, info.name, info.id)
        self._ref = WorkspaceRef(provider='islo', id=info.id)
        await self._wait_until_running(sandbox, info.status)
        self._sandbox = sandbox
        return sandbox

    async def _create(self, client: AsyncIslo) -> SandboxResponse:
        """Create a sandbox under a name kept across retries, so a lost reply is found rather than duplicated."""
        name = self._create_name
        with anyio.move_on_after(_CREATE_TIMEOUT):
            while True:
                try:
                    return await client.sandboxes.create_sandbox(
                        name=name, request_options=_ONCE, **self._create_options
                    )
                except (httpx.UnsupportedProtocol, httpx.LocalProtocolError):
                    # The request never left the client, so nothing was created and a retry cannot help.
                    raise
                except httpx.TransportError:
                    pass  # The sandbox may have been created before the reply was lost; its name finds it.
                except Exception as error:
                    # The name is this backend's own, so a sandbox already under it is one an earlier attempt made.
                    if not _already_exists(error):
                        translated = _translate(error, 'Could not start Islo sandbox')
                        if translated is error:
                            # A server error can come after Islo made the sandbox.
                            if (found := await _find(client, name)) is not None:
                                return found
                            raise
                        # Islo can refuse a create, for lack of credit say, after making the sandbox.
                        await _discard(client, name)
                        if type(translated) is WorkspaceError:
                            # Islo refused the request (an unknown image, an invalid setting): no retry can fix it.
                            translated = WorkspaceUnavailableError(str(translated))
                        raise translated from error
                if (found := await _find(client, name)) is not None:
                    return found
                await anyio.sleep(_RETRY_DELAY)
        # A stalled control plane is transient, not a command deadline.
        raise TimeoutError(
            f'Islo sandbox creation did not complete within {_CREATE_TIMEOUT}s; the Islo control plane may be unreachable.'
        )

    async def _attach(self, client: AsyncIslo, key: str) -> SandboxResponse:
        """Islo's record of the sandbox to attach to, by ID or name."""
        async with _mapped_errors(f'Could not attach to Islo sandbox {key!r}'):
            try:
                with anyio.fail_after(_CREATE_TIMEOUT):
                    return await _lookup(client, key)
            except ApiError as error:
                if error.status_code == 404:
                    raise WorkspaceUnavailableError(_not_found_message(key)) from error
                raise

    async def _wait_until_running(self, sandbox: _Sandbox, state: str) -> None:
        """Wait until Islo reports the sandbox running, resuming it if Islo paused or stopped it."""
        with anyio.move_on_after(_CREATE_TIMEOUT):
            async with _mapped_errors(f'Could not attach to Islo sandbox {sandbox.id!r}'):
                while state != _RUNNING:
                    if state in _PAUSED_STATES:
                        try:
                            await sandbox.client.sandboxes.resume_sandbox(sandbox.name, request_options=_ONCE)
                        except ApiError as error:
                            # Islo refuses a resume while the sandbox is between states.
                            if error.status_code != 409:
                                raise
                    elif (unusable := _state_error(sandbox.id, state)) is not None:
                        raise unusable
                    await anyio.sleep(_READY_POLL)
                    try:
                        state = (
                            await sandbox.client.sandboxes.get_sandbox(sandbox.name, request_options=_LOOKUP)
                        ).status
                    except ApiError as error:
                        if error.status_code == 404:
                            raise WorkspaceUnavailableError(_unavailable_message(sandbox.id)) from error
                        raise
                return
        raise TimeoutError(f'Connecting to Islo sandbox {sandbox.id!r} did not complete within {_CREATE_TIMEOUT}s.')

    async def _make_working_dir(self, sandbox: _Sandbox) -> None:
        directory = self._working_dir
        assert directory is not None
        with anyio.CancelScope(shield=True):
            result = await _exec(sandbox, ['mkdir', '-p', '--', directory], cwd=None)
            if result.exit_code != 0:
                raise WorkspaceError(
                    f'Could not create working_dir {directory!r} in Islo sandbox {sandbox.id}: {result.stderr.strip()}'
                )
            self._working_dir_pending = False

    async def aclose(self) -> None:
        """Close the API client this backend opened for itself.

        A client passed as `client=` is left alone, and the sandbox keeps running. A later
        operation opens a new client and attaches by `ref`. `IsloSandbox` calls this when the run
        that used the backend ends.
        """
        if self._http is None or not running_on_asyncio():
            return
        if self._operations:
            # An operation in flight, such as a cancelled command being stopped, still needs the
            # client: the last one to finish closes it.
            self._close_after_operation = True
            return

        async def close() -> None:
            # Under the lock, so a sandbox still being created is recorded before its client closes.
            async with self._lock:
                http = self._detach()
            if http is not None:
                await _close_quietly(http)

        await run_to_completion(close)

    def _detach(self) -> httpx.AsyncClient | None:
        """Forget the client this backend opened, so the next operation opens its own; return its HTTP client."""
        http, self._http = self._http, None
        if http is not None:
            self._client, self._sandbox, self._resolved_working_dir = None, None, None
        return http

    @asynccontextmanager
    async def _operation(self) -> AsyncGenerator[None]:
        self._operations += 1
        try:
            yield
        finally:
            self._operations -= 1
            if self._operations == 0 and self._close_after_operation and self._http is not None:
                # Detached before any await, so an operation starting meanwhile opens its own client.
                http = self._detach()
                assert http is not None
                await run_to_completion(lambda: _close_quietly(http))

    def _shell(self, sandbox: _Sandbox) -> _ShellFilesystem:
        return _ShellFilesystem(_SyncShell(sandbox, self.working_dir))

    async def _absolute(self, path: str) -> str:
        # Islo's file API refuses a relative path.
        return path if posixpath.isabs(path) else posixpath.join(await self.working_dir(), path)

    async def working_dir(self) -> str:
        """Return the canonical directory commands start in, probed once per connection."""
        async with self._operation():
            if self._resolved_working_dir is None:
                sandbox = await self._connect()
                result = await _exec(sandbox, ['pwd', '-P'], cwd=self._working_dir)
                printed = result.stdout.removesuffix('\n')
                if result.exit_code != 0 or not posixpath.isabs(printed):
                    raise WorkspaceError(
                        f'Could not determine the working directory of Islo sandbox {sandbox.id}: {result.stderr}'
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
        async with self._operation():
            check_timeout(timeout)
            argv = command_argv(command, shell)
            check_env(env)
            # Acquiring the sandbox has its own bound; the timeout is the command's alone.
            sandbox = await self._connect()
            native_env, argv = _split_env({**self._env, **(env or {})}, argv)
            process = _IsloProcess(sandbox)

            async def stop() -> None:
                # An operation of its own, so a stop a repeated cancellation leaves running keeps the
                # owned client open until it ends.
                async with self._operation():
                    await process.kill()

            async with command_deadline(timeout, stop=stop, output=lambda: (process.stdout, process.stderr)):
                await process.start(argv, cwd=self._working_dir, env=native_env)
                result = await process.wait()
                if process.missing_cwd:
                    raise _missing_cwd(self._working_dir, sandbox.id)
            return result

    async def realpath(self, path: str) -> str:
        # Native, so it is one internal command instead of a `run()` with its wrapper.
        async with self._operation():
            sandbox = await self._connect()
            return await self._shell(sandbox).realpath(await self._absolute(path))

    async def read_bytes(self, path: str) -> bytes:
        async with self._operation():
            sandbox = await self._connect()
            target = await self._absolute(path)
            # Checked first: Islo's file API reads a directory as a missing file and never answers for a FIFO.
            if (await self._shell(sandbox).stat(target)).is_dir:
                raise IsADirectoryError(errno.EISDIR, 'Is a directory in the Islo sandbox', target)
            data = bytearray()
            async with _mapped_errors(f'Could not read {target!r}', sandbox=sandbox, path=target):
                async for chunk in sandbox.client.sandboxes.download_file(
                    sandbox.name, path=target, request_options=_LOOKUP
                ):
                    data.extend(chunk)
            return bytes(data)

    async def write_bytes(self, path: str, data: bytes) -> None:
        async with self._operation():
            sandbox = await self._connect()
            target = await self._absolute(path)
            # Islo creates missing parents, follows a symlink, and keeps an existing file's mode.
            async with _mapped_errors(f'Could not write {target!r}', sandbox=sandbox, path=target):
                try:
                    await sandbox.client.sandboxes.upload_file(
                        sandbox.name,
                        path=target,
                        file=(posixpath.basename(target), data, 'application/octet-stream'),
                        request_options=_ONCE,
                    )
                except ApiError as error:
                    # Islo answers a write the path refuses with a bare 500, so the path is checked for why.
                    if error.status_code == 500 and (refusal := await _write_refusal(sandbox, target)) is not None:
                        raise refusal from error
                    raise

    # The other file operations are the ones Pydantic AI derives from commands, run as internal commands.
    async def stat(self, path: str) -> FileEntry:
        async with self._operation():
            sandbox = await self._connect()
            return await self._shell(sandbox).stat(await self._absolute(path))

    async def list_dir(self, path: str) -> Sequence[FileEntry]:
        async with self._operation():
            sandbox = await self._connect()
            return await self._shell(sandbox).list_dir(await self._absolute(path))

    async def make_dir(self, path: str) -> None:
        async with self._operation():
            sandbox = await self._connect()
            await self._shell(sandbox).make_dir(await self._absolute(path))

    async def remove(self, path: str) -> None:
        async with self._operation():
            sandbox = await self._connect()
            await self._shell(sandbox).remove(await self._absolute(path))

    async def exists(self, path: str) -> bool:
        async with self._operation():
            sandbox = await self._connect()
            return await self._shell(sandbox).exists(await self._absolute(path))


async def _write_refusal(sandbox: _Sandbox, path: str) -> OSError | None:
    """The built-in error for a write Islo's file API refused without saying why, if the path explains it."""
    result = await _exec(sandbox, ['sh', '-c', _WRITE_REFUSAL, 'sh', path], cwd=None)
    if result.exit_code == _IS_A_DIRECTORY:
        return IsADirectoryError(errno.EISDIR, 'Is a directory in the Islo sandbox', path)
    if result.exit_code == _NOT_A_DIRECTORY:
        return NotADirectoryError(errno.ENOTDIR, 'Not a directory in the Islo sandbox', path)
    if result.exit_code == _NOT_REGULAR:
        return OSError(f'Not a regular file in the Islo sandbox: {path!r}')
    return None
