"""A stateful fake of Boat's Python SDK (`boat-sdk`), at the boundary the backend calls.

`FakeBoat.client` stands in for `boat_sdk.BoatApi`, and `FakeBoat.follow_command` for
`boat_sdk.boat_helpers.follow_command`, the SDK helper that streams a detached command's output over
HTTP; the other helpers the backend calls run for real against the fake client. Answers are the
SDK's own models, and refusals its own exceptions carrying Boat's real codes and messages (observed
live on 2026-10-21), so the backend's translation runs against what production returns. The backend
calls the SDK in worker threads, so the fake is called in them too.

Each sandbox is a pair of host directories, its home and its `/tmp`: commands are real `bash -c`
processes in their own session, so stopping one kills its process group, and file calls act on the
host filesystem, limited to those two directories like Boat's file API.

Knobs, all keyed by route name (`create`, `get`, `update`, `resume`, `delete`, `run`, `start`,
`follow`, `stop`, `read`, `write`, `upload_url`, `download_url`, `transfer`):

- `errors[route]`: exceptions raised, in order, instead of handling the call; a `None` entry lets that
  call through. `run` and `start` also take an SDK answer to return, `transfer` an `Answer`, and `follow`
  a `Frames` to serve.
- `lost_replies[route]`: how many calls are handled and then fail as a reply lost to a read timeout.
- `gates[route]`: a `threading.Event` a call waits for before it is handled; `reached[route]` is set first.

`calls` records every call that reached a route, with its arguments.

Boat's refusals of a malformed request are kept as tripwires, marked uncovered: a correct backend
never sends one, and the refusal fails the test that made it.
"""

from __future__ import annotations

import base64
import codecs
import errno
import json
import os
import signal
import stat
import subprocess
import threading
import time
import uuid
from collections import defaultdict
from collections.abc import Iterator
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from http import HTTPStatus
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlsplit

import boat_sdk
import urllib3
from boat_sdk.exceptions import (
    ApiException,
    BadRequestException,
    ConflictException,
    ForbiddenException,
    NotFoundException,
    ServiceException,
    UnauthorizedException,
)

MACHINE_HOST = 'box-node-fake.on.boat.dev'
MACHINE_TYPES = ('small', 'default', 'large', 'xlarge')
_ID_ALPHABET = '23456789abcdefghjkmnpqrstuvwxyz'
_STREAM_LIMIT = 8 * 1024 * 1024
_FILE_LIMIT = 5 * 1024 * 1024
_COMMAND_LIMIT = 131071
_ROOTS_MESSAGE = 'Path must resolve under /home/user or /tmp (work directory: /home/user).'
_NODE_ERRORS = {
    errno.ENOENT: 'ENOENT: no such file or directory',
    errno.EACCES: 'EACCES: permission denied',
    errno.ENOTDIR: 'ENOTDIR: not a directory',
    errno.EISDIR: 'EISDIR: illegal operation on a directory',
    errno.ELOOP: 'ELOOP: too many symbolic links encountered',
}
_EXCEPTIONS: dict[int, type[ApiException]] = {
    400: BadRequestException,
    401: UnauthorizedException,
    403: ForbiddenException,
    404: NotFoundException,
    409: ConflictException,
}
# A pool for the network errors the fake raises; building one opens no connection.
_POOL = urllib3.HTTPSConnectionPool('boat.dev')


def boat_error(status: int, code: str, message: str, **details: object) -> ApiException:
    """A refusal in Boat's envelope, as the SDK raises it: its exception class for the status."""
    body = {
        'ok': False,
        'type': 'sandbox.error',
        'status': status,
        'code': code,
        'message': message,
        'error': {'code': code, 'message': message, 'status': status, 'details': {'error': code, **details}},
        'requestId': 'req_00000000000000000000000000000000',
    }
    kind = _EXCEPTIONS.get(status, ServiceException if status >= 500 else ApiException)
    return kind(status=status, reason=HTTPStatus(status).phrase, body=json.dumps(body))


def answered(error: BaseException) -> tuple[object, object]:
    """The HTTP status and body of one of the SDK's errors, which it leaves untyped."""
    fields = vars(error)
    return fields.get('status'), fields.get('body')


def lost_reply() -> urllib3.exceptions.ReadTimeoutError:
    """What the SDK raises when Boat's reply to a request it does not send again is lost."""
    return urllib3.exceptions.ReadTimeoutError(_POOL, '/api/v1', 'Read timed out.')


def unreachable() -> urllib3.exceptions.MaxRetryError:
    """What the SDK raises once it gives up connecting: the request never reached Boat."""
    reason = urllib3.exceptions.ConnectTimeoutError(None, 'Connection to boat.dev timed out.')
    return urllib3.exceptions.MaxRetryError(_POOL, '/api/v1', reason)


def gave_up_reading() -> urllib3.exceptions.MaxRetryError:
    """What the SDK raises once it gives up on a request it sends again, after every reply timed out."""
    return urllib3.exceptions.MaxRetryError(_POOL, '/api/v1', lost_reply())


def finished(
    *, exit_code: int | None = 0, stdout: str = '', stderr: str = '', timed_out: bool = False, truncated: bool = False
) -> boat_sdk.Command200Response:
    """Boat's answer to a synchronous command."""
    return boat_sdk.Command200Response(
        actual_instance=boat_sdk.CommandResponse.model_validate(
            {
                'ok': True,
                'type': 'command.finished',
                'success': exit_code == 0,
                'exitCode': exit_code,
                'stdout': stdout,
                'stderr': stderr,
                'stdoutTruncated': truncated,
                'stderrTruncated': False,
                'timedOut': timed_out,
            }
        )
    )


def detached(pid: int, command: str = 'true') -> boat_sdk.Command200Response:
    """Boat's answer to a detached command."""
    return boat_sdk.Command200Response(
        actual_instance=boat_sdk.CommandStartedResponse.model_validate(
            {
                'ok': True,
                'type': 'command.started',
                'success': True,
                'processId': pid,
                'pid': pid,
                'command': command,
                'cwd': '.',
                'startedAt': datetime.now(UTC).isoformat(),
            }
        )
    )


@dataclass
class Frames:
    """Frames `follow_command` serves instead of following a command."""

    frames: list[dict[str, object]]


@dataclass
class Call:
    """A call that reached one of the fake's routes."""

    route: str
    arguments: dict[str, object] = field(default_factory=dict[str, object])


class FakeCommand:
    """A detached command: a host process writing its output to log files, as Boat's agent does."""

    def __init__(self, process: subprocess.Popen[bytes], out: Path, err: Path) -> None:
        self.process = process
        self.out = out
        self.err = err
        self.stopped = False

    def kill(self) -> None:
        """Kill the command's process group, as Boat kills a stopped command's whole tree."""
        # The group outlives its leader while a background child is left in it.
        with suppress(ProcessLookupError, PermissionError):
            os.killpg(self.process.pid, signal.SIGKILL)
        self.process.wait()


class FakeSandbox:
    def __init__(self, sandbox_id: str, root: Path, *, machine_type: str) -> None:
        self.id = sandbox_id
        self.state = 'provisioning'
        self.machine_type = machine_type
        self.home = root / sandbox_id / 'home'
        self.tmp = root / sandbox_id / 'tmp'
        self.logs = root / sandbox_id / 'logs'
        for directory in (self.home, self.tmp, self.logs):
            directory.mkdir(parents=True)
        # The states successive lookups find it in, such as while it starts; then it is ready.
        self.next_states: list[str] = []
        # Boat's auto-stop in seconds, `None` when it is off.
        self.ttl: int | None = 3600
        self.resume_calls = 0
        self.commands: dict[int, FakeCommand] = {}

    def model(self) -> boat_sdk.Sandbox:
        return boat_sdk.Sandbox.model_validate(
            {
                'id': self.id,
                'name': f'Box {self.id}',
                'state': self.state,
                'type': self.machine_type,
                'ttlSeconds': self.ttl,
                'desktopAvailable': False,
                'snapshotAvailable': False,
            }
        )

    def environment(self) -> dict[str, str]:
        return {'PATH': os.environ.get('PATH', '/usr/bin:/bin'), 'HOME': str(self.home), 'TMPDIR': str(self.tmp)}

    def kill_all(self) -> None:
        for command in self.commands.values():
            command.kill()


class Answer:
    """What the SDK's connection pool returns: enough of `urllib3.BaseHTTPResponse` for the backend and the SDK."""

    def __init__(self, status: int, data: bytes) -> None:
        self.status = status
        self.reason = HTTPStatus(status).phrase
        self.data = data
        self.headers: dict[str, str] = {}


class _Pool:
    """The SDK client's connection pool, which the backend sends signed file transfers through."""

    def __init__(self, fake: FakeBoat) -> None:
        self._fake = fake

    def request(
        self,
        method: str,
        url: str,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
        timeout: object = None,
    ) -> Answer:
        # The URL is signed: Boat's API key must not travel to the sandbox's machine.
        assert 'Authorization' not in (headers or {}), 'the Boat API key was sent to a file-transfer URL'
        if isinstance(answer := self._fake.enter('transfer', method=method, url=url), Answer):
            return answer
        return self._fake.transfer(method, url, body or b'')


class _Rest:
    def __init__(self, fake: FakeBoat) -> None:
        self.pool_manager = _Pool(fake)


class _ApiClient:
    def __init__(self, fake: FakeBoat) -> None:
        self.rest_client = _Rest(fake)


class FakeClient:
    """Stands in for `boat_sdk.BoatApi`: each method is the SDK call of the same name, answered by the fake."""

    def __init__(self, fake: FakeBoat, api_key: str) -> None:
        self.fake = fake
        self.api_key = api_key
        self.api_client = _ApiClient(fake)
        # The SDK's own client the backend built this one with, if it did.
        self.sdk_client: boat_sdk.ApiClient | None = None

    def check_key(self) -> None:
        if self.api_key != self.fake.api_key:
            raise boat_error(401, 'unauthorized', 'unauthorized')

    def create(
        self,
        idempotency_key: str | None = None,
        create_sandbox_request: boat_sdk.CreateSandboxRequest | None = None,
        _request_timeout: object = None,
    ) -> boat_sdk.CreateSandboxResponse:
        self.check_key()
        assert create_sandbox_request is not None
        body = create_sandbox_request.to_dict()
        self.fake.enter('create', idempotency_key=idempotency_key, body=body)
        answer = self.fake.create(idempotency_key, body)
        self.fake.lose_reply('create')
        return answer

    def get(self, sandbox_id: str, _request_timeout: object = None) -> boat_sdk.SandboxInfoResponse:
        self.check_key()
        self.fake.enter('get', sandbox_id=sandbox_id)
        return self.fake.get(sandbox_id)

    def update(
        self, sandbox_id: str, update_sandbox_request: boat_sdk.UpdateSandboxRequest, _request_timeout: object = None
    ) -> boat_sdk.SandboxInfoResponse:
        self.check_key()
        body = update_sandbox_request.to_dict()
        self.fake.enter('update', sandbox_id=sandbox_id, body=body)
        return self.fake.update(sandbox_id, body)

    def resume(self, sandbox_id: str, _request_timeout: object = None) -> boat_sdk.SandboxActionResponse:
        self.check_key()
        self.fake.enter('resume', sandbox_id=sandbox_id)
        return self.fake.resume(sandbox_id)

    def delete_sandbox(
        self, x_ascii_confirm_delete: str, sandbox_id: str, _request_timeout: object = None
    ) -> boat_sdk.DeletionOperationResponse:
        self.check_key()
        self.fake.enter('delete', sandbox_id=sandbox_id, confirm=x_ascii_confirm_delete)
        return self.fake.delete(sandbox_id, x_ascii_confirm_delete)

    def command(
        self, sandbox_id: str, command_request: boat_sdk.CommandRequest, _request_timeout: object = None
    ) -> boat_sdk.Command200Response:
        self.check_key()
        body = command_request.to_dict()
        route = 'start' if command_request.detached else 'run'
        if isinstance(answer := self.fake.enter(route, sandbox_id=sandbox_id, body=body), boat_sdk.Command200Response):
            return answer
        result = self.fake.command(sandbox_id, command_request)
        self.fake.lose_reply(route)
        return result

    def stop_command(
        self, sandbox_id: str, process_id: int, _request_timeout: object = None
    ) -> boat_sdk.CommandStopResponse:
        self.check_key()
        self.fake.enter('stop', sandbox_id=sandbox_id, process_id=process_id)
        return self.fake.stop_command(sandbox_id, process_id)

    def read_file(
        self, sandbox_id: str, path: str, encoding: str | None = None, _request_timeout: object = None
    ) -> boat_sdk.FileReadResponse:
        self.check_key()
        self.fake.enter('read', sandbox_id=sandbox_id, path=path, encoding=encoding)
        return self.fake.read_file(sandbox_id, path, encoding)

    def write_file(
        self, sandbox_id: str, file_write_request: boat_sdk.FileWriteRequest, _request_timeout: object = None
    ) -> boat_sdk.FileWriteResponse:
        self.check_key()
        self.fake.enter('write', sandbox_id=sandbox_id, path=file_write_request.path)
        answer = self.fake.write_file(sandbox_id, file_write_request)
        self.fake.lose_reply('write')
        return answer

    def create_download_url(
        self, sandbox_id: str, file_url_request: boat_sdk.FileUrlRequest, _request_timeout: object = None
    ) -> boat_sdk.FileUrlResponse:
        self.check_key()
        self.fake.enter('download_url', sandbox_id=sandbox_id, path=file_url_request.path)
        return self.fake.transfer_url(sandbox_id, file_url_request.path, 'read')

    def create_upload_url(
        self, sandbox_id: str, file_url_request: boat_sdk.FileUrlRequest, _request_timeout: object = None
    ) -> boat_sdk.FileUrlResponse:
        self.check_key()
        self.fake.enter('upload_url', sandbox_id=sandbox_id, path=file_url_request.path)
        return self.fake.transfer_url(sandbox_id, file_url_request.path, 'write')


class FakeBoat:
    def __init__(self, root: Path, api_key: str) -> None:
        self.root = root
        self.api_key = api_key
        self.sandboxes: dict[str, FakeSandbox] = {}
        self.deleted: set[str] = set()
        # Named snapshots a create may start from.
        self.snapshots: set[str] = set()
        self.create_bodies: list[dict[str, object]] = []
        self.calls: list[Call] = []
        self.errors: dict[str, list[object]] = defaultdict(list)
        self.lost_replies: dict[str, int] = defaultdict(int)
        self.gates: dict[str, threading.Event] = {}
        self.reached: dict[str, threading.Event] = defaultdict(threading.Event)
        self.clients: list[FakeClient] = []
        self._lock = threading.Lock()
        self._created = 0
        self._started = 0
        self._idempotency: dict[str, tuple[dict[str, object], str]] = {}

    # --- Control surface ---------------------------------------------------------------------------------------------

    def client(self, api_client: boat_sdk.ApiClient | None = None) -> FakeClient:
        """Stands in for `boat_sdk.BoatApi(api_client)`, authenticated with the key of the client's configuration."""
        key: object = getattr(getattr(api_client, 'configuration', None), 'access_token', None)
        client = FakeClient(self, key if isinstance(key, str) else '')
        client.sdk_client = api_client
        self.clients.append(client)
        return client

    def routes(self) -> list[str]:
        return [call.route for call in self.calls]

    def bodies(self, route: str) -> list[object]:
        """The request bodies of the calls that reached `route`, in order."""
        return [call.arguments['body'] for call in self.calls if call.route == route]

    def sandbox(self, state: str = 'idle') -> FakeSandbox:
        """A sandbox that exists before the test, as one an earlier run created."""
        sandbox = self._new_sandbox('default')
        sandbox.state = state
        return sandbox

    def stop(self, sandbox_id: str) -> None:
        """Stop a sandbox as Boat's auto-stop does; its processes keep running, as on Boat's first minutes."""
        self.sandboxes[sandbox_id].state = 'archived'

    def close(self) -> None:
        for gate in self.gates.values():
            gate.set()
        for sandbox in self.sandboxes.values():
            sandbox.kill_all()

    # --- Dispatch ----------------------------------------------------------------------------------------------------

    def enter(self, route: str, **arguments: object) -> object:
        """Record a call and hold it at its gate; raise what a test injected for it, or return a non-exception answer."""
        self.calls.append(Call(route, arguments))
        self.reached[route].set()
        if (gate := self.gates.get(route)) is not None:
            assert gate.wait(30), f'the {route!r} gate was never opened'
        injected = self.errors[route].pop(0) if self.errors[route] else None
        if isinstance(injected, BaseException):
            raise injected
        return injected

    def lose_reply(self, route: str) -> None:
        """Fail a handled call as its reply lost, if a test asked for that."""
        if self.lost_replies[route]:
            self.lost_replies[route] -= 1
            # The SDK sends a `PUT` again until it gives up; a `POST` it does not.
            raise gave_up_reading() if route == 'write' else lost_reply()

    def _sandbox(self, sandbox_id: str) -> FakeSandbox:
        sandbox = self.sandboxes.get(sandbox_id)
        if sandbox is None:
            raise boat_error(404, 'not_found', 'not_found')
        return sandbox

    def _running(self, sandbox_id: str, *, files: bool = False) -> FakeSandbox:
        """The sandbox an agent call reaches, or Boat's refusal while it is stopped.

        While a stop is still archiving the sandbox, its machine answers file calls but refuses commands.
        """
        sandbox = self._sandbox(sandbox_id)
        if sandbox.state == 'archived':
            raise boat_error(409, 'machine_not_running', 'Sandbox machine is not running.')
        if sandbox.state == 'archiving' and not files:
            message = 'Sandbox is not ready (state: archiving). Resume or recover it before running commands.'
            raise boat_error(409, 'sandbox_not_ready', message, retryable=False, state='archiving')
        return sandbox

    def _new_sandbox(self, machine_type: str) -> FakeSandbox:
        with self._lock:
            number, self._created = self._created, self._created + 1
        suffix = ''
        for _ in range(8):
            number, digit = divmod(number, len(_ID_ALPHABET))
            suffix = _ID_ALPHABET[digit] + suffix
        sandbox = FakeSandbox(f'bx_{suffix}', self.root, machine_type=machine_type)
        self.sandboxes[sandbox.id] = sandbox
        return sandbox

    # --- Sandboxes ---------------------------------------------------------------------------------------------------

    def create(self, key: str | None, body: dict[str, object]) -> boat_sdk.CreateSandboxResponse:
        assert key is not None, 'every creation is sent with an idempotency key, so a retry finds its sandbox'
        if key in self._idempotency:
            earlier, sandbox_id = self._idempotency[key]
            if earlier != body:  # pragma: no cover - tripwire: a retry must repeat the body it was first sent with
                message = 'This Idempotency-Key was already used with a different request body.'
                raise boat_error(409, 'idempotency_key_reused', message)
            sandbox = self.sandboxes[sandbox_id]
            return self._created_answer('ready', sandbox)
        machine_type = body.get('type', 'default')
        # The SDK refuses any other size before sending it.
        assert machine_type in MACHINE_TYPES
        snapshot = body.get('from')
        if snapshot is not None and snapshot not in self.snapshots:
            raise boat_error(
                404, 'snapshot_not_found', f'No named snapshot "{snapshot}". Run boat snapshots to list yours.'
            )
        ttl = body.get('ttlSeconds', 3600)
        assert isinstance(machine_type, str) and (ttl is None or isinstance(ttl, int))
        sandbox = self._new_sandbox(machine_type)
        sandbox.ttl = ttl
        self.create_bodies.append(body)
        self._idempotency[key] = (body, sandbox.id)
        return self._created_answer('provisioning', sandbox)

    @staticmethod
    def _created_answer(status: str, sandbox: FakeSandbox) -> boat_sdk.CreateSandboxResponse:
        return boat_sdk.CreateSandboxResponse.model_validate(
            {
                'ok': True,
                'type': 'sandbox.created',
                'status': status,
                'ttlSeconds': sandbox.ttl,
                'sandbox': sandbox.model().to_dict(),
            }
        )

    def get(self, sandbox_id: str) -> boat_sdk.SandboxInfoResponse:
        sandbox = self._sandbox(sandbox_id)
        if sandbox.next_states:
            sandbox.state = sandbox.next_states.pop(0)
        elif sandbox.state in ('provisioning', 'cloning'):
            sandbox.state = 'idle'
        return self._info('sandbox.info', sandbox)

    @staticmethod
    def _info(kind: str, sandbox: FakeSandbox) -> boat_sdk.SandboxInfoResponse:
        return boat_sdk.SandboxInfoResponse.model_validate(
            {'ok': True, 'type': kind, 'sandbox': sandbox.model().to_dict()}
        )

    def update(self, sandbox_id: str, body: dict[str, object]) -> boat_sdk.SandboxInfoResponse:
        sandbox = self._sandbox(sandbox_id)
        # The backend only sets the auto-stop again, with the sandbox's own.
        assert body == {'ttlSeconds': sandbox.ttl} and sandbox.ttl is not None
        return self._info('sandbox.updated', sandbox)

    def resume(self, sandbox_id: str) -> boat_sdk.SandboxActionResponse:
        sandbox = self._sandbox(sandbox_id)
        # The backend resumes only a sandbox it just found stopped.
        assert sandbox.state in ('archived', 'archiving')
        sandbox.resume_calls += 1
        sandbox.state = 'ready'
        return boat_sdk.SandboxActionResponse.model_validate(
            {
                'ok': True,
                'type': 'sandbox.resuming',
                'id': sandbox.id,
                'status': 'running',
                'sandbox': sandbox.model().to_dict(),
            }
        )

    def delete(self, sandbox_id: str, confirm: str) -> boat_sdk.DeletionOperationResponse:
        operation = {
            'id': f'bdop_{uuid.uuid4().hex}',
            'kind': 'sandbox',
            'targetId': sandbox_id,
            'reason': 'explicit',
            'status': 'pending',
            'attemptCount': 0,
            'requestedAt': datetime.now(UTC).isoformat(),
            'completedAt': None,
        }
        answer = boat_sdk.DeletionOperationResponse.model_validate(
            {'ok': True, 'type': 'sandbox.deleting', 'operation': operation}
        )
        if sandbox_id in self.deleted:
            return answer
        sandbox = self._sandbox(sandbox_id)
        if confirm != sandbox_id:  # pragma: no cover - tripwire
            message = f'Set X-Ascii-Confirm-Delete to the exact target id: {sandbox_id}'
            raise boat_error(409, 'delete_confirmation_required', message, targetId=sandbox_id)
        # Deleting a sandbox stops its machine at once, and every process on it. Forgotten first, so a
        # command's output stream breaks off rather than reporting an exit.
        del self.sandboxes[sandbox_id]
        self.deleted.add(sandbox_id)
        sandbox.kill_all()
        return answer

    # --- Commands ----------------------------------------------------------------------------------------------------

    def command(self, sandbox_id: str, request: boat_sdk.CommandRequest) -> boat_sdk.Command200Response:
        sandbox = self._running(sandbox_id)
        assert request.command and request.cwd is None
        if len(request.command.encode()) > _COMMAND_LIMIT:
            raise boat_error(400, 'sandbox_direct_failed', "E2BIG: argument list too long, posix_spawn 'systemd-run'")
        if request.detached:
            return self._start(sandbox, request.command)
        timeout = request.timeout_seconds
        assert timeout is not None and 1 <= timeout <= 600, 'Boat refuses a timeout outside 1-600 seconds'
        return _run_sync(sandbox, request.command, timeout)

    def _start(self, sandbox: FakeSandbox, command: str) -> boat_sdk.Command200Response:
        with self._lock:
            number, self._started = self._started, self._started + 1
        out, err = sandbox.logs / f'{number}.log', sandbox.logs / f'{number}.err.log'
        with out.open('wb') as stdout, err.open('wb') as stderr:
            process = subprocess.Popen(
                ['bash', '-c', command],
                cwd=sandbox.home,
                env=sandbox.environment(),
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
        sandbox.commands[process.pid] = FakeCommand(process, out, err)
        return detached(process.pid, command)

    def follow_command(self, api: object, sandbox_id: str, process_id: int) -> Iterator[dict[str, object]]:
        """Stands in for `boat_helpers.follow_command`: a detached command's output from its start, then its exit.

        As the SDK's helper, it reconnects through a dropped stream, so a sandbox deleted or stopped
        meanwhile surfaces as the refusal its reconnection gets.
        """
        assert isinstance(api, FakeClient)
        api.check_key()
        if isinstance(frames := self.enter('follow', sandbox_id=sandbox_id, process_id=process_id), Frames):
            yield from frames.frames
        elif (command := self._running(sandbox_id).commands.get(process_id)) is None:
            yield {'type': 'exit', 'status': 'lost', 'exitCode': None}
        else:
            yield from self._output(sandbox_id, command)

    def _output(self, sandbox_id: str, command: FakeCommand) -> Iterator[dict[str, object]]:
        """A command's output as its log files grow, then its exit."""
        offsets = {'stdout': 0, 'stderr': 0}
        decoders = {name: codecs.getincrementaldecoder('utf-8')(errors='replace') for name in offsets}
        code = None
        while code is None:
            code = command.process.poll()
            for name, path in (('stdout', command.out), ('stderr', command.err)):
                data = path.read_bytes()[offsets[name] :]
                offsets[name] += len(data)
                if text := decoders[name].decode(data, final=code is not None):
                    yield {'type': name, 'data': text, 'nextOffset': offsets[name]}
            # A sandbox deleted or stopped meanwhile breaks the stream off, and its reconnection is refused.
            self._running(sandbox_id)
            if code is None:
                time.sleep(0.005)
        yield {'type': 'exit', 'status': 'exited', 'exitCode': code if code >= 0 else 128 - code}

    def stop_command(self, sandbox_id: str, process_id: int) -> boat_sdk.CommandStopResponse:
        sandbox = self._running(sandbox_id)
        command = sandbox.commands.get(process_id)
        if command is None:
            raise boat_error(
                400, 'sandbox_direct_failed', f'Process {process_id} was not started by a detached command.'
            )
        signaled = command.process.poll() is None
        command.kill()
        command.stopped = True
        return boat_sdk.CommandStopResponse.model_validate(
            {
                'ok': True,
                'type': 'command.stopped',
                'processId': process_id,
                'running': False,
                'signaled': signaled,
                'command': None,
            }
        )

    # --- Files -------------------------------------------------------------------------------------------------------

    def _resolve(self, sandbox: FakeSandbox, raw: str) -> Path:
        """The host path of a file-API path, or Boat's refusal of a path outside its roots."""
        path = Path(os.path.normpath(sandbox.home / raw))
        if not _under(path, sandbox.home, sandbox.tmp):
            raise boat_error(400, 'invalid_path', _ROOTS_MESSAGE)
        # A symlink is followed, and its target must resolve under the roots as well.
        if not _under(Path(os.path.realpath(path)), sandbox.home, sandbox.tmp):
            raise boat_error(400, 'sandbox_direct_failed', _ROOTS_MESSAGE)
        return path

    def read_file(self, sandbox_id: str, raw: str, encoding: str | None) -> boat_sdk.FileReadResponse:
        sandbox = self._running(sandbox_id, files=True)
        path = self._resolve(sandbox, raw)
        try:
            info = path.stat()
        except OSError as error:
            raise boat_error(400, 'sandbox_direct_failed', _node_error(error, 'stat', path)) from None
        if not stat.S_ISREG(info.st_mode):
            # A FIFO too, without opening it.
            raise boat_error(400, 'sandbox_direct_failed', 'Path is not a file.')
        if info.st_size > _FILE_LIMIT:
            raise boat_error(
                400,
                'sandbox_direct_failed',
                f'File is too large for read_file ({info.st_size} bytes > {_FILE_LIMIT}). '
                f'Use a download URL instead: POST /sandboxes/{sandbox.id}/files/download-url.',
            )
        # Boat's file API reads as root, so a file it found is a file it can read.
        data = path.read_bytes()
        assert encoding == 'base64', 'a file read without base64 loses bytes that are not UTF-8'
        return boat_sdk.FileReadResponse.model_validate(
            {
                'ok': True,
                'type': 'file.read',
                'success': True,
                'path': str(path),
                'encoding': encoding,
                'size': len(data),
                'content': base64.b64encode(data).decode(),
            }
        )

    def write_file(self, sandbox_id: str, request: boat_sdk.FileWriteRequest) -> boat_sdk.FileWriteResponse:
        sandbox = self._running(sandbox_id, files=True)
        path = self._resolve(sandbox, request.path)
        assert request.encoding == 'base64', 'a file written without base64 cannot hold bytes that are not UTF-8'
        data = base64.b64decode(request.content)
        assert len(data) <= _FILE_LIMIT, 'a larger file goes through an upload URL'
        if path.is_symlink() and not path.exists():
            raise boat_error(400, 'sandbox_direct_failed', _ROOTS_MESSAGE)
        if path.exists() and not path.is_file() and not path.is_dir():
            # Boat's agent answers a write to a FIFO only once something opens it for reading, so the
            # SDK's request times out, and so do its repeats; opening it here would block.
            raise gave_up_reading()
        if (error := _write_file(path, data)) is not None:
            raise boat_error(400, 'sandbox_direct_failed', error)
        return boat_sdk.FileWriteResponse.model_validate(
            {
                'ok': True,
                'type': 'file.written',
                'success': True,
                'path': str(path),
                'encoding': 'base64',
                'size': len(data),
            }
        )

    def transfer_url(self, sandbox_id: str, raw: str, op: str) -> boat_sdk.FileUrlResponse:
        sandbox = self._running(sandbox_id)
        path = Path(os.path.normpath(sandbox.home / raw))
        # Boat signs the URL without looking at the file.
        if not _under(path, sandbox.home, sandbox.tmp):
            raise boat_error(400, 'invalid_path', _ROOTS_MESSAGE)
        query = urlencode({'op': op, 'sandbox': sandbox.id, 'path': str(path), 'sig': 'fake-signature'})
        return boat_sdk.FileUrlResponse.model_validate(
            {
                'ok': True,
                'type': f'file.{"upload" if op == "write" else "download"}_url',
                'url': f'https://{MACHINE_HOST}/file-transfer?{query}',
                'method': 'PUT' if op == 'write' else 'GET',
                'path': str(path),
                'expiresAt': datetime.now(UTC).isoformat(),
            }
        )

    def transfer(self, method: str, url: str, body: bytes) -> Answer:
        parts = urlsplit(url)
        assert parts.hostname == MACHINE_HOST, f'not a signed file-transfer URL: {url}'
        query = parse_qs(parts.query)
        sandbox = self.sandboxes.get(query['sandbox'][0])
        if sandbox is None or sandbox.state not in ('ready', 'idle', 'running'):
            # The URL points at the sandbox's machine, which stops answering once Boat stops it.
            raise urllib3.exceptions.ProtocolError('Connection aborted.')
        path = Path(query['path'][0])
        if query['op'][0] == 'read':
            assert method == 'GET'
            if not path.exists():
                return Answer(404, json.dumps({'error': 'not_found', 'message': 'File not found.'}).encode())
            # A download follows the file API refusing the file as too large, so it is one.
            return Answer(200, path.read_bytes())
        assert method == 'PUT'
        if path.exists() and not path.is_file():
            message = 'Path exists and is not a regular file.'
            return Answer(400, json.dumps({'error': 'not_a_file', 'message': message}).encode())
        if (error := _write_file(path, body)) is not None:
            return Answer(400, json.dumps({'error': 'write_failed', 'message': error}).encode())
        return Answer(200, json.dumps({'success': True, 'path': str(path), 'size': len(body)}).encode())


def _under(path: Path, *roots: Path) -> bool:
    return any(path == root or root in path.parents for root in roots)


def _node_error(error: OSError, syscall: str, path: Path) -> str:
    """The Node.js wording Boat's agent reports a failed file call in."""
    code = _NODE_ERRORS.get(error.errno or 0, f'EIO: i/o error ({error.strerror})')
    return f"{code}, {syscall} '{path}'"


def _write_file(path: Path, data: bytes) -> str | None:
    """Write like Boat's agent: missing parents are created, in place, through a symlink; or its error message."""
    for parent in reversed(path.parents):
        if parent.exists() and not parent.is_dir():
            return f"EEXIST: file already exists, mkdir '{parent}'"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('wb') as file:
            file.write(data)
    except OSError as error:
        return _node_error(error, 'open', path)
    return None


def _run_sync(sandbox: FakeSandbox, command: str, timeout: int) -> boat_sdk.Command200Response:
    """A synchronous command: Boat answers once its output closes or its timeout stops it."""
    process = subprocess.Popen(
        ['bash', '-c', command],
        cwd=sandbox.home,
        env=sandbox.environment(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    timed_out = False
    try:
        out, err = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:  # pragma: no cover - only an internal command that hangs reaches it
        timed_out = True
        with suppress(ProcessLookupError, PermissionError):
            os.killpg(process.pid, signal.SIGKILL)
        out, err = process.communicate()
    code = process.returncode
    return finished(
        exit_code=code if code >= 0 else 128 - code,
        stdout=out[-_STREAM_LIMIT:].decode(errors='replace'),
        stderr=err[-_STREAM_LIMIT:].decode(errors='replace'),
        timed_out=timed_out,
        truncated=len(out) > _STREAM_LIMIT,
    )
