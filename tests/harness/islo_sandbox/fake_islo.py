"""A stateful fake of the Islo SDK's `AsyncIslo`, the boundary `IsloSandboxBackend` calls.

The answers are the SDK's own types: responses are the real `SandboxResponse`, `ExecResponse` and
`ExecResultResponse`, and failures the real `ApiError` subclasses with the bodies Islo sends (observed
live on 2026-10-22), so the backend's translation runs against what production raises. As in the SDK,
some errors carry a parsed `ErrorResponse` and others the raw JSON `dict`.

Each sandbox is a host directory: commands are real processes started there, so `run()`'s wrapper,
its stop and the shell file operations run as they would in a sandbox, and file calls act on the host
filesystem with Islo's quirks (absolute paths only, a directory read as a missing file, a bare 500
for a write the path refuses).

Knobs, keyed by method name (`create_sandbox`, `get_sandbox`, `get_sandbox_by_id`, `delete_sandbox`,
`resume_sandbox`, `exec_in_sandbox`, `get_exec_result`, `download_file`, `upload_file`):

- `errors[method]`: exceptions raised, in order, instead of handling the call; a `None` entry lets that
  call through.
- `late_errors[method]`: exceptions raised, in order, after handling the call, such as an
  `httpx.ReadTimeout` for a reply that was lost; a `None` entry lets that call's answer through.
- `gates[method]`: an event a call waits for before it is handled; `reached[method]` is set first.

`calls` records every call with its arguments.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import signal
import stat
import uuid
from collections import defaultdict
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import httpx
from islo.core.api_error import ApiError
from islo.core.file import File
from islo.core.request_options import RequestOptions
from islo.errors import ConflictError, NotFoundError, PaymentRequiredError
from islo.types import (
    ErrorCode,
    ErrorResponse,
    ExecResponse,
    ExecResultResponse,
    FileUploadStatusResponse,
    LifecyclePolicy,
    SandboxResponse,
)

DEFAULT_IMAGE = 'ghcr.io/islo-labs/islo-runner:latest'
# The most of each stream Islo keeps of a command's output.
STREAM_LIMIT = 1024 * 1024
_MARKER = re.compile(r'pydantic-ai-[0-9a-f]{32}')


def error_body(code: ErrorCode, message: str) -> ErrorResponse:
    return ErrorResponse(code=code, message=message)


def not_found(name: str) -> NotFoundError:
    return NotFoundError(body={'code': 'SANDBOX_NOT_FOUND', 'message': f"Sandbox '{name}' not found"})


def invalid_state() -> ConflictError:
    return ConflictError(
        body=error_body(ErrorCode.SANDBOX_INVALID_STATE, 'Sandbox is not in the required state for this operation')
    )


def unavailable() -> ApiError:
    return ApiError(
        status_code=503, body={'code': 'RESOURCE_UNAVAILABLE', 'message': 'This operation is currently unavailable'}
    )


def internal_error() -> ApiError:
    return ApiError(status_code=500, body={'code': 'INTERNAL_ERROR', 'message': 'An internal error occurred'})


def billing(action: str) -> PaymentRequiredError:
    return PaymentRequiredError(
        body=error_body(ErrorCode.BILLING_NOT_ALLOWED, f'Insufficient credit balance to {action} a sandbox')
    )


@dataclass
class FakeExec:
    process: asyncio.subprocess.Process
    output: asyncio.Task[tuple[bytes, bytes]]
    command: list[str]
    env: dict[str, str | None]


@dataclass
class FakeSandbox:
    name: str
    id: str
    root: Path
    status: str = 'running'
    image: str = DEFAULT_IMAGE
    lifecycle: LifecyclePolicy | None = None
    options: dict[str, object] = field(default_factory=dict[str, object])
    execs: dict[str, FakeExec] = field(default_factory=dict[str, FakeExec])
    resumes: int = 0

    def record(self) -> SandboxResponse:
        return SandboxResponse(
            created_at=datetime.now(UTC).isoformat(),
            id=self.id,
            image=self.image,
            name=self.name,
            status=self.status,
            lifecycle=self.lifecycle,
        )

    def kill(self) -> None:
        """Kill every process this sandbox's commands started, as deleting a VM would."""
        for run in self.execs.values():
            groups = [run.process.pid]
            # `run()`'s wrapper records the process group `setsid` gave the command.
            for word in run.command:
                for marker in _MARKER.findall(word):
                    with suppress(OSError, ValueError):
                        groups.append(int(Path(f'/tmp/{marker}.pid').read_text(encoding='ascii')))
            for group in groups:
                with suppress(ProcessLookupError, PermissionError):
                    os.killpg(group, signal.SIGKILL)


class FakeSandboxesClient:
    def __init__(self, owner: FakeIslo) -> None:
        self.owner = owner

    async def create_sandbox(
        self, *, request_options: RequestOptions | None = None, **options: object
    ) -> SandboxResponse:
        await self.owner.enter('create_sandbox', options)
        name = options.pop('name')
        assert isinstance(name, str)
        if name in self.owner.sandboxes:
            raise ConflictError(body=error_body(ErrorCode.SANDBOX_ALREADY_EXISTS, f"Sandbox '{name}' already exists"))
        if (image := options.get('image', DEFAULT_IMAGE)) not in self.owner.images:
            raise ApiError(status_code=422, body={'code': 'VALIDATION_ERROR', 'message': f'unknown image: {image}'})
        lifecycle = options.pop('lifecycle', None)
        assert lifecycle is None or isinstance(lifecycle, LifecyclePolicy)
        sandbox = self.owner.add(name, status=self.owner.create_status, lifecycle=lifecycle)
        sandbox.image = str(image)
        sandbox.options = options
        await self.owner.leave('create_sandbox')
        return sandbox.record()

    async def get_sandbox(self, sandbox_name: str, *, request_options: RequestOptions | None = None) -> SandboxResponse:
        await self.owner.enter('get_sandbox', {'sandbox_name': sandbox_name})
        sandbox = self.owner.find(sandbox_name)
        # A sandbox that is starting reports `running` after one more lookup.
        record = sandbox.record()
        if sandbox.status in ('creating', 'resuming'):
            sandbox.status = 'running'
        await self.owner.leave('get_sandbox')
        return record

    async def get_sandbox_by_id(self, id: str, *, request_options: RequestOptions | None = None) -> SandboxResponse:
        await self.owner.enter('get_sandbox_by_id', {'id': id})
        # Unlike a lookup by name, this one finds a deleted sandbox too.
        for sandbox in [*self.owner.sandboxes.values(), *self.owner.deleted]:
            if sandbox.id == id:
                await self.owner.leave('get_sandbox_by_id')
                return sandbox.record()
        raise NotFoundError(body={'code': 'SANDBOX_NOT_FOUND', 'message': f"Sandbox '{id}' not found"})

    async def delete_sandbox(self, sandbox_name: str, *, request_options: RequestOptions | None = None) -> None:
        await self.owner.enter('delete_sandbox', {'sandbox_name': sandbox_name})
        self.owner.find(sandbox_name)
        self.owner.delete(sandbox_name)
        await self.owner.leave('delete_sandbox')

    async def resume_sandbox(
        self, sandbox_name: str, *, request_options: RequestOptions | None = None
    ) -> SandboxResponse:
        await self.owner.enter('resume_sandbox', {'sandbox_name': sandbox_name})
        sandbox = self.owner.find(sandbox_name)
        sandbox.resumes += 1
        sandbox.status = 'resuming'
        await self.owner.leave('resume_sandbox')
        return sandbox.record()

    async def exec_in_sandbox(
        self,
        sandbox_name: str,
        *,
        command: Sequence[str],
        env: dict[str, str | None] | None = None,
        request_options: RequestOptions | None = None,
    ) -> ExecResponse:
        await self.owner.enter('exec_in_sandbox', {'command': list(command), 'env': env})
        sandbox = self.owner.running(sandbox_name, invalid_state)
        exec_id = str(uuid.uuid4())
        if (delay := self.owner.late_start) is not None:
            # Islo took the request, but its reply is lost and the command starts only after `delay`.
            self.owner.late_start = None

            async def start_late() -> None:
                await asyncio.sleep(delay)
                await self.owner.spawn(sandbox, exec_id, command, env)

            self.owner.background.append(asyncio.ensure_future(start_late()))
            raise httpx.ReadTimeout('reply lost')
        await self.owner.spawn(sandbox, exec_id, command, env)
        await self.owner.leave('exec_in_sandbox')
        return ExecResponse(exec_id=exec_id, sandbox_id=sandbox.id, status='started')

    async def get_exec_result(
        self, sandbox_name: str, exec_id: str, *, request_options: RequestOptions | None = None
    ) -> ExecResultResponse:
        await self.owner.enter('get_exec_result', {'exec_id': exec_id})
        sandbox = self.owner.running(sandbox_name, invalid_state)
        run = sandbox.execs[exec_id]
        if not run.output.done():
            await asyncio.sleep(0)
        if not run.output.done() or self.owner.hold_results:
            await self.owner.leave('get_exec_result')
            return ExecResultResponse(exec_id=exec_id, status='running', stdout='', stderr='', truncated=False)
        stdout, stderr = run.output.result()
        code = run.process.returncode
        assert code is not None
        await self.owner.leave('get_exec_result')
        status, exit_code = self.owner.exec_outcome or ('completed', code if code >= 0 else 128 - code)
        return ExecResultResponse(
            exec_id=exec_id,
            status=status,
            exit_code=exit_code,
            stdout=stdout[-STREAM_LIMIT:].decode(errors='replace'),
            stderr=stderr[-STREAM_LIMIT:].decode(errors='replace'),
            truncated=len(stdout) > STREAM_LIMIT or len(stderr) > STREAM_LIMIT,
        )

    async def download_file(
        self, sandbox_name: str, *, path: str, request_options: RequestOptions | None = None
    ) -> AsyncIterator[bytes]:
        await self.owner.enter('download_file', {'path': path})
        self.owner.running(sandbox_name, unavailable)
        # Islo refuses a relative path; the backend never sends one.
        assert path.startswith('/'), path
        target = Path(path)
        # Islo answers a directory as a missing file and never answers for a FIFO, so the backend checks first.
        assert target.is_file(), f'the backend read what is not a regular file: {path}'
        data = target.read_bytes()
        await self.owner.leave('download_file')
        for start in range(0, len(data), 1024 * 1024):
            yield data[start : start + 1024 * 1024]

    async def upload_file(
        self, sandbox_name: str, *, path: str, file: File, request_options: RequestOptions | None = None
    ) -> FileUploadStatusResponse:
        await self.owner.enter('upload_file', {'path': path})
        self.owner.running(sandbox_name, unavailable)
        assert isinstance(file, tuple) and len(file) == 3
        content = file[1]
        assert isinstance(content, bytes)
        assert path.startswith('/'), path
        target = Path(path)
        try:
            # Like Islo: missing parents are created, and a symlink is written through.
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and not stat.S_ISREG(target.stat().st_mode):
                raise IsADirectoryError(path)
            target.write_bytes(content)
        except OSError:
            raise internal_error() from None
        await self.owner.leave('upload_file')
        return FileUploadStatusResponse(status='ok')


class FakeClient:
    """Stands in for `islo.AsyncIslo`; records the arguments it was built with."""

    def __init__(self, owner: FakeIslo, kwargs: Mapping[str, object]) -> None:
        self.kwargs = dict(kwargs)
        self.sandboxes = FakeSandboxesClient(owner)


class FakeIslo:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.api_key = 'fake-islo-key'
        self.sandboxes: dict[str, FakeSandbox] = {}
        self.deleted: list[FakeSandbox] = []
        self.clients: list[FakeClient] = []
        self.images = {DEFAULT_IMAGE, 'python:3.13'}
        # The state a new sandbox starts in; `creating` reports `running` from its next lookup on.
        self.create_status = 'running'
        # Answers every command check with `running` while set.
        self.hold_results = False
        # Replaces a finished command's status and exit code.
        self.exec_outcome: tuple[str, int | None] | None = None
        # When set, the next command's reply is lost and the command starts this many seconds later.
        self.late_start: float | None = None
        self.background: list[asyncio.Future[None]] = []
        # Every command started, in order.
        self.runs: list[FakeExec] = []
        self.errors: defaultdict[str, list[BaseException | None]] = defaultdict(list)
        self.late_errors: defaultdict[str, list[BaseException | None]] = defaultdict(list)
        self.gates: dict[str, asyncio.Event] = {}
        self.reached: defaultdict[str, asyncio.Event] = defaultdict(asyncio.Event)
        self.calls: list[tuple[str, dict[str, object]]] = []

    def client(self, **kwargs: object) -> FakeClient:
        client = FakeClient(self, kwargs)
        self.clients.append(client)
        return client

    def add(
        self, name: str | None = None, *, status: str = 'running', lifecycle: LifecyclePolicy | None = None
    ) -> FakeSandbox:
        """A sandbox that exists in this fake Islo, as if created elsewhere."""
        name = name or f'existing-{len(self.sandboxes) + 1}'
        root = self.root / name
        root.mkdir(parents=True)
        sandbox = FakeSandbox(name=name, id=str(uuid.uuid4()), root=root.resolve(), status=status, lifecycle=lifecycle)
        self.sandboxes[name] = sandbox
        return sandbox

    def find(self, name: str) -> FakeSandbox:
        try:
            return self.sandboxes[name]
        except KeyError:
            raise not_found(name) from None

    def running(self, name: str, refusal: Callable[[], ApiError]) -> FakeSandbox:
        """The sandbox named `name`, or `refusal()` when it is not running."""
        sandbox = self.find(name)
        if sandbox.status != 'running':
            raise refusal()
        return sandbox

    def delete(self, name: str) -> None:
        sandbox = self.sandboxes.pop(name)
        sandbox.status = 'deleted'
        self.deleted.append(sandbox)
        sandbox.kill()
        shutil.rmtree(sandbox.root, ignore_errors=True)

    async def enter(self, method: str, arguments: Mapping[str, object]) -> None:
        self.calls.append((method, dict(arguments)))
        self.reached[method].set()
        if (gate := self.gates.get(method)) is not None:
            await gate.wait()
        if self.errors[method] and (error := self.errors[method].pop(0)) is not None:
            raise error

    async def leave(self, method: str) -> None:
        if self.late_errors[method] and (error := self.late_errors[method].pop(0)) is not None:
            raise error

    async def spawn(
        self, sandbox: FakeSandbox, exec_id: str, command: Sequence[str], env: dict[str, str | None] | None
    ) -> None:
        """Start `command` in the sandbox's directory, in a session of its own, with Islo's `env` added."""
        environment = {**os.environ, **{name: value for name, value in (env or {}).items() if value is not None}}
        # Islo runs a command as root in the sandbox's workdir; this one runs as the host user in its directory.
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=sandbox.root,
            env=environment,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        run = FakeExec(process, asyncio.ensure_future(process.communicate()), list(command), dict(env or {}))
        sandbox.execs[exec_id] = run
        self.runs.append(run)

    def methods(self) -> list[str]:
        return [method for method, _ in self.calls]

    async def close(self) -> None:
        """Kill what the tests' commands left running, and reap them."""
        for task in self.background:
            await task
        for sandbox in [*self.sandboxes.values(), *self.deleted]:
            sandbox.kill()
            for run in sandbox.execs.values():
                await run.output
