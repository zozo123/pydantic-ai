"""Protocol and SDK-boundary tests for Boat sandboxes, over the fake Boat SDK.

They are not VCR tests: what they pin is how the backend reacts to answers Boat gives only on a bad
day (lost replies, refusals, a sandbox stopped or deleted mid-run, a command whose result is lost),
which a recording of a working session does not hold. The fake answers with the SDK's own models
and exceptions, and runs real commands.
"""

from __future__ import annotations

import asyncio
import errno
import importlib
import logging
import os
import pickle
import re
import sys
import threading
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import anyio
import boat_sdk
import pytest
import urllib3
from boat_sdk.exceptions import ApiException, ServiceException

from pydantic_ai.workspaces import (
    CommandResult,
    WorkspaceError,
    WorkspaceOutputLimitError,
    WorkspaceRef,
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
)
from pydantic_ai_harness.boat_sandbox import BoatSandbox, BoatSandboxBackend

from .fake_boat import (
    Answer,
    FakeBoat,
    FakeCommand,
    FakeSandbox,
    Frames,
    answered,
    boat_error,
    detached,
    finished,
    gave_up_reading,
    lost_reply,
    unreachable,
)

pytestmark = pytest.mark.anyio(backends=['asyncio'])

_BACKEND = 'pydantic_ai_harness.boat_sandbox._backend'
_AUTH = 'Boat rejected the credentials. Set BOAT_API_KEY, or pass a configured `boat_sdk.BoatApi` as `client=`.'
_FRESH = "Pass `workspace='new'` to start a fresh sandbox."
_WRAPPER = 'exec sh -c \'"$@" </dev/null; exit $?\' sh'


async def started(backend: BoatSandboxBackend) -> BoatSandboxBackend:
    """Resolve `backend` now: building one does no I/O, so a test that asserts on creating or attaching touches it first."""
    await backend.working_dir()
    return backend


def only(fake: FakeBoat) -> FakeSandbox:
    [sandbox] = fake.sandboxes.values()
    return sandbox


def only_command(fake: FakeBoat) -> FakeCommand:
    [command] = only(fake).commands.values()
    return command


async def until(condition: Callable[[], bool]) -> None:
    with anyio.fail_after(10):
        while not condition():
            await anyio.sleep(0.001)


def advance_clock(monkeypatch: pytest.MonkeyPatch) -> Callable[[float], None]:
    """A function that moves the clock the backend reads forward by some seconds, without waiting them out."""
    offset = 0.0

    def advance(seconds: float) -> None:
        nonlocal offset
        offset += seconds

    monkeypatch.setattr(f'{_BACKEND}.monotonic', lambda: time.monotonic() + offset)
    return advance


def sdk_client(api_key: str) -> boat_sdk.BoatApi:
    """An SDK client as an application builds one to pass as `client=`; the fake answers it."""
    return boat_sdk.BoatApi(boat_sdk.ApiClient(boat_sdk.Configuration(access_token=api_key)))


def unavailable(sandbox_id: str) -> str:
    return f'The Boat sandbox {sandbox_id!r} no longer exists (it was deleted). {_FRESH}'


def failed(sandbox_id: str) -> str:
    return f"The Boat sandbox {sandbox_id!r} is in the 'error' state and cannot be used. {_FRESH}"


def stopped(sandbox_id: str) -> str:
    return f'The Boat sandbox {sandbox_id!r} was stopped; a later run that attaches to it resumes it. {_FRESH}'


@pytest.mark.parametrize(
    ('build', 'error', 'message'),
    [
        pytest.param(
            lambda: BoatSandboxBackend(ref=WorkspaceRef(provider='daytona', id='x')),
            ValueError,
            r"^unsupported workspace provider 'daytona'; expected 'boat'$",
            id='foreign-ref',
        ),
        pytest.param(
            lambda: BoatSandboxBackend(ref=WorkspaceRef(provider='boat', id='  ')),
            ValueError,
            r'^Boat workspace ref id cannot be empty$',
            id='blank-ref',
        ),
        pytest.param(
            lambda: BoatSandboxBackend(working_dir='repo'),
            ValueError,
            r"^working_dir must be an absolute workspace path or None, got 'repo'\.$",
            id='relative-working-dir',
        ),
        pytest.param(
            lambda: BoatSandboxBackend(env={'PORT': 3000}),  # pyright: ignore[reportArgumentType]
            TypeError,
            r'^env keys and values must be strings$',
            id='non-string-env',
        ),
    ],
)
def test_invalid_settings_fail_at_construction(
    build: Callable[[], object], error: type[Exception], message: str
) -> None:
    # No fake: the autouse tripwire fails the test if construction touched the SDK.
    with pytest.raises(error, match=message):
        build()


def test_missing_boat_extra_has_an_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, 'boat_sdk', None)
    for name in [name for name in sys.modules if name.startswith('pydantic_ai_harness.boat_sandbox')]:
        monkeypatch.delitem(sys.modules, name)
    with pytest.raises(ImportError, match=r'^Install `pydantic-ai-harness\[boat\]` to use BoatSandbox\.$'):
        importlib.import_module('pydantic_ai_harness.boat_sandbox')


class TestCreation:
    @pytest.mark.parametrize(
        ('build', 'body'),
        [
            pytest.param(BoatSandboxBackend, {'noEnv': True}, id='defaults'),
            pytest.param(
                lambda: BoatSandboxBackend(machine_type='small', snapshot='base', auto_stop=900),
                {'noEnv': True, 'type': 'small', 'from': 'base', 'ttlSeconds': 900},
                id='settings',
            ),
            # Boat reads `0` as its default of an hour; `null` turns the auto-stop off.
            pytest.param(
                lambda: BoatSandboxBackend(auto_stop=0), {'noEnv': True, 'ttlSeconds': None}, id='no-auto-stop'
            ),
        ],
    )
    async def test_creation_sends_the_settings_and_keeps_the_account_credentials_out(
        self, fake_boat: FakeBoat, build: Callable[[], BoatSandboxBackend], body: dict[str, object]
    ) -> None:
        # Only what the backend sets is sent: none of the SDK's own defaults, such as an `environment`.
        fake_boat.snapshots.add('base')
        backend = await started(build())
        assert fake_boat.create_bodies == [body]
        [create] = [call.arguments for call in fake_boat.calls if call.route == 'create']
        assert re.fullmatch(r'pydantic-ai-[0-9a-f]{32}', str(create['idempotency_key']))
        assert backend.ref == WorkspaceRef(provider='boat', id=only(fake_boat).id)

    async def test_a_new_sandbox_gets_its_working_dir_and_an_attached_one_does_not(
        self, fake_boat: FakeBoat, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        directory = tmp_path.resolve() / 'project' / 'nested'
        with caplog.at_level(logging.INFO, logger=_BACKEND):
            backend = await started(BoatSandboxBackend(working_dir=str(directory)))
        sandbox = only(fake_boat)
        # Logged at creation, so a sandbox whose run ended before its ref was stored can be found.
        assert caplog.messages == [f'Created Boat sandbox {sandbox.id}']
        assert await backend.working_dir() == str(directory)
        made = [body for body in fake_boat.bodies('run') if 'mkdir' in str(body)]
        assert made == [
            {'command': f'{_WRAPPER} mkdir -p -- {directory}', 'timeoutSeconds': 30, 'detached': False, 'stream': False}
        ]

        attached = await started(BoatSandboxBackend(ref=backend.ref, working_dir=str(directory)))
        assert await attached.working_dir() == str(directory)
        assert [body for body in fake_boat.bodies('run') if 'mkdir' in str(body)] == made
        assert len(fake_boat.create_bodies) == 1

    async def test_a_failed_working_dir_is_retried_by_the_next_operation(
        self, fake_boat: FakeBoat, tmp_path: Path
    ) -> None:
        blocker = tmp_path.resolve() / 'file'
        blocker.write_bytes(b'')
        directory = blocker / 'project'
        backend = BoatSandboxBackend(working_dir=str(directory))
        with pytest.raises(
            WorkspaceError,
            match=rf'^Could not create working_dir {re.escape(repr(str(directory)))} in Boat sandbox bx_\w+: '
            r'mkdir: .*Not a directory$',
        ):
            await backend.run(['true'])
        # The sandbox is recorded, so the caller can clean it up, and the command never started.
        assert backend.ref == WorkspaceRef(provider='boat', id=only(fake_boat).id)
        assert fake_boat.bodies('start') == []
        blocker.unlink()
        assert await backend.working_dir() == str(directory)
        assert len([body for body in fake_boat.bodies('run') if 'mkdir' in str(body)]) == 2
        assert len(fake_boat.create_bodies) == 1

    async def test_a_lost_creation_reply_recovers_the_accepted_sandbox(self, fake_boat: FakeBoat) -> None:
        fake_boat.lost_replies['create'] = 1
        backend = await started(BoatSandboxBackend())
        assert backend.ref == WorkspaceRef(provider='boat', id=only(fake_boat).id)
        keys = [call.arguments['idempotency_key'] for call in fake_boat.calls if call.route == 'create']
        assert len(keys) == 2 and len(set(keys)) == 1

    async def test_a_creation_still_in_progress_elsewhere_is_waited_for(self, fake_boat: FakeBoat) -> None:
        message = 'A request with this Idempotency-Key is still in progress.'
        fake_boat.errors['create'] = [boat_error(409, 'idempotency_in_progress', message)]
        await started(BoatSandboxBackend())
        assert fake_boat.routes().count('create') == 2
        assert len(fake_boat.sandboxes) == 1

    async def test_a_retried_creation_finds_the_sandbox_a_lost_reply_left(self, fake_boat: FakeBoat) -> None:
        # The sandbox is created but its reply is lost, and the repeat that would recover it is refused.
        refused = boat_error(503, 'no_ready_machine', 'No machine is ready.')
        fake_boat.errors['create'] = [None, refused]
        fake_boat.lost_replies['create'] = 1
        backend = BoatSandboxBackend()
        with pytest.raises(ServiceException) as exc_info:
            await backend.working_dir()
        assert exc_info.value is refused
        assert backend.ref is None
        # The next operation repeats the creation under the same key instead of starting a second sandbox.
        await backend.working_dir()
        assert backend.ref == WorkspaceRef(provider='boat', id=only(fake_boat).id)

    @pytest.mark.parametrize(
        ('answer', 'message'),
        [
            pytest.param((401, 'unauthorized', 'unauthorized'), f'Credentials rejected. {_AUTH}', id='unauthorized'),
            pytest.param((403, 'forbidden', 'forbidden'), f'Credentials rejected. {_AUTH}', id='forbidden'),
            pytest.param(
                (403, 'trial_machine_class_not_allowed', 'Trial accounts start small machines only.'),
                'Could not start Boat sandbox: Trial accounts start small machines only.',
                id='plan',
            ),
            pytest.param(
                (402, 'billing_required', 'Add a payment method to start sandboxes.'),
                'Could not start Boat sandbox: Add a payment method to start sandboxes.',
                id='billing',
            ),
            # A 429 that no timed retry clears: the trial ends only by paying.
            pytest.param(
                (429, 'trial_compute_limit_reached', 'The trial used its 25 hours.'),
                'Could not start Boat sandbox: The trial used its 25 hours.',
                id='trial-hours',
            ),
            pytest.param(
                (409, 'idempotency_key_reused', 'This Idempotency-Key was already used with a different request body.'),
                'Could not start Boat sandbox: Boat answered HTTP 409 (idempotency_key_reused): '
                'This Idempotency-Key was already used with a different request body.',
                id='key-reused',
            ),
        ],
    )
    async def test_a_refused_creation_is_unavailable(
        self, fake_boat: FakeBoat, answer: tuple[int, str, str], message: str
    ) -> None:
        fake_boat.errors['create'] = [boat_error(*answer)]
        with pytest.raises(WorkspaceUnavailableError) as exc_info:
            await BoatSandboxBackend().run(['true'])
        assert str(exc_info.value) == message
        assert isinstance(exc_info.value.__cause__, ApiException)
        assert not fake_boat.sandboxes

    @pytest.mark.parametrize(
        ('build', 'message'),
        [
            pytest.param(
                lambda: BoatSandboxBackend(snapshot='nope'),
                'Could not start Boat sandbox: Boat answered HTTP 404 (snapshot_not_found): '
                'No named snapshot "nope". Run boat snapshots to list yours.',
                id='unknown-snapshot',
            ),
            # A size outside `MachineType`, as an untyped caller can pass: the SDK refuses it before sending it.
            pytest.param(
                lambda: BoatSandboxBackend(machine_type='huge'),  # pyright: ignore[reportArgumentType]
                'Could not start Boat sandbox: 1 validation error for CreateSandboxRequest\ntype\n'
                "  Value error, must be one of enum values ('small', 'default', 'large', 'xlarge')",
                id='unknown-machine-type',
            ),
        ],
    )
    async def test_settings_boat_refuses_are_unavailable(
        self, fake_boat: FakeBoat, build: Callable[[], BoatSandboxBackend], message: str
    ) -> None:
        # No retry can fix them, so the run ends rather than being retried by a durable engine.
        with pytest.raises(WorkspaceUnavailableError) as exc_info:
            await build().run(['true'])
        assert str(exc_info.value).startswith(message)
        assert not fake_boat.sandboxes

    @pytest.mark.parametrize(
        'answer',
        [
            pytest.param(lambda: boat_error(429, 'rate_limited', 'Too many starts.'), id='429'),
            pytest.param(lambda: boat_error(503, 'out_of_capacity', 'No capacity.'), id='503'),
            # A proxy's answer, which is not Boat's envelope.
            pytest.param(
                lambda: ServiceException(status=502, reason='Bad Gateway', body='<html>Bad Gateway</html>'),
                id='not-json',
            ),
        ],
    )
    async def test_a_transient_creation_failure_propagates_unchanged(
        self, fake_boat: FakeBoat, answer: Callable[[], ApiException]
    ) -> None:
        # So a durable engine retries it.
        fake_boat.errors['create'] = [error := answer()]
        with pytest.raises(ApiException) as exc_info:
            await BoatSandboxBackend().run(['true'])
        assert exc_info.value is error

    async def test_a_transient_failure_survives_pickling(self, fake_boat: FakeBoat) -> None:
        # Durable engines and process pools pickle the errors an operation raises.
        fake_boat.errors['create'] = [boat_error(503, 'out_of_capacity', 'No capacity.', retryable=True)]
        with pytest.raises(ServiceException) as exc_info:
            await BoatSandboxBackend().run(['true'])
        error = exc_info.value
        restored = pickle.loads(pickle.dumps(error))
        assert type(restored) is ServiceException
        assert vars(restored) == vars(error)
        assert str(restored) == str(error)

    async def test_rejected_credentials_do_not_expose_the_key(
        self, fake_boat: FakeBoat, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_boat.errors['create'] = [boat_error(401, 'unauthorized', 'expired token secret-value')]
        with pytest.raises(WorkspaceUnavailableError) as exc_info:
            await BoatSandboxBackend().working_dir()
        assert str(exc_info.value) == f'Credential expired. {_AUTH}'
        # A key Boat no longer accepts: the client the backend built carries it, the error does not.
        monkeypatch.setenv('BOAT_API_KEY', 'rotated-key-secret')
        with pytest.raises(WorkspaceUnavailableError) as exc_info:
            await BoatSandboxBackend().working_dir()
        assert str(exc_info.value) == f'Credentials rejected. {_AUTH}'
        assert 'rotated-key-secret' not in str(exc_info.value)

    @pytest.mark.parametrize('key', [None, '  \n'])
    async def test_missing_api_key_is_unavailable(
        self, fake_boat: FakeBoat, monkeypatch: pytest.MonkeyPatch, key: str | None
    ) -> None:
        if key is None:
            monkeypatch.delenv('BOAT_API_KEY')
        else:
            monkeypatch.setenv('BOAT_API_KEY', key)
        with pytest.raises(WorkspaceUnavailableError) as exc_info:
            await BoatSandboxBackend().run(['true'])
        assert str(exc_info.value) == (
            'No Boat credentials found. Set BOAT_API_KEY, or pass a configured `boat_sdk.BoatApi` as `client=`.'
        )
        assert fake_boat.clients == []

    async def test_a_create_that_never_answers_is_transient(
        self, fake_boat: FakeBoat, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(f'{_BACKEND}._CREATE_TIMEOUT', 0.0)
        fake_boat.errors['create'] = [lost_reply()]
        backend = BoatSandboxBackend()
        with pytest.raises(TimeoutError) as exc_info:
            await backend.run(['true'])
        # The builtin, not a command timeout: a durable engine retries it.
        assert type(exc_info.value) is TimeoutError
        assert str(exc_info.value) == (
            'Boat sandbox creation did not complete within 0s; the Boat control plane may be unreachable.'
        )
        assert backend.ref is None


class TestClients:
    async def test_the_client_built_from_boat_api_key_is_shared_by_every_backend_with_that_key(
        self, fake_boat: FakeBoat, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv('BOAT_API_KEY', f' {fake_boat.api_key}\n')
        first, second = BoatSandboxBackend(), BoatSandboxBackend()
        assert (await first.run(['echo', 'hi'])).stdout == 'hi\n'
        assert (await second.run(['echo', 'hi'])).stdout == 'hi\n'
        # One SDK client, for Boat's API and the key without its whitespace; nothing to close.
        [client] = fake_boat.clients
        assert client.api_key == fake_boat.api_key
        assert (
            client.sdk_client is not None and vars(client.sdk_client)['configuration'].host == 'https://boat.dev/api/v1'
        )
        assert len(fake_boat.sandboxes) == 2

    async def test_a_supplied_client_is_used(self, fake_boat: FakeBoat, monkeypatch: pytest.MonkeyPatch) -> None:
        client = sdk_client(fake_boat.api_key)
        monkeypatch.delenv('BOAT_API_KEY')
        backend = BoatSandboxBackend(client=client)
        assert (await backend.run(['echo', 'mine'])).stdout == 'mine\n'
        assert fake_boat.clients == [client]


class TestCancellationDuringCreation:
    async def test_native_cancellation_during_committed_creation_keeps_ref(self, fake_boat: FakeBoat) -> None:
        gate = fake_boat.gates['create'] = threading.Event()
        backend = BoatSandboxBackend()
        task = asyncio.create_task(backend.working_dir())
        await until(fake_boat.reached['create'].is_set)
        task.cancel()
        task.cancel()
        gate.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        # The creation finishes in its thread and records the sandbox; the next operation uses it.
        await until(lambda: backend.ref is not None)
        assert backend.ref == WorkspaceRef(provider='boat', id=only(fake_boat).id)
        await backend.working_dir()
        assert len(fake_boat.create_bodies) == 1

    async def test_a_creation_failure_after_cancellation_is_not_reported_to_the_loop(self, fake_boat: FakeBoat) -> None:
        # Creation keeps going after the caller is cancelled, then Boat refuses it.
        gate = fake_boat.gates['create'] = threading.Event()
        fake_boat.errors['create'] = [boat_error(400, 'invalid_type', 'type must be one of small, default')]
        backend = BoatSandboxBackend()
        # What reaches the loop's exception handler, such as a failure reported as never retrieved.
        loop = asyncio.get_running_loop()
        errors: list[dict[str, Any]] = []
        previous = loop.get_exception_handler()
        loop.set_exception_handler(lambda _, context: errors.append(context))
        try:
            task = asyncio.create_task(backend.working_dir())
            await until(fake_boat.reached['create'].is_set)
            task.cancel()
            await asyncio.sleep(0)
            gate.set()
            # The caller's cancellation wins; the refusal is dropped, not reported.
            with pytest.raises(asyncio.CancelledError):
                await task
            await until(lambda: 'get' in fake_boat.routes() or not fake_boat.errors['create'])
            await asyncio.sleep(0.05)
        finally:
            loop.set_exception_handler(previous)
        assert errors == []

    async def test_cancellation_during_creation_keeps_the_created_sandbox(
        self, fake_boat: FakeBoat, tmp_path: Path
    ) -> None:
        directory = tmp_path.resolve() / 'work'
        gate = fake_boat.gates['create'] = threading.Event()
        backend = BoatSandboxBackend(working_dir=str(directory))
        with anyio.CancelScope() as scope:

            async def cancel_once_created() -> None:
                await until(fake_boat.reached['create'].is_set)
                scope.cancel()
                gate.set()

            async with anyio.create_task_group() as tasks:
                tasks.start_soon(cancel_once_created)
                await backend.run(['true'])
        assert scope.cancelled_caught
        # Acquiring the sandbox finished before the cancellation took effect, `working_dir` included;
        # the command never started.
        assert backend.ref == WorkspaceRef(provider='boat', id=only(fake_boat).id)
        assert directory.is_dir()
        assert fake_boat.bodies('start') == []
        assert await backend.working_dir() == str(directory)
        assert len(fake_boat.create_bodies) == 1

    async def test_the_command_timeout_starts_once_the_sandbox_is_acquired(self, fake_boat: FakeBoat) -> None:
        gate = fake_boat.gates['create'] = threading.Event()
        # A command Boat finishes at once, so only acquiring the sandbox outlasts the timeout.
        fake_boat.errors['start'] = [detached(4242)]
        fake_boat.errors['follow'] = [Frames([{'type': 'stdout', 'data': 'ready\n'}, {'type': 'exit', 'exitCode': 0}])]

        async def release() -> None:
            await anyio.sleep(0.1)
            gate.set()

        async with anyio.create_task_group() as tasks:
            tasks.start_soon(release)
            result = await BoatSandboxBackend().run(['echo', 'ready'], timeout=0.05)
            assert result == CommandResult(exit_code=0, stdout='ready\n', stderr='')


class TestAttach:
    async def test_attaching_resumes_a_stopped_sandbox(self, fake_boat: FakeBoat) -> None:
        existing = fake_boat.sandbox('archived')
        backend = BoatSandboxBackend(ref=WorkspaceRef(provider='boat', id=existing.id))
        assert (await backend.run(['echo', 'resumed'])).stdout == 'resumed\n'
        assert existing.resume_calls == 1
        assert fake_boat.create_bodies == []
        # Checked again until ready, and not extended: a resume sets the auto-stop deadline itself.
        assert fake_boat.routes()[:4] == ['get', 'resume', 'get', 'start']

    async def test_operations_wait_until_a_starting_sandbox_is_ready(self, fake_boat: FakeBoat) -> None:
        # File calls sent to a starting sandbox can report an existing file as missing, so none is sent early.
        existing = fake_boat.sandbox('provisioning')
        existing.next_states = ['cloning', 'cloning']
        backend = BoatSandboxBackend(ref=WorkspaceRef(provider='boat', id=existing.id))
        assert await backend.exists(f'{existing.home}/missing') is False
        # The first operation on an attached sandbox extends its auto-stop.
        assert fake_boat.routes()[:5] == ['get', 'get', 'get', 'update', 'run']

    @pytest.mark.parametrize('operation', ['run', 'read_bytes', 'exists', 'working_dir'])
    async def test_a_missing_sandbox_remains_terminal(
        self, fake_boat: FakeBoat, tmp_path: Path, operation: str
    ) -> None:
        backend = BoatSandboxBackend(ref=WorkspaceRef(provider='boat', id='bx_missing'))
        calls: dict[str, Callable[[], Awaitable[object]]] = {
            'run': lambda: backend.run(['true']),
            'read_bytes': lambda: backend.read_bytes(f'{tmp_path}/note'),
            'exists': lambda: backend.exists(f'{tmp_path}/note'),
            'working_dir': backend.working_dir,
        }
        for _ in range(2):
            with pytest.raises(WorkspaceUnavailableError) as exc_info:
                await calls[operation]()
            assert str(exc_info.value) == (
                "The Boat sandbox 'bx_missing' was not found (it was deleted, or never existed in this Boat "
                f'account). {_FRESH}'
            )
        # A dead reference is reported, not replaced with a fresh sandbox.
        assert backend.ref == WorkspaceRef(provider='boat', id='bx_missing')
        assert fake_boat.create_bodies == []

    @pytest.mark.parametrize('state', ['error', 'cancelled'])
    async def test_a_failed_sandbox_is_unavailable(self, fake_boat: FakeBoat, state: str) -> None:
        existing = fake_boat.sandbox(state)
        with pytest.raises(WorkspaceUnavailableError) as exc_info:
            await BoatSandboxBackend(ref=WorkspaceRef(provider='boat', id=existing.id)).run(['true'])
        assert str(exc_info.value) == (
            f'The Boat sandbox {existing.id!r} is in the {state!r} state and cannot be used. {_FRESH}'
        )

    async def test_a_sandbox_that_fails_while_starting_is_unavailable(self, fake_boat: FakeBoat) -> None:
        existing = fake_boat.sandbox('provisioning')
        existing.next_states = ['cloning', 'error']
        with pytest.raises(WorkspaceUnavailableError) as exc_info:
            await BoatSandboxBackend(ref=WorkspaceRef(provider='boat', id=existing.id)).working_dir()
        assert str(exc_info.value) == (
            f'The Boat sandbox {existing.id!r} stopped or failed while starting (Sandbox entered terminal state '
            f'error). {_FRESH}'
        )

    @pytest.mark.parametrize('code', ['sandbox_resuming', 'sandbox_stopping', 'sandbox_starting'])
    async def test_a_resume_refused_while_the_sandbox_changes_state_is_sent_again(
        self, fake_boat: FakeBoat, code: str
    ) -> None:
        existing = fake_boat.sandbox('archived')
        fake_boat.errors['resume'] = [boat_error(409, code, 'The sandbox is changing state.')]
        backend = BoatSandboxBackend(ref=WorkspaceRef(provider='boat', id=existing.id))
        assert (await backend.run(['echo', 'resumed'])).stdout == 'resumed\n'
        assert fake_boat.routes()[:5] == ['get', 'resume', 'get', 'resume', 'get']
        assert existing.resume_calls == 1

    async def test_a_resume_refused_until_the_bound_is_transient(
        self, fake_boat: FakeBoat, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(f'{_BACKEND}._CREATE_TIMEOUT', 0.0)
        existing = fake_boat.sandbox('archived')
        fake_boat.errors['resume'] = [boat_error(409, 'sandbox_stopping', 'The sandbox is changing state.')]
        with pytest.raises(TimeoutError) as exc_info:
            await BoatSandboxBackend(ref=WorkspaceRef(provider='boat', id=existing.id)).working_dir()
        assert str(exc_info.value) == f'Resuming Boat sandbox {existing.id!r} did not complete within 0s.'

    @pytest.mark.parametrize(
        ('code', 'message'),
        [
            ('snapshot_data_expired', 'The sandbox stayed stopped for 14 days.'),
            ('resume_failed', 'The snapshot of this sandbox has expired.'),
        ],
    )
    async def test_a_sandbox_boat_cannot_resume_is_unavailable(
        self, fake_boat: FakeBoat, code: str, message: str
    ) -> None:
        existing = fake_boat.sandbox('archived')
        fake_boat.errors['resume'] = [boat_error(409, code, message)]
        with pytest.raises(WorkspaceUnavailableError) as exc_info:
            await BoatSandboxBackend(ref=WorkspaceRef(provider='boat', id=existing.id)).run(['true'])
        assert str(exc_info.value) == f'Could not resume Boat sandbox {existing.id!r}: {message}'
        assert fake_boat.routes() == ['get', 'resume']

    async def test_attach_errors_are_translated_or_propagate(self, fake_boat: FakeBoat) -> None:
        existing = fake_boat.sandbox()
        ref = WorkspaceRef(provider='boat', id=existing.id)
        offline = unreachable()
        fake_boat.errors['get'] = [
            boat_error(400, 'invalid_request', 'Bad request.'),
            boat_error(401, 'unauthorized', 'unauthorized'),
            limited := boat_error(429, 'rate_limited', 'Too many requests.'),
            offline,
        ]
        with pytest.raises(WorkspaceError) as refused:
            await BoatSandboxBackend(ref=ref).working_dir()
        assert type(refused.value) is WorkspaceError
        assert str(refused.value) == (
            f'Could not attach to Boat sandbox {existing.id!r}: Boat answered HTTP 400 (invalid_request): Bad request.'
        )
        with pytest.raises(WorkspaceUnavailableError, match=re.escape(_AUTH)):
            await BoatSandboxBackend(ref=ref).working_dir()
        with pytest.raises(ApiException) as raised_limit:
            await BoatSandboxBackend(ref=ref).working_dir()
        assert raised_limit.value is limited
        with pytest.raises(urllib3.exceptions.MaxRetryError) as raised:
            await BoatSandboxBackend(ref=ref).working_dir()
        assert raised.value is offline

    async def test_an_attach_that_never_becomes_ready_is_transient(
        self, fake_boat: FakeBoat, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(f'{_BACKEND}._CREATE_TIMEOUT', 0.0)
        existing = fake_boat.sandbox('provisioning')
        existing.next_states = ['cloning', 'cloning']
        with pytest.raises(TimeoutError) as exc_info:
            await BoatSandboxBackend(ref=WorkspaceRef(provider='boat', id=existing.id)).working_dir()
        assert type(exc_info.value) is TimeoutError


class TestAutoStop:
    """Boat stops a sandbox once its auto-stop runs out from creation or resume, whatever runs in it."""

    # At most once a minute, or once per half of a shorter auto-stop.
    @pytest.mark.parametrize(('auto_stop', 'interval'), [(900, 60), (100, 50)])
    async def test_an_operation_extends_the_auto_stop_at_most_once_an_interval(
        self, fake_boat: FakeBoat, monkeypatch: pytest.MonkeyPatch, auto_stop: int, interval: int
    ) -> None:
        advance = advance_clock(monkeypatch)
        backend = await started(BoatSandboxBackend(auto_stop=auto_stop))
        path = f'{only(fake_boat).home}/note'
        advance(interval - 1)
        assert await backend.exists(path) is False
        assert fake_boat.bodies('update') == []
        advance(1)
        assert await backend.exists(path) is False
        # A full auto-stop from now, which the next operations do not repeat until the interval has passed again.
        assert fake_boat.bodies('update') == [{'ttlSeconds': auto_stop}]
        advance(interval - 1)
        assert await backend.exists(path) is False
        assert fake_boat.bodies('update') == [{'ttlSeconds': auto_stop}]
        advance(1)
        assert await backend.exists(path) is False
        assert fake_boat.bodies('update') == [{'ttlSeconds': auto_stop}, {'ttlSeconds': auto_stop}]

    @pytest.mark.parametrize(('state', 'updates'), [('idle', [{'ttlSeconds': 3600}]), ('archived', [])])
    async def test_attaching_extends_a_running_sandbox_and_a_resume_sets_it_again(
        self, fake_boat: FakeBoat, state: str, updates: list[dict[str, object]]
    ) -> None:
        # When a running sandbox's deadline was set is unknown, so its first operation sets it.
        existing = fake_boat.sandbox(state)
        backend = BoatSandboxBackend(ref=WorkspaceRef(provider='boat', id=existing.id))
        for _ in range(2):
            assert await backend.exists(f'{existing.home}/note') is False
        assert fake_boat.bodies('update') == updates

    async def test_a_sandbox_without_an_auto_stop_is_not_extended(
        self, fake_boat: FakeBoat, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        advance = advance_clock(monkeypatch)
        monkeypatch.setattr(f'{_BACKEND}._KEEP_ALIVE_INTERVAL', 0.01)
        backend = await started(BoatSandboxBackend(auto_stop=0))
        advance(30 * 24 * 3600)
        assert await backend.exists(f'{only(fake_boat).home}/note') is False
        assert (await backend.run(['sleep', '0.1'])).exit_code == 0
        assert 'update' not in fake_boat.routes()

    async def test_a_running_command_extends_the_auto_stop(
        self, fake_boat: FakeBoat, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Boat does not count a running command as activity; the same rule at a small interval.
        backend = await started(BoatSandboxBackend(auto_stop=900))
        monkeypatch.setattr(f'{_BACKEND}._KEEP_ALIVE_INTERVAL', 0.02)
        assert (await backend.run(['sleep', '0.3'])).exit_code == 0
        routes = fake_boat.routes()
        assert routes[routes.index('follow') :].count('update') >= 2

    async def test_a_command_outlives_an_extension_that_finds_the_sandbox_gone(
        self, fake_boat: FakeBoat, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The command's own stream reports a deleted or stopped sandbox; extending stops at the first sign.
        advance = advance_clock(monkeypatch)
        backend = await started(BoatSandboxBackend(auto_stop=900))
        monkeypatch.setattr(f'{_BACKEND}._KEEP_ALIVE_INTERVAL', 0.02)
        advance(3600)
        fake_boat.errors['update'] = [None, boat_error(404, 'not_found', 'not_found')]
        assert (await backend.run(['sleep', '0.3'])).exit_code == 0
        assert fake_boat.routes().count('update') == 2

    async def test_a_failed_extension_does_not_fail_the_operation(
        self, fake_boat: FakeBoat, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        advance = advance_clock(monkeypatch)
        existing = fake_boat.sandbox()
        offline = urllib3.exceptions.ProtocolError('offline')
        fake_boat.errors['update'] = [offline, offline]
        backend = BoatSandboxBackend(ref=WorkspaceRef(provider='boat', id=existing.id))
        path = f'{existing.home}/note'
        with caplog.at_level(logging.DEBUG, logger=_BACKEND):
            # Tried again once the interval has passed, not on the next operation; only the first
            # failure in a row is a warning.
            for elapsed in (0, 0, 60, 60):
                advance(elapsed)
                assert await backend.exists(path) is False
        assert fake_boat.routes().count('update') == 3
        assert [(record.levelno, record.getMessage()) for record in caplog.records] == [
            (
                logging.WARNING,
                f"Could not extend the auto-stop of Boat sandbox {existing.id}: ProtocolError('offline')",
            ),
            (logging.DEBUG, f"Could not extend the auto-stop of Boat sandbox {existing.id}: ProtocolError('offline')"),
        ]

    async def test_an_extension_boat_refuses_for_the_key_is_not_asked_again(
        self, fake_boat: FakeBoat, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        advance = advance_clock(monkeypatch)
        existing = fake_boat.sandbox()
        fake_boat.errors['update'] = [boat_error(403, 'api_key_action_forbidden', 'The action is not on the key.')]
        backend = BoatSandboxBackend(ref=WorkspaceRef(provider='boat', id=existing.id))
        with caplog.at_level(logging.DEBUG, logger=_BACKEND):
            for _ in range(3):
                assert await backend.exists(f'{existing.home}/note') is False
                advance(3600)
        assert fake_boat.routes().count('update') == 1
        assert [(record.levelno, record.getMessage()) for record in caplog.records] == [
            (
                logging.WARNING,
                f'Could not extend the auto-stop of Boat sandbox {existing.id}: Boat answered HTTP 403 '
                '(api_key_action_forbidden): The action is not on the key.',
            ),
        ]

    @pytest.mark.parametrize('change', ['deleted', 'stopped'])
    async def test_an_extension_that_finds_the_sandbox_gone_is_unavailable(
        self, fake_boat: FakeBoat, monkeypatch: pytest.MonkeyPatch, change: str
    ) -> None:
        advance = advance_clock(monkeypatch)
        backend = await started(BoatSandboxBackend())
        assert backend.ref is not None
        sandbox_id = backend.ref.id
        if change == 'deleted':
            await BoatSandbox().destroy(backend.ref)
            expected = unavailable(sandbox_id)
        else:
            fake_boat.errors['update'] = [boat_error(409, 'machine_not_running', 'Sandbox machine is not running.')]
            expected = stopped(sandbox_id)
        advance(1800)
        with pytest.raises(WorkspaceUnavailableError) as exc_info:
            await backend.run(['true'])
        assert str(exc_info.value) == expected
        assert fake_boat.routes()[-1] == 'update'


class TestCommands:
    @pytest.mark.parametrize(
        ('case', 'error'),
        [
            ('string-without-shell', TypeError),
            ('argv-with-shell', TypeError),
            ('empty-argv', TypeError),
            ('zero-timeout', ValueError),
            ('blank-env-name', ValueError),
        ],
    )
    async def test_invalid_arguments_do_not_create_a_sandbox(
        self, fake_boat: FakeBoat, case: str, error: type[Exception]
    ) -> None:
        backend = BoatSandboxBackend()
        calls: dict[str, Callable[[], Awaitable[object]]] = {
            'string-without-shell': lambda: backend.run('echo hi'),
            'argv-with-shell': lambda: backend.run(['true'], shell=True),
            'empty-argv': lambda: backend.run([]),
            'zero-timeout': lambda: backend.run(['true'], timeout=0),
            'blank-env-name': lambda: backend.run(['true'], env={'': 'x'}),
        }
        with pytest.raises(error):
            await calls[case]()
        assert fake_boat.calls == []

    async def test_commands_start_detached_with_their_env_inside_the_wrapper(self, fake_boat: FakeBoat) -> None:
        # `env` runs inside the wrapper's `sh`; outside it, a dash `sh` would drop `A.B` before the command ran.
        backend = await started(BoatSandboxBackend(env={'A': '1', 'B': '2'}))
        result = await backend.run(['sh', '-c', 'printf "%s %s" "$A" "$B"'], env={'B': '3', 'A.B': 'x'})
        assert result == CommandResult(exit_code=0, stdout='1 3', stderr='')
        result = await backend.run('printf hello | tr a-z A-Z', shell=True)
        assert result.stdout == 'HELLO'
        # Boat ignores the SDK's default `timeoutSeconds` on a detached command.
        assert fake_boat.bodies('start') == [
            {
                'command': f'{_WRAPPER} env -- A=1 B=3 A.B=x sh -c \'printf "%s %s" "$A" "$B"\'',
                'timeoutSeconds': 30,
                'detached': True,
                'stream': False,
            },
            {
                'command': f"{_WRAPPER} env -- A=1 B=2 /bin/sh -c 'printf hello | tr a-z A-Z'",
                'timeoutSeconds': 30,
                'detached': True,
                'stream': False,
            },
        ]

    async def test_a_command_killed_by_a_signal_reports_its_status(self, fake_boat: FakeBoat) -> None:
        # The wrapper's `sh` stays the command's parent and exits with the status the shell reports.
        backend = await started(BoatSandboxBackend())
        assert (await backend.run(['sh', '-c', 'kill -KILL $$'])).exit_code == 137

    async def test_env_names_that_are_not_shell_identifiers_reach_the_command(self, fake_boat: FakeBoat) -> None:
        backend = BoatSandboxBackend(env={'A.B': 'sandbox'})
        result = await backend.run(['env'], env={'C-D': 'command'})
        assert {'A.B=sandbox', 'C-D=command'} <= set(result.stdout.splitlines())

    async def test_commands_start_in_the_working_dir_and_a_missing_one_is_a_workspace_error(
        self, fake_boat: FakeBoat
    ) -> None:
        owner = await started(BoatSandboxBackend())
        sandbox = only(fake_boat)
        (sandbox.home / 'sub').mkdir()
        attached = BoatSandboxBackend(ref=owner.ref, working_dir=f'{sandbox.home}/sub')
        assert (await attached.run(['pwd', '-P'])).stdout == f'{sandbox.home}/sub\n'

        # An attached sandbox must already have the configured `working_dir`.
        missing = f'{sandbox.home}/missing'
        attached = BoatSandboxBackend(ref=owner.ref, working_dir=missing)
        message = (
            f'working_dir {missing!r} does not exist in Boat sandbox {sandbox.id}. '
            'Create it there, or pass a working_dir that exists.'
        )
        with pytest.raises(WorkspaceError) as exc_info:
            await attached.run(['true'])
        assert str(exc_info.value) == message
        with pytest.raises(WorkspaceError) as exc_info:
            await attached.working_dir()
        assert str(exc_info.value) == message
        # The command had exited, so nothing was stopped; and a command's own exit 125 is its result.
        assert 'stop' not in fake_boat.routes()
        assert await owner.run(['sh', '-c', 'echo out; exit 125']) == CommandResult(
            exit_code=125, stdout='out\n', stderr=''
        )

    async def test_output_over_the_limit_stops_the_command_and_raises_with_a_preview(
        self, fake_boat: FakeBoat, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The same rule as the 8 MiB limit, at a small size.
        monkeypatch.setattr(f'{_BACKEND}._MAX_OUTPUT_BYTES', 8192)
        monkeypatch.setattr(f'{_BACKEND}._OUTPUT_PREVIEW_BYTES', 64)
        backend = await started(BoatSandboxBackend())
        fake_boat.errors['follow'] = [
            Frames(
                [
                    {'type': 'stderr', 'data': 'x' + 'é' * 40},
                    {'type': 'stdout', 'data': 'b'},
                    {'type': 'stdout', 'data': 'a' * 10000},
                    {'type': 'stdout', 'data': 'never read'},
                ]
            )
        ]
        with pytest.raises(WorkspaceOutputLimitError) as exc_info:
            await backend.run(['sleep', '30'])
        error = exc_info.value
        # The preview is the beginning of each stream. It is cut at 64 bytes, not characters, and the
        # two-byte character split at the cut is dropped.
        assert (error.limit, error.stdout, error.stderr) == (8192, 'b' + 'a' * 63, 'x' + 'é' * 31)
        # Reading stops at the limit, and the command is stopped rather than left writing.
        assert only_command(fake_boat).stopped

    async def test_the_timeout_carries_partial_output_and_stops_the_command(
        self, fake_boat: FakeBoat, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The end of each stream, as at most 4 KiB of it; the same rule at a small size.
        monkeypatch.setattr(f'{_BACKEND}._PARTIAL_OUTPUT_LIMIT', 8)
        backend = await started(BoatSandboxBackend())
        command = 'echo first; echo partial out; echo partial err >&2; exec sleep 30'
        with anyio.fail_after(10), pytest.raises(WorkspaceTimeoutError) as exc_info:
            await backend.run(['sh', '-c', command], timeout=0.5)
        assert (exc_info.value.stdout, exc_info.value.stderr) == ('ial out\n', 'ial err\n')
        assert str(exc_info.value) == 'Command timed out after 0.5 seconds'
        command_ = only_command(fake_boat)
        assert command_.stopped and command_.process.poll() is not None

    @pytest.mark.parametrize('answered', [pytest.param(True, id='answered'), pytest.param(False, id='cut-off')])
    async def test_a_stop_waits_for_boats_answer_until_the_kill_timeout(
        self, fake_boat: FakeBoat, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, answered: bool
    ) -> None:
        # Boat answers a stop once the command exits, about 5.4 seconds for one that ignores SIGTERM: longer
        # than the 2 seconds a stop gets by default. The same rule at a small size.
        monkeypatch.setattr(f'{_BACKEND}._KILL_TIMEOUT', 0.5 if answered else 0.05)
        backend = await started(BoatSandboxBackend())
        release = fake_boat.gates['stop'] = threading.Event()

        async def answer_late() -> None:
            await until(fake_boat.reached['stop'].is_set)
            await anyio.sleep(0.1)
            if answered:
                release.set()

        # Well under the default grace, so a stop bounded by it instead would fail the test.
        with anyio.fail_after(1.5):
            async with anyio.create_task_group() as tasks:
                tasks.start_soon(answer_late)
                with pytest.raises(WorkspaceTimeoutError):
                    await backend.run(['sleep', '30'], timeout=0.05)
        assert only_command(fake_boat).stopped is answered
        assert ('it may still be running' in caplog.text) is not answered

    @pytest.mark.parametrize(
        ('answer', 'logged'),
        [
            pytest.param((500, 'internal_error', 'Internal error.'), True, id='failed'),
            # Boat no longer knows the command: nothing is left running.
            pytest.param(
                (400, 'sandbox_direct_failed', 'Process 1 was not started by a detached command.'), False, id='gone'
            ),
        ],
    )
    async def test_a_failed_stop_is_logged_and_the_timeout_still_raises(
        self, fake_boat: FakeBoat, caplog: pytest.LogCaptureFixture, answer: tuple[int, str, str], logged: bool
    ) -> None:
        backend = await started(BoatSandboxBackend())
        fake_boat.errors['stop'] = [boat_error(*answer)]
        with anyio.fail_after(10), pytest.raises(WorkspaceTimeoutError):
            await backend.run(['sleep', '30'], timeout=0.05)
        assert fake_boat.routes().count('stop') == 1
        assert ('it may still be running' in caplog.text) is logged

    @pytest.mark.parametrize('change', ['deleted', 'stopped'])
    async def test_a_timeout_whose_stop_finds_the_sandbox_gone_is_unavailable(
        self, fake_boat: FakeBoat, caplog: pytest.LogCaptureFixture, change: str
    ) -> None:
        # A deleted sandbox's output stream can hang until the timeout, rather than report the deletion.
        backend = await started(BoatSandboxBackend())
        sandbox = only(fake_boat)
        fake_boat.gates['follow'] = threading.Event()
        task = asyncio.create_task(backend.run(['sleep', '30'], timeout=0.3))
        with anyio.fail_after(10):
            await until(fake_boat.reached['follow'].is_set)
            if change == 'deleted':
                await BoatSandbox().destroy(WorkspaceRef(provider='boat', id=sandbox.id))
            else:
                fake_boat.stop(sandbox.id)
            with pytest.raises(WorkspaceUnavailableError) as exc_info:
                await task
        assert str(exc_info.value) == (unavailable if change == 'deleted' else stopped)(sandbox.id)
        assert isinstance(exc_info.value.__cause__, WorkspaceTimeoutError)
        assert 'it may still be running' not in caplog.text

    async def test_a_command_boat_refuses_to_start_is_a_workspace_error(self, fake_boat: FakeBoat) -> None:
        backend = await started(BoatSandboxBackend())
        with pytest.raises(WorkspaceError) as exc_info:
            await backend.run(['echo', 'x' * 140_000])
        assert type(exc_info.value) is WorkspaceError
        assert str(exc_info.value) == (
            'Could not start the command: Boat answered HTTP 400 (sandbox_direct_failed): '
            "E2BIG: argument list too long, posix_spawn 'systemd-run'"
        )
        # Nothing started, so nothing is looked up or stopped.
        assert fake_boat.routes()[-1] == 'start'

    @pytest.mark.parametrize(
        'failure',
        [
            # Boat documents that a refused start did not run.
            pytest.param(lambda: boat_error(429, 'rate_limited', 'Too many commands.'), id='rate-limited'),
            # The request never reached Boat.
            pytest.param(unreachable, id='not-sent'),
        ],
    )
    async def test_a_start_that_did_not_run_propagates_unchanged(
        self, fake_boat: FakeBoat, caplog: pytest.LogCaptureFixture, failure: Callable[[], Exception]
    ) -> None:
        # So a durable engine may retry it.
        backend = await started(BoatSandboxBackend())
        fake_boat.errors['start'] = [error := failure()]
        with pytest.raises(type(error)) as exc_info:
            await backend.run(['true'])
        assert exc_info.value is error
        # A lookup finds the sandbox running; there is nothing to stop.
        assert fake_boat.routes()[-2:] == ['start', 'get']
        assert caplog.records == []

    @pytest.mark.parametrize(
        ('failure', 'reason'),
        [
            # Boat started the command, but its answer was lost.
            pytest.param(None, r'ReadTimeoutError\(.*: Read timed out\.\"\)', id='reply-lost'),
            pytest.param(
                lambda: boat_error(502, 'sandbox_direct_failed', 'Machine did not answer.', retryable=False),
                re.escape('Boat answered HTTP 502 (sandbox_direct_failed): Machine did not answer.'),
                id='server-error',
            ),
        ],
    )
    async def test_a_command_whose_start_was_not_confirmed_is_a_workspace_error(
        self,
        fake_boat: FakeBoat,
        caplog: pytest.LogCaptureFixture,
        failure: Callable[[], Exception] | None,
        reason: str,
    ) -> None:
        backend = await started(BoatSandboxBackend())
        if failure is None:
            fake_boat.lost_replies['start'] = 1
        else:
            fake_boat.errors['start'] = [failure()]
        with pytest.raises(WorkspaceError) as exc_info:
            await backend.run(['sleep', '30'])
        # Not transient: a retry could run the command a second time.
        assert type(exc_info.value) is WorkspaceError
        assert re.fullmatch(
            rf'Boat did not confirm that the command started, so it may be running: {reason}', str(exc_info.value)
        )
        # Without its process ID there is no command to stop, which is logged.
        assert 'stop' not in fake_boat.routes()
        assert 'it may still be running' in caplog.text

    async def test_a_start_answered_without_detaching_is_a_workspace_error(self, fake_boat: FakeBoat) -> None:
        backend = await started(BoatSandboxBackend())
        fake_boat.errors['start'] = [finished(stdout='ran')]
        with pytest.raises(WorkspaceError, match=r'^Boat ran the command without detaching it\.$'):
            await backend.run(['true'])
        assert 'stop' not in fake_boat.routes()

    async def test_a_command_boat_lost_track_of_is_unknown_and_stopped(self, fake_boat: FakeBoat) -> None:
        # Boat forgets a command's exit when the sandbox's agent restarts; the command may still run.
        backend = await started(BoatSandboxBackend())
        fake_boat.errors['follow'] = [Frames([{'type': 'exit', 'status': 'lost', 'exitCode': None}])]
        with pytest.raises(WorkspaceError) as exc_info:
            await backend.run(['sleep', '30'])
        assert type(exc_info.value) is WorkspaceError
        assert str(exc_info.value) == 'Boat lost track of the command; its result is unknown.'
        assert only_command(fake_boat).stopped

    async def test_stopping_a_command_boat_forgot_is_quiet(
        self, fake_boat: FakeBoat, caplog: pytest.LogCaptureFixture
    ) -> None:
        backend = await started(BoatSandboxBackend())
        # Boat confirms the start, then knows nothing of the process: its output stream and its stop say so.
        fake_boat.errors['start'] = [detached(999_999)]
        with pytest.raises(WorkspaceError, match=r'^Boat lost track of the command; its result is unknown\.$'):
            await backend.run(['true'])
        assert fake_boat.routes()[-4:] == ['start', 'follow', 'get', 'stop']
        assert caplog.records == []

    async def test_a_command_lost_with_its_sandbox_is_unavailable(self, fake_boat: FakeBoat) -> None:
        backend = await started(BoatSandboxBackend())
        sandbox_id = only(fake_boat).id
        fake_boat.errors['follow'] = [Frames([{'type': 'error', 'message': 'agent restarted'}])]
        fake_boat.errors['get'] = [boat_error(404, 'not_found', 'not_found')]
        with pytest.raises(WorkspaceUnavailableError) as exc_info:
            await backend.run(['true'])
        assert str(exc_info.value) == unavailable(sandbox_id)

    @pytest.mark.parametrize(
        ('failure', 'message'),
        [
            pytest.param(
                lambda: urllib3.exceptions.ProtocolError('offline'),
                "Could not read the result of a command that may still be running: ProtocolError('offline')",
                id='network',
            ),
            pytest.param(
                lambda: boat_error(500, 'internal_error', 'Internal error.'),
                'Could not read the result of a command that may still be running: '
                'Boat answered HTTP 500 (internal_error): Internal error.',
                id='server',
            ),
            pytest.param(
                lambda: Frames([{'type': 'stdout', 'data': 'partial'}]),
                'Boat ended the output of the command without reporting its exit.',
                id='cut-short',
            ),
        ],
    )
    async def test_an_unreadable_result_is_a_workspace_error_and_the_command_is_stopped(
        self, fake_boat: FakeBoat, failure: Callable[[], object], message: str
    ) -> None:
        # The SDK reconnects a dropped stream for a minute before it gives up.
        backend = await started(BoatSandboxBackend())
        fake_boat.errors['follow'] = [failure()]
        with pytest.raises(WorkspaceError) as exc_info:
            await backend.run(['sleep', '30'])
        assert type(exc_info.value) is WorkspaceError
        assert str(exc_info.value) == message
        assert only_command(fake_boat).stopped

    async def test_a_command_whose_caller_is_cancelled_during_its_start_is_stopped(self, fake_boat: FakeBoat) -> None:
        # The start goes on in its thread and records the process ID, which the stop waits for.
        backend = await started(BoatSandboxBackend())
        release = fake_boat.gates['start'] = threading.Event()
        task = asyncio.create_task(backend.run(['sleep', '30']))
        with anyio.fail_after(10):
            await until(fake_boat.reached['start'].is_set)
            task.cancel()
            await anyio.sleep(0.05)
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            await until(lambda: bool(only(fake_boat).commands) and only_command(fake_boat).stopped)

    async def test_a_stop_a_repeated_cancellation_leaves_running_still_stops_the_command(
        self, fake_boat: FakeBoat
    ) -> None:
        backend = await started(BoatSandboxBackend())
        release = fake_boat.gates['stop'] = threading.Event()
        task = asyncio.create_task(backend.run(['sleep', '30']))
        with anyio.fail_after(10):
            await until(fake_boat.reached['follow'].is_set)
            task.cancel()
            await until(fake_boat.reached['stop'].is_set)
            # The second cancellation ends the run's wait for the stop, which keeps going.
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not only_command(fake_boat).stopped
            release.set()
            await until(lambda: only_command(fake_boat).stopped)


class TestErrorTranslation:
    @pytest.mark.parametrize(
        ('answer', 'expected', 'message'),
        [
            pytest.param(
                (401, 'unauthorized', 'unauthorized'),
                WorkspaceUnavailableError,
                f'Credentials rejected. {_AUTH}',
                id='unauthorized',
            ),
            pytest.param(
                (403, 'org_suspended', 'This organization is suspended.'),
                WorkspaceUnavailableError,
                'Could not access {path!r} in the sandbox: This organization is suspended.',
                id='suspended',
            ),
            pytest.param(
                (409, 'conflict', 'Busy.'),
                WorkspaceError,
                'Could not access {path!r} in the sandbox: Boat answered HTTP 409 (conflict): Busy.',
                id='other-4xx',
            ),
            # A POSIX error Boat's message does not name is not a path failure the protocol knows.
            pytest.param(
                (400, 'sandbox_direct_failed', 'EIO: i/o error, read'),
                WorkspaceError,
                'Could not access {path!r} in the sandbox: Boat answered HTTP 400 (sandbox_direct_failed): '
                'EIO: i/o error, read',
                id='unknown-errno',
            ),
            pytest.param(
                (400, 'sandbox_direct_failed', 'ENOTDIR: not a directory, open'),
                NotADirectoryError,
                "[Errno 20] Not a directory in the Boat sandbox: '{path}'",
                id='errno',
            ),
            pytest.param((429, 'rate_limited', 'Too many requests.'), ApiException, None, id='rate-limited'),
            pytest.param((503, 'no_ready_machine', 'No machine is ready.'), ServiceException, None, id='server'),
        ],
    )
    async def test_file_call_failures_are_translated(
        self, fake_boat: FakeBoat, answer: tuple[int, str, str], expected: type[Exception], message: str | None
    ) -> None:
        backend = await started(BoatSandboxBackend())
        path = f'{only(fake_boat).home}/note'
        fake_boat.errors['read'] = [boat_error(*answer)]
        with pytest.raises(expected) as exc_info:
            await backend.read_bytes(path)
        assert type(exc_info.value) is expected
        if message is None:
            # Transient: it propagates unchanged once a lookup finds the sandbox still running.
            assert isinstance(exc_info.value, ApiException) and answered(exc_info.value)[0] == answer[0]
            assert fake_boat.routes()[-2:] == ['read', 'get']
        else:
            assert str(exc_info.value) == message.format(path=path)

    @pytest.mark.parametrize(
        ('change', 'expected'),
        [
            pytest.param('deleted', unavailable, id='deleted'),
            pytest.param('archived', stopped, id='stopped'),
            pytest.param('error', failed, id='failed'),
            # The lookup finds it running, or cannot tell: the failure itself is what the caller sees.
            pytest.param('none', None, id='running'),
            pytest.param('lookup-offline', None, id='lookup-offline'),
            pytest.param('lookup-failed', None, id='lookup-failed'),
        ],
    )
    async def test_a_transient_failure_ends_the_run_when_the_sandbox_is_gone_or_stopped(
        self, fake_boat: FakeBoat, change: str, expected: Callable[[str], str] | None
    ) -> None:
        backend = await started(BoatSandboxBackend())
        sandbox = only(fake_boat)
        fake_boat.errors['read'] = [boat_error(503, 'no_ready_machine', 'No machine is ready.')]
        if change == 'deleted':
            await BoatSandbox().destroy(WorkspaceRef(provider='boat', id=sandbox.id))
        elif change == 'lookup-offline':
            fake_boat.errors['get'] = [urllib3.exceptions.ProtocolError('offline')]
        elif change == 'lookup-failed':
            fake_boat.errors['get'] = [boat_error(500, 'internal_error', 'Internal error.')]
        elif change != 'none':
            sandbox.state = change
        with pytest.raises(Exception) as exc_info:
            await backend.read_bytes(f'{sandbox.home}/note')
        if expected is None:
            assert isinstance(exc_info.value, ServiceException) and answered(exc_info.value)[0] == 503
        else:
            assert type(exc_info.value) is WorkspaceUnavailableError
            assert str(exc_info.value) == expected(sandbox.id)
            assert isinstance(exc_info.value.__cause__, ServiceException)

    @pytest.mark.parametrize(
        ('state', 'operation'),
        [
            # While Boat archives it, the machine still answers file calls but refuses commands.
            pytest.param('archiving', 'run', id='archiving'),
            pytest.param('archived', 'run', id='archived-run'),
            pytest.param('archived', 'read_bytes', id='archived-read'),
        ],
    )
    async def test_a_sandbox_stopped_under_a_held_backend_is_unavailable_until_attached_again(
        self, fake_boat: FakeBoat, state: str, operation: str
    ) -> None:
        backend = await started(BoatSandboxBackend())
        sandbox = only(fake_boat)
        path = f'{sandbox.home}/kept'
        await backend.write_bytes(path, b'kept')
        sandbox.state = state
        with pytest.raises(WorkspaceUnavailableError) as exc_info:
            await (backend.run(['true']) if operation == 'run' else backend.read_bytes(path))
        assert str(exc_info.value) == stopped(sandbox.id)
        # Ending the run is the held backend's answer; attaching again resumes it.
        assert sandbox.resume_calls == 0
        assert await BoatSandboxBackend(ref=backend.ref).read_bytes(path) == b'kept'
        assert sandbox.resume_calls == 1

    async def test_a_sandbox_stopped_while_a_command_runs_is_unavailable(self, fake_boat: FakeBoat) -> None:
        backend = await started(BoatSandboxBackend())
        sandbox = only(fake_boat)
        task = asyncio.create_task(backend.run(['sleep', '30']))
        with anyio.fail_after(10):
            await until(fake_boat.reached['follow'].is_set)
            fake_boat.stop(sandbox.id)
            with pytest.raises(WorkspaceUnavailableError) as exc_info:
                await task
        assert str(exc_info.value) == stopped(sandbox.id)

    @pytest.mark.parametrize(
        ('answer', 'expected', 'message'),
        [
            pytest.param(
                lambda: finished(exit_code=137, timed_out=True),
                WorkspaceError,
                'An internal command did not finish within 30s in Boat sandbox {id}.',
                id='timed-out',
            ),
            pytest.param(
                lambda: finished(stdout='x', truncated=True),
                WorkspaceOutputLimitError,
                'Boat command output exceeded the 8 MiB limit per stream; '
                'redirect it to a file and read part of it instead',
                id='truncated',
            ),
            pytest.param(
                lambda: finished(exit_code=None),
                WorkspaceError,
                'Boat did not report how an internal command exited in Boat sandbox {id}.',
                id='no-exit-code',
            ),
            pytest.param(
                lambda: detached(1),
                WorkspaceError,
                'Boat did not report how an internal command exited in Boat sandbox {id}.',
                id='detached',
            ),
            pytest.param(
                lambda: finished(exit_code=1, stderr='pwd: boom'),
                WorkspaceError,
                'Could not determine the working directory of Boat sandbox {id}: pwd: boom',
                id='failed',
            ),
            pytest.param(
                lambda: finished(stdout='relative\n'),
                WorkspaceError,
                'Could not determine the working directory of Boat sandbox {id}: ',
                id='relative',
            ),
        ],
    )
    async def test_internal_command_failures_are_workspace_errors(
        self,
        fake_boat: FakeBoat,
        answer: Callable[[], boat_sdk.Command200Response],
        expected: type[Exception],
        message: str,
    ) -> None:
        existing = fake_boat.sandbox()
        # Probing the working directory is the first command, and gets this answer.
        fake_boat.errors['run'] = [answer()]
        with pytest.raises(expected) as exc_info:
            await BoatSandboxBackend(ref=WorkspaceRef(provider='boat', id=existing.id)).working_dir()
        assert type(exc_info.value) is expected
        assert str(exc_info.value) == message.format(id=existing.id)


class TestDeletedSandbox:
    """A deleted sandbox must never look like a missing file."""

    @pytest.mark.parametrize(
        'operation',
        ['read_bytes', 'write_bytes', 'stat', 'list_dir', 'make_dir', 'remove', 'exists', 'realpath', 'run'],
    )
    async def test_operations_on_a_deleted_sandbox_are_unavailable(self, fake_boat: FakeBoat, operation: str) -> None:
        backend = await started(BoatSandboxBackend())
        sandbox = only(fake_boat)
        path = f'{sandbox.home}/note'
        calls: dict[str, Callable[[], Awaitable[object]]] = {
            'read_bytes': lambda: backend.read_bytes(path),
            'write_bytes': lambda: backend.write_bytes(path, b'x'),
            'stat': lambda: backend.stat(path),
            'list_dir': lambda: backend.list_dir(str(sandbox.home)),
            'make_dir': lambda: backend.make_dir(path),
            'remove': lambda: backend.remove(path),
            'exists': lambda: backend.exists(path),
            'realpath': lambda: backend.realpath(path),
            'run': lambda: backend.run(['true']),
        }
        assert backend.ref is not None
        await BoatSandbox().destroy(backend.ref)
        with pytest.raises(WorkspaceUnavailableError) as exc_info:
            await calls[operation]()
        assert str(exc_info.value) == unavailable(sandbox.id)

    async def test_the_working_dir_is_resolved_once_and_an_attach_finds_the_sandbox_gone(
        self, fake_boat: FakeBoat
    ) -> None:
        backend = BoatSandboxBackend()
        first = await backend.working_dir()
        assert await backend.working_dir() == first
        assert fake_boat.routes().count('run') == 1
        assert backend.ref is not None
        await BoatSandbox().destroy(backend.ref)
        with pytest.raises(WorkspaceUnavailableError, match='was not found'):
            await BoatSandboxBackend(ref=backend.ref).working_dir()


class TestFiles:
    @pytest.mark.parametrize('through', ['path', 'symlink'])
    async def test_paths_outside_the_file_api_roots_go_through_commands(
        self, fake_boat: FakeBoat, through: str
    ) -> None:
        backend = await started(BoatSandboxBackend())
        sandbox = only(fake_boat)
        # Beyond the sandbox's home and `/tmp`, which are all Boat's file API reaches.
        outside = sandbox.home.parent / 'outside'
        outside.mkdir()
        if through == 'symlink':
            (sandbox.home / 'link').symlink_to(outside)
            path = f'{sandbox.home}/link/nested/data.bin'
        else:
            path = f'{outside}/nested/data.bin'
        data = bytes(range(256))
        await backend.write_bytes(path, data)
        assert (outside / 'nested' / 'data.bin').read_bytes() == data
        assert await backend.read_bytes(path) == data
        # Boat's file API was asked first, and refused.
        assert (fake_boat.routes().count('write'), fake_boat.routes().count('read')) == (1, 1)

    async def test_a_large_file_goes_through_a_signed_download(
        self, fake_boat: FakeBoat, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = await started(BoatSandboxBackend())
        path = only(fake_boat).home / 'big.bin'
        path.write_bytes(data := os.urandom(5 * 1024 * 1024 + 1))
        assert await backend.read_bytes(str(path)) == data
        assert [route for route in fake_boat.routes() if route in ('read', 'download_url', 'transfer')] == [
            'read',
            'download_url',
            'transfer',
        ]

    @pytest.mark.parametrize(
        ('answer', 'expected'),
        [
            # Boat signs the URL without checking the path, so the machine answers for a file gone meanwhile.
            pytest.param(None, FileNotFoundError, id='gone'),
            pytest.param(lambda: Answer(502, b'Bad Gateway'), ApiException, id='not-json'),
        ],
    )
    async def test_a_failed_download_of_a_large_file(
        self, fake_boat: FakeBoat, answer: Callable[[], Answer] | None, expected: type[Exception]
    ) -> None:
        backend = await started(BoatSandboxBackend())
        path = f'{only(fake_boat).home}/big.bin'
        fake_boat.errors['read'] = [
            boat_error(400, 'sandbox_direct_failed', 'File is too large for read_file (6000000 bytes > 5242880).')
        ]
        if answer is not None:
            fake_boat.errors['transfer'] = [answer()]
        with pytest.raises(expected) as exc_info:
            await backend.read_bytes(path)
        if isinstance(exc_info.value, FileNotFoundError):
            assert (exc_info.value.errno, exc_info.value.filename) == (errno.ENOENT, path)
        else:
            assert isinstance(exc_info.value, ApiException)
            assert answered(exc_info.value) == (502, 'Bad Gateway')
        assert [route for route in fake_boat.routes() if route in ('download_url', 'transfer')] == [
            'download_url',
            'transfer',
        ]

    async def test_a_sandbox_stopped_during_a_download_is_unavailable(self, fake_boat: FakeBoat) -> None:
        # The signed URL points at the sandbox's machine, which stops answering once Boat stops it.
        backend = await started(BoatSandboxBackend())
        sandbox = only(fake_boat)
        fake_boat.errors['read'] = [
            boat_error(400, 'sandbox_direct_failed', 'File is too large for read_file (6000000 bytes > 5242880).')
        ]
        release = fake_boat.gates['transfer'] = threading.Event()
        read = asyncio.create_task(backend.read_bytes(f'{sandbox.home}/big.bin'))
        with anyio.fail_after(10):
            await until(fake_boat.reached['transfer'].is_set)
            sandbox.state = 'archived'
            release.set()
            with pytest.raises(WorkspaceUnavailableError) as exc_info:
                await read
        assert str(exc_info.value) == stopped(sandbox.id)
        assert isinstance(exc_info.value.__cause__, urllib3.exceptions.ProtocolError)

    @pytest.mark.parametrize(
        'target',
        [
            'inside',
            # Boat refuses to sign a URL for a path outside its file API's roots.
            'outside',
            # A write below a file fails the same way whichever way it goes.
            'below-a-file',
            # The machine refuses it as `not_a_file`, which the backend checks like an inline refusal.
            'onto-a-directory',
            'stopped',
        ],
    )
    async def test_a_large_write_goes_through_a_signed_upload(
        self, fake_boat: FakeBoat, monkeypatch: pytest.MonkeyPatch, target: str
    ) -> None:
        # The same rule as Boat's 5 MiB inline limit, at a small size.
        monkeypatch.setattr(f'{_BACKEND}._MAX_INLINE_FILE_BYTES', 16)
        backend = await started(BoatSandboxBackend())
        sandbox = only(fake_boat)
        data = bytes(range(100))
        (sandbox.home / 'file').write_bytes(b'')
        (sandbox.home / 'dir').mkdir()
        (sandbox.home.parent / 'outside').mkdir()
        paths = {
            'inside': sandbox.home / 'new' / 'big.bin',
            'outside': sandbox.home.parent / 'outside' / 'big.bin',
            'below-a-file': sandbox.home / 'file' / 'big.bin',
            'onto-a-directory': sandbox.home / 'dir',
            'stopped': sandbox.home / 'big.bin',
        }
        path = paths[target]
        if target == 'stopped':
            sandbox.state = 'archived'
        if target in ('inside', 'outside'):
            await backend.write_bytes(str(path), data)
            assert path.read_bytes() == data
            # Never inline, and the API key never reaches the machine the URL points at (the fake checks it).
            assert 'write' not in fake_boat.routes()
            assert ('transfer' in fake_boat.routes()) is (target == 'inside')
        else:
            expected = {
                'below-a-file': NotADirectoryError,
                'onto-a-directory': IsADirectoryError,
                'stopped': WorkspaceUnavailableError,
            }[target]
            with pytest.raises(expected):
                await backend.write_bytes(str(path), data)

    @pytest.mark.parametrize('target', ['fifo', 'regular-file', 'unreachable'])
    async def test_a_write_that_times_out_is_refused_on_a_fifo(self, fake_boat: FakeBoat, target: str) -> None:
        # Boat answers a write to a FIFO only once something reads it, so the SDK's request times out,
        # and so do its repeats; the path is checked then, rather than before every write.
        backend = await started(BoatSandboxBackend())
        path = only(fake_boat).home / target
        if target == 'fifo':
            os.mkfifo(path)
        else:
            path.write_bytes(b'')
            fake_boat.errors['write'] = [gave_up_reading() if target == 'regular-file' else unreachable()]
        with pytest.raises(OSError if target == 'fifo' else urllib3.exceptions.MaxRetryError) as exc_info:
            await backend.write_bytes(str(path), b'data')
        if target == 'fifo':
            assert type(exc_info.value) is OSError
            assert str(exc_info.value) == f'Not a regular file in the Boat sandbox: {str(path)!r}'
            assert isinstance(exc_info.value.__cause__, urllib3.exceptions.MaxRetryError)
        # Only a write that timed out is checked.
        assert ('run' in fake_boat.routes()[-2:]) is (target != 'unreachable')

    async def test_a_write_through_a_dangling_symlink_creates_its_target(self, fake_boat: FakeBoat) -> None:
        # Boat's file API refuses a dangling link as if it led outside its roots; the shell writes through it.
        backend = await started(BoatSandboxBackend())
        home = only(fake_boat).home
        (home / 'link').symlink_to(home / 'target.txt')
        await backend.write_bytes(f'{home}/link', b'data')
        assert (home / 'link').is_symlink()
        assert (home / 'target.txt').read_bytes() == b'data'

    async def test_a_path_boat_calls_not_a_file_that_is_a_regular_file_when_checked(self, fake_boat: FakeBoat) -> None:
        # Replaced between Boat's answer and the check: reported, rather than read as if nothing happened.
        backend = await started(BoatSandboxBackend())
        path = f'{only(fake_boat).home}/note'
        await backend.write_bytes(path, b'x')
        fake_boat.errors['read'] = [boat_error(400, 'sandbox_direct_failed', 'Path is not a file.')]
        with pytest.raises(OSError) as exc_info:
            await backend.read_bytes(path)
        assert type(exc_info.value) is OSError
        assert str(exc_info.value) == f'Not a regular file in the Boat sandbox: {path!r}'

    @pytest.mark.parametrize(
        'operation', ['read_bytes', 'write_bytes', 'stat', 'list_dir', 'make_dir', 'remove', 'exists', 'realpath']
    )
    async def test_a_newline_in_a_path_is_refused_before_acquisition(self, fake_boat: FakeBoat, operation: str) -> None:
        backend = BoatSandboxBackend()
        path = '/home/user/bad\nname'
        calls: dict[str, Callable[[], Awaitable[object]]] = {
            'read_bytes': lambda: backend.read_bytes(path),
            'write_bytes': lambda: backend.write_bytes(path, b'x'),
            'stat': lambda: backend.stat(path),
            'list_dir': lambda: backend.list_dir(path),
            'make_dir': lambda: backend.make_dir(path),
            'remove': lambda: backend.remove(path),
            'exists': lambda: backend.exists(path),
            'realpath': lambda: backend.realpath(path),
        }
        with pytest.raises(ValueError, match=r'^Boat file paths cannot contain a newline or carriage return$'):
            await calls[operation]()
        assert fake_boat.calls == []
