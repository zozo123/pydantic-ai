"""`IsloSandboxBackend` tests over the fake Islo SDK.

They are not VCR tests: the backend talks to Islo through its SDK, and what they pin is how it
reacts to Islo's answers, including failures and orderings (a lost reply, a refusal after a
create, a sandbox paused or deleted mid-command) that a recording cannot produce on demand. The
conformance suite covers the protocol rules; these cover the Islo-specific paths around them.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import sys
from pathlib import Path

import anyio
import httpx
import islo
import pytest
from islo.core.api_error import ApiError
from islo.errors import BadRequestError, ForbiddenError, NotFoundError, UnauthorizedError
from islo.types import AutoResumePolicy, ErrorCode, LifecyclePolicy

from pydantic_ai.exceptions import UserError
from pydantic_ai.workspaces import (
    WorkspaceError,
    WorkspaceOutputLimitError,
    WorkspaceRef,
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
)
from pydantic_ai_harness.islo_sandbox import IsloSandbox, IsloSandboxBackend

from . import fake_islo as fake_module
from .fake_islo import FakeIslo, FakeSandbox, billing, error_body, internal_error, invalid_state, not_found

pytestmark = pytest.mark.anyio(backends=['asyncio'])

_BACKEND = 'pydantic_ai_harness.islo_sandbox._backend'
_AUTH = 'Islo rejected the credentials. Set ISLO_API_KEY, or pass a configured `islo.AsyncIslo` as `client=`.'


def only(fake: FakeIslo) -> FakeSandbox:
    [sandbox] = fake.sandboxes.values()
    return sandbox


def ref(sandbox: FakeSandbox) -> WorkspaceRef:
    return WorkspaceRef(provider='islo', id=sandbox.id)


def execs(fake: FakeIslo) -> list[list[str]]:
    return [run.command for run in fake.runs]


def test_missing_islo_extra_has_an_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, 'islo', None)
    for name in [name for name in sys.modules if name.startswith('pydantic_ai_harness.islo_sandbox')]:
        monkeypatch.delitem(sys.modules, name)
    with pytest.raises(ImportError, match=r'^Install `pydantic-ai-harness\[islo\]` to use IsloSandbox\.$'):
        importlib.import_module('pydantic_ai_harness.islo_sandbox')


class TestCreation:
    async def test_new_sandbox_gets_the_settings_and_a_named_record(
        self, fake_islo: FakeIslo, caplog: pytest.LogCaptureFixture
    ) -> None:
        backend = IsloSandboxBackend(
            image='python:3.13', vcpus=1, memory_mb=1024, gateway_profile='strict', pause_after_idle=60
        )
        assert backend.ref is None
        with caplog.at_level(logging.INFO, logger=_BACKEND):
            await backend.working_dir()
        sandbox = only(fake_islo)
        assert backend.ref == ref(sandbox)
        assert sandbox.name.startswith('pydantic-ai-') and len(sandbox.name) == len('pydantic-ai-') + 16
        assert sandbox.options == {'image': 'python:3.13', 'vcpus': 1, 'memory_mb': 1024, 'gateway_profile': 'strict'}
        assert sandbox.lifecycle == LifecyclePolicy(pause_after_idle=60, auto_resume=AutoResumePolicy.ON_ACTIVITY)
        assert f'Created Islo sandbox {sandbox.id} ({sandbox.name})' in caplog.text
        record = await backend.get_sandbox()
        assert (record.id, record.name, record.status) == (sandbox.id, sandbox.name, 'running')

    async def test_defaults_leave_islo_defaults_and_pause_after_fifteen_idle_minutes(self, fake_islo: FakeIslo) -> None:
        await IsloSandboxBackend().working_dir()
        assert only(fake_islo).options == {}
        assert only(fake_islo).lifecycle == LifecyclePolicy(
            pause_after_idle=900, auto_resume=AutoResumePolicy.ON_ACTIVITY
        )

    async def test_no_idle_pause_sends_no_lifecycle(self, fake_islo: FakeIslo) -> None:
        await IsloSandboxBackend(pause_after_idle=None).working_dir()
        assert only(fake_islo).lifecycle is None

    async def test_a_starting_sandbox_is_waited_for(self, fake_islo: FakeIslo) -> None:
        fake_islo.create_status = 'creating'
        await IsloSandboxBackend().working_dir()
        assert only(fake_islo).status == 'running'
        # The first lookup still finds it starting.
        assert fake_islo.methods().count('get_sandbox') == 2

    async def test_a_lost_create_reply_finds_the_sandbox_by_name(self, fake_islo: FakeIslo) -> None:
        fake_islo.late_errors['create_sandbox'] = [httpx.ReadTimeout('reply lost')]
        backend = IsloSandboxBackend()
        await backend.working_dir()
        assert backend.ref == ref(only(fake_islo))
        assert fake_islo.methods().count('create_sandbox') == 1

    async def test_a_create_that_never_arrived_is_sent_again(self, fake_islo: FakeIslo) -> None:
        fake_islo.errors['create_sandbox'] = [httpx.ReadTimeout('no answer')]
        await IsloSandboxBackend().working_dir()
        assert fake_islo.methods().count('create_sandbox') == 2
        assert len(fake_islo.sandboxes) == 1

    async def test_a_retried_create_that_finds_its_name_taken_uses_that_sandbox(self, fake_islo: FakeIslo) -> None:
        # The first reply and the lookup after it are both lost; the retry finds the name taken.
        fake_islo.late_errors['create_sandbox'] = [httpx.ReadTimeout('reply lost')]
        fake_islo.errors['get_sandbox'] = [httpx.ReadTimeout('lookup lost')]
        backend = IsloSandboxBackend()
        await backend.working_dir()
        assert fake_islo.methods().count('create_sandbox') == 2
        assert backend.ref == ref(only(fake_islo))

    async def test_a_request_that_never_left_the_client_is_not_retried(self, fake_islo: FakeIslo) -> None:
        fake_islo.errors['create_sandbox'] = [httpx.UnsupportedProtocol('bad URL')]
        with pytest.raises(httpx.UnsupportedProtocol):
            await IsloSandboxBackend().working_dir()
        assert fake_islo.methods() == ['create_sandbox']

    async def test_a_server_error_after_the_sandbox_was_made_uses_it(self, fake_islo: FakeIslo) -> None:
        fake_islo.late_errors['create_sandbox'] = [internal_error()]
        backend = IsloSandboxBackend()
        await backend.working_dir()
        assert backend.ref == ref(only(fake_islo))

    async def test_a_server_error_with_no_sandbox_propagates_as_transient(self, fake_islo: FakeIslo) -> None:
        fake_islo.errors['create_sandbox'] = [internal_error()]
        backend = IsloSandboxBackend()
        with pytest.raises(ApiError) as exc_info:
            await backend.working_dir()
        assert exc_info.value.status_code == 500
        assert backend.ref is None and not fake_islo.sandboxes

    async def test_a_refusal_after_the_sandbox_was_made_deletes_it(
        self, fake_islo: FakeIslo, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Islo can refuse a create for lack of credit after making the sandbox.
        fake_islo.late_errors['create_sandbox'] = [billing('create')]
        backend = IsloSandboxBackend()
        with caplog.at_level(logging.INFO, logger=_BACKEND):
            with pytest.raises(WorkspaceUnavailableError) as exc_info:
                await backend.working_dir()
        assert str(exc_info.value) == 'Could not start Islo sandbox: Insufficient credit balance to create a sandbox'
        assert backend.ref is None and not fake_islo.sandboxes
        assert 'which a refused create left behind' in caplog.text

    async def test_an_unknown_image_is_unavailable(self, fake_islo: FakeIslo, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger=_BACKEND):
            with pytest.raises(WorkspaceUnavailableError, match=r'^Could not start Islo sandbox: unknown image: nope$'):
                await IsloSandboxBackend(image='nope').working_dir()
        assert 'refused create' not in caplog.text

    async def test_a_create_that_does_not_finish_times_out(
        self, fake_islo: FakeIslo, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(f'{_BACKEND}._CREATE_TIMEOUT', 0.05)
        fake_islo.gates['create_sandbox'] = asyncio.Event()
        with pytest.raises(TimeoutError, match=r'Islo sandbox creation did not complete within 0\.05s'):
            await IsloSandboxBackend().working_dir()

    async def test_a_sandbox_that_never_starts_times_out(
        self, fake_islo: FakeIslo, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(f'{_BACKEND}._CREATE_TIMEOUT', 0.05)
        fake_islo.create_status = 'creating'
        fake_islo.gates['get_sandbox'] = asyncio.Event()
        with pytest.raises(TimeoutError, match=r"^Connecting to Islo sandbox '.+' did not complete within 0.05s\.$"):
            await IsloSandboxBackend().working_dir()

    async def test_a_missing_key_is_unavailable(self, fake_islo: FakeIslo, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv('ISLO_API_KEY')
        message = (
            r'^No Islo credentials found\. Set ISLO_API_KEY, or pass a configured `islo\.AsyncIslo` as `client=`\.$'
        )
        with pytest.raises(WorkspaceUnavailableError, match=message):
            await IsloSandboxBackend().working_dir()
        assert fake_islo.clients == []

    @pytest.mark.parametrize(
        'error',
        [
            pytest.param(
                httpx.HTTPStatusError(
                    'refused',
                    request=httpx.Request('POST', 'https://api.islo.dev/auth/token'),
                    response=httpx.Response(401),
                ),
                id='key-exchange',
            ),
            pytest.param(UnauthorizedError(body=error_body(ErrorCode.AUTH_REQUIRED, 'token expired')), id='401'),
        ],
    )
    async def test_rejected_credentials_are_unavailable(self, fake_islo: FakeIslo, error: Exception) -> None:
        fake_islo.errors['create_sandbox'] = [error]
        with pytest.raises(WorkspaceUnavailableError, match=r'\. ' + _AUTH.replace('.', r'\.').replace('`', '.') + '$'):
            await IsloSandboxBackend().working_dir()

    async def test_other_http_errors_propagate(self, fake_islo: FakeIslo) -> None:
        request = httpx.Request('GET', 'https://ca.compute.islo.dev/sandboxes')
        error = httpx.HTTPStatusError('busy', request=request, response=httpx.Response(503, request=request))
        fake_islo.errors['create_sandbox'] = [error]
        with pytest.raises(httpx.HTTPStatusError):
            await IsloSandboxBackend().working_dir()

    async def test_a_forbidden_create_is_unavailable(self, fake_islo: FakeIslo) -> None:
        fake_islo.errors['create_sandbox'] = [ForbiddenError(body=error_body(ErrorCode.ACCESS_DENIED, 'no sandboxes'))]
        with pytest.raises(WorkspaceUnavailableError, match=r'^Could not start Islo sandbox: no sandboxes$'):
            await IsloSandboxBackend().working_dir()

    async def test_working_dir_is_made_on_a_new_sandbox(self, fake_islo: FakeIslo, tmp_path: Path) -> None:
        directory = str(tmp_path.resolve() / 'project' / 'nested')
        backend = IsloSandboxBackend(working_dir=directory)
        assert await backend.working_dir() == directory
        assert (await backend.run(['pwd', '-P'])).stdout == f'{directory}\n'

    async def test_a_working_dir_that_cannot_be_made_fails(self, fake_islo: FakeIslo) -> None:
        with pytest.raises(WorkspaceError, match=r"^Could not create working_dir '/dev/null/x' in Islo sandbox "):
            await IsloSandboxBackend(working_dir='/dev/null/x').working_dir()

    async def test_concurrent_first_use_creates_one_sandbox(self, fake_islo: FakeIslo) -> None:
        backend = IsloSandboxBackend()
        async with anyio.create_task_group() as group:
            for _ in range(3):
                group.start_soon(backend.working_dir)
        assert len(fake_islo.sandboxes) == 1

    async def test_a_cancelled_creation_is_still_recorded(self, fake_islo: FakeIslo, tmp_path: Path) -> None:
        directory = str(tmp_path.resolve() / 'project')
        backend = IsloSandboxBackend(working_dir=directory)
        fake_islo.gates['create_sandbox'] = gate = asyncio.Event()
        with anyio.move_on_after(5) as scope:
            async with anyio.create_task_group() as group:
                group.start_soon(backend.working_dir)
                await fake_islo.reached['create_sandbox'].wait()
                group.cancel_scope.cancel()
                gate.set()
        assert not scope.cancelled_caught
        # The cancelled caller's sandbox was recorded, and the next operation makes the directory.
        assert backend.ref == ref(only(fake_islo))
        assert await backend.working_dir() == directory


class TestAttach:
    async def test_by_id_or_name(self, fake_islo: FakeIslo) -> None:
        existing = fake_islo.add('my-box')
        by_id = IsloSandboxBackend(ref=ref(existing))
        by_name = IsloSandboxBackend(ref=WorkspaceRef(provider='islo', id='my-box'))
        assert await by_id.working_dir() == await by_name.working_dir() == str(existing.root)
        assert by_id.ref == by_name.ref == ref(existing)
        assert 'create_sandbox' not in fake_islo.methods()

    async def test_a_sandbox_named_like_an_id(self, fake_islo: FakeIslo) -> None:
        name = '01a163b0-38b7-7d30-a982-b713fa0ad9ae'
        existing = fake_islo.add(name)
        backend = IsloSandboxBackend(ref=WorkspaceRef(provider='islo', id=name))
        await backend.working_dir()
        assert backend.ref == ref(existing)

    @pytest.mark.parametrize('key', ['gone', '01a163b0-38b7-7d30-a982-b713fa0ad9ae'])
    async def test_a_missing_sandbox_is_unavailable(self, fake_islo: FakeIslo, key: str) -> None:
        message = (
            rf"^The Islo sandbox '{key}' was not found \(it was deleted, or never existed in this Islo organization\)"
        )
        with pytest.raises(WorkspaceUnavailableError, match=message):
            await IsloSandboxBackend(ref=WorkspaceRef(provider='islo', id=key)).working_dir()
        assert 'create_sandbox' not in fake_islo.methods()

    async def test_a_failed_lookup_by_id_propagates(self, fake_islo: FakeIslo) -> None:
        existing = fake_islo.add()
        fake_islo.errors['get_sandbox_by_id'] = [internal_error()]
        with pytest.raises(ApiError):
            await IsloSandboxBackend(ref=ref(existing)).working_dir()

    async def test_a_paused_sandbox_is_resumed(self, fake_islo: FakeIslo) -> None:
        existing = fake_islo.add(status='paused')
        backend = IsloSandboxBackend(ref=ref(existing))
        assert (await backend.run(['true'])).exit_code == 0
        assert existing.resumes == 1

    async def test_a_resume_refused_while_between_states_is_tried_again(self, fake_islo: FakeIslo) -> None:
        existing = fake_islo.add(status='stopped')
        fake_islo.errors['resume_sandbox'] = [invalid_state()]
        await IsloSandboxBackend(ref=ref(existing)).working_dir()
        assert existing.resumes == 1
        assert fake_islo.methods().count('resume_sandbox') == 2

    async def test_a_resume_refused_for_lack_of_credit_is_unavailable(self, fake_islo: FakeIslo) -> None:
        existing = fake_islo.add(status='paused')
        fake_islo.errors['resume_sandbox'] = [billing('resume')]
        message = (
            rf"^Could not attach to Islo sandbox '{existing.id}': Insufficient credit balance to resume a sandbox$"
        )
        with pytest.raises(WorkspaceUnavailableError, match=message):
            await IsloSandboxBackend(ref=ref(existing)).working_dir()

    async def test_a_failed_sandbox_is_unavailable(self, fake_islo: FakeIslo) -> None:
        existing = fake_islo.add(status='failed')
        message = rf"^The Islo sandbox '{existing.id}' is in the 'failed' state and cannot be used\."
        with pytest.raises(WorkspaceUnavailableError, match=message):
            await IsloSandboxBackend(ref=ref(existing)).working_dir()

    async def test_a_sandbox_deleted_while_resuming_is_unavailable(self, fake_islo: FakeIslo) -> None:
        existing = fake_islo.add(status='paused')
        fake_islo.errors['get_sandbox'] = [not_found(existing.name)]
        with pytest.raises(WorkspaceUnavailableError, match=r'no longer exists \(it was deleted\)'):
            await IsloSandboxBackend(ref=ref(existing)).working_dir()

    async def test_a_failed_check_while_resuming_propagates(self, fake_islo: FakeIslo) -> None:
        existing = fake_islo.add(status='paused')
        fake_islo.errors['get_sandbox'] = [internal_error()]
        with pytest.raises(ApiError):
            await IsloSandboxBackend(ref=ref(existing)).working_dir()

    async def test_an_attached_working_dir_must_exist(self, fake_islo: FakeIslo) -> None:
        existing = fake_islo.add()
        backend = IsloSandboxBackend(ref=ref(existing), working_dir='/nonexistent/project')
        message = r"^working_dir '/nonexistent/project' does not exist in Islo sandbox "
        with pytest.raises(WorkspaceError, match=message):
            await backend.working_dir()
        with pytest.raises(WorkspaceError, match=message):
            await backend.run(['true'])

    @pytest.mark.parametrize(
        ('build', 'message'),
        [
            (lambda: IsloSandboxBackend(ref=WorkspaceRef(provider='other', id='x')), "expected 'islo'"),
            (lambda: IsloSandboxBackend(ref=WorkspaceRef(provider='islo', id=' ')), 'cannot be empty'),
            (lambda: IsloSandboxBackend(working_dir='relative'), 'working_dir must be an absolute workspace path'),
        ],
    )
    def test_invalid_arguments(self, build: object, message: str) -> None:
        assert callable(build)
        with pytest.raises(ValueError, match=message):
            build()


class TestRun:
    async def test_env_layers_and_names_that_are_not_identifiers_travel_in_the_command(
        self, fake_islo: FakeIslo
    ) -> None:
        backend = IsloSandboxBackend(env={'A': 'base', 'B': 'base'})
        result = await backend.run(['env'], env={'B': 'call', 'a-b': 'dash'})
        assert {'A=base', 'B=call', 'a-b=dash'} <= set(result.stdout.splitlines())
        # Islo sets the shell identifiers; the other name is in the command line.
        assert fake_islo.runs[-1].command[-4:] == ['env', '--', 'a-b=dash', 'env']
        assert fake_islo.runs[-1].env == {'A': 'base', 'B': 'call'}

    async def test_output_past_the_limit_raises_with_its_head(
        self, fake_islo: FakeIslo, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(f'{_BACKEND}._MAX_OUTPUT_BYTES', 10)
        monkeypatch.setattr(f'{_BACKEND}._OUTPUT_PREVIEW_BYTES', 4)
        with pytest.raises(WorkspaceOutputLimitError) as exc_info:
            await IsloSandboxBackend().run('printf 0123456789ab; printf err >&2', shell=True)
        assert (exc_info.value.stdout, exc_info.value.stderr, exc_info.value.limit) == ('0123', 'err', 10)

    async def test_output_islo_cut_raises(self, fake_islo: FakeIslo, monkeypatch: pytest.MonkeyPatch) -> None:
        # Output a background child adds after the wrapper measured it can pass what Islo keeps.
        backend = IsloSandboxBackend()
        await backend.working_dir()
        monkeypatch.setattr(fake_module, 'STREAM_LIMIT', 4)
        with pytest.raises(WorkspaceOutputLimitError) as exc_info:
            await backend.run(['printf', '123456'])
        assert (exc_info.value.stdout, exc_info.value.stderr) == ('', '')

    async def test_a_timeout_stops_the_command_and_keeps_its_partial_output(self, fake_islo: FakeIslo) -> None:
        backend = IsloSandboxBackend()
        with pytest.raises(WorkspaceTimeoutError, match=r'^Command timed out after 0\.5 seconds$') as exc_info:
            await backend.run('echo PARTIAL; echo E >&2; exec sleep 30', shell=True, timeout=0.5)
        assert (exc_info.value.stdout, exc_info.value.stderr) == ('PARTIAL\n', 'E\n')
        # The stop is a command of its own; the command's shell saw its process killed.
        command, stop = fake_islo.runs
        assert 'kill -KILL' in stop.command[-1]
        await command.output
        assert command.process.returncode == 137

    async def test_a_lost_start_reply_is_stopped_by_its_marker(self, fake_islo: FakeIslo) -> None:
        backend = IsloSandboxBackend()
        await backend.working_dir()
        fake_islo.late_errors['exec_in_sandbox'] = [httpx.ReadTimeout('reply lost')]
        with pytest.raises(httpx.ReadTimeout):
            await backend.run(['sleep', '30'])
        _, command, stop = fake_islo.runs
        assert 'kill -KILL' in stop.command[-1]
        # The probe found the sandbox running, so the transient error propagated.
        assert fake_islo.methods().count('get_sandbox') == 1
        await command.output
        assert command.process.returncode == 137

    async def test_a_start_that_arrives_after_its_stop_does_not_run(self, fake_islo: FakeIslo, tmp_path: Path) -> None:
        marker = tmp_path / 'ran'
        backend = IsloSandboxBackend()
        await backend.working_dir()
        fake_islo.late_start = 0.8
        with pytest.raises(httpx.ReadTimeout):
            await backend.run(['touch', str(marker)])
        [late] = fake_islo.background
        await late
        _, stop, command = fake_islo.runs
        assert 'kill -KILL' in stop.command[-1]
        await command.output
        assert command.process.returncode == 137
        assert not marker.exists()

    async def test_a_refused_start_is_not_stopped(self, fake_islo: FakeIslo) -> None:
        fake_islo.errors['exec_in_sandbox'] = [
            None,
            BadRequestError(body=error_body(ErrorCode.VALIDATION_ERROR, 'bad')),
        ]
        backend = IsloSandboxBackend()
        await backend.working_dir()
        with pytest.raises(WorkspaceError, match=r'^Could not start the command: bad$'):
            await backend.run(['true'])
        assert fake_islo.methods().count('exec_in_sandbox') == 2

    async def test_a_start_that_never_left_is_not_stopped(self, fake_islo: FakeIslo) -> None:
        fake_islo.errors['exec_in_sandbox'] = [None, httpx.ConnectError('down')]
        backend = IsloSandboxBackend()
        await backend.working_dir()
        with pytest.raises(httpx.ConnectError):
            await backend.run(['true'])
        assert fake_islo.methods().count('exec_in_sandbox') == 2

    async def test_a_failed_stop_is_logged(self, fake_islo: FakeIslo, caplog: pytest.LogCaptureFixture) -> None:
        backend = IsloSandboxBackend()
        await backend.working_dir()
        fake_islo.reached.clear()
        fake_islo.hold_results = True
        with caplog.at_level(logging.WARNING, logger=_BACKEND):
            async with anyio.create_task_group() as group:
                group.start_soon(backend.run, ['sleep', '5'])
                await fake_islo.reached['get_exec_result'].wait()
                fake_islo.errors['exec_in_sandbox'] = [internal_error()]
                group.cancel_scope.cancel()
        assert 'Could not stop a command in Islo sandbox' in caplog.text

    async def test_a_sandbox_deleted_mid_command_is_unavailable_and_not_stopped(self, fake_islo: FakeIslo) -> None:
        backend = IsloSandboxBackend()
        await backend.working_dir()
        fake_islo.reached.clear()
        fake_islo.hold_results = True

        async def delete() -> None:
            await fake_islo.reached['get_exec_result'].wait()
            fake_islo.delete(only(fake_islo).name)

        deleting = asyncio.ensure_future(delete())
        with pytest.raises(WorkspaceUnavailableError, match=r'no longer exists \(it was deleted\)'):
            await backend.run(['sleep', '30'])
        await deleting
        assert fake_islo.methods().count('exec_in_sandbox') == 2

    async def test_a_sandbox_paused_mid_command_is_unavailable(self, fake_islo: FakeIslo) -> None:
        backend = IsloSandboxBackend()
        await backend.working_dir()
        sandbox = only(fake_islo)
        sandbox.status = 'paused'
        message = rf"^The Islo sandbox '{sandbox.id}' was paused; a later run that attaches to it resumes it\."
        with pytest.raises(WorkspaceUnavailableError, match=message):
            await backend.run(['true'])

    async def test_a_sandbox_that_failed_mid_command_is_unavailable(self, fake_islo: FakeIslo) -> None:
        backend = IsloSandboxBackend()
        await backend.working_dir()
        sandbox = only(fake_islo)
        sandbox.status = 'error'
        message = rf"^The Islo sandbox '{sandbox.id}' is in the 'error' state and cannot be used\."
        with pytest.raises(WorkspaceUnavailableError, match=message):
            await backend.run(['true'])

    async def test_a_command_islo_ended_without_an_exit_code_fails(self, fake_islo: FakeIslo) -> None:
        backend = IsloSandboxBackend()
        await backend.working_dir()
        fake_islo.exec_outcome = ('failed', None)
        with pytest.raises(WorkspaceError, match=r"^Islo reported the command 'failed' without an exit code: "):
            await backend.run(['true'])

    async def test_a_transient_failure_on_a_running_sandbox_propagates(self, fake_islo: FakeIslo) -> None:
        backend = IsloSandboxBackend()
        await backend.working_dir()
        fake_islo.errors['get_exec_result'] = [internal_error()]
        with pytest.raises(ApiError):
            await backend.run(['true'])

    @pytest.mark.parametrize(
        'probe_error',
        [pytest.param(internal_error(), id='server-error'), pytest.param(httpx.ConnectError('down'), id='network')],
    )
    async def test_an_inconclusive_probe_keeps_the_original_failure(
        self, fake_islo: FakeIslo, probe_error: Exception
    ) -> None:
        backend = IsloSandboxBackend()
        await backend.working_dir()
        fake_islo.errors['exec_in_sandbox'] = [internal_error()]
        fake_islo.errors['get_sandbox'] = [probe_error]
        with pytest.raises(ApiError) as exc_info:
            await backend.run(['true'])
        assert exc_info.value.status_code == 500

    async def test_a_failure_on_a_deleted_sandbox_is_unavailable(self, fake_islo: FakeIslo) -> None:
        backend = IsloSandboxBackend()
        await backend.working_dir()
        fake_islo.delete(only(fake_islo).name)
        fake_islo.errors['exec_in_sandbox'] = [ApiError(status_code=502, body='bad gateway')]
        with pytest.raises(WorkspaceUnavailableError, match=r'no longer exists \(it was deleted\)'):
            await backend.run(['true'])

    async def test_an_answer_without_a_status_propagates(self, fake_islo: FakeIslo) -> None:
        backend = IsloSandboxBackend()
        await backend.working_dir()
        fake_islo.errors['exec_in_sandbox'] = [ApiError(status_code=None, body=None)]
        fake_islo.errors['get_sandbox'] = [internal_error()]
        with pytest.raises(ApiError):
            await backend.run(['true'])

    @pytest.mark.parametrize('anyio_backend', ['trio'])
    async def test_trio_is_refused_before_any_islo_call(self) -> None:
        message = r'^Islo needs the asyncio event loop: the Islo SDK runs its calls on asyncio\.$'
        backend = IsloSandboxBackend()
        with pytest.raises(UserError, match=message):
            await backend.run(['true'])
        with pytest.raises(UserError, match=message):
            await IsloSandbox[None]().destroy(WorkspaceRef(provider='islo', id='target'))
        await backend.aclose()


class TestInternalCommands:
    async def test_an_internal_command_islo_cut_raises(
        self, fake_islo: FakeIslo, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(fake_module, 'STREAM_LIMIT', 4)
        with pytest.raises(WorkspaceOutputLimitError):
            await IsloSandboxBackend().working_dir()

    async def test_an_internal_command_that_does_not_finish_fails(
        self, fake_islo: FakeIslo, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(f'{_BACKEND}._HELPER_TIMEOUT', 0.1)
        fake_islo.hold_results = True
        message = r'^An internal command did not finish within 0.1s in Islo sandbox .+; it may still be running\.$'
        with pytest.raises(WorkspaceError, match=message):
            await IsloSandboxBackend().working_dir()

    async def test_a_working_dir_islo_cannot_report_fails(self, fake_islo: FakeIslo) -> None:
        backend = IsloSandboxBackend()
        await backend.realpath('/')  # creates the sandbox
        fake_islo.exec_outcome = ('completed', 1)
        with pytest.raises(WorkspaceError, match=r'^Could not determine the working directory of Islo sandbox '):
            await backend.working_dir()

    async def test_working_dir_is_probed_once(self, fake_islo: FakeIslo) -> None:
        backend = IsloSandboxBackend()
        assert await backend.working_dir() == await backend.working_dir()
        assert sum('pwd' in command for command in execs(fake_islo)) == 1


class TestFiles:
    async def test_relative_paths_resolve_against_the_working_dir(self, fake_islo: FakeIslo) -> None:
        backend = IsloSandboxBackend()
        await backend.write_bytes('a/b.txt', b'\x00data')
        root = await backend.working_dir()
        assert (Path(root) / 'a' / 'b.txt').read_bytes() == b'\x00data'
        assert await backend.read_bytes('a/b.txt') == b'\x00data'
        assert (await backend.stat('a/b.txt')).size == 5
        assert [entry.name for entry in await backend.list_dir('a')] == ['b.txt']
        assert await backend.exists('a') and not await backend.exists('missing')
        await backend.make_dir('c/d')
        assert await backend.realpath('c/../a') == f'{root}/a'
        await backend.remove('a')
        assert not await backend.exists('a/b.txt')
        assert [call['path'] for method, call in fake_islo.calls if method in ('upload_file', 'download_file')] == [
            f'{root}/a/b.txt',
            f'{root}/a/b.txt',
        ]

    async def test_reading_a_directory_raises_is_a_directory(self, fake_islo: FakeIslo) -> None:
        with pytest.raises(IsADirectoryError, match=r'\[Errno 21\] Is a directory in the Islo sandbox: '):
            await IsloSandboxBackend().read_bytes('/')
        assert 'download_file' not in fake_islo.methods()

    async def test_a_file_removed_before_its_download_is_not_found(self, fake_islo: FakeIslo) -> None:
        backend = IsloSandboxBackend()
        await backend.write_bytes('f', b'x')
        fake_islo.errors['download_file'] = [
            NotFoundError(body={'code': 'FILE_NOT_FOUND', 'message': 'file not found'})
        ]
        with pytest.raises(FileNotFoundError, match=r'No such file or directory in the Islo sandbox'):
            await backend.read_bytes('f')

    @pytest.mark.parametrize(
        ('setup', 'path', 'error'),
        [
            ('mkdir dir', 'dir', IsADirectoryError),
            ('touch file', 'file/child', NotADirectoryError),
            ('touch file', 'file/deeper/child', NotADirectoryError),
            ('mkfifo fifo', 'fifo', OSError),
        ],
    )
    async def test_a_refused_write_raises_the_builtin_error(
        self, fake_islo: FakeIslo, setup: str, path: str, error: type[OSError]
    ) -> None:
        backend = IsloSandboxBackend()
        assert (await backend.run(setup, shell=True)).exit_code == 0
        with pytest.raises(error):
            await backend.write_bytes(path, b'data')

    async def test_a_write_refused_for_another_reason_propagates(self, fake_islo: FakeIslo) -> None:
        backend = IsloSandboxBackend()
        await backend.working_dir()
        fake_islo.errors['upload_file'] = [internal_error()]
        with pytest.raises(ApiError):
            await backend.write_bytes('ok', b'data')

    async def test_file_calls_on_a_paused_sandbox_are_unavailable(self, fake_islo: FakeIslo) -> None:
        backend = IsloSandboxBackend()
        await backend.write_bytes('f', b'x')
        only(fake_islo).status = 'paused'
        # The file API answers a paused sandbox with a 503; the lookup that follows finds it paused.
        with pytest.raises(WorkspaceUnavailableError, match='was paused'):
            await backend.write_bytes('f', b'y')
        assert fake_islo.methods()[-2:] == ['upload_file', 'get_sandbox']


class TestDestroy:
    async def test_by_id_or_name_without_resuming(self, fake_islo: FakeIslo) -> None:
        paused = fake_islo.add('paused-box', status='paused')
        other = fake_islo.add()
        await IsloSandbox[None]().destroy(ref(paused))
        await IsloSandbox[None]().destroy(WorkspaceRef(provider='islo', id=other.name))
        assert not fake_islo.sandboxes
        assert 'resume_sandbox' not in fake_islo.methods()
        # The capability opened a client for each, and closed it.
        assert len(fake_islo.clients) == 2

    async def test_a_supplied_client_is_used_and_left_open(self, fake_islo: FakeIslo) -> None:
        existing = fake_islo.add()
        client = islo.AsyncIslo()
        await IsloSandbox[None](client=client).destroy(ref(existing))
        assert not fake_islo.sandboxes
        assert fake_islo.clients == [client]

    async def test_a_deleted_sandbox_whose_name_was_reused_is_left_alone(self, fake_islo: FakeIslo) -> None:
        deleted = fake_islo.add('box')
        fake_islo.delete('box')
        newer = fake_islo.add('box')
        await IsloSandbox[None]().destroy(ref(deleted))
        assert fake_islo.sandboxes == {'box': newer}
        message = rf"^The Islo sandbox '{deleted.id}' no longer exists \(it was deleted\)\."
        with pytest.raises(WorkspaceUnavailableError, match=message):
            await IsloSandboxBackend(ref=ref(deleted)).working_dir()

    async def test_a_sandbox_already_gone_returns_quietly(self, fake_islo: FakeIslo) -> None:
        await IsloSandbox[None]().destroy(WorkspaceRef(provider='islo', id='gone'))
        existing = fake_islo.add()
        fake_islo.errors['delete_sandbox'] = [not_found(existing.name)]
        await IsloSandbox[None]().destroy(ref(existing))

    async def test_rejected_credentials_are_unavailable(self, fake_islo: FakeIslo) -> None:
        fake_islo.errors['get_sandbox'] = [
            UnauthorizedError(body=error_body(ErrorCode.AUTH_REQUIRED, 'Credentials expired'))
        ]
        with pytest.raises(WorkspaceUnavailableError, match=r'^Credential expired\. '):
            await IsloSandbox[None]().destroy(WorkspaceRef(provider='islo', id='box'))

    async def test_a_foreign_or_blank_ref_is_refused(self, fake_islo: FakeIslo) -> None:
        with pytest.raises(ValueError, match=r"^unsupported workspace provider 'other'; expected 'islo'$"):
            await IsloSandbox[None]().destroy(WorkspaceRef(provider='other', id='x'))
        with pytest.raises(ValueError, match=r'^Islo workspace ref id cannot be empty$'):
            await IsloSandbox[None]().destroy(WorkspaceRef(provider='islo', id=' '))
        assert fake_islo.clients == []


class TestClientLifecycle:
    async def test_aclose_closes_the_client_it_opened_and_a_later_operation_opens_another(
        self, fake_islo: FakeIslo
    ) -> None:
        backend = IsloSandboxBackend()
        await backend.write_bytes('note', b'hi')
        [first] = fake_islo.clients
        http = first.kwargs['httpx_client']
        assert isinstance(http, httpx.AsyncClient)
        await backend.aclose()
        assert http.is_closed
        assert await backend.read_bytes('note') == b'hi'
        assert len(fake_islo.clients) == 2
        await backend.aclose()

    async def test_a_supplied_client_is_left_alone(self, fake_islo: FakeIslo) -> None:
        client = islo.AsyncIslo()
        backend = IsloSandboxBackend(client=client)
        await backend.working_dir()
        await backend.aclose()
        assert fake_islo.clients == [client]

    async def test_aclose_during_an_operation_closes_when_it_ends(self, fake_islo: FakeIslo) -> None:
        backend = IsloSandboxBackend()
        await backend.working_dir()
        http = fake_islo.clients[0].kwargs['httpx_client']
        assert isinstance(http, httpx.AsyncClient)
        fake_islo.reached.clear()
        fake_islo.hold_results = True
        async with anyio.create_task_group() as group:
            group.start_soon(backend.run, ['true'])
            await fake_islo.reached['get_exec_result'].wait()
            await backend.aclose()
            assert not http.is_closed
            fake_islo.hold_results = False
        assert http.is_closed

    async def test_concurrent_acloses_close_once(self, fake_islo: FakeIslo) -> None:
        backend = IsloSandboxBackend()
        await backend.working_dir()
        async with anyio.create_task_group() as group:
            group.start_soon(backend.aclose)
            group.start_soon(backend.aclose)
        http = fake_islo.clients[0].kwargs['httpx_client']
        assert isinstance(http, httpx.AsyncClient) and http.is_closed

    async def test_a_failed_or_stuck_close_is_logged(
        self, fake_islo: FakeIslo, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        async def fail() -> None:
            raise RuntimeError('boom')

        async def hang() -> None:
            await anyio.sleep_forever()

        monkeypatch.setattr(f'{_BACKEND}._TEARDOWN_TIMEOUT', 0.05)
        with caplog.at_level(logging.WARNING, logger=_BACKEND):
            for close in (fail, hang):
                backend = IsloSandboxBackend()
                await backend.working_dir()
                monkeypatch.setattr(fake_islo.clients[-1].kwargs['httpx_client'], 'aclose', close)
                await backend.aclose()
        assert "Could not close the Islo API client: RuntimeError('boom')" in caplog.text
        assert 'Closing the Islo API client did not complete within 0.05s.' in caplog.text
