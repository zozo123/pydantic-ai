"""Public `BoatSandbox` capability tests, over the fake Boat SDK.

They drive the capability through `Agent` runs and its own methods. They are not VCR tests: what
they pin is which sandbox a run creates or attaches to, and with which SDK client, which no recorded
exchange shows.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path

import anyio
import boat_sdk
import pytest
from boat_sdk.exceptions import ServiceException

from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage
from pydantic_ai.workspaces import (
    ReadOnlyWorkspace,
    Workspace,
    WorkspaceReadOnlyError,
    WorkspaceRef,
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
)
from pydantic_ai_harness.boat_sandbox import BoatSandbox, BoatSandboxBackend
from pydantic_ai_harness.coder import Coder

from .._tool_calls import call_tools
from .fake_boat import FakeBoat, FakeSandbox, boat_error

pytestmark = pytest.mark.anyio(backends=['asyncio'])

_AUTH = 'Boat rejected the credentials. Set BOAT_API_KEY, or pass a configured `boat_sdk.BoatApi` as `client=`.'


def _ctx() -> RunContext[None]:
    return RunContext(deps=None, model=TestModel(), usage=RunUsage())


def only(fake: FakeBoat) -> FakeSandbox:
    [sandbox] = fake.sandboxes.values()
    return sandbox


def sdk_client(api_key: str) -> boat_sdk.BoatApi:
    """An SDK client as an application builds one to pass as `client=`; the fake answers it."""
    return boat_sdk.BoatApi(boat_sdk.ApiClient(boat_sdk.Configuration(access_token=api_key)))


async def until(condition: Callable[[], bool]) -> None:
    with anyio.fail_after(10):
        while not condition():
            await anyio.sleep(0.001)


def test_capability_repr_hides_command_environment() -> None:
    assert 'secret-sentinel' not in repr(BoatSandbox(env={'TOKEN': 'secret-sentinel'}))


async def test_capability_is_lazy_and_forwards_options(fake_boat: FakeBoat, tmp_path: Path) -> None:
    directory = f'{tmp_path.resolve()}/work'
    fake_boat.snapshots.add('python')
    capability = BoatSandbox[None](
        machine_type='large', snapshot='python', auto_stop=600, working_dir=directory, env={'A': 'b'}
    )
    backend = capability.get_workspace(_ctx(), ref=None)
    assert isinstance(backend, BoatSandboxBackend)
    assert fake_boat.calls == []
    assert await backend.working_dir() == directory
    assert fake_boat.create_bodies == [{'noEnv': True, 'type': 'large', 'from': 'python', 'ttlSeconds': 600}]
    # The env reaches commands, not the create request.
    assert (await backend.run(['sh', '-c', 'printf %s "$A"'])).stdout == 'b'


@pytest.mark.parametrize(
    ('build', 'error', 'message'),
    [
        pytest.param(
            lambda: BoatSandbox(working_dir='repo'),
            UserError,
            r"^working_dir must be an absolute POSIX path or None, got 'repo'\.$",
            id='relative-working-dir',
        ),
        pytest.param(
            lambda: BoatSandbox(auto_stop=-1),
            UserError,
            r'^auto_stop must be an integer of at least 0 or None, got -1\.$',
            id='negative-auto-stop',
        ),
        pytest.param(
            lambda: BoatSandbox(auto_stop=1.5),  # pyright: ignore[reportArgumentType]
            UserError,
            r'^auto_stop must be an integer of at least 0 or None, got 1\.5\.$',
            id='fractional-auto-stop',
        ),
        pytest.param(
            lambda: BoatSandbox(defer_loading=True, id='sandbox'),
            UserError,
            r'^`BoatSandbox` does not support `defer_loading=True`: '
            r'the workspace is selected before deferred capabilities load\.$',
            id='defer-loading',
        ),
        pytest.param(
            lambda: BoatSandbox(env={'PORT': 3000}),  # pyright: ignore[reportArgumentType]
            TypeError,
            r'^env keys and values must be strings$',
            id='non-string-env',
        ),
    ],
)
def test_invalid_settings_fail_at_construction(
    build: Callable[[], object], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        build()


async def test_backend_for_a_ref_attaches_with_the_capability_settings(fake_boat: FakeBoat) -> None:
    existing = fake_boat.sandbox()
    (existing.home / 'work').mkdir()
    capability = BoatSandbox[None](working_dir=f'{existing.home}/work', env={'A': 'b'}, machine_type='large')
    backend = capability.backend(WorkspaceRef(provider='boat', id=existing.id))
    assert (await backend.run(['sh', '-c', 'printf "%s %s" "$A" "$(pwd -P)"'])).stdout == f'b {existing.home}/work'
    assert fake_boat.create_bodies == []


def test_foreign_ref_is_declined_and_the_backend_rejects_it() -> None:
    assert BoatSandbox[None]().get_workspace(_ctx(), ref=WorkspaceRef(provider='other', id='x')) is None
    with pytest.raises(ValueError, match="expected 'boat'"):
        BoatSandbox[None]().backend(WorkspaceRef(provider='other', id='x'))


async def test_destroy_rejects_a_foreign_or_blank_ref(fake_boat: FakeBoat) -> None:
    with pytest.raises(ValueError, match=r"^unsupported workspace provider 'other'; expected 'boat'$"):
        await BoatSandbox[None]().destroy(WorkspaceRef(provider='other', id='x'))
    with pytest.raises(ValueError, match=r'^Boat workspace ref id cannot be empty$'):
        await BoatSandbox[None]().destroy(WorkspaceRef(provider='boat', id=' '))
    assert (fake_boat.clients, fake_boat.calls) == ([], [])


async def test_destroy_deletes_a_stopped_sandbox_without_resuming_it(fake_boat: FakeBoat) -> None:
    existing = fake_boat.sandbox('archived')
    capability = BoatSandbox[None]()
    await capability.destroy(WorkspaceRef(provider='boat', id=existing.id))
    assert existing.id not in fake_boat.sandboxes
    assert existing.resume_calls == 0
    # Boat deletes only with the target's ID repeated as the confirmation.
    [delete] = [call.arguments for call in fake_boat.calls if call.route == 'delete']
    assert delete['confirm'] == existing.id
    # A sandbox already deleted, or one Boat never knew, is already gone.
    await capability.destroy(WorkspaceRef(provider='boat', id=existing.id))
    await capability.destroy(WorkspaceRef(provider='boat', id='bx_unknown'))
    assert fake_boat.routes() == ['delete', 'delete', 'delete']


async def test_destroy_translates_rejected_credentials_and_propagates_transient_errors(fake_boat: FakeBoat) -> None:
    existing = fake_boat.sandbox()
    ref = WorkspaceRef(provider='boat', id=existing.id)
    refused = boat_error(503, 'out_of_capacity', 'No capacity.')
    fake_boat.errors['delete'] = [boat_error(401, 'unauthorized', 'expired token secret-value'), refused]
    with pytest.raises(WorkspaceUnavailableError) as exc_info:
        await BoatSandbox[None]().destroy(ref)
    assert str(exc_info.value) == f'Credential expired. {_AUTH}'
    with pytest.raises(ServiceException) as transient:
        await BoatSandbox[None]().destroy(ref)
    assert transient.value is refused
    assert existing.id in fake_boat.sandboxes


async def test_destroy_with_a_supplied_client_uses_it(fake_boat: FakeBoat, monkeypatch: pytest.MonkeyPatch) -> None:
    existing = fake_boat.sandbox()
    client = sdk_client(fake_boat.api_key)
    monkeypatch.delenv('BOAT_API_KEY')
    await BoatSandbox[None](client=client).destroy(WorkspaceRef(provider='boat', id=existing.id))
    assert existing.id not in fake_boat.sandboxes
    assert fake_boat.clients == [client]


async def test_agent_without_workspace_use_does_not_create(fake_boat: FakeBoat) -> None:
    result = await Agent(TestModel(custom_output_text='done'), capabilities=[BoatSandbox()]).run('hello')
    assert result.output == 'done'
    assert fake_boat.calls == []


async def test_agent_runs_without_history_create_fresh_sandboxes(fake_boat: FakeBoat) -> None:
    agent = Agent(TestModel(call_tools=['run_command']), capabilities=[BoatSandbox()])

    @agent.tool
    async def run_command(ctx: RunContext[object]) -> str:
        return (await ctx.workspace.run(['echo', 'ok'])).stdout

    await agent.run('run')
    await agent.run('run')
    assert len(fake_boat.create_bodies) == 2


async def test_agent_history_attaches_and_resumes_the_persisted_sandbox(fake_boat: FakeBoat) -> None:
    def model(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        if isinstance(messages[-1], ModelRequest) and any(isinstance(p, ToolReturnPart) for p in messages[-1].parts):
            return ModelResponse(parts=[TextPart(content='done')])
        return ModelResponse(parts=[ToolCallPart(tool_name='remember', args={}, tool_call_id='call')])

    agent = Agent(FunctionModel(model), capabilities=[BoatSandbox()])

    @agent.tool
    async def remember(ctx: RunContext[object]) -> str:
        if await ctx.workspace.exists('marker'):
            return (await ctx.workspace.read_bytes('marker')).decode()
        await ctx.workspace.write_bytes('marker', b'saved')
        return 'created'

    first = await agent.run('remember')
    sandbox = only(fake_boat)
    # Boat stops an idle sandbox after its auto-stop; the next run attaching to it resumes it.
    fake_boat.stop(sandbox.id)
    second = await agent.run('continue', message_history=first.all_messages())
    assert (first.output, second.output) == ('done', 'done')
    returned = [
        part.content for message in second.new_messages() for part in message.parts if isinstance(part, ToolReturnPart)
    ]
    assert returned == ['saved']
    assert len(fake_boat.create_bodies) == 1
    assert sandbox.resume_calls == 1
    assert second.workspace.ref == first.workspace.ref == WorkspaceRef(provider='boat', id=sandbox.id)


async def test_read_only_facade_identity_and_denied_command(fake_boat: FakeBoat) -> None:
    backend = BoatSandboxBackend()
    facade = ReadOnlyWorkspace(Workspace(backend))
    agent = Agent(TestModel(call_tools=['check']), capabilities=[BoatSandbox()])

    @agent.tool
    async def check(ctx: RunContext[object]) -> str:
        assert ctx.workspace is facade
        with pytest.raises(WorkspaceReadOnlyError, match='read-only'):
            await ctx.workspace.run(['echo', 'blocked'])
        return 'ok'

    result = await agent.run('check', workspace=facade)
    assert result.workspace is facade
    assert 'ok' in result.output
    assert fake_boat.calls == []


async def _touch(ctx: RunContext[None]) -> str:
    await ctx.workspace.write_bytes('note', b'kept')
    return 'ok'


async def test_a_cancelled_run_stops_its_command(fake_boat: FakeBoat) -> None:
    async def long_command(ctx: RunContext[None]) -> str:
        return (await ctx.workspace.run(['sleep', '30'])).stdout

    agent = Agent(
        TestModel(call_tools=['long_command']), deps_type=type(None), tools=[long_command], capabilities=[BoatSandbox()]
    )
    task = asyncio.create_task(agent.run('go'))
    await until(fake_boat.reached['follow'].is_set)
    task.cancel()
    with anyio.fail_after(10), pytest.raises(asyncio.CancelledError):
        await task
    [command] = only(fake_boat).commands.values()
    assert command.stopped and command.process.poll() is not None


async def test_the_result_workspace_keeps_working_after_the_run(fake_boat: FakeBoat) -> None:
    agent = Agent(TestModel(call_tools=['_touch']), deps_type=type(None), tools=[_touch], capabilities=[BoatSandbox()])
    result = await agent.run('go')
    # The run leaves nothing to close: the process's one SDK client keeps serving its sandbox.
    assert await result.workspace.read_bytes('note') == b'kept'
    assert len(fake_boat.clients) == 1
    assert len(fake_boat.create_bodies) == 1


async def test_a_backend_rebuilt_from_its_ref_reaches_the_same_sandbox(fake_boat: FakeBoat) -> None:
    # What a Temporal activity does: rebuild the backend from the ref, use it, and drop it.
    backend = BoatSandbox[None]().get_workspace(_ctx(), ref=None)
    assert isinstance(backend, BoatSandboxBackend)
    path = f'{await backend.working_dir()}/note'
    await backend.write_bytes(path, b'hi')
    attached = BoatSandbox[None]().get_workspace(_ctx(), ref=backend.ref)
    assert isinstance(attached, BoatSandboxBackend)
    first, second = await asyncio.gather(attached.read_bytes(path), attached.exists(path))
    assert (first, second) == (b'hi', True)
    assert len(fake_boat.clients) == 1
    assert len(fake_boat.create_bodies) == 1


@pytest.mark.parametrize('anyio_backend', ['trio'])
async def test_trio_runs_commands_files_and_timeouts(fake_boat: FakeBoat) -> None:
    """The SDK runs in worker threads and holds no event loop state, so the backend works on Trio too."""
    backend = BoatSandbox[None]().get_workspace(_ctx(), ref=None)
    assert isinstance(backend, BoatSandboxBackend)
    assert (await backend.run(['echo', 'trio'])).stdout == 'trio\n'
    path = f'{await backend.working_dir()}/note'
    await backend.write_bytes(path, b'on trio')
    assert await backend.read_bytes(path) == b'on trio'
    with anyio.fail_after(10), pytest.raises(WorkspaceTimeoutError):
        await backend.run(['sleep', '30'], timeout=0.05)
    stopped = [command for command in only(fake_boat).commands.values() if command.stopped]
    assert len(stopped) == 1
    assert backend.ref is not None
    await BoatSandbox[None]().destroy(backend.ref)
    assert fake_boat.sandboxes == {}


@pytest.mark.parametrize('supplied', ['client', 'workspace'])
async def test_a_run_uses_a_supplied_client_or_workspace(fake_boat: FakeBoat, supplied: str) -> None:
    client = sdk_client(fake_boat.api_key) if supplied == 'client' else None
    workspace = BoatSandboxBackend() if supplied == 'workspace' else None
    agent = Agent(
        TestModel(call_tools=['_touch']),
        deps_type=type(None),
        tools=[_touch],
        capabilities=[BoatSandbox[None](client=client)],
    )
    result = await agent.run('go', workspace=workspace)
    assert (only(fake_boat).home / 'note').read_bytes() == b'kept'
    assert workspace is None or result.workspace.ref == workspace.ref
    assert len(fake_boat.clients) == 1


async def test_a_child_run_works_in_the_parent_runs_sandbox(fake_boat: FakeBoat) -> None:
    child = Agent(TestModel(call_tools=['_touch']), deps_type=type(None), tools=[_touch], capabilities=[BoatSandbox()])

    async def delegate(ctx: RunContext[None]) -> str:
        await child.run('go', workspace=ctx.workspace)
        return 'ok'

    parent = Agent(
        TestModel(call_tools=['_touch', 'delegate']),
        deps_type=type(None),
        tools=[_touch, delegate],
        capabilities=[BoatSandbox()],
    )
    await parent.run('go')
    assert len(fake_boat.create_bodies) == 1
    assert fake_boat.routes().count('write') == 2


async def test_coder_works_in_the_sandbox(fake_boat: FakeBoat) -> None:
    results = await call_tools(
        [BoatSandbox(), Coder()],
        [
            ('write_file', {'path': 'greet.sh', 'content': 'echo hi from the sandbox\n'}),
            ('shell', {'command': 'sh greet.sh'}),
        ],
    )
    assert 'hi from the sandbox' in results[1]
    assert (only(fake_boat).home / 'greet.sh').read_text() == 'echo hi from the sandbox\n'
