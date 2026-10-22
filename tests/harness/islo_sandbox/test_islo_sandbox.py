"""Public `IsloSandbox` capability tests, over the fake Islo SDK.

They drive the capability through `Agent` runs and its own methods. They are not VCR tests: what
they pin is which client a run opens and closes, and which sandbox a run creates or attaches to,
which no recorded exchange shows.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import anyio
import httpx
import islo
import pytest

from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage
from pydantic_ai.workspaces import ReadOnlyWorkspace, Workspace, WorkspaceReadOnlyError, WorkspaceRef
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.islo_sandbox import IsloSandbox, IsloSandboxBackend

from .._tool_calls import call_tools
from .fake_islo import FakeIslo, FakeSandbox

pytestmark = pytest.mark.anyio(backends=['asyncio'])


def _ctx() -> RunContext[None]:
    return RunContext(deps=None, model=TestModel(), usage=RunUsage())


def only(fake: FakeIslo) -> FakeSandbox:
    [sandbox] = fake.sandboxes.values()
    return sandbox


def closed(fake: FakeIslo) -> int:
    """How many of the HTTP clients the backends opened are closed."""
    clients = [client.kwargs.get('httpx_client') for client in fake.clients]
    return sum(isinstance(client, httpx.AsyncClient) and client.is_closed for client in clients)


def test_capability_repr_hides_command_environment() -> None:
    assert 'secret-sentinel' not in repr(IsloSandbox(env={'TOKEN': 'secret-sentinel'}))


async def test_capability_is_lazy_and_forwards_options(fake_islo: FakeIslo) -> None:
    directory = f'{fake_islo.root}/work'
    capability = IsloSandbox[None](
        image='python:3.13',
        vcpus=4,
        memory_mb=4096,
        gateway_profile='strict',
        pause_after_idle=120,
        working_dir=directory,
        env={'A': 'b'},
    )
    backend = capability.get_workspace(_ctx(), ref=None)
    assert isinstance(backend, IsloSandboxBackend)
    assert fake_islo.calls == []
    assert await backend.working_dir() == directory
    sandbox = only(fake_islo)
    assert sandbox.options == {'image': 'python:3.13', 'vcpus': 4, 'memory_mb': 4096, 'gateway_profile': 'strict'}
    assert sandbox.lifecycle is not None and sandbox.lifecycle.pause_after_idle == 120
    # The env reaches commands, not the create request.
    assert (await backend.run(['sh', '-c', 'printf %s "$A"'])).stdout == 'b'


@pytest.mark.parametrize(
    ('build', 'error', 'message'),
    [
        pytest.param(
            lambda: IsloSandbox(working_dir='repo'),
            UserError,
            r"^working_dir must be an absolute POSIX path or None, got 'repo'\.$",
            id='relative-working-dir',
        ),
        pytest.param(
            lambda: IsloSandbox(vcpus=0),
            UserError,
            r'^vcpus must be an integer of at least 1 or None, got 0\.$',
            id='no-vcpus',
        ),
        pytest.param(
            lambda: IsloSandbox(memory_mb=True),
            UserError,
            r'^memory_mb must be an integer of at least 1 or None, got True\.$',
            id='bool-memory',
        ),
        pytest.param(
            lambda: IsloSandbox(pause_after_idle=1.5),  # pyright: ignore[reportArgumentType]
            UserError,
            r'^pause_after_idle must be an integer of at least 1 or None, got 1\.5\.$',
            id='fractional-pause',
        ),
        pytest.param(
            lambda: IsloSandbox(defer_loading=True, id='sandbox'),
            UserError,
            r'^`IsloSandbox` does not support `defer_loading=True`: '
            r'the workspace is selected before deferred capabilities load\.$',
            id='defer-loading',
        ),
        pytest.param(
            lambda: IsloSandbox(env={'PORT': 3000}),  # pyright: ignore[reportArgumentType]
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


async def test_backend_for_a_ref_attaches_with_the_capability_settings(fake_islo: FakeIslo) -> None:
    existing = fake_islo.add()
    (existing.root / 'work').mkdir()
    capability = IsloSandbox[None](working_dir=f'{existing.root}/work', env={'A': 'b'}, vcpus=8)
    backend = capability.backend(WorkspaceRef(provider='islo', id=existing.id))
    assert (await backend.run(['sh', '-c', 'printf "%s %s" "$A" "$(pwd -P)"'])).stdout == f'b {existing.root}/work'
    assert 'create_sandbox' not in fake_islo.methods()


async def test_foreign_ref_is_declined(fake_islo: FakeIslo) -> None:
    assert IsloSandbox[None]().get_workspace(_ctx(), ref=WorkspaceRef(provider='other', id='x')) is None


async def test_agent_without_workspace_use_does_not_create(fake_islo: FakeIslo) -> None:
    result = await Agent(TestModel(custom_output_text='done'), capabilities=[IsloSandbox()]).run('hello')
    assert result.output == 'done'
    assert not fake_islo.sandboxes


async def test_agent_runs_create_fresh_sandboxes_without_history(fake_islo: FakeIslo) -> None:
    agent = Agent(TestModel(call_tools=['run_command']), capabilities=[IsloSandbox()])

    @agent.tool
    async def run_command(ctx: RunContext[object]) -> str:
        return (await ctx.workspace.run(['echo', 'ok'])).stdout

    await agent.run('run')
    await agent.run('run')
    assert len(fake_islo.sandboxes) == 2


async def test_agent_history_attaches_the_recorded_sandbox(fake_islo: FakeIslo) -> None:
    def model(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        if isinstance(messages[-1], ModelRequest) and any(isinstance(p, ToolReturnPart) for p in messages[-1].parts):
            return ModelResponse(parts=[TextPart(content='done')])
        return ModelResponse(parts=[ToolCallPart(tool_name='remember', args={}, tool_call_id='call')])

    agent = Agent(FunctionModel(model), capabilities=[IsloSandbox()])

    @agent.tool
    async def remember(ctx: RunContext[object]) -> str:
        if await ctx.workspace.exists('marker'):
            return (await ctx.workspace.read_bytes('marker')).decode()
        await ctx.workspace.write_bytes('marker', b'saved')
        return 'created'

    first = await agent.run('remember')
    only(fake_islo).status = 'paused'
    second = await agent.run('continue', message_history=first.all_messages())
    assert 'done' in first.output and 'done' in second.output
    assert fake_islo.methods().count('create_sandbox') == 1
    assert only(fake_islo).resumes == 1


async def test_read_only_facade_identity_and_denied_command(fake_islo: FakeIslo) -> None:
    backend = IsloSandboxBackend()
    facade = ReadOnlyWorkspace(Workspace(backend))
    agent = Agent(TestModel(call_tools=['check']), capabilities=[IsloSandbox()])

    @agent.tool
    async def check(ctx: RunContext[object]) -> str:
        assert ctx.workspace is facade
        with pytest.raises(WorkspaceReadOnlyError, match='read-only'):
            await ctx.workspace.run(['echo', 'blocked'])
        return 'ok'

    result = await agent.run('check', workspace=facade)
    assert result.workspace is facade
    assert 'ok' in result.output
    assert not fake_islo.sandboxes


async def _touch(ctx: RunContext[None]) -> str:
    await ctx.workspace.write_bytes('note', b'kept')
    return 'ok'


async def test_run_closes_the_client_it_opened_and_the_result_workspace_reopens(fake_islo: FakeIslo) -> None:
    agent = Agent(TestModel(call_tools=['_touch']), deps_type=type(None), tools=[_touch], capabilities=[IsloSandbox()])
    result = await agent.run('go')
    # The run keeps one client across its operations and closes it at the end.
    assert (len(fake_islo.clients), closed(fake_islo)) == (1, 1)

    # After the run nothing would close the backend, so each operation closes its own client.
    assert await result.workspace.read_bytes(str(only(fake_islo).root / 'note')) == b'kept'
    assert (len(fake_islo.clients), closed(fake_islo)) == (2, 2)
    assert len(fake_islo.sandboxes) == 1


async def test_backend_used_outside_a_run_closes_its_client_after_each_operation(fake_islo: FakeIslo) -> None:
    # What a Temporal activity does: rebuild the backend from the ref, use it, and drop it.
    backend = IsloSandbox[None]().get_workspace(_ctx(), ref=None)
    assert isinstance(backend, IsloSandboxBackend)
    await backend.write_bytes('note', b'hi')
    assert (len(fake_islo.clients), closed(fake_islo)) == (1, 1)

    attached = IsloSandbox[None]().get_workspace(_ctx(), ref=backend.ref)
    assert isinstance(attached, IsloSandboxBackend)
    # One operation, however many requests it makes, and concurrent operations share one client,
    # closed once the last of them ends.
    first, second = await asyncio.gather(attached.read_bytes('note'), attached.exists('note'))
    assert (first, second) == (b'hi', True)
    assert (len(fake_islo.clients), closed(fake_islo)) == (2, 2)


async def test_cancelled_run_closes_its_client_after_stopping_the_command(fake_islo: FakeIslo) -> None:
    async def long_command(ctx: RunContext[None]) -> str:
        return (await ctx.workspace.run(['sleep', '30'])).stdout

    agent = Agent(
        TestModel(call_tools=['long_command']), deps_type=type(None), tools=[long_command], capabilities=[IsloSandbox()]
    )
    task = asyncio.create_task(agent.run('go'))
    with anyio.fail_after(10):
        await fake_islo.reached['get_exec_result'].wait()
    task.cancel()
    with anyio.fail_after(10), pytest.raises(asyncio.CancelledError):
        await task
    # The stop ran before the client closed.
    command, stop = fake_islo.runs
    assert 'kill -KILL' in stop.command[-1]
    await command.output
    assert command.process.returncode == 137
    assert closed(fake_islo) == 1


@pytest.mark.parametrize('supplied', ['client', 'workspace'])
async def test_run_leaves_a_supplied_client_or_workspace_open(fake_islo: FakeIslo, supplied: str) -> None:
    capability = IsloSandbox[None](client=islo.AsyncIslo()) if supplied == 'client' else IsloSandbox[None]()
    workspace = IsloSandboxBackend() if supplied == 'workspace' else None
    agent = Agent(TestModel(call_tools=['_touch']), deps_type=type(None), tools=[_touch], capabilities=[capability])
    await agent.run('go', workspace=workspace)
    assert closed(fake_islo) == 0


async def test_a_child_run_leaves_the_parent_runs_client_open(fake_islo: FakeIslo) -> None:
    child = Agent(TestModel(call_tools=['_touch']), deps_type=type(None), tools=[_touch], capabilities=[IsloSandbox()])
    closed_after_child: list[int] = []

    async def delegate(ctx: RunContext[None]) -> str:
        await child.run('go', workspace=ctx.workspace)
        closed_after_child.append(closed(fake_islo))
        return 'ok'

    parent = Agent(
        TestModel(call_tools=['_touch', 'delegate']),
        deps_type=type(None),
        tools=[_touch, delegate],
        capabilities=[IsloSandbox()],
    )
    await parent.run('go')
    assert closed_after_child == [0]
    assert closed(fake_islo) == 1


async def test_coder_works_in_the_sandbox(fake_islo: FakeIslo) -> None:
    results = await call_tools(
        [IsloSandbox(), Coder()],
        [
            ('write_file', {'path': 'hello.txt', 'content': 'hi from the sandbox\n'}),
            ('shell', {'command': 'cat hello.txt'}),
        ],
    )
    assert 'hi from the sandbox' in results[1]
    assert (only(fake_islo).root / 'hello.txt').read_text() == 'hi from the sandbox\n'
