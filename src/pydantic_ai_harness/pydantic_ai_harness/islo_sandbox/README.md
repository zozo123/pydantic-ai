# Islo Sandbox

Run your agent's commands and file edits in an isolated [Islo](https://islo.dev) cloud sandbox instead of on your machine.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/islo_sandbox/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://pydantic.dev/docs/ai/harness/#version-policy).

## Install

uv:

```bash
uv add "pydantic-ai-harness[islo,anthropic]"
```

pip:

```bash
pip install "pydantic-ai-harness[islo,anthropic]"
```

The `anthropic` extra is there because the examples use an Anthropic model; swap it for your model provider's extra. Then set `ISLO_API_KEY` to an [Islo API key](https://docs.islo.dev/cli/api-keys) (`islo api-key create <name>` makes one), and `ANTHROPIC_API_KEY` for the examples.

## Quick start

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.islo_sandbox import IsloSandbox

agent = Agent('anthropic:claude-opus-5-5', capabilities=[IsloSandbox(), Coder()])
result = agent.run_sync('Clone https://github.com/pydantic/pydantic-ai and summarize how capabilities work.')
```

`Coder`'s shell and file tools now run in the sandbox, not on your machine. With `Coder`, `RepoContext` creates the sandbox when the run starts, even without a tool call; use `Coder(repo_context=False)` for lazy creation. It keeps running, and costs money, after the run ends until Islo pauses it 15 minutes after its last command or file operation, and it keeps its disk until you delete it; see [Clean up](#clean-up).

Give the agent a project directory with `IsloSandbox(working_dir='/workspace/project')`: a new sandbox gets it created for you, and commands and relative paths start there.

Islo's default image runs Debian with git and Python, and commands run as `root` in `/workspace`, so the agent can install what else it needs. It has no ripgrep (`rg`): `Coder` can search without it, but installing `rg` makes searches faster. For your project's dependencies, pass an image that has them, such as `IsloSandbox(image='ghcr.io/my-org/my-image:latest')`. An image Islo cannot use fails on first use with `WorkspaceUnavailableError`.

## Continue in the same sandbox

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.islo_sandbox import IsloSandbox

agent = Agent('anthropic:claude-opus-5-5', capabilities=[IsloSandbox(), Coder()])
result = agent.run_sync('Clone https://github.com/pydantic/pydantic-ai and summarize how capabilities work.')

followup = agent.run_sync(
    'Which capability would you add next, and where would it live?',
    message_history=result.all_messages(),
)
```

The follow-up run finds the sandbox in the message history and works in it, so the clone is still there. Without the history, a run starts a new sandbox. If Islo paused the sandbox, the follow-up run resumes it. If the sandbox has been deleted, the run raises `WorkspaceUnavailableError` instead of starting over in an empty one.

## Choose the tools

For a narrower agent, use [`Shell`](../shell/) and [`FileSystem`](../filesystem/) instead of `Coder`, or write your own tool that runs in the sandbox:

```python {names="defined"}
from pydantic_ai import Agent, RunContext
from pydantic_ai_harness.filesystem import FileSystem
from pydantic_ai_harness.islo_sandbox import IsloSandbox
from pydantic_ai_harness.shell import Shell

agent = Agent('anthropic:claude-opus-5-5', capabilities=[IsloSandbox(), Shell(), FileSystem()])


@agent.tool
async def run_python(ctx: RunContext, code: str) -> str:
    """Run a Python snippet in the sandbox."""
    result = await ctx.workspace.run(['python3', '-c', code], timeout=10)
    return result.stdout + result.stderr
```

Commands run under `sh -c` as the image's default user, `root` on Islo's default image. File reads and writes go through Islo's file API, which acts as `root` whatever the commands run as. File reads refuse FIFOs rather than waiting for a writer. File writes create missing parent directories, write through a symlink to its target, and keep an existing file's mode.

### What a timeout stops

A command timeout starts after the sandbox is ready. Islo has no way to stop a single command and no deadline of its own, so the backend starts each command in a process group of its own (with `setsid`, which Islo's default image has) and enforces the timeout itself. A timeout or a cancelled run then runs a second command that kills that process group, background children included, and the sandbox keeps running for other commands. `WorkspaceTimeoutError` carries the last 4 KiB of each stream's output so far. If that second command does not finish within 2 seconds, the command may still be running: inspect the sandbox or delete it yourself. On an image without `setsid`, only the command's own process is killed. `timeout=None` removes the command deadline, not Islo's idle pause.

`run()` returns when the command exits, even if a background child it started is still running; output that child writes afterwards is not returned. A command whose stdout or stderr passes 1 MiB, the most of each stream Islo keeps, raises `WorkspaceOutputLimitError` once it exits, with the first 64 KiB of each stream in `stdout` and `stderr`. Redirect large or background output to a file instead; `Shell.start_command` manages its own output log.

See [Workspaces](https://pydantic.dev/docs/ai/core-concepts/workspace/) for more.

## Reattach later

To come back to the sandbox without the message history, pass its ref back as `workspace=`:

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.islo_sandbox import IsloSandbox

agent = Agent('anthropic:claude-opus-5-5', capabilities=[IsloSandbox(), Coder()])

result = agent.run_sync('Clone https://github.com/pydantic/pydantic-ai and summarize how capabilities work.')
ref = result.workspace.ref  # store this, e.g. in your database

later = agent.run_sync('Which capability would you add next, and where would it live?', workspace=ref)
```

The ref holds no credentials, so the process that reattaches needs `ISLO_API_KEY` too. Pass `workspace='new'` to start a fresh sandbox even when the message history names one.

`image`, `vcpus`, `memory_mb`, `gateway_profile`, and `pause_after_idle` only shape a new sandbox; `working_dir` and `env` apply to every command, including after you reattach. A new sandbox gets `working_dir` created for it; on an attached sandbox it must already exist, or commands fail with `WorkspaceError`.

Already have an Islo sandbox, such as one from `islo use my-sandbox`? Pass `workspace=WorkspaceRef(provider='islo', id='my-sandbox')` to a run, with `WorkspaceRef` from `pydantic_ai.workspaces`: the `id` can be the sandbox's name or its Islo ID, and the ref the run records holds the ID. The sandbox keeps the settings it was created with.

A run leaves a backend you built open. Call `await backend.aclose()` when you are done with it: that closes the client it opened from `ISLO_API_KEY`, not the sandbox, and a later operation opens a new one. A `client=` you passed is never closed.

## Prepare a sandbox before the run

To seed files before the agent starts, build the backend yourself, write through a `Workspace`, and pass the backend to the run:

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai.workspaces import Workspace
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.islo_sandbox import IsloSandbox, IsloSandboxBackend

agent = Agent('anthropic:claude-opus-5-5', capabilities=[IsloSandbox(), Coder()])


async def run_in_prepared_sandbox() -> str:
    backend = IsloSandboxBackend(working_dir='/workspace/project')
    try:
        await Workspace(backend).write_text('TASK.md', 'Add a `--version` flag to cli.py.\n')
        result = await agent.run('Do the task in TASK.md.', workspace=backend)
        return result.output
    finally:
        await backend.aclose()
        ref = backend.ref  # set once the sandbox exists
        if ref is not None:
            await IsloSandbox().destroy(ref)
```

The run uses the backend as it is, so `IsloSandbox`'s settings don't apply to it, and leaves it open: close it with `aclose()`. To keep the sandbox instead, store `backend.ref` and pass it as `workspace=` to reattach later.

## Clean up

Pydantic AI never pauses or deletes the sandbox. A sandbox `IsloSandbox` creates is paused by Islo after 15 minutes without commands or file operations (set `pause_after_idle=` to change that, or `None` to keep it running), and a later command or run resumes it. A paused sandbox stops costing compute but keeps its disk, and Islo never deletes it. Delete it with the ref you stored:

```python {names="defined"}
from pydantic_ai.workspaces import WorkspaceRef
from pydantic_ai_harness.islo_sandbox import IsloSandbox


async def delete_sandbox(ref: WorkspaceRef) -> None:
    await IsloSandbox().destroy(ref)
```

`destroy(ref)` deletes by sandbox ID or name without resuming a paused sandbox, and a sandbox that no longer exists returns quietly. Only destroy sandboxes you own. Use `backend(ref)` to construct a lazy backend for an existing sandbox; `get_sandbox()` attaches to it and returns Islo's record of it, whose `name` is what the Islo SDK's sandbox calls take (on a backend with no sandbox yet, it creates one).

A failed run returns no result, so there is no ref to store. To delete its sandbox, clean up in an `on_run_error` hook; `after_run` doesn't run when a run fails:

```python
from typing import Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Hooks
from pydantic_ai.run import AgentRunResult
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.islo_sandbox import IsloSandbox

hooks = Hooks()


@hooks.on.run_error
async def delete_failed_run_sandbox(ctx: RunContext[None], *, error: BaseException) -> AgentRunResult[Any]:
    if ctx.workspace.ref is not None:
        await IsloSandbox().destroy(ctx.workspace.ref)
    raise error


agent = Agent('anthropic:claude-opus-5-5', capabilities=[IsloSandbox(), Coder(), hooks])
```

A run that ends before its ref is stored, such as a crash, Ctrl-C, or a killed worker right after creation, leaves the sandbox behind. Every sandbox the backend creates is named `pydantic-ai-` and 16 hex characters, so `islo ls` shows it, and its ID is logged at `INFO` as `Created Islo sandbox <id> (<name>)`; delete it with `await IsloSandbox().destroy(WorkspaceRef(provider='islo', id=...))`, passing the ID or the name.

## Configuration

| Option | What it does |
| --- | --- |
| `client` | An `islo.AsyncIslo` to share across runs on one event loop, or to authenticate another way, such as its `async_token=`, or to reach other Islo URLs. Default: `None`, a client each run opens from `ISLO_API_KEY` (and `ISLO_BASE_URL` and `ISLO_COMPUTE_URL`, when set) and closes when it ends. You own a client you pass; `IsloSandbox` never closes it. |
| `image` | Container image a new sandbox runs. Default: `None`, Islo's default image. An image Islo cannot use fails on first use. |
| `vcpus` | Virtual CPUs of a new sandbox. Default: `None`, Islo's default (2). |
| `memory_mb` | Memory of a new sandbox, in MiB. Default: `None`, Islo's default (2048). |
| `gateway_profile` | Islo [gateway profile](https://docs.islo.dev/cli/gateways) of a new sandbox, which sets what it can reach and which credentials Islo adds to its requests, so the sandbox does not hold them. Default: `None`, Islo's default. |
| `pause_after_idle` | Seconds without commands or file operations before Islo pauses a new sandbox, keeping its disk; a later command resumes it. Default: `900`. `None` keeps Islo's default, which never pauses it, so it runs and is billed until you delete it. |
| `working_dir` | Absolute directory commands start in and relative paths resolve against. Default: `None`, the sandbox's own (`/workspace` on Islo's default image); set a project directory, or use relative paths for portable code. Created on a new sandbox; on an attached sandbox it must already exist. |
| `env` | Environment variables every command gets. Default: `None`. Nothing from your machine's environment reaches the sandbox. Islo sets names that are shell identifiers on each command; any other name travels in the command's command line, where the sandbox's process list shows it. |

Islo runs on the asyncio event loop only: its SDK uses asyncio, so under Trio the backend raises `UserError`.

## Durable execution

Each activity opens its own Islo client and closes it once its operation ends, so the worker has nothing to close. Run a Temporal dev server on `localhost:7233` first.

```python
import asyncio
import uuid

from pydantic_ai import Agent
from pydantic_ai.durable_exec.temporal import PydanticAIPlugin, PydanticAIWorkflow, TemporalDurability
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.islo_sandbox import IsloSandbox
from temporalio import workflow
from temporalio.client import Client
from temporalio.worker import Worker

agent = Agent(
    'anthropic:claude-opus-5-5',
    name='islo_coder',
    capabilities=[IsloSandbox(), Coder(), TemporalDurability()],
)


@workflow.defn
class SandboxWorkflow(PydanticAIWorkflow):
    __pydantic_ai_agents__ = [agent]

    @workflow.run
    async def run(self, prompt: str) -> str:
        return (await agent.run(prompt)).output


async def main() -> None:
    client = await Client.connect('localhost:7233', plugins=[PydanticAIPlugin()])
    async with Worker(client, task_queue='sandbox', workflows=[SandboxWorkflow]):
        print(
            await client.execute_workflow(
                SandboxWorkflow.run,
                'Use the shell tool to run pwd.',
                id=f'sandbox-{uuid.uuid4()}', task_queue='sandbox',
            )
        )


if __name__ == '__main__':
    asyncio.run(main())
```

`Coder`, `Shell`, and `FileSystem` work under DBOS, Temporal and Prefect. See the [Coder](https://pydantic.dev/docs/ai/harness/coder/#durable-execution), [Shell](https://pydantic.dev/docs/ai/harness/shell/#durable-execution), and [FileSystem](https://pydantic.dev/docs/ai/harness/filesystem/#durable-execution) guides for engine-specific limits.

Removing a capability while workflows using it are still running changes their replay history. Drain those workflows or use [Temporal worker versioning](https://docs.temporal.io/production-deployment/worker-deployments/worker-versioning) before deploying the change.

## Telemetry

`IsloSandbox` emits no spans of its own. Core's [instrumentation](https://pydantic.dev/docs/ai/capabilities/instrumentation/) records the sandbox on the agent run span as `pydantic_ai.workspace.provider` and `pydantic_ai.workspace.id`, and each command or file operation a tool makes runs inside that tool call's span; operations at run start, such as `RepoContext` loading repo instructions, run in the agent run span. Creating a sandbox logs its ID and name at `INFO` (`Created Islo sandbox <id> (<name>)`) on the `pydantic_ai_harness.islo_sandbox._backend` logger, so a sandbox that outlives its run can still be found.

## API reference

::: pydantic_ai_harness.islo_sandbox.IsloSandbox

::: pydantic_ai_harness.islo_sandbox.IsloSandboxBackend
