"""Integration tests that require a real Islo sandbox.

The fake-backed suites cover the harness-owned logic: deadline handling, protocol translation,
and exception mapping. This live tier admits only what a correctly written fake could not catch:
real process execution, a client-owned deadline over a real process group, one filesystem shared
by Islo's file API and its commands, and real lifecycle state in Islo's control plane.

Admission rule:
  A test belongs here only when its docstring can name the fake-encoded assumption it
  validates against real Islo behavior.

Gating:
  * `islo_live` marker separates this tier from fake-backed tests.
  * skipped unless `PYDANTIC_AI_HARNESS_ISLO_LIVE=1` opts in explicitly.
  * also requires a non-empty `ISLO_API_KEY`; without one it skips, or fails when
    `ISLO_REQUIRE_LIVE=1` (CI's live job). The gate is `pytest_runtest_setup` in `conftest.py`.
  * one sandbox runs at a time: tests that need no settings of their own share one for a few seconds
    at a time, and every sandbox is deleted, confirming Islo no longer has it.

An organization without credit is a poor fit: observed on 2026-10-22, Islo paused each of its
sandboxes within about a minute of starting it, refused to resume one (`402 BILLING_NOT_ALLOWED`),
and refused some creations the same way, more often the more sandboxes it had started recently.

Run locally:
`PYDANTIC_AI_HARNESS_ISLO_LIVE=1 uv run pytest -m islo_live tests/harness/islo_sandbox`
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager

import anyio
import httpx
import islo
import pytest
from islo.types import AutoResumePolicy
from pytest_examples import CodeExample

from pydantic_ai.workspaces import (
    WorkspaceOutputLimitError,
    WorkspaceRef,
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
)
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.islo_sandbox import IsloSandbox, IsloSandboxBackend

from .._docs_examples import documented_cleanup, python_blocks, run_block
from .._tool_calls import call_tools
from .conftest import LIVE_PAUSE_AFTER_IDLE, SharedSandbox, destroy_and_confirm

pytestmark = pytest.mark.islo_live


def _unique(prefix: str) -> str:
    return f'{prefix}-{uuid.uuid4().hex}'


@pytest.fixture
async def client() -> AsyncIterator[islo.AsyncIslo]:
    """A caller-owned API client for one test's backends, closed when the test ends."""
    async with httpx.AsyncClient(timeout=60) as http:
        yield islo.AsyncIslo(httpx_client=http)


@pytest.fixture(scope='module')
async def shared() -> AsyncIterator[SharedSandbox]:
    """A small sandbox the tests that need no settings of their own share, a few at a time."""
    shared = SharedSandbox()
    try:
        yield shared
    finally:
        await shared.close()


@asynccontextmanager
async def _owned(
    client: islo.AsyncIslo, *, vcpus: int | None = None, memory_mb: int | None = None, working_dir: str | None = None
) -> AsyncGenerator[IsloSandboxBackend]:
    """Create a sandbox and delete it on the way out, even when the test deleted it already."""
    backend = IsloSandboxBackend(
        client=client,
        pause_after_idle=LIVE_PAUSE_AFTER_IDLE,
        vcpus=vcpus,
        memory_mb=memory_mb,
        working_dir=working_dir,
    )
    try:
        await backend.working_dir()
        yield backend
    finally:
        if backend.ref is not None:
            await destroy_and_confirm(backend.ref, client)


async def test_creates_a_sandbox_with_its_settings_and_runs_a_command(client: islo.AsyncIslo) -> None:
    """Validates the fake-encoded assumptions that a command reports its real output and exit code, that
    the creation settings reach Islo (the idle pause with resume on activity, and the size), that
    `mkdir -p` makes `working_dir` on a new sandbox, so the first command starts there, that a relative
    path the file API writes lands in it, and that the file API keeps an existing file's mode, as the
    docs say."""
    directory = f'/workspace/{_unique("project")}/nested'
    async with _owned(client, vcpus=1, memory_mb=1024, working_dir=directory) as backend:
        result = await backend.run('echo out; echo err 1>&2; exit 3', shell=True, timeout=60)
        assert (result.stdout, result.stderr, result.exit_code) == ('out\n', 'err\n', 3)

        record = await backend.get_sandbox()
        assert backend.ref == WorkspaceRef(provider='islo', id=record.id)
        assert record.name.startswith('pydantic-ai-')
        assert record.lifecycle is not None
        assert (record.lifecycle.pause_after_idle, record.lifecycle.auto_resume) == (
            LIVE_PAUSE_AFTER_IDLE,
            AutoResumePolicy.ON_ACTIVITY,
        )
        assert record.spec is not None
        assert (record.spec.vcpus, record.spec.memory_mb) == (1, 1024)

        assert await backend.working_dir() == directory
        await backend.write_bytes('relative.txt', b'here')
        assert (await backend.run(['cat', 'relative.txt'], timeout=60)).stdout == 'here'
        assert (await backend.run(['chmod', '750', 'relative.txt'], timeout=60)).exit_code == 0
        await backend.write_bytes('relative.txt', b'again')
        assert (await backend.run(['stat', '-c', '%a', 'relative.txt'], timeout=60)).stdout == '750\n'


async def test_reattach_to_a_deleted_sandbox_is_unavailable(client: islo.AsyncIslo) -> None:
    """Validates the fake-encoded assumptions that `get_sandbox_by_id` reports a deleted sandbox `deleted`
    at once, so a new attach is unavailable, and that the held backend's calls answer `404
    SANDBOX_NOT_FOUND` once Islo has removed the sandbox, rather than a missing file or a retryable
    error. Observed on 2026-10-22: those calls can still succeed for some seconds after the delete."""
    path = f'/tmp/{_unique("deleted")}.txt'
    async with _owned(client) as owner:
        await owner.write_bytes(path, b'before-delete')
        assert owner.ref is not None
        await destroy_and_confirm(owner.ref, client)

        with pytest.raises(WorkspaceUnavailableError):
            await IsloSandboxBackend(client=client, ref=owner.ref).working_dir()
        with anyio.fail_after(60):
            while True:
                try:
                    await owner.read_bytes(path)
                except WorkspaceUnavailableError:
                    break
                await anyio.sleep(1)
        with pytest.raises(WorkspaceUnavailableError):
            await owner.run(['true'], timeout=30)


async def test_destroy_deletes_a_paused_sandbox_without_resuming_it(
    client: islo.AsyncIslo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Validates the fake-encoded assumption that Islo deletes a paused sandbox as it is, so `destroy` never
    resumes one, and that destroying it again returns quietly."""
    async with _owned(client) as backend:
        record = await backend.get_sandbox()
        await client.sandboxes.pause_sandbox(record.name)

        async def refuse(*args: object, **kwargs: object) -> None:
            raise AssertionError('destroy resumed the sandbox')

        monkeypatch.setattr(client.sandboxes, 'resume_sandbox', refuse)
        assert backend.ref is not None
        await destroy_and_confirm(backend.ref, client)
        await IsloSandbox(client=client).destroy(backend.ref)


# The README's Python blocks are the same as this page's.
_DOCS_BLOCKS = python_blocks('docs/harness/islo-sandbox.md')


@pytest.mark.parametrize('example', [pytest.param(block, id=f'line {block.start_line}') for block in _DOCS_BLOCKS])
def test_docs_example(example: CodeExample) -> None:
    """Every example on the docs page runs as written, and its agent's tools do their work in a real sandbox.

    The fake stands in for Islo, so only this shows the page's code, its default settings, and its
    cleanup work against the real service. A follow-up run, from the message history or a stored ref,
    works in the first run's sandbox.
    """
    _, runs = run_block(example, cleanup=documented_cleanup(_DOCS_BLOCKS, 'delete_sandbox'))
    assert all(run.used_sandbox for run in runs), runs
    assert len({run.ref for run in runs}) <= 1, runs


# The tests below share one sandbox, so they run last: none of the tests above, which start their own,
# runs while it does.
async def test_output_past_the_stream_limit_raises_with_its_head(shared: SharedSandbox) -> None:
    """Validates the fake-encoded assumption that Islo keeps 1 MiB of each stream: output just under it
    comes back whole, and output past it raises `WorkspaceOutputLimitError` with the first 64 KiB."""
    backend = await shared.get()
    result = await backend.run(['seq', '1', '150000'], timeout=60)
    assert result.stdout == ''.join(f'{i}\n' for i in range(1, 150001))

    with pytest.raises(WorkspaceOutputLimitError) as exc_info:
        await backend.run(['sh', '-c', 'seq 1 400000; echo tail >&2'], timeout=60)
    assert exc_info.value.stdout == ''.join(f'{i}\n' for i in range(1, 400001))[: 64 * 1024]
    assert exc_info.value.stderr == 'tail\n'


async def test_reattach_by_id_or_name_reaches_the_same_sandbox(client: islo.AsyncIslo, shared: SharedSandbox) -> None:
    """Validates the fake-encoded assumptions that `get_sandbox_by_id` finds the sandbox a ref names and
    that its name, as `islo use` users know it, attaches to it too, recording the ID."""
    path = f'/tmp/{_unique("reattach")}.txt'
    owner = await shared.get()
    await owner.write_bytes(path, b'shared')
    assert owner.ref is not None
    name = (await owner.get_sandbox()).name

    by_id = IsloSandboxBackend(client=client, ref=owner.ref)
    by_name = IsloSandboxBackend(client=client, ref=WorkspaceRef(provider='islo', id=name))

    assert await by_id.read_bytes(path) == b'shared'
    assert (await by_name.run(['cat', path])).stdout == 'shared'
    assert by_id.ref == by_name.ref == owner.ref


async def test_coder_tools_run_in_the_sandbox(shared: SharedSandbox) -> None:
    """Validates the fake-encoded assumption that a relative path the file API writes is where a command
    started in the working directory finds it, so `Coder`'s tools agree."""
    results = await call_tools(
        [Coder()],
        [
            ('write_file', {'path': 'hello.py', 'content': "print('hi from islo')\n"}),
            ('shell', {'command': 'python3 hello.py'}),
        ],
        workspace=await shared.get(),
    )

    assert 'hi from islo' in results[1]


async def test_commands_get_a_usable_environment(shared: SharedSandbox) -> None:
    """Validates the fake-encoded assumption that a command sees the image's `PATH` and `HOME`, and what the
    docs say the default image has: `root` in `/workspace`, git and Python. A per-call `env` adds to the
    environment, including a name that is not a shell identifier, which travels through `env --`."""
    backend = await shared.get()
    result = await backend.run(
        'echo "$PATH"; echo "$HOME"; whoami; pwd -P; git --version; python3 --version', shell=True, timeout=60
    )
    path, home, user, cwd, git, python = result.stdout.splitlines()
    assert (home, user, cwd) == ('/root', 'root', '/workspace')
    assert path and git.startswith('git version') and python.startswith('Python 3')
    assert await backend.working_dir() == '/workspace'

    script = 'import os; print(os.environ["FOO"], os.environ["a-b"], os.environ["PATH"] == os.environ["P"])'
    result = await backend.run(['python3', '-c', script], env={'FOO': '1', 'a-b': '2', 'P': path}, timeout=60)
    assert result.stdout == '1 2 True\n'


async def test_timeout_kills_the_command_and_its_children(shared: SharedSandbox) -> None:
    """Validates the fake-encoded assumptions that the command's output files hold what it printed before
    the deadline, and that the stop kills the process group `setsid` gave it, a background child included."""
    marker = f'/tmp/{_unique("after-deadline")}'
    backend = await shared.get()
    with pytest.raises(WorkspaceTimeoutError) as exc_info:
        await backend.run(
            f'echo DIAGNOSTIC; (sleep 6; touch {marker}.child) & sleep 6; touch {marker}', shell=True, timeout=2
        )

    assert exc_info.value.stdout == 'DIAGNOSTIC\n'
    # Outlasts the stopped command's `sleep 6`, so a process left running would have written a marker.
    assert (await backend.run(['sleep', '6'], timeout=30)).exit_code == 0
    assert not await backend.exists(marker)
    assert not await backend.exists(f'{marker}.child')
