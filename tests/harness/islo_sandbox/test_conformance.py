"""Pydantic AI's workspace backend conformance suite, run against `IsloSandboxBackend`.

`TestFakeIsloSandboxBackend` runs everywhere, over the fake Islo SDK, whose sandboxes run real
commands and file operations in temporary host directories. `TestLiveIsloSandboxBackend` runs the
same rules against real sandboxes and is gated by the `islo_live` marker.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable

import pytest

from pydantic_ai.workspaces import WorkspaceBackend, WorkspaceRef
from pydantic_ai.workspaces.conformance import WorkspaceBackendSuite
from pydantic_ai_harness.islo_sandbox import IsloSandbox, IsloSandboxBackend

from .conftest import LIVE_PAUSE_AFTER_IDLE, destroy_and_confirm
from .fake_islo import FakeIslo


def _attach(ref: WorkspaceRef) -> WorkspaceBackend:
    return IsloSandboxBackend(ref=ref)


async def _delete(backend: WorkspaceBackend) -> None:
    assert backend.ref is not None
    await IsloSandbox().destroy(backend.ref)


class TestFakeIsloSandboxBackend(WorkspaceBackendSuite):
    @pytest.fixture
    def backend(self, fake_islo: FakeIslo) -> IsloSandboxBackend:
        return IsloSandboxBackend()

    @pytest.fixture
    def fresh_backend(self, fake_islo: FakeIslo) -> Callable[[], WorkspaceBackend]:
        return IsloSandboxBackend

    @pytest.fixture
    def attach_backend(self) -> Callable[[WorkspaceRef], WorkspaceBackend]:
        return _attach

    @pytest.fixture
    def destroy_environment(self) -> Callable[[WorkspaceBackend], Awaitable[None]]:
        return _delete

    @pytest.fixture
    def filesystem_honors_shell_permissions(self) -> bool:
        # Islo's file API and its default commands run as root, which permissions do not stop; the
        # fake's run as the host user, and opt out to match Islo.
        return False


@pytest.mark.islo_live
# The live tier runs without coverage; in CI only its gate fixtures run, to skip it.
class TestLiveIsloSandboxBackend(WorkspaceBackendSuite):  # pragma: lax no cover
    # One sandbox per rule rather than one for the class: an Islo organization without credit runs one
    # sandbox at a time and pauses it within about a minute (see `test_islo_live.py`), and each rule
    # finishes well inside that. The sandbox is deleted after its rule, and its client closed.
    @pytest.fixture
    async def backend(self) -> AsyncIterator[IsloSandboxBackend]:
        backend = IsloSandboxBackend(pause_after_idle=LIVE_PAUSE_AFTER_IDLE)
        try:
            yield backend
        finally:
            try:
                if backend.ref is not None:
                    await destroy_and_confirm(backend.ref)
            finally:
                await backend.aclose()

    # Destructive rules build their own sandbox; each is deleted here if a failed rule left it behind.
    @pytest.fixture
    async def destructive_backend(self) -> AsyncIterator[Callable[[], WorkspaceBackend]]:
        created: list[IsloSandboxBackend] = []

        def create() -> WorkspaceBackend:
            backend = IsloSandboxBackend(pause_after_idle=LIVE_PAUSE_AFTER_IDLE)
            created.append(backend)
            return backend

        yield create
        for backend in created:
            try:
                if backend.ref is not None:
                    await destroy_and_confirm(backend.ref)
            finally:
                await backend.aclose()

    # The suite drops the backend it attaches; each one opened its own API client, which is closed
    # here rather than left to the garbage collector with a connection still open.
    @pytest.fixture
    async def attach_backend(self) -> AsyncIterator[Callable[[WorkspaceRef], WorkspaceBackend]]:
        attached: list[IsloSandboxBackend] = []

        def attach(ref: WorkspaceRef) -> WorkspaceBackend:
            backend = IsloSandboxBackend(ref=ref)
            attached.append(backend)
            return backend

        yield attach
        for backend in attached:
            await backend.aclose()

    @pytest.fixture
    def destroy_environment(self) -> Callable[[WorkspaceBackend], Awaitable[None]]:
        return _delete
