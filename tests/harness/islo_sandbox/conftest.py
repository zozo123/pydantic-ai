"""Shared fixtures for IsloSandbox tests."""

from __future__ import annotations

import importlib.util
import os
import types
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

_HAS_ISLO = importlib.util.find_spec('islo') is not None
collect_ignore = (
    [] if _HAS_ISLO else ['test_backend.py', 'test_conformance.py', 'test_islo_live.py', 'test_islo_sandbox.py']
)

# CI stops the live job after 15 minutes, before a `finally` can delete what it started. Live tests
# create sandboxes that Islo pauses after this many idle seconds, so one left behind that way stops
# running instead of running until someone deletes it.
LIVE_PAUSE_AFTER_IDLE = 300

if TYPE_CHECKING or _HAS_ISLO:  # pragma: no branch - the SDK-installed and slim jobs take opposite branches
    import anyio
    import httpx
    import islo
    from islo.core.api_error import ApiError

    from pydantic_ai.workspaces import WorkspaceRef
    from pydantic_ai_harness.islo_sandbox import IsloSandbox

    from .fake_islo import FakeIslo


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Gate the live tier, before any live fixture can start a sandbox.

    It runs only with `PYDANTIC_AI_HARNESS_ISLO_LIVE=1` and a non-empty `ISLO_API_KEY`. Otherwise
    it skips, unless `ISLO_REQUIRE_LIVE=1` (CI's live job sets it), where a missing key fails
    instead: a live job that tested nothing must not pass as a silent skip.
    """
    if item.get_closest_marker('islo_live') is None:
        return
    opted_in = os.getenv('PYDANTIC_AI_HARNESS_ISLO_LIVE') == '1'
    if opted_in and os.getenv('ISLO_API_KEY'):  # pragma: no cover - live job
        return
    reason = 'requires PYDANTIC_AI_HARNESS_ISLO_LIVE=1 and a non-empty ISLO_API_KEY'
    if os.getenv('ISLO_REQUIRE_LIVE') == '1':  # pragma: no cover - live job
        pytest.fail(f'ISLO_REQUIRE_LIVE=1 but the live tier cannot run: it {reason}.')
    pytest.skip(reason)


async def destroy_and_confirm(ref: WorkspaceRef, client: islo.AsyncIslo | None = None) -> None:  # pragma: no cover
    """Delete a live test's sandbox and wait until Islo no longer has it; the live tier runs without coverage.

    Islo can accept a delete and keep the sandbox, as observed on 2026-10-22 for one it was pausing at
    the time, so the delete is sent again until a lookup no longer finds it.
    """
    async with httpx.AsyncClient(timeout=60) as http:
        client = client or islo.AsyncIslo(httpx_client=http)
        with anyio.fail_after(90):
            while True:
                await IsloSandbox(client=client).destroy(ref)
                for _ in range(10):
                    try:
                        # Islo still finds a deleted sandbox by its ID, in the `deleted` state.
                        if (await client.sandboxes.get_sandbox_by_id(ref.id)).status == 'deleted':
                            return
                    except ApiError as error:
                        if error.status_code == 404:
                            return
                        raise
                    await anyio.sleep(1)


class _PoisonedIslo(types.ModuleType):
    """An `islo` stand-in that fails loudly on any attribute access.

    Real islo is installed in the dev venv for the live tier, so a unit test that forgets the
    `fake_islo` fixture would otherwise reach the real API and, with a developer's `ISLO_API_KEY`
    exported, create real billed sandboxes.
    """

    def __getattr__(self, name: str) -> object:  # pragma: no cover - tripwire, hit only by a misbehaving test
        raise AssertionError(
            'An islo_sandbox unit test touched the real `islo` package. '
            'Use the `fake_islo` fixture, or mark the test `islo_live`.'
        )


@pytest.fixture(autouse=True)
def _no_real_islo(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Poison `islo` for every test here except the opt-in live tier."""
    if 'islo_live' in request.keywords:  # pragma: no cover - live tier runs without coverage
        yield
        return
    monkeypatch.setattr('pydantic_ai_harness.islo_sandbox._backend.islo', _PoisonedIslo('islo'))
    monkeypatch.delenv('ISLO_API_KEY', raising=False)
    yield


if _HAS_ISLO:  # pragma: no branch - the fixture cannot be defined without its SDK-backed fake

    @pytest.fixture
    async def fake_islo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> AsyncIterator[FakeIslo]:
        """Swap the SDK's `AsyncIslo` for the fake and return its control surface.

        The real module stays in place for its error and response types, which the backend matches
        on. Its sandboxes live under `tmp_path`, resolved so the working directory `pwd -P` reports
        matches it on hosts whose temp directory sits behind a symlink.
        """
        fake = FakeIslo(tmp_path.resolve())
        monkeypatch.setattr('pydantic_ai_harness.islo_sandbox._backend.islo', islo)
        monkeypatch.setattr(islo, 'AsyncIslo', fake.client)
        monkeypatch.setenv('ISLO_API_KEY', fake.api_key)
        # The fake answers at once, so the backend's waits between checks would only slow the tests.
        for name, delay in (('_RETRY_DELAY', 0.0), ('_READY_POLL', 0.0), ('_POLL_INTERVAL', 0.01)):
            monkeypatch.setattr(f'pydantic_ai_harness.islo_sandbox._backend.{name}', delay)
        try:
            yield fake
        finally:
            await fake.close()
