"""Shared fixtures for BoatSandbox tests."""

from __future__ import annotations

import importlib.util
import os
import time
import types
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import pytest

_HAS_BOAT = importlib.util.find_spec('boat_sdk') is not None
collect_ignore = (
    [] if _HAS_BOAT else ['test_backend.py', 'test_boat_live.py', 'test_boat_sandbox.py', 'test_conformance.py']
)

# CI stops the live job after 15 minutes, before a `finally` can delete what it started. Live tests
# create sandboxes that Boat stops at most this many seconds after their last operation, so one left
# behind that way stops running instead of waiting for Boat's default hour.
LIVE_AUTO_STOP = 900
# Boat refuses sandbox starts over the account's per-minute limit (12 on its smallest plan) with
# `429 rate_limited`; a live test starts at most this many.
LIVE_STARTS_PER_TEST = 4

_BACKEND = 'pydantic_ai_harness.boat_sandbox._backend'

if TYPE_CHECKING or _HAS_BOAT:  # pragma: no branch - the SDK-installed and slim jobs take opposite branches
    import boat_sdk
    from boat_sdk import boat_helpers

    from .fake_boat import FakeBoat


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Gate the live tier, before any live fixture can start a sandbox.

    It runs only with `PYDANTIC_AI_HARNESS_BOAT_LIVE=1` and a non-empty `BOAT_API_KEY`. Otherwise
    it skips, unless `BOAT_REQUIRE_LIVE=1` (CI's live job sets it), where a missing key fails
    instead: a live job that tested nothing must not pass as a silent skip.
    """
    if item.get_closest_marker('boat_live') is None:
        return
    opted_in = os.getenv('PYDANTIC_AI_HARNESS_BOAT_LIVE') == '1'
    if opted_in and os.getenv('BOAT_API_KEY'):  # pragma: no cover - live job
        return
    reason = 'requires PYDANTIC_AI_HARNESS_BOAT_LIVE=1 and a non-empty BOAT_API_KEY'
    if os.getenv('BOAT_REQUIRE_LIVE') == '1':  # pragma: no cover - live job
        pytest.fail(f'BOAT_REQUIRE_LIVE=1 but the live tier cannot run: it {reason}.')
    pytest.skip(reason)


def _wait_for_start_budget() -> None:  # pragma: no cover - live tier
    """Wait, up to about a minute, until Boat's start limit leaves room for one live test's sandboxes."""
    headers = {'Authorization': f'Bearer {os.environ["BOAT_API_KEY"]}'}
    deadline = time.monotonic() + 65
    while time.monotonic() < deadline:
        response = httpx.get('https://boat.dev/api/v1/limits', headers=headers, timeout=30)
        # A key without `account.read` cannot read its limits: start anyway.
        if not response.is_success or response.json()['starts']['minute']['remaining'] >= LIVE_STARTS_PER_TEST:
            return
        time.sleep(5)


class _PoisonedModule(types.ModuleType):
    """A `boat_sdk` stand-in that fails loudly on any attribute access.

    Real boat-sdk is installed in the dev venv for the live tier, so a unit test that forgets the
    `fake_boat` fixture would otherwise reach the real SDK and, with a developer's `BOAT_API_KEY`
    exported, create real billed sandboxes.
    """

    def __getattr__(self, name: str) -> object:  # pragma: no cover - tripwire, hit only by a misbehaving test
        raise AssertionError(
            'A boat_sandbox unit test touched the real `boat_sdk` package. '
            'Use the `fake_boat` fixture, or mark the test `boat_live`.'
        )


@pytest.fixture(autouse=True)
def _no_real_boat(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Poison `boat_sdk` for every test here except the opt-in live tier."""
    if 'boat_live' in request.keywords:  # pragma: no cover - live tier runs without coverage
        _wait_for_start_budget()
        yield
        return
    monkeypatch.setattr(f'{_BACKEND}.boat', _PoisonedModule('boat_sdk'))
    monkeypatch.setattr(f'{_BACKEND}.boat_helpers', _PoisonedModule('boat_sdk.boat_helpers'))
    monkeypatch.delenv('BOAT_API_KEY', raising=False)
    yield


if _HAS_BOAT:  # pragma: no branch - the fixture cannot be defined without its SDK-backed fake

    @pytest.fixture
    def fake_boat(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[FakeBoat]:
        """Swap the SDK's `BoatApi` and its streaming helper for the fake, and return its control surface.

        The real module stays in place for its models, exceptions and other helpers, which the backend
        builds and matches on. Each test gets an API key of its own, so the SDK client the backend
        keeps per key is the fake of that test. Its sandboxes live under `tmp_path`, resolved so the
        working directory `pwd -P` reports matches it on hosts whose temp directory sits behind a symlink.
        """
        fake = FakeBoat(tmp_path.resolve(), api_key=f'test-key-{uuid.uuid4().hex}')
        monkeypatch.setenv('BOAT_API_KEY', fake.api_key)
        monkeypatch.setattr(f'{_BACKEND}.boat', boat_sdk)
        monkeypatch.setattr(f'{_BACKEND}.boat_helpers', boat_helpers)
        monkeypatch.setattr(boat_sdk, 'BoatApi', fake.client)
        monkeypatch.setattr(boat_helpers, 'follow_command', fake.follow_command)
        # The fake answers at once, so the backend's waits between checks would only slow the tests.
        for name in ('_RETRY_DELAY', '_READY_POLL'):
            monkeypatch.setattr(f'{_BACKEND}.{name}', 0.0)
        try:
            yield fake
        finally:
            fake.close()
