"""Capability that supplies a Boat sandbox to an agent run.

Unlike `DaytonaSandbox`, it has no per-run client lifecycle (no backend a run claims, and no client
closed when the run ends or after each operation of a Temporal activity): Boat's SDK is synchronous
and holds no event loop state, the backend calls it in worker threads, and one `boat_sdk.BoatApi`
serves every run in the process, so a run leaves nothing to close.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.workspaces import WorkspaceBackend, WorkspaceRef
from pydantic_ai_harness._workspace_provider import check_env, check_integer, check_working_dir
from pydantic_ai_harness.boat_sandbox._backend import BoatSandboxBackend, MachineType, destroy_sandbox

if TYPE_CHECKING:
    from boat_sdk import BoatApi


@dataclass(kw_only=True)
class BoatSandbox(AbstractCapability[AgentDepsT]):
    """Supply an isolated [Boat](https://boat.dev) sandbox as the run's workspace.

    A run with no reference creates a fresh sandbox on first use; pass a `WorkspaceRef` to attach
    to an existing one. Ending a run does not delete the sandbox: Boat stops it at most `auto_stop`
    seconds after its last operation and keeps it until you delete it. Commands run under `sh -c` as the
    sandbox's `user` account, and a new sandbox gets none of your Boat account's credentials.

    This capability supplies the workspace only. Compose it with capabilities that use it, such
    as `Coder`, `Shell`, and `FileSystem`. See
    [Workspaces](https://pydantic.dev/docs/ai/core-concepts/workspace/) for more.
    """

    client: BoatApi | None = None
    """A `boat_sdk.BoatApi` to create, attach to and delete sandboxes with, for example one whose
    `boat_sdk.Configuration` routes through a proxy; you own it. When `None`, one is built from
    `BOAT_API_KEY` on first use and shared by every run in the process with that key. The SDK holds
    no event loop state, so one client serves runs on any event loop, and nothing needs closing."""

    machine_type: MachineType | None = None
    """Boat machine size for a newly created sandbox: `'small'`, `'default'`, `'large'` or
    `'xlarge'`; Boat's default when `None`. A size Boat refuses raises `WorkspaceUnavailableError`
    on first use."""

    snapshot: str | None = None
    """Named Boat snapshot a newly created sandbox starts from; Boat's base image when `None`.
    An unknown snapshot raises `WorkspaceUnavailableError` on first use."""

    auto_stop: int | None = None
    """Seconds a newly created sandbox runs without the backend extending it before Boat stops it,
    keeping its disk; `0` disables it, and `None` keeps Boat's default of an hour. Operations, and
    running commands, extend it at most once a minute (once per half of it when it is under two
    minutes), so a run that works in the sandbox more often than `auto_stop` minus that interval
    keeps it up, and an idle one stops. Attaching resumes a stopped sandbox."""

    working_dir: str | None = None
    """Absolute directory commands start in and relative paths resolve against; `None` uses the sandbox's own.
    Created on a new sandbox; an attached sandbox must already have it."""

    env: Mapping[str, str] | None = field(default=None, repr=False)
    """Environment variables every command gets, also on an attached sandbox; a command's `env` is
    layered on top. Nothing is read from the host. Values travel in each command's command line, so
    the sandbox's process list and Boat's command history show them."""

    def __post_init__(self) -> None:
        if self.defer_loading:
            # Core picks the run's workspace from the always-on capabilities only.
            raise UserError(
                '`BoatSandbox` does not support `defer_loading=True`: '
                'the workspace is selected before deferred capabilities load.'
            )
        # Checked here rather than at the first workspace operation, so a bad value fails where it is written.
        check_working_dir(self.working_dir)
        check_integer('auto_stop', self.auto_stop, minimum=0, optional=True)
        check_env(self.env)

    def backend(self, ref: WorkspaceRef) -> BoatSandboxBackend:
        """Construct a lazy backend for an existing sandbox."""
        return BoatSandboxBackend(client=self.client, ref=ref, working_dir=self.working_dir, env=self.env)

    async def destroy(self, ref: WorkspaceRef) -> None:
        """Delete a sandbox by ID without resuming it, including one Boat has stopped.

        A sandbox that no longer exists is already gone, so destroying it returns quietly.
        """
        if ref.provider != 'boat':
            raise ValueError(f"unsupported workspace provider {ref.provider!r}; expected 'boat'")
        await destroy_sandbox(self.client, ref.id)

    def get_workspace(self, ctx: RunContext[AgentDepsT], *, ref: WorkspaceRef | None) -> WorkspaceBackend | None:
        """Build the backend for this run. No I/O here: it attaches or creates on first use."""
        if ref is not None and ref.provider != 'boat':
            return None
        return BoatSandboxBackend(
            client=self.client,
            ref=ref,
            machine_type=self.machine_type,
            snapshot=self.snapshot,
            auto_stop=self.auto_stop,
            working_dir=self.working_dir,
            env=self.env,
        )
