"""Capability that supplies an Islo sandbox to an agent run."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic_ai.capabilities import AbstractCapability, WrapRunHandler
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.workspaces import WorkspaceBackend, WorkspaceRef
from pydantic_ai_harness._workspace import innermost_backend
from pydantic_ai_harness._workspace_provider import check_env, check_integer, check_working_dir
from pydantic_ai_harness.islo_sandbox._backend import IsloSandboxBackend, destroy_sandbox

if TYPE_CHECKING:
    from islo import AsyncIslo

    from pydantic_ai.agent import AgentRunResult


class _SuppliedBackend(IsloSandboxBackend):
    """The backend `IsloSandbox` built for one run; the capability closes its client when that run ends."""

    def __init__(self, *, run_id: str | None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.run_id = run_id
        # Until a run claims it, nothing closes this backend: Temporal builds one per activity and
        # drops it, so its own client closes after each operation instead.
        self._close_after_operation = True

    def claim(self, owned: bool) -> None:
        """Keep the client open across operations while the run that closes it is in progress."""
        self._close_after_operation = not owned


@dataclass(kw_only=True)
class IsloSandbox(AbstractCapability[AgentDepsT]):
    """Supply an isolated [Islo](https://islo.dev) sandbox as the run's workspace.

    A run with no reference creates a fresh sandbox on first use; pass a `WorkspaceRef` to attach
    to an existing one. Ending a run does not delete the sandbox: Islo pauses it after
    `pause_after_idle` seconds without commands or file operations and keeps it until you delete
    it. Commands run under `sh -c` as the image's default user.

    This capability supplies the workspace only. Compose it with capabilities that use it, such
    as `Coder`, `Shell`, and `FileSystem`. See
    [Workspaces](https://pydantic.dev/docs/ai/core-concepts/workspace/) for more.
    """

    client: AsyncIslo | None = None
    """An `islo.AsyncIslo` to share across runs on one event loop; you own it. When `None`, each run
    opens one from `ISLO_API_KEY` and closes it when the run ends."""

    image: str | None = None
    """Container image a newly created sandbox runs; Islo's default image when `None`. An image Islo
    cannot use raises `WorkspaceUnavailableError` on first use."""

    vcpus: int | None = None
    """Virtual CPUs of a newly created sandbox; Islo's default (2) when `None`."""

    memory_mb: int | None = None
    """Memory of a newly created sandbox, in MiB; Islo's default (2048) when `None`."""

    gateway_profile: str | None = None
    """Islo gateway profile of a newly created sandbox, which sets what it can reach and which
    credentials Islo injects into its requests; Islo's default when `None`."""

    pause_after_idle: int | None = 900
    """Seconds without commands or file operations before Islo pauses a newly created sandbox,
    keeping its disk; a later command resumes it. `None` keeps Islo's default, which never pauses
    it, so it runs and is billed until you delete it. Attaching resumes a paused sandbox."""

    working_dir: str | None = None
    """Absolute directory commands start in and relative paths resolve against; `None` uses the sandbox's own.
    Created on a new sandbox; an attached sandbox must already have it."""

    env: Mapping[str, str] | None = field(default=None, repr=False)
    """Environment variables every command gets, also on an attached sandbox; a command's `env` is
    layered on top. Nothing is read from the host."""

    def __post_init__(self) -> None:
        if self.defer_loading:
            # Core picks the run's workspace from the always-on capabilities only.
            raise UserError(
                '`IsloSandbox` does not support `defer_loading=True`: '
                'the workspace is selected before deferred capabilities load.'
            )
        # Checked here rather than at the first workspace operation, so a bad value fails where it is written.
        check_working_dir(self.working_dir)
        check_integer('vcpus', self.vcpus, optional=True)
        check_integer('memory_mb', self.memory_mb, optional=True)
        check_integer('pause_after_idle', self.pause_after_idle, optional=True)
        check_env(self.env)

    def backend(self, ref: WorkspaceRef) -> IsloSandboxBackend:
        """Construct a lazy backend for an existing sandbox."""
        return IsloSandboxBackend(client=self.client, ref=ref, working_dir=self.working_dir, env=self.env)

    async def destroy(self, ref: WorkspaceRef) -> None:
        """Delete a sandbox by ID or name without resuming it, including one Islo has paused.

        A sandbox that no longer exists is already gone, so destroying it returns quietly.
        """
        if ref.provider != 'islo':
            raise ValueError(f"unsupported workspace provider {ref.provider!r}; expected 'islo'")
        await destroy_sandbox(self.client, ref.id)

    def get_workspace(self, ctx: RunContext[AgentDepsT], *, ref: WorkspaceRef | None) -> WorkspaceBackend | None:
        """Build the backend for this run. No I/O here: it attaches or creates on first use."""
        if ref is not None and ref.provider != 'islo':
            return None
        return _SuppliedBackend(
            run_id=ctx.run_id,
            client=self.client,
            ref=ref,
            image=self.image,
            vcpus=self.vcpus,
            memory_mb=self.memory_mb,
            gateway_profile=self.gateway_profile,
            pause_after_idle=self.pause_after_idle,
            working_dir=self.working_dir,
            env=self.env,
        )

    async def wrap_run(self, ctx: RunContext[AgentDepsT], *, handler: WrapRunHandler) -> AgentRunResult[Any]:
        # Only the backend this run built: one passed in as `workspace=`, such as a parent run's
        # handed to a subagent, belongs to the run that built it and may still be in use.
        backend = innermost_backend(ctx.workspace)
        owned = backend if isinstance(backend, _SuppliedBackend) and backend.run_id == ctx.run_id else None
        if owned is not None:
            owned.claim(True)
        try:
            return await handler()
        finally:
            if owned is not None:
                # `result.workspace` outlives the run, with no one left to close it.
                owned.claim(False)
                await owned.aclose()
