"""Islo sandbox capability: gives agents an isolated cloud computer to work in.

`IsloSandbox` is the supported entry point; build an agent with it and add tools that use the
workspace, such as `Coder`. `IsloSandboxBackend` is the Islo implementation of Pydantic AI's
workspace backend protocol, public for applications that want to create or attach to a sandbox
themselves and pass it to a run as `workspace=`.
"""

from pydantic_ai_harness.islo_sandbox._backend import IsloSandboxBackend
from pydantic_ai_harness.islo_sandbox._capability import IsloSandbox

__all__ = [
    'IsloSandbox',
    'IsloSandboxBackend',
]
