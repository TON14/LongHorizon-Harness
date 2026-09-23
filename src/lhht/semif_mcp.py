"""The in-role scorer MCP server, shared across agent adapters.

``[run.semif] mcp_tool`` hands every role session a ``semif-scorer`` stdio MCP
server (the ``score`` tool, backed by the resident GPU scorer on :8790 through
``scripts/semif_mcp.py``).  Adapters render the same interpreter+script command
in their own wire format: ZCode takes per-session server objects through the
app-server protocol, Claude Code a ``--mcp-config`` JSON document.  The switch
and both paths live in the project config, so no per-project file ever needs
to be placed by hand.
"""

from __future__ import annotations

from typing import Any

SERVER_NAME = "semif-scorer"


def semif_mcp_command(defaults: dict[str, Any]) -> tuple[str, list[str]] | None:
    """``(command, args)`` for the scorer MCP server, or ``None`` when off.

    Any gap -- switch off, interpreter or script path missing -- means role
    sessions simply start without the tool; registration must never fail a run.
    """

    if not defaults.get("semif_mcp_tool"):
        return None
    python_path = defaults.get("semif_mcp_python")
    script_path = defaults.get("semif_mcp_script")
    if not (
        isinstance(python_path, str)
        and python_path
        and isinstance(script_path, str)
        and script_path
    ):
        return None
    return python_path, [script_path]
