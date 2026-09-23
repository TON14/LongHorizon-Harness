"""Per-role MCP server visibility, shared by every agent adapter.

``[run] mcp_allow`` / ``mcp_blocked`` (replaceable per role in
``[run.roles.<role>]``) decide which MCP servers each role may load, on every
backend. Defaults -- allow ``["*"]``, block ``[]`` -- keep each adapter's
hands-off behavior: claude discovers its own servers, codex and zcode load
only what lhht registers. The moment either list restricts anything, a role
gets ONLY admitted servers; the block list always wins over the allow list.
Server names are the middle segment of a tool id (``mcp__<server>__<tool>``);
``semif-scorer`` is the harness's own scorer server and is filtered like any
other.
"""

from __future__ import annotations

from typing import Any

OPEN_ALLOW = ("*",)


def load_mcp_defaults() -> dict[str, Any]:
    """Project defaults for the lists, best-effort: config errors stay open."""

    try:
        from .config import load_run_defaults

        return load_run_defaults()
    except Exception:
        return {}


def effective_mcp_lists(
    defaults: dict[str, Any], role: str | None
) -> tuple[list[str], list[str]]:
    """(allow, blocked) for one role; a role-specific list replaces the global."""

    allow = defaults.get(f"{role}_mcp_allow", defaults.get("mcp_allow", list(OPEN_ALLOW)))
    blocked = defaults.get(f"{role}_mcp_blocked", defaults.get("mcp_blocked", []))
    return list(allow), list(blocked)


def mcp_restricted(allow: list[str], blocked: list[str]) -> bool:
    """True when the lists restrict anything and the adapter must take control."""

    return not (allow == list(OPEN_ALLOW) and not blocked)


def mcp_admits(allow: list[str], blocked: list[str], name: str) -> bool:
    """Whether one server name survives the lists; block wins over allow."""

    return (allow == list(OPEN_ALLOW) or name in allow) and name not in blocked
