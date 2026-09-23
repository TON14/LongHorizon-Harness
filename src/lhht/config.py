from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from .types import MAX_ROUNDS

PROJECT_CONFIG_PATH = Path(".lhht/config.toml")

_AGENT_CHOICES = {"claude_code", "codex", "deepseek_harness", "opencode", "zcode"}
_ROLE_NAMES = {
    "manager",
    "executor",
    "gui_executor",
    "cli_executor",
    "auditor",
    "gui_auditor",
    "cli_auditor",
    "final_response",
}
_TIMEOUT_NAMES = {"manager", "gui_executor", "cli_executor", "auditor"}
_SEMIF_KEYS = {
    "enabled",
    "command",
    "model",
    "revision",
    "gguf",
    "threshold",
    "timeout_seconds",
    "auditor_fast",
    "auditor_fast_threshold",
    "effort_routing",
    "effort_threshold",
    "report_selection",
    "report_selection_k",
    "report_selection_threshold",
}
_RUN_KEYS = {
    "agent",
    "model",
    "reasoning_effort",
    "env",
    "runs_root",
    "workspace",
    "harness_dir",
    "log_dir",
    "base_url",
    "prompt_language",
    "claude_mcp_config",
    "codex_mcp_config",
    "claude_isolation",
    "claude_allowed_plugins",
    "claude_allowed_skills",
    "mcp_add_dirs",
    "guard_exclude_paths",
    "guard_exclude_git",
    "max_rounds",
    "dashboard",
    "dashboard_port",
    "roles",
    "timeouts",
    "semif",
}
_STRING_KEYS = {
    "model",
    "runs_root",
    "workspace",
    "harness_dir",
    "log_dir",
    "base_url",
    "claude_mcp_config",
    "codex_mcp_config",
}

CONFIG_TEMPLATE = """# LongHorizon-Harness project defaults.
# Explicit CLI arguments override these values.

[run]
agent = "codex"
model = "gpt-5.6-sol"

# Reasoning depth, forwarded to whichever backend exposes it (Codex through
# `model_reasoning_effort`, Claude Code through `--effort`, OpenCode through
# `--variant`). Any value the backend accepts is allowed, so a newer tier does
# not need a harness release. Leave unset to keep the provider's own setting.
# reasoning_effort = "high"

env = "local"
runs_root = "./.lhht/runs"
# Agents work in the directory lhht was started from unless set here.
# workspace = "./workspace"
# harness_dir = "./.lhht/runs/<run-id>/harness"
# log_dir = "./lhht"

# base_url = "https://api.example.com/v1"

prompt_language = "en"
# Each agent reads its own format; installed plugins are loaded automatically.
# claude_mcp_config = "/path/to/mcp.json"
# codex_mcp_config = "/path/to/mcp.toml"
mcp_add_dirs = []

# Start Claude Code agents without this account's own plugins, skills, hooks
# and user-level CLAUDE.md, so an operator's toolbox cannot leak into runs
# that never asked for it. The CLI's built-in skills stay available. The
# allow-lists re-admit exactly the named installed plugins / user skills, and
# naming anything implies isolation. To allow nothing, leave the lists out
# entirely (or []): [""] is an empty *name* and is rejected at startup.
# claude_isolation = true
# claude_allowed_plugins = ["<installed-plugin-name>"]
# claude_allowed_skills = ["<skill-name>"]

# Build/cache directories the auditor read-only guard should not snapshot,
# e.g. ["target", "node_modules", "build", ".venv"]. Agents can still read
# them. Exclusions must stay inside the workspace; ".git" and harness-owned
# control/state paths are rejected at startup, and the effective list is
# echoed at run start and recorded in each audited episode's metadata.
# Passing --guard-exclude-path replaces this list rather than adding to it.
guard_exclude_paths = []
# Deliberately drop .git from audit snapshots. This is an audit blind spot --
# hooks, refs and history become unwatched -- so it has its own named switch
# instead of hiding in the list above. Meant for workspaces where concurrent
# runs legitimately share one repository and every sibling commit would
# otherwise invalidate an open audit window.
# guard_exclude_git = true

max_rounds = 25
dashboard = true
# Embedded dashboards use an OS-assigned port by default so concurrent runs
# cannot accidentally share or race a fixed listener. Standalone `web` keeps
# its explicit 8799 default for the operator-facing control plane.
dashboard_port = 0

[run.timeouts]
manager = 300
gui_executor = 1800
cli_executor = 1800
auditor = 300

[run.roles.manager]
# agent = "codex"
# model = "gpt-5.6-sol"
# reasoning_effort = "high"

[run.roles.executor]
# agent = "codex"
# model = "gpt-5.6-sol"

[run.roles.gui_executor]
# agent = "codex"
# model = "gpt-5.6-sol"

[run.roles.cli_executor]
# agent = "codex"
# model = "gpt-5.6-sol"

[run.roles.auditor]
# agent = "codex"
# model = "gpt-5.6-sol"

[run.roles.gui_auditor]
# agent = "codex"
# model = "gpt-5.6-sol"

[run.roles.cli_auditor]
# agent = "codex"
# model = "gpt-5.6-sol"

# Writes the closing reply to you; falls back to the manager's agent/model.
[run.roles.final_response]
# agent = "codex"
# model = "gpt-5.6-sol"
"""


class ProjectConfigError(ValueError):
    pass


def create_project_config(
    path: str | Path = PROJECT_CONFIG_PATH,
    *,
    force: bool = False,
) -> Path:
    target = Path(path)
    if target.exists() and not force:
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(CONFIG_TEMPLATE, encoding="utf-8")
    return target


def load_run_defaults(path: str | Path = PROJECT_CONFIG_PATH) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        return {}
    try:
        with source.open("rb") as fh:
            payload = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ProjectConfigError(f"could not read {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ProjectConfigError(f"{source} must contain a TOML table")
    unknown_root = set(payload) - {"run"}
    if unknown_root:
        raise ProjectConfigError(f"unknown top-level key(s): {_names(unknown_root)}")
    run = payload.get("run", {})
    if not isinstance(run, dict):
        raise ProjectConfigError("[run] must be a TOML table")
    return _flatten_run_table(run)


def _flatten_run_table(run: dict[str, Any]) -> dict[str, Any]:
    unknown = set(run) - _RUN_KEYS
    if unknown:
        raise ProjectConfigError(f"unknown [run] key(s): {_names(unknown)}")

    defaults: dict[str, Any] = {}
    for key in _STRING_KEYS:
        if key in run:
            defaults[key] = _string(run[key], f"run.{key}")

    if "agent" in run:
        defaults["agent"] = _choice(run["agent"], "run.agent", _AGENT_CHOICES)
    if "reasoning_effort" in run:
        defaults["reasoning_effort"] = _reasoning_effort(
            run["reasoning_effort"], "run.reasoning_effort"
        )
    if "env" in run:
        defaults["env"] = _choice(run["env"], "run.env", {"local"})
    if "prompt_language" in run:
        defaults["prompt_language"] = _choice(
            run["prompt_language"], "run.prompt_language", {"en", "zh"}
        )
    if "max_rounds" in run:
        defaults["max_rounds"] = _positive_int(run["max_rounds"], "run.max_rounds")
    if "dashboard" in run:
        defaults["dashboard"] = _boolean(run["dashboard"], "run.dashboard")
    if "dashboard_port" in run:
        defaults["dashboard_port"] = _port(run["dashboard_port"], "run.dashboard_port")
    if "claude_isolation" in run:
        defaults["claude_isolation"] = _boolean(run["claude_isolation"], "run.claude_isolation")
    for key, dest in (
        ("claude_allowed_plugins", "claude_allowed_plugin"),
        ("claude_allowed_skills", "claude_allowed_skill"),
    ):
        if key in run:
            value = run[key]
            if not isinstance(value, list) or not all(
                isinstance(item, str) and item for item in value
            ):
                # [""] is the natural but wrong spelling of "allow nothing",
                # so the refusal must name the right one.
                raise ProjectConfigError(
                    f"run.{key} must be an array of non-empty names; "
                    "to allow nothing, use [] or omit the key"
                )
            defaults[dest] = list(value)
    if "mcp_add_dirs" in run:
        value = run["mcp_add_dirs"]
        if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
            raise ProjectConfigError("run.mcp_add_dirs must be an array of non-empty strings")
        defaults["mcp_add_dir"] = list(value)
    if "guard_exclude_git" in run:
        defaults["guard_exclude_git"] = _boolean(run["guard_exclude_git"], "run.guard_exclude_git")
    if "guard_exclude_paths" in run:
        value = run["guard_exclude_paths"]
        if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
            raise ProjectConfigError("run.guard_exclude_paths must be an array of non-empty strings")
        defaults["guard_exclude_path"] = list(value)

    roles = run.get("roles", {})
    if not isinstance(roles, dict):
        raise ProjectConfigError("[run.roles] must be a TOML table")
    unknown_roles = set(roles) - _ROLE_NAMES
    if unknown_roles:
        raise ProjectConfigError(f"unknown role(s): {_names(unknown_roles)}")
    for role, values in roles.items():
        if not isinstance(values, dict):
            raise ProjectConfigError(f"[run.roles.{role}] must be a TOML table")
        unknown_role_keys = set(values) - {"agent", "model", "reasoning_effort"}
        if unknown_role_keys:
            raise ProjectConfigError(
                f"unknown [run.roles.{role}] key(s): {_names(unknown_role_keys)}"
            )
        if "agent" in values:
            defaults[f"{role}_agent"] = _choice(
                values["agent"], f"run.roles.{role}.agent", _AGENT_CHOICES
            )
        if "model" in values:
            defaults[f"{role}_model"] = _string(
                values["model"], f"run.roles.{role}.model"
            )
        if "reasoning_effort" in values:
            defaults[f"{role}_reasoning_effort"] = _reasoning_effort(
                values["reasoning_effort"], f"run.roles.{role}.reasoning_effort"
            )

    timeouts = run.get("timeouts", {})
    if not isinstance(timeouts, dict):
        raise ProjectConfigError("[run.timeouts] must be a TOML table")
    unknown_timeouts = set(timeouts) - _TIMEOUT_NAMES
    if unknown_timeouts:
        raise ProjectConfigError(f"unknown timeout role(s): {_names(unknown_timeouts)}")
    for role, value in timeouts.items():
        defaults[f"{role}_timeout"] = _positive_int(value, f"run.timeouts.{role}")

    # Semantic salvage stays off unless the operator opts in; the semif_*
    # keys exist only when the table was written, so an absent section
    # leaves the defaults byte-for-byte identical to a config without it.
    semif = run.get("semif", {})
    if not isinstance(semif, dict):
        raise ProjectConfigError("[run.semif] must be a TOML table")
    unknown_semif = set(semif) - _SEMIF_KEYS
    if unknown_semif:
        raise ProjectConfigError(f"unknown [run.semif] key(s): {_names(unknown_semif)}")
    if "enabled" in semif:
        defaults["semif_enabled"] = _boolean(semif["enabled"], "run.semif.enabled")
    for key in ("command", "model", "revision", "gguf"):
        if key in semif:
            defaults[f"semif_{key}"] = _string(semif[key], f"run.semif.{key}")
    if "threshold" in semif:
        defaults["semif_threshold"] = _threshold(
            semif["threshold"], "run.semif.threshold"
        )
    if "timeout_seconds" in semif:
        defaults["semif_timeout_seconds"] = _positive_int(
            semif["timeout_seconds"], "run.semif.timeout_seconds"
        )
    # The auditor-fast pre-gate has its own switch and threshold inside the
    # same table, but unlike salvage it stays silently off when the scorer is
    # unusable: skipping the slow auditor is an optimization, so a missing or
    # misconfigured gate must degrade to today's behavior, not refuse to load.
    if "auditor_fast" in semif:
        defaults["semif_auditor_fast"] = _boolean(
            semif["auditor_fast"], "run.semif.auditor_fast"
        )
    if "auditor_fast_threshold" in semif:
        defaults["semif_auditor_fast_threshold"] = _threshold(
            semif["auditor_fast_threshold"], "run.semif.auditor_fast_threshold"
        )
    # Effort routing is the same kind of optimization: picking among executor
    # adapters can wait for a usable scorer, so it too stays silently off
    # when the scorer is not configured rather than refusing to load.
    if "effort_routing" in semif:
        defaults["semif_effort_routing"] = _boolean(
            semif["effort_routing"], "run.semif.effort_routing"
        )
    if "effort_threshold" in semif:
        defaults["semif_effort_threshold"] = _threshold(
            semif["effort_threshold"], "run.semif.effort_threshold"
        )
    # Report selection is the same kind of optimization: ranking which past
    # audit reports ride along in the round's prompts can wait for a usable
    # scorer, so it too stays silently off when the scorer is not configured
    # rather than refusing to load.
    if "report_selection" in semif:
        defaults["semif_report_selection"] = _boolean(
            semif["report_selection"], "run.semif.report_selection"
        )
    if "report_selection_k" in semif:
        defaults["semif_report_selection_k"] = _positive_int(
            semif["report_selection_k"], "run.semif.report_selection_k"
        )
    if "report_selection_threshold" in semif:
        defaults["semif_report_selection_threshold"] = _threshold(
            semif["report_selection_threshold"],
            "run.semif.report_selection_threshold",
        )
    if defaults.get("semif_enabled"):
        missing = [
            key for key in ("command", "model", "revision")
            if f"semif_{key}" not in defaults
        ]
        if missing:
            # Salvage silently doing nothing is the one failure an operator
            # cannot see from the run output, so an enabled-but-unusable
            # section must refuse to load rather than quietly disable.
            raise ProjectConfigError(
                "run.semif.enabled = true requires "
                + ", ".join(f"run.semif.{key}" for key in missing)
            )
    return defaults


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProjectConfigError(f"{name} must be a non-empty string")
    return value


def _choice(value: Any, name: str, choices: set[str]) -> str:
    result = _string(value, name)
    if result not in choices:
        raise ProjectConfigError(f"{name} must be one of: {_names(choices)}")
    return result


def _reasoning_effort(value: Any, name: str) -> str:
    from .agent_registry import normalise_reasoning_effort

    try:
        return normalise_reasoning_effort(_string(value, name))
    except ValueError as exc:
        raise ProjectConfigError(f"{name}: {exc}") from exc


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ProjectConfigError(f"{name} must be an integer of at least 1")
    if name.endswith("max_rounds") and value > MAX_ROUNDS:
        raise ProjectConfigError(f"{name} must be at most {MAX_ROUNDS}")
    return value


def _port(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 65535:
        raise ProjectConfigError(f"{name} must be an integer from 0 to 65535")
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ProjectConfigError(f"{name} must be true or false")
    return value


def _threshold(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not 0 < value <= 1
    ):
        raise ProjectConfigError(
            f"{name} must be a number greater than 0 and at most 1"
        )
    return float(value)


def _names(values: set[str]) -> str:
    return ", ".join(sorted(values))
