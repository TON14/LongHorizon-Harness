from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

from ..agent_logs import visible_output as extract_claude_visible_output
from ..agent_registry import normalise_reasoning_effort
from .claude_isolation import build_skills_plugin, resolve_plugin_dirs
from .claude_permissions import (
    ClaudeRole,
    is_auditor_role,
    path_deny_rules,
    policy_for_role,
    snapshot_workspace,
    workspace_snapshot_diff,
)
from ..environment.base import Environment
from ..provider_errors import GUARD_REJECTION_MESSAGE
from ..semif_mcp import SERVER_NAME
from ..types import (
    DEFAULT_CLAUDE_MODEL,
    DEFAULT_TMP_DIR,
    DEFAULT_WORKSPACE_PATH,
    EpisodeBudget,
    EpisodeResult,
)
from .cli_agent import CommandAgentAdapter


def _read_mcp_config_document(path: str) -> dict[str, Any] | None:
    """Parse an operator-authored ``.mcp.json`` document's server table."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    servers = payload.get("mcpServers") if isinstance(payload, dict) else None
    return servers if isinstance(servers, dict) else None


def _mcp_server_table(path: str) -> dict[str, Any]:
    """Best-effort ``mcpServers`` table from any ``.mcp.json``-shaped file."""

    document = _read_mcp_config_document(path)
    return dict(document) if document else {}


def _claude_user_mcp_servers() -> dict[str, Any]:
    """The operator's user-scope servers, as claude itself would discover them."""

    return _mcp_server_table(str(Path.home() / ".claude.json"))


def _workspace_mcp_servers(workspace_path: str) -> dict[str, Any]:
    """Project-scope servers declared by the workspace's own ``.mcp.json``."""

    return _mcp_server_table(str(Path(workspace_path) / ".mcp.json"))


def _semif_mcp_server(defaults: dict[str, Any]) -> dict[str, Any] | None:
    """The scorer's stdio server entry for ``--mcp-config``, or None when off."""

    from ..semif_mcp import SERVER_NAME, semif_mcp_command

    command = semif_mcp_command(defaults)
    if command is None:
        return None
    return {SERVER_NAME: {"command": command[0], "args": command[1]}}


def _effective_mcp_lists(defaults: dict[str, Any], role: str) -> tuple[list[str], list[str]]:
    """Per-role (allow, blocked) lists; defaults keep everything visible.

    A role-specific list replaces the global one; ``["*"]`` in ``mcp_allow``
    means no allow-filtering. Anything else restricts: the role then loads
    ONLY admitted servers.
    """

    allow = defaults.get(f"{role}_mcp_allow", defaults.get("mcp_allow", ["*"]))
    blocked = defaults.get(f"{role}_mcp_blocked", defaults.get("mcp_blocked", []))
    return list(allow), list(blocked)


def _load_mcp_defaults() -> dict[str, Any]:
    try:
        from ..config import load_run_defaults

        return load_run_defaults()
    except Exception:
        return {}


class ClaudeCodeAdapter(CommandAgentAdapter):
    def __init__(
        self,
        *,
        model: str = DEFAULT_CLAUDE_MODEL,
        api_key: str | None = None,
        base_url: str | None = None,
        workspace_path: str = DEFAULT_WORKSPACE_PATH,
        prompt_dir: str = f"{DEFAULT_TMP_DIR}/prompts",
        mcp_config: str | None = None,
        add_dirs: list[str] | None = None,
        role: ClaudeRole = "cli_executor",
        hidden_paths: tuple[str, ...] = (),
        guard_exclude_paths: tuple[str, ...] = (),
        reasoning_effort: str | None = None,
        isolation: bool = False,
        allowed_plugins: tuple[str, ...] = (),
        allowed_skills: tuple[str, ...] = (),
    ) -> None:
        policy = policy_for_role(role)
        # Naming what is allowed only makes sense against a clean slate, so a
        # non-empty allow-list implies isolation rather than silently doing
        # nothing without the boolean.
        isolation = bool(isolation or allowed_plugins or allowed_skills)
        effort = normalise_reasoning_effort(reasoning_effort)
        env_overrides: dict[str, str] = {}
        if api_key:
            env_overrides["ANTHROPIC_API_KEY"] = api_key
            env_overrides["ANTHROPIC_AUTH_TOKEN"] = api_key
        if base_url:
            raw_url = base_url.rstrip("/")
            if raw_url.endswith("/v1"):
                raw_url = raw_url[:-3]
            env_overrides["ANTHROPIC_BASE_URL"] = raw_url
        env_overrides.update(
            {
                "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
                "CLAUDE_CODE_SKIP_PROMPT_HISTORY": "1",
                "LHHT_CLAUDE_ROLE": role,
            }
        )

        # MCP support remains opt-in. --strict-mcp-config keeps unrelated
        # user/project MCP servers out of every role.
        mcp_config = mcp_config or os.getenv("LHHT_CLAUDECODE_MCP_CONFIG")
        if mcp_config:
            candidate = Path(mcp_config).expanduser()
            if candidate.is_file():
                mcp_config = str(candidate.resolve())
        resolved_add_dirs = list(add_dirs or [])
        env_add_dirs = os.getenv("LHHT_CLAUDECODE_ADD_DIRS") or os.getenv(
            "LHHT_MCP_ADD_DIRS"
        )
        if env_add_dirs:
            resolved_add_dirs.extend(part for part in env_add_dirs.split(os.pathsep) if part)
        if resolved_add_dirs:
            raise ValueError(
                "Claude Code role isolation does not allow additional directories; "
                "put task files inside the run workspace instead."
            )

        if is_auditor_role(role):
            env_overrides.update(
                {
                    "GIT_OPTIONAL_LOCKS": "0",
                    "GIT_PAGER": "cat",
                    "PAGER": "cat",
                }
            )

        argv = [
            "claude",
            "--print",
            "--output-format",
            "stream-json",
            "--verbose",
            "--dangerously-skip-permissions",
        ]
        if isolation:
            # Project-only setting sources drop everything the *operator's
            # account* accumulated -- user plugins, skills, hooks, user-level
            # CLAUDE.md -- while keeping what the workspace repo itself
            # declares, which the task may rely on. Not `--bare`: bare mode
            # also skips the account's OAuth credentials, so every run without
            # an explicit API key dies with "Not logged in".
            argv.extend(["--setting-sources", "project"])
            plugin_dirs = resolve_plugin_dirs(tuple(allowed_plugins))
            skills_plugin = build_skills_plugin(
                tuple(allowed_skills),
                Path(prompt_dir).parent / "claude-allowed-skills",
            )
            if skills_plugin:
                plugin_dirs.append(skills_plugin)
            for plugin_dir in plugin_dirs:
                argv.extend(["--plugin-dir", plugin_dir])
            # Deliberately no --disable-slash-commands: isolation removes the
            # *operator account's* layer, and project-only setting sources
            # already keep user skills from resolving. The CLI's own built-in
            # skills ship identically for everyone and stay available, exactly
            # like its built-in tools.
        deny_tools = [*policy.disallowed_tools, *path_deny_rules(hidden_paths)]
        if deny_tools:
            argv.append("--disallowedTools")
            argv.extend(deny_tools)
        # Per-role MCP policy: [run] mcp_allow/mcp_blocked, replaceable per
        # role. Defaults -- allow ["*"], block [] -- keep claude's own server
        # discovery untouched: lhht only ADDS its registrations (the scorer
        # tool via [run.semif] mcp_tool, plus the computer-use plugin's
        # opt-in file) in one merged --mcp-config document, because the flag
        # is variadic and repeated flags race. The moment either list
        # restricts anything, lhht takes full control instead: the role loads
        # ONLY admitted servers, resolved from the operator's claude config,
        # the workspace .mcp.json, and lhht's own registrations, enforced
        # with --strict-mcp-config. An operator computer-use file that fails
        # to parse is always passed through untouched so the CLI reports the
        # broken config instead of lhht silently dropping their servers, and
        # restriction enforcement waits for a fixed file.
        defaults = _load_mcp_defaults()
        self.mcp_allow, self.mcp_blocked = _effective_mcp_lists(defaults, role)
        self.computer_mcp_configured = bool(policy.load_computer_mcp and mcp_config)
        computer_servers = (
            _read_mcp_config_document(mcp_config) if self.computer_mcp_configured else None
        )
        operator_file_broken = self.computer_mcp_configured and computer_servers is None
        semif_server = None if operator_file_broken else _semif_mcp_server(defaults)
        harness_servers = {**(computer_servers or {}), **(semif_server or {})}

        def write_config(servers: dict[str, Any]) -> None:
            merged_path = Path(prompt_dir) / f"mcp_config_{uuid.uuid4().hex[:8]}.json"
            merged_path.parent.mkdir(parents=True, exist_ok=True)
            merged_path.write_text(
                json.dumps({"mcpServers": servers}), encoding="utf-8"
            )
            argv.extend(["--mcp-config", str(merged_path)])

        self.semif_mcp_configured = semif_server is not None
        self.mcp_servers_loaded: list[str] | None = None
        restricted = (
            not (self.mcp_allow == ["*"] and not self.mcp_blocked)
            and not operator_file_broken
        )
        if restricted:
            universe = {
                **_claude_user_mcp_servers(),
                **_workspace_mcp_servers(workspace_path),
                **harness_servers,
            }
            admitted = {
                name: definition
                for name, definition in universe.items()
                if (self.mcp_allow == ["*"] or name in self.mcp_allow)
                and name not in self.mcp_blocked
            }
            self.semif_mcp_configured = SERVER_NAME in admitted
            self.mcp_servers_loaded = sorted(admitted)
            if admitted:
                write_config(admitted)
            argv.append("--strict-mcp-config")
        elif semif_server is not None:
            write_config(harness_servers)
        elif self.computer_mcp_configured:
            argv.extend(["--mcp-config", mcp_config])
        argv.extend(["--model", model])
        # Claude Code warns and continues at its default when the value is not
        # one it knows, so an unusable effort will not fail the run here.
        if effort:
            argv.extend(["--effort", effort])

        self.role = role
        self.policy = policy
        self.reasoning_effort = effort
        self.isolation = isolation
        self.allowed_plugins = tuple(allowed_plugins)
        self.allowed_skills = tuple(allowed_skills)
        # Snapshot-only exclusions: unlike hidden_paths these are not denied
        # to the agent — the guard just refrains from walking directories that
        # legitimately churn (build outputs) during an audit window.
        self.guard_exclude_paths = tuple(guard_exclude_paths)
        super().__init__(
            argv=argv,
            env=env_overrides,
            prompt_dir=prompt_dir,
            workspace_path=workspace_path,
            visible_output_parser=extract_claude_visible_output,
            hidden_paths=hidden_paths,
        )

    async def run_episode(
        self,
        prompt: str,
        env: Environment,
        budget: EpisodeBudget,
        live_trajectory_path: str | None = None,
    ) -> EpisodeResult:
        before = (
            snapshot_workspace(
                self.workspace_path,
                (*self.hidden_paths, *self.guard_exclude_paths),
            )
            if is_auditor_role(self.role)
            else None
        )
        result = await super().run_episode(
            prompt,
            env,
            budget,
            live_trajectory_path=live_trajectory_path,
        )
        result.metadata.update(
            {
                "claude_role": self.role,
                "claude_permission_mode": self.policy.permission_mode,
                "claude_dangerously_skip_permissions": True,
                "claude_hooks_enabled": False,
                "claude_native_sandbox_enabled": False,
                "claude_tool_policy": "default-minus-disallowed",
                "claude_disallowed_tools": list(self.policy.disallowed_tools),
                "claude_computer_mcp_loaded": self.computer_mcp_configured,
                "claude_semif_mcp_loaded": self.semif_mcp_configured,
                "claude_mcp_allow": list(self.mcp_allow),
                "claude_mcp_blocked": list(self.mcp_blocked),
                # None = claude's own discovery posture (lists unrestricted);
                # otherwise exactly the server names the role may load.
                "claude_mcp_servers_loaded": self.mcp_servers_loaded,
                "claude_workspace_read_only": self.policy.workspace_read_only,
                "claude_reasoning_effort": self.reasoning_effort,
            }
        )
        if before is not None:
            after = snapshot_workspace(
                self.workspace_path,
                (*self.hidden_paths, *self.guard_exclude_paths),
            )
            diff = workspace_snapshot_diff(before, after)
            result.metadata.update(diff)
            # Record the effective exclusions with every audited episode so
            # the guard's reduced coverage is visible in the run artifacts.
            result.metadata["verifier_guard_exclude_paths"] = list(self.guard_exclude_paths)
            snapshot_errors = diff.get("verifier_workspace_snapshot_errors")
            if snapshot_errors:
                # Escalate only a successful status: a real timeout (or
                # cancellation) is stronger evidence and must stay visible to
                # the runtime-failure classifier.
                if result.status == "done":
                    result.status = "error"
                guard_error = GUARD_REJECTION_MESSAGE
                result.error = f"{result.error}\n{guard_error}".strip() if result.error else guard_error
        return result
