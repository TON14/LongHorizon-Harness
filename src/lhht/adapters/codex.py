from __future__ import annotations

import json
import os
import tomllib

from ..types import DEFAULT_CODEX_MODEL, DEFAULT_TMP_DIR, DEFAULT_WORKSPACE_PATH
from ..agent_logs import visible_output as extract_codex_visible_output
from ..agent_registry import normalise_reasoning_effort
from ..mcp_policy import (
    effective_mcp_lists,
    load_mcp_defaults,
    mcp_admits,
    mcp_restricted,
)
from ..utils.agent_cli import resolve_codex_binary
from .cli_agent import CommandAgentAdapter

# Codex resolves this provider id against `model_providers.<id>` so a run can
# target any OpenAI-compatible endpoint without editing ~/.codex/config.toml.
_PROVIDER_ID = "lhht"
_DEFAULT_BASE_URL = "https://api.openai.com/v1"


class CodexAdapter(CommandAgentAdapter):
    def __init__(
        self,
        *,
        model: str | None = DEFAULT_CODEX_MODEL,
        api_key: str | None = None,
        base_url: str | None = None,
        workspace_path: str = DEFAULT_WORKSPACE_PATH,
        prompt_dir: str = f"{DEFAULT_TMP_DIR}/prompts",
        mcp_config: str | None = None,
        add_dirs: list[str] | None = None,
        sandbox_mode: str | None = None,
        hidden_paths: tuple[str, ...] = (),
        reasoning_effort: str | None = None,
        role: str | None = None,
    ) -> None:
        effort = normalise_reasoning_effort(reasoning_effort)
        env_overrides: dict[str, str] = {}
        if api_key:
            env_overrides["OPENAI_API_KEY"] = api_key
            env_overrides["CODEX_API_KEY"] = api_key

        # Resolve once when an adapter is built.  A LongHorizon run must use
        # the same authenticated Codex installation as the desktop client when
        # both the standalone PATH CLI and ChatGPT.app are present.
        codex_binary = resolve_codex_binary() or "codex"
        argv = [
            codex_binary,
            "exec",
            "--json",
            "--skip-git-repo-check",
        ]
        # Under its default sandbox `codex exec` cannot touch the filesystem,
        # which would block every CLI subtask. The harness already runs inside an
        # isolated environment, so bypass Codex's own sandbox unless the caller
        # picked an explicit policy.
        if sandbox_mode:
            argv.extend(["--sandbox", sandbox_mode])
        else:
            argv.append("--dangerously-bypass-approvals-and-sandbox")

        for override in _config_overrides(base_url=base_url, api_key=api_key):
            argv.extend(["-c", override])

        # Codex has no `--effort` flag; the reasoning depth is a config value.
        # Passing nothing leaves the user's own ~/.codex/config.toml in charge.
        if effort:
            argv.extend(["-c", f"model_reasoning_effort={json.dumps(effort)}"])

        # MCP support is opt-in and uses Codex's own format: a TOML file holding
        # `[mcp_servers.*]` tables, replayed as `-c mcp_servers.<name>=...`
        # overrides because `--profile` only reads files inside $CODEX_HOME.
        # `codex exec` loads ONLY what arrives this way -- config.toml servers
        # never reach a headless run (verified live) -- so the per-role
        # mcp_allow/mcp_blocked lists simply filter what lhht passes: the
        # operator file's servers plus the scorer tool, ours last so the
        # harness definition wins a name clash.
        mcp_config = mcp_config or os.getenv("LHHT_CODEX_MCP_CONFIG")
        defaults = load_mcp_defaults()
        allow, blocked = effective_mcp_lists(defaults, role)
        operator_tables = mcp_server_tables(mcp_config) if mcp_config else {}
        servers: dict = dict(operator_tables)
        self.semif_mcp_configured = False
        semif_override = _semif_mcp_override(defaults)
        if semif_override is not None:
            name, spec = semif_override
            servers[name] = spec
        if mcp_restricted(allow, blocked):
            servers = {
                name: spec
                for name, spec in servers.items()
                if mcp_admits(allow, blocked, name)
            }
        if semif_override is not None:
            self.semif_mcp_configured = semif_override[0] in servers
        for name, spec in servers.items():
            argv.extend(["-c", f"mcp_servers.{name}={_toml_inline(spec)}"])

        resolved_add_dirs = list(add_dirs or [])
        env_add_dirs = os.getenv("LHHT_CODEX_ADD_DIRS") or os.getenv("LHHT_MCP_ADD_DIRS")
        if env_add_dirs:
            resolved_add_dirs.extend(part for part in env_add_dirs.split(os.pathsep) if part)
        for add_dir in resolved_add_dirs:
            argv.extend(["--add-dir", add_dir])

        if model:
            argv.extend(["--model", model])
        # `-` makes Codex read the prompt from stdin, which is where the harness
        # sends it: long prompts stay off the command line and out of the
        # process table.
        argv.append("-")

        super().__init__(
            argv=argv,
            env=env_overrides,
            prompt_dir=prompt_dir,
            workspace_path=workspace_path,
            visible_output_parser=extract_codex_visible_output,
            hidden_paths=hidden_paths,
        )


def _semif_mcp_overrides() -> list[str]:
    """`-c mcp_servers.*` overrides for the scorer tool, empty when off."""

    try:
        from ..config import load_run_defaults
        from ..semif_mcp import SERVER_NAME, semif_mcp_command

        command = semif_mcp_command(load_run_defaults())
        if command is None:
            return []
        spec = {"command": command[0], "args": command[1]}
        return [f"mcp_servers.{SERVER_NAME}={_toml_inline(spec)}"]
    except Exception:
        return []


def _config_overrides(*, base_url: str | None, api_key: str | None) -> list[str]:
    """Build the `-c key=value` overrides that point Codex at our endpoint."""
    if not base_url and not api_key:
        return []
    provider = {
        "name": "LongHorizon-Harness",
        "base_url": _normalize_base_url(base_url),
        "wire_api": "responses",
    }
    if api_key:
        provider["env_key"] = "OPENAI_API_KEY"
    return [
        f"model_providers.{_PROVIDER_ID}={_toml_inline(provider)}",
        f"model_provider={json.dumps(_PROVIDER_ID)}",
    ]


def _normalize_base_url(base_url: str | None) -> str:
    if not base_url:
        return _DEFAULT_BASE_URL
    trimmed = base_url.rstrip("/")
    # Codex requests `<base_url>/responses`, so the URL must carry the API
    # version segment that Anthropic-style base URLs usually omit.
    return trimmed if trimmed.endswith("/v1") else f"{trimmed}/v1"


def _semif_mcp_override(defaults: dict) -> tuple[str, dict] | None:
    """(name, spec) for the scorer's mcp_servers table, or None when off."""

    from ..semif_mcp import SERVER_NAME, semif_mcp_command

    command = semif_mcp_command(defaults)
    if command is None:
        return None
    return SERVER_NAME, {"command": command[0], "args": command[1]}


def mcp_server_tables(path: str) -> dict:
    """Read `[mcp_servers.*]` tables from a Codex TOML file, name -> spec."""

    try:
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    servers = data.get("mcp_servers") if isinstance(data, dict) else None
    if not isinstance(servers, dict):
        return {}
    return {
        str(name): dict(spec)
        for name, spec in servers.items()
        if isinstance(spec, dict) and spec and str(name).strip()
    }


def mcp_server_overrides(path: str) -> list[str]:
    """Read `[mcp_servers.*]` tables from a Codex TOML file as `-c` overrides."""

    return [
        f"mcp_servers.{name}={_toml_inline(spec)}"
        for name, spec in mcp_server_tables(path).items()
    ]


def _toml_inline(value) -> str:
    """Render a value as inline TOML, which is what `codex -c` parses."""
    if isinstance(value, dict):
        body = ", ".join(f"{key} = {_toml_inline(item)}" for key, item in value.items())
        return "{" + body + "}"
    if isinstance(value, list):
        return "[" + ", ".join(_toml_inline(item) for item in value) + "]"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value))
