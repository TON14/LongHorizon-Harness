from __future__ import annotations

import json
import sys
import uuid
from collections.abc import Sequence
from pathlib import Path

from ..agent_logs import visible_output as extract_visible_output
from ..agent_registry import ZCODE_EFFORT_LEVELS, normalise_reasoning_effort
from ..environment.base import Environment
from ..types import (
    DEFAULT_TMP_DIR,
    DEFAULT_WORKSPACE_PATH,
    DEFAULT_ZCODE_MODEL,
    EpisodeBudget,
    EpisodeResult,
)
from ..utils.agent_cli import resolve_zcode_binary
from .cli_agent import CommandAgentAdapter
from .zcode_provider_config import _DEFAULT_BASE_URL as DEFAULT_PROVIDER_BASE_URL
from .zcode_provider_config import PROVIDER_ID, ensure_provider_config

_READ_ONLY_ROLES = {
    "manager",
    "final_response",
    "gui_auditor",
    "cli_auditor",
    "auditor_format_repair",
}
_WORKSPACE_WRITE_ROLES = {"gui_executor", "cli_executor"}


# The desktop app keeps its configured providers (with their API keys) in its
# own state directory. Reading it is a convenience fallback for operators who
# keep a provider entry there; the supported credential paths are `--api-key`
# and the personal provider config written by `ensure_provider_config`.
_DESKTOP_CONFIG_PATH = Path.home() / ".zcode" / "v2" / "config.json"
_DESKTOP_PROVIDER_IDS = ("builtin:zai-coding-plan", "builtin:zai", PROVIDER_ID)


def _semif_mcp_json(prompt_dir: str) -> str | None:
    """Register the scorer MCP tool for role sessions when configured.

    ZCode's headless app-server does not discover workspace ``.mcp.json``
    (verified empirically); it accepts stdio MCP servers per ``session/create``.
    When ``[run.semif] mcp_tool`` is on with ``mcp_python``/``mcp_script``
    paths, write the server definition once per adapter into the run's prompt
    directory and hand the runner its path. Any gap or error returns None --
    sessions simply start without the tool.
    """
    try:
        from ..config import load_run_defaults

        defaults = load_run_defaults()
        if not defaults.get("semif_mcp_tool"):
            return None
        python_path = defaults.get("semif_mcp_python")
        script_path = defaults.get("semif_mcp_script")
        if not (isinstance(python_path, str) and python_path
                and isinstance(script_path, str) and script_path):
            return None
        servers = [{
            "name": "semif-scorer",
            "command": python_path,
            "args": [script_path],
            "env": [],
            "isolation": "session",
        }]
        path = Path(prompt_dir) / f"mcp_servers_{uuid.uuid4().hex[:8]}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(servers), encoding="utf-8")
        return str(path)
    except Exception:
        return None


def permission_mode_for_role(role: str) -> str:
    if role in _READ_ONLY_ROLES:
        return "plan"
    if role in _WORKSPACE_WRITE_ROLES:
        return "yolo"
    raise ValueError(f"unsupported ZCode role: {role}")


def split_model_ref(model: str) -> tuple[str, str]:
    """Split a harness model id into (ignored provider, model id).

    ZCode 0.16.x builds its model registry from the personal provider config
    where the harness registers every model under one internal provider, so
    only the model id survives; a legacy ``provider/model`` value is accepted
    and its provider part ignored.
    """

    model_id = model.rpartition("/")[2].strip()
    return PROVIDER_ID, model_id


def _desktop_api_key() -> str | None:
    try:
        payload = json.loads(_DESKTOP_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    providers = payload.get("provider") if isinstance(payload, dict) else None
    if not isinstance(providers, dict):
        return None
    for provider_id in _DESKTOP_PROVIDER_IDS:
        entry = providers.get(provider_id)
        if not isinstance(entry, dict):
            continue
        options = entry.get("options")
        if not isinstance(options, dict):
            continue
        api_key = options.get("apiKey")
        if isinstance(api_key, str) and api_key.strip():
            return api_key.strip()
    return None


class ZCodeAdapter(CommandAgentAdapter):
    """Run ZCode headlessly through LongHorizon Harness.

    ZCode 0.16.x no longer resolves a model for ``-p``-style headless runs,
    so episodes go through the ZCode Protocol app-server instead: the runner
    opens a session with the model and the role's reasoning level pinned and
    sends the prompt over the protocol. Roles still map onto permission
    modes: executors run ``yolo`` inside the workspace the harness scopes,
    while the manager and auditors run ``plan``.

    The model itself is registered in ZCode's personal provider config
    (``~/.zcode/v2/provider_config.json``) under one internal provider,
    authenticated with the operator's Z.ai API key.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_ZCODE_MODEL,
        api_key: str | None = None,
        base_url: str | None = None,
        workspace_path: str = DEFAULT_WORKSPACE_PATH,
        prompt_dir: str = f"{DEFAULT_TMP_DIR}/prompts",
        role: str = "cli_executor",
        add_dirs: Sequence[str] = (),
        hidden_paths: Sequence[str] = (),
        reasoning_effort: str | None = None,
    ) -> None:
        normalized_model = model.strip()
        if not normalized_model:
            raise ValueError("ZCode model must not be empty")
        if "\x00" in normalized_model or len(normalized_model) > 256:
            raise ValueError("ZCode model contains invalid characters or is too long")
        _, model_id = split_model_ref(normalized_model)
        # Rejected rather than ignored: ZCode silently runs at its default
        # depth when the level does not match the model, so a typo must fail
        # here instead of degrading the run without a trace.
        normalized_effort = normalise_reasoning_effort(reasoning_effort, agent_id="zcode")
        if normalized_effort and normalized_effort not in ZCODE_EFFORT_LEVELS:
            raise ValueError(
                "ZCode reasoning effort must be one of: " + ", ".join(ZCODE_EFFORT_LEVELS)
            )
        if add_dirs:
            raise ValueError("ZCode integration does not support additional directories")

        permission_mode = permission_mode_for_role(role)
        zcode_binary = resolve_zcode_binary() or "zcode"
        endpoint = (base_url or "").strip().rstrip("/") or DEFAULT_PROVIDER_BASE_URL
        resolved_key = api_key or _desktop_api_key()
        if not resolved_key:
            raise ValueError(
                "ZCode integration requires a Z.ai API key: pass --api-key or set it in "
                f"{_DESKTOP_CONFIG_PATH}"
            )
        config_path = ensure_provider_config(resolved_key, model_id, base_url=endpoint)

        command = [
            sys.executable,
            "-m",
            "lhht.adapters.zcode_runner",
            "--binary",
            zcode_binary,
            "--prompt",
            "{prompt_path}",
            "--model",
            model_id,
            "--mode",
            permission_mode,
            "--workspace",
            workspace_path,
        ]
        if normalized_effort:
            command += ["--thought-level", normalized_effort]
        mcp_json_path = _semif_mcp_json(prompt_dir)
        if mcp_json_path:
            command += ["--mcp-json", mcp_json_path]

        super().__init__(
            argv=command,
            env={},
            workspace_path=workspace_path,
            prompt_dir=prompt_dir,
            visible_output_parser=extract_visible_output,
            hidden_paths=tuple(hidden_paths),
        )
        self.model = model_id
        self.role = role
        self.permission_mode = permission_mode
        self.reasoning_effort = normalized_effort
        self.provider_config_path = str(config_path)

    async def run_episode(
        self,
        prompt: str,
        env: Environment,
        budget: EpisodeBudget,
        live_trajectory_path: str | None = None,
    ) -> EpisodeResult:
        result = await super().run_episode(
            prompt,
            env,
            budget,
            live_trajectory_path=live_trajectory_path,
        )
        result.metadata.update(
            {
                "zcode_role": self.role,
                "zcode_model": self.model,
                "zcode_mode": self.permission_mode,
                "zcode_reasoning_effort": self.reasoning_effort,
                "zcode_provider_config": self.provider_config_path,
            }
        )
        return result
