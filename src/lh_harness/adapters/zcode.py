from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path

from ..agent_logs import visible_output as extract_visible_output
from ..agent_registry import normalise_reasoning_effort
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

_READ_ONLY_ROLES = {
    "manager",
    "final_response",
    "gui_auditor",
    "cli_auditor",
    "auditor_format_repair",
}
_WORKSPACE_WRITE_ROLES = {"gui_executor", "cli_executor"}

# ZCode headless selects the model through `ZCODE_MODEL` as `provider/model`.
# The `zai` provider speaks the Anthropic wire protocol, and without an
# explicit `ZCODE_BASE_URL` the CLI falls back to api.anthropic.com, so the
# endpoint is always pinned (explicitly overridable) rather than inherited.
_ZAI_PROVIDER = "zai"
_DEFAULT_ZCODE_BASE_URL = "https://api.z.ai/api/anthropic"

# The desktop app keeps its configured providers (with their API keys) in its
# own state directory. Reading it is a convenience fallback for operators who
# are already logged in on the desktop -- the supported credential paths are
# `--api-key` and exporting `ZCODE_API_KEY`/`ZAI_API_KEY`, which the child
# inherits untouched.
_DESKTOP_CONFIG_PATH = Path.home() / ".zcode" / "v2" / "config.json"
_DESKTOP_PROVIDER_IDS = ("builtin:zai-coding-plan", "builtin:zai")


def permission_mode_for_role(role: str) -> str:
    if role in _READ_ONLY_ROLES:
        return "plan"
    if role in _WORKSPACE_WRITE_ROLES:
        return "yolo"
    raise ValueError(f"unsupported ZCode role: {role}")


def _qualified_model(model: str) -> str:
    if "/" in model:
        return model
    return f"{_ZAI_PROVIDER}/{model}"


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

    ZCode's headless mode is one ``-p <task>`` invocation per episode; the
    final answer comes back as a JSON object which ``zcode_runner`` re-emits
    as the same ``zcode.result`` record shape the log parsers normalise.
    Roles map onto ZCode's permission modes: executors run ``yolo`` inside the
    workspace the harness already scopes, while the manager and auditors run
    ``plan`` so they can read and investigate without editing.
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
        # Rejected rather than ignored: silently dropping it would make the
        # workbench claim a reasoning depth the run never applied.
        normalise_reasoning_effort(reasoning_effort, agent_id="zcode")
        if add_dirs:
            raise ValueError("ZCode integration does not support additional directories")

        permission_mode = permission_mode_for_role(role)
        zcode_binary = resolve_zcode_binary() or "zcode"

        environment = {
            "ZCODE_MODEL": _qualified_model(normalized_model),
            "ZCODE_BASE_URL": (base_url or "").strip().rstrip("/") or _DEFAULT_ZCODE_BASE_URL,
        }
        resolved_key = api_key or _desktop_api_key()
        if resolved_key:
            environment["ZCODE_API_KEY"] = resolved_key

        command = [
            sys.executable,
            "-m",
            "lh_harness.adapters.zcode_runner",
            "--binary",
            zcode_binary,
            "--prompt",
            "{prompt_path}",
            "--model",
            normalized_model,
            "--mode",
            permission_mode,
        ]
        super().__init__(
            argv=command,
            env=environment,
            workspace_path=workspace_path,
            prompt_dir=prompt_dir,
            visible_output_parser=extract_visible_output,
            hidden_paths=hidden_paths,
        )
        self.model = normalized_model
        self.role = role
        self.permission_mode = permission_mode

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
            }
        )
        return result
