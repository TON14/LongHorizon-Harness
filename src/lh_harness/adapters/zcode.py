from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
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

# The reasoning dial ZCode exposes for GLM-5.x models. The CLI keeps it in a
# per-user `local_setting` row of its session database (there is no headless
# flag), so the adapter seeds an isolated copy of that database instead of
# touching the operator's own ZCode state. The list itself is owned by the
# agent registry, which also offers it to the workbench.


# The desktop app keeps its configured providers (with their API keys) in its
# own state directory. Reading it is a convenience fallback for operators who
# are already logged in on the desktop -- the supported credential paths are
# `--api-key` and exporting `ZCODE_API_KEY`/`ZAI_API_KEY`, which the child
# inherits untouched.
_DESKTOP_CONFIG_PATH = Path.home() / ".zcode" / "v2" / "config.json"
_DESKTOP_PROVIDER_IDS = ("builtin:zai-coding-plan", "builtin:zai")

_LOCAL_SETTING_DDL = """
CREATE TABLE IF NOT EXISTS local_setting (
        scope text not null,
        scope_id text not null,
        namespace text not null,
        key text not null,
        value text not null,
        schema_version integer not null,
        time_created integer not null,
        time_updated integer not null,
        primary key(scope, scope_id, namespace, key)
      )
"""


def permission_mode_for_role(role: str) -> str:
    if role in _READ_ONLY_ROLES:
        return "plan"
    if role in _WORKSPACE_WRITE_ROLES:
        return "yolo"
    raise ValueError(f"unsupported ZCode role: {role}")


def split_model_ref(model: str) -> tuple[str, str]:
    """Split a harness model id into ZCode's (provider, model) pair."""

    if "/" in model:
        provider, _, model_id = model.partition("/")
        return provider.strip() or _ZAI_PROVIDER, model_id.strip()
    return _ZAI_PROVIDER, model


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


def _write_project_config(
    workspace_path: str,
    *,
    provider_id: str,
    model_id: str,
    api_key: str,
    base_url: str,
) -> str:
    """Place ZCode's project config so the provider reaches the request.

    A provider defined only through ``ZCODE_*`` environment variables never
    picks up the model catalogue's reasoning capability, so the effort dial
    silently does nothing. The same provider declared in a ``.zcode/
    config.json`` next to the workspace does, and the file also carries the
    endpoint and key. An existing file is never overwritten -- it may be the
    operator's own -- and the adapter reports that so the run record says
    which path a role actually took.
    """

    config_path = Path(workspace_path) / ".zcode" / "config.json"
    if config_path.exists():
        return "pre-existing"
    payload = {
        "model": {"main": f"{provider_id}/{model_id}"},
        "provider": {
            provider_id: {
                "kind": "anthropic",
                "options": {"apiKey": api_key, "baseURL": base_url},
                "models": {model_id: {}},
            }
        },
    }
    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = config_path.with_name(f"{config_path.name}.lh-harness.tmp")
        with open(temp_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        os.chmod(temp_path, 0o600)
        os.replace(temp_path, config_path)
    except OSError as exc:
        raise ValueError(f"could not write ZCode project config: {exc}") from exc
    return "created"


def _seed_session_db(db_path: Path, level: str) -> None:
    """Prepare the isolated ZCode session store with the effort level.

    ZCode migrates a fresh database on first use, and a pre-existing
    ``local_setting`` table survives that migration untouched -- which is what
    makes per-run isolation possible without writing to ``~/.zcode``.
    """

    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(db_path)
        try:
            connection.execute(_LOCAL_SETTING_DDL)
            if level:
                now = int(time.time() * 1000)
                connection.execute(
                    "INSERT INTO local_setting VALUES "
                    "('user','default','model','reasoningLevel',?,1,?,?) "
                    "ON CONFLICT(scope, scope_id, namespace, key) DO UPDATE SET "
                    "value=excluded.value, time_updated=excluded.time_updated",
                    (json.dumps({"level": level}), now, now),
                )
            connection.commit()
        finally:
            connection.close()
    except (OSError, sqlite3.Error) as exc:
        raise ValueError(f"could not prepare the ZCode session store: {exc}") from exc


class ZCodeAdapter(CommandAgentAdapter):
    """Run ZCode headlessly through LongHorizon Harness.

    ZCode's headless mode is one ``-p <task>`` invocation per episode; the
    final answer comes back as a JSON object which ``zcode_runner`` re-emits
    as the same ``zcode.result`` record shape the log parsers normalise.
    Roles map onto ZCode's permission modes: executors run ``yolo`` inside the
    workspace the harness already scopes, while the manager and auditors run
    ``plan`` so they can read and investigate without editing.

    The reasoning effort rides in an isolated copy of ZCode's session
    database (``ZCODE_SESSION_DB_PATH``); it only reaches the provider when
    the model and its provider are declared in a ``.zcode/config.json``
    inside the workspace, so the adapter writes one when it can and keeps the
    environment-variable path for everything else.
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
        provider_id, model_id = split_model_ref(normalized_model)
        # Rejected rather than ignored: ZCode silently runs at its default
        # depth when the level does not match the model, so a typo must fail
        # here instead of degrading the run without a trace.
        normalized_effort = normalise_reasoning_effort(reasoning_effort, agent_id="zcode")
        if (
            normalized_effort
            and provider_id == _ZAI_PROVIDER
            and normalized_effort not in ZCODE_EFFORT_LEVELS
        ):
            raise ValueError(
                "ZCode reasoning effort for zai models must be one of: "
                + ", ".join(ZCODE_EFFORT_LEVELS)
            )
        if add_dirs:
            raise ValueError("ZCode integration does not support additional directories")

        permission_mode = permission_mode_for_role(role)
        zcode_binary = resolve_zcode_binary() or "zcode"
        endpoint = (base_url or "").strip().rstrip("/") or _DEFAULT_ZCODE_BASE_URL
        resolved_key = api_key or _desktop_api_key()

        environment: dict[str, str] = {}
        # With a key in hand the provider is declared in a project config,
        # which is the only path that also applies the effort dial. Without
        # one, the env-configured provider keeps working as before.
        if resolved_key:
            project_config = _write_project_config(
                workspace_path,
                provider_id=provider_id,
                model_id=model_id,
                api_key=resolved_key,
                base_url=endpoint,
            )
        else:
            project_config = "env-configured"
            environment["ZCODE_MODEL"] = f"{provider_id}/{model_id}"
            environment["ZCODE_BASE_URL"] = endpoint

        # The session store is always isolated: concurrent runs then never
        # fight over the operator's ~/.zcode, and the effort row is the only
        # state a run writes.
        session_db = (
            Path(prompt_dir.rstrip("/")).parent / "zcode-db" / "session.db"
        )
        _seed_session_db(session_db, normalized_effort)
        environment["ZCODE_SESSION_DB_PATH"] = str(session_db)

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
        scoped_hidden_paths = tuple(hidden_paths)
        if project_config == "created":
            # The harness wrote this one; the agents must not treat it as
            # task content or leak the key inside it.
            scoped_hidden_paths += (".zcode",)
        super().__init__(
            argv=command,
            env=environment,
            workspace_path=workspace_path,
            prompt_dir=prompt_dir,
            visible_output_parser=extract_visible_output,
            hidden_paths=scoped_hidden_paths,
        )
        self.model = normalized_model
        self.role = role
        self.permission_mode = permission_mode
        self.reasoning_effort = normalized_effort
        self.project_config = project_config
        self.session_db = str(session_db)

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
                "zcode_project_config": self.project_config,
            }
        )
        return result
