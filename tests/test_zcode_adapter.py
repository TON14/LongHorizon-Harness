from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from lh_harness import agent_logs
from lh_harness.adapters import zcode as zcode_adapter_module
from lh_harness.adapters.zcode import ZCodeAdapter, permission_mode_for_role
from lh_harness.adapters.zcode_runner import run
from lh_harness.environment.local import LocalEnvironment
from lh_harness.types import EpisodeBudget
from lh_harness.utils.agent_cli import resolve_zcode_binary
from lh_harness.webapi import server as web_server

from .fake_cli import fake_cli as _executable


@pytest.fixture(autouse=True)
def _no_desktop_credentials(monkeypatch):
    """Keep tests off the operator's real desktop ZCode login.

    The adapter falls back to ~/.zcode/v2/config.json when no key is given;
    a machine with the desktop app installed must not change what the tests
    observe.
    """
    monkeypatch.setattr(zcode_adapter_module, "_desktop_api_key", lambda: None)


def test_zcode_binary_environment_override() -> None:
    assert (
        resolve_zcode_binary(
            environ={"LH_HARNESS_ZCODE_BINARY": "/custom/ZCode/zcode.cjs"},
            platform_name="linux",
        )
        == "/custom/ZCode/zcode.cjs"
    )


def test_zcode_reasoning_is_declared_per_model() -> None:
    from lh_harness.agent_registry import agent_spec, supports_reasoning_effort

    spec = agent_spec("zcode")
    assert supports_reasoning_effort("zcode") is True
    assert spec.reasoning is not None
    assert spec.reasoning.transport == "session_db"
    assert spec.reasoning.declared_choices == ("low", "high", "max")


def test_zcode_permission_modes_are_role_scoped() -> None:
    assert permission_mode_for_role("manager") == "plan"
    assert permission_mode_for_role("cli_auditor") == "plan"
    assert permission_mode_for_role("cli_executor") == "yolo"


def test_zcode_adapter_builds_runner_command_and_env(
    monkeypatch,
    tmp_path: Path,
) -> None:
    binary = str(tmp_path / "ZCode" / "zcode.cjs")
    monkeypatch.setattr(zcode_adapter_module, "resolve_zcode_binary", lambda: binary)
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    adapter = ZCodeAdapter(
        model="glm-5.3",
        api_key="sk-test",
        workspace_path=str(workspace),
        prompt_dir=str(tmp_path / "run with spaces" / "prompts"),
        role="manager",
    )

    argv = adapter.argv
    assert "lh_harness.adapters.zcode_runner" in argv
    assert argv[argv.index("--binary") + 1] == binary
    assert argv[argv.index("--mode") + 1] == "plan"
    assert adapter.permission_mode == "plan"
    # With a key in hand the provider rides in the project config, so the
    # model must NOT also come from ZCODE_MODEL: the env-configured provider
    # would win and silently drop the effort dial.
    assert "ZCODE_MODEL" not in adapter.env
    assert adapter.env["ZCODE_SESSION_DB_PATH"].endswith("zcode-db/session.db")
    assert adapter.project_config == "created"
    config = json.loads((workspace / ".zcode" / "config.json").read_text(encoding="utf-8"))
    assert config["model"]["main"] == "zai/glm-5.3"
    assert config["provider"]["zai"]["options"]["apiKey"] == "sk-test"
    assert config["provider"]["zai"]["options"]["baseURL"] == "https://api.z.ai/api/anthropic"
    assert config["provider"]["zai"]["models"] == {"glm-5.3": {}}


def test_zcode_project_config_permissions_and_reuse(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(zcode_adapter_module, "resolve_zcode_binary", lambda: "zcode")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    prompt_dir = tmp_path / "prompts"

    first = ZCodeAdapter(
        api_key="sk-test",
        workspace_path=str(workspace),
        prompt_dir=str(prompt_dir),
    )
    config_path = workspace / ".zcode" / "config.json"
    assert oct(config_path.stat().st_mode & 0o777) == "0o600"
    assert ".zcode" in first.hidden_paths

    config_path.write_text('{"model": {"main": "custom/model"}}', encoding="utf-8")
    second = ZCodeAdapter(
        api_key="sk-test",
        workspace_path=str(workspace),
        prompt_dir=str(prompt_dir),
    )
    # The operator's own file is never overwritten.
    assert second.project_config == "pre-existing"
    assert "custom/model" in config_path.read_text(encoding="utf-8")
    assert ".zcode" not in second.hidden_paths


def test_zcode_adapter_falls_back_to_env_config_without_a_key(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(zcode_adapter_module, "resolve_zcode_binary", lambda: "zcode")
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    adapter = ZCodeAdapter(
        workspace_path=str(workspace),
        prompt_dir=str(tmp_path / "prompts"),
    )

    assert adapter.project_config == "env-configured"
    assert not (workspace / ".zcode").exists()
    assert adapter.env["ZCODE_MODEL"] == "zai/glm-5.3"
    assert adapter.env["ZCODE_BASE_URL"] == "https://api.z.ai/api/anthropic"
    assert "ZCODE_SESSION_DB_PATH" in adapter.env


def test_zcode_adapter_keeps_qualified_model_and_custom_endpoint(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(zcode_adapter_module, "resolve_zcode_binary", lambda: "zcode")

    adapter = ZCodeAdapter(
        model="other/glm-5.3-flash",
        base_url="https://proxy.example.com/anthropic/",
        workspace_path=str(tmp_path / "ws"),
        prompt_dir=str(tmp_path / "prompts"),
        role="cli_executor",
    )

    assert adapter.env["ZCODE_MODEL"] == "other/glm-5.3-flash"
    assert adapter.env["ZCODE_BASE_URL"] == "https://proxy.example.com/anthropic"
    assert adapter.permission_mode == "yolo"


def test_zcode_reasoning_effort_seeds_the_isolated_session_db(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(zcode_adapter_module, "resolve_zcode_binary", lambda: "zcode")

    adapter = ZCodeAdapter(
        api_key="sk-test",
        workspace_path=str(tmp_path / "ws"),
        prompt_dir=str(tmp_path / "prompts"),
        reasoning_effort="high",
    )

    assert adapter.reasoning_effort == "high"
    import sqlite3

    connection = sqlite3.connect(adapter.env["ZCODE_SESSION_DB_PATH"])
    try:
        row = connection.execute(
            "select value from local_setting where namespace='model' and key='reasoningLevel'"
        ).fetchone()
    finally:
        connection.close()
    assert json.loads(row[0]) == {"level": "high"}

    # Re-seeding the same store updates the level instead of failing.
    ZCodeAdapter(
        api_key="sk-test",
        workspace_path=str(tmp_path / "ws"),
        prompt_dir=str(tmp_path / "prompts"),
        reasoning_effort="low",
    )
    connection = sqlite3.connect(adapter.env["ZCODE_SESSION_DB_PATH"])
    try:
        row = connection.execute(
            "select value from local_setting where namespace='model' and key='reasoningLevel'"
        ).fetchone()
    finally:
        connection.close()
    assert json.loads(row[0]) == {"level": "low"}


def test_zcode_rejects_unknown_zai_effort_but_keeps_custom_providers(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(zcode_adapter_module, "resolve_zcode_binary", lambda: "zcode")

    with pytest.raises(ValueError, match="low, high, max"):
        ZCodeAdapter(
            workspace_path=str(tmp_path / "ws"),
            prompt_dir=str(tmp_path / "prompts"),
            reasoning_effort="medium",
        )
    # A custom provider owns its own level names; the harness passes them on.
    adapter = ZCodeAdapter(
        model="other/glm-x",
        workspace_path=str(tmp_path / "ws"),
        prompt_dir=str(tmp_path / "prompts"),
        reasoning_effort="medium",
    )
    assert adapter.reasoning_effort == "medium"


def test_zcode_runner_parses_the_json_final_answer(
    tmp_path: Path,
    capsys,
) -> None:
    binary = _executable(
        tmp_path / "bin" / "zcode",
        "import json\n"
        'print(json.dumps({"sessionId": "sess_1", "response": "done by zcode", '
        '"usage": {"inputTokens": 5}}))\n',
    )
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("fix the project", encoding="utf-8")

    assert run(binary, prompt_path, "glm-5.3", mode="yolo") == 0

    record = json.loads(capsys.readouterr().out)
    assert record["type"] == "zcode.result"
    assert record["is_error"] is False
    assert record["text"] == "done by zcode"
    assert record["session_id"] == "sess_1"
    assert record["usage"] == {"inputTokens": 5}


def test_zcode_runner_parses_a_pretty_printed_json_answer(
    tmp_path: Path,
    capsys,
) -> None:
    # The real CLI pretty-prints `--json` across many lines; the reply must
    # still come from `response`, not the raw document.
    payload = (
        "{\n"
        '  "sessionId": "sess_p",\n'
        '  "response": "Next: cli\\nTask: do the work",\n'
        '  "usage": {"inputTokens": 7}\n'
        "}\n"
    )
    binary = _executable(
        tmp_path / "bin" / "zcode",
        "import sys\nsys.stdout.write(" + json.dumps(payload) + ")\n",
    )
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("plan", encoding="utf-8")

    assert run(binary, prompt_path, "glm-5.3", mode="plan") == 0

    record = json.loads(capsys.readouterr().out)
    assert record["text"] == "Next: cli\nTask: do the work"
    assert record["session_id"] == "sess_p"
    assert record["usage"] == {"inputTokens": 7}


def test_zcode_runner_passes_prompt_and_mode_to_the_cli(
    tmp_path: Path,
    capsys,
) -> None:
    binary = _executable(
        tmp_path / "bin" / "zcode",
        "print(' '.join(sys.argv[1:]))\n",
    )
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("fix the project", encoding="utf-8")

    assert run(binary, prompt_path, "glm-5.3", mode="plan") == 0

    record = json.loads(capsys.readouterr().out)
    text = record["text"]
    assert text.startswith("--json")
    assert "--mode plan" in text
    assert text.endswith("fix the project")


def test_zcode_runner_preserves_failure_and_stderr(tmp_path: Path, capfd) -> None:
    binary = _executable(
        tmp_path / "bin" / "zcode",
        "sys.stderr.write('model config is missing\\n')\nsys.exit(9)\n",
    )
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("task", encoding="utf-8")

    assert run(binary, prompt_path, "glm-5.3") == 9

    record = json.loads(capfd.readouterr().out)
    assert record["is_error"] is True
    assert record["exit_code"] == 9
    # The CLI's stderr rides inside the record rather than the runner's own
    # stderr, so the episode log carries the provider's reason.
    assert "model config is missing" in record["error"]


def test_zcode_jsonl_views() -> None:
    raw = json.dumps(
        {
            "type": "zcode.result",
            "text": "implemented and verified",
            "is_error": False,
            "exit_code": 0,
        }
    )

    assert agent_logs.detect_format(raw) == agent_logs.ZCODE_RESULT_JSONL
    assert agent_logs.visible_output(raw) == "implemented and verified"
    assert agent_logs.assistant_texts(raw) == ["implemented and verified"]
    steps = agent_logs.parse_trajectory(raw)
    assert steps[0]["kind"] == "result"
    assert steps[0]["text"] == "implemented and verified"
    assert steps[0]["is_error"] is False


def test_zcode_adapter_runs_end_to_end_with_fake_zcode(
    monkeypatch,
    tmp_path: Path,
) -> None:
    binary = _executable(
        tmp_path / "bin" / "zcode",
        "import sys, json\n"
        'print(json.dumps({"response": "done by zcode"}))\n',
    )
    monkeypatch.setattr(zcode_adapter_module, "resolve_zcode_binary", lambda: binary)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    prompt_dir = tmp_path / "run state" / "prompts"
    adapter = ZCodeAdapter(
        workspace_path=str(workspace),
        prompt_dir=str(prompt_dir),
        role="cli_executor",
    )

    result = asyncio.run(
        adapter.run_episode(
            "complete the task",
            LocalEnvironment(tmp_dir=str(tmp_path / "tmp")),
            EpisodeBudget(max_duration_seconds=10),
        )
    )

    assert result.status == "done"
    assert result.metadata["assistant_visible_output"] == "done by zcode"
    assert result.metadata["runtime_signals"] == []
    assert json.loads(result.actions_log)["type"] == "zcode.result"
    assert result.metadata["zcode_mode"] == "yolo"


def test_web_meta_exposes_zcode_backend_and_default_model(
    monkeypatch,
    tmp_path: Path,
) -> None:
    # Availability is proven by running `--version`, so the stub answers it.
    binary = _executable(tmp_path / "bin" / "zcode", 'print("zcode 0.16.5")\n')
    monkeypatch.setattr(web_server, "resolve_zcode_binary", lambda: binary)

    client = TestClient(web_server.create_app(runs_root=tmp_path / "runs"))
    meta = client.get("/api/meta").json()
    agent = next(item for item in meta["agents"] if item["id"] == "zcode")

    assert agent["label"] == "ZCode"
    assert agent["available"] is True
    assert agent["availability"] == "usable"
    assert agent["version"] == "0.16.5"
    assert agent["binary"] == binary
    assert agent["default_model"] == "glm-5.3"
    assert meta["models"]["zcode"][0]["id"] == "glm-5.3"
    assert meta["models"]["zcode"][1]["id"] == "glm-5.3-flash"
    # The workbench offers the GLM-5.x reasoning dial for ZCode runs.
    assert agent["reasoning"]["supported"] is True
    assert [choice["id"] for choice in agent["reasoning"]["choices"]] == [
        "low",
        "high",
        "max",
    ]
