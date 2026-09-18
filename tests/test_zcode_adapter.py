from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from lhht import agent_logs
from lhht.adapters import zcode as zcode_adapter_module
from lhht.adapters.zcode import ZCodeAdapter, permission_mode_for_role
from lhht.adapters.zcode_provider_config import (
    PROVIDER_ID,
    ensure_provider_config,
)
from lhht.adapters.zcode_protocol import _Client, run_episode
from lhht.adapters.zcode_runner import run
from lhht.utils.agent_cli import resolve_zcode_binary, zcode_spawn_command


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
            environ={"LHHT_ZCODE_BINARY": "/custom/ZCode/zcode.cjs"},
            platform_name="linux",
        )
        == "/custom/ZCode/zcode.cjs"
    )


def test_zcode_reasoning_is_declared_per_model() -> None:
    from lhht.agent_registry import agent_spec, supports_reasoning_effort

    spec = agent_spec("zcode")
    assert supports_reasoning_effort("zcode") is True
    assert spec.reasoning is not None
    assert spec.reasoning.transport == "protocol"
    assert spec.reasoning.declared_choices == ("low", "high", "max")


def test_zcode_permission_modes_are_role_scoped() -> None:
    assert permission_mode_for_role("manager") == "plan"
    assert permission_mode_for_role("cli_auditor") == "plan"
    assert permission_mode_for_role("cli_executor") == "yolo"


def test_zcode_spawn_command_wraps_node_bundles_only() -> None:
    assert zcode_spawn_command("node", ["x"]) == ["node", "x"]
    wrapped = zcode_spawn_command("/opt/ZCode/resources/glm/zcode.cjs", ["--version"])
    assert wrapped[0].lower().endswith(("node", "node.exe"))
    assert wrapped[1] == "/opt/ZCode/resources/glm/zcode.cjs"
    assert zcode_spawn_command("C:\\bin\\zcode.exe", ["--version"]) == [
        "C:\\bin\\zcode.exe",
        "--version",
    ]


def test_zcode_adapter_builds_runner_command(monkeypatch, tmp_path: Path) -> None:
    binary = str(tmp_path / "ZCode" / "zcode.cjs")
    monkeypatch.setattr(zcode_adapter_module, "resolve_zcode_binary", lambda: binary)
    seen: dict = {}

    def fake_ensure(api_key, model_id, *, base_url=""):
        seen.update(api_key=api_key, model_id=model_id, base_url=base_url)
        return tmp_path / "provider_config.json"

    monkeypatch.setattr(zcode_adapter_module, "ensure_provider_config", fake_ensure)
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    adapter = ZCodeAdapter(
        model="zai/glm-5.3-flash",
        api_key="sk-test",
        workspace_path=str(workspace),
        prompt_dir=str(tmp_path / "run with spaces" / "prompts"),
        role="manager",
        reasoning_effort="low",
    )

    argv = adapter.argv
    assert "lhht.adapters.zcode_runner" in argv
    assert argv[argv.index("--binary") + 1] == binary
    assert argv[argv.index("--model") + 1] == "glm-5.3-flash"
    assert argv[argv.index("--mode") + 1] == "plan"
    assert argv[argv.index("--thought-level") + 1] == "low"
    assert argv[argv.index("--workspace") + 1] == str(workspace)
    assert adapter.reasoning_effort == "low"
    assert seen == {
        "api_key": "sk-test",
        "model_id": "glm-5.3-flash",
        "base_url": "https://api.z.ai/api/anthropic",
    }


def test_zcode_adapter_requires_a_key(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(zcode_adapter_module, "resolve_zcode_binary", lambda: "zcode")
    with pytest.raises(ValueError, match="API key"):
        ZCodeAdapter(model="glm-5.3-flash", workspace_path=str(tmp_path))


def test_zcode_adapter_rejects_unknown_effort(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="reasoning effort"):
        ZCodeAdapter(
            model="glm-5.3-flash",
            api_key="sk-test",
            workspace_path=str(tmp_path),
            reasoning_effort="ultra",
        )


def test_ensure_provider_config_merges_and_preserves(tmp_path: Path) -> None:
    config_path = tmp_path / "provider_config.json"
    config_path.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "config": {
                    "providerConfigRules": {
                        "providerRules": [
                            {"providerId": "user-own", "config": {"group": "standard-personal"}}
                        ]
                    },
                    "modelConfigRules": {
                        "providerModelRules": [],
                        "manualProviderModelRules": [
                            {"providerId": "user-own", "modelId": "m1", "config": {"enabled": True}}
                        ],
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    ensure_provider_config("key-1", "glm-5.3-flash", path=config_path)
    first = json.loads(config_path.read_text(encoding="utf-8"))
    rules = first["config"]["providerConfigRules"]["providerRules"]
    harness_rules = [r for r in rules if r["providerId"] == PROVIDER_ID]
    user_rules = [r for r in rules if r["providerId"] == "user-own"]
    assert len(harness_rules) == 1 and len(user_rules) == 1
    rule = harness_rules[0]
    assert rule["config"]["group"] == "standard-personal"
    assert "builtinModelIds" not in rule["config"]
    assert rule["config"]["personalModelIds"] == ["glm-5.3-flash"]
    assert rule["config"]["access"]["apiKey"] == "key-1"

    manual = first["config"]["modelConfigRules"]["manualProviderModelRules"]
    assert [(m["providerId"], m["modelId"]) for m in manual] == [
        ("user-own", "m1"),
        (PROVIDER_ID, "glm-5.3-flash"),
    ]
    spec = manual[-1]["config"]["optionSpecs"]["reasoningLevel"]
    assert spec["values"] == ["low", "high", "max"]
    # The map is a JSON *string* of per-level patches, per the runtime schema.
    assert isinstance(spec["map"], str)
    assert json.loads(spec["map"])["low"] == {}

    # Idempotent second write, with a key rotation: same single rule, new key.
    ensure_provider_config("key-2", "glm-5.3-flash", path=config_path)
    second = json.loads(config_path.read_text(encoding="utf-8"))
    rules2 = [
        r
        for r in second["config"]["providerConfigRules"]["providerRules"]
        if r["providerId"] == PROVIDER_ID
    ]
    assert len(rules2) == 1
    assert rules2[0]["config"]["access"]["apiKey"] == "key-2"
    assert len(second["config"]["providerConfigRules"]["providerRules"]) == 2


def test_ensure_provider_config_rejects_empty_model(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        ensure_provider_config("key", "", path=tmp_path / "p.json")


def _fake_app_server_script() -> str:
    """A tiny ZCode Protocol app-server lookalike for runner tests."""
    return r"""
import json, sys

def send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()

created = {"result": {"protocol": {"name": "ZCode Protocol", "version": 1},
                      "session": {"sessionId": "sess_fake", "mode": "yolo"}}}
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    msg = json.loads(line)
    method = msg.get("method")
    if method == "session/requestRuntimePreferences":
        send({"id": msg["id"], "result": {"nativeSearchEnhancementsEnabled": False}})
    elif method == "session/create":
        send({"id": msg["id"], **created})
        send({"method": "state.updated", "params": {"status": "running"}})
    elif method == "session/send":
        send({"id": msg["id"], "result": {"accepted": True, "sessionId": msg["params"]["sessionId"]}})
    elif method == "session/messages":
        send({"id": msg["id"], "result": {"messages": [{
            "info": {"role": "assistant", "finish": "stop", "tokens": {"output": 3}},
            "parts": [{"type": "text", "text": "OK"}],
        }]}})
    elif "id" in msg:
        send({"id": msg["id"], "result": {}})
"""
def test_zcode_runner_runs_an_episode_through_the_protocol(tmp_path: Path, capsys) -> None:
    script = tmp_path / "fake_app_server.py"
    script.write_text(_fake_app_server_script(), encoding="utf-8")
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("Reply with exactly: OK", encoding="utf-8")

    exit_code = run(
        sys.executable,
        prompt_path,
        "glm-5.3-flash",
        mode="yolo",
        thought_level="high",
        workspace=str(tmp_path),
        server_args=["-u", str(script)],
    )

    assert exit_code == 0
    record = json.loads(capsys.readouterr().out)
    assert record["type"] == "zcode.result"
    assert record["is_error"] is False
    assert record["text"] == "OK"
    assert record["session_id"] == "sess_fake"


def test_zcode_runner_surfaces_protocol_failures(tmp_path: Path, capsys) -> None:
    script = tmp_path / "failing_app_server.py"
    script.write_text(
        "\n".join(
            [
                "import json, sys",
                "def send(o):",
                "    sys.stdout.write(json.dumps(o) + chr(10)); sys.stdout.flush()",
                "for line in sys.stdin:",
                "    msg = json.loads(line)",
                "    if msg.get('method') == 'session/create':",
                "        send({'id': msg['id'], 'error': {'code': -32000, 'message': 'boom'}})",
            ]
        ),
        encoding="utf-8",
    )
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("task", encoding="utf-8")

    exit_code = run(
        sys.executable,
        prompt_path,
        "glm-5.3-flash",
        mode="plan",
        thought_level="low",
        workspace=str(tmp_path),
        server_args=["-u", str(script)],
    )

    assert exit_code == 1
    record = json.loads(capsys.readouterr().out)
    assert record["is_error"] is True
    assert "boom" in record["error"]


def test_zcode_protocol_client_answers_client_bound_requests(tmp_path: Path) -> None:
    """The server blocks on client-bound requests; the client must answer.

    session/requestRuntimePreferences gates every session/create, so a
    client that stays silent deadlocks the harness on the very first
    episode. The fake server only completes the create after it has seen
    the answer to its question.
    """
    script = tmp_path / "asking_app_server.py"
    script.write_text(
        "\n".join(
            [
                "import json, sys",
                "def send(o):",
                "    sys.stdout.write(json.dumps(o) + chr(10)); sys.stdout.flush()",
                "for line in sys.stdin:",
                "    msg = json.loads(line)",
                "    if msg.get('method') == 'session/requestRuntimePreferences':",
                "        send({'id': msg['id'], 'result': {'nativeSearchEnhancementsEnabled': True}})",
                "        continue",
                "    if msg.get('method') == 'session/create':",
                "        send({'id': msg['id'], 'result': {'session': {'sessionId': 'sess_ok'}}})",
                "    elif 'id' in msg:",
                "        send({'id': msg['id'], 'result': {}})",
            ]
        ),
        encoding="utf-8",
    )
    process = subprocess.Popen(
        [sys.executable, "-u", str(script)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
    )
    client = _Client(process)
    try:
        assert client.create_session(
            workspace_path=str(tmp_path),
            workspace_key=str(tmp_path),
            provider_id="p",
            model_id="m",
            reasoning_level="low",
            thought_level="low",
            mode="plan",
            timeout=20,
        ) == "sess_ok"
    finally:
        client.close()


def test_zcode_run_episode_reaches_the_app_server(tmp_path: Path) -> None:
    """Full run_episode path: create pinned to the model/effort, then send."""
    seen: dict = {}

    script = tmp_path / "asserting_app_server.py"
    checks = json.dumps(
        {
            "provider": "zai-direct",
            "model": "glm-5.3-flash",
            "level": "max",
            "mode": "plan",
            "workspace": str(tmp_path),
        }
    )
    script.write_text(
        "\n".join(
            [
                "import json, sys",
                "expected = json.loads(%r)" % checks,
                "def send(o):",
                "    sys.stdout.write(json.dumps(o) + chr(10)); sys.stdout.flush()",
                "for line in sys.stdin:",
                "    msg = json.loads(line)",
                "    method = msg.get('method')",
                "    if method == 'session/create':",
                "        p = msg['params']",
                "        sel = p['model']",
                "        assert sel['providerId'] == expected['provider'], sel",
                "        assert sel['modelId'] == expected['model'], sel",
                "        assert sel['options']['reasoningLevel'] == expected['level'], sel",
                "        assert p['thoughtLevel'] == expected['level'], p",
                "        assert p['mode'] == expected['mode'], p",
                "        assert p['workspace']['workspacePath'] == expected['workspace'], p",
                "        send({'id': msg['id'], 'result': {'session': {'sessionId': 'sess_e2e'}}})",
                "    elif method == 'session/send':",
                "        assert 'unique-tail-token' in msg['params']['content']",
                "        send({'id': msg['id'], 'result': {'accepted': True, 'sessionId': 'sess_e2e'}})",
                "    elif method == 'session/messages':",
                "        send({'id': msg['id'], 'result': {'messages': [{'info': {'role': 'assistant', 'finish': 'stop', 'tokens': {'output': 2}}, 'parts': [{'type': 'text', 'text': 'done'}]}]}})",
                "    elif 'id' in msg:",
                "        send({'id': msg['id'], 'result': {}})",
            ]
        ),
        encoding="utf-8",
    )

    result = run_episode(
        argv=[sys.executable, "-u", str(script)],
        workspace_path=str(tmp_path),
        workspace_key=str(tmp_path),
        provider_id="zai-direct",
        model_id="glm-5.3-flash",
        reasoning_level="max",
        thought_level="max",
        mode="plan",
        content="task with unique-tail-token",
        timeout=60,
    )
    assert result["text"] == "done"
    assert result["session_id"] == "sess_e2e"


def test_zcode_jsonl_views() -> None:
    raw = '{"type":"zcode.result","text":"hi","is_error":false,"exit_code":0}\n'
    assert agent_logs.visible_output(raw) == "hi"
