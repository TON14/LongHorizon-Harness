"""The scorer's in-role MCP registration, shared by the agent adapters.

`[run.semif] mcp_tool` must hand every backend the same stdio server in its
own wire format: ZCode a per-session object list, Claude Code a merged
`--mcp-config` document. Any gap leaves sessions without the tool instead of
failing the run.
"""

from __future__ import annotations

import json
from pathlib import Path

import lhht.config
from lhht.adapters.claude_code import ClaudeCodeAdapter
from lhht.adapters.zcode import _semif_mcp_json
from lhht.semif_mcp import SERVER_NAME, semif_mcp_command

_ENABLED = {
    "semif_mcp_tool": True,
    "semif_mcp_python": "C:/sidecar/python.exe",
    "semif_mcp_script": "C:/repo/scripts/semif_mcp.py",
}


def _patch_defaults(monkeypatch, defaults):
    monkeypatch.setattr(lhht.config, "load_run_defaults", lambda *a, **k: dict(defaults))


# --- the shared gate ----------------------------------------------------


def test_command_gate_returns_interpreter_and_script() -> None:
    assert semif_mcp_command(_ENABLED) == (
        "C:/sidecar/python.exe",
        ["C:/repo/scripts/semif_mcp.py"],
    )


def test_command_gate_is_closed_when_off_or_incomplete() -> None:
    assert semif_mcp_command({}) is None
    off = dict(_ENABLED, semif_mcp_tool=False)
    assert semif_mcp_command(off) is None
    no_script = {k: v for k, v in _ENABLED.items() if k != "semif_mcp_script"}
    assert semif_mcp_command(no_script) is None
    assert semif_mcp_command({"semif_mcp_tool": True, "semif_mcp_python": "", "semif_mcp_script": "s"}) is None


# --- Claude Code: one merged --mcp-config document ----------------------


def test_claude_adapter_passes_merged_mcp_config_when_tool_on(tmp_path, monkeypatch) -> None:
    _patch_defaults(monkeypatch, _ENABLED)
    adapter = ClaudeCodeAdapter(
        workspace_path=str(tmp_path),
        prompt_dir=str(tmp_path / "prompts"),
        role="cli_executor",
    )

    assert adapter.semif_mcp_configured is True
    argv = adapter.argv
    assert "--mcp-config" in argv
    document = json.loads(Path(argv[argv.index("--mcp-config") + 1]).read_text(encoding="utf-8"))
    assert document["mcpServers"][SERVER_NAME] == {
        "command": "C:/sidecar/python.exe",
        "args": ["C:/repo/scripts/semif_mcp.py"],
    }


def test_claude_adapter_skips_mcp_config_when_tool_off(tmp_path, monkeypatch) -> None:
    _patch_defaults(monkeypatch, {"semif_mcp_tool": False})
    adapter = ClaudeCodeAdapter(
        workspace_path=str(tmp_path),
        prompt_dir=str(tmp_path / "prompts"),
        role="cli_executor",
    )

    assert adapter.semif_mcp_configured is False
    assert "--mcp-config" not in adapter.argv


def test_claude_adapter_merges_operator_computer_mcp_servers(tmp_path, monkeypatch) -> None:
    operator_config = tmp_path / "operator-mcp.json"
    operator_config.write_text(
        json.dumps({"mcpServers": {"computer": {"command": "computer-server"}}}),
        encoding="utf-8",
    )
    _patch_defaults(monkeypatch, _ENABLED)
    adapter = ClaudeCodeAdapter(
        workspace_path=str(tmp_path),
        prompt_dir=str(tmp_path / "prompts"),
        role="cli_executor",
        mcp_config=str(operator_config),
    )

    assert adapter.computer_mcp_configured is True
    argv = adapter.argv
    document = json.loads(Path(argv[argv.index("--mcp-config") + 1]).read_text(encoding="utf-8"))
    assert set(document["mcpServers"]) == {"computer", SERVER_NAME}


def test_claude_adapter_passes_broken_operator_config_through(tmp_path, monkeypatch) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("not json", encoding="utf-8")
    _patch_defaults(monkeypatch, _ENABLED)
    adapter = ClaudeCodeAdapter(
        workspace_path=str(tmp_path),
        prompt_dir=str(tmp_path / "prompts"),
        role="cli_executor",
        mcp_config=str(broken),
    )

    # The CLI must report the operator's broken file; lhht does not hide it,
    # and the scorer tool sits this episode out.
    argv = adapter.argv
    assert argv[argv.index("--mcp-config") + 1] == str(broken)
    assert adapter.semif_mcp_configured is False


def test_claude_adapter_without_scorer_passes_operator_path_untouched(
    tmp_path, monkeypatch
) -> None:
    operator_config = tmp_path / "operator-mcp.json"
    operator_config.write_text(
        json.dumps({"mcpServers": {"computer": {"command": "computer-server"}}}),
        encoding="utf-8",
    )
    _patch_defaults(monkeypatch, {"semif_mcp_tool": False})
    adapter = ClaudeCodeAdapter(
        workspace_path=str(tmp_path),
        prompt_dir=str(tmp_path / "prompts"),
        role="cli_executor",
        mcp_config=str(operator_config),
    )

    argv = adapter.argv
    assert argv[argv.index("--mcp-config") + 1] == str(operator_config)


# --- ZCode: per-session protocol registration ---------------------------


def test_zcode_registration_writes_protocol_server_list(tmp_path, monkeypatch) -> None:
    _patch_defaults(monkeypatch, _ENABLED)

    path = _semif_mcp_json(str(tmp_path))

    assert path is not None
    servers = json.loads(Path(path).read_text(encoding="utf-8"))
    assert servers == [{
        "name": SERVER_NAME,
        "command": "C:/sidecar/python.exe",
        "args": ["C:/repo/scripts/semif_mcp.py"],
        "env": [],
        "isolation": "session",
    }]


def test_zcode_registration_returns_none_when_off(tmp_path, monkeypatch) -> None:
    _patch_defaults(monkeypatch, {})
    assert _semif_mcp_json(str(tmp_path)) is None


def test_zcode_registration_respects_the_block_list(tmp_path, monkeypatch) -> None:
    _patch_defaults(monkeypatch, {**_ENABLED, "manager_mcp_blocked": [SERVER_NAME]})

    assert _semif_mcp_json(str(tmp_path), role="manager") is None
    assert _semif_mcp_json(str(tmp_path), role="cli_executor") is not None


# --- Codex: mcp_servers config overrides --------------------------------


def test_codex_adapter_registers_scorer_as_config_override(tmp_path, monkeypatch) -> None:
    import tomllib

    from lhht.adapters.codex import CodexAdapter

    _patch_defaults(monkeypatch, _ENABLED)
    adapter = CodexAdapter(
        workspace_path=str(tmp_path),
        prompt_dir=str(tmp_path / "prompts"),
    )

    argv = adapter.argv
    overrides = [
        argv[index + 1] for index, item in enumerate(argv) if item == "-c"
    ]
    mcp = [o for o in overrides if o.startswith(f"mcp_servers.{SERVER_NAME}=")]
    assert mcp, "the scorer server must ride as an mcp_servers override"
    spec = tomllib.loads("x=" + mcp[0].split("=", 1)[1])["x"]
    assert spec == {
        "command": "C:/sidecar/python.exe",
        "args": ["C:/repo/scripts/semif_mcp.py"],
    }
    assert adapter.semif_mcp_configured is True


def test_codex_adapter_skips_registration_when_tool_off(tmp_path, monkeypatch) -> None:
    from lhht.adapters.codex import CodexAdapter

    _patch_defaults(monkeypatch, {"semif_mcp_tool": False})
    adapter = CodexAdapter(
        workspace_path=str(tmp_path),
        prompt_dir=str(tmp_path / "prompts"),
    )

    assert not [
        item for item in adapter.argv if str(item).startswith(f"mcp_servers.{SERVER_NAME}")
    ]
    assert adapter.semif_mcp_configured is False


# --- Per-role mcp_allow / mcp_blocked ------------------------------------


def test_manager_policy_leaves_mcp_visibility_to_the_lists() -> None:
    from lhht.adapters.claude_permissions import policy_for_role

    manager = policy_for_role("manager")
    # Visibility is the operator's call via mcp_allow/mcp_blocked, not a
    # blanket tool deny that would block even the read-only scorer.
    assert "mcp__*" not in manager.disallowed_tools
    assert "Bash" in manager.disallowed_tools


def test_defaults_leave_claude_discovery_untouched(tmp_path, monkeypatch) -> None:
    _patch_defaults(monkeypatch, _ENABLED)
    adapter = ClaudeCodeAdapter(
        workspace_path=str(tmp_path),
        prompt_dir=str(tmp_path / "prompts"),
        role="manager",
    )

    # allow ["*"] + blocked [] = hands-off: claude loads its own servers, lhht
    # only adds its registrations; no strict flag anywhere.
    assert "--strict-mcp-config" not in adapter.argv
    assert adapter.mcp_servers_loaded is None
    assert adapter.semif_mcp_configured is True


def test_role_allow_list_restricts_to_admitted_servers(tmp_path, monkeypatch) -> None:
    _patch_defaults(
        monkeypatch,
        {**_ENABLED, "manager_mcp_allow": ["semif-scorer"]},
    )
    monkeypatch.setattr(
        "lhht.adapters.claude_code._claude_user_mcp_servers",
        lambda: {"gitlab": {"command": "g"}, "playwright": {"command": "p"}},
    )
    monkeypatch.setattr(
        "lhht.adapters.claude_code._workspace_mcp_servers",
        lambda workspace: {"workspace-srv": {"command": "w"}},
    )
    adapter = ClaudeCodeAdapter(
        workspace_path=str(tmp_path),
        prompt_dir=str(tmp_path / "prompts"),
        role="manager",
    )

    argv = adapter.argv
    assert "--strict-mcp-config" in argv
    document = json.loads(Path(argv[argv.index("--mcp-config") + 1]).read_text(encoding="utf-8"))
    assert list(document["mcpServers"]) == [SERVER_NAME]
    assert adapter.mcp_servers_loaded == [SERVER_NAME]


def test_block_list_subtracts_from_everything(tmp_path, monkeypatch) -> None:
    _patch_defaults(
        monkeypatch,
        {**_ENABLED, "cli_executor_mcp_blocked": ["playwright", SERVER_NAME]},
    )
    monkeypatch.setattr(
        "lhht.adapters.claude_code._claude_user_mcp_servers",
        lambda: {"gitlab": {"command": "g"}, "playwright": {"command": "p"}},
    )
    monkeypatch.setattr(
        "lhht.adapters.claude_code._workspace_mcp_servers", lambda workspace: {}
    )
    adapter = ClaudeCodeAdapter(
        workspace_path=str(tmp_path),
        prompt_dir=str(tmp_path / "prompts"),
        role="cli_executor",
    )

    argv = adapter.argv
    assert "--strict-mcp-config" in argv
    document = json.loads(Path(argv[argv.index("--mcp-config") + 1]).read_text(encoding="utf-8"))
    assert list(document["mcpServers"]) == ["gitlab"]
    assert adapter.semif_mcp_configured is False  # the block list wins over registration


def test_allow_list_naming_an_absent_server_loads_nothing(tmp_path, monkeypatch) -> None:
    _patch_defaults(monkeypatch, {"manager_mcp_allow": ["no-such-server"]})
    monkeypatch.setattr(
        "lhht.adapters.claude_code._claude_user_mcp_servers", lambda: {}
    )
    monkeypatch.setattr(
        "lhht.adapters.claude_code._workspace_mcp_servers", lambda workspace: {}
    )
    adapter = ClaudeCodeAdapter(
        workspace_path=str(tmp_path),
        prompt_dir=str(tmp_path / "prompts"),
        role="manager",
    )

    # Strict with no --mcp-config: zero MCP servers, claude's own included.
    assert "--strict-mcp-config" in adapter.argv
    assert "--mcp-config" not in adapter.argv
    assert adapter.mcp_servers_loaded == []


def test_config_lists_validate_star_and_types(tmp_path) -> None:
    import pytest

    from lhht.config import ProjectConfigError, load_run_defaults

    def config_with(body: str) -> None:
        (tmp_path / ".lhht").mkdir(exist_ok=True)
        (tmp_path / ".lhht" / "config.toml").write_text(body, encoding="utf-8")

    config_with('[run]\nmcp_blocked = ["*"]\n')
    with pytest.raises(ProjectConfigError, match="mcp_blocked"):
        load_run_defaults(tmp_path / ".lhht" / "config.toml")

    config_with('[run]\nmcp_allow = ["*", "gitlab"]\n')
    with pytest.raises(ProjectConfigError, match="exactly"):
        load_run_defaults(tmp_path / ".lhht" / "config.toml")

    config_with('[run.roles.manager]\nmcp_allow = ["semif-scorer"]\n')
    defaults = load_run_defaults(tmp_path / ".lhht" / "config.toml")
    assert defaults["manager_mcp_allow"] == ["semif-scorer"]
    assert "mcp_allow" not in defaults  # no global list written


# --- Shared mcp_allow / mcp_blocked policy --------------------------------


def test_policy_lists_default_open_and_role_overrides_global() -> None:
    from lhht.mcp_policy import (
        effective_mcp_lists,
        mcp_admits,
        mcp_restricted,
    )

    assert effective_mcp_lists({}, None) == (["*"], [])
    assert mcp_restricted(["*"], []) is False

    allow, blocked = effective_mcp_lists(
        {"mcp_blocked": ["gitlab"], "cli_auditor_mcp_allow": [SERVER_NAME]},
        "cli_auditor",
    )
    assert allow == [SERVER_NAME]  # role list replaces the global one
    assert blocked == ["gitlab"]  # ...per list, independently
    assert mcp_restricted(allow, blocked) is True
    assert mcp_admits(allow, blocked, SERVER_NAME) is True
    assert mcp_admits(allow, blocked, "playwright") is False
    # The block list wins over the allow list.
    assert mcp_admits(["*"], [SERVER_NAME], SERVER_NAME) is False
    assert mcp_admits([SERVER_NAME], [SERVER_NAME], SERVER_NAME) is False


def test_codex_blocked_list_removes_the_tool_for_that_role_only(
    tmp_path, monkeypatch
) -> None:
    from lhht.adapters.codex import CodexAdapter

    _patch_defaults(
        monkeypatch,
        {**_ENABLED, "cli_executor_mcp_blocked": [SERVER_NAME]},
    )
    blocked_adapter = CodexAdapter(
        workspace_path=str(tmp_path), prompt_dir=str(tmp_path / "p"),
        role="cli_executor",
    )
    other_adapter = CodexAdapter(
        workspace_path=str(tmp_path), prompt_dir=str(tmp_path / "p2"),
        role="cli_auditor",
    )

    assert blocked_adapter.semif_mcp_configured is False
    assert not [
        item for item in blocked_adapter.argv
        if str(item).startswith(f"mcp_servers.{SERVER_NAME}")
    ]
    assert other_adapter.semif_mcp_configured is True


def test_codex_allow_list_filters_operator_file_servers(tmp_path, monkeypatch) -> None:
    import tomllib

    from lhht.adapters.codex import CodexAdapter

    operator_config = tmp_path / "operators.toml"
    operator_config.write_text(
        "[mcp_servers.gitlab]\n"
        'command = "gitlab-server"\n'
        "[mcp_servers.playwright]\n"
        'command = "playwright-server"\n',
        encoding="utf-8",
    )
    _patch_defaults(
        monkeypatch,
        {**_ENABLED, "cli_executor_mcp_allow": ["gitlab", SERVER_NAME]},
    )
    adapter = CodexAdapter(
        workspace_path=str(tmp_path),
        prompt_dir=str(tmp_path / "p"),
        mcp_config=str(operator_config),
        role="cli_executor",
    )

    mcp_values = [
        adapter.argv[index + 1]
        for index, item in enumerate(adapter.argv)
        if item == "-c" and str(adapter.argv[index + 1]).startswith("mcp_servers.")
    ]
    names = sorted(value.split(".", 2)[1].split("=", 1)[0] for value in mcp_values)
    assert names == ["gitlab", SERVER_NAME]
    for value in mcp_values:
        tomllib.loads("x=" + value.split("=", 1)[1])  # every override is valid TOML


# --- `lhht init` scaffolds the scorer without turning it on -------------


def test_init_template_parses_with_fork_defaults(tmp_path) -> None:
    from lhht.config import create_project_config, load_run_defaults

    config_path = create_project_config(tmp_path / ".lhht" / "config.toml")
    defaults = load_run_defaults(config_path)

    assert defaults["max_rounds"] == 40
    assert defaults["manager_timeout"] == 10800
    assert defaults["cli_executor_timeout"] == 10800
    assert defaults["auditor_timeout"] == 10800
    # The [run.semif] scaffold is commented out, so a fresh project stays
    # byte-for-byte identical to a config without the section.
    assert not [key for key in defaults if key.startswith("semif_")]


def test_init_template_semif_scaffold_documents_the_switches(tmp_path) -> None:
    from lhht.config import CONFIG_TEMPLATE, create_project_config

    create_project_config(tmp_path / ".lhht" / "config.toml")
    template = (tmp_path / ".lhht" / "config.toml").read_text(encoding="utf-8")

    assert template == CONFIG_TEMPLATE
    for marker in (
        "[run.semif]",
        "mcp_tool = true",
        "auditor_fast = true",
        "effort_routing = true",
        "lhht server doctor",
    ):
        assert f"# {marker}" in template or marker in template
