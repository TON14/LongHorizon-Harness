"""Isolation keeps the operator's Claude toolbox out of harness agents.

The default is untouched: without opting in, the CLI is launched exactly as
before and inherits whatever the account has installed. Opting in must start
from a clean slate and re-admit only what was named.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lh_harness.adapters.claude_code import ClaudeCodeAdapter
from lh_harness.adapters.claude_isolation import (
    SKILLS_PLUGIN_NAME,
    build_skills_plugin,
    resolve_plugin_dirs,
)


def _claude_home(tmp_path: Path, *, plugins: dict[str, Path] = {}, skills: tuple[str, ...] = ()) -> Path:
    home = tmp_path / "claude-home"
    entries = {}
    for qualified, install in plugins.items():
        (install / ".claude-plugin").mkdir(parents=True)
        (install / ".claude-plugin" / "plugin.json").write_text(
            json.dumps({"name": qualified.split("@", 1)[0]}), encoding="utf-8"
        )
        entries[qualified] = [{"scope": "user", "installPath": str(install)}]
    (home / "plugins").mkdir(parents=True)
    (home / "plugins" / "installed_plugins.json").write_text(
        json.dumps({"version": 2, "plugins": entries}), encoding="utf-8"
    )
    for name in skills:
        skill = home / "skills" / name
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text(f"# {name}\n", encoding="utf-8")
        (skill / "references").mkdir()
        (skill / "references" / "notes.md").write_text("notes\n", encoding="utf-8")
    return home


# --- default behaviour is unchanged -----------------------------------------


def test_without_opt_in_no_isolation_flags_appear(tmp_path):
    adapter = ClaudeCodeAdapter(workspace_path=str(tmp_path), prompt_dir=str(tmp_path / "p"))
    assert "--setting-sources" not in adapter.argv
    assert "--disable-slash-commands" not in adapter.argv
    assert "--plugin-dir" not in adapter.argv


# --- bare isolation ----------------------------------------------------------


def test_isolation_alone_is_bare_with_no_skills(tmp_path):
    adapter = ClaudeCodeAdapter(
        workspace_path=str(tmp_path), prompt_dir=str(tmp_path / "p"), isolation=True
    )
    idx = adapter.argv.index("--setting-sources")
    assert adapter.argv[idx + 1] == "project"
    assert "--disable-slash-commands" in adapter.argv
    assert "--plugin-dir" not in adapter.argv


def test_allow_lists_imply_isolation(tmp_path, monkeypatch):
    home = _claude_home(tmp_path, skills=("graphify",))
    monkeypatch.setattr(
        "lh_harness.adapters.claude_isolation.default_claude_home", lambda: home
    )
    adapter = ClaudeCodeAdapter(
        workspace_path=str(tmp_path),
        prompt_dir=str(tmp_path / "run" / "prompts"),
        allowed_skills=("graphify",),
    )
    assert adapter.isolation is True
    assert "--setting-sources" in adapter.argv
    # Something was re-admitted, so skills must stay resolvable.
    assert "--disable-slash-commands" not in adapter.argv


# --- plugin resolution -------------------------------------------------------


def test_plugins_resolve_through_the_install_manifest(tmp_path):
    install = tmp_path / "cache" / "playwright" / "1.0.0"
    home = _claude_home(tmp_path, plugins={"playwright@official": install})
    dirs = resolve_plugin_dirs(["playwright"], claude_home=home)
    assert dirs == [str(install)]


def test_unknown_plugin_is_a_startup_error(tmp_path):
    home = _claude_home(tmp_path)
    with pytest.raises(ValueError, match="not installed"):
        resolve_plugin_dirs(["ghost"], claude_home=home)


def test_ambiguous_bare_name_requires_qualification(tmp_path):
    home = _claude_home(
        tmp_path,
        plugins={
            "tool@market-a": tmp_path / "a" / "tool",
            "tool@market-b": tmp_path / "b" / "tool",
        },
    )
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_plugin_dirs(["tool"], claude_home=home)
    assert resolve_plugin_dirs(["tool@market-a"], claude_home=home) == [
        str(tmp_path / "a" / "tool")
    ]


# --- synthesised skills plugin ----------------------------------------------


def test_allowed_skills_ride_in_a_synthesised_plugin(tmp_path):
    home = _claude_home(tmp_path, skills=("graphify", "unrelated"))
    target = tmp_path / "run" / "claude-allowed-skills"
    plugin = build_skills_plugin(("graphify",), target, claude_home=home)
    assert plugin == str(target)
    manifest = json.loads((target / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))
    assert manifest["name"] == SKILLS_PLUGIN_NAME
    # The named skill travels whole; the unnamed one does not travel at all.
    assert (target / "skills" / "graphify" / "SKILL.md").is_file()
    assert (target / "skills" / "graphify" / "references" / "notes.md").is_file()
    assert not (target / "skills" / "unrelated").exists()


def test_missing_skill_is_a_startup_error(tmp_path):
    home = _claude_home(tmp_path, skills=("graphify",))
    with pytest.raises(ValueError, match="not found"):
        build_skills_plugin(("ghost",), tmp_path / "out", claude_home=home)


def test_skill_names_cannot_traverse_paths(tmp_path):
    home = _claude_home(tmp_path, skills=("graphify",))
    with pytest.raises(ValueError, match="invalid skill name"):
        build_skills_plugin(("../secrets",), tmp_path / "out", claude_home=home)


def test_adapter_passes_every_allowed_dir_to_the_cli(tmp_path, monkeypatch):
    install = tmp_path / "cache" / "playwright" / "1.0.0"
    home = _claude_home(tmp_path, plugins={"playwright@official": install}, skills=("graphify",))
    monkeypatch.setattr(
        "lh_harness.adapters.claude_isolation.default_claude_home", lambda: home
    )
    adapter = ClaudeCodeAdapter(
        workspace_path=str(tmp_path),
        prompt_dir=str(tmp_path / "run" / "prompts"),
        allowed_plugins=("playwright",),
        allowed_skills=("graphify",),
    )
    plugin_dirs = [
        adapter.argv[i + 1] for i, part in enumerate(adapter.argv) if part == "--plugin-dir"
    ]
    assert str(install) in plugin_dirs
    assert str(tmp_path / "run" / "claude-allowed-skills") in plugin_dirs
