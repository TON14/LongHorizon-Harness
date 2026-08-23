"""Keep the operator's Claude Code customisations out of harness agents.

A harness agent is a subprocess of the operator's account, so by default the
``claude`` CLI loads everything that account has accumulated: plugins, skills,
hooks, user-level CLAUDE.md. None of that was chosen for the run. It spends
context tokens in every episode, a skill description can trigger on task text
it was never meant for, and a hook can rewrite tool calls the harness believes
it controls. The harness already isolates the DeepSeek CLI behind its own
``DSH_HOME`` and scopes MCP with ``--strict-mcp-config``; this module gives
Claude Code the same property.

Isolation is opt-in and selective, because sometimes part of the toolbox is
wanted: ``--bare`` starts the CLI without the account's context, and the
allow-lists re-admit exactly the named plugins and skills. Plugins resolve
through the CLI's own install manifest rather than by guessing cache paths.
A skill has no standalone CLI switch, but a plugin may carry skills, so the
allowed skills ride in a synthesised single-purpose plugin directory.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

SKILLS_PLUGIN_NAME = "lh-harness-allowed-skills"


def _checked_name(name: str, *, kind: str) -> str:
    cleaned = str(name or "").strip()
    if not _NAME_RE.match(cleaned):
        raise ValueError(
            f"invalid {kind} name {name!r}: letters, digits, '.', '_' and '-' only"
        )
    return cleaned


def default_claude_home() -> Path:
    return Path.home() / ".claude"


def resolve_plugin_dirs(
    names: list[str] | tuple[str, ...],
    *,
    claude_home: Path | None = None,
) -> list[str]:
    """Map installed-plugin names to their on-disk directories.

    Reads the CLI's own ``installed_plugins.json`` (the authority on what is
    installed and where) instead of walking the cache layout, which encodes
    marketplace and version segments that are not a stable interface. A name
    matches either bare (``playwright``) or fully qualified
    (``playwright@claude-plugins-official``); a bare name that matches more
    than one marketplace is an error rather than a guess.
    """

    if not names:
        return []
    home = claude_home or default_claude_home()
    manifest_path = home / "plugins" / "installed_plugins.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(
            f"no Claude Code plugins are installed (missing {manifest_path})"
        ) from None
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read {manifest_path}: {exc}") from exc

    entries: dict[str, list[dict]] = manifest.get("plugins") or {}
    resolved: list[str] = []
    for raw in names:
        wanted = _checked_name(raw, kind="plugin") if "@" not in str(raw) else str(raw).strip()
        matches = [
            installs
            for qualified, installs in entries.items()
            if qualified == wanted or qualified.split("@", 1)[0] == wanted
        ]
        if not matches:
            installed = ", ".join(sorted(entries)) or "none"
            raise ValueError(
                f"Claude Code plugin {wanted!r} is not installed (installed: {installed})"
            )
        if len(matches) > 1:
            raise ValueError(
                f"Claude Code plugin name {wanted!r} is ambiguous across marketplaces; "
                "qualify it as name@marketplace"
            )
        installs = matches[0]
        install_path = str((installs or [{}])[0].get("installPath") or "")
        if not install_path or not (Path(install_path) / ".claude-plugin" / "plugin.json").is_file():
            raise ValueError(
                f"Claude Code plugin {wanted!r} has no usable install at {install_path!r}"
            )
        resolved.append(install_path)
    return resolved


def build_skills_plugin(
    names: list[str] | tuple[str, ...],
    target_dir: Path,
    *,
    claude_home: Path | None = None,
) -> str | None:
    """Materialise the allowed user skills as one synthesised plugin.

    Copies (never symlinks: creating links needs a privilege most Windows
    accounts lack) each ``~/.claude/skills/<name>`` into a plugin-shaped
    directory under the run's own tree. The copy also freezes the skill for
    the run: an edit to the live skill mid-run cannot change agent behaviour
    between episodes.
    """

    if not names:
        return None
    home = claude_home or default_claude_home()
    skills_root = home / "skills"
    target_dir = Path(target_dir)
    if target_dir.exists():
        shutil.rmtree(target_dir)
    (target_dir / ".claude-plugin").mkdir(parents=True)
    (target_dir / ".claude-plugin" / "plugin.json").write_text(
        json.dumps(
            {
                "name": SKILLS_PLUGIN_NAME,
                "version": "0.0.0",
                "description": "User skills explicitly allowed for this LongHorizon-Harness run.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    for raw in names:
        name = _checked_name(raw, kind="skill")
        source = skills_root / name
        if not (source / "SKILL.md").is_file():
            available = ", ".join(sorted(p.name for p in skills_root.glob("*/") )) or "none"
            raise ValueError(
                f"Claude Code skill {name!r} not found under {skills_root} "
                f"(available: {available})"
            )
        shutil.copytree(source, target_dir / "skills" / name)
    return str(target_dir)
