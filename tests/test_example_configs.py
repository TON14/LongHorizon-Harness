"""The shipped example configs must parse with today's loader.

Each examples/config-<backend>.toml is what a new operator copies verbatim;
if a key drifts out of the schema, the copy must fail loudly here, not in
their first run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lhht.config import load_run_defaults

_EXAMPLES = sorted(Path(__file__).resolve().parents[1].glob("examples/config-*.toml"))

_BACKENDS = {"zcode", "claude_code", "codex"}


def test_example_configs_exist_for_every_supported_backend() -> None:
    assert {path.stem.removeprefix("config-") for path in _EXAMPLES} == {
        "zcode",
        "claude",
        "codex",
    }


@pytest.mark.parametrize("path", _EXAMPLES, ids=lambda path: path.name)
def test_example_config_parses_with_battle_defaults(path: Path) -> None:
    defaults = load_run_defaults(path)

    assert defaults["agent"] in _BACKENDS
    assert defaults["model"]
    assert defaults["max_rounds"] == 40
    for role in ("manager", "cli_executor", "auditor"):
        assert defaults[f"{role}_timeout"] == 10800
    # The scorer section is complete: server mode needs no model
    # coordinates, but the tool registration paths must be present.
    assert defaults["semif_enabled"] is True
    assert defaults["semif_server"]
    assert defaults["semif_mcp_tool"] is True
    assert defaults["semif_mcp_python"]
    assert defaults["semif_mcp_script"]
