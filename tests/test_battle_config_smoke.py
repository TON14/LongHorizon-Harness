"""Bootstrap smoke with every [run.semif] feature enabled.

Catches the enabled-CLI-path crash class: the effort-router NameError
(2026-09-23) shipped green because no test exercised the wiring that only
runs when the flags are on. This module boots the real resolvers and the
CLI variant builder with a full battle config and stub agents — no LLM,
no subprocess — and asserts the machinery constructs.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from lhht import cli, manager
from lhht import auditor_agent, role_prompts
from lhht.config import _flatten_run_table

FULL_SEMIF_TABLE = {
    "enabled": True,
    "command": "C:/fake/semif-score.exe",
    "model": "Qwen/Qwen3.5-4B",
    "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    "threshold": 0.8,
    "timeout_seconds": 60,
    "mcp_tool": True,
    "mcp_python": "C:/fake/python.exe",
    "mcp_script": "C:/fake/semif_mcp.py",
    "auditor_fast": True,
    "auditor_fast_threshold": 0.95,
    "cross_check": True,
    "cross_check_threshold": 0.9,
    "round_dedup": True,
    "round_dedup_threshold": 0.9,
    "effort_routing": True,
    "effort_threshold": 0.9,
    "report_selection": True,
    "report_selection_k": 3,
    "report_selection_threshold": 0.6,
}


def test_full_semif_table_flattens():
    defaults = _flatten_run_table({"semif": FULL_SEMIF_TABLE, "agent": "zcode"})
    for key in (
        "semif_enabled",
        "semif_auditor_fast",
        "semif_cross_check",
        "semif_round_dedup",
        "semif_effort_routing",
        "semif_report_selection",
    ):
        assert defaults[key] is True


def test_cli_effort_variants_boot_with_all_features_enabled():
    calls = []

    def build_role_agent(role, *, reasoning_effort=None, **kwargs):
        calls.append((role, reasoning_effort))
        return SimpleNamespace(role=role, effort=reasoning_effort)

    args = SimpleNamespace(
        cli_executor_agent=None,
        cli_executor_model=None,
        cli_executor_reasoning_effort=None,
        executor_agent=None,
        executor_model=None,
        executor_reasoning_effort="high",
        agent="zcode",
        model=None,
        reasoning_effort=None,
    )
    defaults = _flatten_run_table({"semif": FULL_SEMIF_TABLE, "agent": "zcode"})
    variants, default_effort = cli._cli_executor_effort_variants(
        defaults, args, build_role_agent
    )
    assert default_effort == "high"
    assert set(variants) == set(manager.EFFORT_VARIANTS)
    assert ("cli_executor", "low") in calls


def test_manager_side_resolvers_accept_the_battle_config(monkeypatch):
    defaults = _flatten_run_table({"semif": FULL_SEMIF_TABLE, "agent": "zcode"})
    # Stub the subprocess scorer so no real command runs.
    from lhht import semantic_salvage

    monkeypatch.setattr(
        semantic_salvage, "SemifCliScorer", lambda *a, **k: SimpleNamespace(
            score=lambda *a, **k: None
        )
    )
    monkeypatch.setattr(manager, "load_run_defaults", lambda: defaults)
    monkeypatch.setattr(
        "lhht.auditor_agent.load_run_defaults", lambda: defaults
    )
    monkeypatch.setattr(
        "lhht.role_prompts.load_run_defaults", lambda: defaults
    )
    # The round-dedup resolver lives in its own module, so it needs its own
    # patch: unlike the gate/cross-check it is smoke-booted from the flattened
    # battle config rather than the project's on-disk table.
    monkeypatch.setattr(
        "lhht.round_dedup.load_run_defaults", lambda: defaults
    )
    # Fresh lazy caches, then resolve everything the loop would resolve.
    auditor_agent._PROJECT_SALVAGE_SETTINGS.pop("settings", None)
    role_prompts._ROUTE_SALVAGE_SETTINGS.pop("settings", None)
    scorer, gate_threshold = manager.resolve_fast_gate()
    assert scorer is not None and gate_threshold == 0.95
    dedup_scorer, dedup_threshold = manager.resolve_round_dedup()
    assert dedup_scorer is not None and dedup_threshold == 0.9
    if hasattr(manager, "_resolve_effort_router_settings"):
        manager._resolve_effort_router_settings.cache_clear() if hasattr(manager._resolve_effort_router_settings, "cache_clear") else None
    # resolve_effort_router(variants, default_effort) is exercised by the
    # manager tests; here the router construction from config is the point.
    router = manager.resolve_effort_router(
        variants={"low": object(), "high": object(), "max": object()},
        default_effort="high",
    )
    assert router is not None
