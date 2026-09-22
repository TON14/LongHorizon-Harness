"""Scorer-based executor effort routing: per-round cli-executor variants.

The classifier question, the threshold cascade, the escalation rule, and the
round-record fields are all exercised through a fake scorer -- never the real
shim, so the suite stays hermetic. The wiring tests drive the full manager
loop with fake agents and assert the binding cascade end to end.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pytest

import lhht.cli as cli_module
import lhht.manager as manager_module
from lhht.config import ProjectConfigError, _flatten_run_table
from lhht.environment.local import LocalEnvironment
from lhht.manager import run
from lhht.role_prompts import (
    MANAGER_NEXT_CLI,
    MANAGER_NEXT_GUI,
    MANAGER_NEXT_INVALID,
)
from lhht.types import EpisodeResult, HarnessConfig, ManagedRound

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeScorer:
    """One canned distribution per question, resolved by mapping.

    ``resolver(question)`` returns ``{option_id: probability}``; returning
    ``None`` simulates the CLI scorer's timeout/unusable answer.
    """

    def __init__(self, resolver: Any = None) -> None:
        self._resolver = resolver
        self.calls: list[tuple[str, str, list[str]]] = []

    def score(self, state: str, question: str, options: list[dict[str, str]]):
        self.calls.append((state, question, [option["id"] for option in options]))
        if self._resolver is None:
            return None
        mapping = self._resolver(question)
        if mapping is None:
            return None
        return [float(mapping.get(option["id"], 0.0)) for option in options]


class RaisingScorer:
    def __init__(self) -> None:
        self.calls = 0

    def score(self, state: str, question: str, options: list[dict[str, str]]):
        self.calls += 1
        raise RuntimeError("scorer exploded")


def _mechanical(_question: str):
    return {"mechanical": 0.97, "standard": 0.02, "deep": 0.01}


def _standard(_question: str):
    return {"mechanical": 0.02, "standard": 0.96, "deep": 0.02}


def _deep(_question: str):
    return {"mechanical": 0.01, "standard": 0.04, "deep": 0.95}


def _router(
    scorer: Any,
    *,
    threshold: float = 0.9,
    default_effort: str = "high",
) -> manager_module._EffortRouter:
    return manager_module._EffortRouter(
        scorer,
        float(threshold),
        {"low": object(), "high": object(), "max": object()},
        default_effort,
    )


class SequencedAgent:
    """Replays a fixed reply sequence and records the prompts it received."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.prompts: list[str] = []

    async def run_episode(self, prompt, _env, _budget, live_trajectory_path=None):
        self.prompts.append(str(prompt))
        reply = self._replies.pop(0) if self._replies else "Next: done"
        return EpisodeResult(status="done", actions_log=reply)


def _harness_config(tmp_path: Path) -> HarnessConfig:
    return HarnessConfig(
        max_total_episodes=2,
        workspace_path=str(tmp_path / "workspace"),
        harness_dir=str(tmp_path / "harness"),
        log_dir=str(tmp_path / "logs"),
    )


def _events(tmp_path: Path) -> list[dict[str, Any]]:
    path = tmp_path / "logs" / "role_orchestration" / "events.jsonl"
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _recorded_rounds(tmp_path: Path) -> list[dict[str, Any]]:
    path = tmp_path / "logs" / "role_orchestration" / "rounds.jsonl"
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# ---------------------------------------------------------------------------
# config: [run.semif] keys
# ---------------------------------------------------------------------------


def test_semif_effort_keys_flatten_into_defaults() -> None:
    defaults = _flatten_run_table(
        {"semif": {"effort_routing": True, "effort_threshold": 0.85}}
    )

    assert defaults["semif_effort_routing"] is True
    assert defaults["semif_effort_threshold"] == 0.85


def test_absent_semif_effort_keys_leave_defaults_empty() -> None:
    for defaults in (
        _flatten_run_table({}),
        _flatten_run_table({"semif": {"enabled": False}}),
        _flatten_run_table({"semif": {"command": "scripts/semif_shim.bat"}}),
    ):
        assert "semif_effort_routing" not in defaults
        assert "semif_effort_threshold" not in defaults


def test_unknown_semif_effort_key_is_refused() -> None:
    with pytest.raises(ProjectConfigError, match="unknown \\[run.semif\\] key"):
        _flatten_run_table({"semif": {"effort_routing_x": True}})


@pytest.mark.parametrize("bad", [0, 1.5, True, "high"])
def test_bad_effort_threshold_is_refused(bad: Any) -> None:
    with pytest.raises(ProjectConfigError, match="effort_threshold"):
        _flatten_run_table({"semif": {"effort_threshold": bad}})


def test_effort_routing_requires_a_bool() -> None:
    with pytest.raises(ProjectConfigError, match="effort_routing"):
        _flatten_run_table({"semif": {"effort_routing": "yes"}})


# ---------------------------------------------------------------------------
# router resolution
# ---------------------------------------------------------------------------


def test_router_off_for_absent_or_disabled_config_without_building_a_scorer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _no_scorer(_defaults: Any) -> None:  # pragma: no cover - tripwire
        raise AssertionError("no scorer may be constructed for routing off")

    monkeypatch.setattr(manager_module, "scorer_from_config", _no_scorer)

    assert manager_module.effort_router_from_defaults({}, {"low": 1}, "high") is None
    assert (
        manager_module.effort_router_from_defaults(
            {"semif_effort_routing": False}, {"low": 1}, "high"
        )
        is None
    )


def test_router_silently_off_when_the_scorer_is_not_configured() -> None:
    # scorer_from_config itself degrades to None without command/model/revision.
    assert (
        manager_module.effort_router_from_defaults(
            {"semif_effort_routing": True}, {"low": 1}, "high"
        )
        is None
    )


def test_router_off_without_variants_or_default_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _no_scorer(_defaults: Any) -> None:  # pragma: no cover - tripwire
        raise AssertionError("no scorer may be constructed without variants")

    monkeypatch.setattr(manager_module, "scorer_from_config", _no_scorer)

    assert (
        manager_module.effort_router_from_defaults(
            {"semif_effort_routing": True}, None, "high"
        )
        is None
    )
    assert (
        manager_module.effort_router_from_defaults(
            {"semif_effort_routing": True}, {"low": 1}, ""
        )
        is None
    )


def test_router_resolves_scorer_and_threshold_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()
    monkeypatch.setattr(
        manager_module, "scorer_from_config", lambda defaults: sentinel
    )

    router = manager_module.effort_router_from_defaults(
        {
            "semif_effort_routing": True,
            "semif_effort_threshold": 0.8,
        },
        {"low": 1, "high": 2, "max": 3},
        "high",
    )

    assert router is not None
    assert router.scorer is sentinel
    assert router.threshold == 0.8
    assert router.default_effort == "high"
    assert router.variants == {"low": 1, "high": 2, "max": 3}


def test_router_threshold_defaults_to_0_9_and_degrades_on_garbage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(manager_module, "scorer_from_config", lambda defaults: object())

    router = manager_module.effort_router_from_defaults(
        {"semif_effort_routing": True}, {"low": 1}, "high"
    )
    assert router is not None and router.threshold == 0.9

    router_bad = manager_module.effort_router_from_defaults(
        {"semif_effort_routing": True, "semif_effort_threshold": True},
        {"low": 1},
        "high",
    )
    assert router_bad is not None and router_bad.threshold == 0.9


# ---------------------------------------------------------------------------
# the routing decision
# ---------------------------------------------------------------------------


def _route(
    scorer: Any,
    *,
    rounds: list[ManagedRound] | None = None,
    round_index: int = 1,
    threshold: float = 0.9,
    default_effort: str = "high",
    memo: dict[int, Any] | None = None,
    plan_text: str = "the plan",
) -> tuple[str | None, dict[str, Any]]:
    return manager_module._route_effort(
        _router(scorer, threshold=threshold, default_effort=default_effort),
        plan_text=plan_text,
        round_index=round_index,
        rounds=rounds or [],
        memo=memo if memo is not None else {},
    )


def test_confident_classification_selects_its_variant() -> None:
    for resolver, expected in (
        (_mechanical, "low"),
        (_standard, "high"),
        (_deep, "max"),
    ):
        level, payload = _route(FakeScorer(resolver))

        assert level == expected
        assert payload["variant"] == expected
        assert payload["used_default"] is (expected == "high")
        assert payload["skipped"] is None
        assert payload["escalated"] is False
        assert payload["probabilities"] is not None
        assert sorted(payload["probabilities"]) == ["deep", "mechanical", "standard"]


def test_variant_equal_to_the_default_counts_as_default_used() -> None:
    level, payload = _route(FakeScorer(_standard), default_effort="high")

    assert level == "high"
    assert payload["variant"] == "high"
    assert payload["used_default"] is True


def test_low_confidence_keeps_the_default() -> None:
    scorer = FakeScorer(lambda q: {"mechanical": 0.7, "standard": 0.2, "deep": 0.1})

    level, payload = _route(scorer)

    assert level is None
    assert payload["used_default"] is True
    assert payload["variant"] == "high"
    assert payload["classified"] == "mechanical"
    assert payload["skipped"] == "below_threshold"
    assert payload["probabilities"] is not None, "low confidence is still recorded"


def test_threshold_is_inclusive() -> None:
    scorer = FakeScorer(lambda q: {"mechanical": 0.9, "standard": 0.06, "deep": 0.04})

    level, _payload = _route(scorer, threshold=0.9)

    assert level == "low"


def test_scorer_timeout_answer_keeps_the_default() -> None:
    level, payload = _route(FakeScorer(lambda q: None))

    assert level is None
    assert payload["used_default"] is True
    assert payload["probabilities"] is None
    assert payload["skipped"] == "scorer_unavailable"


def test_scorer_error_keeps_the_default() -> None:
    level, payload = _route(RaisingScorer())

    assert level is None
    assert payload["used_default"] is True
    assert payload["skipped"] == "scorer_unavailable"


def test_malformed_probabilities_keep_the_default() -> None:
    level, payload = _route(FakeScorer(lambda q: {"mechanical": "not-a-number"}))

    assert level is None
    assert payload["skipped"] == "scorer_unavailable"


def test_plan_text_question_and_options_reach_the_scorer() -> None:
    plan = "Next: cli\n\nTask: fix the flaky retry test\n"
    scorer = FakeScorer(_mechanical)

    _route(scorer, plan_text=plan)

    assert len(scorer.calls) == 1
    state, question, option_ids = scorer.calls[0]
    assert state == plan
    assert question == "How much reasoning depth does this executor subtask need?"
    assert option_ids == ["mechanical", "standard", "deep"]


def test_decision_is_memoized_per_round() -> None:
    scorer = FakeScorer(_mechanical)
    memo: dict[int, Any] = {}

    first = _route(scorer, round_index=4, memo=memo)
    second = _route(scorer, round_index=4, memo=memo)

    assert first == second
    assert len(scorer.calls) == 1, "one round classifies its plan exactly once"
    _route(scorer, round_index=5, memo=memo)
    assert len(scorer.calls) == 2, "a new round classifies again"


# ---------------------------------------------------------------------------
# escalation: no second downgrade before a clean-complete audit
# ---------------------------------------------------------------------------

_CLEAN_AUDIT = (
    "Status: complete\nIntegrity: clean\nContract audit: aligned\n\n"
    "Summary:\nthe change is verified"
)
_DIRTY_AUDIT = (
    "Status: incomplete\nIntegrity: suspect\nContract audit: unknown\n\n"
    "Summary:\nthe deliverable is missing"
)


def _routed_round(
    round_index: int, variant: str, used_default: bool, audit: str
) -> ManagedRound:
    return ManagedRound(
        round_index=round_index,
        next_step=MANAGER_NEXT_CLI,
        plan_text="plan",
        auditor_report=audit,
        executor_status={
            "effort_routing": {
                "variant": variant,
                "used_default": used_default,
                "default_effort": "high",
            }
        },
    )


def test_downgrade_after_failed_downgraded_round_is_suppressed() -> None:
    rounds = [_routed_round(1, "low", False, _DIRTY_AUDIT)]

    level, payload = _route(FakeScorer(_mechanical), rounds=rounds, round_index=2)

    assert level is None
    assert payload["used_default"] is True
    assert payload["variant"] == "high"
    assert payload["classified"] == "mechanical"
    assert payload["escalated"] is True
    assert payload["skipped"] == "escalated_to_default"


def test_clean_complete_audit_redeems_the_downgrade() -> None:
    rounds = [_routed_round(1, "low", False, _CLEAN_AUDIT)]

    level, payload = _route(FakeScorer(_mechanical), rounds=rounds, round_index=2)

    assert level == "low"
    assert payload["escalated"] is False


def test_escalation_still_allows_deep() -> None:
    rounds = [_routed_round(1, "low", False, _DIRTY_AUDIT)]

    level, payload = _route(FakeScorer(_deep), rounds=rounds, round_index=2)

    assert level == "max"
    assert payload["used_default"] is False
    assert payload["escalated"] is False


def test_escalation_allows_the_default_level() -> None:
    rounds = [_routed_round(1, "low", False, _DIRTY_AUDIT)]

    level, payload = _route(FakeScorer(_standard), rounds=rounds, round_index=2)

    assert level == "high"
    assert payload["used_default"] is True
    assert payload["escalated"] is False


def test_no_escalation_without_a_prior_downgrade() -> None:
    for rounds in (
        [],
        [_routed_round(1, "high", True, _DIRTY_AUDIT)],
        [_routed_round(1, "max", False, _DIRTY_AUDIT)],
    ):
        level, payload = _route(FakeScorer(_mechanical), rounds=rounds, round_index=2)

        assert level == "low"
        assert payload["escalated"] is False


def test_escalation_walks_past_manager_only_rounds() -> None:
    rounds = [
        _routed_round(1, "low", False, _DIRTY_AUDIT),
        ManagedRound(
            round_index=2,
            next_step=MANAGER_NEXT_INVALID,
            plan_text="Next: invalid",
            auditor_status={"invalid_plan": True},
        ),
    ]

    level, payload = _route(FakeScorer(_mechanical), rounds=rounds, round_index=3)

    assert level is None
    assert payload["escalated"] is True


def test_missing_audit_on_a_downgraded_round_escalates() -> None:
    rounds = [_routed_round(1, "low", False, "")]

    level, payload = _route(FakeScorer(_mechanical), rounds=rounds, round_index=2)

    assert level is None
    assert payload["escalated"] is True


# ---------------------------------------------------------------------------
# _executor_binding: disabled stays today's binding; GUI never routes
# ---------------------------------------------------------------------------


def test_binding_without_a_router_is_today_cli_binding() -> None:
    cli_agent = object()
    gui_agent = object()
    budget = manager_module.EpisodeBudget(max_duration_seconds=60)

    agent, bound_budget, routing = manager_module._executor_binding(
        next_step=MANAGER_NEXT_CLI,
        gui_executor_agent=gui_agent,
        cli_executor_agent=cli_agent,
        gui_executor_budget=budget,
        cli_executor_budget=budget,
    )

    assert agent is cli_agent
    assert bound_budget is budget
    assert routing == {}


def test_gui_binding_never_consults_the_router() -> None:
    scorer = FakeScorer(_mechanical)
    variants = {"low": object(), "high": object(), "max": object()}
    gui_agent, cli_agent = object(), object()
    gui_budget = manager_module.EpisodeBudget(max_duration_seconds=61)
    cli_budget = manager_module.EpisodeBudget(max_duration_seconds=62)

    agent, bound_budget, routing = manager_module._executor_binding(
        next_step=MANAGER_NEXT_GUI,
        gui_executor_agent=gui_agent,
        cli_executor_agent=cli_agent,
        gui_executor_budget=gui_budget,
        cli_executor_budget=cli_budget,
        effort_router=manager_module._EffortRouter(
            scorer, 0.9, variants, "high"
        ),
        plan_text="plan",
        round_index=1,
    )

    assert agent is gui_agent
    assert bound_budget is gui_budget
    assert routing == {}
    assert scorer.calls == []


def test_confident_binding_returns_the_chosen_variant_adapter() -> None:
    scorer = FakeScorer(_deep)
    low, high, default = object(), object(), object()
    variants = {"low": low, "high": high, "max": object()}

    agent, _budget, routing = manager_module._executor_binding(
        next_step=MANAGER_NEXT_CLI,
        gui_executor_agent=object(),
        cli_executor_agent=default,
        gui_executor_budget=manager_module.EpisodeBudget(max_duration_seconds=1),
        cli_executor_budget=manager_module.EpisodeBudget(max_duration_seconds=2),
        effort_router=manager_module._EffortRouter(
            scorer, 0.9, variants, "high"
        ),
        plan_text="plan",
        round_index=1,
    )

    assert agent is variants["max"]
    assert routing["variant"] == "max"


# ---------------------------------------------------------------------------
# the CLI variant builder
# ---------------------------------------------------------------------------


def _variant_args(effort: str | None) -> argparse.Namespace:
    return argparse.Namespace(
        agent="zcode",
        cli_executor_agent="zcode",
        cli_executor_reasoning_effort=effort,
    )


def _recording_builder(record: list[tuple[str, str | None]]):
    def build(role: str, *, permission_role=None, reasoning_effort=None):
        record.append((role, reasoning_effort))
        return f"adapter:{reasoning_effort}"

    return build


def test_cli_builds_no_variants_and_no_scorer_when_the_flag_is_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _no_scorer(_defaults: Any) -> None:  # pragma: no cover - tripwire
        raise AssertionError("no scorer may be constructed for routing off")

    monkeypatch.setattr(cli_module, "scorer_from_config", _no_scorer)
    record: list[tuple[str, str | None]] = []

    variants, default_effort = cli_module._cli_executor_effort_variants(
        {}, _variant_args("high"), _recording_builder(record)
    )

    assert variants is None
    assert default_effort == ""
    assert record == []


def test_cli_builds_nothing_when_the_scorer_is_not_configured() -> None:
    # The real scorer_from_config degrades to None without the scorer keys.
    record: list[tuple[str, str | None]] = []

    variants, default_effort = cli_module._cli_executor_effort_variants(
        {"semif_effort_routing": True},
        _variant_args("high"),
        _recording_builder(record),
    )

    assert variants is None
    assert default_effort == ""
    assert record == []


def test_cli_builds_three_variants_around_the_configured_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cli_module, "scorer_from_config", lambda defaults: object()
    )
    record: list[tuple[str, str | None]] = []

    variants, default_effort = cli_module._cli_executor_effort_variants(
        {"semif_effort_routing": True},
        _variant_args("high"),
        _recording_builder(record),
    )

    assert default_effort == "high"
    assert set(variants) == {"low", "high", "max"}
    assert [effort for _role, effort in record] == ["low", "high", "max"]
    assert variants["low"] == "adapter:low"


def test_cli_builds_nothing_for_an_unrankable_or_unset_effort(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(
        cli_module, "scorer_from_config", lambda defaults: object()
    )
    record: list[tuple[str, str | None]] = []

    variants, default_effort = cli_module._cli_executor_effort_variants(
        {"semif_effort_routing": True},
        _variant_args("ultra"),
        _recording_builder(record),
    )

    assert variants is None
    assert default_effort == ""
    assert record == [], "no adapter is built for an unrankable default effort"
    assert "routing stays off" in capsys.readouterr().err

    variants_unset, default_unset = cli_module._cli_executor_effort_variants(
        {"semif_effort_routing": True},
        _variant_args(None),
        _recording_builder(record),
    )

    assert variants_unset is None
    assert default_unset == ""
    assert record == []


def test_cli_degrades_to_off_when_a_backend_rejects_a_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cli_module, "scorer_from_config", lambda defaults: object()
    )

    def rejecting_builder(role, *, permission_role=None, reasoning_effort=None):
        if reasoning_effort == "max":
            raise ValueError("no max on this backend")
        return object()

    variants, default_effort = cli_module._cli_executor_effort_variants(
        {"semif_effort_routing": True},
        _variant_args("high"),
        rejecting_builder,
    )

    assert variants is None
    assert default_effort == ""


# ---------------------------------------------------------------------------
# Wiring: the manager loop with fake agents
# ---------------------------------------------------------------------------

_PLAN = (
    "Next: cli\n\n"
    "Current Task State:\nround 1 in flight\n\n"
    "Task contract:\n"
    "Acceptance constraints:\n"
    "1. The deliverable must exist in the workspace\n"
)
_EXECUTOR = "executor output: the mechanical change is in"
_DONE = "Next: done\n\nCurrent Task State:\nall finished"


def _enable_routing(monkeypatch: pytest.MonkeyPatch, scorer: Any) -> None:
    monkeypatch.setattr(
        manager_module,
        "load_run_defaults",
        lambda: {
            "semif_effort_routing": True,
            "semif_enabled": True,
            "semif_command": "scripts/semif_shim.bat",
            "semif_model": "m",
            "semif_revision": "r",
        },
    )
    monkeypatch.setattr(manager_module, "scorer_from_config", lambda defaults: scorer)


def _disable_routing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(manager_module, "load_run_defaults", lambda: {})

    def _no_scorer(_defaults: Any) -> None:  # pragma: no cover - tripwire
        raise AssertionError("no scorer may be constructed for routing off")

    monkeypatch.setattr(manager_module, "scorer_from_config", _no_scorer)


@pytest.mark.asyncio
async def test_disabled_routing_keeps_today_binding_and_calls_no_scorer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _disable_routing(monkeypatch)
    manager = SequencedAgent([_PLAN, _DONE])
    default_exec = SequencedAgent([_EXECUTOR])
    low_exec = SequencedAgent([_EXECUTOR])
    auditor = SequencedAgent([_CLEAN_AUDIT])

    report = await run(
        task="finish the feature",
        env=LocalEnvironment(str(tmp_path / "tmp")),
        config=_harness_config(tmp_path),
        manager_agent=manager,
        gui_executor_agent=default_exec,
        cli_executor_agent=default_exec,
        gui_auditor_agent=auditor,
        cli_auditor_agent=auditor,
        # Variants are passed anyway: the off flag must make the loop ignore them.
        cli_executor_effort_agents={"low": low_exec, "high": default_exec, "max": SequencedAgent([])},
        cli_executor_default_effort="high",
    )

    # Today's behavior end to end: the default executor ran and completed the
    # run through a real audit; no variant agent, no routing record fields.
    assert report["completion_satisfied"] is True
    assert report["status"] == "complete"
    assert len(default_exec.prompts) == 1
    assert low_exec.prompts == []
    round_one = _recorded_rounds(tmp_path)[0]
    assert "effort_routing" not in round_one["executor_status"]
    assert round_one["executor_status"]["status"] == "done"


@pytest.mark.asyncio
async def test_confident_mechanical_round_runs_the_low_variant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scorer = FakeScorer(_mechanical)
    _enable_routing(monkeypatch, scorer)
    manager = SequencedAgent([_PLAN, _DONE])
    default_exec = SequencedAgent([_EXECUTOR])
    low_exec = SequencedAgent([_EXECUTOR])
    max_exec = SequencedAgent([_EXECUTOR])
    auditor = SequencedAgent([_CLEAN_AUDIT])

    report = await run(
        task="finish the feature",
        env=LocalEnvironment(str(tmp_path / "tmp")),
        config=_harness_config(tmp_path),
        manager_agent=manager,
        gui_executor_agent=default_exec,
        cli_executor_agent=default_exec,
        gui_auditor_agent=auditor,
        cli_auditor_agent=auditor,
        cli_executor_effort_agents={
            "low": low_exec,
            "high": default_exec,
            "max": max_exec,
        },
        cli_executor_default_effort="high",
    )

    # The low variant received the executor prompt; everyone else stayed idle.
    assert len(low_exec.prompts) == 1
    assert default_exec.prompts == []
    assert max_exec.prompts == []
    # Manager and auditor roles ran normally: routing only rebinds the
    # executor. The manager agent also writes the run's closing reply.
    assert len(manager.prompts) == 3
    assert len(auditor.prompts) == 1
    assert report["completion_satisfied"] is True

    # The classifier saw exactly the round's (stripped) plan text, once.
    assert len(scorer.calls) == 1
    assert scorer.calls[0][0] == _PLAN.strip()

    # The round record and events carry the decision.
    round_one = _recorded_rounds(tmp_path)[0]
    routing = round_one["executor_status"]["effort_routing"]
    assert routing["variant"] == "low"
    assert routing["used_default"] is False
    assert routing["classified"] == "mechanical"
    assert routing["probabilities"]["mechanical"] == pytest.approx(0.97)
    recorded_events = [
        item
        for item in _events(tmp_path)
        if item["event"] == "managed_round_recorded"
    ]
    assert recorded_events[0]["executor_status"]["effort_routing"] == routing


@pytest.mark.asyncio
async def test_low_confidence_round_runs_the_default_variant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scorer = FakeScorer(lambda q: {"mechanical": 0.6, "standard": 0.3, "deep": 0.1})
    _enable_routing(monkeypatch, scorer)
    manager = SequencedAgent([_PLAN, _DONE])
    default_exec = SequencedAgent([_EXECUTOR])
    low_exec = SequencedAgent([_EXECUTOR])
    auditor = SequencedAgent([_CLEAN_AUDIT])

    await run(
        task="finish the feature",
        env=LocalEnvironment(str(tmp_path / "tmp")),
        config=_harness_config(tmp_path),
        manager_agent=manager,
        gui_executor_agent=default_exec,
        cli_executor_agent=default_exec,
        gui_auditor_agent=auditor,
        cli_auditor_agent=auditor,
        cli_executor_effort_agents={
            "low": low_exec,
            "high": default_exec,
            "max": SequencedAgent([_EXECUTOR]),
        },
        cli_executor_default_effort="high",
    )

    assert len(default_exec.prompts) == 1
    assert low_exec.prompts == []
    round_one = _recorded_rounds(tmp_path)[0]
    routing = round_one["executor_status"]["effort_routing"]
    assert routing["used_default"] is True
    assert routing["variant"] == "high"
    assert routing["skipped"] == "below_threshold"


@pytest.mark.asyncio
async def test_escalation_after_a_failed_downgraded_round(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scorer = FakeScorer(_mechanical)
    _enable_routing(monkeypatch, scorer)
    manager = SequencedAgent([_PLAN, _PLAN])
    default_exec = SequencedAgent([_EXECUTOR])
    low_exec = SequencedAgent([_EXECUTOR])
    auditor = SequencedAgent([_DIRTY_AUDIT, _CLEAN_AUDIT])

    await run(
        task="finish the feature",
        env=LocalEnvironment(str(tmp_path / "tmp")),
        config=_harness_config(tmp_path),
        manager_agent=manager,
        gui_executor_agent=default_exec,
        cli_executor_agent=default_exec,
        gui_auditor_agent=auditor,
        cli_auditor_agent=auditor,
        cli_executor_effort_agents={
            "low": low_exec,
            "high": default_exec,
            "max": SequencedAgent([_EXECUTOR]),
        },
        cli_executor_default_effort="high",
    )

    # Round 1 downgraded to low; its audit was not clean-complete, so round 2
    # must not downgrade again even though the classifier stayed confident.
    assert len(low_exec.prompts) == 1
    assert len(default_exec.prompts) == 1
    assert len(scorer.calls) == 2, "each round classifies its own plan"

    rounds = _recorded_rounds(tmp_path)
    first = rounds[0]["executor_status"]["effort_routing"]
    assert first["variant"] == "low"
    assert first["used_default"] is False
    second = rounds[1]["executor_status"]["effort_routing"]
    assert second["variant"] == "high"
    assert second["used_default"] is True
    assert second["escalated"] is True
    assert second["classified"] == "mechanical"
    assert second["skipped"] == "escalated_to_default"
