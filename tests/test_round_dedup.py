"""The round-dedup detector: an advisory flag for re-planned subtasks.

Each round's manager plan is compared with the previous round's plan by one
scored decision; a confident "essentially the same work" marks the round as a
possible re-planning loop. These tests drive the comparison with fake
scorers -- never the real shim -- and the wiring tests run the full manager
loop with fake agents to assert the advisory-only contract end to end:
routing, feedback, the plan text, and the manager's own output records never
change in any case.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import lhht.manager as manager_module
from lhht.auditor_agent import parse_audit_report
from lhht.config import ProjectConfigError, _flatten_run_table
from lhht.environment.local import LocalEnvironment
from lhht.manager import run
from lhht.role_prompts import extract_role_manager_plan_text
from lhht.round_dedup import (
    DEFAULT_REPEAT_THRESHOLD,
    REPEAT_QUESTION,
    detect_repeat,
    resolve_round_dedup,
    round_dedup_from_defaults,
)
from lhht.types import EpisodeResult, HarnessConfig, ManagedRound

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeScorer:
    """One canned distribution per call, resolved against (question, state).

    Returning ``None`` simulates the CLI scorer's timeout/unusable answer.
    Calls are recorded so tests can assert exactly which rounds consulted
    the scorer and what each comparison saw.
    """

    def __init__(self, resolver: Any = None) -> None:
        self._resolver = resolver
        self.calls: list[tuple[str, str]] = []

    def score(self, state: str, question: str, options: list[dict[str, str]]):
        self.calls.append((question, state))
        if self._resolver is None:
            return None
        mapping = self._resolver(question, state)
        if mapping is None:
            return None
        return [float(mapping.get(option["id"], 0.0)) for option in options]


class RaisingScorer:
    def __init__(self) -> None:
        self.calls = 0

    def score(self, state: str, question: str, options: list[dict[str, str]]):
        self.calls += 1
        raise RuntimeError("scorer exploded")


class RawScorer:
    """Returns a fixed answer object verbatim, however malformed."""

    def __init__(self, answer: Any) -> None:
        self._answer = answer

    def score(self, state: str, question: str, options: list[dict[str, str]]):
        return self._answer


def _same_work(question: str, state: str):
    return {"same_work": 0.97, "different_work": 0.03}


def _different_work(question: str, state: str):
    return {"different_work": 0.95, "same_work": 0.05}


def _same_work_except_done_plans(question: str, state: str):
    # Confidently same only for the working-plan comparisons; a completion
    # plan is genuinely different work.
    current = state.split("This round's manager plan:\n", 1)[1]
    if "Next: done" in current:
        return {"different_work": 0.95, "same_work": 0.05}
    return {"same_work": 0.97, "different_work": 0.03}


# ---------------------------------------------------------------------------
# detect_repeat: decision semantics
# ---------------------------------------------------------------------------


def test_confident_same_work_flags_with_the_comparison_probability() -> None:
    result = detect_repeat(
        FakeScorer(_same_work), "plan: fix the tests", "plan: fix the tests", 0.9
    )

    assert result is not None
    assert result.same is True
    assert result.flagged is True
    assert result.probability == pytest.approx(0.97)
    assert result.probabilities == {"same_work": 0.97, "different_work": 0.03}
    assert result.threshold == 0.9
    payload = result.payload()
    assert payload["flagged"] is True
    assert payload["same"] is True
    assert payload["probability"] == pytest.approx(0.97)
    assert payload["probabilities"] == {"same_work": 0.97, "different_work": 0.03}
    assert payload["threshold"] == 0.9


def test_different_work_records_without_flagging() -> None:
    result = detect_repeat(
        FakeScorer(_different_work), "plan: fix the tests", "plan: write the docs", 0.9
    )

    assert result is not None
    assert result.same is False
    assert result.flagged is False
    # The measurement survives: how much mass the scorer gave "same".
    assert result.probability == pytest.approx(0.05)


def test_below_threshold_same_work_records_without_flagging() -> None:
    scorer = FakeScorer(lambda question, state: {"same_work": 0.7, "different_work": 0.3})

    result = detect_repeat(scorer, "plan: fix the tests", "plan: fix the tests", 0.9)

    assert result is not None
    assert result.same is False
    assert result.flagged is False
    assert result.probability == pytest.approx(0.7)


def test_threshold_is_inclusive() -> None:
    scorer = FakeScorer(lambda question, state: {"same_work": 0.90, "different_work": 0.10})

    result = detect_repeat(scorer, "plan: a", "plan: a", 0.9)

    assert result is not None
    assert result.same is True
    assert result.probability == pytest.approx(0.90)


def test_tie_goes_to_the_earliest_option() -> None:
    scorer = FakeScorer(lambda question, state: {"same_work": 0.5, "different_work": 0.5})

    result = detect_repeat(scorer, "plan: a", "plan: b", 0.5)

    assert result is not None
    assert result.same is True


def test_one_decision_with_labeled_condensed_separated_plans() -> None:
    scorer = FakeScorer(_same_work)
    long_plan = "fix the tests  " + "and the suite " * 400

    result = detect_repeat(scorer, long_plan + "\n\n", "previous:\t" + long_plan, 0.9)

    assert result is not None
    assert len(scorer.calls) == 1, "exactly one scored decision"
    question, state = scorer.calls[0]
    assert question == (
        "Is this round's planned subtask essentially the same work as the "
        "previous round's?"
    )
    assert question == REPEAT_QUESTION
    # Clearly separated, labeled, and condensed: whitespace collapses and each
    # plan is clipped to the bounded slice the local model can judge.
    previous_part, current_part = state.split("\n\n")
    assert previous_part.startswith("Previous round's manager plan:\n")
    assert current_part.startswith("This round's manager plan:\n")
    assert previous_part.endswith("…")
    assert current_part.endswith("…")
    assert len(previous_part) <= len("Previous round's manager plan:\n") + 1_500
    assert len(current_part) <= len("This round's manager plan:\n") + 1_500


def test_options_are_same_work_and_different_work() -> None:
    scorer = FakeScorer(_same_work)

    detect_repeat(scorer, "plan: a", "plan: b", 0.9)

    # The fake resolves distributions by option id, so a correct shape here
    # means the decision offered exactly the two specified options.
    assert scorer.calls


def test_missing_previous_plan_returns_none_without_scoring() -> None:
    scorer = FakeScorer(_same_work)  # pragma: no cover - tripwire shape

    assert detect_repeat(scorer, "plan: a", "", 0.9) is None
    assert detect_repeat(scorer, "plan: a", "   \n  ", 0.9) is None
    assert scorer.calls == []


def test_missing_current_plan_returns_none_without_scoring() -> None:
    scorer = FakeScorer(_same_work)

    assert detect_repeat(scorer, "", "plan: b", 0.9) is None
    assert detect_repeat(scorer, None, "plan: b", 0.9) is None  # type: ignore[arg-type]
    assert scorer.calls == []


def test_missing_scorer_returns_none() -> None:
    assert detect_repeat(None, "plan: a", "plan: b", 0.9) is None


def test_raising_scorer_returns_none() -> None:
    scorer = RaisingScorer()

    assert detect_repeat(scorer, "plan: a", "plan: b", 0.9) is None
    assert scorer.calls == 1


def test_unusable_scorer_answers_return_none() -> None:
    assert detect_repeat(FakeScorer(), "plan: a", "plan: b", 0.9) is None
    malformed = FakeScorer(lambda question, state: {"same_work": "not-a-number"})
    assert detect_repeat(malformed, "plan: a", "plan: b", 0.9) is None
    for answer in (None, [], [0.5], [0.5, 0.5, 0.5], "junk", {"same_work": 1.0}):
        assert detect_repeat(RawScorer(answer), "plan: a", "plan: b", 0.9) is None


# ---------------------------------------------------------------------------
# config: [run.semif] keys
# ---------------------------------------------------------------------------


def test_semif_round_dedup_keys_flatten_into_defaults() -> None:
    defaults = _flatten_run_table(
        {
            "semif": {
                "enabled": True,
                "command": "scripts/semif_shim.bat",
                "model": "m",
                "revision": "r",
                "round_dedup": True,
                "round_dedup_threshold": 0.85,
            }
        }
    )

    assert defaults["semif_round_dedup"] is True
    assert defaults["semif_round_dedup_threshold"] == 0.85


def test_absent_round_dedup_keys_leave_no_defaults() -> None:
    defaults = _flatten_run_table(
        {
            "semif": {
                "enabled": True,
                "command": "scripts/semif_shim.bat",
                "model": "m",
                "revision": "r",
            }
        }
    )

    assert "semif_round_dedup" not in defaults
    assert "semif_round_dedup_threshold" not in defaults


def test_unknown_semif_key_is_refused() -> None:
    with pytest.raises(ProjectConfigError, match="unknown \\[run.semif\\] key"):
        _flatten_run_table({"semif": {"round_dedupx": True}})


@pytest.mark.parametrize("bad", [0, 1.5, -0.1, True, "high"])
def test_bad_round_dedup_threshold_is_refused(bad: Any) -> None:
    with pytest.raises(ProjectConfigError, match="round_dedup_threshold"):
        _flatten_run_table({"semif": {"round_dedup_threshold": bad}})


def test_round_dedup_requires_a_bool() -> None:
    with pytest.raises(ProjectConfigError, match="round_dedup"):
        _flatten_run_table({"semif": {"round_dedup": "yes"}})


# ---------------------------------------------------------------------------
# resolvers: defaults -> (scorer, threshold)
# ---------------------------------------------------------------------------


def test_round_dedup_off_for_absent_or_disabled_config_without_building_a_scorer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _no_scorer(_defaults: Any) -> None:  # pragma: no cover - tripwire
        raise AssertionError("no scorer may be constructed for an off detector")

    monkeypatch.setattr("lhht.round_dedup.scorer_from_config", _no_scorer)

    assert round_dedup_from_defaults({}) == (None, DEFAULT_REPEAT_THRESHOLD)
    assert round_dedup_from_defaults({"semif_round_dedup": False}) == (
        None,
        DEFAULT_REPEAT_THRESHOLD,
    )


def test_round_dedup_silently_off_when_the_scorer_is_not_configured() -> None:
    assert round_dedup_from_defaults({"semif_round_dedup": True}) == (
        None,
        DEFAULT_REPEAT_THRESHOLD,
    )


def test_round_dedup_resolves_scorer_and_threshold_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()
    monkeypatch.setattr(
        "lhht.round_dedup.scorer_from_config", lambda defaults: sentinel
    )

    scorer, threshold = round_dedup_from_defaults(
        {
            "semif_round_dedup": True,
            "semif_round_dedup_threshold": 0.8,
        }
    )

    assert scorer is sentinel
    assert threshold == 0.8


@pytest.mark.parametrize("bad", [True, "high", None])
def test_bad_threshold_type_falls_back_to_the_default(
    monkeypatch: pytest.MonkeyPatch, bad: Any
) -> None:
    sentinel = object()
    monkeypatch.setattr(
        "lhht.round_dedup.scorer_from_config", lambda defaults: sentinel
    )

    scorer, threshold = round_dedup_from_defaults(
        {"semif_round_dedup": True, "semif_round_dedup_threshold": bad}
    )

    assert scorer is sentinel
    assert threshold == DEFAULT_REPEAT_THRESHOLD


def test_resolve_round_dedup_degrades_to_off_when_config_load_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom() -> None:
        raise RuntimeError("config unreadable")

    monkeypatch.setattr("lhht.round_dedup.load_run_defaults", _boom)

    assert resolve_round_dedup() == (None, DEFAULT_REPEAT_THRESHOLD)


# ---------------------------------------------------------------------------
# manager_status merge: byte-identical when there is nothing to attach
# ---------------------------------------------------------------------------


def test_with_round_dedup_attaches_only_a_payload() -> None:
    assert manager_module._with_round_dedup(None, None) == {}
    status = {"status": "done"}
    assert manager_module._with_round_dedup(status, None) == {"status": "done"}
    assert manager_module._with_round_dedup(status, None) is status
    merged = manager_module._with_round_dedup(status, {"flagged": True})
    assert merged == {"status": "done", "round_dedup": {"flagged": True}}
    assert status == {"status": "done"}, "the base status is never mutated"


# ---------------------------------------------------------------------------
# Wiring: the manager loop with fake agents
# ---------------------------------------------------------------------------

_PLAN_ROUND1 = (
    "Next: cli\n\n"
    "Current Task State:\nround 1 in flight\n\n"
    "Task contract:\n"
    "Acceptance constraints:\n"
    "1. The deliverable must exist in the workspace\n"
    "2. The suite must stay green\n\n"
    "Dependency assessment:\nnone\n"
)
# The same subtask, re-phrased: exactly the loop the detector exists to surface.
_PLAN_ROUND2_REPEAT = (
    "Next: cli\n\n"
    "Current Task State:\nround 2 in flight\n\n"
    "Task contract:\n"
    "Acceptance constraints:\n"
    "1. The deliverable must exist in the workspace\n"
    "2. The suite must stay green\n\n"
    "Dependency assessment:\nnone\n"
)
_EXECUTOR = "executor output: touched nothing, workspace unchanged"
_DONE = "Next: done\n\nCurrent Task State:\nall finished"
_CLEAN_AUDIT = (
    "Status: complete\nIntegrity: clean\nContract audit: aligned\n\n"
    "Summary:\nthe change is verified"
)


class SequencedAgent:
    """Replays a fixed reply sequence and records the prompts it received."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.prompts: list[str] = []

    async def run_episode(self, prompt, _env, _budget, live_trajectory_path=None):
        self.prompts.append(str(prompt))
        reply = self._replies.pop(0) if self._replies else _DONE
        return EpisodeResult(status="done", actions_log=reply)


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


def _harness_config(tmp_path: Path, *, max_total_episodes: int = 2) -> HarnessConfig:
    return HarnessConfig(
        max_total_episodes=max_total_episodes,
        workspace_path=str(tmp_path / "workspace"),
        harness_dir=str(tmp_path / "harness"),
        log_dir=str(tmp_path / "logs"),
    )


def _disable_other_semif_features(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(manager_module, "resolve_fast_gate", lambda: (None, 0.95))
    monkeypatch.setattr(
        manager_module,
        "resolve_cross_check",
        lambda: (None, 0.9),
    )


def _disable_round_dedup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        manager_module, "resolve_round_dedup", lambda: (None, DEFAULT_REPEAT_THRESHOLD)
    )


def _enable_round_dedup(
    monkeypatch: pytest.MonkeyPatch, scorer: Any, threshold: float = 0.9
) -> None:
    monkeypatch.setattr(
        manager_module, "resolve_round_dedup", lambda: (scorer, threshold)
    )


async def _run_three_round_scenario(tmp_path: Path) -> dict[str, Any]:
    """Two working rounds with essentially the same plan, then done."""
    return await run(
        task="finish the feature",
        env=LocalEnvironment(str(tmp_path / "tmp")),
        config=_harness_config(tmp_path, max_total_episodes=3),
        agent=SequencedAgent(
            [
                _PLAN_ROUND1,
                _EXECUTOR,
                _CLEAN_AUDIT,
                _PLAN_ROUND2_REPEAT,
                _EXECUTOR,
                _CLEAN_AUDIT,
                _DONE,
            ]
        ),
    )


async def _run_repeat_flow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scorer: Any
) -> dict[str, Any]:
    """The repeat scenario with the detector enabled for `scorer`."""
    _disable_other_semif_features(monkeypatch)
    _enable_round_dedup(monkeypatch, scorer)
    return await _run_three_round_scenario(tmp_path)


def _flag_events(tmp_path: Path) -> list[dict[str, Any]]:
    return [item for item in _events(tmp_path) if item["event"] == "round_dedup_flag"]


@pytest.mark.asyncio
async def test_confident_repeat_flags_the_record_and_emits_the_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scorer = FakeScorer(_same_work_except_done_plans)

    report = await _run_repeat_flow(tmp_path, monkeypatch, scorer)

    # Advisory-only: the flagged round still parses clean/complete, routing
    # still dispatched the executor, and done acceptance is unaffected.
    assert report["completion_satisfied"] is True
    recorded = _recorded_rounds(tmp_path)
    assert [item["round_index"] for item in recorded] == [1, 2, 3]
    round_two = recorded[1]
    assert round_two["next_step"] == "cli"
    assert round_two["plan_text"] == extract_role_manager_plan_text(
        _PLAN_ROUND2_REPEAT
    ).strip()
    assert round_two["harness_feedback"] == ""
    parsed = parse_audit_report(round_two["auditor_report"], 2)
    assert (parsed.status, parsed.integrity_status, parsed.contract_audit_status) == (
        "complete",
        "clean",
        "aligned",
    )
    assert "round_dedup" not in round_two["auditor_status"]
    assert "round_dedup" not in round_two["executor_status"]

    # The flag: round 2 (the repeated plan) is marked and surfaced, with the
    # comparison probability carried for measurement.
    payload = round_two["manager_status"]["round_dedup"]
    assert payload["flagged"] is True
    assert payload["same"] is True
    assert payload["probability"] == pytest.approx(0.97)
    assert payload["probabilities"] == {"same_work": 0.97, "different_work": 0.03}
    assert payload["threshold"] == 0.9

    flagged = _flag_events(tmp_path)
    assert [item["round"] for item in flagged] == [2]
    assert flagged[0]["flagged"] is True
    assert flagged[0]["probability"] == pytest.approx(0.97)

    # Rounds 1 (no previous plan) and 3 (a genuinely different done plan)
    # still scored -- every round with a previous plan is compared -- and
    # only round 2 was confidently same.
    assert len(scorer.calls) == 2
    assert "round_dedup" not in recorded[0]["manager_status"]
    assert recorded[2]["manager_status"]["round_dedup"]["same"] is False


@pytest.mark.asyncio
async def test_different_work_attaches_the_record_without_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = await _run_repeat_flow(tmp_path, monkeypatch, FakeScorer(_different_work))

    assert report["completion_satisfied"] is True
    assert _flag_events(tmp_path) == []
    recorded = _recorded_rounds(tmp_path)
    payload = recorded[1]["manager_status"]["round_dedup"]
    assert payload["flagged"] is False
    assert payload["same"] is False
    assert payload["probability"] == pytest.approx(0.05)
    assert payload["probabilities"]["different_work"] == pytest.approx(0.95)


@pytest.mark.asyncio
async def test_below_threshold_attaches_the_record_without_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scorer = FakeScorer(lambda question, state: {"same_work": 0.7, "different_work": 0.3})

    report = await _run_repeat_flow(tmp_path, monkeypatch, scorer)

    assert report["completion_satisfied"] is True
    assert _flag_events(tmp_path) == []
    payload = _recorded_rounds(tmp_path)[1]["manager_status"]["round_dedup"]
    assert payload["flagged"] is False
    assert payload["same"] is False
    assert payload["probability"] == pytest.approx(0.7)


@pytest.mark.asyncio
async def test_first_round_makes_no_scorer_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scorer = FakeScorer(_different_work)
    _disable_other_semif_features(monkeypatch)
    _enable_round_dedup(monkeypatch, scorer)

    report = await run(
        task="finish the feature",
        env=LocalEnvironment(str(tmp_path / "tmp")),
        config=_harness_config(tmp_path),
        agent=SequencedAgent([_PLAN_ROUND1, _EXECUTOR, _CLEAN_AUDIT, _DONE]),
    )

    assert report["completion_satisfied"] is True
    recorded = _recorded_rounds(tmp_path)
    # Round 1 has no previous plan: no scorer call and no annotation, exactly
    # today's record. The single call belongs to round 2's done plan.
    assert len(scorer.calls) == 1
    assert recorded[0]["manager_status"] == {}
    assert "round_dedup" not in recorded[0]["manager_status"]
    state = scorer.calls[0][1]
    assert "Previous round's manager plan:\n" in state
    assert "This round's manager plan:\n" in state
    assert "Next: done" in state.split("This round's manager plan:\n")[1]


@pytest.mark.asyncio
async def test_scorer_error_leaves_no_record_or_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scorer = RaisingScorer()

    report = await _run_repeat_flow(tmp_path, monkeypatch, scorer)

    # The run is untouched: every round still recorded, nothing annotated.
    assert report["completion_satisfied"] is True
    assert scorer.calls == 2
    assert _flag_events(tmp_path) == []
    for item in _recorded_rounds(tmp_path):
        assert "round_dedup" not in item["manager_status"]


@pytest.mark.asyncio
async def test_disabled_dedup_is_byte_identical_and_calls_no_scorer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _disable_other_semif_features(monkeypatch)
    _disable_round_dedup(monkeypatch)

    def _tripwire(*_args: Any, **_kwargs: Any) -> None:  # pragma: no cover - tripwire
        raise AssertionError("the disabled detector must not score anything")

    monkeypatch.setattr(manager_module, "detect_repeat", _tripwire)

    report = await _run_three_round_scenario(tmp_path)

    # Behavior identical to today: all rounds recorded, no detector
    # artifacts, and the manager_status dicts serialize exactly as they did
    # before the feature -- {} on these paths, with no new record keys.
    assert report["completion_satisfied"] is True
    assert _flag_events(tmp_path) == []
    recorded = _recorded_rounds(tmp_path)
    assert [item["round_index"] for item in recorded] == [1, 2, 3]
    expected_fields = {
        "round_index",
        "next_step",
        "plan_text",
        "executor_output",
        "auditor_report",
        "harness_feedback",
        "task_state",
        "task_contract",
        "related_report_refs",
        "manager_status",
        "executor_status",
        "auditor_status",
    }
    for item in recorded:
        assert set(item) == expected_fields, "no new record fields appeared"
        assert item["manager_status"] == {}
        assert "round_dedup" not in item["auditor_status"]
        assert "round_dedup" not in item["executor_status"]


def test_flagged_repeat_never_blocks_done_acceptance() -> None:
    # A round whose auditor report parses complete/clean/aligned satisfies
    # _latest_auditor_is_clean_complete even with a flagged round-dedup
    # payload sitting in manager_status.
    round_one = ManagedRound(
        round_index=1,
        next_step="cli",
        plan_text=_PLAN_ROUND1,
        auditor_report=_CLEAN_AUDIT,
        manager_status={
            "round_dedup": {
                "flagged": True,
                "same": True,
                "probability": 0.97,
                "probabilities": {"same_work": 0.97, "different_work": 0.03},
                "threshold": 0.9,
            }
        },
    )

    assert manager_module._latest_auditor_is_clean_complete([round_one]) is True
