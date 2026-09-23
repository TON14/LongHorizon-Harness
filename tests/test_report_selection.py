"""Scorer-based selection of related past audit reports: fake-scorer cascade.

The pointwise row shape, the threshold/K cascade, the explicit-reference
guarantee, and the round-record payload are all exercised through a fake
scorer -- never the real shim, so the suite stays hermetic. The wiring tests
drive the full manager loop with fake agents and assert the prompt and
record effects end to end.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

import lhht.report_selection as selection_module
from lhht.config import ProjectConfigError, _flatten_run_table
from lhht.environment.local import LocalEnvironment
from lhht.manager import run
from lhht.report_selection import (
    DEFAULT_REPORT_SELECTION_K,
    DEFAULT_REPORT_SELECTION_THRESHOLD,
    ReportCandidate,
    report_candidates,
    report_selector_from_defaults,
    select_related_reports,
    select_reports,
    selection_record,
)
from lhht.role_prompts import format_related_auditor_reports
from lhht.types import EpisodeResult, HarnessConfig, ManagedRound

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

_ROW_ID_RE = re.compile(r"Past audit report (round_\d+):")


class FakeScorer:
    """One P(relevant) per candidate, resolved from the id embedded in the row.

    ``resolver(candidate_id)`` returns P(relevant), or ``None`` to simulate
    the CLI scorer's timeout/unusable answer.
    """

    def __init__(self, resolver: Any) -> None:
        self._resolver = resolver
        self.calls: list[tuple[str, str, list[str]]] = []

    def score(self, state: str, question: str, options: list[dict[str, str]]):
        self.calls.append((state, question, [option["id"] for option in options]))
        match = _ROW_ID_RE.search(state)
        if match is None:  # pragma: no cover - malformed row
            return None
        probability = self._resolver(match.group(1))
        if probability is None:
            return None
        return [float(probability), 1.0 - float(probability)]


class RaisingScorer:
    def __init__(self) -> None:
        self.calls = 0

    def score(self, state: str, question: str, options: list[dict[str, str]]):
        self.calls += 1
        raise RuntimeError("scorer exploded")


class SequencedAgent:
    """Replays a fixed reply sequence and records the prompts it received."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.prompts: list[str] = []

    async def run_episode(self, prompt, _env, _budget, live_trajectory_path=None):
        self.prompts.append(str(prompt))
        reply = self._replies.pop(0) if self._replies else "Next: done"
        return EpisodeResult(status="done", actions_log=reply)


def _candidates(*ids: str) -> list[ReportCandidate]:
    return [ReportCandidate(item, f"audit report body for {item}") for item in ids]


def _selector(
    scorer: Any,
    *,
    k: int = DEFAULT_REPORT_SELECTION_K,
    threshold: float = DEFAULT_REPORT_SELECTION_THRESHOLD,
) -> selection_module.ReportSelector:
    return selection_module.ReportSelector(scorer, k, float(threshold))


# ---------------------------------------------------------------------------
# config: [run.semif] keys
# ---------------------------------------------------------------------------


def test_semif_report_selection_keys_flatten_into_defaults() -> None:
    defaults = _flatten_run_table(
        {
            "semif": {
                "report_selection": True,
                "report_selection_k": 5,
                "report_selection_threshold": 0.7,
            }
        }
    )

    assert defaults["semif_report_selection"] is True
    assert defaults["semif_report_selection_k"] == 5
    assert defaults["semif_report_selection_threshold"] == 0.7


def test_absent_semif_report_selection_keys_leave_defaults_empty() -> None:
    for defaults in (
        _flatten_run_table({}),
        _flatten_run_table({"semif": {"enabled": False}}),
        _flatten_run_table({"semif": {"command": "scripts/semif_shim.bat"}}),
    ):
        assert "semif_report_selection" not in defaults
        assert "semif_report_selection_k" not in defaults
        assert "semif_report_selection_threshold" not in defaults


def test_unknown_semif_report_selection_key_is_refused() -> None:
    with pytest.raises(ProjectConfigError, match="unknown \\[run.semif\\] key"):
        _flatten_run_table({"semif": {"report_selection_x": True}})


def test_report_selection_requires_a_bool() -> None:
    with pytest.raises(ProjectConfigError, match="report_selection"):
        _flatten_run_table({"semif": {"report_selection": "yes"}})


@pytest.mark.parametrize("bad", [0, -1, True, "3", 1.5])
def test_bad_report_selection_k_is_refused(bad: Any) -> None:
    with pytest.raises(ProjectConfigError, match="report_selection_k"):
        _flatten_run_table({"semif": {"report_selection_k": bad}})


@pytest.mark.parametrize("bad", [0, 1.5, True, "high"])
def test_bad_report_selection_threshold_is_refused(bad: Any) -> None:
    with pytest.raises(ProjectConfigError, match="report_selection_threshold"):
        _flatten_run_table({"semif": {"report_selection_threshold": bad}})


# ---------------------------------------------------------------------------
# selector resolution
# ---------------------------------------------------------------------------


def test_selector_off_for_absent_or_disabled_config_without_building_a_scorer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _no_scorer(_defaults: Any) -> None:  # pragma: no cover - tripwire
        raise AssertionError("no scorer may be constructed for selection off")

    monkeypatch.setattr(selection_module, "scorer_from_config", _no_scorer)

    assert report_selector_from_defaults({}) is None
    assert report_selector_from_defaults({"semif_report_selection": False}) is None


def test_selector_silently_off_when_the_scorer_is_not_configured() -> None:
    # scorer_from_config itself degrades to None without command/model/revision.
    assert report_selector_from_defaults({"semif_report_selection": True}) is None


def test_selector_resolves_scorer_k_and_threshold_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()
    monkeypatch.setattr(
        selection_module, "scorer_from_config", lambda defaults: sentinel
    )

    selector = report_selector_from_defaults(
        {
            "semif_report_selection": True,
            "semif_report_selection_k": 5,
            "semif_report_selection_threshold": 0.75,
        }
    )

    assert selector is not None
    assert selector.scorer is sentinel
    assert selector.k == 5
    assert selector.threshold == 0.75


def test_selector_defaults_k_3_threshold_0_6_and_degrades_on_garbage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        selection_module, "scorer_from_config", lambda defaults: object()
    )

    selector = report_selector_from_defaults({"semif_report_selection": True})
    assert selector is not None
    assert selector.k == DEFAULT_REPORT_SELECTION_K == 3
    assert selector.threshold == DEFAULT_REPORT_SELECTION_THRESHOLD == 0.6

    for bad in (True, "3", 0, 2.5, None):
        degraded = report_selector_from_defaults(
            {"semif_report_selection": True, "semif_report_selection_k": bad}
        )
        assert degraded is not None and degraded.k == 3

    for bad in (True, "0.6", None):
        degraded = report_selector_from_defaults(
            {
                "semif_report_selection": True,
                "semif_report_selection_threshold": bad,
            }
        )
        assert degraded is not None and degraded.threshold == 0.6


# ---------------------------------------------------------------------------
# candidates from recorded rounds
# ---------------------------------------------------------------------------


def test_report_candidates_cover_rounds_with_reports_only() -> None:
    rounds = [
        ManagedRound(round_index=1, next_step="cli", plan_text="p", auditor_report="r1"),
        ManagedRound(round_index=2, next_step="cli", plan_text="p", auditor_report="  "),
        ManagedRound(round_index=3, next_step="cli", plan_text="p", auditor_report="r3"),
    ]

    candidates = report_candidates(rounds)

    assert [(item.id, item.text) for item in candidates] == [
        ("round_001", "r1"),
        ("round_003", "r3"),
    ]


# ---------------------------------------------------------------------------
# select_reports: the pointwise cascade
# ---------------------------------------------------------------------------


def test_confident_split_keeps_top_k_ordered_by_probability() -> None:
    scorer = FakeScorer(
        {
            "round_001": 0.95,
            "round_002": 0.9,
            "round_003": 0.85,
            "round_004": 0.8,
            "round_005": 0.05,
        }.get
    )

    selection = select_reports(
        scorer, "the plan", _candidates(*[f"round_00{i}" for i in range(1, 6)]), 3, 0.6
    )

    assert [item.id for item in selection.kept] == [
        "round_001",
        "round_002",
        "round_003",
    ]
    assert selection.degraded is False
    # The threshold-passing but over-K candidate and the irrelevant one are
    # both dropped, each with its recorded probability.
    assert [item.id for item in selection.dropped] == ["round_004", "round_005"]
    assert selection.probabilities["round_004"] == pytest.approx(0.8)
    assert selection.probabilities["round_005"] == pytest.approx(0.05)


def test_ties_keep_candidate_order() -> None:
    scorer = FakeScorer(lambda _id: 0.9)

    selection = select_reports(
        scorer, "plan", _candidates("round_001", "round_002", "round_003"), 3, 0.6
    )

    assert [item.id for item in selection.kept] == ["round_001", "round_002", "round_003"]


def test_threshold_is_inclusive() -> None:
    scorer = FakeScorer(lambda _id: 0.6)

    selection = select_reports(scorer, "plan", _candidates("round_001"), 3, 0.6)

    assert [item.id for item in selection.kept] == ["round_001"]


def test_all_below_threshold_drops_everything_scored() -> None:
    scorer = FakeScorer(lambda _id: 0.59)

    selection = select_reports(
        scorer, "plan", _candidates("round_001", "round_002"), 3, 0.6
    )

    assert selection.kept == []
    assert [item.id for item in selection.dropped] == ["round_001", "round_002"]
    assert selection.degraded is False


def test_scorer_error_returns_all_candidates_unchanged() -> None:
    scorer = RaisingScorer()
    candidates = _candidates("round_001", "round_002")

    selection = select_reports(scorer, "plan", candidates, 3, 0.6)

    assert selection.kept == candidates
    assert selection.dropped == []
    assert selection.probabilities == {}
    assert selection.degraded is True
    assert scorer.calls == 1


def test_scorer_timeout_answer_returns_all_candidates_unchanged() -> None:
    scorer = FakeScorer(lambda _id: None)
    candidates = _candidates("round_001", "round_002")

    selection = select_reports(scorer, "plan", candidates, 3, 0.6)

    assert selection.kept == candidates
    assert selection.degraded is True


def test_one_failing_row_degrades_the_whole_selection() -> None:
    scorer = FakeScorer({"round_001": 0.95}.get)  # round_002 resolves to None
    candidates = _candidates("round_001", "round_002")

    selection = select_reports(scorer, "plan", candidates, 3, 0.6)

    assert selection.kept == candidates
    assert selection.degraded is True
    assert len(scorer.calls) == 2


def test_malformed_probabilities_return_all_candidates_unchanged() -> None:
    class MalformedScorer:
        def score(self, state, question, options):
            return ["not-a-number", "also-not-a-number"]

    candidates = _candidates("round_001")

    selection = select_reports(MalformedScorer(), "plan", candidates, 3, 0.6)

    assert selection.kept == candidates
    assert selection.degraded is True


def test_empty_candidates_return_unchanged_without_scorer_calls() -> None:
    def _explode(_id: str) -> None:  # pragma: no cover - tripwire
        raise AssertionError("empty candidates must not reach the scorer")

    selection = select_reports(FakeScorer(_explode), "plan", [], 3, 0.6)

    assert selection.kept == []
    assert selection.dropped == []
    assert selection.degraded is False


def test_each_row_carries_plan_candidate_question_and_options() -> None:
    scorer = FakeScorer(lambda _id: 0.9)
    plan = "Next: cli\n\nTask: fix the flaky retry test\n"

    select_reports(scorer, plan, _candidates("round_001", "round_002"), 3, 0.6)

    assert len(scorer.calls) == 2
    states = [call[0] for call in scorer.calls]
    # All rows share the plan state and differ only by the embedded candidate.
    assert all(state.startswith("Current subtask plan:\n" + plan) for state in states)
    assert "Past audit report round_001:\naudit report body for round_001" in states[0]
    assert "Past audit report round_002:\naudit report body for round_002" in states[1]
    for _state, question, option_ids in scorer.calls:
        assert question == "Is this past audit report relevant to the current subtask?"
        assert option_ids == ["relevant", "irrelevant"]


def test_overlong_plan_and_reports_are_condensed_head_and_tail() -> None:
    scorer = FakeScorer(lambda _id: 0.9)
    plan = "P" * 20_000
    report = "H" * 5_000 + "\n" + "T" * 5_000

    select_reports(
        scorer,
        plan,
        [ReportCandidate("round_001", report)],
        3,
        0.6,
    )

    state = scorer.calls[0][0]
    assert len(state) < 20_000, "the row must stay bounded"
    # Head and tail of each long input survive; the middle is dropped.
    assert "Current subtask plan:\n" + "P" * 2_000 in state
    assert "H" * 1_000 in state
    assert "T" * 500 in state
    assert "P" * 2_601 not in state
    assert "H" * 1_301 not in state
    assert "...[truncated" in state


# ---------------------------------------------------------------------------
# select_related_reports: explicit references always win
# ---------------------------------------------------------------------------


def test_explicit_refs_survive_even_when_scored_irrelevant() -> None:
    scorer = FakeScorer(
        {
            "round_001": 0.05,
            "round_002": 0.9,
            "round_003": 0.85,
            "round_004": 0.1,
        }.get
    )

    selection = select_related_reports(
        _selector(scorer),
        "the plan",
        _candidates("round_001", "round_002", "round_003", "round_004"),
        ["round_001"],
    )

    assert [item.id for item in selection.kept] == [
        "round_001",
        "round_002",
        "round_003",
    ]
    assert [item.id for item in selection.dropped] == ["round_004"]
    assert selection.degraded is False
    # The explicit ref was still scored, so its probability is recorded.
    assert selection.probabilities["round_001"] == pytest.approx(0.05)


def test_unreferenced_top_k_joins_the_explicit_refs() -> None:
    scorer = FakeScorer(
        {
            "round_001": 0.05,  # explicit, irrelevant to the scorer
            "round_002": 0.95,
            "round_003": 0.9,
            "round_004": 0.05,
        }.get
    )

    selection = select_related_reports(
        _selector(scorer, k=1),
        "the plan",
        _candidates("round_001", "round_002", "round_003", "round_004"),
        ["round_001"],
    )

    assert [item.id for item in selection.kept] == ["round_001", "round_002"]


def test_everything_below_threshold_keeps_exactly_today_ref_set() -> None:
    scorer = FakeScorer(lambda _id: 0.2)

    selection = select_related_reports(
        _selector(scorer),
        "the plan",
        _candidates("round_001", "round_002", "round_003"),
        ["round_002"],
    )

    assert [item.id for item in selection.kept] == ["round_002"]
    assert [item.id for item in selection.dropped] == ["round_001", "round_003"]
    assert selection.degraded is False


def test_scorer_failure_keeps_exactly_today_ref_set() -> None:
    scorer = FakeScorer(lambda _id: None)

    selection = select_related_reports(
        _selector(scorer),
        "the plan",
        _candidates("round_001", "round_002"),
        ["round_002"],
    )

    assert [item.id for item in selection.kept] == ["round_002"]
    assert selection.dropped == []
    assert selection.degraded is True


def test_empty_candidates_keep_today_empty_set() -> None:
    def _explode(_id: str) -> None:  # pragma: no cover - tripwire
        raise AssertionError("empty candidates must not reach the scorer")

    selection = select_related_reports(
        _selector(FakeScorer(_explode)), "plan", [], ["round_001"]
    )

    assert selection.kept == []
    assert selection.degraded is False


# ---------------------------------------------------------------------------
# the recorded payload
# ---------------------------------------------------------------------------


def test_selection_record_lists_kept_and_dropped_with_probabilities() -> None:
    scorer = FakeScorer({"round_001": 0.05, "round_002": 0.95, "round_003": 0.1}.get)

    selection = select_related_reports(
        _selector(scorer),
        "the plan",
        _candidates("round_001", "round_002", "round_003"),
        ["round_001"],
    )
    payload = selection_record(selection, refs=["round_001"], k=3, threshold=0.6)

    assert payload["kept"] == [
        {"id": "round_001", "probability": pytest.approx(0.05)},
        {"id": "round_002", "probability": pytest.approx(0.95)},
    ]
    assert payload["dropped"] == [{"id": "round_003", "probability": pytest.approx(0.1)}]
    assert payload["explicit_refs"] == ["round_001"]
    assert payload["degraded"] is False
    assert payload["k"] == 3
    assert payload["threshold"] == 0.6


def test_selection_record_marks_degradation_without_probabilities() -> None:
    scorer = FakeScorer(lambda _id: None)

    selection = select_related_reports(
        _selector(scorer),
        "the plan",
        _candidates("round_001", "round_002"),
        ["round_002"],
    )
    payload = selection_record(selection, refs=["round_002"], k=3, threshold=0.6)

    assert payload["degraded"] is True
    assert payload["kept"] == [{"id": "round_002", "probability": None}]
    assert payload["dropped"] == []


# ---------------------------------------------------------------------------
# Wiring: the manager loop with fake agents
# ---------------------------------------------------------------------------

_CLEAN_AUDIT = (
    "Status: complete\nIntegrity: clean\nContract audit: aligned\n\n"
    "Summary:\nthe change is verified"
)
_PLAN_ROUND1 = (
    "Next: cli\n\n"
    "Current Task State:\nround 1 in flight\n\n"
    "Task contract:\n"
    "Acceptance constraints:\n"
    "1. The deliverable must exist in the workspace\n"
)
_PLAN_ROUND2_REFERENCING = (
    "Next: cli\n\n"
    "Current Task State:\nround 2 builds on the audited round\n\n"
    "Task contract:\n"
    "Acceptance constraints:\n"
    "1. The deliverable must exist in the workspace\n\n"
    "Related audit reports: round_001 — its audit certified the baseline.\n"
)
_DONE = "Next: done\n\nCurrent Task State:\nall finished"
_EXECUTOR = "executor output: the change is in"


def _harness_config(tmp_path: Path) -> HarnessConfig:
    return HarnessConfig(
        max_total_episodes=3,
        workspace_path=str(tmp_path / "workspace"),
        harness_dir=str(tmp_path / "harness"),
        log_dir=str(tmp_path / "logs"),
    )


def _recorded_rounds(tmp_path: Path) -> list[dict[str, Any]]:
    path = tmp_path / "logs" / "role_orchestration" / "rounds.jsonl"
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


async def _run_referencing_flow(
    tmp_path: Path,
    manager_replies: list[str],
    *,
    executor_replies: list[str],
    audits: list[str],
) -> tuple[dict[str, Any], SequencedAgent, SequencedAgent]:
    manager = SequencedAgent(manager_replies)
    executor = SequencedAgent(executor_replies)
    auditor = SequencedAgent(audits)

    report = await run(
        task="finish the feature",
        env=LocalEnvironment(str(tmp_path / "tmp")),
        config=_harness_config(tmp_path),
        manager_agent=manager,
        gui_executor_agent=executor,
        cli_executor_agent=executor,
        gui_auditor_agent=auditor,
        cli_auditor_agent=auditor,
    )
    return report, executor, auditor


def _disable_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(selection_module, "load_run_defaults", lambda: {})

    def _no_scorer(_defaults: Any) -> None:  # pragma: no cover - tripwire
        raise AssertionError("no scorer may be constructed for selection off")

    monkeypatch.setattr(selection_module, "scorer_from_config", _no_scorer)


def _enable_selection(monkeypatch: pytest.MonkeyPatch, scorer: Any) -> None:
    monkeypatch.setattr(
        selection_module,
        "load_run_defaults",
        lambda: {
            "semif_report_selection": True,
            "semif_enabled": True,
            "semif_command": "scripts/semif_shim.bat",
            "semif_model": "m",
            "semif_revision": "r",
        },
    )
    monkeypatch.setattr(
        selection_module, "scorer_from_config", lambda defaults: scorer
    )


@pytest.mark.asyncio
async def test_disabled_selection_is_byte_identical_and_calls_no_scorer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _disable_selection(monkeypatch)

    report, executor, _auditor = await _run_referencing_flow(
        tmp_path,
        [_PLAN_ROUND1, _PLAN_ROUND2_REFERENCING, _DONE],
        executor_replies=[_EXECUTOR, _EXECUTOR],
        audits=[_CLEAN_AUDIT, _CLEAN_AUDIT],
    )

    assert report["completion_satisfied"] is True
    # Round 2's executor prompt carries exactly the section today's
    # ref-driven formatter produces for the same recorded round and refs.
    recorded = _recorded_rounds(tmp_path)
    round_one = recorded[0]
    today_section = format_related_auditor_reports(
        [
            ManagedRound(
                round_index=round_one["round_index"],
                next_step=round_one["next_step"],
                plan_text=round_one["plan_text"],
                auditor_report=round_one["auditor_report"],
            )
        ],
        recorded[1]["related_report_refs"],
        max_chars=60_000,
        language="en",
    ).strip()
    assert today_section in executor.prompts[1]
    # No scorer was ever constructed or called, and no record carries the
    # selection payload.
    assert all("report_selection" not in item["executor_status"] for item in recorded)


@pytest.mark.asyncio
async def test_enabled_selection_keeps_scored_reference_in_prompt_and_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scorer = FakeScorer({"round_001": 0.95}.get)
    _enable_selection(monkeypatch, scorer)

    report, executor, _auditor = await _run_referencing_flow(
        tmp_path,
        [_PLAN_ROUND1, _PLAN_ROUND2_REFERENCING, _DONE],
        executor_replies=[_EXECUTOR, _EXECUTOR],
        audits=[_CLEAN_AUDIT, _CLEAN_AUDIT],
    )

    assert report["completion_satisfied"] is True
    # Round 1 had no past reports, so the scorer first fires in round 2,
    # exactly once for the single candidate.
    assert len(scorer.calls) == 1
    assert "Past audit report round_001:" in scorer.calls[0][0]
    assert "--- Round 1 auditor report ---" in executor.prompts[1]

    recorded = _recorded_rounds(tmp_path)
    first, second = recorded[0], recorded[1]
    # Round 1 recorded an empty selection (nothing to rank yet)...
    assert first["executor_status"]["report_selection"]["kept"] == []
    # ...round 2 recorded the kept reference with its probability.
    selection = second["executor_status"]["report_selection"]
    assert [item["id"] for item in selection["kept"]] == ["round_001"]
    assert selection["kept"][0]["probability"] == pytest.approx(0.95)
    assert selection["dropped"] == []
    assert selection["explicit_refs"] == ["round_001"]
    assert selection["degraded"] is False
    assert selection["k"] == 3
    assert selection["threshold"] == 0.6


@pytest.mark.asyncio
async def test_explicit_reference_stays_in_prompt_when_scored_irrelevant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scorer = FakeScorer({"round_001": 0.05}.get)
    _enable_selection(monkeypatch, scorer)

    _report, executor, _auditor = await _run_referencing_flow(
        tmp_path,
        [_PLAN_ROUND1, _PLAN_ROUND2_REFERENCING, _DONE],
        executor_replies=[_EXECUTOR, _EXECUTOR],
        audits=[_CLEAN_AUDIT, _CLEAN_AUDIT],
    )

    # Explicit references always win: the scored-irrelevant report still
    # rides along in the prompt, exactly like today.
    assert "--- Round 1 auditor report ---" in executor.prompts[1]
    selection = _recorded_rounds(tmp_path)[1]["executor_status"]["report_selection"]
    assert [item["id"] for item in selection["kept"]] == ["round_001"]
    assert selection["kept"][0]["probability"] == pytest.approx(0.05)
    assert selection["degraded"] is False


@pytest.mark.asyncio
async def test_scorer_failure_in_the_loop_keeps_today_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scorer = FakeScorer(lambda _id: None)
    _enable_selection(monkeypatch, scorer)

    _report, executor, _auditor = await _run_referencing_flow(
        tmp_path,
        [_PLAN_ROUND1, _PLAN_ROUND2_REFERENCING, _DONE],
        executor_replies=[_EXECUTOR, _EXECUTOR],
        audits=[_CLEAN_AUDIT, _CLEAN_AUDIT],
    )

    assert "--- Round 1 auditor report ---" in executor.prompts[1]
    selection = _recorded_rounds(tmp_path)[1]["executor_status"]["report_selection"]
    assert selection["degraded"] is True
    assert [item["id"] for item in selection["kept"]] == ["round_001"]
    assert selection["dropped"] == []
