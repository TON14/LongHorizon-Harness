"""The post-audit verdict cross-check: a second, advisory line of trust.

After the slow auditor's report is parsed, the scorer independently answers
the same three control verdicts against the same live evidence the pre-gate
gathers. These tests drive the comparison with fake scorers -- never the
real shim -- and the wiring tests run the full manager loop with fake agents
to assert the advisory-only contract end to end: verdicts, routing, and
completion acceptance never change in any case.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import lhht.manager as manager_module
from lhht.auditor_agent import parse_audit_report
from lhht.auditor_fast import (
    DEFAULT_CROSS_CHECK_THRESHOLD,
    GateEvidence,
    cross_check,
    cross_check_from_defaults,
)
from lhht.config import ProjectConfigError, _flatten_run_table
from lhht.environment.local import LocalEnvironment
from lhht.manager import run
from lhht.types import AuditReport, EpisodeResult, HarnessConfig, ManagedRound

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeScorer:
    """One canned distribution per question, resolved by substring match.

    The resolver maps a question fragment to ``{option_id: probability}``;
    returning ``None`` simulates the CLI scorer's timeout/unusable answer.
    States are recorded so tests can assert one evidence consumption.
    """

    def __init__(self, resolver: Any = None) -> None:
        self._resolver = resolver
        self.calls: list[str] = []
        self.states: list[str] = []

    def score(self, state: str, question: str, options: list[dict[str, str]]):
        self.calls.append(question)
        self.states.append(state)
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


def _agrees_with_clean_audit(question: str):
    lowered = question.lower()
    if "task status complete" in lowered:
        return {"complete": 0.96, "incomplete": 0.02, "blocked": 0.02}
    if "integrity status clean" in lowered:
        return {"clean": 0.97, "suspect": 0.02, "violation": 0.01}
    if "contract audit aligned" in lowered:
        return {"aligned": 0.95, "unknown": 0.03, "needs_revision": 0.01, "invalid": 0.01}
    return {}


def _disagrees_on_status(question: str):
    lowered = question.lower()
    if "task status complete" in lowered:
        return {"incomplete": 0.97, "complete": 0.02, "blocked": 0.01}
    return _agrees_with_clean_audit(question)


_CLEAN_AUDIT = (
    "Status: complete\nIntegrity: clean\nContract audit: aligned\n\n"
    "Summary:\nthe change is verified"
)


def _evidence() -> GateEvidence:
    return GateEvidence(
        vcs_status="## main\n M src/app.py",
        vcs_tool="git",
        vcs_status_available=True,
        executor_output="executor output text",
        plan_text="plan text for the round",
        task_contract="Task contract:\nAcceptance constraints:\n- do the thing",
        acceptance_constraints=["the deliverable must exist"],
    )


def _clean_report() -> AuditReport:
    return parse_audit_report(_CLEAN_AUDIT, 1)


def _harness_config(tmp_path: Path, workspace: str | None = None) -> HarnessConfig:
    return HarnessConfig(
        max_total_episodes=2,
        workspace_path=workspace or str(tmp_path / "workspace"),
        harness_dir=str(tmp_path / "harness"),
        log_dir=str(tmp_path / "logs"),
    )


# ---------------------------------------------------------------------------
# cross_check: comparison semantics
# ---------------------------------------------------------------------------


def test_agreement_records_all_controls_without_flags() -> None:
    scorer = FakeScorer(_agrees_with_clean_audit)

    result = cross_check(scorer, _evidence(), _clean_report(), 0.9)

    assert result is not None
    assert result.flagged is False
    assert [entry["control"] for entry in result.agreements] == [
        "status",
        "integrity",
        "contract",
    ]
    assert result.disagreements == []
    payload = result.payload()
    assert payload["flagged"] is False
    assert payload["threshold"] == 0.9
    assert len(payload["controls"]) == 3
    status_entry = payload["controls"][0]
    assert status_entry["auditor_verdict"] == "complete"
    assert status_entry["scorer_answer"] == "complete"
    assert status_entry["probabilities"] == {
        "complete": 0.96,
        "incomplete": 0.02,
        "blocked": 0.02,
    }
    assert status_entry["probability"] == pytest.approx(0.96)
    assert status_entry["auditor_probability"] == pytest.approx(0.96)


def test_confident_disagreement_is_flagged_with_both_sides_probabilities() -> None:
    scorer = FakeScorer(_disagrees_on_status)

    result = cross_check(scorer, _evidence(), _clean_report(), 0.9)

    assert result is not None
    assert result.flagged is True
    assert [entry["control"] for entry in result.disagreements] == ["status"]
    disagreement = result.disagreements[0]
    assert disagreement["auditor_verdict"] == "complete"
    assert disagreement["scorer_answer"] == "incomplete"
    assert disagreement["probability"] == pytest.approx(0.97)
    assert disagreement["auditor_probability"] == pytest.approx(0.02)
    assert set(disagreement["probabilities"]) == {"complete", "incomplete", "blocked"}
    # The controls that agreed are still full records, not just the flag.
    assert [entry["control"] for entry in result.agreements] == ["integrity", "contract"]


def test_below_threshold_disagreement_records_but_does_not_flag() -> None:
    def resolver(question: str):
        lowered = question.lower()
        if "task status complete" in lowered:
            return {"incomplete": 0.89, "complete": 0.06, "blocked": 0.05}
        return _agrees_with_clean_audit(question)

    result = cross_check(FakeScorer(resolver), _evidence(), _clean_report(), 0.9)

    assert result is not None
    assert result.flagged is False
    assert result.disagreements == []
    status_entry = next(
        entry for entry in result.controls if entry["control"] == "status"
    )
    assert status_entry["agrees"] is False
    assert status_entry["flagged"] is False
    assert status_entry["scorer_answer"] == "incomplete"
    # A low-confidence mismatch is neither agreement nor disagreement, but its
    # measurement data still reaches the record through controls.
    assert status_entry not in result.agreements
    assert [entry["control"] for entry in result.agreements] == ["integrity", "contract"]


def test_threshold_is_inclusive() -> None:
    def resolver(question: str):
        lowered = question.lower()
        if "task status complete" in lowered:
            return {"incomplete": 0.90, "complete": 0.05, "blocked": 0.05}
        return _agrees_with_clean_audit(question)

    result = cross_check(FakeScorer(resolver), _evidence(), _clean_report(), 0.9)

    assert result is not None
    assert result.flagged is True
    assert result.disagreements[0]["probability"] == pytest.approx(0.90)


def test_tie_goes_to_the_earliest_option() -> None:
    def resolver(question: str):
        lowered = question.lower()
        if "task status complete" in lowered:
            return {"complete": 0.5, "incomplete": 0.5, "blocked": 0.0}
        return _agrees_with_clean_audit(question)

    result = cross_check(FakeScorer(resolver), _evidence(), _clean_report(), 0.9)

    assert result is not None
    status_entry = result.controls[0]
    assert status_entry["scorer_answer"] == "complete"
    assert status_entry["agrees"] is True


def test_questions_use_the_auditor_legal_values_and_one_shared_state() -> None:
    scorer = FakeScorer(_agrees_with_clean_audit)

    result = cross_check(scorer, _evidence(), _clean_report(), 0.9)

    assert result is not None
    assert scorer.calls == [
        "Given the evidence, is the task status complete?",
        "Given the evidence, is the integrity status clean?",
        "Given the evidence, is the contract audit aligned?",
    ]
    by_control = {entry["control"]: entry for entry in result.controls}
    assert by_control["status"]["options"] == ["complete", "incomplete", "blocked"]
    assert by_control["integrity"]["options"] == ["clean", "suspect", "violation"]
    assert by_control["contract"]["options"] == [
        "aligned",
        "unknown",
        "needs_revision",
        "invalid",
    ]
    # The evidence is consumed as-is: all three questions saw the same state.
    assert len(set(scorer.states)) == 1
    assert "executor output text" in scorer.states[0]
    assert "M src/app.py" in scorer.states[0]


def test_raising_scorer_yields_no_record() -> None:
    scorer = RaisingScorer()

    assert cross_check(scorer, _evidence(), _clean_report(), 0.9) is None
    assert scorer.calls == 1, "the first failure stops the cross-check entirely"


def test_unusable_scorer_answers_yield_no_record() -> None:
    assert cross_check(FakeScorer(lambda question: None), _evidence(), _clean_report()) is None


def test_malformed_probabilities_yield_no_record() -> None:
    scorer = FakeScorer(lambda question: {"complete": "not-a-number"})
    assert cross_check(scorer, _evidence(), _clean_report(), 0.9) is None


def test_missing_scorer_yields_no_record() -> None:
    assert cross_check(None, _evidence(), _clean_report(), 0.9) is None


def test_verdict_outside_the_legal_values_yields_no_record() -> None:
    report = AuditReport(round_id="round_1", status="nonsense")

    assert cross_check(FakeScorer(_agrees_with_clean_audit), _evidence(), report) is None


def test_parsed_report_is_never_mutated() -> None:
    report = _clean_report()
    before = (
        report.status,
        report.integrity_status,
        report.contract_audit_status,
        report.report_text,
    )

    cross_check(FakeScorer(_disagrees_on_status), _evidence(), report, 0.9)

    assert (
        report.status,
        report.integrity_status,
        report.contract_audit_status,
        report.report_text,
    ) == before


# ---------------------------------------------------------------------------
# config: [run.semif] keys
# ---------------------------------------------------------------------------


def test_semif_cross_check_keys_flatten_into_defaults() -> None:
    defaults = _flatten_run_table(
        {
            "semif": {
                "enabled": True,
                "command": "scripts/semif_shim.bat",
                "model": "m",
                "revision": "r",
                "cross_check": True,
                "cross_check_threshold": 0.85,
            }
        }
    )

    assert defaults["semif_cross_check"] is True
    assert defaults["semif_cross_check_threshold"] == 0.85


def test_unknown_semif_key_is_refused() -> None:
    with pytest.raises(ProjectConfigError, match="unknown \\[run.semif\\] key"):
        _flatten_run_table({"semif": {"cross_checkk": True}})


@pytest.mark.parametrize("bad", [0, 1.5, True, "high"])
def test_bad_cross_check_threshold_is_refused(bad: Any) -> None:
    with pytest.raises(ProjectConfigError, match="cross_check_threshold"):
        _flatten_run_table({"semif": {"cross_check_threshold": bad}})


def test_cross_check_requires_a_bool() -> None:
    with pytest.raises(ProjectConfigError, match="cross_check"):
        _flatten_run_table({"semif": {"cross_check": "yes"}})


def test_cross_check_off_for_absent_or_disabled_config_without_building_a_scorer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _no_scorer(_defaults: Any) -> None:  # pragma: no cover - tripwire
        raise AssertionError("no scorer may be constructed for an off cross-check")

    monkeypatch.setattr("lhht.auditor_fast.scorer_from_config", _no_scorer)

    assert cross_check_from_defaults({}) == (None, DEFAULT_CROSS_CHECK_THRESHOLD)
    assert cross_check_from_defaults({"semif_cross_check": False}) == (
        None,
        DEFAULT_CROSS_CHECK_THRESHOLD,
    )


def test_cross_check_silently_off_when_the_scorer_is_not_configured() -> None:
    assert cross_check_from_defaults({"semif_cross_check": True}) == (
        None,
        DEFAULT_CROSS_CHECK_THRESHOLD,
    )


def test_cross_check_resolves_scorer_and_threshold_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()
    monkeypatch.setattr(
        "lhht.auditor_fast.scorer_from_config",
        lambda defaults: sentinel,
    )

    scorer, threshold = cross_check_from_defaults(
        {
            "semif_cross_check": True,
            "semif_cross_check_threshold": 0.8,
        }
    )

    assert scorer is sentinel
    assert threshold == 0.8


# ---------------------------------------------------------------------------
# Wiring: the manager loop with fake agents
# ---------------------------------------------------------------------------

_PLAN = (
    "Next: cli\n\n"
    "Current Task State:\nround 1 in flight\n\n"
    "Task contract:\n"
    "Acceptance constraints:\n"
    "1. The deliverable must exist in the workspace\n"
    "2. The suite must stay green\n\n"
    "Dependency assessment:\nnone\n"
)
_EXECUTOR = "executor output: touched nothing, workspace unchanged"
_DONE = "Next: done\n\nCurrent Task State:\nall finished"


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


def _disable_fast_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(manager_module, "resolve_fast_gate", lambda: (None, 0.95))


def _disable_cross_check(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        manager_module, "resolve_cross_check", lambda: (None, DEFAULT_CROSS_CHECK_THRESHOLD)
    )


def _enable_cross_check(
    monkeypatch: pytest.MonkeyPatch, scorer: Any, threshold: float = 0.9
) -> None:
    monkeypatch.setattr(
        manager_module, "resolve_cross_check", lambda: (scorer, threshold)
    )


@pytest.mark.asyncio
async def test_agreement_attached_to_the_round_without_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scorer = FakeScorer(_agrees_with_clean_audit)
    _disable_fast_gate(monkeypatch)
    _enable_cross_check(monkeypatch, scorer)

    report = await run(
        task="finish the feature",
        env=LocalEnvironment(str(tmp_path / "tmp")),
        config=_harness_config(tmp_path),
        agent=SequencedAgent([_PLAN, _EXECUTOR, _CLEAN_AUDIT, _DONE]),
    )

    # The auditor's own verdicts are unchanged and still satisfy done.
    assert report["completion_satisfied"] is True
    assert report["status"] == "complete"
    events = _events(tmp_path)
    assert [item for item in events if item["event"] == "auditor_role_done"]
    assert not [item for item in events if item["event"] == "auditor_cross_check"]

    round_one = _recorded_rounds(tmp_path)[0]
    payload = round_one["auditor_status"]["auditor_cross_check"]
    assert payload["flagged"] is False
    assert payload["threshold"] == 0.9
    assert [entry["control"] for entry in payload["agreements"]] == [
        "status",
        "integrity",
        "contract",
    ]
    assert payload["disagreements"] == []
    parsed = parse_audit_report(round_one["auditor_report"], 1)
    assert (parsed.status, parsed.integrity_status, parsed.contract_audit_status) == (
        "complete",
        "clean",
        "aligned",
    )
    assert scorer.calls, "the cross-check actually consulted the scorer"


@pytest.mark.asyncio
async def test_confident_disagreement_flags_the_record_and_emits_the_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scorer = FakeScorer(_disagrees_on_status)
    _disable_fast_gate(monkeypatch)
    _enable_cross_check(monkeypatch, scorer)
    emitted: list[tuple[str, dict[str, Any]]] = []

    def progress(event: str, payload: dict[str, Any]) -> None:
        emitted.append((event, payload))

    report = await run(
        task="finish the feature",
        env=LocalEnvironment(str(tmp_path / "tmp")),
        config=_harness_config(tmp_path),
        agent=SequencedAgent([_PLAN, _EXECUTOR, _CLEAN_AUDIT, _DONE]),
        progress=progress,
    )

    # Advisory-only: the auditor's complete verdict still satisfies done and
    # the role_done flow reports the auditor's own statuses, not the flag.
    assert report["completion_satisfied"] is True
    assert report["status"] == "complete"
    auditor_done = [
        payload
        for event, payload in emitted
        if event == "role_done" and payload.get("role") == "cli_auditor"
    ]
    assert auditor_done[0]["audit_status"] == "complete"
    assert auditor_done[0]["integrity_status"] == "clean"
    assert auditor_done[0]["contract_audit_status"] == "aligned"

    flagged = [item for item in _events(tmp_path) if item["event"] == "auditor_cross_check"]
    assert [item["round"] for item in flagged] == [1]
    event = flagged[0]
    assert event["flagged"] is True
    assert [entry["control"] for entry in event["disagreements"]] == ["status"]
    disagreement = event["disagreements"][0]
    assert disagreement["auditor_verdict"] == "complete"
    assert disagreement["scorer_answer"] == "incomplete"
    assert disagreement["probability"] == pytest.approx(0.97)
    assert disagreement["auditor_probability"] == pytest.approx(0.02)

    # The persisted rounds.jsonl record carries the same payload with both
    # sides' probabilities next to the auditor's own verdicts.
    round_one = _recorded_rounds(tmp_path)[0]
    payload = round_one["auditor_status"]["auditor_cross_check"]
    assert payload["flagged"] is True
    assert [entry["control"] for entry in payload["disagreements"]] == ["status"]
    assert payload["disagreements"][0] == disagreement
    parsed = parse_audit_report(round_one["auditor_report"], 1)
    assert (parsed.status, parsed.integrity_status, parsed.contract_audit_status) == (
        "complete",
        "clean",
        "aligned",
    )
    assert "invalid_completion" not in round_one["auditor_status"]


@pytest.mark.asyncio
async def test_below_threshold_disagreement_attaches_record_without_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def resolver(question: str):
        lowered = question.lower()
        if "task status complete" in lowered:
            return {"incomplete": 0.89, "complete": 0.06, "blocked": 0.05}
        return _agrees_with_clean_audit(question)

    _disable_fast_gate(monkeypatch)
    _enable_cross_check(monkeypatch, FakeScorer(resolver))

    report = await run(
        task="finish the feature",
        env=LocalEnvironment(str(tmp_path / "tmp")),
        config=_harness_config(tmp_path),
        agent=SequencedAgent([_PLAN, _EXECUTOR, _CLEAN_AUDIT, _DONE]),
    )

    assert report["completion_satisfied"] is True
    assert not [
        item for item in _events(tmp_path) if item["event"] == "auditor_cross_check"
    ]
    round_one = _recorded_rounds(tmp_path)[0]
    payload = round_one["auditor_status"]["auditor_cross_check"]
    assert payload["flagged"] is False
    status_entry = next(
        entry for entry in payload["controls"] if entry["control"] == "status"
    )
    assert status_entry["scorer_answer"] == "incomplete"
    assert status_entry["flagged"] is False


@pytest.mark.asyncio
async def test_scorer_error_leaves_no_record_or_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scorer = RaisingScorer()
    _disable_fast_gate(monkeypatch)
    _enable_cross_check(monkeypatch, scorer)

    report = await run(
        task="finish the feature",
        env=LocalEnvironment(str(tmp_path / "tmp")),
        config=_harness_config(tmp_path),
        agent=SequencedAgent([_PLAN, _EXECUTOR, _CLEAN_AUDIT, _DONE]),
    )

    # The run is untouched: the auditor's verdicts stand and done is accepted.
    assert report["completion_satisfied"] is True
    assert scorer.calls == 1
    assert not [
        item for item in _events(tmp_path) if item["event"] == "auditor_cross_check"
    ]
    round_one = _recorded_rounds(tmp_path)[0]
    assert "auditor_cross_check" not in round_one["auditor_status"]


@pytest.mark.asyncio
async def test_disabled_cross_check_gathers_nothing_and_calls_no_scorer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _disable_fast_gate(monkeypatch)
    _disable_cross_check(monkeypatch)

    def _tripwire(*_args: Any, **_kwargs: Any) -> None:  # pragma: no cover - tripwire
        raise AssertionError("the disabled cross-check must not gather or score")

    monkeypatch.setattr(manager_module, "gather_gate_evidence", _tripwire)
    monkeypatch.setattr(manager_module, "cross_check", _tripwire)

    report = await run(
        task="finish the feature",
        env=LocalEnvironment(str(tmp_path / "tmp")),
        config=_harness_config(tmp_path),
        agent=SequencedAgent([_PLAN, _EXECUTOR, _CLEAN_AUDIT, _DONE]),
    )

    # Behavior identical to today: the auditor ran, no cross-check artifacts.
    assert report["completion_satisfied"] is True
    events = _events(tmp_path)
    assert [item for item in events if item["event"] == "auditor_role_start"]
    assert not [item for item in events if item["event"] == "auditor_cross_check"]
    round_one = _recorded_rounds(tmp_path)[0]
    assert "auditor_cross_check" not in round_one["auditor_status"]


@pytest.mark.asyncio
async def test_cross_check_never_runs_on_the_gate_skip_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _gate_fails(question: str):
        lowered = question.lower()
        if "constraint" in lowered:
            return {"no": 0.99, "yes": 0.005, "undetermined": 0.005}
        if "vcs status changes match" in lowered:
            return {"match": 0.97, "partial": 0.01, "no_changes_claimed": 0.01, "mismatch": 0.01}
        if "empty of the round's expected changes" in lowered:
            return {"no": 0.97, "yes": 0.03}
        return {}

    cross_scorer = FakeScorer(_agrees_with_clean_audit)
    monkeypatch.setattr(manager_module, "resolve_fast_gate", lambda: (FakeScorer(_gate_fails), 0.95))
    _enable_cross_check(monkeypatch, cross_scorer)

    await run(
        task="finish the feature",
        env=LocalEnvironment(str(tmp_path / "tmp")),
        config=_harness_config(tmp_path),
        agent=SequencedAgent([_PLAN, _EXECUTOR, _DONE]),
    )

    # The fast gate skipped the auditor; the cross-check is a post-audit
    # feature, so that path must not consult its scorer at all.
    round_one = _recorded_rounds(tmp_path)[0]
    assert round_one["auditor_status"]["status"] == "skipped_by_fast_gate"
    assert "auditor_cross_check" not in round_one["auditor_status"]
    assert cross_scorer.calls == []
    assert not [
        item for item in _events(tmp_path) if item["event"] == "auditor_cross_check"
    ]


def test_flagged_cross_check_never_blocks_done_acceptance() -> None:
    # A round whose auditor report parses complete/clean/aligned satisfies
    # _latest_auditor_is_clean_complete even with a flagged cross-check
    # payload sitting in auditor_status.
    disagreement = {
        "control": "status",
        "question": "Given the evidence, is the task status complete?",
        "options": ["complete", "incomplete", "blocked"],
        "auditor_verdict": "complete",
        "scorer_answer": "incomplete",
        "probabilities": {"complete": 0.02, "incomplete": 0.97, "blocked": 0.01},
        "probability": 0.97,
        "auditor_probability": 0.02,
        "agrees": False,
        "flagged": True,
    }
    round_one = ManagedRound(
        round_index=1,
        next_step="cli",
        plan_text=_PLAN,
        auditor_report=_CLEAN_AUDIT,
        auditor_status={
            "auditor_cross_check": {
                "threshold": 0.9,
                "flagged": True,
                "agreements": [],
                "disagreements": [disagreement],
                "controls": [disagreement],
            }
        },
    )

    assert manager_module._latest_auditor_is_clean_complete([round_one]) is True
