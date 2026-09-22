"""The auditor-fast pre-gate: fail-only scorer decisions in front of the slow auditor.

The battery judges live evidence (VCS status, executor output, contract
constraints, manager plan) through a fake scorer -- never the real shim, so
the suite stays hermetic. The wiring tests drive the full manager loop with
fake agents and assert the skip-auditor cascade end to end.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

import lhht.manager as manager_module
from lhht.auditor_agent import parse_audit_report
from lhht.auditor_fast import (
    DEFAULT_GATE_THRESHOLD,
    GateEvidence,
    RoundContext,
    extract_acceptance_constraints,
    gate_from_defaults,
    gate_report_text,
    gather_gate_evidence,
    run_gate,
)
from lhht.config import ProjectConfigError, _flatten_run_table
from lhht.environment.local import LocalEnvironment
from lhht.manager import run
from lhht.types import EpisodeResult, HarnessConfig

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeScorer:
    """One canned distribution per question, resolved by substring match.

    The resolver maps a question fragment to ``{option_id: probability}``;
    returning ``None`` simulates the CLI scorer's timeout/unusable answer.
    """

    def __init__(self, resolver: Any = None) -> None:
        self._resolver = resolver
        self.calls: list[str] = []

    def score(self, state: str, question: str, options: list[dict[str, str]]):
        self.calls.append(question)
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


def _yes_on_constraints(question: str):
    lowered = question.lower()
    if "constraint" in lowered:
        return {"yes": 0.99, "no": 0.005, "undetermined": 0.005}
    if "vcs status changes match" in lowered:
        return {"match": 0.97, "partial": 0.01, "no_changes_claimed": 0.01, "mismatch": 0.01}
    if "empty of the round's expected changes" in lowered:
        return {"no": 0.97, "yes": 0.03}
    return {}


def _no_on_constraints(question: str):
    lowered = question.lower()
    if "constraint" in lowered:
        return {"no": 0.99, "yes": 0.005, "undetermined": 0.005}
    if "vcs status changes match" in lowered:
        return {"match": 0.97, "partial": 0.01, "no_changes_claimed": 0.01, "mismatch": 0.01}
    if "empty of the round's expected changes" in lowered:
        return {"no": 0.97, "yes": 0.03}
    return {}


def _evidence(
    *,
    constraints: list[str] | None = None,
    vcs_status: str | None = None,
    executor_output: str = "executor output text",
) -> GateEvidence:
    return GateEvidence(
        vcs_status="" if vcs_status is None else vcs_status,
        vcs_tool="" if vcs_status is None else "git",
        vcs_status_available=vcs_status is not None,
        executor_output=executor_output,
        plan_text="plan text for the round",
        task_contract="Task contract:\nAcceptance constraints:\n- do the thing",
        acceptance_constraints=(
            ["the deliverable must exist"] if constraints is None else constraints
        ),
    )


def _harness_config(tmp_path: Path, workspace: str | None = None) -> HarnessConfig:
    return HarnessConfig(
        max_total_episodes=2,
        workspace_path=workspace or str(tmp_path / "workspace"),
        harness_dir=str(tmp_path / "harness"),
        log_dir=str(tmp_path / "logs"),
    )


# ---------------------------------------------------------------------------
# run_gate: verdict semantics
# ---------------------------------------------------------------------------


def test_confident_yes_passes() -> None:
    scorer = FakeScorer(_yes_on_constraints)

    decision = run_gate(scorer, _evidence(vcs_status="## main\n M src/app.py"), 0.95)

    assert decision.verdict == "pass"
    assert decision.findings == []
    assert all(not entry["votes_fail"] for entry in decision.battery)
    # All three criterion families were asked with their specified options.
    by_id = {entry["id"]: entry for entry in decision.battery}
    assert by_id["constraint_1"]["options"] == ["yes", "no", "undetermined"]
    assert by_id["vcs_claims_match"]["options"] == [
        "match",
        "partial",
        "no_changes_claimed",
        "mismatch",
    ]
    assert by_id["workspace_changes_absent"]["options"] == ["yes", "no"]


def test_confident_constraint_violation_fails() -> None:
    scorer = FakeScorer(_no_on_constraints)

    decision = run_gate(scorer, _evidence(), 0.95)

    assert decision.verdict == "fail"
    assert [finding["criterion"] for finding in decision.findings] == ["constraint_1"]
    assert decision.findings[0]["fail_option"] == "no"
    assert decision.findings[0]["probability"] >= 0.95
    assert decision.battery[0]["votes_fail"] is True
    # The non-failing criteria still ran and did not vote.
    assert all(not entry["votes_fail"] for entry in decision.battery[1:])


def test_low_confidence_passes() -> None:
    scorer = FakeScorer(
        lambda question: (
            {"no": 0.80, "yes": 0.10, "undetermined": 0.10}
            if "constraint" in question.lower()
            else _yes_on_constraints(question)
        )
    )

    decision = run_gate(scorer, _evidence(), 0.95)

    assert decision.verdict == "pass"
    winner = decision.battery[0]
    assert winner["winner"] == "no"
    assert winner["votes_fail"] is False, "below-threshold wins must not vote fail"


def test_threshold_is_inclusive() -> None:
    scorer = FakeScorer(
        lambda question: (
            {"no": 0.95, "yes": 0.03, "undetermined": 0.02}
            if "constraint" in question.lower()
            else {}
        )
    )

    decision = run_gate(scorer, _evidence(), 0.95)

    assert decision.verdict == "fail"


def test_fail_side_below_another_winner_does_not_vote() -> None:
    # "no" is high but "yes" wins the argmax: only a winning fail-side counts.
    scorer = FakeScorer(
        lambda question: (
            {"yes": 0.96, "no": 0.03, "undetermined": 0.01}
            if "constraint" in question.lower()
            else {}
        )
    )

    decision = run_gate(scorer, _evidence(), 0.95)

    assert decision.verdict == "pass"
    assert decision.battery[0]["winner"] == "yes"


def test_tie_goes_to_the_earliest_option() -> None:
    scorer = FakeScorer(
        lambda question: (
            {"yes": 0.5, "no": 0.5, "undetermined": 0.0}
            if "constraint" in question.lower()
            else {}
        )
    )

    decision = run_gate(scorer, _evidence(), 0.95)

    assert decision.battery[0]["winner"] == "yes"
    assert decision.verdict == "pass"


def test_scorer_error_passes() -> None:
    scorer = RaisingScorer()

    decision = run_gate(scorer, _evidence(vcs_status="## main"), 0.95)

    assert decision.verdict == "pass"
    assert scorer.calls == len(decision.battery)
    assert all(entry["skipped"] == "scorer_unavailable" for entry in decision.battery)


def test_scorer_timeout_answer_passes() -> None:
    scorer = FakeScorer(lambda question: None)

    decision = run_gate(scorer, _evidence(), 0.95)

    assert decision.verdict == "pass"
    assert all(entry["probabilities"] is None for entry in decision.battery)


def test_malformed_probabilities_pass() -> None:
    scorer = FakeScorer(lambda question: {"yes": "not-a-number"})

    decision = run_gate(scorer, _evidence(), 0.95)

    assert decision.verdict == "pass"


def test_vcs_status_unavailable_degrades_the_status_criteria_to_pass() -> None:
    scorer = FakeScorer(_yes_on_constraints)

    decision = run_gate(scorer, _evidence(vcs_status=None), 0.95)

    assert decision.verdict == "pass"
    by_id = {entry["id"]: entry for entry in decision.battery}
    assert by_id["vcs_claims_match"]["skipped"] == "vcs_status_unavailable"
    assert by_id["workspace_changes_absent"]["skipped"] == "vcs_status_unavailable"
    # The constraint criterion still ran; the status questions were never asked.
    assert by_id["constraint_1"]["skipped"] is None
    assert scorer.calls == ["Is this constraint satisfied by the evidence? "
                            "Constraint: \"the deliverable must exist\""]


def test_confident_workspace_mismatch_can_fail_the_gate() -> None:
    def resolver(question: str):
        lowered = question.lower()
        if "vcs status changes match" in lowered:
            return {"mismatch": 0.97, "match": 0.01, "partial": 0.01, "no_changes_claimed": 0.01}
        if "empty of the round's expected changes" in lowered:
            return {"no": 0.97, "yes": 0.03}
        return {"yes": 0.97, "no": 0.01, "undetermined": 0.02}

    decision = run_gate(FakeScorer(resolver), _evidence(vcs_status="## main\n M src/app.py"), 0.95)

    assert decision.verdict == "fail"
    assert [finding["criterion"] for finding in decision.findings] == ["vcs_claims_match"]
    assert decision.findings[0]["fail_option"] == "mismatch"


def test_confident_empty_workspace_can_fail_the_gate() -> None:
    def resolver(question: str):
        lowered = question.lower()
        if "empty of the round's expected changes" in lowered:
            return {"yes": 0.97, "no": 0.03}
        if "vcs status changes match" in lowered:
            return {"match": 0.97, "partial": 0.01, "no_changes_claimed": 0.01, "mismatch": 0.01}
        return {"yes": 0.97, "no": 0.01, "undetermined": 0.02}

    decision = run_gate(FakeScorer(resolver), _evidence(vcs_status="## main"), 0.95)

    assert decision.verdict == "fail"
    assert [finding["criterion"] for finding in decision.findings] == ["workspace_changes_absent"]


def test_no_constraints_and_no_vcs_still_pass() -> None:
    scorer = FakeScorer(_no_on_constraints)

    decision = run_gate(
        scorer, _evidence(constraints=[], vcs_status=None), DEFAULT_GATE_THRESHOLD
    )

    assert decision.verdict == "pass"
    assert scorer.calls == [], "no criterion exists to ask"


# ---------------------------------------------------------------------------
# gate_report_text: the synthetic audit record
# ---------------------------------------------------------------------------


def test_gate_report_text_parses_incomplete_clean_unknown() -> None:
    decision = run_gate(FakeScorer(_no_on_constraints), _evidence(vcs_status="## main\n M a.py"), 0.95)

    report = gate_report_text(decision, _evidence(vcs_status="## main\n M a.py"))

    assert report.startswith("auditor-fast gate:")
    parsed = parse_audit_report(report, 1)
    assert parsed.status == "incomplete"
    assert parsed.integrity_status == "clean"
    assert parsed.contract_audit_status == "unknown"
    assert "constraint_1" in report and "p=0.990" in report
    assert "Workspace VCS status lines:" in report


# ---------------------------------------------------------------------------
# gather_gate_evidence: live VCS facts (real subprocesses, temp workspaces)
# ---------------------------------------------------------------------------


def test_vcs_unavailable_records_a_note_and_never_raises(tmp_path: Path) -> None:
    evidence = gather_gate_evidence(
        _harness_config(tmp_path), RoundContext(plan_text="p", executor_output="o")
    )

    assert evidence.vcs_status_available is False
    assert evidence.vcs_status == ""
    assert evidence.notes, "a missing VCS must be explained by a note"
    digest = evidence.digest()
    assert digest["vcs_status_available"] is False
    assert digest["vcs_status_line_count"] == 0
    assert digest["executor_output_chars"] == 1


def _git_workspace(tmp_path: Path) -> Path:
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    workspace = tmp_path / "workspace"
    (workspace / "src").mkdir(parents=True)
    (workspace / "src" / "app.py").write_text("print(1)\n", encoding="utf-8")
    for argv in (
        ("git", "init", "-q"),
        ("git", "add", "."),
        ("git", "-c", "user.email=t@example.com", "-c", "user.name=t", "commit", "-qm", "init"),
    ):
        subprocess.run(argv, cwd=workspace, check=True, capture_output=True)
    (workspace / "src" / "app.py").write_text("print(2)\n", encoding="utf-8")
    (workspace / "target").mkdir()
    (workspace / "target" / "out.log").write_text("build noise\n", encoding="utf-8")
    return workspace


def test_vcs_status_is_gathered_and_digest_counts_lines(tmp_path: Path) -> None:
    workspace = _git_workspace(tmp_path)

    evidence = gather_gate_evidence(
        _harness_config(tmp_path, workspace=str(workspace)),
        RoundContext(plan_text="plan", executor_output="executor output"),
    )

    assert evidence.vcs_status_available is True
    assert evidence.vcs_tool == "git"
    assert any(line.strip().endswith("src/app.py") for line in evidence.vcs_status.splitlines())
    assert evidence.digest()["vcs_status_line_count"] == len(
        [line for line in evidence.vcs_status.splitlines() if line.strip()]
    )
    assert evidence.digest()["executor_output_chars"] == len("executor output")


def test_guard_excluded_paths_are_filtered_from_the_status(tmp_path: Path) -> None:
    workspace = _git_workspace(tmp_path)

    evidence = gather_gate_evidence(
        _harness_config(tmp_path, workspace=str(workspace)),
        RoundContext(guard_exclude_paths=(str(workspace / "target"),)),
    )

    status_lines = [line for line in evidence.vcs_status.splitlines() if line.strip()]
    assert any(line.strip().endswith("src/app.py") for line in status_lines)
    assert not any("target" in line for line in status_lines)
    assert any("guard_exclude_paths" in note for note in evidence.notes)


def test_constraints_are_extracted_from_the_task_contract(tmp_path: Path) -> None:
    contract = (
        "Task contract:\n"
        "Interpretation calibration: judge live evidence only.\n\n"
        "Acceptance constraints:\n"
        "1. The full pytest suite must be green\n"
        "2. The report must land at docs/report.md\n\n"
        "**Acceptable evidence:** pytest output and file contents\n"
    )

    evidence = gather_gate_evidence(
        _harness_config(tmp_path),
        RoundContext(task_contract=contract),
    )

    assert evidence.acceptance_constraints == [
        "The full pytest suite must be green",
        "The report must land at docs/report.md",
    ]
    assert evidence.digest()["acceptance_constraint_count"] == 2


def test_inline_numbered_constraints_split() -> None:
    contract = (
        "Acceptance constraints (all blocking, source = original request): "
        "(1) suite green; (2) no new dependencies."
    )

    items = extract_acceptance_constraints(contract)

    assert items == ["suite green;", "no new dependencies."]


def test_bold_header_with_inline_constraints_splits() -> None:
    contract = (
        "Task contract:\n\n"
        "**Acceptance constraints (all blocking, source = original request):** "
        "(1) full pytest suite green; (2) no new runtime dependencies.\n\n"
        "**Acceptable evidence:** pytest output\n"
    )

    items = extract_acceptance_constraints(contract)

    assert items == [
        "full pytest suite green;",
        "no new runtime dependencies.",
    ]


def test_absent_acceptance_section_yields_no_constraints() -> None:
    assert extract_acceptance_constraints("Task contract:\nTarget state: done.\n") == []


# ---------------------------------------------------------------------------
# config: [run.semif] keys
# ---------------------------------------------------------------------------


def test_semif_keys_flatten_into_defaults() -> None:
    defaults = _flatten_run_table(
        {
            "semif": {
                "enabled": True,
                "command": "scripts/semif_shim.bat",
                "model": "m",
                "revision": "r",
                "auditor_fast": True,
                "auditor_fast_threshold": 0.9,
            }
        }
    )

    assert defaults["semif_auditor_fast"] is True
    assert defaults["semif_auditor_fast_threshold"] == 0.9


def test_unknown_semif_key_is_refused() -> None:
    with pytest.raises(ProjectConfigError, match="unknown \\[run.semif\\] key"):
        _flatten_run_table({"semif": {"auditor_fast_x": True}})


@pytest.mark.parametrize("bad", [0, 1.5, True, "high"])
def test_bad_auditor_fast_threshold_is_refused(bad: Any) -> None:
    with pytest.raises(ProjectConfigError, match="auditor_fast_threshold"):
        _flatten_run_table({"semif": {"auditor_fast_threshold": bad}})


def test_auditor_fast_requires_a_bool() -> None:
    with pytest.raises(ProjectConfigError, match="auditor_fast"):
        _flatten_run_table({"semif": {"auditor_fast": "yes"}})


def test_gate_off_for_absent_or_disabled_config_without_building_a_scorer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _no_scorer(_defaults: Any) -> None:  # pragma: no cover - tripwire
        raise AssertionError("no scorer may be constructed for an off gate")

    monkeypatch.setattr("lhht.auditor_fast.scorer_from_config", _no_scorer)

    assert gate_from_defaults({}) == (None, DEFAULT_GATE_THRESHOLD)
    assert gate_from_defaults({"semif_auditor_fast": False}) == (None, DEFAULT_GATE_THRESHOLD)


def test_gate_silently_off_when_the_scorer_is_not_configured() -> None:
    assert gate_from_defaults({"semif_auditor_fast": True}) == (None, DEFAULT_GATE_THRESHOLD)


def test_gate_resolves_scorer_and_threshold_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()
    monkeypatch.setattr(
        "lhht.auditor_fast.scorer_from_config",
        lambda defaults: sentinel,
    )

    scorer, threshold = gate_from_defaults(
        {
            "semif_auditor_fast": True,
            "semif_auditor_fast_threshold": 0.9,
        }
    )

    assert scorer is sentinel
    assert threshold == 0.9


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


def _enable_gate(monkeypatch: pytest.MonkeyPatch, scorer: Any, threshold: float = 0.95) -> None:
    monkeypatch.setattr(manager_module, "resolve_fast_gate", lambda: (scorer, threshold))


def _disable_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(manager_module, "resolve_fast_gate", lambda: (None, 0.95))


@pytest.mark.asyncio
async def test_gate_fail_skips_the_auditor_episode_and_records_the_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scorer = FakeScorer(_no_on_constraints)
    _enable_gate(monkeypatch, scorer)
    agent = SequencedAgent([_PLAN, _EXECUTOR, _DONE])

    report = await run(
        task="finish the feature",
        env=LocalEnvironment(str(tmp_path / "tmp")),
        config=_harness_config(tmp_path),
        agent=agent,
    )

    # The auditor episode never ran: no auditor prompt or artifacts exist and
    # no auditor lifecycle event was emitted. The remaining prompts are the
    # two manager turns, the executor turn, and the run's closing reply.
    round_one_dir = tmp_path / "logs" / "role_orchestration" / "rounds" / "round_001"
    assert not (round_one_dir / "auditor_input.txt").exists()
    assert not (round_one_dir / "auditor_raw_trajectory.jsonl").exists()
    events = _events(tmp_path)
    assert not [item for item in events if item["event"] == "auditor_role_start"]
    assert not [item for item in events if item["event"] == "auditor_role_done"]

    gate_events = [item for item in events if item["event"] == "auditor_fast_gate"]
    assert [item["round"] for item in gate_events] == [1]
    payload = gate_events[0]
    assert payload["verdict"] == "fail"
    assert payload["battery"], "the full battery must reach the events"
    assert all(
        {"question", "options", "probabilities"} <= set(entry) for entry in payload["battery"]
    )
    assert payload["evidence_digest"]["executor_output_chars"] == len(_EXECUTOR)
    assert "vcs_status_line_count" in payload["evidence_digest"]
    assert any(item["event"] == "auditor_fast_gate_skip" for item in events)

    # The synthesized record is a routed incomplete audit with the findings.
    round_one = _recorded_rounds(tmp_path)[0]
    assert round_one["auditor_report"].startswith("auditor-fast gate:")
    assert round_one["auditor_status"]["status"] == "skipped_by_fast_gate"
    assert round_one["auditor_status"]["auditor_fast_gate"]["verdict"] == "fail"
    parsed = parse_audit_report(round_one["auditor_report"], 1)
    assert (parsed.status, parsed.integrity_status, parsed.contract_audit_status) == (
        "incomplete",
        "clean",
        "unknown",
    )
    assert scorer.calls, "the gate actually consulted the scorer"


@pytest.mark.asyncio
async def test_gate_fail_findings_reach_the_next_manager_round(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable_gate(monkeypatch, FakeScorer(_no_on_constraints))
    agent = SequencedAgent([_PLAN, _EXECUTOR, _DONE])

    await run(
        task="finish the feature",
        env=LocalEnvironment(str(tmp_path / "tmp")),
        config=_harness_config(tmp_path),
        agent=agent,
    )

    # The round-2 manager prompt is rebuilt from the rounds ledger, so the
    # synthetic report is the re-binding signal it plans against.
    assert "auditor-fast gate:" in agent.prompts[2]


@pytest.mark.asyncio
async def test_gate_fail_never_satisfies_done_acceptance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable_gate(monkeypatch, FakeScorer(_no_on_constraints))

    report = await run(
        task="finish the feature",
        env=LocalEnvironment(str(tmp_path / "tmp")),
        config=_harness_config(tmp_path),
        agent=SequencedAgent([_PLAN, _EXECUTOR, _DONE]),
    )

    # The manager said done right after a gate-fired round; the only auditor
    # evidence is the gate's own incomplete synthesis, so completion must be
    # refused and fed back as a repair signal.
    assert report["completion_satisfied"] is False
    assert report["status"] != "complete"
    round_two = _recorded_rounds(tmp_path)[1]
    assert round_two["auditor_status"].get("invalid_completion") is True


@pytest.mark.asyncio
async def test_gate_pass_launches_the_auditor_as_today(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scorer = FakeScorer(_yes_on_constraints)
    _enable_gate(monkeypatch, scorer)

    report = await run(
        task="finish the feature",
        env=LocalEnvironment(str(tmp_path / "tmp")),
        config=_harness_config(tmp_path),
        agent=SequencedAgent([_PLAN, _EXECUTOR, _CLEAN_AUDIT, _DONE]),
    )

    # The slow auditor ran and its clean report still satisfies done.
    assert report["completion_satisfied"] is True
    assert report["status"] == "complete"
    events = _events(tmp_path)
    assert any(item["event"] == "auditor_role_start" for item in events)
    gate_events = [item for item in events if item["event"] == "auditor_fast_gate"]
    assert [item["verdict"] for item in gate_events] == ["pass"]
    # The pass verdict is recorded next to the real audit for later comparison.
    round_one = _recorded_rounds(tmp_path)[0]
    assert round_one["auditor_status"]["auditor_fast_gate"]["verdict"] == "pass"
    assert round_one["auditor_report"].startswith("Status: complete")


@pytest.mark.asyncio
async def test_disabled_gate_builds_nothing_and_calls_no_scorer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _disable_gate(monkeypatch)

    def _tripwire(*_args: Any, **_kwargs: Any) -> None:  # pragma: no cover - tripwire
        raise AssertionError("the disabled gate must not gather evidence")

    monkeypatch.setattr(manager_module, "gather_gate_evidence", _tripwire)
    monkeypatch.setattr(manager_module, "run_gate", _tripwire)

    report = await run(
        task="finish the feature",
        env=LocalEnvironment(str(tmp_path / "tmp")),
        config=_harness_config(tmp_path),
        agent=SequencedAgent([_PLAN, _EXECUTOR, _CLEAN_AUDIT, _DONE]),
    )

    # Behavior identical to today: the auditor ran, no gate artifacts exist.
    assert report["completion_satisfied"] is True
    events = _events(tmp_path)
    assert not [item for item in events if item["event"] == "auditor_fast_gate"]
    assert not [item for item in events if item["event"] == "auditor_fast_gate_skip"]
    round_one = _recorded_rounds(tmp_path)[0]
    assert "auditor_fast_gate" not in round_one["auditor_status"]
