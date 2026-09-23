"""`lhht scorer-stats`: one summary of everything the scorer did in a run.

A synthetic run dir exercises every feature's persisted payloads (a gate
skip, a cross-check flag, effort-routing variants, report selection, a
round-dedup flag) plus the rebuilt salvage ledger through a fake scorer --
never the real shim, so the suite stays hermetic. Empty and old-style run
dirs must render explicit zeros without crashing, fixture appends dedup by
id against the existing file, and the subcommand registers beside run and
doctor.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import lhht.scorer_stats as scorer_stats
from lhht.cli import main as cli_main
from lhht.scorer_stats import (
    append_fixture_rows,
    mine_fixture_rows,
    render_summary,
    summarize_run,
)

# ---------------------------------------------------------------------------
# Fakes and fixtures
# ---------------------------------------------------------------------------


class FakeScorer:
    """Salvage answers keyed by the exact control line (the salvage state).

    ``answers`` maps a line to ``(favored option id, probability)``; every
    other state scores ``None`` (the CLI scorer's timeout/unusable answer).
    """

    def __init__(self, answers: dict[str, tuple[str, float]]) -> None:
        self._answers = answers

    def score(
        self, state: str, question: str, options: list[dict[str, str]]
    ) -> list[float] | None:
        answer = self._answers.get(state.strip())
        if answer is None:
            return None
        favored, probability = answer
        ids = [option["id"] for option in options]
        if favored not in ids:
            return None
        rest = (1.0 - probability) / max(1, len(ids) - 1)
        return [probability if value == favored else rest for value in ids]


def _no_config_scorer(monkeypatch: pytest.MonkeyPatch) -> None:
    # Keep the default scorer resolution hermetic: the checkout's own
    # .lhht/config.toml may enable a real shim, which no test may invoke.
    monkeypatch.setattr(scorer_stats, "scorer_from_config", lambda defaults: None)


def _write_run(
    runs_root: Path, run_id: str, records: list[dict[str, Any]]
) -> Path:
    role_dir = runs_root / run_id / "lhht" / "role_orchestration"
    role_dir.mkdir(parents=True, exist_ok=True)
    with open(role_dir / "rounds.jsonl", "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return runs_root / run_id


def _full_feature_records() -> list[dict[str, Any]]:
    """Three rounds touching every feature the summary must count.

    Round 1: routed executor (low variant) with report selection, a passing
    gate evaluation, a slow audit (60 s) and a cross-check with one flagged
    disagreement; its auditor header needs status salvage and its blocking
    constraints line needs an acceptance_none rescue.
    Round 2: routed executor (default variant, escalated), a gate skip, and
    a round-dedup flag; its route line needs route salvage.
    Round 3: plain slow audit (120 s) with canonical control lines.
    """

    return [
        {
            "round_index": 1,
            "next_step": "cli",
            "plan_text": "Subtask A\nNext: cli",
            "executor_output": "executor did the work",
            "auditor_report": (
                "Status: finished\n"
                "Integrity: clean\n"
                "Contract audit: aligned\n"
                "\n"
                "Blocking constraints: nothing applies here\n"
                "\n"
                "Audit facts: everything checked."
            ),
            "executor_status": {
                "status": "done",
                "duration_ms": 10000,
                "effort_routing": {
                    "default_effort": "medium",
                    "variant": "low",
                    "used_default": False,
                    "probabilities": {
                        "mechanical": 0.93,
                        "standard": 0.05,
                        "deep": 0.02,
                    },
                    "classified": "mechanical",
                    "escalated": False,
                    "skipped": None,
                },
                "report_selection": {
                    "kept": [{"id": "round_001", "probability": 0.9}],
                    "dropped": [{"id": "round_002", "probability": 0.4}],
                    "explicit_refs": [],
                    "degraded": False,
                    "k": 3,
                    "threshold": 0.6,
                },
            },
            "auditor_status": {
                "status": "done",
                "duration_ms": 60000,
                "auditor_fast_gate": {
                    "verdict": "pass",
                    "battery": [],
                    "evidence_digest": {},
                },
                "auditor_cross_check": {
                    "threshold": 0.9,
                    "flagged": True,
                    "agreements": [
                        {"control": "integrity"},
                        {"control": "contract_audit"},
                    ],
                    "disagreements": [
                        {
                            "control": "status",
                            "auditor_verdict": "complete",
                            "scorer_answer": "incomplete",
                            "probability": 0.93,
                        }
                    ],
                    "controls": [
                        {"control": "status"},
                        {"control": "integrity"},
                        {"control": "contract_audit"},
                    ],
                },
            },
        },
        {
            "round_index": 2,
            "next_step": "cli",
            "plan_text": "Subtask B\nNext: do the CLI task",
            "executor_output": "executor did more work",
            "auditor_report": (
                "auditor-fast gate: confident local fail on live evidence; "
                "the slow auditor episode was skipped for this round.\n"
                "\n"
                "Status: incomplete\n"
                "Integrity: clean\n"
                "Contract audit: unknown"
            ),
            "executor_status": {
                "status": "done",
                "duration_ms": 5000,
                "effort_routing": {
                    "default_effort": "medium",
                    "variant": "medium",
                    "used_default": True,
                    "probabilities": {
                        "mechanical": 0.05,
                        "standard": 0.95,
                        "deep": 0.0,
                    },
                    "classified": "standard",
                    "escalated": True,
                    "skipped": "escalated_to_default",
                },
            },
            "auditor_status": {
                "status": "skipped_by_fast_gate",
                "audit_status": "incomplete",
                "integrity_status": "clean",
                "contract_audit_status": "unknown",
                "auditor_fast_gate": {
                    "verdict": "fail",
                    "battery": [],
                    "evidence_digest": {},
                },
            },
            "manager_status": {
                "status": "done",
                "duration_ms": 4000,
                "round_dedup": {
                    "flagged": True,
                    "same": True,
                    "probability": 0.95,
                    "probabilities": {
                        "same_work": 0.95,
                        "different_work": 0.05,
                    },
                    "threshold": 0.9,
                },
            },
        },
        {
            "round_index": 3,
            "next_step": "gui",
            "plan_text": "Subtask C\nNext: gui",
            "executor_output": "executor finished the visuals",
            "auditor_report": (
                "Status: incomplete\n"
                "Integrity: suspect\n"
                "Contract audit: unknown\n"
                "\n"
                "Audit facts: partial delivery."
            ),
            "auditor_status": {"status": "done", "duration_ms": 120000},
        },
    ]


def _feature_scorer() -> FakeScorer:
    # The salvage states are the control lines themselves (audit controls)
    # and the post-colon text (the acceptance guard), exactly as production
    # passes them to salvage_control_value.
    return FakeScorer(
        {
            "Status: finished": ("complete", 0.91),
            "nothing applies here": ("yes", 0.9),
            "Next: do the CLI task": ("cli", 0.88),
        }
    )


# ---------------------------------------------------------------------------
# summarize_run / render_summary over the full-feature fixture
# ---------------------------------------------------------------------------


def test_summary_counts_every_feature(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, "demo", _full_feature_records())

    summary = summarize_run(run_dir, scorer=_feature_scorer())

    assert summary["run_dir"].endswith("demo")
    assert summary["rounds"] == 3
    assert summary["gate"] == {
        "evaluations": 2,
        "passes": 1,
        "skips": 1,
        "slow_audits": 2,
        "median_slow_ms": 90000.0,
        "estimated_minutes_saved": 1.5,
    }
    cross = summary["cross_check"]
    assert cross["rounds"] == 1
    assert cross["verdicts_checked"] == 3
    assert cross["agreements"] == 2
    assert cross["flagged_disagreements"] == [
        {
            "round": 1,
            "control": "status",
            "auditor_verdict": "complete",
            "scorer_answer": "incomplete",
            "scorer_probability": 0.93,
        }
    ]
    routing = summary["effort_routing"]
    assert routing["rounds"] == 2
    assert routing["escalations"] == 1
    assert routing["variants"]["low"] == {
        "rounds": 1,
        "mean_probability": pytest.approx(0.93),
        "mean_duration_ms": 10000.0,
    }
    assert routing["variants"]["medium"] == {
        "rounds": 1,
        "mean_probability": pytest.approx(0.95),
        "mean_duration_ms": 5000.0,
    }
    assert summary["report_selection"] == {
        "rounds": 1,
        "candidates": 2,
        "kept": 1,
        "dropped": 1,
        "degraded": 0,
    }
    assert summary["round_dedup"] == {
        "comparisons": 1,
        "flags": [{"round": 2, "probability": 0.95}],
    }


def test_rendered_summary_carries_the_numbers(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, "demo", _full_feature_records())

    text = render_summary(summarize_run(run_dir, scorer=_feature_scorer()))

    assert "3 recorded round(s)" in text
    # Salvage: one rescue each for route, status, and acceptance_none.
    assert "route: 1 line(s) rescued, max probability 0.880" in text
    assert "status: 1 line(s) rescued, max probability 0.910" in text
    assert "acceptance_none: 1 line(s) rescued, max probability 0.900" in text
    assert "integrity: 0 line(s) rescued" in text
    assert "contract_audit: 0 line(s) rescued" in text
    # Gate, with the estimate explicitly labeled as one.
    assert "Auditor-fast gate: evaluations 2, passes 1, skips 1" in text
    assert "median duration 90.0 s" in text
    assert "estimated audit minutes saved: 1.5 (estimate" in text
    # Cross-check disagreement with both sides' values.
    assert (
        "round 1 status: auditor=complete, scorer=incomplete, p=0.930" in text
    )
    # Effort routing per variant.
    assert "low: 1 round(s), mean routing probability 0.930" in text
    assert "mean executor duration 10.0 s" in text
    assert "medium: 1 round(s), mean routing probability 0.950" in text
    # Report selection and round dedup.
    assert (
        "Report selection: rounds with selection 1, candidates 2, "
        "kept 1, dropped 1, degraded 0" in text
    )
    assert "Round dedup: comparisons 1, flags 1" in text
    assert "round 2 flagged (same-work probability 0.950)" in text


def test_salvage_without_a_scorer_renders_not_recorded(
    tmp_path: Path,
) -> None:
    run_dir = _write_run(tmp_path, "demo", _full_feature_records())

    summary = summarize_run(run_dir, scorer=None)

    assert summary["salvage"]["recorded"] is False
    for entry in summary["salvage"]["rescued"].values():
        assert entry == {"count": 0, "max_probability": None}
    text = render_summary(summary)
    assert "Semantic salvage: not recorded (no scorer configured)" in text
    # The other features still count from the ledger alone.
    assert "skips 1" in text


def test_gate_skips_without_slow_audits_report_the_count_only(
    tmp_path: Path,
) -> None:
    records = [_full_feature_records()[1]]  # the gate-skip round alone
    run_dir = _write_run(tmp_path, "demo", records)

    summary = summarize_run(run_dir, scorer=None)
    gate = summary["gate"]

    assert gate["skips"] == 1
    assert gate["slow_audits"] == 0
    assert gate["median_slow_ms"] is None
    assert gate["estimated_minutes_saved"] is None
    text = render_summary(summary)
    assert "estimated audit minutes saved: not estimable" in text
    assert "no slow audits in this run" in text


# ---------------------------------------------------------------------------
# Degenerate run dirs: explicit zeros, never a crash
# ---------------------------------------------------------------------------


def test_empty_run_dir_renders_explicit_zeros(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_config_scorer(monkeypatch)
    run_dir = _write_run(tmp_path, "empty", [])

    summary = summarize_run(run_dir)

    assert summary["rounds"] == 0
    assert summary["gate"]["evaluations"] == 0
    assert summary["gate"]["skips"] == 0
    assert summary["gate"]["estimated_minutes_saved"] is None
    assert summary["cross_check"]["verdicts_checked"] == 0
    assert summary["effort_routing"]["rounds"] == 0
    assert summary["report_selection"]["rounds"] == 0
    assert summary["round_dedup"]["comparisons"] == 0
    text = render_summary(summary)
    assert "0 recorded round(s)" in text
    assert "evaluations 0, passes 0, skips 0" in text
    assert "verdicts checked 0, agreements 0, flagged disagreements 0" in text
    assert "Semantic salvage: not recorded (no scorer configured)" in text


def test_reserved_run_dir_without_a_ledger_renders_zeros(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_config_scorer(monkeypatch)
    run_dir = tmp_path / "runs" / "reserved"
    run_dir.mkdir(parents=True)

    summary = summarize_run(run_dir)

    assert summary["rounds"] == 0
    assert "0 recorded round(s)" in render_summary(summary)


def test_old_style_round_records_render_zeros(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_config_scorer(monkeypatch)
    records = [
        {
            "round_index": index,
            "next_step": "cli",
            "plan_text": f"Subtask {index}\nNext: cli",
            "executor_output": "old-style round",
            "auditor_report": (
                "Status: incomplete\n"
                "Integrity: clean\n"
                "Contract audit: unknown\n"
                "\n"
                "Audit facts: recorded before any scorer feature existed."
            ),
        }
        for index in (1, 2)
    ]
    run_dir = _write_run(tmp_path, "legacy", records)

    summary = summarize_run(run_dir)

    assert summary["rounds"] == 2
    assert summary["gate"] == {
        "evaluations": 0,
        "passes": 0,
        "skips": 0,
        "slow_audits": 0,
        "median_slow_ms": None,
        "estimated_minutes_saved": None,
    }
    assert summary["cross_check"] == {
        "rounds": 0,
        "verdicts_checked": 0,
        "agreements": 0,
        "flagged_disagreements": [],
    }
    assert summary["effort_routing"] == {
        "rounds": 0,
        "escalations": 0,
        "variants": {},
    }
    assert summary["report_selection"] == {
        "rounds": 0,
        "candidates": 0,
        "kept": 0,
        "dropped": 0,
        "degraded": 0,
    }
    assert summary["round_dedup"] == {"comparisons": 0, "flags": []}
    # No slow-audit duration was ever recorded, so even a skip would stay a
    # bare count; here there is nothing at all.
    assert "skips 0" in render_summary(summary)


def test_malformed_ledger_lines_are_skipped(tmp_path: Path) -> None:
    role_dir = tmp_path / "runs" / "partial" / "lhht" / "role_orchestration"
    role_dir.mkdir(parents=True)
    (role_dir / "rounds.jsonl").write_text(
        "{not json}\n"
        + json.dumps({"round_index": 1, "next_step": "cli", "plan_text": "p"})
        + "\n\n"
        + json.dumps({"round_index": "two"})
        + "\n",
        encoding="utf-8",
    )

    summary = summarize_run(tmp_path / "runs" / "partial", scorer=None)

    assert summary["rounds"] == 1


def test_non_run_directory_is_rejected(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "not-a-run.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="not a readable run directory"):
        summarize_run(runs / "not-a-run.txt")


# ---------------------------------------------------------------------------
# Fixture accumulation
# ---------------------------------------------------------------------------


def _assert_decision_row(row: dict[str, Any]) -> None:
    assert isinstance(row["id"], str) and row["id"]
    assert isinstance(row["kind"], str)
    assert isinstance(row["state"], str) and row["state"]
    assert isinstance(row["question"], str)
    assert [option["id"] for option in row["options"]]
    assert all(
        isinstance(option, dict)
        and isinstance(option["id"], str)
        and isinstance(option["description"], str)
        for option in row["options"]
    )
    assert isinstance(row["label"], int) and not isinstance(row["label"], bool)
    assert 0 <= row["label"] < len(row["options"])


def test_mine_fixture_rows_covers_routes_and_audit_headers(
    tmp_path: Path,
) -> None:
    run_dir = _write_run(tmp_path, "demo", _full_feature_records())

    rows = mine_fixture_rows(run_dir)

    by_id = {row["id"]: row for row in rows}
    # Route rows keep the harness verdict as gold, salvaged routes included.
    assert by_id["demo#r1-route"]["label"] == 1  # "cli"
    assert by_id["demo#r2-route"]["state"] == "Next: do the CLI task"
    assert by_id["demo#r2-route"]["label"] == 1  # salvaged route, gold "cli"
    assert by_id["demo#r3-route"]["label"] == 0  # "gui"
    # Audit header rows carry the canonical parsed value.
    assert by_id["demo#r1-integrity"]["label"] == 0  # clean
    assert by_id["demo#r1-contract_audit"]["label"] == 0  # aligned
    assert by_id["demo#r2-status"]["label"] == 1  # incomplete
    assert by_id["demo#r2-contract_audit"]["label"] == 1  # unknown
    assert by_id["demo#r3-integrity"]["label"] == 1  # suspect
    # A header line the exact value regex missed ("Status: finished") has no
    # trustworthy canonical value, so it is left for human labeling.
    assert "demo#r1-status" not in by_id
    for row in rows:
        _assert_decision_row(row)


def test_append_fixture_rows_dedups_by_id_and_preserves_existing(
    tmp_path: Path,
) -> None:
    run_dir = _write_run(tmp_path, "demo", _full_feature_records())
    fixture = tmp_path / "data" / "semif-fixture.jsonl"
    rows = mine_fixture_rows(run_dir)
    foreign_row = {
        "id": "other-run#r1-route",
        "kind": "route",
        "state": "Next: cli",
        "question": rows[0]["question"],
        "options": rows[0]["options"],
        "label": 1,
    }
    assert append_fixture_rows([foreign_row], fixture) == 1

    added = append_fixture_rows(rows, fixture)

    assert added == len(rows)
    lines = fixture.read_text(encoding="utf-8").splitlines()
    persisted = [json.loads(line) for line in lines]
    assert persisted[0] == foreign_row
    assert {row["id"] for row in persisted[1:]} == {row["id"] for row in rows}
    for row in persisted:
        _assert_decision_row(row)

    # Re-mining the same run adds nothing; a fresh round id adds one row.
    assert append_fixture_rows(mine_fixture_rows(run_dir), fixture) == 0
    records = _full_feature_records() + [
        {
            "round_index": 4,
            "next_step": "blocked",
            "plan_text": "Subtask D\nNext: blocked",
        }
    ]
    rerun_dir = _write_run(tmp_path, "demo2", records)
    assert append_fixture_rows(mine_fixture_rows(rerun_dir), fixture) > 0


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def test_cli_subcommand_help_and_registration(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli_main(["--help"])
    assert excinfo.value.code == 0
    assert "scorer-stats" in capsys.readouterr().out

    with pytest.raises(SystemExit) as excinfo:
        cli_main(["scorer-stats", "--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "usage: lhht scorer-stats" in out
    assert "--fixture" in out


def test_cli_subcommand_prints_the_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_config_scorer(monkeypatch)
    run_dir = _write_run(tmp_path, "demo", _full_feature_records())
    fixture = tmp_path / "fixture.jsonl"

    code = cli_main(
        ["scorer-stats", str(run_dir), "--fixture", str(fixture)]
    )

    assert code == 0
    out = capsys.readouterr().out
    assert out.startswith("Scorer stats:")
    assert "skips 1" in out
    assert f"Fixture rows added: {len(mine_fixture_rows(run_dir))}" in out
    # A second invocation dedups and adds nothing.
    code = cli_main(
        ["scorer-stats", str(run_dir), "--fixture", str(fixture)]
    )
    assert code == 0
    assert "Fixture rows added: 0" in capsys.readouterr().out


def test_cli_subcommand_rejects_a_non_run_directory(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    not_a_run = runs / "not-a-run.txt"
    not_a_run.write_text("x", encoding="utf-8")
    code = cli_main(["scorer-stats", str(not_a_run)])
    assert code == 2
    assert "Cannot summarize run" in capsys.readouterr().err
