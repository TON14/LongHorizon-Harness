"""Per-run measurement rail for every scorer feature: `lhht scorer-stats`.

The scorer is wired into nine points of the management loop, and each one
records rich payloads on the round ledger (``rounds.jsonl``). This module
reads one finished run and produces one plain-text summary of everything
the scorer did in it: semantic salvage, the auditor-fast pre-gate, the
post-audit verdict cross-check, effort routing, report selection, and the
round-dedup detector. Older runs and runs with a feature off render
explicit zeros or "not recorded" -- absent fields never crash a summary.

Salvage provenance is the one thing the ledger does not persist
(``AuditReport.control_salvage`` lives only in memory), so salvage is
rebuilt: each round's ``auditor_report`` is re-parsed and each ``plan_text``
route line re-judged with the configured scorer. With no usable scorer the
section renders "not recorded" instead of guessing.

The optional ``--fixture`` mode grows the SemIf calibration fixture by
mining the run's control lines (manager route lines and audit header
lines) into the established decision-row schema, deduplicating by id.
"""

from __future__ import annotations

import io
import json
import re
from pathlib import Path
from typing import Any, Callable

from .auditor_agent import (
    _CONTRACT_AUDIT_CONTROL_LINE_RE,
    _CONTRACT_AUDIT_LABEL_LINE_RE,
    _CONTRACT_AUDIT_LEGAL_VALUES,
    _CONTRACT_AUDIT_VALUE_DESCRIPTIONS,
    _INTEGRITY_CONTROL_LINE_RE,
    _INTEGRITY_LABEL_LINE_RE,
    _INTEGRITY_LEGAL_VALUES,
    _INTEGRITY_VALUE_DESCRIPTIONS,
    _STATUS_CONTROL_LINE_RE,
    _STATUS_LABEL_LINE_RE,
    _STATUS_LEGAL_VALUES,
    _STATUS_VALUE_DESCRIPTIONS,
    infer_contract_audit_status,
    infer_integrity_findings,
    infer_report_status,
    parse_audit_report,
)
from .config import load_run_defaults
from .manager import _recorded_rounds
from .role_prompts import (
    MANAGER_NEXT_INVALID,
    _MANAGER_ROUTE_LABEL_LINE_RE,
    _MANAGER_ROUTE_LEGAL_VALUES,
    _MANAGER_ROUTE_VALUE_DESCRIPTIONS,
    parse_role_manager_next_step,
)
from .semantic_salvage import (
    DEFAULT_THRESHOLD,
    SALVAGE_QUESTION,
    SemanticScorer,
    salvage_control_value,
    scorer_from_config,
)
from .types import ManagedRound
from .utils.run_boundary import safe_run_role

# The six salvage controls the loop can rescue, in report order.
SALVAGE_CONTROLS = (
    "route",
    "status",
    "integrity",
    "contract_audit",
    "acceptance_none",
)


class _AutoResolve:
    """Sentinel for ``summarize_run``: resolve the scorer from the config."""


_AUTO = _AutoResolve()


class _AbstainingScorer:
    """A scorer whose every answer is unusable, so parsers stay pure-regex.

    Passing it to the control parsers runs their exact string cascade without
    ever resolving the project scorer, which is how the salvage rebuild tells
    a regex hit (production never salvaged) from a genuine miss.
    """

    def score(
        self, state: str, question: str, options: list[dict[str, str]]
    ) -> list[float] | None:
        return None


def summarize_run(
    run_dir: str | Path,
    *,
    scorer: Any = _AUTO,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Summarize everything the scorer did in one finished run directory.

    ``run_dir`` is resolved through the run-boundary helpers, and the round
    ledger is read with the manager's own reader (latest entry per
    ``round_index`` wins, malformed lines skipped). ``scorer`` defaults to
    the project-config scorer; pass one explicitly (tests, tooling) or
    ``None`` to force the "not recorded" salvage section.
    """

    run_path, role_dir = _resolve_run(run_dir)
    if scorer is _AUTO:
        scorer, configured_threshold = _resolve_scorer()
        if threshold is None:
            threshold = configured_threshold
    if threshold is None:
        threshold = DEFAULT_THRESHOLD
    rounds = _recorded_rounds(role_dir)
    return {
        "run_dir": str(run_path),
        "rounds": len(rounds),
        "salvage": _salvage_summary(rounds, scorer, threshold),
        "gate": _gate_summary(rounds),
        "cross_check": _cross_check_summary(rounds),
        "effort_routing": _effort_routing_summary(rounds),
        "report_selection": _report_selection_summary(rounds),
        "round_dedup": _round_dedup_summary(rounds),
    }


def render_summary(summary: dict[str, Any]) -> str:
    """Render one summary dict as neutral plain text."""

    lines: list[str] = []
    lines.append(
        f"Scorer stats: {summary.get('run_dir', '?')} "
        f"({summary.get('rounds', 0)} recorded round(s))"
    )
    lines.extend(_render_salvage(summary.get("salvage")))
    lines.extend(_render_gate(summary.get("gate")))
    lines.extend(_render_cross_check(summary.get("cross_check")))
    lines.extend(_render_effort_routing(summary.get("effort_routing")))
    lines.extend(_render_report_selection(summary.get("report_selection")))
    lines.extend(_render_round_dedup(summary.get("round_dedup")))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Per-feature summaries
# ---------------------------------------------------------------------------


def _salvage_summary(
    rounds: list[ManagedRound], scorer: SemanticScorer | None, threshold: float
) -> dict[str, Any]:
    """Rebuild the salvage ledger by re-parsing every round with the scorer.

    Salvage only ever runs on the regex-miss path, so re-running the same
    parsers with a live scorer reproduces exactly the lines that were (or
    would be) rescued: control header lines and "no blocking constraints"
    phrasings from ``auditor_report`` via ``parse_audit_report``, and the
    manager route line from ``plan_text`` when the exact string cascade
    missed it.
    """

    rescued = {
        control: {"count": 0, "max_probability": None}
        for control in SALVAGE_CONTROLS
    }
    records: list[dict[str, Any]] = []
    if scorer is not None:
        for item in rounds:
            if item.auditor_report.strip():
                try:
                    report = parse_audit_report(
                        item.auditor_report,
                        item.round_index,
                        scorer=scorer,
                        threshold=threshold,
                    )
                except Exception:
                    # The rebuild must never crash on one odd report.
                    report = None
                if report is not None:
                    records.extend(
                        entry
                        for entry in report.control_salvage
                        if isinstance(entry, dict)
                    )
            route = _route_salvage_record(item, scorer, threshold)
            if route is not None:
                records.append(route)
    for record in records:
        control = str(record.get("control") or "")
        if control not in rescued:
            continue
        rescued[control]["count"] += 1
        probability = _winner_probability(record)
        if probability is not None:
            current = rescued[control]["max_probability"]
            rescued[control]["max_probability"] = (
                probability if current is None else max(current, probability)
            )
    return {"recorded": scorer is not None, "rescued": rescued}


def _route_salvage_record(
    item: ManagedRound, scorer: SemanticScorer, threshold: float
) -> dict[str, Any] | None:
    """One salvage record for the round's route line, or None.

    Mirrors ``role_prompts._salvage_manager_route``: the first route-label
    line is re-judged only when the exact string cascade missed it (a regex
    hit never reaches the scorer in production).
    """

    plan_text = str(item.plan_text or "")
    line = next(
        (
            stripped
            for stripped in (raw.strip() for raw in plan_text.splitlines())
            if _MANAGER_ROUTE_LABEL_LINE_RE.match(stripped)
        ),
        None,
    )
    if line is None:
        return None
    try:
        strict = parse_role_manager_next_step(plan_text, scorer=_AbstainingScorer())
        if strict != MANAGER_NEXT_INVALID:
            return None
        result = salvage_control_value(
            line,
            _MANAGER_ROUTE_LEGAL_VALUES,
            _MANAGER_ROUTE_VALUE_DESCRIPTIONS,
            scorer,
            threshold,
        )
    except Exception:
        return None
    if result is None:
        return None
    return {
        "control": "route",
        "value": result.value,
        "probabilities": dict(zip(_MANAGER_ROUTE_LEGAL_VALUES, result.probabilities)),
    }


def _gate_summary(rounds: list[ManagedRound]) -> dict[str, Any]:
    """Auditor-fast pre-gate counts and the estimated saved audit minutes.

    A slow audit is any round whose slow auditor episode actually ran (a
    numeric ``duration_ms``); gate skips carry no duration. The saved-minutes
    figure is skips times this run's own median slow-audit duration -- an
    estimate, printed as the count alone when the run has no slow audits to
    measure against.
    """

    evaluations = 0
    skips = 0
    slow_durations: list[float] = []
    for item in rounds:
        status = _status_dict(item.auditor_status)
        skipped = str(status.get("status") or "") == "skipped_by_fast_gate"
        if skipped:
            skips += 1
        if isinstance(status.get("auditor_fast_gate"), dict):
            evaluations += 1
        duration = status.get("duration_ms")
        if (
            not skipped
            and isinstance(duration, (int, float))
            and not isinstance(duration, bool)
        ):
            slow_durations.append(float(duration))
    median_slow_ms = _median(slow_durations)
    estimated = (
        skips * median_slow_ms / 60_000.0
        if skips and median_slow_ms is not None
        else None
    )
    return {
        "evaluations": evaluations,
        "passes": max(0, evaluations - skips),
        "skips": skips,
        "slow_audits": len(slow_durations),
        "median_slow_ms": median_slow_ms,
        "estimated_minutes_saved": estimated,
    }


def _cross_check_summary(rounds: list[ManagedRound]) -> dict[str, Any]:
    """Post-audit verdict cross-check: agreements and flagged disagreements."""

    rounds_checked = 0
    verdicts = 0
    agreements = 0
    disagreements: list[dict[str, Any]] = []
    for item in rounds:
        cross = _status_dict(item.auditor_status).get("auditor_cross_check")
        if not isinstance(cross, dict):
            continue
        rounds_checked += 1
        controls = cross.get("controls")
        if isinstance(controls, list):
            verdicts += len(controls)
        agreed = cross.get("agreements")
        if isinstance(agreed, list):
            agreements += len(agreed)
        flagged = cross.get("disagreements")
        if not isinstance(flagged, list):
            continue
        for entry in flagged:
            if not isinstance(entry, dict):
                continue
            disagreements.append(
                {
                    "round": item.round_index,
                    "control": str(entry.get("control") or ""),
                    "auditor_verdict": str(entry.get("auditor_verdict") or ""),
                    "scorer_answer": str(entry.get("scorer_answer") or ""),
                    "scorer_probability": _number_or_none(
                        entry.get("probability")
                    ),
                }
            )
    return {
        "rounds": rounds_checked,
        "verdicts_checked": verdicts,
        "agreements": agreements,
        "flagged_disagreements": disagreements,
    }


def _effort_routing_summary(rounds: list[ManagedRound]) -> dict[str, Any]:
    """Rounds per effort variant, with routing confidence and executor time."""

    routed = 0
    escalations = 0
    variants: dict[str, dict[str, Any]] = {}
    for item in rounds:
        status = _status_dict(item.executor_status)
        routing = status.get("effort_routing")
        if not isinstance(routing, dict) or not routing:
            continue
        routed += 1
        if routing.get("escalated"):
            escalations += 1
        variant = str(routing.get("variant") or "unknown")
        entry = variants.setdefault(
            variant, {"rounds": 0, "probabilities": [], "durations_ms": []}
        )
        entry["rounds"] += 1
        classified = routing.get("classified")
        probabilities = routing.get("probabilities")
        if (
            isinstance(probabilities, dict)
            and isinstance(classified, str)
            and isinstance(probabilities.get(classified), (int, float))
            and not isinstance(probabilities.get(classified), bool)
        ):
            entry["probabilities"].append(float(probabilities[classified]))
        duration = status.get("duration_ms")
        if isinstance(duration, (int, float)) and not isinstance(duration, bool):
            entry["durations_ms"].append(float(duration))
    return {
        "rounds": routed,
        "escalations": escalations,
        "variants": {
            name: {
                "rounds": data["rounds"],
                "mean_probability": _mean(data["probabilities"]),
                "mean_duration_ms": _mean(data["durations_ms"]),
            }
            for name, data in sorted(variants.items())
        },
    }


def _report_selection_summary(rounds: list[ManagedRound]) -> dict[str, Any]:
    """Rounds that ran a report selection, and the candidate split."""

    rounds_with = 0
    kept = 0
    dropped = 0
    degraded = 0
    for item in rounds:
        selection = _status_dict(item.executor_status).get("report_selection")
        if not isinstance(selection, dict):
            continue
        rounds_with += 1
        if selection.get("degraded"):
            degraded += 1
        kept_entries = selection.get("kept")
        if isinstance(kept_entries, list):
            kept += len(kept_entries)
        dropped_entries = selection.get("dropped")
        if isinstance(dropped_entries, list):
            dropped += len(dropped_entries)
    return {
        "rounds": rounds_with,
        "candidates": kept + dropped,
        "kept": kept,
        "dropped": dropped,
        "degraded": degraded,
    }


def _round_dedup_summary(rounds: list[ManagedRound]) -> dict[str, Any]:
    """Round-dedup comparisons and the flagged repeats with probabilities."""

    comparisons = 0
    flags: list[dict[str, Any]] = []
    for item in rounds:
        dedup = _status_dict(item.manager_status).get("round_dedup")
        if not isinstance(dedup, dict):
            continue
        comparisons += 1
        if dedup.get("flagged"):
            flags.append(
                {
                    "round": item.round_index,
                    "probability": _number_or_none(dedup.get("probability")),
                }
            )
    return {"comparisons": comparisons, "flags": flags}


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _render_salvage(salvage: Any) -> list[str]:
    if not isinstance(salvage, dict):
        return ["Semantic salvage: not recorded"]
    rescued = salvage.get("rescued")
    rescued = rescued if isinstance(rescued, dict) else {}
    if not salvage.get("recorded"):
        return ["Semantic salvage: not recorded (no scorer configured)"]
    lines = ["Semantic salvage: recorded (rebuilt by re-parsing this run)"]
    for control in SALVAGE_CONTROLS:
        entry = rescued.get(control)
        entry = entry if isinstance(entry, dict) else {}
        count = entry.get("count")
        count = count if isinstance(count, int) else 0
        line = f"  {control}: {count} line(s) rescued"
        probability = entry.get("max_probability")
        if count and probability is not None:
            line += f", max probability {_fmt_probability(probability)}"
        lines.append(line)
    return lines


def _render_gate(gate: Any) -> list[str]:
    if not isinstance(gate, dict):
        return ["Auditor-fast gate: evaluations 0, passes 0, skips 0"]
    lines = [
        "Auditor-fast gate: evaluations {evaluations}, passes {passes}, "
        "skips {skips}".format(
            evaluations=gate.get("evaluations", 0),
            passes=gate.get("passes", 0),
            skips=gate.get("skips", 0),
        )
    ]
    slow_audits = gate.get("slow_audits")
    slow_audits = slow_audits if isinstance(slow_audits, int) else 0
    median = gate.get("median_slow_ms")
    lines.append(f"  slow audits: {slow_audits}")
    if isinstance(median, (int, float)):
        lines[-1] += f", median duration {_fmt_seconds(median / 1000.0)}"
    estimated = gate.get("estimated_minutes_saved")
    if isinstance(estimated, (int, float)):
        lines.append(
            f"  estimated audit minutes saved: {_fmt_minutes(estimated)} "
            "(estimate: skips x this run's median slow-audit duration)"
        )
    elif gate.get("skips"):
        lines.append(
            "  estimated audit minutes saved: not estimable "
            "(no slow audits in this run); skip count above stands alone"
        )
    return lines


def _render_cross_check(cross: Any) -> list[str]:
    if not isinstance(cross, dict):
        return ["Cross-check: verdicts checked 0, agreements 0, flagged disagreements 0"]
    lines = [
        "Cross-check: verdicts checked {verdicts}, agreements {agreements}, "
        "flagged disagreements {flagged}".format(
            verdicts=cross.get("verdicts_checked", 0),
            agreements=cross.get("agreements", 0),
            flagged=len(cross.get("flagged_disagreements") or []),
        )
    ]
    for entry in cross.get("flagged_disagreements") or []:
        if not isinstance(entry, dict):
            continue
        probability = entry.get("scorer_probability")
        confidence = (
            f", p={probability:.3f}" if isinstance(probability, (int, float)) else ""
        )
        lines.append(
            "  round {round} {control}: auditor={auditor}, scorer={scorer}{confidence}".format(
                round=entry.get("round", "?"),
                control=entry.get("control", "?"),
                auditor=entry.get("auditor_verdict", "?"),
                scorer=entry.get("scorer_answer", "?"),
                confidence=confidence,
            )
        )
    return lines


def _render_effort_routing(routing: Any) -> list[str]:
    if not isinstance(routing, dict):
        return ["Effort routing: rounds 0, escalations 0"]
    lines = [
        "Effort routing: rounds {rounds}, escalations {escalations}".format(
            rounds=routing.get("rounds", 0),
            escalations=routing.get("escalations", 0),
        )
    ]
    variants = routing.get("variants")
    for name, entry in sorted(
        (variants or {}).items()
        if isinstance(variants, dict)
        else {}
    ):
        if not isinstance(entry, dict):
            continue
        line = "  {name}: {rounds} round(s)".format(
            name=name, rounds=entry.get("rounds", 0)
        )
        probability = entry.get("mean_probability")
        if probability is not None:
            line += f", mean routing probability {_fmt_probability(probability)}"
        duration = entry.get("mean_duration_ms")
        if duration is not None:
            line += f", mean executor duration {_fmt_seconds(duration / 1000.0)}"
        lines.append(line)
    return lines


def _render_report_selection(selection: Any) -> list[str]:
    if not isinstance(selection, dict):
        return [
            "Report selection: rounds with selection 0, candidates 0, "
            "kept 0, dropped 0, degraded 0"
        ]
    return [
        "Report selection: rounds with selection {rounds}, candidates {candidates}, "
        "kept {kept}, dropped {dropped}, degraded {degraded}".format(
            rounds=selection.get("rounds", 0),
            candidates=selection.get("candidates", 0),
            kept=selection.get("kept", 0),
            dropped=selection.get("dropped", 0),
            degraded=selection.get("degraded", 0),
        )
    ]


def _render_round_dedup(dedup: Any) -> list[str]:
    if not isinstance(dedup, dict):
        return ["Round dedup: comparisons 0, flags 0"]
    flags = dedup.get("flags")
    flags = flags if isinstance(flags, list) else []
    lines = [
        "Round dedup: comparisons {comparisons}, flags {flags}".format(
            comparisons=dedup.get("comparisons", 0),
            flags=len(flags),
        )
    ]
    for entry in flags:
        if not isinstance(entry, dict):
            continue
        probability = entry.get("probability")
        confidence = (
            f" (same-work probability {probability:.3f})"
            if isinstance(probability, (int, float))
            else ""
        )
        lines.append(f"  round {entry.get('round', '?')} flagged{confidence}")
    return lines


def _fmt_probability(value: float) -> str:
    return f"{value:.3f}"


def _fmt_seconds(value: float) -> str:
    return f"{value:.1f} s"


def _fmt_minutes(value: float) -> str:
    return f"{value:.1f}"


# ---------------------------------------------------------------------------
# Fixture mining
# ---------------------------------------------------------------------------

# Per control: the label regex locates the header line, the value regex
# proves the exact parser accepted it, and infer yields the canonical value
# for the gold label. Only regex-accepted lines are mined: a line the value
# regex missed has no trustworthy canonical value (the infer_* fallback
# would fabricate one), which is exactly the review flow the original
# mining pipeline kept out of the fixture.
_FIXTURE_CONTROLS: dict[
    str,
    tuple[re.Pattern[str], re.Pattern[str], list[str], dict[str, str], Callable[[str], str]],
] = {
    "status": (
        _STATUS_LABEL_LINE_RE,
        _STATUS_CONTROL_LINE_RE,
        _STATUS_LEGAL_VALUES,
        _STATUS_VALUE_DESCRIPTIONS,
        infer_report_status,
    ),
    "integrity": (
        _INTEGRITY_LABEL_LINE_RE,
        _INTEGRITY_CONTROL_LINE_RE,
        _INTEGRITY_LEGAL_VALUES,
        _INTEGRITY_VALUE_DESCRIPTIONS,
        lambda text: infer_integrity_findings(text)[0],
    ),
    "contract_audit": (
        _CONTRACT_AUDIT_LABEL_LINE_RE,
        _CONTRACT_AUDIT_CONTROL_LINE_RE,
        _CONTRACT_AUDIT_LEGAL_VALUES,
        _CONTRACT_AUDIT_VALUE_DESCRIPTIONS,
        infer_contract_audit_status,
    ),
}

_FIXTURE_WINDOW_LINES = 8


def mine_fixture_rows(run_dir: str | Path) -> list[dict[str, Any]]:
    """Mine one run's control lines into SemIf decision rows.

    Manager route lines come from ``plan_text`` (the last route-label line,
    the one the harness plan extractor keys on) with gold = the record's own
    ``next_step``; invalid routes are skipped. Audit header lines come from
    the first non-empty lines of ``auditor_report`` with the canonical value
    the ``infer_*`` parsers derive -- but only when the exact value regex
    accepted the line (a miss has no trustworthy gold and needs a human
    label, as in the original mining pipeline). Row ids embed the run id and
    round, so ``append_fixture_rows`` can dedup by id across invocations.
    """

    run_path, role_dir = _resolve_run(run_dir)
    run_id = run_path.name
    rows: list[dict[str, Any]] = []
    for item in _recorded_rounds(role_dir):
        rid = f"{run_id}#r{item.round_index}"
        plan = str(item.plan_text or "")
        candidates = [
            line
            for line in plan.splitlines()
            if _MANAGER_ROUTE_LABEL_LINE_RE.match(line.strip())
        ]
        if candidates:
            gold = item.next_step
            if gold != MANAGER_NEXT_INVALID and gold in _MANAGER_ROUTE_LEGAL_VALUES:
                rows.append(
                    {
                        "id": f"{rid}-route",
                        "kind": "route",
                        "state": _norm_line(candidates[-1]),
                        "question": SALVAGE_QUESTION,
                        "options": [
                            {"id": value, "description": _MANAGER_ROUTE_VALUE_DESCRIPTIONS[value]}
                            for value in _MANAGER_ROUTE_LEGAL_VALUES
                        ],
                        "label": _MANAGER_ROUTE_LEGAL_VALUES.index(gold),
                    }
                )
        report = str(item.auditor_report or "")
        window = [line for line in report.splitlines() if line.strip()][
            :_FIXTURE_WINDOW_LINES
        ]
        for control, (
            label_re,
            value_re,
            values,
            descriptions,
            infer,
        ) in _FIXTURE_CONTROLS.items():
            hits = [line for line in window if label_re.match(line)]
            if not hits or not value_re.match(hits[0].strip()):
                continue
            canonical = infer(_norm_line(hits[0]) + "\nx\nx")
            if canonical not in values:
                continue
            rows.append(
                {
                    "id": f"{rid}-{control}",
                    "kind": control,
                    "state": _norm_line(hits[0]),
                    "question": SALVAGE_QUESTION,
                    "options": [
                        {"id": value, "description": descriptions[value]}
                        for value in values
                    ],
                    "label": values.index(canonical),
                }
            )
    return rows


def append_fixture_rows(
    rows: list[dict[str, Any]], fixture_path: str | Path
) -> int:
    """Append new decision rows to the fixture, dedup by id; return count added."""

    path = Path(fixture_path)
    existing: set[str] = set()
    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except ValueError:
                continue
            if isinstance(payload, dict) and isinstance(payload.get("id"), str):
                existing.add(payload["id"])
    fresh: list[dict[str, Any]] = []
    seen = set(existing)
    for row in rows:
        row_id = row.get("id")
        if not isinstance(row_id, str) or row_id in seen:
            continue
        seen.add(row_id)
        fresh.append(row)
    if fresh:
        path.parent.mkdir(parents=True, exist_ok=True)
        with io.open(path, "a", encoding="utf-8") as handle:
            for row in fresh:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(fresh)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _resolve_run(run_dir: str | Path) -> tuple[Path, Path]:
    """Resolve a run directory through the boundary helpers.

    Returns ``(run_path, role_dir)``. The run must be exactly one directory
    below its parent (the runs root the caller named); the role ledger may
    be missing entirely for a reserved or partially written run.
    """

    run_path = Path(run_dir).expanduser()
    role_dir = safe_run_role(run_path.parent, run_path, allow_missing=True)
    if role_dir is None:
        raise ValueError(f"not a readable run directory: {run_dir}")
    return run_path, role_dir


def _resolve_scorer() -> tuple[SemanticScorer | None, float]:
    """The project-config scorer and salvage threshold; failure = none."""

    try:
        defaults = load_run_defaults()
    except Exception:
        return None, DEFAULT_THRESHOLD
    scorer = scorer_from_config(defaults)
    threshold = defaults.get("semif_threshold", DEFAULT_THRESHOLD)
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        threshold = DEFAULT_THRESHOLD
    return scorer, float(threshold)


def _status_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _winner_probability(record: dict[str, Any]) -> float | None:
    probabilities = record.get("probabilities")
    if not isinstance(probabilities, dict):
        return None
    probability = probabilities.get(record.get("value"))
    if isinstance(probability, (int, float)) and not isinstance(probability, bool):
        return float(probability)
    return None


def _number_or_none(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _norm_line(raw: str) -> str:
    return re.sub(r"\s+", " ", raw.strip())
