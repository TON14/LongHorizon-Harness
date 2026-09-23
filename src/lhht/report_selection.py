"""Pointwise scorer selection of related past audit reports for role prompts.

The executor and auditor prompts carry the past audit reports the manager
referenced (``round_NNN`` refs). In long runs the report history grows and
every referenced report rides along wholesale, while the current subtask
usually needs only a few. When ``[run.semif].report_selection`` enables it,
the same resident scorer the salvage layer configures ranks every available
past audit report against the round's plan: one pointwise row per candidate
-- state = the plan text plus that candidate's condensed report, question
"Is this past audit report relevant to the current subtask?", options
relevant/irrelevant -- and the prompt keeps the top-K by P(relevant) plus
every explicitly referenced round (explicit references always win).

Batching: every row shares the plan-text prefix, but the ``SemanticScorer``
protocol (and the ``SemifCliScorer`` behind it) scores exactly one row per
``score()`` invocation, so selection loops one call per candidate; a scorer
that grows a multi-row API only needs to replace that loop.

Safety: selection is an optimization, never a gate. With the switch absent
or false no scorer is built or called; a scorer failure, timeout, or empty
candidate list leaves the round with exactly today's referenced-reports set;
explicit references are kept whatever the scores say. Nothing here can raise
into the management loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

from .config import load_run_defaults
from .semantic_salvage import (
    SemanticScorer,
    _valid_probabilities,
    scorer_from_config,
)
from .types import ManagedRound

DEFAULT_REPORT_SELECTION_K = 3
DEFAULT_REPORT_SELECTION_THRESHOLD = 0.6

SELECTION_QUESTION = "Is this past audit report relevant to the current subtask?"
_SELECTION_OPTIONS = (
    {
        "id": "relevant",
        "description": (
            "The report's findings bear directly on the current subtask."
        ),
    },
    {
        "id": "irrelevant",
        "description": (
            "The report's findings have no bearing on the current subtask."
        ),
    },
)

# Head+tail caps keep one row (and therefore the whole batch) bounded: the
# plan prefix repeats in every row, and a long audit report only needs its
# opening verdicts and closing state update to judge relevance.
_MAX_PLAN_CHARS = 4_000
_MAX_CANDIDATE_CHARS = 2_000


class ReportCandidate(NamedTuple):
    """One available past audit report: ``round_NNN`` id plus report text."""

    id: str
    text: str


class Selection(NamedTuple):
    kept: list[ReportCandidate]
    dropped: list[ReportCandidate]
    probabilities: dict[str, float]
    degraded: bool


def report_candidates(rounds: list[ManagedRound]) -> list[ReportCandidate]:
    """One candidate per recorded round that carries an audit report."""
    return [
        ReportCandidate(
            f"round_{item.round_index:03d}", item.auditor_report.strip()
        )
        for item in rounds
        if item.auditor_report.strip()
    ]


def _condense(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head_chars = max(1, int(max_chars * 0.65))
    tail_chars = max(1, max_chars - head_chars)
    return (
        text[:head_chars].rstrip()
        + f"\n...[truncated {len(text) - max_chars} chars]...\n"
        + text[-tail_chars:].lstrip()
    )


def select_reports(
    scorer: SemanticScorer,
    plan_text: str,
    candidates: list[ReportCandidate],
    k: int,
    threshold: float,
) -> Selection:
    """Rank candidates against the plan; keep P(relevant) >= threshold, top-K.

    One pointwise row per candidate: state = the round's (condensed) plan
    text followed by the candidate's (condensed) report, question
    ``SELECTION_QUESTION``, options relevant/irrelevant. Kept candidates are
    ordered by probability (ties keep candidate order); ``k`` caps how many
    survive. Rows share the plan prefix but go through the scorer one
    ``score()`` call each -- the protocol is single-row (see the module
    docstring). An empty candidate list, and any scorer failure (exception,
    ``None`` answer, or malformed probabilities on any row), degrades to
    today's behavior: every candidate is returned unchanged, unscored.
    """
    if not candidates:
        return Selection(list(candidates), [], {}, False)
    state_prefix = _condense(str(plan_text or ""), _MAX_PLAN_CHARS)
    scored: list[tuple[ReportCandidate, float]] = []
    for candidate in candidates:
        row_state = (
            f"Current subtask plan:\n{state_prefix}\n\n"
            f"Past audit report {candidate.id}:\n"
            + _condense(candidate.text, _MAX_CANDIDATE_CHARS)
        )
        try:
            probabilities = scorer.score(
                row_state, SELECTION_QUESTION, list(_SELECTION_OPTIONS)
            )
        except Exception:
            return Selection(list(candidates), [], {}, True)
        if not _valid_probabilities(probabilities, len(_SELECTION_OPTIONS)):
            return Selection(list(candidates), [], {}, True)
        scored.append((candidate, float(probabilities[0])))
    kept_scores = sorted(
        ((candidate, probability) for candidate, probability in scored if probability >= threshold),
        key=lambda item: item[1],
        reverse=True,
    )[: max(0, int(k))]
    kept_ids = {candidate.id for candidate, _ in kept_scores}
    return Selection(
        [candidate for candidate, _ in kept_scores],
        [candidate for candidate in candidates if candidate.id not in kept_ids],
        {candidate.id: probability for candidate, probability in scored},
        False,
    )


@dataclass(frozen=True)
class ReportSelector:
    """Run-scoped selection state: scorer, K, threshold."""

    scorer: SemanticScorer
    k: int = DEFAULT_REPORT_SELECTION_K
    threshold: float = DEFAULT_REPORT_SELECTION_THRESHOLD


def report_selector_from_defaults(
    defaults: dict[str, Any],
) -> ReportSelector | None:
    """Resolve the selector an optional ``[run.semif]`` table configures.

    Like the auditor-fast gate and effort routing, selection is an
    optimization: an absent or false flag, or an unusable scorer, silently
    keeps today's referenced-reports behavior.
    """
    if not defaults.get("semif_report_selection"):
        return None
    scorer = scorer_from_config(defaults)
    if scorer is None:
        return None
    k = defaults.get("semif_report_selection_k", DEFAULT_REPORT_SELECTION_K)
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        k = DEFAULT_REPORT_SELECTION_K
    threshold = defaults.get(
        "semif_report_selection_threshold", DEFAULT_REPORT_SELECTION_THRESHOLD
    )
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        threshold = DEFAULT_REPORT_SELECTION_THRESHOLD
    return ReportSelector(scorer, k, float(threshold))


def resolve_report_selector() -> ReportSelector | None:
    """Resolve the selector once per run from the project config; failure = off."""
    try:
        return report_selector_from_defaults(load_run_defaults())
    except Exception:
        # A missing or broken config must disable selection, never crash a run.
        return None


def select_related_reports(
    selector: ReportSelector,
    plan_text: str,
    candidates: list[ReportCandidate],
    refs: list[str],
) -> Selection:
    """Apply one round's selection: explicit refs always win.

    Every candidate is scored -- explicit references included, so their
    probabilities land in the record for measurement -- but a candidate whose
    id matches a ref is kept whatever its score, while the rest compete for
    the K slots by P(relevant) >= threshold. On degradation the round keeps
    exactly today's set, the referenced candidates, so a scorer failure can
    neither drop an explicit reference nor smuggle in extra reports. The kept
    list preserves candidate order (the prompt formatter emits rounds in run
    order anyway); the ranking itself lives in ``probabilities``.
    """
    explicit_ids = {
        str(ref).strip().lower() for ref in refs if str(ref or "").strip()
    }
    explicit = [
        candidate
        for candidate in candidates
        if candidate.id.lower() in explicit_ids
    ]
    selection = select_reports(
        selector.scorer, plan_text, candidates, selector.k, selector.threshold
    )
    if selection.degraded:
        return Selection(explicit, [], {}, True)
    kept_ids = {candidate.id for candidate in selection.kept} | {
        candidate.id for candidate in explicit
    }
    return Selection(
        [candidate for candidate in candidates if candidate.id in kept_ids],
        # A ref-scored-irrelevant candidate is kept, not dropped: dropped is
        # exactly the complement of kept, so the two lists never overlap.
        [candidate for candidate in candidates if candidate.id not in kept_ids],
        selection.probabilities,
        False,
    )


def selection_record(
    selection: Selection,
    *,
    refs: list[str],
    k: int,
    threshold: float,
) -> dict[str, Any]:
    """The measurement payload stored on the round that ran a selection."""
    return {
        "kept": [
            {
                "id": candidate.id,
                "probability": selection.probabilities.get(candidate.id),
            }
            for candidate in selection.kept
        ],
        "dropped": [
            {
                "id": candidate.id,
                "probability": selection.probabilities.get(candidate.id),
            }
            for candidate in selection.dropped
        ],
        "explicit_refs": [str(ref) for ref in refs],
        "degraded": selection.degraded,
        "k": k,
        "threshold": threshold,
    }
