"""Advisory detector for the manager re-planning the same subtask.

Long-horizon failure mode: the manager re-plans essentially the same subtask
round after round -- phrasing varies, work repeats -- and burns the round
budget in a loop nothing surfaces. When ``[run.semif]`` enables
``round_dedup``, each round's manager plan is compared with the previous
round's plan by one SemIf decision: a confident "essentially the same work"
marks the round as a possible loop.

The detector is purely advisory: the record on the round and the
``round_dedup_flag`` event are annotations for the operator. Routing,
feedback, the plan text, and every other field of the round record stay
exactly as today; the detector never blocks, rewrites, or re-judges
anything. Any scorer failure, a missing previous plan, or an absent/disabled
switch leaves no record at all. See docs/round-dedup.md for the formats.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import load_run_defaults
from .semantic_salvage import (
    SemanticScorer,
    _valid_probabilities,
    scorer_from_config,
)

DEFAULT_REPEAT_THRESHOLD = 0.9

REPEAT_QUESTION = (
    "Is this round's planned subtask essentially the same work as the previous round's?"
)

_REPEAT_OPTIONS = [
    {
        "id": "same_work",
        "description": "The two plans describe essentially the same subtask work.",
    },
    {
        "id": "different_work",
        "description": "The two plans describe substantially different work.",
    },
]

# The shim's local model answers narrow questions on small states, so each
# plan slice is bounded exactly like the gate's plan evidence.
_MAX_PLAN_CHARS = 1_500


@dataclass
class Repeat:
    """One advisory same-work decision over two consecutive manager plans.

    ``same`` is True only for a confident ``same_work`` win (argmax plus a
    probability >= threshold); ``probability`` is the scorer's mass on
    ``same_work`` -- the comparison measurement the record carries even when
    the answer is not confident enough to flag.
    """

    same: bool
    probability: float
    probabilities: dict[str, float] = field(default_factory=dict)
    threshold: float = DEFAULT_REPEAT_THRESHOLD

    @property
    def flagged(self) -> bool:
        return self.same

    def payload(self) -> dict[str, Any]:
        return {
            "flagged": self.flagged,
            "same": self.same,
            "probability": self.probability,
            "probabilities": dict(self.probabilities),
            "threshold": self.threshold,
        }


def detect_repeat(
    scorer: SemanticScorer | None,
    plan_text: str,
    previous_plan_text: str,
    threshold: float = DEFAULT_REPEAT_THRESHOLD,
) -> Repeat | None:
    """Ask the scorer whether this round's plan repeats the previous round's.

    One decision over both plan texts -- each condensed, clearly labeled, and
    separated in the state. A confident ``same_work`` (the argmax option with
    probability >= ``threshold``) yields ``Repeat(same=True, ...)``; any
    other scored answer yields ``Repeat(same=False, ...)`` so the comparison
    probability stays measurable. A missing scorer, a missing plan on either
    side, or any scorer failure or unusable answer returns None: the detector
    then leaves no record and never blocks the round.
    """
    if scorer is None:
        return None
    if not str(plan_text or "").strip() or not str(previous_plan_text or "").strip():
        return None
    state = _comparison_state(plan_text, previous_plan_text)
    try:
        probabilities = scorer.score(state, REPEAT_QUESTION, _REPEAT_OPTIONS)
    except Exception:
        # The detector must never leak a scorer failure into the manager loop.
        return None
    if not _valid_probabilities(probabilities, len(_REPEAT_OPTIONS)):
        return None
    values = [float(value) for value in probabilities]
    by_id = {
        option["id"]: value for option, value in zip(_REPEAT_OPTIONS, values)
    }
    # Ties go to the earliest option, mirroring salvage semantics.
    top = max(range(len(values)), key=values.__getitem__)
    same = _REPEAT_OPTIONS[top]["id"] == "same_work" and values[top] >= threshold
    return Repeat(
        same=same,
        probability=by_id["same_work"],
        probabilities=by_id,
        threshold=float(threshold),
    )


def round_dedup_from_defaults(
    defaults: dict[str, Any],
) -> tuple[SemanticScorer | None, float]:
    """Resolve ``(scorer, threshold)`` when ``[run.semif].round_dedup`` is on.

    Like the cross-check, the detector is an advisory extra: it is silently
    off unless the switch is set AND the salvage scorer is configured, so an
    absent/disabled flag or an unusable scorer yields ``(None, default)`` and
    the caller records nothing.
    """
    if not defaults.get("semif_round_dedup"):
        return None, DEFAULT_REPEAT_THRESHOLD
    scorer = scorer_from_config(defaults)
    if scorer is None:
        return None, DEFAULT_REPEAT_THRESHOLD
    threshold = defaults.get(
        "semif_round_dedup_threshold", DEFAULT_REPEAT_THRESHOLD
    )
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        threshold = DEFAULT_REPEAT_THRESHOLD
    return scorer, float(threshold)


def resolve_round_dedup() -> tuple[SemanticScorer | None, float]:
    """Resolve the detector once per run from the project config; failure = off."""
    try:
        return round_dedup_from_defaults(load_run_defaults())
    except Exception:
        # A missing or broken config must disable the detector, never crash a run.
        return None, DEFAULT_REPEAT_THRESHOLD


def _comparison_state(plan_text: str, previous_plan_text: str) -> str:
    """Both plans, each condensed and labeled, clearly separated."""
    return (
        "Previous round's manager plan:\n"
        f"{_clip(previous_plan_text)}\n\n"
        "This round's manager plan:\n"
        f"{_clip(plan_text)}"
    )


def _clip(text: str, limit: int = _MAX_PLAN_CHARS) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"
