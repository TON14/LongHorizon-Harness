"""Fail-only scorer pre-gate in front of the slow auditor episode.

Auditor episodes are the run's biggest time sink, and many of them land on
hopeless rounds the manager bounces anyway. When ``[run.semif]`` enables
``auditor_fast``, the manager asks the local SemIf scorer a small battery of
narrow questions about LIVE evidence -- the workspace VCS status, the
executor's visible output, the task contract's acceptance constraints, and
the manager's own plan -- after the executor episode and before launching
the auditor. Only a CONFIDENT fail (a fail-side option winning with
probability >= threshold) short-circuits the slow auditor; everything else
proceeds exactly as today.

Safety rules, in order:
- the gate is fail-only: it can produce "skip the slow audit", never a
  clean/complete verdict, and the ``done`` acceptance logic never consults it;
- every criterion judges live evidence, never the executor's self-report
  prose on its own (the VCS-match question cross-checks the output against
  the repository state, it does not score the narrative's tone);
- any scorer error, timeout, or unusable answer abstains, and an abstaining
  criterion never votes fail -- a broken gate degrades to today's behavior;
- an absent/disabled config or a missing scorer keeps the gate off, so the
  run stays byte-for-byte identical to a run without this module.

The scorer is the existing ``SemifCliScorer`` subprocess sidecar (one row per
``score`` call); the battery is therefore a loop of one scorer invocation per
criterion. See docs/auditor-fast.md for the record formats.

This module also hosts the post-audit verdict cross-check: after the slow
auditor's report is parsed, ``cross_check`` asks the same scorer the
auditor's three control verdicts against the same live evidence the gate
gathers, and records confident disagreements on the round. It is purely
advisory -- it never modifies the auditor's verdicts, routing, or completion
acceptance. See docs/cross-check.md for the record formats.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import load_run_defaults
from .semantic_salvage import (
    SemanticScorer,
    _valid_probabilities,
    scorer_from_config,
)
from .types import AuditReport, HarnessConfig

DEFAULT_GATE_THRESHOLD = 0.95
DEFAULT_CROSS_CHECK_THRESHOLD = 0.9
VCS_STATUS_TIMEOUT_SECONDS = 120

# The shim's local model answers narrow questions on small states, so every
# evidence slice fed to a criterion is bounded.
_MAX_CONSTRAINTS = 8
_MAX_CONSTRAINT_CHARS = 400
_MAX_EXECUTOR_OUTPUT_CHARS = 3_000
_MAX_PLAN_CHARS = 1_500
_MAX_VCS_STATUS_CHARS = 4_000
_MAX_VCS_REPORT_LINES = 120

_CONSTRAINT_QUESTION = "Is this constraint satisfied by the evidence?"
_VCS_MATCH_QUESTION = (
    "Do the VCS status changes match the executor's claims about which files it changed?"
)
_WORKSPACE_EMPTY_QUESTION = "Is the workspace status empty of the round's expected changes?"

_CONSTRAINT_OPTIONS = [
    {"id": "yes", "description": "The evidence shows the constraint is satisfied."},
    {"id": "no", "description": "The evidence shows the constraint is violated or unmet."},
    {"id": "undetermined", "description": "The evidence is insufficient to decide."},
]
_VCS_MATCH_OPTIONS = [
    {"id": "match", "description": "The status changes agree with the claimed file changes."},
    {
        "id": "partial",
        "description": "Some claimed changes appear in the status; others are unclear.",
    },
    {"id": "no_changes_claimed", "description": "The executor claimed it changed no files."},
    {
        "id": "mismatch",
        "description": "The status contradicts the executor's claims about changed files.",
    },
]
_WORKSPACE_EMPTY_OPTIONS = [
    {
        "id": "yes",
        "description": "The workspace status shows none of the round's expected changes.",
    },
    {"id": "no", "description": "The workspace status shows changes from this round."},
]

_CROSS_CHECK_STATUS_QUESTION = "Given the evidence, is the task status complete?"
_CROSS_CHECK_INTEGRITY_QUESTION = "Given the evidence, is the integrity status clean?"
_CROSS_CHECK_CONTRACT_QUESTION = "Given the evidence, is the contract audit aligned?"

# The options are the auditor's own legal verdict values, so a scorer answer
# maps one-to-one onto what the parsed control header already recorded.
_CROSS_CHECK_STATUS_OPTIONS = [
    {"id": "complete", "description": "The evidence shows the task reached its target state."},
    {"id": "incomplete", "description": "The evidence shows work remains or claims are unverified."},
    {"id": "blocked", "description": "The evidence shows a blocker the executor cannot pass."},
]
_CROSS_CHECK_INTEGRITY_OPTIONS = [
    {"id": "clean", "description": "The evidence shows no integrity problem."},
    {"id": "suspect", "description": "The evidence raises integrity doubts."},
    {"id": "violation", "description": "The evidence shows an integrity violation."},
]
_CROSS_CHECK_CONTRACT_OPTIONS = [
    {"id": "aligned", "description": "The evidence matches the task contract."},
    {"id": "unknown", "description": "The evidence is insufficient to judge contract alignment."},
    {"id": "needs_revision", "description": "The evidence shows the task contract needs revision."},
    {"id": "invalid", "description": "The evidence shows the round or contract is invalid."},
]

_CROSS_CHECK_CONTROLS: tuple[tuple[str, str, list[dict[str, str]]], ...] = (
    ("status", _CROSS_CHECK_STATUS_QUESTION, _CROSS_CHECK_STATUS_OPTIONS),
    ("integrity", _CROSS_CHECK_INTEGRITY_QUESTION, _CROSS_CHECK_INTEGRITY_OPTIONS),
    ("contract", _CROSS_CHECK_CONTRACT_QUESTION, _CROSS_CHECK_CONTRACT_OPTIONS),
)


@dataclass
class RoundContext:
    """The live per-round facts the gate is allowed to see.

    The workspace itself comes from the HarnessConfig passed to
    ``gather_gate_evidence``; this context carries only round-specific text
    plus the effective guard exclusions.
    """

    plan_text: str = ""
    executor_output: str = ""
    task_contract: str = ""
    guard_exclude_paths: tuple[str, ...] = ()


@dataclass
class GateEvidence:
    """Live evidence gathered after the executor episode, before any audit."""

    vcs_status: str = ""
    vcs_tool: str = ""
    vcs_status_available: bool = False
    notes: list[str] = field(default_factory=list)
    executor_output: str = ""
    plan_text: str = ""
    task_contract: str = ""
    acceptance_constraints: list[str] = field(default_factory=list)

    def digest(self) -> dict[str, Any]:
        return {
            "vcs_status_available": self.vcs_status_available,
            "vcs_tool": self.vcs_tool,
            "vcs_status_line_count": len(
                [line for line in self.vcs_status.splitlines() if line.strip()]
            ),
            "executor_output_chars": len(self.executor_output),
            "acceptance_constraint_count": len(self.acceptance_constraints),
            "notes": list(self.notes),
        }


@dataclass
class GateDecision:
    verdict: str
    findings: list[dict[str, Any]] = field(default_factory=list)
    battery: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class CrossCheck:
    """Advisory comparison of the scorer's verdicts with the auditor's own.

    ``controls`` carries one entry per control with both sides' probabilities;
    ``agreements`` / ``disagreements`` are the subsets where the scorer's
    argmax matched the auditor's verdict / was confidently different.
    """

    threshold: float
    controls: list[dict[str, Any]] = field(default_factory=list)

    @property
    def agreements(self) -> list[dict[str, Any]]:
        return [entry for entry in self.controls if entry["agrees"]]

    @property
    def disagreements(self) -> list[dict[str, Any]]:
        return [entry for entry in self.controls if entry["flagged"]]

    @property
    def flagged(self) -> bool:
        return bool(self.disagreements)

    def payload(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "flagged": self.flagged,
            "agreements": self.agreements,
            "disagreements": self.disagreements,
            "controls": self.controls,
        }


def gate_from_defaults(defaults: dict[str, Any]) -> tuple[SemanticScorer | None, float]:
    """Resolve ``(scorer, threshold)`` when ``[run.semif].auditor_fast`` is on.

    The gate needs the same scorer the salvage layer uses, so it is silently
    off unless that scorer is configured: an absent/disabled flag or an
    unusable scorer yields ``(None, default)`` and the caller skips the gate.
    """
    if not defaults.get("semif_auditor_fast"):
        return None, DEFAULT_GATE_THRESHOLD
    scorer = scorer_from_config(defaults)
    if scorer is None:
        return None, DEFAULT_GATE_THRESHOLD
    threshold = defaults.get("semif_auditor_fast_threshold", DEFAULT_GATE_THRESHOLD)
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        threshold = DEFAULT_GATE_THRESHOLD
    return scorer, float(threshold)


def resolve_fast_gate() -> tuple[SemanticScorer | None, float]:
    """Resolve the gate once per run from the project config; failure = off."""
    try:
        return gate_from_defaults(load_run_defaults())
    except Exception:
        # A missing or broken config must disable the gate, never crash a run.
        return None, DEFAULT_GATE_THRESHOLD


def gather_gate_evidence(config: HarnessConfig, round_ctx: RoundContext) -> GateEvidence:
    """Collect the live evidence for one round's gate, best-effort.

    VCS status comes from ``git status --porcelain=v1 -b`` with an ``svn
    status`` fallback, each capped at 120 s; any failure or timeout only adds
    a note -- the two VCS-dependent criteria then abstain. Status lines whose
    path falls under a ``guard_exclude_paths`` entry are filtered exactly like
    the auditor guard's snapshot exclusions, so excluded churn (build outputs)
    cannot vote.
    """
    workspace = str(config.workspace_path or "")
    notes: list[str] = []
    status, tool = _workspace_vcs_status(workspace, notes)
    excludes = tuple(round_ctx.guard_exclude_paths)
    if status and excludes:
        lines = status.splitlines()
        kept = [line for line in lines if not _line_excluded(line, workspace, excludes)]
        dropped = len(lines) - len(kept)
        if dropped:
            notes.append(f"filtered {dropped} status line(s) matching guard_exclude_paths")
        status = "\n".join(kept)
    return GateEvidence(
        vcs_status=status,
        vcs_tool=tool,
        vcs_status_available=bool(tool),
        notes=notes,
        executor_output=str(round_ctx.executor_output or ""),
        plan_text=str(round_ctx.plan_text or ""),
        task_contract=str(round_ctx.task_contract or ""),
        acceptance_constraints=extract_acceptance_constraints(round_ctx.task_contract),
    )


def run_gate(
    scorer: SemanticScorer | None,
    evidence: GateEvidence,
    threshold: float = DEFAULT_GATE_THRESHOLD,
) -> GateDecision:
    """Score the battery; "fail" only on at least one confident fail-side win.

    Each criterion is one narrow scorer question. A criterion votes fail only
    when its fail-side option both wins the argmax and reaches ``threshold``;
    scorer errors, timeouts, and malformed answers abstain. The verdict is
    therefore "fail" only with hard local evidence, and any scorer failure
    anywhere degrades to "pass" (the slow auditor runs as today).
    """
    battery: list[dict[str, Any]] = []
    findings: list[dict[str, Any]] = []
    for criterion in _criteria(evidence):
        entry: dict[str, Any] = {
            "id": criterion["id"],
            "question": criterion["question"],
            "options": [option["id"] for option in criterion["options"]],
            "fail_option": criterion["fail_option"],
            "skipped": criterion.get("skipped"),
            "probabilities": None,
            "winner": None,
            "votes_fail": False,
        }
        if entry["skipped"] is None:
            probabilities = _score(
                scorer, criterion["state"], criterion["question"], criterion["options"]
            )
            if probabilities is None:
                entry["skipped"] = "scorer_unavailable"
            else:
                option_ids = entry["options"]
                entry["probabilities"] = {
                    option_id: probability
                    for option_id, probability in zip(option_ids, probabilities)
                }
                # Ties go to the earliest option, mirroring salvage semantics.
                top = max(range(len(probabilities)), key=probabilities.__getitem__)
                entry["winner"] = option_ids[top]
                entry["votes_fail"] = (
                    option_ids[top] == criterion["fail_option"]
                    and probabilities[top] >= threshold
                )
                if entry["votes_fail"]:
                    finding: dict[str, Any] = {
                        "criterion": criterion["id"],
                        "question": criterion["question"],
                        "fail_option": criterion["fail_option"],
                        "probability": probabilities[top],
                    }
                    if criterion["id"].startswith("constraint_"):
                        finding["constraint"] = criterion["constraint"]
                    findings.append(finding)
        battery.append(entry)
    return GateDecision(
        verdict="fail" if findings else "pass",
        findings=findings,
        battery=battery,
    )


def gate_report_text(decision: GateDecision, evidence: GateEvidence) -> str:
    """The synthetic audit report a gate fail leaves in the round record.

    The text starts with the gate line and then carries a valid three-line
    control header (``incomplete`` / ``clean`` / ``unknown``) within the
    parsers' leading window, so downstream consumers read the same verdicts
    the manager recorded and completion can never be inferred from it.
    """
    lines = [
        "auditor-fast gate: confident local fail on live evidence; "
        "the slow auditor episode was skipped for this round.",
        "",
        "Status: incomplete",
        "Integrity: clean",
        "Contract audit: unknown",
        "",
        "Findings:",
    ]
    for finding in decision.findings:
        lines.append(
            f"- [{finding['criterion']}] {finding['fail_option']} "
            f"(p={finding['probability']:.3f}): {finding['question']}"
        )
    status_lines = [
        line for line in evidence.vcs_status.splitlines() if line.strip()
    ][:_MAX_VCS_REPORT_LINES]
    if status_lines:
        lines.append("")
        lines.append("Workspace VCS status lines:")
        lines.extend(status_lines)
    if evidence.notes:
        lines.append("")
        lines.append("Evidence notes: " + "; ".join(evidence.notes))
    return "\n".join(lines)


def cross_check(
    scorer: SemanticScorer | None,
    evidence: GateEvidence,
    parsed_report: AuditReport,
    threshold: float = DEFAULT_CROSS_CHECK_THRESHOLD,
) -> CrossCheck | None:
    """Score the auditor's three control verdicts against live evidence.

    One question per control, with the auditor's own legal values as the
    options, over one evidence state built from the same ``GateEvidence`` the
    pre-gate gathers (the evidence is consumed as given; nothing is re-gathered
    here). A control is flagged when the scorer's argmax differs from the
    parsed report's verdict AND the scorer's probability for its own answer
    reaches ``threshold`` (inclusive). Any scorer failure anywhere returns
    None: an advisory second opinion that cannot be computed leaves no record
    at all, and the result is never wired back into the auditor's verdicts,
    routing, or completion acceptance.
    """
    if scorer is None:
        return None
    verdicts = {
        "status": parsed_report.status,
        "integrity": parsed_report.integrity_status,
        "contract": parsed_report.contract_audit_status,
    }
    state = _cross_check_state(evidence)
    controls: list[dict[str, Any]] = []
    for control, question, options in _CROSS_CHECK_CONTROLS:
        verdict = verdicts[control]
        option_ids = [option["id"] for option in options]
        if verdict not in option_ids:
            # A verdict outside the legal values cannot be compared; leave no record.
            return None
        probabilities = _score(scorer, state, question, options)
        if probabilities is None:
            return None
        by_id = {
            option_id: probability
            for option_id, probability in zip(option_ids, probabilities)
        }
        # Ties go to the earliest option, mirroring salvage semantics.
        top = max(range(len(probabilities)), key=probabilities.__getitem__)
        answer = option_ids[top]
        agrees = answer == verdict
        controls.append(
            {
                "control": control,
                "question": question,
                "options": option_ids,
                "auditor_verdict": verdict,
                "scorer_answer": answer,
                "probabilities": by_id,
                "probability": probabilities[top],
                "auditor_probability": by_id[verdict],
                "agrees": agrees,
                "flagged": (not agrees) and probabilities[top] >= threshold,
            }
        )
    return CrossCheck(threshold=threshold, controls=controls)


def cross_check_from_defaults(
    defaults: dict[str, Any],
) -> tuple[SemanticScorer | None, float]:
    """Resolve ``(scorer, threshold)`` when ``[run.semif].cross_check`` is on.

    Like the pre-gate, the cross-check is an optimization-adjacent advisory
    extra: it is silently off unless the switch is set AND the salvage scorer
    is configured, so an absent/disabled flag or an unusable scorer yields
    ``(None, default)`` and the caller records nothing.
    """
    if not defaults.get("semif_cross_check"):
        return None, DEFAULT_CROSS_CHECK_THRESHOLD
    scorer = scorer_from_config(defaults)
    if scorer is None:
        return None, DEFAULT_CROSS_CHECK_THRESHOLD
    threshold = defaults.get(
        "semif_cross_check_threshold", DEFAULT_CROSS_CHECK_THRESHOLD
    )
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        threshold = DEFAULT_CROSS_CHECK_THRESHOLD
    return scorer, float(threshold)


def resolve_cross_check() -> tuple[SemanticScorer | None, float]:
    """Resolve the cross-check once per run from the project config; failure = off."""
    try:
        return cross_check_from_defaults(load_run_defaults())
    except Exception:
        # A missing or broken config must disable the cross-check, never crash a run.
        return None, DEFAULT_CROSS_CHECK_THRESHOLD


def extract_acceptance_constraints(
    contract: str, *, limit: int = _MAX_CONSTRAINTS
) -> list[str]:
    """Split the task contract's acceptance-constraints section into items.

    The contract is manager-maintained free text (see
    ``extract_role_task_contract``); its acceptance-constraints section lists
    items as bullets, numbered markers, or one paragraph with ``(N)`` markers.
    The split is best-effort: an absent section yields no constraints, and the
    per-constraint criteria simply do not exist while the two VCS criteria
    still run.
    """
    text = str(contract or "")
    lines = text.splitlines()
    header_index: int | None = None
    inline_rest = ""
    for index, line in enumerate(lines):
        match = _ACCEPTANCE_HEADER_RE.match(line.strip())
        if match:
            header_index = index
            # A bold header leaves its closing "**" after the colon.
            inline_rest = match.group("rest").strip().lstrip("*").strip()
            break
    if header_index is None:
        return []
    body_parts = [inline_rest] if inline_rest else []
    for line in lines[header_index + 1 :]:
        stripped = line.strip()
        if not stripped:
            continue
        if _SECTION_BOUNDARY_RE.match(stripped):
            break
        body_parts.append(stripped)
    return _split_constraint_items("\n".join(part for part in body_parts if part))[:limit]


# The section header inside a task contract: "Acceptance constraints (...):"
# in any of the contract's spellings, possibly bolded, possibly with inline
# items after the colon.
_ACCEPTANCE_HEADER_RE = re.compile(
    r"(?i)^(?:\*\*)?\s*(?:验收约束|验收标准|acceptance\s+constraints?|acceptance\s+criteria)"
    r"(?:[^:：]{0,80})?[:：]\s*(?P<rest>.*)$"
)
# A following labeled section ("**Acceptable evidence:**", "Boundary:", ...)
# ends the acceptance-constraints list. The common contract section names are
# recognized with or without bold markup and whether or not content follows
# the colon; list items (bullets, "(N)", "N.") never match.
_SECTION_BOUNDARY_RE = re.compile(
    r"(?i)^(?:\*\*)?\s*(?:可接受证据|不可接受的捷径|不可接受捷径|解释校准|已验证环境事实"
    r"|未验证假设|权威输入|状态生产过程|状态载体|持久化边界|提交边界|候选选择|污染边界"
    r"|边界|依赖判断|下一步"
    r"|acceptable\s+evidence|unacceptable\s+shortcuts?|interpretation\s+calibration"
    r"|verified\s+environment\s+facts|unverified\s+hypotheses|authoritative\s+inputs?"
    r"|state[-\s]*production\s+process|state\s+carrier|persistence\s+boundary"
    r"|commit/persistence\s+boundary|candidate[-\s]*selection|contamination\s+boundary"
    r"|boundar(?:y|ies)|dependency\s+assessment|next)"
    r"(?:[^:：]{0,60})?[:：]"
)
_ITEM_MARKER_RE = re.compile(r"(?:^|\s)[(（]\d{1,3}[)）、.]\s*")
_BULLET_START_RE = re.compile(r"^\s*(?:[-*•·]|\d{1,3}[.)、])\s+")


def _split_constraint_items(body: str) -> list[str]:
    """Split the acceptance-constraints body into individual constraints.

    Bullet/numbered lines each start an item; other lines continue the open
    item; inline ``(N)`` markers inside one chunk split it further. A body
    without any marker is one paragraph-sized constraint.
    """
    chunks: list[list[str]] = []
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if _BULLET_START_RE.match(stripped):
            chunks.append([re.sub(r"^\s*(?:[-*•·]|\d{1,3}[.)、])\s+", "", stripped)])
        elif chunks:
            chunks[-1].append(stripped)
        else:
            chunks.append([stripped])
    items: list[str] = []
    for chunk in chunks:
        text = " ".join(part for part in chunk if part).strip()
        if not text:
            continue
        for part in _ITEM_MARKER_RE.split(text):
            part = part.strip().strip("*").strip()
            # Markup fragments ("**") are not constraints; keep real text only.
            if re.search(r"[\w\u4e00-\u9fff]", part):
                items.append(part)
    return [_clip(item) for item in items if item.strip()]


def _clip(text: str, limit: int = _MAX_CONSTRAINT_CHARS) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _criteria(evidence: GateEvidence) -> list[dict[str, Any]]:
    """Build the battery: one entry per acceptance constraint plus the two
    VCS criteria. VCS-dependent criteria are emitted as skipped (never voted)
    when the workspace status is unavailable."""
    executor_output = _clip(evidence.executor_output, _MAX_EXECUTOR_OUTPUT_CHARS)
    plan_text = _clip(evidence.plan_text, _MAX_PLAN_CHARS)
    vcs_status = _clip(evidence.vcs_status, _MAX_VCS_STATUS_CHARS)
    vcs_block = (
        vcs_status if evidence.vcs_status_available
        else "(workspace VCS status unavailable)"
    )

    criteria: list[dict[str, Any]] = []
    for index, constraint in enumerate(evidence.acceptance_constraints):
        criteria.append(
            {
                "id": f"constraint_{index + 1}",
                "question": f"{_CONSTRAINT_QUESTION} Constraint: \"{constraint}\"",
                "options": _CONSTRAINT_OPTIONS,
                "fail_option": "no",
                "constraint": constraint,
                "state": (
                    "Task contract acceptance constraint under audit:\n"
                    f"{constraint}\n\n"
                    "Manager plan for this round:\n"
                    f"{plan_text}\n\n"
                    "Executor visible output:\n"
                    f"{executor_output}\n\n"
                    "Workspace VCS status:\n"
                    f"{vcs_block}"
                ),
            }
        )
    criteria.append(
        {
            "id": "vcs_claims_match",
            "question": _VCS_MATCH_QUESTION,
            "options": _VCS_MATCH_OPTIONS,
            "fail_option": "mismatch",
            "skipped": None if evidence.vcs_status_available else "vcs_status_unavailable",
            "state": (
                "Workspace VCS status:\n"
                f"{vcs_block}\n\n"
                "Executor visible output (its claims about changed files):\n"
                f"{executor_output}"
            ),
        }
    )
    criteria.append(
        {
            "id": "workspace_changes_absent",
            "question": _WORKSPACE_EMPTY_QUESTION,
            "options": _WORKSPACE_EMPTY_OPTIONS,
            "fail_option": "yes",
            "skipped": None if evidence.vcs_status_available else "vcs_status_unavailable",
            "state": (
                "Manager plan for this round:\n"
                f"{plan_text}\n\n"
                "Executor visible output:\n"
                f"{executor_output}\n\n"
                "Workspace VCS status:\n"
                f"{vcs_block}"
            ),
        }
    )
    return criteria


def _score(
    scorer: SemanticScorer | None,
    state: str,
    question: str,
    options: list[dict[str, str]],
) -> list[float] | None:
    if scorer is None:
        return None
    try:
        probabilities = scorer.score(state, question, options)
    except Exception:
        # The gate must never leak a scorer failure as anything but a pass.
        return None
    if not _valid_probabilities(probabilities, len(options)):
        return None
    return [float(value) for value in probabilities]


def _cross_check_state(evidence: GateEvidence) -> str:
    """One bounded evidence text shared by the three verdict questions."""
    executor_output = _clip(evidence.executor_output, _MAX_EXECUTOR_OUTPUT_CHARS)
    plan_text = _clip(evidence.plan_text, _MAX_PLAN_CHARS)
    vcs_status = _clip(evidence.vcs_status, _MAX_VCS_STATUS_CHARS)
    vcs_block = (
        vcs_status if evidence.vcs_status_available
        else "(workspace VCS status unavailable)"
    )
    constraints = "\n".join(
        f"- {constraint}" for constraint in evidence.acceptance_constraints
    )
    return (
        "Manager plan for this round:\n"
        f"{plan_text}\n\n"
        "Executor visible output:\n"
        f"{executor_output}\n\n"
        "Workspace VCS status:\n"
        f"{vcs_block}\n\n"
        "Task contract acceptance constraints:\n"
        f"{constraints or '(none extracted)'}"
    )


def _workspace_vcs_status(workspace: str, notes: list[str]) -> tuple[str, str]:
    """Best-effort workspace status: git first, svn fallback, never raises."""
    if not str(workspace or "").strip():
        notes.append("workspace path unavailable for VCS status")
        return "", ""
    for tool, argv in (
        ("git", ("git", "-C", workspace, "status", "--porcelain=v1", "-b")),
        ("svn", ("svn", "status")),
    ):
        try:
            completed = subprocess.run(
                argv,
                cwd=workspace,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=VCS_STATUS_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired:
            notes.append(f"{tool} status timed out after {VCS_STATUS_TIMEOUT_SECONDS}s")
        except OSError as exc:
            notes.append(f"{tool} status unavailable: {exc}")
        else:
            if completed.returncode == 0:
                stderr = completed.stderr or ""
                # `svn status` exits 0 with a W155007 warning when the
                # directory is not a working copy at all; an empty status
                # from that path is not evidence of a clean workspace (it
                # made the gate false-fail read-only rounds in plain
                # folders), so keep looking instead of returning it.
                if tool == "svn" and (
                    "W155007" in stderr or "not a working copy" in stderr.lower()
                ):
                    notes.append("svn status: directory is not a working copy")
                    continue
                return completed.stdout or "", tool
            notes.append(f"{tool} status exited with code {completed.returncode}")
    return "", ""


def _line_excluded(line: str, workspace: str, excludes: tuple[str, ...]) -> bool:
    """True when a status line's path falls under a guard-excluded path."""
    path = _status_line_path(line)
    if path is None:
        return False
    try:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = Path(workspace) / candidate
        resolved = os.path.normcase(str(candidate.resolve()))
    except (OSError, ValueError, RuntimeError):
        return False
    for item in excludes:
        try:
            excluded = os.path.normcase(str(Path(item).resolve()))
        except (OSError, ValueError, RuntimeError):
            continue
        if resolved == excluded or resolved.startswith(excluded + os.sep):
            return True
    return False


def _status_line_path(line: str) -> str | None:
    """Extract the path a porcelain/svn status line reports, if any."""
    text = str(line or "").rstrip("\n")
    if not text.strip():
        return None
    if text.startswith("##"):
        # The porcelain branch header is not a path.
        return None
    if text.startswith("??") and len(text) > 3:
        path = text[3:]
    elif len(text) > 3 and text[2] == " ":
        # Porcelain v1 (two status columns, one space, then the path); the
        # `svn status` shape ("M       path") strips to the same path here.
        path = text[3:]
    else:
        return None
    path = path.strip().strip('"')
    if " -> " in path:
        # A rename reports "old -> new"; the current location is the vote.
        path = path.rsplit(" -> ", 1)[1]
    return path or None
