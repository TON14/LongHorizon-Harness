from __future__ import annotations

import re
from typing import Any

from .agent_logs import assistant_texts as decode_agent_assistant_texts
from .agent_logs import visible_output as decode_agent_visible_output
from .config import load_run_defaults
from .runtime_signals import hard_signal_labels
from .semantic_salvage import (
    DEFAULT_THRESHOLD,
    SALVAGE_QUESTION,
    SalvageResult,
    SemanticScorer,
    salvage_control_value,
    scorer_from_config,
)
from .types import DEFAULT_WORKSPACE_PATH, EpisodeResult, AuditReport, PromptLanguage

COMPACT_REPORT_CHARS = 2_500
STATE_SUMMARY_CHARS = 1_000
ACTION_GUIDANCE_CHARS = 1_200
# Metadata keys holding assistant-visible text, most specific first. Anything
# rewriting a report must go through these, not `actions_log`, which is ignored
# whenever one of them is present.
VISIBLE_OUTPUT_KEYS = (
    "executor_agent_visible_output",
    "visible_executor_output",
    "assistant_visible_output",
    "output_text",
)
_STATUS_CONTROL_LINE_RE = re.compile(
    r"^\s*(?:\*\*)?\s*(?:状态|status|статус)\s*[:：]\s*"
    r"(complete|incomplete|blocked|完成|未完成|阻塞"
    r"|завершено|выполнено|незавершено|незавершён|заблокирован)"
    r"\s*(?:\*\*)?\s*$",
    re.I,
)
_INTEGRITY_CONTROL_LINE_RE = re.compile(
    r"^\s*(?:\*\*)?\s*(?:完整性|integrity|целостность)\s*[:：]\s*"
    r"(clean|suspect|violation|чисто|чистая|без\s+нарушений|нет\s+нарушений|нарушений\s+нет|подозрительно|нарушение|с\s+нарушениями)"
    r"\s*(?:\*\*)?\s*$",
    re.I,
)
_CONTRACT_AUDIT_CONTROL_LINE_RE = re.compile(
    r"^\s*(?:\*\*)?\s*(?:契约审计|contract(?:[_\s-]*audit)?|аудит\s*контракта|контракт[-_\s]*аудит)\s*[:：]\s*"
    r"(aligned|unknown|needs[_\s-]*revision|invalid|对齐|未知|需修订|需要修订|无效"
    r"|согласован|выровнен|выровнено|соответствует|неизвестно|требуется[_\s]*доработка|требует[_\s]*доработки|невалиден)"
    r"\s*(?:\*\*)?\s*$",
    re.I,
)
# Label-only prefixes of the three control regexes above. Semantic salvage
# uses them to locate the line whose *value* the regex missed: the label is
# there, the wording is just not one the value alternation knows yet.
_STATUS_LABEL_LINE_RE = re.compile(
    r"^\s*(?:\*\*)?\s*(?:状态|status|статус)\s*[:：]",
    re.I,
)
_INTEGRITY_LABEL_LINE_RE = re.compile(
    r"^\s*(?:\*\*)?\s*(?:完整性|integrity|целостность)\s*[:：]",
    re.I,
)
_CONTRACT_AUDIT_LABEL_LINE_RE = re.compile(
    r"^\s*(?:\*\*)?\s*(?:契约审计|contract(?:[_\s-]*audit)?|аудит\s*контракта|контракт[-_\s]*аудит)\s*[:：]",
    re.I,
)
# Canonical salvage outcomes per control line, in the parsers' own value
# order, with scorer-facing descriptions in the control prompts' bilingual
# spirit; the canonical spellings anchor the decision.
_STATUS_LEGAL_VALUES = ["complete", "incomplete", "blocked"]
_STATUS_VALUE_DESCRIPTIONS = {
    "complete": "Task complete; every deliverable is done (завершено, 完成).",
    "incomplete": "Task incomplete; deliverables remain (незавершено, 未完成).",
    "blocked": "Task blocked; a dependency failed (заблокирован, 阻塞).",
}
_INTEGRITY_LEGAL_VALUES = ["clean", "suspect", "violation"]
_INTEGRITY_VALUE_DESCRIPTIONS = {
    "clean": "Integrity clean; no violations found (без нарушений).",
    "suspect": "Integrity suspect; evidence unverified (подозрительно).",
    "violation": "Integrity violation confirmed (с нарушениями).",
}
_CONTRACT_AUDIT_LEGAL_VALUES = ["aligned", "unknown", "needs_revision", "invalid"]
_CONTRACT_AUDIT_VALUE_DESCRIPTIONS = {
    "aligned": "Execution aligns with the task contract (согласован, 对齐).",
    "unknown": "Contract compliance could not be determined (неизвестно, 未知).",
    "needs_revision": "The task contract needs revision (требуется доработка, 需修订).",
    "invalid": "The contract audit itself is invalid (невалиден, 无效).",
}
_BLOCKING_ACCEPTANCE_SECTION_RE = re.compile(
    r"^\s*(?:[-*]\s*)?(?:阻断约束|blocking\s+(?:acceptance\s+)?(?:constraints?|claims?))\s*[:：]\s*(?P<rest>.*)$",
    re.I,
)
_NO_BLOCKING_ACCEPTANCE_RE = re.compile(
    r"(?ix)^\s*(?:[-*+]\s*|\d+[.)]\s*)?(?:无|没有|暂无|none|nothing|n/?a|not\s+applicable)(?:\s*[。.,，;；:].*)?\s*$"
)
# Salvage decision for the acceptance guard: a "Blocking constraints" line the
# regex above missed is re-judged as a plain yes/no question rather than a
# multi-value control, mirroring the control-line cascade (control name
# "acceptance_none" in the salvage ledger).
_ACCEPTANCE_NONE_QUESTION = (
    "Does this line state that there are no blocking acceptance constraints?"
)
_ACCEPTANCE_NONE_LEGAL_VALUES = ["yes", "no"]
_ACCEPTANCE_NONE_VALUE_DESCRIPTIONS = {
    "yes": "The line states there are no blocking acceptance constraints (Блокирующих ограничений нет, 无阻断约束).",
    "no": "The line states or lists actual blocking acceptance constraints (есть блокирующие ограничения, 存在阻断约束).",
}
# Every heading the auditor contract prompt mandates must terminate the blocking
# section; otherwise a reordered report leaks the next section into it and the
# acceptance guard downgrades an otherwise clean audit.
_ACCEPTANCE_SECTION_BOUNDARY_RE = re.compile(
    r"(?im)^\s*(?:[-*]\s*)?(?:契约结论|可能评分风险|过窄或错误解释|建议契约修订|审计事实|证据|缺口|下一步|给任务管理器的状态更新"
    r"|验收约束反查|原题约束清单|契约覆盖检查|逐项反查"
    r"|contract\s+conclusion|possible\s+scoring\s+risks|over-narrow|recommended\s+contract\s+revision"
    r"|audit\s+facts|evidence|gaps?|next\s+step|state\s+update\s+for\s+manager|status|integrity|contract\s+audit"
    r"|acceptance[-\s]*constraint\s+backcheck|original\s+constraint\s+inventory|contract\s+coverage\s+check"
    r"|per[-\s]*constraint\s+backcheck)\s*[:：]"
)
_DELETE_DECLARATION_RE = re.compile(
    r"^\s*(?:[-*]\s*)?"
    r"(?:删除伪造产物|删除不可信产物|删除作弊产物|"
    r"delete fabricated artifact|delete untrusted artifact|"
    r"deleted fabricated artifact|deleted untrusted artifact)"
    r"\s*[:：]\s*(?P<rest>.+)$",
    re.I,
)

# Harness-synthesized report fragments. The auditor's own report stays verbatim;
# only these harness-authored lines follow the run's prompt language, and the
# control-header keywords match what the parser above accepts for each language.
_CONTROL_HEADER: dict[PromptLanguage, tuple[str, str, str]] = {
    "en": ("Status", "Integrity", "Contract audit"),
    "zh": ("状态", "完整性", "契约审计"),
}
_SYNTHETIC_TEXT: dict[str, dict[PromptLanguage, str]] = {
    "invalid_header_facts": {
        "en": "Audit facts: the auditor report lacks a valid three-line control header; the harness will not guess completion, integrity, or contract audit from the body.",
        "zh": "审计事实: auditor 报告缺少有效的前三行控制头，harness 不会根据正文猜测完成状态、完整性或契约审计。",
    },
    "invalid_header_gap": {
        "en": "Gap: the auditor must state status, integrity, and contract audit on the first three lines.",
        "zh": "缺口: auditor 必须以前三行明确写出状态、完整性和契约审计。",
    },
    "invalid_header_excerpt": {
        "en": "Original auditor output excerpt:",
        "zh": "原始 auditor 输出摘录:",
    },
    "deleted_ledger": {
        "en": "Deletion ledger: the auditor deleted executor artifacts it confirmed to be fabricated or untrusted; those paths can no longer count as valid deliverables.",
        "zh": "删除记录: auditor 在审计中删除了确认伪造或不可信的 executor 产物；这些路径不能再作为有效 deliverable。",
    },
    "deleted_paths": {
        "en": "Deleted paths: ",
        "zh": "已删除路径: ",
    },
    "read_only_violation": {
        "en": "Auditor read-only violation: the auditor window changed task workspace files while checking.",
        "zh": "auditor 只读违规: auditor 在审计过程中修改了任务 workspace 文件。",
    },
    "restored": {
        "en": "The harness restored the workspace snapshot taken before audit.",
        "zh": "harness 已恢复审计前的 workspace 快照。",
    },
    "not_restored": {
        "en": "The harness detected a workspace mutation but did not confirm restoration; treat the audit as invalid.",
        "zh": "harness 检测到 workspace 变化但未确认恢复；本次审计视为无效。",
    },
    "violation_claims_invalid": {
        "en": "Any claim based on auditor-created or auditor-modified files is invalid.",
        "zh": "任何基于 auditor 创建或修改的文件得出的结论都无效。",
    },
    "changed_paths": {
        "en": "Changed paths observed: ",
        "zh": "观察到的变更路径: ",
    },
    "executor_must_repair": {
        "en": "The next executor must check and finish the missing deliverables itself; the auditor may only report state, never repair it.",
        "zh": "下一轮 executor 必须自行检查并完成缺失 deliverable；auditor 只能报告状态，不能修复。",
    },
    "diagnostics_only": {
        "en": "Original auditor report, for diagnostics only:",
        "zh": "原始 auditor 报告，仅供诊断:",
    },
    "activity_archive": {
        "en": (
            f"Environment activity note: active processes or unstable file state were already observed under `{DEFAULT_WORKSPACE_PATH}` before the auditor ran, "
            "and the workspace changed again during the audit window. The harness did not roll these changes back, to avoid misreading real executor or "
            "background-service progress as auditor writes. This note does not by itself invalidate the round's audit report."
        ),
        "zh": (
            f"环境活动归档: auditor 运行前已观察到 `{DEFAULT_WORKSPACE_PATH}` 存在活动进程或不稳定文件状态；"
            "审计窗口内 workspace 又发生变化。harness 没有回滚这些变化，避免把 executor 或后台服务的真实进展"
            "误判为 auditor 写入。该归档不自动废弃本轮审计报告。"
        ),
    },
    "completion_guard": {
        "en": "Harness completion guard: the auditor listed nonempty blocking constraints, so `complete` and an aligned contract decision are not accepted.",
        "zh": "harness 完成守卫: auditor 列出了非空阻断约束，因此不接受 `complete` 和 aligned 契约结论。",
    },
    "runtime_failed": {
        "en": "Auditor runtime failed before producing a trustworthy report.",
        "zh": "auditor 运行时在产出可信报告前失败。",
    },
    "runtime_mutation_restored": {
        "en": "Auditor also changed task workspace files; the harness restored the pre-audit snapshot.",
        "zh": "auditor 还修改了任务 workspace 文件；harness 已恢复审计前快照。",
    },
    "runtime_mutation_not_restored": {
        "en": "Auditor also changed task workspace files, and restoration was not confirmed.",
        "zh": "auditor 还修改了任务 workspace 文件，且未确认恢复。",
    },
    "no_claim_from_logs": {
        "en": "No completion claim was accepted from raw runtime logs or echoed prompt text.",
        "zh": "harness 不接受来自原始运行日志或回显 prompt 文本的完成声明。",
    },
    "runtime_failed_guidance": {
        "en": "Auditor runtime failed; rerun the audit after fixing the runtime issue.",
        "zh": "auditor 运行时失败；修复运行时问题后重新审计。",
    },
}


def _language(language: str) -> PromptLanguage:
    return "zh" if str(language or "").strip().lower() == "zh" else "en"


def _text(key: str, language: str) -> str:
    return _SYNTHETIC_TEXT[key][_language(language)]


def _control_header(
    status: str,
    integrity: str,
    contract_audit: str,
    *,
    language: str,
) -> str:
    status_label, integrity_label, contract_label = _CONTROL_HEADER[_language(language)]
    return (
        f"{status_label}: {status}\n"
        f"{integrity_label}: {integrity}\n"
        f"{contract_label}: {contract_audit}"
    )


def audit_report_from_episode_result(
    result: EpisodeResult,
    round_index: int,
    *,
    language: str = "en",
    scorer: SemanticScorer | None = None,
    threshold: float | None = None,
) -> AuditReport:
    """Convert a auditor role episode into the structured report used by the runner.

    The role manager stores auditor output as natural language, but the
    final report and stop checks need a compact status, integrity flag, and
    artifact-deletion ledger.
    """
    hard_runtime_signals = hard_signal_labels(result.metadata.get("runtime_signals"))
    if result.status != "done" or hard_runtime_signals:
        return AuditReport(
            round_id=f"round_{round_index}",
            status="blocked",
            report_text=_runtime_failure_report(
                result, hard_runtime_signals, language=language
            ),
            action_guidance=_text("runtime_failed_guidance", language),
        )

    salvage = _ControlSalvage(scorer=scorer, threshold=threshold)
    report_text = compact_auditor_report_text(
        extract_auditor_report_text(_episode_visible_output(result))
    )
    if not _has_valid_control_header(report_text, salvage):
        report_text = _invalid_control_header_report(report_text, language=language)
    status = infer_report_status(report_text, salvage)
    state_summary = extract_state_summary(report_text)
    action_guidance = extract_action_guidance(report_text)
    integrity_status, integrity_findings = infer_integrity_findings(report_text, salvage)
    contract_audit_status = infer_contract_audit_status(report_text, salvage)
    artifact_actions = extract_deleted_artifact_actions(report_text) if integrity_status == "violation" else []
    if result.metadata.get("verifier_workspace_mutation_detected"):
        paths = _mutation_paths(result.metadata.get("verifier_workspace_mutations"))
        restore_on_mutation = bool(result.metadata.get("verifier_workspace_restore_on_mutation", True))
        restored = bool(result.metadata.get("verifier_workspace_restored"))
        allowed_delete_paths = _allowed_auditor_delete_paths(
            result.metadata,
            integrity_status=integrity_status,
            declared_actions=artifact_actions,
        )
        if allowed_delete_paths:
            artifact_actions = _reconcile_deletion_actions(
                declared_actions=artifact_actions,
                confirmed_deleted_paths=allowed_delete_paths,
            )
            suffix = "\n\n" + _text("deleted_ledger", language)
            suffix += "\n" + _text("deleted_paths", language) + ", ".join(allowed_delete_paths[:20])
            report_text = compact_auditor_report_text(report_text + suffix)
            state_summary = extract_state_summary(report_text)
            action_guidance = extract_action_guidance(report_text)
        elif restore_on_mutation:
            restore_text = _text("restored" if restored else "not_restored", language)
            prefix = (
                _control_header("blocked", "violation", "unknown", language=language)
                + "\n\n"
                + _text("read_only_violation", language)
                + f" {restore_text} "
                + _text("violation_claims_invalid", language)
                + "\n"
            )
            if paths:
                prefix += _text("changed_paths", language) + ", ".join(paths[:20]) + "\n"
            prefix += (
                _text("executor_must_repair", language)
                + "\n\n"
                + _text("diagnostics_only", language)
                + "\n"
            )
            report_text = compact_auditor_report_text(prefix + report_text)
            status = "blocked"
            state_summary = extract_state_summary(report_text)
            action_guidance = extract_action_guidance(report_text)
            integrity_status = "violation"
            contract_audit_status = "unknown"
            integrity_findings.append(
                {
                    "type": "verifier_workspace_write",
                    "severity": "violation",
                    "evidence": "Auditor changed task workspace files during read-only audit.",
                    "paths": paths,
                    "restored": restored,
                }
            )
        else:
            suffix = "\n\n" + _text("activity_archive", language)
            if paths:
                suffix += "\n" + _text("changed_paths", language) + ", ".join(paths[:20])
            report_text = compact_auditor_report_text(report_text + suffix)
            state_summary = extract_state_summary(report_text)
            action_guidance = extract_action_guidance(report_text)
    artifact_actions = _mark_unconfirmed_deletion_declarations(artifact_actions)
    report_text, status, contract_audit_status = _apply_acceptance_constraint_guard(
        report_text, status, contract_audit_status, language=language, salvage=salvage
    )
    if (integrity_status == "violation" or contract_audit_status != "aligned") and status == "complete":
        status = "incomplete"
    return AuditReport(
        round_id=f"round_{round_index}",
        status=status,
        report_text=report_text,
        state_summary=state_summary,
        action_guidance=action_guidance,
        integrity_status=integrity_status,
        contract_audit_status=contract_audit_status,
        integrity_findings=integrity_findings,
        artifact_actions=artifact_actions,
        control_salvage=salvage.records,
    )


def parse_audit_report(
    raw: str,
    round_index: int,
    *,
    language: str = "en",
    scorer: SemanticScorer | None = None,
    threshold: float | None = None,
) -> AuditReport:
    salvage = _ControlSalvage(scorer=scorer, threshold=threshold)
    report_text = compact_auditor_report_text(extract_auditor_report_text(raw))
    if not _has_valid_control_header(report_text, salvage):
        report_text = _invalid_control_header_report(report_text, language=language)
    status = infer_report_status(report_text, salvage)
    integrity_status, integrity_findings = infer_integrity_findings(report_text, salvage)
    contract_audit_status = infer_contract_audit_status(report_text, salvage)
    report_text, status, contract_audit_status = _apply_acceptance_constraint_guard(
        report_text, status, contract_audit_status, language=language, salvage=salvage
    )
    if (integrity_status == "violation" or contract_audit_status != "aligned") and status == "complete":
        status = "incomplete"
    return AuditReport(
        round_id=f"round_{round_index}",
        status=status,
        report_text=report_text,
        state_summary=extract_state_summary(report_text),
        action_guidance=extract_action_guidance(report_text),
        integrity_status=integrity_status,
        contract_audit_status=contract_audit_status,
        integrity_findings=integrity_findings,
        artifact_actions=extract_deleted_artifact_actions(report_text) if integrity_status == "violation" else [],
        control_salvage=salvage.records,
    )


def auditor_report_text_from_episode_result(result: EpisodeResult) -> str:
    return compact_auditor_report_text(
        extract_auditor_report_text(_episode_visible_output(result))
    )


def _episode_visible_output(result: EpisodeResult) -> str:
    """Prefer adapter-provided assistant text over the diagnostic trajectory."""
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    for key in VISIBLE_OUTPUT_KEYS:
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    if metadata.get("actions_log_diagnostics_only"):
        return ""
    raw = result.actions_log or ""
    decoded = decode_agent_visible_output(raw)
    return decoded if decoded else raw


def has_valid_auditor_control_header(text: str) -> bool:
    # The manager's format-repair trigger reads this predicate: a header
    # recovered by [run.semif] salvage counts as valid here too, so a
    # recoverable wording no longer burns the repair episode. With salvage
    # not configured this stays the regex-only check it always was.
    return _has_valid_control_header(str(text or ""), _ControlSalvage())


def infer_report_status(
    text: str, salvage: _ControlSalvage | None = None
) -> str:
    return _parse_status_control_header(text, salvage) or "blocked"


def infer_contract_audit_status(
    text: str, salvage: _ControlSalvage | None = None
) -> str:
    return _parse_contract_audit_control_header(text, salvage) or "unknown"


def _apply_acceptance_constraint_guard(
    report_text: str,
    status: str,
    contract_audit_status: str,
    *,
    language: str = "en",
    salvage: _ControlSalvage | None = None,
) -> tuple[str, str, str]:
    if status != "complete" or not _has_blocking_acceptance_constraints(
        report_text, salvage
    ):
        return report_text, status, contract_audit_status
    guarded = compact_auditor_report_text(
        report_text + "\n\n" + _text("completion_guard", language)
    )
    return guarded, "incomplete", "unknown" if contract_audit_status == "aligned" else contract_audit_status


def _has_blocking_acceptance_constraints(
    text: str, salvage: _ControlSalvage | None = None
) -> bool:
    lines = str(text or "").splitlines()
    for index, line in enumerate(lines):
        match = _BLOCKING_ACCEPTANCE_SECTION_RE.match(line)
        if not match:
            continue
        rest = match.group("rest").strip()
        if rest:
            return not _is_no_blocking_acceptance(rest, salvage)
        section_lines: list[str] = []
        for following in lines[index + 1 :]:
            stripped = following.strip()
            if not stripped:
                continue
            if _ACCEPTANCE_SECTION_BOUNDARY_RE.match(stripped):
                break
            section_lines.append(stripped)
        return bool(section_lines) and not all(
            _is_no_blocking_acceptance(item, salvage) for item in section_lines
        )
    return False


def _is_no_blocking_acceptance(
    text: str, salvage: _ControlSalvage | None = None
) -> bool:
    stripped = str(text or "").strip().strip("`")
    if _NO_BLOCKING_ACCEPTANCE_RE.match(stripped):
        return True
    # The regex missed; one semantic decision may still recognize an
    # unfamiliar "none" phrasing. Salvage can only prevent a downgrade, so
    # a lost decision (no scorer, failure, below threshold, argmax "no")
    # keeps today's not-a-"none"-phrasing reading.
    result = (
        salvage.resolve(
            "acceptance_none",
            stripped,
            _ACCEPTANCE_NONE_LEGAL_VALUES,
            _ACCEPTANCE_NONE_VALUE_DESCRIPTIONS,
            question=_ACCEPTANCE_NONE_QUESTION,
        )
        if salvage is not None
        else None
    )
    return result is not None and result.value == "yes"


# Semantic salvage for control lines the regexes missed. Salvage runs only
# after a control regex missed -- a regex hit never resolves the scorer at
# all. An optional [run.semif] table supplies the scorer; with the section
# absent or disabled nothing is constructed and every miss keeps today's
# fallback verdict.
_PROJECT_SALVAGE_SETTINGS: dict[str, tuple[SemanticScorer | None, float]] = {}


def _project_salvage_settings() -> tuple[SemanticScorer | None, float]:
    """Resolve the optional [run.semif] scorer and threshold once per process."""
    if "settings" not in _PROJECT_SALVAGE_SETTINGS:
        try:
            defaults = load_run_defaults()
            threshold = defaults.get("semif_threshold", DEFAULT_THRESHOLD)
            if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
                threshold = DEFAULT_THRESHOLD
            _PROJECT_SALVAGE_SETTINGS["settings"] = (
                scorer_from_config(defaults),
                float(threshold),
            )
        except Exception:
            # A missing or broken config must degrade to no salvage, never
            # crash control-line parsing.
            _PROJECT_SALVAGE_SETTINGS["settings"] = (None, DEFAULT_THRESHOLD)
    return _PROJECT_SALVAGE_SETTINGS["settings"]


class _ControlSalvage:
    """One report's salvage context: lazy scorer, one decision per missed
    line, and the provenance ledger the parsed report records.

    The header validity check and the three verdict extractions re-parse the
    same text, so decisions are memoized per (control, line): the scorer runs
    at most once per missed control line per report.
    """

    def __init__(
        self,
        scorer: SemanticScorer | None = None,
        threshold: float | None = None,
    ) -> None:
        self._scorer = scorer
        self._threshold = threshold
        self._resolved = False
        self._memo: dict[tuple[str, str], SalvageResult | None] = {}
        self.records: list[dict[str, Any]] = []

    def resolve(
        self,
        control: str,
        line: str,
        legal_values: list[str],
        descriptions: dict[str, str],
        *,
        question: str = SALVAGE_QUESTION,
    ) -> SalvageResult | None:
        if not self._resolved:
            if self._scorer is None:
                configured_scorer, configured_threshold = _project_salvage_settings()
                self._scorer = configured_scorer
                if self._threshold is None:
                    self._threshold = configured_threshold
            if self._threshold is None:
                # An explicitly passed scorer keeps the default threshold
                # instead of reading the project config.
                self._threshold = DEFAULT_THRESHOLD
            self._resolved = True
        key = (control, line)
        if key not in self._memo:
            result = salvage_control_value(
                line,
                legal_values,
                descriptions,
                self._scorer,
                self._threshold,
                question=question,
            )
            self._memo[key] = result
            if result is not None:
                self.records.append(
                    {
                        "control": control,
                        "value": result.value,
                        "probabilities": dict(
                            zip(legal_values, result.probabilities)
                        ),
                    }
                )
        return self._memo[key]


def _salvage_control_line(
    text: str,
    salvage: _ControlSalvage | None,
    control: str,
    label_line_re: re.Pattern[str],
    legal_values: list[str],
    descriptions: dict[str, str],
) -> str | None:
    """Recover a control value for a label line the value regex missed."""
    if salvage is None:
        return None
    line = next(
        (window for window in _control_lines_window(text) if label_line_re.match(window)),
        None,
    )
    if line is None:
        # No line in the window even carries the control label; there is
        # nothing to re-judge.
        return None
    result = salvage.resolve(control, line, legal_values, descriptions)
    return result.value if result is not None else None


def _parse_status_control_header(
    text: str, salvage: _ControlSalvage | None = None
) -> str | None:
    lines = _first_nonempty_lines(text, 1)
    if not lines:
        return None
    match = None
    for line in _control_lines_window(text):
        candidate = _STATUS_CONTROL_LINE_RE.match(line)
        if candidate:
            match = candidate
            break
    if not match:
        return _salvage_control_line(
            text,
            salvage,
            "status",
            _STATUS_LABEL_LINE_RE,
            _STATUS_LEGAL_VALUES,
            _STATUS_VALUE_DESCRIPTIONS,
        )
    value = match.group(1).lower()
    if value in {"complete", "完成", "завершено", "выполнено"}:
        return "complete"
    if value in {"blocked", "阻塞", "заблокирован"}:
        return "blocked"
    return "incomplete"


def _parse_integrity_control_header(
    text: str, salvage: _ControlSalvage | None = None
) -> str | None:
    lines = _first_nonempty_lines(text, 2)
    if len(lines) < 2:
        return None
    match = None
    for line in _control_lines_window(text):
        candidate = _INTEGRITY_CONTROL_LINE_RE.match(line)
        if candidate:
            match = candidate
            break
    if not match:
        return _salvage_control_line(
            text,
            salvage,
            "integrity",
            _INTEGRITY_LABEL_LINE_RE,
            _INTEGRITY_LEGAL_VALUES,
            _INTEGRITY_VALUE_DESCRIPTIONS,
        )
    value = match.group(1).lower()
    ru = {
        "чисто": "clean",
        "чистая": "clean",
        "без нарушений": "clean",
        "нет нарушений": "clean",
        "нарушений нет": "clean",
        "подозрительно": "suspect",
        "нарушение": "violation",
        "с нарушениями": "violation",
    }
    return ru.get(value, value)


def _parse_contract_audit_control_header(
    text: str, salvage: _ControlSalvage | None = None
) -> str | None:
    lines = _first_nonempty_lines(text, 3)
    if len(lines) < 3:
        return None
    match = None
    for line in _control_lines_window(text):
        candidate = _CONTRACT_AUDIT_CONTROL_LINE_RE.match(line)
        if candidate:
            match = candidate
            break
    if not match:
        return _salvage_control_line(
            text,
            salvage,
            "contract_audit",
            _CONTRACT_AUDIT_LABEL_LINE_RE,
            _CONTRACT_AUDIT_LEGAL_VALUES,
            _CONTRACT_AUDIT_VALUE_DESCRIPTIONS,
        )
    value = match.group(1).lower().replace("-", "_").replace(" ", "_")
    if value in {"aligned", "对齐", "согласован", "выровнен", "выровнено", "соответствует"}:
        return "aligned"
    if value in {"needs_revision", "需修订", "需要修订", "требуется_доработка", "требует_доработки"}:
        return "needs_revision"
    if value in {"invalid", "无效", "невалиден"}:
        return "invalid"
    return "unknown"


def _has_valid_control_header(
    text: str, salvage: _ControlSalvage | None = None
) -> bool:
    return (
        _parse_status_control_header(text, salvage) is not None
        and _parse_integrity_control_header(text, salvage) is not None
        and _parse_contract_audit_control_header(text, salvage) is not None
    )


def _invalid_control_header_report(raw: str, *, language: str = "en") -> str:
    clipped = compact_auditor_report_text(raw, max_chars=1_800)
    return (
        _control_header("blocked", "suspect", "unknown", language=language)
        + "\n"
        + _text("invalid_header_facts", language)
        + "\n"
        + _text("invalid_header_gap", language)
        + "\n\n"
        + _text("invalid_header_excerpt", language)
        + f"\n{clipped}"
    )


def extract_auditor_report_text(raw: str, *, max_chars: int = 8_000) -> str:
    assistant_texts = decode_agent_assistant_texts(raw)
    if not assistant_texts:
        assistant_texts = _assistant_texts_from_compacted_openclaw_log(raw)
    for text in reversed(assistant_texts):
        if _looks_like_report(text):
            return _clip_report_source(_trim_to_report_start(text), max_chars)
    if assistant_texts:
        return _clip_report_source(
            _trim_to_report_start("\n\n".join(text.strip() for text in assistant_texts if text.strip())),
            max_chars,
        )
    return _clip_report_source(_trim_to_report_start(raw.strip()), max_chars)


def compact_auditor_report_text(text: str, *, max_chars: int = COMPACT_REPORT_CHARS) -> str:
    stripped = text.strip()
    if len(stripped) <= max_chars:
        return stripped
    head_chars = max(1, int(max_chars * 0.7))
    tail_chars = max(1, max_chars - head_chars)
    return (
        stripped[:head_chars].rstrip()
        + f"\n\n...[auditor report truncated {len(stripped) - max_chars} chars; kept head and tail]...\n\n"
        + stripped[-tail_chars:].lstrip()
    )


def extract_state_summary(text: str, *, max_chars: int = STATE_SUMMARY_CHARS) -> str:
    lines = [line.rstrip() for line in text.splitlines()]
    for index, line in enumerate(lines):
        if not re.search(r"(状态摘要|真实状态|当前状态|state\s+summary|state\s+card)", line, re.I):
            continue
        collected = [_strip_heading(line)]
        for follow in lines[index + 1 : index + 7]:
            stripped = follow.strip()
            if not stripped:
                break
            if _is_major_heading(stripped):
                break
            collected.append(stripped)
        summary = "\n".join(item for item in collected if item).strip()
        if summary:
            return _head(summary, max_chars)

    collected = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if re.search(r"(下一步|行动建议|next\s+step|next\s+round|task\s+agent)", stripped, re.I):
            break
        collected.append(stripped)
        if len(collected) >= 6:
            break
    return _head("\n".join(collected), max_chars)


def extract_action_guidance(text: str, *, max_chars: int = ACTION_GUIDANCE_CHARS) -> str:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if not re.search(r"(下一步|行动建议|建议|next\s+step|next\s+round|task\s+agent)", line, re.I):
            continue
        collected: list[str] = []
        first = re.sub(
            r"^\s*(?:[-*]\s*)?(?:下一轮\s*task\s*agent\s*应该具体做什么|下一步|行动建议|建议|next\s+step|next\s+round|task\s+agent)\s*[:：-]?\s*",
            "",
            line,
            flags=re.I,
        ).strip()
        if first:
            collected.append(first)
        for follow in lines[index + 1 : index + 6]:
            stripped = follow.strip()
            if not stripped:
                break
            if re.search(r"^(状态|已完成|完成|待完成|还缺|缺少|证据|命令|检查|status|completed|missing|evidence|commands)\s*[:：]", stripped, re.I):
                break
            collected.append(stripped)
        guidance = "\n".join(collected).strip()
        if guidance:
            return _tail(guidance, max_chars)
    return ""


def infer_integrity_findings(
    text: str, salvage: _ControlSalvage | None = None
) -> tuple[str, list[dict[str, Any]]]:
    integrity_status = _parse_integrity_control_header(text, salvage)
    lines = _first_nonempty_lines(text, 2)
    evidence = lines[1] if len(lines) >= 2 else "missing integrity control header"
    if integrity_status == "clean":
        return "clean", []
    if integrity_status == "violation":
        return "violation", [
            {
                "type": "integrity_control_header",
                "severity": "violation",
                "evidence": evidence,
                "paths": [],
            }
        ]
    return "suspect", [
        {
            "type": "integrity_control_header",
            "severity": "suspect",
            "evidence": evidence,
            "paths": [],
        }
    ]


def extract_deleted_artifact_actions(text: str) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for line in text.splitlines():
        normalized = line.strip()
        if not normalized:
            continue
        match = _DELETE_DECLARATION_RE.match(normalized)
        if not match:
            continue
        rest = match.group("rest").strip()
        paths = extract_candidate_artifact_paths(rest)
        for path in paths:
            if any(action.get("path") == path for action in actions):
                continue
            actions.append(
                {
                    "action": "delete",
                    "status": "delete_declared_unverified",
                    "path": path,
                    "reason": _extract_delete_reason(rest),
                    "declaration": _head(normalized, 600),
                }
            )
    return actions


def extract_candidate_artifact_paths(text: str) -> list[str]:
    patterns = [
        rf"(?P<path>{re.escape(DEFAULT_WORKSPACE_PATH)}/[^\s`'\"<>:;|，。；：]+)",
        r"(?<![A-Za-z0-9_./-])(?P<path>(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+\.(?:png|jpg|jpeg|gif|webp|svg|pdf|txt|md|csv|json|html|htm|mp4|mov|wav|zip|tar|gz))",
    ]
    paths: list[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            path = match.group("path").rstrip(".,)]}，。；：")
            if path and path not in paths:
                paths.append(path)
    return paths[:20]


def _extract_delete_reason(line: str) -> str:
    match = re.search(r"(?:原因|reason)\s*[:：]\s*(.+)$", line, re.I)
    if match:
        return _head(match.group(1).strip(), 300)
    return _head(line, 300)


def _looks_like_report(text: str) -> bool:
    lowered = text.lower()
    return "状态" in text or "status" in lowered or "已完成" in text or "missing" in lowered


def _trim_to_report_start(text: str) -> str:
    match = re.search(
        r"(?im)^\s*(?:\*\*)?\s*(?:状态|status|статус)\s*[:：]\s*"
        r"(?:complete|incomplete|blocked|完成|未完成|阻塞|завершено|выполнено|незавершено|незавершён|заблокирован)",
        text,
    )
    if match:
        return text[match.start() :].strip()
    return text.strip()


def _assistant_texts_from_compacted_openclaw_log(raw: str) -> list[str]:
    texts: list[str] = []
    role: str | None = None
    current: list[str] = []

    def flush() -> None:
        nonlocal current
        if role == "assistant":
            text = "\n".join(current).strip()
            if text:
                texts.append(text)
        current = []

    for line in raw.splitlines():
        marker = re.match(r"^\[(assistant|toolResult|tool)(?::[^\]]*)?\]\s*$", line.strip())
        if marker:
            flush()
            role = marker.group(1)
            continue
        if re.match(r"^\[toolCall:[^\]]+\]", line.strip()):
            flush()
            role = None
            continue
        if role == "assistant":
            current.append(line)
    flush()
    return texts


def _tail(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _clip_report_source(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head_chars = max(1, int(max_chars * 0.7))
    tail_chars = max(1, max_chars - head_chars)
    return (
        text[:head_chars].rstrip()
        + f"\n\n...[auditor source truncated {len(text) - max_chars} chars; kept head and tail]...\n\n"
        + text[-tail_chars:].lstrip()
    )


def _head(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + f"\n...[truncated {len(text) - max_chars} chars]"


def _first_nonempty_lines(text: str, count: int) -> list[str]:
    lines: list[str] = []
    for line in str(text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lines.append(stripped)
        if len(lines) >= count:
            break
    return lines


# Auditors -- reliably the wordier ones -- sometimes open the report with a
# one-line preamble ("All checks complete. Compiling the audit report.")
# before the three control lines, which pushed Status/Integrity/Contract
# out of their fixed first/second/third positions and burned the whole
# round on an "invalid control header" verdict. The header parsers below
# therefore scan a small window of leading non-empty lines for the k-th
# control line IN ORDER, tolerating preamble noise but not body text:
# the window is short and each line may be claimed by at most one header.
def _control_lines_window(text: str, window: int = 8) -> list[str]:
    return _first_nonempty_lines(text, window)


def _strip_heading(line: str) -> str:
    return re.sub(
        r"^\s*(?:[-*]\s*)?(?:状态摘要|真实状态|当前状态|state\s+summary|state\s+card)\s*[:：-]?\s*",
        "",
        line,
        flags=re.I,
    ).strip()


def _is_major_heading(line: str) -> bool:
    return bool(
        re.search(
            r"^(状态|已完成|完成|待完成|还缺|缺少|证据|命令|检查|状态摘要|真实状态|当前状态|行动建议|下一步|status|completed|missing|evidence|commands|state|next)\s*[:：]",
            line,
            re.I,
        )
    )


def _allowed_auditor_delete_paths(
    metadata: dict[str, Any],
    *,
    integrity_status: str,
    declared_actions: list[dict[str, Any]],
) -> list[str]:
    counts = metadata.get("verifier_workspace_mutation_counts")
    mutations = metadata.get("verifier_workspace_mutations")
    if not isinstance(counts, dict) or not isinstance(mutations, dict):
        return []
    deleted_count = int(counts.get("deleted") or 0)
    non_delete_count = (
        int(counts.get("added") or 0)
        + int(counts.get("changed") or 0)
        + int(counts.get("type_changed") or 0)
    )
    if deleted_count <= 0:
        return []
    if metadata.get("verifier_workspace_restored"):
        return []
    deleted_paths = [str(path) for path in mutations.get("deleted", []) if str(path).strip()]
    declared = {
        _workspace_relpath(action.get("path"))
        for action in declared_actions
        if action.get("action") == "delete" and _workspace_relpath(action.get("path"))
    }
    if non_delete_count or integrity_status != "violation" or not declared:
        return []
    return [path for path in deleted_paths if _workspace_relpath(path) in declared]


def _reconcile_deletion_actions(
    *,
    declared_actions: list[dict[str, Any]],
    confirmed_deleted_paths: list[str],
) -> list[dict[str, Any]]:
    confirmed = {_workspace_relpath(path): path for path in confirmed_deleted_paths}
    reconciled: list[dict[str, Any]] = []
    matched: set[str] = set()
    for action in declared_actions:
        copied = dict(action)
        key = _workspace_relpath(copied.get("path"))
        if key and key in confirmed:
            copied["status"] = "deleted_by_auditor"
            copied["path"] = str(copied.get("path") or confirmed[key])
            matched.add(key)
        else:
            copied["status"] = "delete_declared_unverified"
            copied.setdefault(
                "reason",
                "auditor declared deletion, but the workspace diff did not confirm that this path was deleted",
            )
        reconciled.append(copied)

    for path in confirmed_deleted_paths:
        if _workspace_relpath(path) in matched:
            continue
        reconciled.append(
            {
                "action": "delete",
                "status": "deleted_by_auditor",
                "path": path,
                "reason": "workspace diff recorded auditor deletion during integrity audit",
                "declaration": "Auditor deleted this path during integrity audit; see report_text for rationale.",
            }
        )
    return reconciled


def _mark_unconfirmed_deletion_declarations(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    marked: list[dict[str, Any]] = []
    for action in actions:
        copied = dict(action)
        if copied.get("status") == "deleted_by_auditor":
            marked.append(copied)
            continue
        if copied.get("action") == "delete":
            copied["status"] = "delete_declared_unverified"
            copied.setdefault(
                "reason",
                "auditor declared deletion, but no matching workspace deletion was observed",
            )
        marked.append(copied)
    return marked


def _workspace_relpath(raw: object) -> str:
    path = str(raw or "").strip()
    if not path:
        return ""
    path = path.rstrip(".,)]}，。；：")
    prefix = f"{DEFAULT_WORKSPACE_PATH}/"
    if path == DEFAULT_WORKSPACE_PATH:
        return "."
    if path.startswith(prefix):
        return path[len(prefix) :]
    return path.lstrip("./")


def _mutation_paths(raw: object) -> list[str]:
    if not isinstance(raw, dict):
        return []
    paths: list[str] = []
    for key in ("added", "changed", "deleted", "type_changed"):
        value = raw.get(key)
        if isinstance(value, list):
            paths.extend(str(item) for item in value[:20])
    return paths


def _runtime_failure_report(result, runtime_signals: list[str], *, language: str = "en") -> str:  # noqa: ANN001
    lines = [
        _control_header("blocked", "suspect", "unknown", language=language),
        "",
        _text("runtime_failed", language),
    ]
    if result.status != "done":
        lines.append(f"Episode status: {result.status}.")
    if result.error:
        lines.append(f"Runtime error: {result.error}")
    if runtime_signals:
        lines.append("Runtime signals: " + ", ".join(runtime_signals[:8]))

    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    episode_dir = metadata.get("episode_dir")
    agent_log_path = metadata.get("agent_log_path")
    chat_jsonl_path = metadata.get("chat_jsonl_path")
    if episode_dir:
        lines.append(f"Episode dir: {episode_dir}")
    if agent_log_path:
        lines.append(f"Agent log: {agent_log_path}")
    if chat_jsonl_path:
        lines.append(f"Chat log: {chat_jsonl_path}")

    if metadata.get("verifier_workspace_mutation_detected"):
        paths = _mutation_paths(metadata.get("verifier_workspace_mutations"))
        restored = bool(metadata.get("verifier_workspace_restored"))
        lines.append(
            _text(
                "runtime_mutation_restored" if restored else "runtime_mutation_not_restored",
                language,
            )
        )
        if paths:
            lines.append(_text("changed_paths", language) + ", ".join(paths[:20]))

    lines.append("")
    lines.append(_text("no_claim_from_logs", language))
    return "\n".join(lines).strip()
