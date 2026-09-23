# Scorer stats: measuring what the scorer did in one run (`lhht scorer-stats`)

## What and why

Every scorer feature in the loop — semantic salvage, the auditor-fast
pre-gate, the post-audit verdict cross-check, effort routing, report
selection, and the round-dedup detector — records rich payloads on the
round ledger as it runs. Reading those records has so far meant ad-hoc
scripts that live outside the repository. `lhht scorer-stats` reads one
finished run directory and prints a single plain-text summary of
everything the scorer did in it: the deploy-then-measure doctrine's
first-class measuring stick.

```
lhht scorer-stats <run-dir> [--fixture PATH]
```

- `<run-dir>` — the run directory to summarize (`<runs-root>/<run-id>`).
  It is resolved through the same run-boundary helpers the dashboard
  uses; a symlinked or foreign layout is refused, while a reserved run
  with no ledger yet summarizes as zero rounds.
- `--fixture PATH` — also grow the SemIf calibration fixture by mining
  the run's control lines into `PATH` (default
  `data/semif-fixture.jsonl`, created on first use; pass an empty string
  to skip). Rows are appended in the established decision format —
  `{id, kind, state, question, options, label}` — and deduplicated by
  id, so re-summarizing a run never duplicates rows. The command prints
  how many rows were added.

The summary never crashes on absent fields: an older run, or a run with
a feature switched off, renders explicit zeros or a "not recorded" line
for that feature.

## What each number means

`src/lhht/scorer_stats.py` reads the run's
`lhht/role_orchestration/rounds.jsonl` with the manager's own ledger
reader (the latest entry per `round_index` wins; malformed lines are
skipped) and reports one block per feature.

**Semantic salvage** — lines rescued per control (`route`, `status`,
`integrity`, `contract_audit`, `acceptance_none`), with the maximum
winning probability among the rescues. Salvage provenance is not
persisted (`AuditReport.control_salvage` lives only in memory), so this
block is *rebuilt*: every round's `auditor_report` is re-parsed and
every route-label line in `plan_text` re-judged with the scorer
configured in this project's `[run.semif]` table. Because salvage only
ever runs on the regex-miss path, the rebuild reproduces exactly the
lines that were (or would be) rescued. With no usable scorer the block
renders `not recorded (no scorer configured)` rather than guessing. A
salvage probability from a machine without the shim is therefore "not
measured", never zero.

**Auditor-fast gate** — `evaluations` (rounds that gathered evidence and
scored the battery), `skips` (rounds whose slow auditor episode was
skipped after a confident fail), and `passes` (evaluations that let the
slow auditor run). `slow audits` and its median duration cover the
rounds where the slow auditor episode actually ran.

`estimated audit minutes saved` is **an estimate, not a measurement**:
it multiplies the skip count by this run's own median slow-audit
duration. It assumes a skipped audit would have taken about as long as
the run's typical audit — the gate skips precisely the rounds it
believes hopeless, which need not resemble the audited rounds. Runs
with no slow audits to measure against report the skip count alone
(`not estimable`).

**Cross-check** — how many verdicts were checked (three per round that
ran a cross-check), how many agreed with the auditor's parsed verdicts,
and each flagged disagreement with both sides' values (the auditor's
verdict, the scorer's answer, and the scorer's confidence). The
cross-check is advisory; disagreements never altered the run.

**Effort routing** — rounds per effort variant, each with the mean
routing probability (the scorer's confidence in the classification that
chose the variant) and the mean executor episode duration, plus the
escalation count (rounds held at least at the default effort after a
downgrade that was not audited clean-complete).

**Report selection** — rounds that ran a selection, and across them the
candidate past audit reports kept in prompts vs. dropped, plus how many
selections degraded to today's keep-the-references behavior on a scorer
failure.

**Round dedup** — how many consecutive-plan comparisons were recorded,
and each flagged repeat with its same-work probability. Flags are
advisory loop suspicions, nothing more.

## Where the data comes from

| feature           | ledger location                                  | key fields                                                         |
| ----------------- | ------------------------------------------------ | ------------------------------------------------------------------ |
| salvage           | rebuilt from `auditor_report` / `plan_text`      | (re-parsed; not persisted)                                          |
| gate              | `auditor_status`                                 | `auditor_fast_gate`, `status == "skipped_by_fast_gate"`, `duration_ms` |
| cross-check       | `auditor_status`                                 | `auditor_cross_check` (`agreements`, `disagreements`, `controls`)   |
| effort routing    | `executor_status`                                | `effort_routing` (`variant`, `probabilities`, `classified`, `escalated`), `duration_ms` |
| report selection  | `executor_status`                                | `report_selection` (`kept`, `dropped`, `degraded`)                  |
| round dedup       | `manager_status`                                 | `round_dedup` (`flagged`, `probability`)                            |

## Fixture mining details

With `--fixture`, the run's control lines become calibration rows:
manager route lines are taken from `plan_text` (the last route-label
line — the one the harness plan extractor keys on) with the recorded
`next_step` as gold; invalid routes are skipped. Audit header lines are
taken from the first non-empty lines of `auditor_report` with the
canonical value the `infer_*` parsers derive. A header line the exact
value regex did not accept has no trustworthy canonical value (the
parsers would substitute their fallback), so it is skipped for human
labeling — the same review separation the original mining pipeline kept.
Row ids embed the run id and round (`<run-id>#r<index>-route`,
`<run-id>#r<index>-<control>`), which is what makes dedup-by-id stable
across invocations.
