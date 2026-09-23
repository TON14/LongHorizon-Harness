# Cross-check: a post-audit second opinion on the verdicts (`[run.semif]`)

## What and why

The auditor-fast gate screens rounds *before* the slow auditor runs, but
nothing today double-checks the slow auditor's own verdicts *after* it runs.
When enabled, the cross-check asks the same local SemIf scorer the salvage
layer and the gate use to independently answer the auditor's three control
verdicts — status (`complete` / `incomplete` / `blocked`), integrity
(`clean` / `suspect` / `violation`), contract audit (`aligned` / `unknown` /
`needs_revision` / `invalid`) — against the **same live evidence** the gate
gathers: the workspace VCS status, the executor's visible output, the task
contract's acceptance constraints, and the manager's plan.

A verdict is **flagged** when the scorer's argmax answer differs from the
auditor's verdict AND the scorer's probability for its own answer reaches
`cross_check_threshold` (default 0.9). A flag is doubt surfaced for the
operator, nothing more.

## Configuration

The cross-check shares the optional `[run.semif]` table in the project
config (`.lhht/config.toml`):

| key                      | type  | default | meaning                                                  |
| ------------------------ | ----- | ------- | -------------------------------------------------------- |
| `cross_check`            | bool  | `false` | cross-check switch; nothing runs unless this is `true`    |
| `cross_check_threshold`  | float | `0.9`   | min probability for the scorer's own answer to flag (inclusive) |

It reuses the table's scorer keys (`enabled`, `command`, `model`,
`revision`, `gguf`, `timeout_seconds`) through the same `scorer_from_config`
factory as the gate. Like the gate, it is an advisory extra rather than a
correctness requirement: with `cross_check = true` but no usable scorer it
silently stays off instead of refusing to load. With `cross_check` absent or
`false`, no scorer is constructed, no evidence is gathered, and runtime
behavior is byte-for-byte identical to a config without the keys.

```toml
[run.semif]
enabled = true
command = "scripts/semif_shim.bat"
model = "<model-id>"
revision = "<revision-sha>"
cross_check = true
cross_check_threshold = 0.9
```

## How it runs

`src/lhht/auditor_fast.py` implements the comparison (`cross_check`);
`src/lhht/manager.py` invokes it on the normal audit path — after the
auditor report is parsed and before the round record is persisted — once per
audited round. The fast-gate skip path never runs it (there is no auditor
report to cross-check).

1. **Gather evidence once** — the manager calls the same
   `gather_gate_evidence` the pre-gate uses, with the same round context
   (plan, executor output, task contract, guard exclusions); the cross-check
   itself consumes the evidence as given and never re-gathers.
2. **Ask three questions** — one per control, all over one shared bounded
   evidence state: "Given the evidence, is the task status complete?",
   "Given the evidence, is the integrity status clean?", "Given the
   evidence, is the contract audit aligned?" The options are the auditor's
   own legal values, so a scorer answer maps one-to-one onto a parsed
   control-header verdict. Ties go to the earliest option (the same
   determinism rule as salvage and the gate).
3. **Compare** — per control, the scorer's argmax is set against the parsed
   report's verdict. `agrees` records a match; `flagged` records a mismatch
   whose winning probability reached the threshold.

## The advisory-only contract

- **Never modifies anything.** The cross-check runs after the report is
  parsed and only *adds* a record. It cannot change the auditor's verdict
  fields, the `role_done` flow, routing, or `done` acceptance
  (`_latest_auditor_is_clean_complete` still reads the auditor report alone;
  a flagged round still parses complete + clean + aligned and still
  satisfies completion).
- **All-or-nothing on scorer health.** Any scorer failure — an exception, a
  timeout/`None` answer, malformed probabilities, a verdict outside the
  legal values — yields **no cross-check record at all**, silently. A
  broken second opinion never leaves a half-record behind and never blocks
  the run.
- **Off by default, byte-identical when off.** With `cross_check` absent or
  `false`, no scorer is constructed, no evidence is gathered, and no
  cross-check events or record fields exist.

## Reading cross-check records

Every audited round where the cross-check ran is fully recorded:

- **rounds.jsonl / round.json**: the round record's `auditor_status` gains
  an `auditor_cross_check` payload:
  - `threshold` — the threshold actually applied;
  - `flagged` — whether any control confidently disagreed;
  - `agreements` — controls where the scorer's argmax matched the auditor;
  - `disagreements` — controls flagged by the threshold rule;
  - `controls` — the full per-control measurement base, including
    below-threshold mismatches (in neither list above).
  Each control entry carries `control`, `question`, `options`,
  `auditor_verdict`, `scorer_answer`, `probabilities` (the scorer's full
  per-option distribution), `probability` (the scorer's mass on its own
  answer), `auditor_probability` (the scorer's mass on the auditor's
  verdict — both sides of the disagreement are measurable), `agrees`, and
  `flagged`.
- **events.jsonl**: one `auditor_cross_check` event — `round` plus the same
  payload — **only** when a disagreement was flagged. Agreements and
  degraded runs emit nothing.

That makes the operator's scan cheap: watch for `auditor_cross_check`
events (or `auditor_status.auditor_cross_check.flagged` in the ledger),
then read the flagged control's probabilities to see which evidence the
scorer weighed, at what confidence, against the auditor's verdict.
Aggregate measurements fall out of the same records — flag rate over runs,
per-control disagreement rates, and the distribution of
`auditor_probability` on flagged rounds (how much mass the scorer still
gave the auditor's side).
