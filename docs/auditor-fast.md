# Auditor-fast: a scorer-based pre-gate for the slow auditor (`[run.semif]`)

## What and why

Auditor episodes are the run's biggest time sink — on this machine's real
runs, a median of 16.4 minutes, a worst case of 106 minutes, and 25.4 hours
across 15 runs — and a large share of those audits land on rounds the
manager bounces anyway. When enabled, a fast gate runs after the executor
episode and **before** the auditor episode is launched: it gathers live
evidence and asks the same local SemIf scorer the semantic-salvage layer
uses a small battery of narrow classification questions. Only a CONFIDENT
fail short-circuits the slow auditor for that round; everything else
proceeds exactly as today.

The evidence is deliberately independent of the executor's honesty: the
workspace VCS status, the executor's *visible output*, the task contract's
acceptance constraints, and the manager's own plan. An earlier battery
design judged the executor's self-report prose and was rejected for
precision 0.30; this gate never scores the narrative's tone. The one
question that mentions the executor's claims cross-checks them against the
repository state ("do the VCS status changes match the claims about which
files changed"), so a mismatch is grounded in live VCS evidence, not in how
convincing the output reads.

## Configuration

The gate shares the optional `[run.semif]` table in the project config
(`.lhht/config.toml`) with semantic salvage:

| key                       | type  | default | meaning                                              |
| ------------------------- | ----- | ------- | ---------------------------------------------------- |
| `auditor_fast`            | bool  | `false` | gate switch; nothing runs unless this is `true`       |
| `auditor_fast_threshold`  | float | `0.95`  | min probability for a fail-side option to vote fail   |

The gate reuses the table's scorer keys (`enabled`, `command`, `model`,
`revision`, `gguf`, `timeout_seconds`) through the same
`scorer_from_config` factory. **A scorer must be configured** — with
`auditor_fast = true` but no usable scorer, the gate silently stays off
(unlike salvage, which refuses to load: skipping the slow auditor is an
optimization, so a misconfigured gate degrades to today's behavior instead
of blocking the run). When `auditor_fast` is absent or `false`, no gate is
constructed and runtime behavior is byte-for-byte identical to a config
without it.

```toml
[run.semif]
enabled = true
command = "scripts/semif_shim.bat"
model = "<model-id>"
revision = "<revision-sha>"
auditor_fast = true
auditor_fast_threshold = 0.95
```

## How the gate runs

`src/lhht/auditor_fast.py` implements the gate; `src/lhht/manager.py`
invokes it between the executor episode result and the auditor episode
launch, once per round, only for rounds whose executor episode succeeded.

1. **Gather evidence** (`gather_gate_evidence`) — best-effort, never raises:
   - workspace VCS status via `git status --porcelain=v1 -b`, falling back
     to `svn status`, each capped at 120 s; a failure or timeout records a
     note and the two VCS-dependent criteria abstain;
   - status lines whose path falls under a `guard_exclude_paths` entry are
     filtered exactly like the auditor guard's snapshot exclusions, so
     excluded churn (build outputs) cannot vote;
   - the executor's visible output, the manager's plan text, and the task
     contract's acceptance constraints (reusing the contract the manager
     already maintains; the acceptance-constraints section is split into
     individual constraints, bullets / `(N)` / `N.` markers, at most 8).
2. **Run the battery** (`run_gate`) — one narrow scored question per
   criterion, each over a bounded evidence state:
   - per acceptance constraint: "Is this constraint satisfied by the
     evidence?" — options `yes` / `no` / `undetermined`, fail side `no`;
   - "Do the VCS status changes match the executor's claims about which
     files it changed?" — options `match` / `partial` / `no_changes_claimed`
     / `mismatch`, fail side `mismatch`;
   - "Is the workspace status empty of the round's expected changes?" —
     options `yes` / `no`, fail side `yes`.
   The last two are emitted as skipped (never scored, never voted) when the
   workspace status is unavailable.
3. **Decide** — a criterion votes fail only when its fail-side option both
   wins the argmax and reaches `auditor_fast_threshold`; ties go to the
   earliest option (same determinism rule as salvage). The verdict is
   `fail` only if at least one criterion confidently fails; otherwise
   `pass`, and the auditor episode launches exactly as today.

**Batching:** the `SemanticScorer` protocol (and the `SemifCliScorer` shim
behind it) accepts one row per `score()` call, so the battery is a loop of
one scorer invocation per criterion — with the default `timeout_seconds`
of 30 s per call, a worst-case battery of ~10 criteria is bounded well
below the minutes a single auditor episode costs. Batching every question
into one invocation would need a new multi-row scorer API; that was not
added, to keep the reuse of the existing plumbing exact.

## What a fail does (and never does)

On `fail` the manager skips the auditor episode for the round and
synthesizes the round's audit record: `status=incomplete`,
`integrity=clean`, `contract=unknown`, with report text starting
`auditor-fast gate:` followed by the failing criteria with their
probabilities and the matched VCS status lines. The report embeds a valid
three-line control header (`Status: incomplete` / `Integrity: clean` /
`Contract audit: unknown`) below the gate line, so every downstream
consumer parses the same verdicts the manager recorded. The record is
routed through the existing incomplete-audit flow: the next round's
manager prompt contains the findings and re-binds the executor.

## Safety rules

- **Fail-only.** The gate can produce exactly one action: skip the slow
  auditor. It has no path to a clean or complete outcome, and the `done`
  acceptance logic (`_latest_auditor_is_clean_complete`) is untouched — it
  still requires a real auditor report that parses complete + clean +
  aligned, which a gate-fired round never has (its report parses
  incomplete by construction).
- **Threshold semantics.** `auditor_fast_threshold` (default 0.95) is the
  minimum probability for a *winning* fail-side option. High-but-losing
  fail probabilities do not vote; below-threshold winners do not vote.
- **Degrades to today on any failure.** Scorer errors, timeouts, malformed
  answers, unavailable VCS status, and config-resolution failures all
  abstain or disable the gate — the auditor then runs exactly as today.
  Nothing in the gate can raise into the management loop.
- **Off by default, byte-identical when off.** With `auditor_fast` absent
  or `false`, no gate is constructed, no scorer is invoked, and no gate
  events or record fields exist.
- **Never judges self-report prose as primary evidence.** Every criterion
  is grounded in live evidence (VCS status, contract constraints, plan);
  the executor output only enters as the object of a cross-check against
  the repository state.

## Reading gate records (effectiveness measurement)

Every round where the gate ran — fired or not — is fully recorded:

- **events.jsonl**: one `auditor_fast_gate` event per gate run with
  `round`, `verdict`, the full `battery` (per criterion: `id`, `question`,
  `options`, `fail_option`, `probabilities`, `winner`, `votes_fail`,
  `skipped`), and an `evidence_digest` (`vcs_status_available`, `vcs_tool`,
  `vcs_status_line_count`, `executor_output_chars`,
  `acceptance_constraint_count`, `notes`). A fired round additionally has
  an `auditor_fast_gate_skip` event.
- **rounds.jsonl / round.json**: the round record's `auditor_status`
  carries the same `auditor_fast_gate` payload — on fired rounds next to
  `status: "skipped_by_fast_gate"` with the synthesized
  incomplete/clean/unknown values, and on audited rounds next to the slow
  auditor's own verdicts.

That makes three measurements directly possible from real runs:

1. **How often the gate fired:** count `auditor_fast_gate_skip` events (or
   rounds with `auditor_status.status == "skipped_by_fast_gate"`).
2. **What the slow auditor said when the gate did NOT fire:** for rounds
   with `auditor_status.auditor_fast_gate.verdict == "pass"`, compare the
   adjacent auditor verdicts — the payload sits next to them in the same
   record.
3. **Whether gate-fired rounds were indeed bounced:** the synthetic report
   is an `incomplete` audit in the rounds ledger, so the next manager turn
   re-binds the executor; check the following round's plan for the re-bind
   and the absence of any later clean-complete audit claiming that round's
   deliverables.

False-positive analysis (rounds the gate fired on that a real audit would
have passed) reduces to reading the recorded battery: each failing
criterion's question, option set, and probabilities show exactly which
evidence convinced the scorer, at what confidence.
