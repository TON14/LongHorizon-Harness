# Report selection: scorer-ranked related audit reports (`[run.semif]`)

## What and why

The executor and auditor prompts carry the past audit reports the manager
referenced for the round (`round_NNN` refs). In long runs the report history
grows, and the prompts bloat with everything ever referenced while the
current subtask usually needs only a few of them. When enabled, report
selection asks the same local SemIf scorer the semantic-salvage layer uses
one narrow question per available past audit report — *is this past audit
report relevant to the current subtask* — and the prompts keep the top-K by
P(relevant) plus every explicitly referenced round. Explicit references
always win: a report the manager named rides along whatever the scorer says.

## Configuration

Report selection shares the optional `[run.semif]` table in the project
config (`.lhht/config.toml`) with semantic salvage:

| key                          | type  | default | meaning                                              |
| ---------------------------- | ----- | ------- | ---------------------------------------------------- |
| `report_selection`           | bool  | `false` | selection switch; nothing runs unless this is `true`  |
| `report_selection_k`         | int   | `3`     | how many scored-relevant reports join the refs        |
| `report_selection_threshold` | float | `0.6`   | min P(relevant) for a candidate to be keepable        |

The selection keys reuse the table's scorer keys (`enabled`, `command`,
`model`, `revision`, `gguf`, `timeout_seconds`) through the same
`scorer_from_config` factory. **A scorer must be configured** — with
`report_selection = true` but no usable scorer, selection silently stays off
(same policy as the auditor-fast gate and effort routing: selection is an
optimization, so a misconfigured selector degrades to today's behavior
instead of blocking the run).

```toml
[run.semif]
enabled = true
command = "scripts/semif_shim.bat"
model = "<model-id>"
revision = "<revision-sha>"
report_selection = true
report_selection_k = 3
report_selection_threshold = 0.6
```

When `report_selection` is absent or `false`, no selector is resolved, no
scorer is built or called, no selection fields appear in any record, and
runtime behavior is byte-for-byte identical to a config without the keys.

## How selection runs

`src/lhht/report_selection.py` implements the decision; `src/lhht/manager.py`
applies it where the round's prompts are built from related reports, right
before `format_related_auditor_reports` formatting.

1. **Collect candidates** — every recorded round that carries an audit
   report, as `round_NNN` id plus report text.
2. **Score pointwise** (`select_reports`, once per round) — one row per
   candidate: state = the round's plan text (head+tail-condensed past 4 000
   chars) followed by that candidate's condensed report (head+tail past
   2 000 chars), question "Is this past audit report relevant to the current
   subtask?", options `relevant`/`irrelevant`. Every row shares the plan
   prefix, but the `SemanticScorer` protocol — and the `SemifCliScorer`
   behind it — scores exactly one row per `score()` invocation, so selection
   loops one call per candidate; a scorer that grows a multi-row API only
   needs to replace that loop.
3. **Select** — candidates with P(relevant) >= `report_selection_threshold`
   are kept in probability order (ties keep candidate order) and capped at
   `report_selection_k`; the kept set is unioned with the explicitly
   referenced rounds, which never compete for the K slots and are never
   dropped. Explicit refs are scored too, so their probabilities land in the
   record for measurement.

## Safety properties

- **Explicit references always survive.** A `round_NNN` reference the
  manager emitted stays in the prompt regardless of its score; degradation
  and low scores can only shrink the unreferenced additions, never the
  refs. `dropped` is exactly the complement of `kept`, so no report is both.
- **Degrades to today on any failure.** A scorer error, timeout, or
  malformed probabilities on any row — or an empty candidate list — leaves
  the round with exactly today's referenced-reports set (`degraded: true` in
  the record). Nothing in selection can raise into the management loop.
- **Only the related-reports section changes.** Selection rewrites the refs
  handed to the existing formatter; the manager prompt, histories, and every
  other prompt section are untouched.
- **Off by default, byte-identical when off.** No selector resolution side
  effects, no scorer calls, and no selection fields in rounds.jsonl or
  events.jsonl.

## Reading selection records

Every round whose executor/auditor prompts were built while selection was on
carries the decision in its record (`rounds.jsonl` / `round.json`, and the
`managed_round_recorded` event) under `executor_status.report_selection`:

| field           | meaning                                                       |
| --------------- | ------------------------------------------------------------- |
| `kept`          | ids that reached the prompt, each with its probability (null when unscored) |
| `dropped`       | ids removed by threshold/K, each with its probability         |
| `explicit_refs` | the round's `round_NNN` references, for overlap analysis      |
| `degraded`      | true when a scorer failure fell the round back to today's set |
| `k` / `threshold` | the effective settings for the round                        |

That makes the value measurable from real runs: compare rounds whose
selection dropped reports against their audit outcomes, and read the
recorded probabilities to see which past reports the scorer found relevant
to each subtask.
