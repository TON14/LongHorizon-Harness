# Effort routing: scorer-picked executor effort per round (`[run.semif]`)

## What and why

Executor episodes run long — real runs on this machine sit between 10 and 50
minutes — while the reasoning effort behind them is a static config value.
Most rounds are mechanical (run the tests, apply a reviewed edit, re-check a
log) and never need the top depth, while a few genuinely do. When enabled,
effort routing asks the same local SemIf scorer the semantic-salvage layer
uses one narrow question per cli round — *how much reasoning depth does this
subtask need* — and binds one of three **pre-built** executor adapters
(low / high / max) instead of the single configured one. The reference
result that motivated this shape: a fast classifier routing each subtask to
the right model/effort saved ~70% of cost because most subtasks never needed
the top model.

Effort is fixed at adapter construction (`ZcodeAdapter.__init__` normalizes
and stores `reasoning_effort`), so routing cannot retune a live adapter: it
selects among adapters built once at startup, around the currently
configured effort as the **default variant**.

## Configuration

Effort routing shares the optional `[run.semif]` table in the project config
(`.lhht/config.toml`) with semantic salvage:

| key                 | type  | default | meaning                                                |
| ------------------- | ----- | ------- | ------------------------------------------------------ |
| `effort_routing`    | bool  | `false` | routing switch; nothing runs unless this is `true`      |
| `effort_threshold`  | float | `0.9`   | min probability for the winning option to leave the default |

The routing keys reuse the table's scorer keys (`enabled`, `command`,
`model`, `revision`, `gguf`, `timeout_seconds`) through the same
`scorer_from_config` factory. **A scorer must be configured** — with
`effort_routing = true` but no usable scorer, routing silently stays off
(same policy as the auditor-fast gate: routing is an optimization, so a
misconfigured router degrades to today's behavior instead of blocking the
run). One extra requirement is specific to routing: the cli executor's
configured `reasoning_effort` must be one of the ranked tiers
(`minimal`/`low`/`medium`/`high`/`xhigh`/`max` — every level the agent
registry declares), because the escalation rule compares efforts. An unset
or exotic effort leaves routing off.

```toml
[run.semif]
enabled = true
command = "scripts/semif_shim.bat"
model = "<model-id>"
revision = "<revision-sha>"
effort_routing = true
effort_threshold = 0.9
```

When `effort_routing` is absent or `false`, no variants are constructed, no
scorer is built or called, no routing fields appear in any record, and
runtime behavior is byte-for-byte identical to a config without the keys.

## How routing runs

`src/lhht/manager.py` implements the decision at `_executor_binding()`;
`src/lhht/cli.py` builds the variants at role construction.

1. **Build variants** (`_cli_executor_effort_variants`, cli.py) — only when
   the flag is on and the scorer is configured: three cli-executor adapters
   at efforts low / high / max, identical to the configured executor except
   for effort. The configured effort itself is never rebuilt — the adapter
   cache is keyed by effort, so the matching variant *is* the default
   adapter. A backend that rejects one of the levels (none of the supported
   agents do) degrades to routing off with a warning, never a failed run.
2. **Classify** (`_route_effort`, once per cli round, memoized by round
   index) — state is the round's extracted plan text, question is "How much
   reasoning depth does this executor subtask need?", options are
   `mechanical` (→ low), `standard` (→ high), `deep` (→ max) with
   one-line descriptions. The plan text is passed unclipped.
3. **Select** — the top option must reach `effort_threshold` to leave the
   default; ties go to the earliest option (the same determinism rule as
   salvage). Scorer errors, timeouts, and malformed answers all keep the
   default. A confident pick equal to the configured effort is recorded as
   the default having run.

## The escalation rule (self-correction)

A downgrade is only allowed while it keeps auditing clean. If the most
recent round that ran a routed executor was **downgraded** (its variant
ranks below the configured default) and its audit did **not** parse
complete + clean + aligned, the next routed round must not downgrade again:
a below-default pick is suppressed to the default effort, while `deep`
(≥ default) still runs. Manager-only rounds in between never redeem a
downgrade — the walk-back skips them — and a failed, timed-out, cancelled,
or fast-gate-skipped audit all count as not passed, so a downgrade is only
repeated after a clean-complete audit redeemed it. The suppressed decision
is recorded (`escalated: true`, `skipped: "escalated_to_default"`) together
with what the classifier actually said.

## Safety properties

- **Only the cli-executor binding changes.** Routing picks among
  pre-built cli-executor adapters at the one point the loop binds the
  executor for a round. It never constructs, rebinds, or retunes the
  manager, auditor, GUI-executor, format-repair, or final-response roles;
  GUI rounds bypass the router entirely and bind exactly as today.
- **Variants share no mutable per-run state.** Each `CommandAgentAdapter`
  episode writes its prompt under a UUID-suffixed filename, so a shared
  `prompt_dir` cannot collide; ZCode protocol sessions are created and
  discarded inside each episode's runner subprocess (nothing persists
  between episodes); the provider config the variants ensure is identical
  for the same model and written only at construction.
- **Degrades to today on any failure.** Scorer errors, timeouts, malformed
  probabilities, config-resolution failures, missing variants, and
  unrankable default efforts all fall back to the configured default
  adapter. Nothing in routing can raise into the management loop.
- **Off by default, byte-identical when off.** No variants, no scorer
  calls, and no routing fields in rounds.jsonl or events.jsonl.

## Reading routing records

Every round whose executor ran while routing was on carries the decision in
its record (`rounds.jsonl` / `round.json`, and the `managed_round_recorded`
event) under `executor_status.effort_routing`:

| field           | meaning                                                            |
| --------------- | ------------------------------------------------------------------ |
| `variant`       | the effort that actually ran (`low` / `high` / `max`, or the default) |
| `used_default`  | true when the configured default effort ran (fallback or match)     |
| `probabilities` | one probability per option (`mechanical`/`standard`/`deep`), or null |
| `classified`    | the argmax option id, or null when the scorer was unusable          |
| `escalated`     | true when a below-default pick was suppressed by the escalation rule |
| `skipped`       | why the default ran: `scorer_unavailable`, `below_threshold`, `escalated_to_default`, or null |
| `default_effort`| the configured default effort for context                           |

That makes the cost question measurable from real runs: compare episode
durations of downgraded rounds (`variant` below `default_effort`) against
their audit outcomes, count how often escalation had to correct a downgrade,
and read the recorded probabilities to see which plans the classifier found
mechanical versus deep.
