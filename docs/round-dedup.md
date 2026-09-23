# Round-dedup: flagging re-planned subtasks (`[run.semif]`)

## What and why

Long-horizon runs have a quiet failure mode: the manager re-plans essentially
the same subtask round after round — the phrasing varies, the work repeats —
and the round budget burns away in a loop nothing surfaces. Each round looks
locally reasonable; only the *comparison* between consecutive plans exposes
the pattern.

When enabled, the round-dedup detector makes that comparison explicit. Right
after the manager's plan is extracted each round, the same local SemIf scorer
the salvage layer and the gate use answers one question over both plans:
**"Is this round's planned subtask essentially the same work as the previous
round's?"** — options `same_work` / `different_work`. A round is **flagged**
when `same_work` both wins the argmax and reaches `round_dedup_threshold`
(default 0.9, inclusive; ties go to the earliest option, the same determinism
rule as salvage and the gate). A flag is a surfaced suspicion of a planning
loop, nothing more.

The first round of a run has no previous plan and is never compared.

## Configuration

The detector shares the optional `[run.semif]` table in the project config
(`.lhht/config.toml`):

| key                      | type  | default | meaning                                                  |
| ------------------------ | ----- | ------- | -------------------------------------------------------- |
| `round_dedup`            | bool  | `false` | detector switch; nothing runs unless this is `true`       |
| `round_dedup_threshold`  | float | `0.9`   | min probability for a confident `same_work` to flag (inclusive) |

It reuses the table's scorer keys (`enabled`, `command`, `model`,
`revision`, `gguf`, `timeout_seconds`) through the same `scorer_from_config`
factory as the gate and the cross-check. Like them, it is an advisory extra
rather than a correctness requirement: with `round_dedup = true` but no
usable scorer it silently stays off instead of refusing to load. With
`round_dedup` absent or `false`, no scorer is constructed, no comparison is
made, and runtime behavior is byte-for-byte identical to a config without
the keys.

```toml
[run.semif]
enabled = true
command = "scripts/semif_shim.bat"
model = "<model-id>"
revision = "<revision-sha>"
round_dedup = true
round_dedup_threshold = 0.9
```

## How it runs

`src/lhht/round_dedup.py` implements the comparison (`detect_repeat`);
`src/lhht/manager.py` invokes it once per round, right after the manager
plan is extracted and written — and only when a previous round's plan exists.

1. **Build one state** — both plan texts, each condensed to a bounded slice
   (whitespace collapsed, capped at 1 500 characters), clearly labeled
   ("Previous round's manager plan:" / "This round's manager plan:") and
   separated, so the small local model sees exactly the two things being
   compared.
2. **Ask one question** — "Is this round's planned subtask essentially the
   same work as the previous round's?" with the two options above.
3. **Decide** — `same_work` winning the argmax with probability >= threshold
   marks the round (`flagged`); any other scored answer is recorded as not
   flagged so the comparison probability stays measurable.

## The advisory-only contract

- **Never modifies anything.** The detector runs after the plan is extracted
  and only *adds* an annotation. It cannot change the plan text, routing
  (`next_step`), harness feedback, the executor/auditor flow, or `done`
  acceptance (`_latest_auditor_is_clean_complete` still reads the auditor
  report alone; a flagged round still parses complete + clean + aligned and
  still satisfies completion). Records of the manager's own outputs are
  untouched; only the dedup annotation is added.
- **All-or-nothing on scorer health.** Any scorer failure — an exception, a
  timeout/`None` answer, malformed probabilities — yields **no dedup record
  at all**, silently, and never blocks the round. A missing previous plan
  (the first round) and a missing current plan behave the same.
- **Off by default, byte-identical when off.** With `round_dedup` absent or
  `false`, no scorer is constructed, `detect_repeat` is never called, and no
  dedup events or record fields exist: round records serialize exactly as
  they did before the feature.

## Reading round-dedup records

Every round whose plan was compared (round 2 onward, feature on, scorer
healthy) is fully recorded:

- **rounds.jsonl / round.json**: the round record's `manager_status` gains
  a `round_dedup` payload:
  - `threshold` — the threshold actually applied;
  - `flagged` / `same` — whether this round was confidently judged the same
    work (the two are equal; `same` names the decision, `flagged` its
    operational meaning);
  - `probability` — the scorer's mass on `same_work`: **the comparison
    probability**, present whether or not the threshold was met, so
    near-misses stay measurable;
  - `probabilities` — the scorer's full distribution over both options.
- **events.jsonl**: one `round_dedup_flag` event — `round` plus the same
  payload — **only** when a repeat was flagged. Not-flagged comparisons and
  degraded runs emit nothing.

The operator's scan is cheap: watch for `round_dedup_flag` events (or
`manager_status.round_dedup.flagged` in the ledger), then read the flagged
round's plan next to the previous round's (`rounds/<n>/manager_plan.txt`) to
see the loop for yourself. Aggregates fall out of the same records — flag
rate over runs, and the distribution of `probability` on unflagged rounds
(how close the manager came to repeating itself without crossing the line).
