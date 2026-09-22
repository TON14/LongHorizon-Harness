# Semantic salvage for control lines (`[run.semif]`)

## What and why

Role control lines — the auditor's three verdict headers and the manager's
`Next:` route line — are parsed by exact trilingual regexes and string sets.
Every observed synonym had to be added one by one (recent commits: the RU
spellings «выровнено», «соответствует», «без нарушений»), and a miss still
defaults the verdict (auditor: blocked / suspect / unknown; manager:
`MANAGER_NEXT_INVALID`), burning a round or triggering a format-repair
episode of up to 600 s.

The semantic salvage layer replaces that treadmill with one SemIf decision on
the miss path only: the legal values of the missed control line become typed
options, and a small local model — the sibling project SemIf — returns one
probability per option, read directly from its logits (one forward pass, no
text generation). A miss can then be recovered as the highest-probability
legal value instead of silently failing.

## Configuration

Salvage is off unless the operator opts in via an optional `[run.semif]`
table in the project config (`.lhht/config.toml`):

| key               | type   | default    | meaning                                                     |
| ----------------- | ------ | ---------- | ----------------------------------------------------------- |
| `enabled`         | bool   | `false`    | master switch; nothing runs unless this is `true`            |
| `command`         | string | —          | path to the real `semif-score` executable (see below)        |
| `model`           | string | —          | model id, passed as `--model`                               |
| `revision`        | string | —          | model revision, passed as `--revision`                      |
| `gguf`            | string | unset      | optional GGUF checkpoint for the llamacpp backend            |
| `threshold`       | float  | `0.8`      | minimum top-option probability (must be > 0 and at most 1)   |
| `timeout_seconds` | int    | `30`       | hard wall-clock box per `semif-score` subprocess             |

`command`, `model` and `revision` are required when `enabled = true`;
an enabled-but-unusable section refuses to load at startup instead of
quietly disabling salvage. Unknown keys are rejected.

When the section is absent or `enabled = false`, runtime behavior is
byte-for-byte identical to a config without it: no scorer is ever
constructed or invoked.

```toml
[run.semif]
enabled = true
command = "/path/to/semif-score"    # the real executable — see the caveat below
model = "<model-id>"
revision = "<revision-sha>"
# gguf = "/path/to/checkpoint.gguf"
# threshold = 0.8
# timeout_seconds = 30
```

## Where it is wired

- **Auditor verdict headers** (`src/lhht/auditor_agent.py`): each of the
  three parsers — `_parse_status_control_header` (complete / incomplete /
  blocked), `_parse_integrity_control_header` (clean / suspect / violation)
  and `_parse_contract_audit_control_header` (aligned / unknown /
  needs_revision / invalid) — scans the leading control lines with its exact
  trilingual regex first; only when that misses does it hand the
  label-carrying line to the scorer. The parsed audit report records the
  provenance: its `control_salvage` field lists, per salvaged line, the
  control name, the recovered value, and one probability per legal value, so
  rounds.jsonl / report.json stay auditable. A header recovered by salvage
  also counts as valid for the format-repair trigger
  (`has_valid_auditor_control_header`), so a recoverable wording no longer
  burns the repair episode.
- **Manager route line** (`src/lhht/role_prompts.py`): after the exact
  string sets miss, the miss path of `parse_role_manager_next_step()` tries
  salvage over the five legal routes — gui, cli, ask, done, blocked — before
  falling back to `MANAGER_NEXT_INVALID`.

## Safety rules

- **Miss-only.** Salvage runs only when the exact regex / string-set match
  failed. A regex hit is never re-judged and never constructs or invokes the
  scorer.
- **Off by default.** With `[run.semif]` absent or disabled, no scorer
  object exists and behavior is byte-for-byte today's.
- **Threshold semantics.** The top option probability must be at least
  `threshold` (default 0.8) for the salvaged value to be accepted — the
  value and its probabilities are then used and recorded. Below the
  threshold, or on any scorer failure, the caller keeps today's
  default/fallback verdict. Ties go to the earliest legal value, keeping
  salvage deterministic.
- **Fail-closed.** Every scorer failure mode — invalid row, missing
  executable, non-zero exit, timeout, missing or malformed output — returns
  `None` and degrades to today's fallback; nothing ever raises into the
  parsers.
- **Subprocess-only.** SemIf is never imported in-process and is not a
  runtime dependency; the scorer runs `semif-score` as a child process.
- **Clean temp files.** Each call uses its own `TemporaryDirectory` under
  the system temp area (prefix `lhht-semif-`): the input JSONL is written
  there, SemIf creates the output file, and both vanish with the directory.
- **Memoized.** At most one scorer call per missed control line per report.

## The `command` key must be the real executable

`SemifCliScorer` launches the configured `command` directly as a subprocess
(argv[0], no shell), passing `--mode direct --model <model> --revision
<revision> --input <tmp>/input.jsonl --output <tmp>/output.jsonl`, plus
`--backend llamacpp --gguf <path>` whenever `gguf` is set — `semif-score`
rejects `--gguf` without the llamacpp backend, and llamacpp refuses to start
without a GGUF checkpoint, so the two flags always travel together.

`command` must therefore be a single executable path that speaks the
`semif-score` CLI: the installed entry-point executable, or a wrapper that
execs it with the arguments verbatim. A shell command line
(`python -m ...`, `uv run ...`) or a shim that rewrites flags will not be
split or translated, so it will simply fail — which, by design, degrades to
no salvage rather than crashing the run.
