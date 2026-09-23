# LongHorizon-Harness — TON14 fork

This is my actively maintained fork of
[AMAP-ML/LongHorizon-Harness](https://github.com/AMAP-ML/LongHorizon-Harness),
a role-orchestrated harness (Manager → Executor → Auditor) that turns agent
CLIs into long-horizon computer-use systems. Upstream went quiet after
2026-08-20 (v0.1.7), so the fork carries the project forward on its own line.

The complete original project documentation — install walkthroughs, the loop
engineering story, benchmarks, configuration reference — is preserved verbatim
from the v0.1.7 tag: see [`docs/upstream/README.md`](docs/upstream/README.md)
(English) and [`docs/upstream/README.zh-CN.md`](docs/upstream/README.zh-CN.md)
(中文). Everything below is specific to this fork.

## What the fork adds

- **Three production backends, one harness.** ZCode (`lhht run --agent zcode`):
  drives the headless runtime bundled with the ZCode desktop app on GLM models
  (`glm-5.3` default), with role-scoped permission modes and effort riding the
  app-server session protocol. Claude Code: `claude-opus-5-5` default, five
  explicit effort tiers. Codex: the npm CLI, which shares the desktop app's
  login through `~/.codex` (no separate sign-in). Per-role agent/model/effort
  mixing never translates values across backends. Battle-tested operator
  guide for the ZCode line: [`ZCODE-RUN-GUIDE.md`](ZCODE-RUN-GUIDE.md).
- **A local semantic scorer wired into the loop (SemIf sidecar).** A resident
  GPU server (`lhht server start`, shared by every parallel run) answers
  narrow classification questions by reading option probabilities straight
  from a 4B model's logits — no text generation, ~0.1 s per decision. The
  harness consults it at nine decision points:
  [control-line salvage](docs/semif-salvage.md) (manager route + auditor
  verdict headers; regex-first, never overriding a deterministic parse),
  acceptance-none salvage, the fail-only [auditor-fast pre-gate](docs/auditor-fast.md)
  (skips the slow auditor solely on a confident fail; can never certify
  work), the advisory post-audit [cross-check](docs/cross-check.md),
  [effort routing](docs/effort-routing.md) (mechanical/standard/deep →
  low/high/max executor variants), [report selection](docs/report-selection.md),
  [round dedup](docs/round-dedup.md), and an in-role `score` MCP tool that
  every backend auto-registers for its roles. Every feature silently degrades
  to plain harness behavior while the server is down; `lhht server doctor`
  checks the whole chain, and `lhht scorer-stats <run-dir>` reports what the
  scorer actually gave a finished run.
- **Per-role MCP visibility.** `[run] mcp_allow` / `mcp_blocked`, replaceable
  per role: by default everything is allowed and lhht does not interfere with
  the agent's own server discovery; the moment a list restricts anything,
  that role loads only the admitted servers — enforced identically on all
  three backends (the block list always wins).
- **Bilingual control tokens**: the manager/auditor parsers accept route
  lines and control markers in both English and Russian.
- **A modern toolchain floor**: Python ≥ 3.14 (tomli fallbacks removed),
  Node.js ≥ 22 with CI provisioning 24; `make check` is the pre-push gate.
- **Honest CI**: `tests.yml` runs the Python suite on ubuntu/windows × 3.14
  plus a web job (frontend core suite + typecheck) on every push and pull
  request; `release.yml` builds and verifies distributions without publishing
  (the PyPI name belongs to upstream).

## Configuring

`lhht init` scaffolds a `.lhht/config.toml` with the fork's battle-tested
defaults (10800 s episode timeouts, 40 rounds, a fully commented
`[run.semif]` section). Ready-to-copy example configs per backend live in
[`examples/`](examples/):

- [`examples/config-zcode.toml`](examples/config-zcode.toml) — GLM on the Z.ai plan
- [`examples/config-claude.toml`](examples/config-claude.toml) — Claude Code
- [`examples/config-codex.toml`](examples/config-codex.toml) — Codex / OpenAI

Each is self-sufficient (roles, timeouts, the scorer section) and marked
where a path is machine-specific. `lhht doctor` run from the project
directory verifies the chosen backend; `lhht server doctor` verifies the
scorer chain end to end.

## Installing

The PyPI package `lh-harness` is upstream's release and does not contain the
work above — install from this repository:

```bash
pip install lhht            # or: uv tool install lhht
```

> **Naming note:** the fork owns its identity end to end — the PyPI
> distribution, the console command, the import module (`import lhht`), the
> `LHHT_*` environment variables, and the `.lhht/` state directory are all
> `lhht`. The original `lh-harness` installs alongside it cleanly, so both
> can drive the same project folder for comparison.

Building from source (what the development loop uses):

```bash
git clone https://github.com/TON14/LongHorizon-Harness.git
cd LongHorizon-Harness
npm run build --prefix frontend/web   # the Web workbench bundle
uv tool install --force .   # installs the `lhht` command
```

## Developing

```bash
make help        # all targets
make check       # both suites + typecheck, everything expected green
make dev-api     # control API on 127.0.0.1:8799
make dev-web     # Vite dev server on :5173, proxying /api
```

The tool installs as **`lhht`** (import module and the `.lhht/` state directory keep their
historical names). The harness is also used to develop itself: a repo-root `.lhht/config.toml`
drives the roles on the ZCode backend with the semantic scorer enabled, and tasks run with
`lhht run --task @task.md`. Agents edit the working tree; the operator reviews, runs
`make check`, and commits. Feature docs live in [`docs/`](docs/).

## Credits

The foundation is the AMAP-ML team's work — every contributor is listed on the
[contributors page](https://github.com/TON14/LongHorizon-Harness/graphs/contributors).
The fork's line is maintained by [TON14](https://github.com/TON14).
