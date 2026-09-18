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

- **ZCode agent backend** (`lhht run --agent zcode`): drives the headless runtime
  bundled with the ZCode desktop app on GLM models (`glm-5.3` default,
  `glm-5.3-flash`), with role-scoped permission modes (`plan` for the manager
  and auditors, `yolo` for executors). The reasoning effort (`low`/`high`/`max`)
  rides in an isolated per-run copy of ZCode's session database plus a
  `.zcode/config.json` provider declaration written into the workspace — the
  operator's `~/.zcode` is never touched.
- **Reasoning effort for every backend**: on top of upstream's
  `reasoning_effort` chain (Codex, Claude Code, OpenCode), the DeepSeek
  Harness backend forwards the level through an `llm-deepseek` patch layer,
  and ZCode through its session store. Per-role, verbatim, no cross-backend
  mapping.
- **Bilingual control tokens**: the manager/auditor parsers accept route
  lines and control markers in both English and Russian.
- **A modern toolchain floor**: Python ≥ 3.14 (tomli fallbacks removed),
  Node.js ≥ 22 with CI provisioning 24; `make check` is the pre-push gate.
- **Honest CI**: `tests.yml` runs the Python suite on ubuntu/windows × 3.14
  plus a web job (frontend core suite + typecheck) on every push and pull
  request; `release.yml` builds and verifies distributions without publishing
  (the PyPI name belongs to upstream).

## Installing

The PyPI package `lh-harness` is upstream's release and does not contain the
work above — install from this repository:

```bash
pip install lhht            # or: uv tool install lhht
```

> **Naming note:** the PyPI distribution and the console command are `lhht`,
> while the importable Python module keeps the historical name — `import
> lh_harness` — and the per-project state lives in `.lh-harness/`. Nothing
> else about the tool is called lh-harness.

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

The tool installs as **`lhht`** (import module and the `.lh-harness/` state directory keep their
historical names). The harness is also used to develop itself: a repo-root `.lh-harness/config.toml`
defines the roles (manager/executor on `glm-5.3-flash`, auditor on `glm-5.3`,
effort `max`), and tasks run with `lhht run --task @task.md`. Agents edit
the working tree; the operator reviews, runs `make check`, and commits.

## Credits

The foundation is the AMAP-ML team's work — every contributor is listed on the
[contributors page](https://github.com/TON14/LongHorizon-Harness/graphs/contributors).
The fork's line is maintained by [TON14](https://github.com/TON14).
