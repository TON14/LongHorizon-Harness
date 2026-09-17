# Developer entry points for LongHorizon-Harness.
#
# Until now these commands lived only in .github/workflows/release.yml, where a
# contributor had to read a release job to find them. Every recipe below is one
# command deep on purpose: it names the command rather than replacing it, so a
# shell without `make` (Windows, a minimal image) can run the same thing by
# hand, and release.yml stays the authority for what a release does. Recipes
# are echoed for the same reason: the Makefile teaches the command it runs.
# The require-* prerequisites add no work of their own; they only check that
# the toolchain can run that command, and say what to do when it cannot.
#
# Override the interpreters if yours are named differently:
#   make test PYTHON=python NPM=pnpm

# Prefer the checkout's uv-managed venv when present; override with PYTHON=...
PYTHON      ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
NPM         ?= npm
PYTEST_ARGS ?=
WEB         := frontend/web

# Minimum toolchain versions, checked before anything installs or runs.
# PYTHON_MIN tracks pyproject's requires-python; NODE_MIN tracks the Node
# version .github/workflows/release.yml provisions to build the Web bundle.
PYTHON_MIN  := 3.14
NODE_MIN    := 24

.DEFAULT_GOAL := help
.PHONY: help install install-py install-web check test test-py test-web \
        typecheck build-web dist dev-api dev-web clean \
        require-python require-node require-py-deps require-web-deps

help: ## List the available targets
	@echo 'Usage: make <target> [PYTHON=... NPM=... PYTEST_ARGS=...]'
	@grep -hE '^[a-zA-Z][a-zA-Z0-9_-]*:.*## ' $(MAKEFILE_LIST) \
		| sed -e 's/:[^#]*## /|/' \
		| awk -F'|' '{ printf "  %-13s %s\n", $$1, $$2 }'

install: install-py install-web ## Install everything needed to develop and test

install-py: require-python ## Install the package in editable mode with its test extras
	$(PYTHON) -m pip install -e '.[test]'

install-web: require-node ## Install the Web app dependencies from package-lock.json
	$(NPM) ci --prefix $(WEB) --no-audit --no-fund

check: test typecheck ## Run everything expected to be green before pushing

test: test-py test-web ## Run the Python and frontend suites

test-py: require-py-deps ## Run the Python suite (PYTEST_ARGS="-k webapi" narrows it)
	$(PYTHON) -m pytest -q $(PYTEST_ARGS)

# frontend/core has no lockfile of its own; its tests run through the Web
# app's npm scripts, which own the toolchain both packages share.
test-web: require-web-deps ## Run the shared frontend/core suite
	$(NPM) test --prefix $(WEB)

typecheck: require-web-deps ## Type-check the Web app without emitting output
	$(NPM) run typecheck --prefix $(WEB)

# The bundle is a build artifact, not source: the directory is git-ignored and
# CI rebuilds it before packaging, so the wheel has one source of truth for it.
# Never commit it. One webapi test skips itself until this has run.
build-web: require-web-deps ## Build the Web bundle into src/lh_harness/_frontend/web/dist
	$(NPM) run build --prefix $(WEB)

dist: build-web ## Build the sdist and wheel
	@$(PYTHON) -c 'import build' >/dev/null 2>&1 \
		|| { echo 'The build front-end is missing. Run: $(PYTHON) -m pip install build'; exit 1; }
	$(PYTHON) -m build

dev-api: require-py-deps ## Serve the control API on 127.0.0.1:8799
	$(PYTHON) -m lh_harness web --no-open

# Run alongside dev-api in a second shell. Set LH_HARNESS_WEB_API to proxy to a
# control API running somewhere other than 127.0.0.1:8799.
dev-web: require-web-deps ## Serve the Vite dev server on :5173, proxying /api to dev-api
	$(NPM) run dev --prefix $(WEB)

clean: ## Remove build artifacts and caches (rebuild the bundle with build-web)
	rm -rf dist build src/lh_harness/_frontend .pytest_cache
	find src tests -name '__pycache__' -type d -prune -exec rm -rf {} +
	find . -maxdepth 2 -name '*.egg-info' -type d -prune -exec rm -rf {} +

# An unusable environment should name its own fix rather than fail somewhere
# inside the tool that was supposed to run.
require-python:
	@$(PYTHON) -c 'import sys; raise SystemExit(sys.version_info < tuple(int(part) for part in "$(PYTHON_MIN)".split(".")))' 2>/dev/null \
		|| { echo 'Python $(PYTHON_MIN) or later is required; $(PYTHON) is '"$$($(PYTHON) -V 2>/dev/null || echo 'not on PATH')"'. Override with PYTHON=/path/to/python.'; exit 1; }

require-node:
	@node -e 'process.exit(+process.versions.node.split(".")[0] >= $(NODE_MIN) ? 0 : 1)' 2>/dev/null \
		|| { echo 'Node.js $(NODE_MIN) or later is required; found '"$$(node -v 2>/dev/null || echo 'no node on PATH')"'.'; exit 1; }

require-py-deps: require-python
	@$(PYTHON) -c 'import lh_harness, pytest' >/dev/null 2>&1 \
		|| { echo 'The editable install with test extras is missing. Run: make install-py'; exit 1; }

require-web-deps: require-node
	@test -d $(WEB)/node_modules \
		|| { echo 'The Web dependencies are missing. Run: make install-web'; exit 1; }
