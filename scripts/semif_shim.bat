@echo off
rem Forward a semif-score CLI invocation to the resident scoring server
rem (lhht server start / lhht.scorer_server). Point [run.semif] command at
rem this file. The interpreter comes from SEMIF_PY (the SemIf sidecar venv);
rem without it, plain `python` from PATH is used -- semif_client_cli.py is
rem stdlib-only, so any Python 3.10+ works.
setlocal
if defined SEMIF_PY (set "PY=%SEMIF_PY%") else set "PY=python"
"%PY%" "%~dp0semif_client_cli.py" %*
