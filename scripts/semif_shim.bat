@echo off
rem Forward a semif-score CLI invocation to the resident scoring server
rem (scripts/semif_server.py). Point [run.semif] command at this file.
"D:\semif\.venv\Scripts\python.exe" "D:\lhht\scripts\semif_client_cli.py" %*
