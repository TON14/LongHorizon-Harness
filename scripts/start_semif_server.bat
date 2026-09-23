@echo off
rem Start the resident SemIf scoring server (CPU, llamacpp) in its own window.
rem All parallel lhht runs share this one loaded model via scripts/semif_shim.bat.
cd /d D:\lhht
"D:\semif\.venv\Scripts\python.exe" src\lhht\scorer_server.py --port 8790 --backend llamacpp
