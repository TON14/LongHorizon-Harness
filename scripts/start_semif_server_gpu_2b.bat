@echo off
rem VRAM-safe variant: MiniCPM5-2B BF16 (~4.5 GB VRAM) instead of the 4B
rem (~8.2 GB). Validated 309/309 on the control-line calibration fixture.
cd /d D:\lhht
"D:\semif\.venv\Scripts\python.exe" src\lhht\scorer_server.py --port 8790 --backend torch --device cuda --dtype bfloat16 --model openbmb/MiniCPM5-2B --revision 12a3808a956f869c767195e9266b59c4d21d92e2
