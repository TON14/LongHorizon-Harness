@echo off
rem Start the resident SemIf scoring server on the GPU (torch + CUDA, BF16).
rem Needs ~9 GB free VRAM for the 4B BF16 model. First start also downloads
rem the HF weights (~8 GB) into the huggingface cache.
cd /d D:\lhht
"D:\semif\.venv\Scripts\python.exe" src\lhht\scorer_server.py --port 8790 --backend torch --device cuda --dtype bfloat16
