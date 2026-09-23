"""`lhht server` — explicit operator control of the resident scorer server.

The harness NEVER starts or stops the server on its own: every scorer
feature only checks whether a healthy server answers on the configured
port and silently degrades when it does not. Starting one is an operator
decision, made here.

    lhht server status            # is a server answering /health?
    lhht server start [--port N] [--cpu] [--model M] [--gguf P]
    lhht server stop [--port N]

`start` refuses to double-start (a healthy server wins), spawns the server
detached (it survives this process), waits for /health, and reports. The
server module ships inside the package (``lhht/scorer_server.py``) and runs
under the SemIf sidecar interpreter from ``[run.semif] mcp_python`` — the
only machine-specific path the command needs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from .config import load_run_defaults

DEFAULT_PORT = 8790
_START_TIMEOUT_SECONDS = 120.0


def _health(port: int, timeout: float = 2.0) -> dict | None:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/health", timeout=timeout
        ) as response:
            payload = json.loads(response.read())
            return payload if payload.get("ok") else None
    except Exception:
        return None


def _server_module_path() -> Path:
    import lhht

    return Path(lhht.__file__).resolve().parent / "scorer_server.py"


def _status_command(args: argparse.Namespace) -> int:
    health = _health(args.port)
    if health is None:
        print(f"scorer server: NOT reachable on 127.0.0.1:{args.port} "
              "(scorer features stay off in runs)")
        return 1
    print(f"scorer server: reachable on 127.0.0.1:{args.port} "
          f"(backend={health.get('backend')}, scored={health.get('scored')})")
    return 0


def _start_command(args: argparse.Namespace) -> int:
    if _health(args.port) is not None:
        print(f"scorer server: already running on 127.0.0.1:{args.port}")
        return 0
    defaults = load_run_defaults()
    python_path = defaults.get("semif_mcp_python")
    module_path = _server_module_path()
    if not (isinstance(python_path, str) and python_path and module_path.is_file()):
        print("cannot start: set [run.semif] mcp_python to the SemIf sidecar "
              "python (the interpreter that has semif_phase1 installed)",
              file=sys.stderr)
        return 2
    argv = [python_path, str(module_path), "--port", str(args.port)]
    if args.cpu:
        argv += ["--backend", "llamacpp"]
    else:
        argv += ["--backend", "torch", "--device", "cuda", "--dtype", "bfloat16"]
    if args.model:
        argv += ["--model", args.model]
    if args.revision:
        argv += ["--revision", args.revision]
    if args.gguf:
        argv += ["--backend", "llamacpp", "--gguf", args.gguf]
    log_path = Path(f"scorer-server-{args.port}.log")
    creationflags = 0
    if sys.platform == "win32":
        creationflags = 0x00000008 | 0x00000200  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    with log_path.open("ab") as log:
        subprocess.Popen(
            argv, stdout=log, stderr=log, stdin=subprocess.DEVNULL,
            creationflags=creationflags, close_fds=True,
        )
    print(f"scorer server: starting ({'CPU' if args.cpu else 'GPU'}) "
          f"on 127.0.0.1:{args.port}, log: {log_path}")
    deadline = time.monotonic() + (args.wait or _START_TIMEOUT_SECONDS)
    while time.monotonic() < deadline:
        health = _health(args.port, timeout=2.0)
        if health is not None:
            print(f"scorer server: ready (backend={health.get('backend')})")
            return 0
        time.sleep(1.0)
    print("scorer server: did not become healthy in time — check the log",
          file=sys.stderr)
    return 1


def _stop_command(args: argparse.Namespace) -> int:
    if _health(args.port) is None:
        print(f"scorer server: nothing to stop on 127.0.0.1:{args.port}")
        return 0
    request = urllib.request.Request(
        f"http://127.0.0.1:{args.port}/shutdown", data=b"{}", method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            json.loads(response.read())
    except Exception as exc:
        print(f"scorer server: stop request failed: {exc}", file=sys.stderr)
        return 1
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        if _health(args.port, timeout=1.0) is None:
            print("scorer server: stopped")
            return 0
        time.sleep(0.5)
    print("scorer server: stop accepted but still answering", file=sys.stderr)
    return 1


def _probe(interpreter: str, code: str, timeout: float = 15.0) -> str | None:
    """Run a short python probe under the sidecar interpreter."""
    try:
        completed = subprocess.run(
            [interpreter, "-c", code],
            capture_output=True, text=True, timeout=timeout, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def _doctor_command(args: argparse.Namespace) -> int:
    from pathlib import Path as _P

    failures = 0
    try:
        defaults = load_run_defaults()
        configured = True
    except Exception:
        defaults, configured = {}, False

    def line(mark: str, label: str, detail: str = "") -> None:
        nonlocal failures
        if mark == "FAIL":
            failures += 1
        suffix = f" {detail}" if detail else ""
        print(f"[{mark:4s}] {label}{suffix}")

    line("OK" if configured else "FAIL", "Project config",
         "parses" if configured else "cannot read .lhht/config.toml")
    if not configured:
        return 1

    python_path = defaults.get("semif_mcp_python")
    if isinstance(python_path, str) and python_path and _P(python_path).is_file():
        version = _probe(python_path, "import sys; print(sys.version.split()[0])")
        line("OK", "Sidecar interpreter", f"{version or 'runs'} {python_path}")
        semif = _probe(
            python_path,
            "import semif_phase1; print(getattr(semif_phase1, '__version__', 'ok'))",
        )
        line("OK" if semif else "FAIL", "SemIf package (semif_phase1)",
             semif or "not importable under the sidecar interpreter")
        cuda = _probe(
            python_path,
            "import torch; print('cuda' if torch.cuda.is_available() else 'cpu-only')",
        )
        if cuda == "cuda":
            line("OK", "Torch backend", "CUDA available")
        elif cuda == "cpu-only":
            line("WARN", "Torch backend",
                 "installed but CPU-only (GPU start needs the cu128 wheel)")
        else:
            line("WARN", "Torch backend", "torch not importable (CPU/GPU scoring off)")
    else:
        line("FAIL", "Sidecar interpreter",
             "[run.semif] mcp_python missing or not a file")

    for label, key in (("Shim (semif-score CLI)", "command"),
                       ("MCP tool script", "mcp_script")):
        value = defaults.get(f"semif_{key}")
        ok = isinstance(value, str) and value and _P(value).is_file()
        line("OK" if ok else ("WARN" if not value else "FAIL"), label,
             value if ok else "not configured" if not value else "path missing")

    gguf = defaults.get("semif_gguf")
    if gguf:
        line("OK" if _P(gguf).is_file() else "FAIL", "GGUF checkpoint", gguf)
    model = defaults.get("semif_model")
    line("OK" if model else "WARN", "Scorer model",
         f"{model or 'not configured'}")

    module_path = _server_module_path()
    line("OK" if module_path.is_file() else "FAIL", "Server module",
         str(module_path))

    health = _health(args.port)
    if health is None:
        line("WARN", "Server", f"not running on 127.0.0.1:{args.port} "
                               "(scorer features off; `lhht server start`)")
    else:
        line("OK", "Server",
             f"backend={health.get('backend')} model={health.get('model')} "
             f"scored={health.get('scored')}")

    print(f"Doctor result: {'ready' if failures == 0 else f'{failures} failure(s)'}")
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lhht server",
        description="Explicit start/stop/status for the resident scorer server. "
                    "Runs never touch it: they only use a healthy server or "
                    "silently run without scorer features.",
    )
    sub = parser.add_subparsers(dest="action", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--port", type=int, default=DEFAULT_PORT)

    status = sub.add_parser("status", help="is a scorer server answering?")
    common(status)
    status.set_defaults(func=_status_command)

    start = sub.add_parser("start", help="start the server (refuses if running)")
    common(start)
    start.add_argument("--cpu", action="store_true",
                       help="llamacpp CPU backend instead of GPU torch")
    start.add_argument("--model", default=None)
    start.add_argument("--revision", default=None)
    start.add_argument("--gguf", default=None, help="GGUF checkpoint for --cpu")
    start.add_argument("--wait", type=float, default=None,
                       help="seconds to wait for health (default 120)")
    start.set_defaults(func=_start_command)

    stop = sub.add_parser("stop", help="ask a running server to shut down")
    common(stop)
    stop.set_defaults(func=_stop_command)

    doctor = sub.add_parser(
        "doctor", help="check the scorer stack: SemIf, interpreter, files, server"
    )
    common(doctor)
    doctor.set_defaults(func=_doctor_command)
    return parser


def server_command(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
