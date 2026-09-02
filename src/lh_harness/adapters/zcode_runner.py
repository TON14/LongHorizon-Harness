from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence


def _emit_result(
    *,
    text: str = "",
    is_error: bool,
    exit_code: int,
    error: str = "",
    session_id: str = "",
    usage: dict | None = None,
) -> None:
    record: dict[str, object] = {
        "type": "zcode.result",
        "text": text,
        "is_error": is_error,
        "exit_code": exit_code,
    }
    if error:
        record["error"] = error
    if session_id:
        record["session_id"] = session_id
    if usage:
        record["usage"] = usage
    sys.stdout.write(json.dumps(record, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _final_response(stdout: str) -> tuple[str, str, dict | None]:
    """Read the assistant reply out of ``--json`` output.

    A successful headless run prints one JSON object whose ``response`` field
    is the final answer; the CLI pretty-prints it across many lines. Anything
    unparseable keeps its raw stdout as the episode text, so evidence is never
    lost to a formatting change.
    """
    for candidate in (stdout, _outermost_json(stdout)):
        if not candidate.lstrip().startswith("{"):
            continue
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        response = payload.get("response")
        session_id = payload.get("sessionId")
        usage = payload.get("usage")
        return (
            response if isinstance(response, str) else "",
            session_id if isinstance(session_id, str) else "",
            usage if isinstance(usage, dict) else None,
        )
    return (stdout.strip(), "", None)


def _outermost_json(stdout: str) -> str:
    start = stdout.find("{")
    end = stdout.rfind("}")
    if 0 <= start < end:
        return stdout[start : end + 1]
    return ""


def run(binary: str, prompt_path: Path, model: str, *, mode: str = "yolo") -> int:
    """Bridge one episode to the ZCode headless CLI.

    The shared adapter plumbing delivers the prompt as a file and over stdin;
    ZCode only accepts a headless task as a ``-p`` argument, so this bridge
    reads the file and re-launches the CLI with the task in argv -- the same
    contract ``deepseek_runner`` has with ``dsh``.
    """
    try:
        prompt = prompt_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        message = f"could not read ZCode prompt: {exc}"
        sys.stderr.write(message + "\n")
        _emit_result(is_error=True, exit_code=2, error=message)
        return 2

    command = [binary, "--json", "--mode", mode, "-p", prompt]
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        message = f"could not start ZCode binary {binary!r}: {exc}"
        sys.stderr.write(message + "\n")
        _emit_result(is_error=True, exit_code=127, error=message)
        return 127

    stdout = completed.stdout or ""
    if completed.returncode == 0:
        text, session_id, usage = _final_response(stdout)
        _emit_result(
            text=text,
            is_error=False,
            exit_code=0,
            session_id=session_id,
            usage=usage,
        )
    else:
        # `--json` is only promised for successful runs; on failure the CLI
        # writes a human-readable error, which stderr already carries.
        _emit_result(
            text=stdout.strip(),
            is_error=True,
            exit_code=completed.returncode,
            error=(completed.stderr or "").strip()[-2000:],
        )
    return completed.returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LongHorizon ZCode JSONL bridge")
    parser.add_argument("--binary", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", default="yolo")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args.binary, Path(args.prompt), args.model, mode=args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
