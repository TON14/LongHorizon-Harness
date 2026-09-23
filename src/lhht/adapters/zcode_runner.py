from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from lhht.adapters.zcode_protocol import ProtocolError, run_episode


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


def run(
    binary: str,
    prompt_path: Path,
    model: str,
    *,
    mode: str = "yolo",
    thought_level: str = "high",
    workspace: str,
    server_args: Sequence[str] = ("app-server", "--stdio"),
    mcp_servers: list[dict] | None = None,
) -> int:
    """Bridge one episode to the ZCode Protocol app-server.

    ZCode 0.16.x headless ``-p`` runs cannot resolve a model ("Select a model
    before continuing"), so episodes go through ``zcode.cjs app-server
    --stdio`` instead: ``session/create`` pins the model and the role's
    reasoning level, ``session/send`` carries the task, and the transcript is
    polled until the assistant reply completes. The prompt travels inside the
    JSON body, so the Windows command-line length ceiling never applies.
    ``server_args`` replaces the app-server invocation; tests inject a fake
    protocol server there.
    """
    try:
        prompt = prompt_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        message = f"could not read ZCode prompt: {exc}"
        sys.stderr.write(message + "\n")
        _emit_result(is_error=True, exit_code=2, error=message)
        return 2

    from lhht.utils.agent_cli import zcode_spawn_command

    try:
        episode = run_episode(
            argv=zcode_spawn_command(binary, list(server_args)),
            workspace_path=workspace,
            workspace_key=workspace,
            provider_id="zai-direct",
            model_id=model,
            reasoning_level=thought_level,
            thought_level=thought_level,
            mode=mode,
            content=prompt,
            mcp_servers=mcp_servers,
        )
    except ProtocolError as exc:
        message = str(exc)
        sys.stderr.write(message + "\n")
        _emit_result(is_error=True, exit_code=1, error=message)
        return 1
    except OSError as exc:
        message = f"could not start ZCode app-server {binary!r}: {exc}"
        sys.stderr.write(message + "\n")
        _emit_result(is_error=True, exit_code=127, error=message)
        return 127

    _emit_result(
        text=episode.get("text", ""),
        is_error=False,
        exit_code=0,
        session_id=episode.get("session_id", ""),
        usage=episode.get("usage"),
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LongHorizon ZCode JSONL bridge")
    parser.add_argument("--binary", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", default="yolo")
    parser.add_argument("--thought-level", default="high")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--mcp-json", default=None,
                        help="JSON file with stdio MCP servers for session/create")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    mcp_servers = None
    if getattr(args, "mcp_json", None):
        try:
            loaded = json.loads(Path(args.mcp_json).read_text(encoding="utf-8"))
            if isinstance(loaded, list) and loaded:
                mcp_servers = loaded
        except (OSError, ValueError):
            mcp_servers = None  # a broken registration must never kill a run
    return run(
        args.binary,
        Path(args.prompt),
        args.model,
        mode=args.mode,
        thought_level=args.thought_level,
        workspace=args.workspace,
        mcp_servers=mcp_servers,
    )


if __name__ == "__main__":
    raise SystemExit(main())
