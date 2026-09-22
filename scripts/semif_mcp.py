"""MCP server exposing the resident SemIf scorer as a tool for agent roles.

Stdlib-only JSON-RPC over stdio (Model Context Protocol, 2025-06-18 shape:
initialize / notifications/initialized / ping / tools/list / tools/call).
The scoring itself is delegated to the resident HTTP server
(scripts/semif_server.py, default http://127.0.0.1:8790), so the model is
loaded once and every ZCode session in every parallel lhht run shares it.

Register from a workspace .mcp.json (or the user-level MCP config):

    {
      "mcpServers": {
        "semif-scorer": {
          "command": "D:\\python\\git\\SemIf\\.venv\\Scripts\\python.exe",
          "args": ["D:\\python\\git\\LongHorizon-Harness\\scripts\\semif_mcp.py"]
        }
      }
    }

Tool contract (score):
    state     str                     the text to judge (a line, a report, a claim)
    question  str                     one narrow question about the state
    options   [{id, description}]*    2-16 typed choices
    -> content text: JSON {option_id: probability, ...} ordered by probability
Any error (server down, bad args) returns an MCP error result — the agent
sees a clean failure, never a hang.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request

SCORER_URL = os.environ.get("SEMIF_SCORER_URL", "http://127.0.0.1:8790")
TOOL_SCHEMA = {
    "name": "score",
    "description": (
        "Fast local semantic classifier (no text generation): given a state "
        "text, one narrow question, and 2-16 typed options, returns the "
        "probability of each option. Answers arrive in ~0.1-1 s. Use it for "
        "cheap yes/no or menu decisions instead of reasoning them out, e.g. "
        "'does this file content match the requirement?', 'which of these "
        "categories does this line belong to?'."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "state": {"type": "string", "description": "The text to judge."},
            "question": {"type": "string", "description": "One narrow question about the state."},
            "options": {
                "type": "array",
                "minItems": 2,
                "maxItems": 16,
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "description": {"type": "string"},
                    },
                    "required": ["id"],
                },
            },
        },
        "required": ["state", "question", "options"],
    },
}


def _score(state: str, question: str, options: list) -> dict:
    row = {"id": "mcp", "state": state, "question": question,
           "options": [{"id": o.get("id", ""), "description": o.get("description", o.get("id", ""))}
                       for o in options]}
    request = urllib.request.Request(
        f"{SCORER_URL}/score",
        data=json.dumps({"rows": [row]}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        result = json.loads(response.read())["results"][0]
    pairs = sorted(zip(result["option_ids"], result["probabilities"]),
                   key=lambda p: -p[1])
    return {option_id: round(prob, 4) for option_id, prob in pairs}


def dispatch(message: dict) -> dict | None:
    method = message.get("method", "")
    msg_id = message.get("id")
    if method == "initialize":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {
            "protocolVersion": "2025-06-18",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "semif-scorer", "version": "1.0.0"},
        }}
    if method == "ping":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": msg_id,
                "result": {"tools": [TOOL_SCHEMA]}}
    if method == "tools/call":
        params = message.get("params") or {}
        if params.get("name") != "score":
            return {"jsonrpc": "2.0", "id": msg_id, "error": {
                "code": -32602, "message": f"unknown tool: {params.get('name')}"}}
        try:
            args = params.get("arguments") or {}
            outcome = _score(args["state"], args["question"], args["options"])
            return {"jsonrpc": "2.0", "id": msg_id, "result": {
                "content": [{"type": "text", "text": json.dumps(outcome)}],
                "isError": False,
            }}
        except Exception as exc:
            return {"jsonrpc": "2.0", "id": msg_id, "result": {
                "content": [{"type": "text",
                             "text": f"scorer unavailable: {type(exc).__name__}: {exc}"}],
                "isError": True,
            }}
    if msg_id is not None:  # unknown request; notifications pass silently
        return {"jsonrpc": "2.0", "id": msg_id, "error": {
            "code": -32601, "message": f"method not found: {method}"}}
    return None


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        response = dispatch(message)
        if response is not None:
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
