"""Resident SemIf scoring server: load the model once, serve every lhht run.

All parallel lhht processes point their ``[run.semif] command`` at the shim
(``scripts/semif_shim.bat``), which forwards each semif-score CLI invocation to
this server over HTTP. The model is loaded exactly once per server lifetime.

Run under the SemIf sidecar venv:

    D:\\python\\git\\SemIf\\.venv\\Scripts\\python.exe scripts\\semif_server.py \
        --port 8790 --backend llamacpp \
        --gguf D:\\python\\git\\SemIf\\models\\Qwen_Qwen3.5-4B-Q4_K_M.gguf

GPU (single visible CUDA device, BF16 weights, ~9 GB VRAM for 4B):

    ... --backend torch --model Qwen/Qwen3.5-4B --revision <sha> --device cuda

API (localhost only):
    GET  /health            -> {"ok": true, "backend": ..., "scored": n}
    POST /score  {"rows": [decision rows]}  -> {"results": [result rows]}

Rows use SemIf's decision schema ({id, state, question, options}); results
carry {id, option_ids, probabilities, option_logits, forward_seconds}. One
row per request is the salvage case; batches are scored in order. Scoring is
serialized behind a lock (the model is not thread-safe); concurrent clients
queue.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from semif_phase1 import core
from semif_phase1.direct import score as direct_score

_STATE: dict = {"score_fn": None, "shared_fn": None, "backend": None,
                "scored": 0, "lock": threading.Lock()}


def _load(args: argparse.Namespace):
    if args.backend == "llamacpp":
        from semif_phase1 import llamacpp_backend as backend

        model, tokenizer, metadata = backend.load_model(
            args.model, args.revision, args.gguf, threads=args.threads
        )

        def score(row):
            return backend.score(model, tokenizer, row, metadata,
                                 max_tokens=args.max_tokens)
        shared_score = None
    elif args.backend == "torch":
        from semif_phase1.core import load_causal_model
        from semif_phase1.shared import score_shared

        model, tokenizer, metadata = load_causal_model(
            args.model, args.revision, device=args.device, dtype=args.dtype
        )

        def score(row):
            return direct_score(model, tokenizer, row, metadata,
                                max_tokens=args.max_tokens)

        def shared_score(rows):
            results, _timing = score_shared(
                model, tokenizer, rows, metadata, max_tokens=args.max_tokens
            )
            return results
    else:
        raise SystemExit(f"unsupported backend: {args.backend}")
    _STATE["score_fn"] = score
    _STATE["shared_fn"] = shared_score
    _STATE["backend"] = args.backend


def _score_rows(rows: list) -> list:
    """Score rows, routing consecutive same-state runs through shared prefixes.

    A battery over one big evidence state (the auditor-fast gate) then pays
    one prefill instead of re-encoding thousands of tokens per question.
    SemIf notes BF16 cache reuse can shift a few argmaxes (5-6/777 on their
    fixture), so any shared-path failure falls back to per-row direct scoring
    and short mixed batches stay direct.
    """
    score = _STATE["score_fn"]
    shared = _STATE["shared_fn"]
    results: list = []
    index = 0
    while index < len(rows):
        end = index
        while end < len(rows) and rows[end]["state"] == rows[index]["state"]:
            end += 1
        group = rows[index:end]
        if len(group) > 1 and shared is not None:
            try:
                results.extend(shared(group))
                index = end
                continue
            except Exception:
                pass  # fall back to per-row direct below
        results.extend(score(row) for row in group)
        index = end
    return results


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *vals):  # one quiet line per request
        print(f"[{time.strftime('%H:%M:%S')}] {self.address_string()} {fmt % vals}",
              flush=True)

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"ok": _STATE["score_fn"] is not None,
                             "backend": _STATE["backend"],
                             "scored": _STATE["scored"]})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/score":
            self._json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            rows = payload.get("rows")
            if not isinstance(rows, list) or not rows:
                raise ValueError("rows must be a non-empty list")
            results = []
            with _STATE["lock"]:
                for row in rows:
                    core.validate_row(row)
                results = _score_rows(rows)
                _STATE["scored"] += len(results)
            self._json(200, {"results": results})
        except Exception as exc:  # strict SemIf errors become 400s, not crashes
            self._json(400, {"error": f"{type(exc).__name__}: {exc}"})


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8790)
    p.add_argument("--backend", choices=("llamacpp", "torch"), default="llamacpp")
    p.add_argument("--model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--revision",
                   default="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
    p.add_argument("--gguf",
                   default=r"D:\semif\models\Qwen_Qwen3.5-4B-Q4_K_M.gguf")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--max-tokens", type=int, default=4096)
    args = p.parse_args()

    print(f"loading model (backend={args.backend}) ...", flush=True)
    t0 = time.perf_counter()
    _load(args)
    print(f"model ready in {time.perf_counter()-t0:.1f}s; "
          f"serving on http://{args.host}:{args.port}", flush=True)

    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
