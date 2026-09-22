"""semif-score CLI shim: forward one CLI invocation to the resident server.

Argv-compatible with the subset of ``semif-score`` flags lhht's
``SemifCliScorer`` emits, so ``[run.semif] command`` can point at the shim
instead of the real CLI and every parallel lhht run shares one loaded model:

    command = "D:\\python\\git\\LongHorizon-Harness\\scripts\\semif_shim.bat"

Model/backend flags are accepted and ignored (the server owns the model).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

DEFAULT_URL = "http://127.0.0.1:8790"


def main() -> int:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--mode")
    p.add_argument("--backend")
    p.add_argument("--model")
    p.add_argument("--revision")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--gguf")
    p.add_argument("--max-tokens", type=int)
    p.add_argument("--device")
    p.add_argument("--dtype")
    p.add_argument("--llama-threads", type=int)
    p.add_argument("--url", default=DEFAULT_URL)
    args, _unknown = p.parse_known_args()

    rows = [json.loads(line) for line in
            Path(args.input).read_text(encoding="utf-8").splitlines() if line.strip()]
    body = json.dumps({"rows": rows}).encode("utf-8")
    request = urllib.request.Request(
        f"{args.url}/score", data=body,
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=120) as response:
        results = json.loads(response.read())["results"]

    out = Path(args.output)
    if out.exists():  # SemIf refuses to overwrite; keep that contract
        print(f"refusing to overwrite existing output: {out}", file=sys.stderr)
        return 1
    with out.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    started = time.perf_counter()
    code = main()
    print(f"shim: exit={code} in {time.perf_counter()-started:.2f}s",
          file=sys.stderr)
    sys.exit(code)
