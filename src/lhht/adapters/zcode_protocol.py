"""Client for the ZCode Protocol stdio app-server (zcode.cjs app-server).

ZCode 0.16.x removed every headless path this harness relied on: the
``-p/--prompt`` runner resolves no model selection at all ("Select a model
before continuing"), and the ``ZCODE_MODEL`` environment variable is gone.
The one supported entry point left is the same one the desktop app drives:
``zcode.cjs app-server --stdio``, a newline-delimited JSON protocol whose
``session/create`` takes the model (with an explicit reasoning level) and a
thought level per session. That maps one-to-one onto the harness roles, so
this module speaks it directly.

Wire notes learned the hard way, kept here so nobody re-derives them from a
minified bundle:

- Messages are JSON objects per line, but *not* JSON-RPC: requests are
  ``{"id", "method", "params"}`` and the ``jsonrpc`` key is rejected.
- The server also sends *requests* to the client and blocks on the answers.
  ``session/requestRuntimePreferences`` must be answered with
  ``{"nativeSearchEnhancementsEnabled": bool}``;
  ``interaction/requestOfficialMcpAuthHeaders`` with ``{"headers": {...}}``.
- A model only exists if it is registered in the personal provider config
  (see ``zcode_provider_config``). The registry otherwise answers
  ``Provider Registry 中不存在 Model``.
- The turn runs asynchronously after ``session/send``; poll
  ``session/messages`` until the last assistant message reports
  ``finish: "completed"`` and read the ``text`` parts.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time


class ProtocolError(RuntimeError):
    pass


class _Client:
    def __init__(self, process: subprocess.Popen) -> None:
        self._process = process
        self._lines: list[dict] = []
        self._lock = threading.Lock()
        self._answered: set[tuple[int, object]] = set()
        self._stop = threading.Event()
        self._threads = [
            threading.Thread(target=self._pump, args=(process.stdout,), daemon=True),
            threading.Thread(target=self._pump, args=(process.stderr,), daemon=True),
        ]
        for thread in self._threads:
            thread.start()
        self._responder = threading.Thread(target=self._autorespond, daemon=True)
        self._responder.start()

    # -- plumbing ---------------------------------------------------------

    def _pump(self, stream) -> None:
        for raw in stream:
            line = raw.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except ValueError:
                continue
            if isinstance(message, dict):
                with self._lock:
                    self._lines.append(message)

    def _autorespond(self) -> None:
        """Answer the server's client-bound requests.

        ``session/requestRuntimePreferences`` and
        ``interaction/requestOfficialMcpAuthHeaders`` gate session creation.
        Tool permission requests are granted: executors run in yolo mode and
        the auditors/manager are read-only by the harness's own design.
        """
        while not self._stop.is_set():
            with self._lock:
                pending = list(self._lines)
            for index, message in enumerate(pending):
                key = ("ask", index, message.get("id"))
                if key in self._answered:
                    continue
                method = message.get("method")
                if not method or "id" not in message:
                    continue
                self._answered.add(key)
                if method == "session/requestRuntimePreferences":
                    result = {"nativeSearchEnhancementsEnabled": False}
                elif method == "interaction/requestOfficialMcpAuthHeaders":
                    result = {"headers": {}}
                elif "permission" in method.lower():
                    result = {"decision": "allow"}
                else:
                    result = {}
                self.send({"id": message["id"], "result": result})
            time.sleep(0.2)

    def send(self, message: dict) -> None:
        self._process.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
        self._process.stdin.flush()

    def request(self, method: str, params: dict, *, timeout: float = 60.0) -> dict:
        request_id = int(time.time() * 1000) % 1000000
        self.send({"id": request_id, "method": method, "params": params})
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._process.poll() is not None:
                raise ProtocolError(f"zcode app-server exited with {self._process.returncode}")
            with self._lock:
                for message in self._lines:
                    if message.get("id") == request_id and ("result" in message or "error" in message):
                        if "error" in message:
                            error = message["error"]
                            detail = error.get("message", "") if isinstance(error, dict) else str(error)
                            raise ProtocolError(f"{method} failed: {detail[:400]}")
                        return message.get("result", {})
            time.sleep(0.3)
        raise ProtocolError(f"{method} timed out after {timeout:.0f}s")

    # -- protocol ---------------------------------------------------------

    def create_session(
        self,
        *,
        workspace_path: str,
        workspace_key: str,
        provider_id: str,
        model_id: str,
        reasoning_level: str,
        thought_level: str,
        mode: str,
        timeout: float = 90.0,
    ) -> str:
        params = {
            "workspace": {
                "workspacePath": workspace_path,
                "workspaceKey": workspace_key,
            },
            "model": {
                "providerId": provider_id,
                "modelId": model_id,
                "options": {"reasoningLevel": reasoning_level},
            },
            "thoughtLevel": thought_level,
            "mode": mode,
        }
        result = self.request("session/create", params, timeout=timeout)
        session = (result or {}).get("session", {})
        session_id = session.get("sessionId")
        if not session_id or session_id == "unknown":
            raise ProtocolError(f"session/create returned no session id: {str(result)[:200]}")
        return session_id

    def send_prompt(self, session_id: str, content: str) -> None:
        result = self.request(
            "session/send", {"sessionId": session_id, "content": content}, timeout=120.0
        )
        if not result.get("accepted", False):
            raise ProtocolError(f"session/send was not accepted: {str(result)[:200]}")

    def messages(self, session_id: str) -> list[dict]:
        result = self.request("session/messages", {"sessionId": session_id}, timeout=60.0)
        return (result or {}).get("messages", [])

    def wait_for_assistant_reply(
        self, session_id: str, *, timeout: float = 600.0, poll: float = 2.0
    ) -> tuple[str, dict]:
        """Poll the transcript until the turn's final answer is complete.

        A finished assistant message carries ``finish`` ("stop" for a normal
        model stop, "completed" on timeline scaffolding); a streaming one has
        ``finish: null``. The turn is done when the transcript ends with a
        finished assistant message that has stayed unchanged across a poll —
        the extra beat keeps tool-using turns from being cut off after their
        first intermediate step.
        """
        deadline = time.time() + timeout
        last_error = ""
        stable_count = 0
        last_snapshot = None
        while time.time() < deadline:
            try:
                messages = self.messages(session_id)
            except ProtocolError as exc:
                last_error = str(exc)
                messages = []

            last = messages[-1] if messages else None
            info = (last or {}).get("info", {})
            finished = info.get("role") == "assistant" and bool(info.get("finish"))
            if info.get("finish") in ("error", "aborted", "cancelled"):
                raise ProtocolError(f"turn ended with finish={info.get('finish')}: {last_error}")
            if finished:
                text = "".join(
                    part.get("text", "")
                    for part in last.get("parts", [])
                    if part.get("type") == "text"
                )
                snapshot = json.dumps(last, ensure_ascii=False, sort_keys=True)
                stable_count = stable_count + 1 if snapshot == last_snapshot else 1
                last_snapshot = snapshot
                if stable_count >= 2:
                    usage = {
                        "tokens": info.get("tokens", {}),
                        "modelId": info.get("modelId"),
                    }
                    return text, usage
            else:
                stable_count = 0
            time.sleep(poll)
        raise ProtocolError(
            f"no completed assistant reply within {timeout:.0f}s"
            + (f" (last: {last_error})" if last_error else "")
        )

    def close(self) -> None:
        self._stop.set()
        try:
            self._process.kill()
        except OSError:
            pass


def run_episode(
    *,
    argv: list[str],
    workspace_path: str,
    workspace_key: str,
    provider_id: str,
    model_id: str,
    reasoning_level: str,
    thought_level: str,
    mode: str,
    content: str,
    timeout: float = 600.0,
) -> dict:
    """Run one prompt through the protocol and return text/session/usage.

    ``argv`` is the full command line to launch the app-server, already
    wrapped for the platform (Node script bundles need ``node`` in front).
    """
    process = subprocess.Popen(
        argv,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    client = _Client(process)
    try:
        session_id = client.create_session(
            workspace_path=workspace_path,
            workspace_key=workspace_key,
            provider_id=provider_id,
            model_id=model_id,
            reasoning_level=reasoning_level,
            thought_level=thought_level,
            mode=mode,
        )
        client.send_prompt(session_id, content)
        text, usage = client.wait_for_assistant_reply(session_id, timeout=timeout)
        return {"text": text, "session_id": session_id, "usage": usage}
    finally:
        client.close()
