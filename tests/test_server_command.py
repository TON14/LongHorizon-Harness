"""`lhht server` subcommand: explicit operator control, never auto-start."""

from __future__ import annotations

import http.server
import threading

import pytest

from lhht import server_command


class _TinyServer:
    """Minimal /health + /shutdown responder for command tests."""

    def __init__(self, port: int) -> None:
        class H(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path == "/health":
                    body = b'{"ok": true, "backend": "fake", "scored": 0}'
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_response(404)
                    self.end_headers()

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), H)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self) -> "_TinyServer":
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def test_status_reports_reachable(capsys):
    with _TinyServer(18790):
        assert server_command.server_command(["status", "--port", "18790"]) == 0
    out = capsys.readouterr().out
    assert "reachable" in out and "fake" in out


def test_status_reports_absent(capsys):
    assert server_command.server_command(["status", "--port", "18791"]) == 1
    assert "NOT reachable" in capsys.readouterr().out


def test_start_refuses_when_already_running(capsys):
    with _TinyServer(18792):
        assert server_command.server_command(["start", "--port", "18792"]) == 0
    assert "already running" in capsys.readouterr().out


def test_start_requires_configured_interpreter(capsys, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # no project config -> no mcp_python
    rc = server_command.server_command(
        ["start", "--port", "18793", "--wait", "0.1"]
    )
    assert rc == 2
    assert "mcp_python" in capsys.readouterr().err


def test_stop_is_idempotent(capsys):
    assert server_command.server_command(["stop", "--port", "18794"]) == 0
    assert "nothing to stop" in capsys.readouterr().out


def test_doctor_reports_absent_server_and_config(capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rc = server_command.server_command(["doctor", "--port", "18795"])
    out = capsys.readouterr().out
    assert "Project config" in out
    assert "not running" in out
