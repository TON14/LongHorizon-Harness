"""The resident-server HTTP scorer: same contract as the CLI scorer.

One POST of one row per decision; every failure mode must return None so
callers degrade exactly like they do when the CLI scorer is down.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from lhht.semantic_salvage import (
    DEFAULT_TIMEOUT_SECONDS,
    SemifCliScorer,
    SemifHttpScorer,
    scorer_from_config,
)

_OPTIONS = [
    {"id": "yes", "description": "true"},
    {"id": "no", "description": "false"},
]


class _Responder(BaseHTTPRequestHandler):
    mode = "ok"

    def do_POST(self):  # noqa: N802 - http.server API
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length)
        if self.mode == "garbage":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"not json")
            return
        rows = json.loads(body)["rows"]
        self.send_response(200)
        self.end_headers()
        results = [
            {
                "id": row["id"],
                "option_ids": [option["id"] for option in row["options"]],
                "probabilities": [0.9, 0.1],
            }
            for row in rows
        ]
        self.wfile.write(json.dumps({"results": results}).encode("utf-8"))

    def log_message(self, *args):  # silence the test output
        pass


@pytest.fixture()
def server_url():
    httpd = HTTPServer(("127.0.0.1", 0), _Responder)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()
    httpd.server_close()


def test_scores_one_row_over_http(server_url) -> None:
    scorer = SemifHttpScorer(server_url)

    assert scorer.score("The sky is blue.", "Is it true?", _OPTIONS) == [0.9, 0.1]


def test_server_down_returns_none() -> None:
    scorer = SemifHttpScorer("http://127.0.0.1:1", timeout_seconds=2)

    assert scorer.score("state", "question?", _OPTIONS) is None


def test_malformed_response_returns_none(server_url, monkeypatch) -> None:
    monkeypatch.setattr(_Responder, "mode", "garbage")
    scorer = SemifHttpScorer(server_url)

    assert scorer.score("state", "question?", _OPTIONS) is None


def test_invalid_rows_return_none_without_a_request(server_url) -> None:
    scorer = SemifHttpScorer(server_url)

    assert scorer.score("", "question?", _OPTIONS) is None
    assert scorer.score("state", "", _OPTIONS) is None
    assert scorer.score("state", "question?", [{"id": "only"}]) is None


def test_factory_prefers_the_server_over_the_cli() -> None:
    defaults = {
        "semif_enabled": True,
        "semif_server": "http://127.0.0.1:8790",
        "semif_command": "semif-score",
        "semif_model": "Qwen/Qwen3.5-4B",
        "semif_revision": "r",
        "semif_timeout_seconds": 5,
    }

    scorer = scorer_from_config(defaults)

    assert isinstance(scorer, SemifHttpScorer)
    assert not isinstance(scorer, SemifCliScorer)


def test_factory_still_builds_the_cli_scorer_without_a_server() -> None:
    scorer = scorer_from_config(
        {
            "semif_enabled": True,
            "semif_command": "semif-score",
            "semif_model": "Qwen/Qwen3.5-4B",
            "semif_revision": "r",
            "semif_timeout_seconds": DEFAULT_TIMEOUT_SECONDS,
        }
    )

    assert isinstance(scorer, SemifCliScorer)


def test_config_server_mode_needs_no_model_coordinates(tmp_path) -> None:
    from lhht.config import load_run_defaults

    config = tmp_path / ".lhht"
    config.mkdir()
    (config / "config.toml").write_text(
        '[run.semif]\nenabled = true\nserver = "http://127.0.0.1:8790"\n',
        encoding="utf-8",
    )

    defaults = load_run_defaults(config / "config.toml")
    assert defaults["semif_server"] == "http://127.0.0.1:8790"

    # The CLI path still refuses an enabled section without coordinates.
    (config / "config.toml").write_text(
        "[run.semif]\nenabled = true\n", encoding="utf-8"
    )
    from lhht.config import ProjectConfigError

    with pytest.raises(ProjectConfigError, match="run.semif.server"):
        load_run_defaults(config / "config.toml")
