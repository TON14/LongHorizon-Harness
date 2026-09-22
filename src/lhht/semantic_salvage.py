"""Semantic salvage for control lines the exact regexes missed.

Role control lines are parsed by trilingual regexes; a miss defaults the
verdict and burns a round. Instead of accumulating ever more hard-coded
synonyms, a miss can be salvaged by one SemIf decision: the legal values
become typed options and a small local model returns one probability per
option. Salvage only ever runs on the regex-miss path -- a successful regex
match is never re-judged -- and every scorer failure degrades to None so
callers keep today's fallback behavior.
"""

from __future__ import annotations

import json
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Any, NamedTuple, Protocol

DEFAULT_THRESHOLD = 0.8
DEFAULT_TIMEOUT_SECONDS = 30

SALVAGE_QUESTION = "Which control value does the line express?"

# SemIf validates every row before scoring: nonempty string id and question,
# nonempty state, and 2-16 options with unique string ids and descriptions
# (D:\semif\src\semif_phase1\core.py, validate_row).
_SALVAGE_ROW_ID = "lhht-salvage"


class SemanticScorer(Protocol):
    """One decision: one probability per option, or None on any failure."""

    def score(
        self, state: str, question: str, options: list[dict[str, str]]
    ) -> list[float] | None: ...


class SalvageResult(NamedTuple):
    value: str
    probabilities: list[float]


class SemifCliScorer:
    """`SemanticScorer` backed by the `semif-score` CLI as a subprocess.

    SemIf is deliberately strict -- it validates rows and refuses to overwrite
    an existing output file -- so each call gets its own fresh temp directory:
    the input JSONL is written there, the output path is left for SemIf to
    create, and both vanish with the directory. Any failure (bad command,
    non-zero exit, timeout, missing or malformed output) returns None instead
    of raising.
    """

    def __init__(
        self,
        command: str,
        model: str,
        revision: str,
        *,
        gguf: str | None = None,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._command = command
        self._model = model
        self._revision = revision
        self._gguf = gguf
        self._timeout_seconds = timeout_seconds

    def score(
        self, state: str, question: str, options: list[dict[str, str]]
    ) -> list[float] | None:
        if not state or not question or not 2 <= len(options) <= 16:
            return None
        row = {
            "id": _SALVAGE_ROW_ID,
            "state": state,
            "question": question,
            "options": options,
        }
        try:
            with tempfile.TemporaryDirectory(
                prefix="lhht-semif-", ignore_cleanup_errors=True
            ) as workdir:
                input_path = Path(workdir) / "input.jsonl"
                output_path = Path(workdir) / "output.jsonl"
                # ensure_ascii keeps the file locale-agnostic: pure ASCII is
                # readable regardless of the platform's default encoding.
                input_path.write_text(
                    json.dumps(row) + "\n", encoding="ascii"
                )
                argv = [
                    self._command,
                    "--mode", "direct",
                    "--model", self._model,
                    "--revision", self._revision,
                    "--input", str(input_path),
                    "--output", str(output_path),
                ]
                if self._gguf is not None:
                    # semif-score rejects --gguf without the llamacpp backend,
                    # and llamacpp refuses to start without a GGUF checkpoint,
                    # so the two flags always travel together.
                    argv += ["--backend", "llamacpp", "--gguf", self._gguf]
                completed = subprocess.run(
                    argv,
                    capture_output=True,
                    check=False,
                    timeout=self._timeout_seconds,
                )
                if completed.returncode != 0 or not output_path.is_file():
                    return None
                output_lines = output_path.read_text(
                    encoding="utf-8"
                ).splitlines()
                if not output_lines:
                    return None
                result = json.loads(output_lines[0])
                probabilities = result.get("probabilities")
                if (
                    result.get("id") != row["id"]
                    or result.get("option_ids") != [o["id"] for o in options]
                    or not _valid_probabilities(probabilities, len(options))
                ):
                    return None
                return [float(p) for p in probabilities]
        except Exception:
            # SemIf raises rather than degrades; salvage must never leak that
            # into the control-line parsers.
            return None


def scorer_from_config(defaults: dict[str, Any]) -> SemifCliScorer | None:
    """Build the scorer an optional ``[run.semif]`` table configures.

    ``defaults`` is the flattened run-config mapping `lhht.config` produces:
    the ``semif_*`` keys exist only when the operator wrote the table, so an
    absent or disabled section yields no scorer and callers keep today's
    behavior. Enabling without command/model/revision is refused at config
    load time; this factory still degrades to None rather than raise, so an
    unexpected defaults mapping can never crash a caller.
    """
    if not defaults.get("semif_enabled"):
        return None
    command = defaults.get("semif_command")
    model = defaults.get("semif_model")
    revision = defaults.get("semif_revision")
    if not all(
        isinstance(value, str) and value for value in (command, model, revision)
    ):
        return None
    return SemifCliScorer(
        command,
        model,
        revision,
        gguf=defaults.get("semif_gguf"),
        timeout_seconds=defaults.get(
            "semif_timeout_seconds", DEFAULT_TIMEOUT_SECONDS
        ),
    )


def _valid_probabilities(values: object, expected: int) -> bool:
    return (
        isinstance(values, list)
        and len(values) == expected
        and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            for value in values
        )
    )


def salvage_control_value(
    line: str,
    legal_values: list[str],
    descriptions_by_value: dict[str, str],
    scorer: SemanticScorer | None,
    threshold: float = DEFAULT_THRESHOLD,
) -> SalvageResult | None:
    """Recover a control value for a line the regexes did not match.

    Returns the winning legal value together with one probability per legal
    value (same order), or None when no scorer is configured, the scorer
    fails or raises, or the top probability stays below `threshold`. Ties go
    to the earliest legal value, keeping salvage deterministic.
    """
    if scorer is None or not legal_values:
        return None
    options = [
        {"id": value, "description": descriptions_by_value.get(value, value)}
        for value in legal_values
    ]
    try:
        probabilities = scorer.score(line, SALVAGE_QUESTION, options)
    except Exception:
        return None
    if not _valid_probabilities(probabilities, len(legal_values)):
        return None
    top = max(range(len(probabilities)), key=probabilities.__getitem__)
    if probabilities[top] < threshold:
        return None
    return SalvageResult(
        legal_values[top], [float(p) for p in probabilities]
    )
