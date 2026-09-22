"""Semantic salvage fallback for control lines the regexes missed.

The regexes stay the only fast path; salvage runs only on a miss, and every
scorer failure must degrade to None (today's fallback) rather than raise.
`SemifCliScorer` is exercised end to end against a fake but strict
`semif-score` executable that mirrors the real CLI contract: required
--mode/--model/--revision/--input/--output flags, an output file the CLI
itself creates (and refuses to pre-create), and the --gguf/--backend
pairing the real parser errors on when split.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from lhht import auditor_agent
from lhht import role_prompts
from lhht import semantic_salvage as semantic_salvage_module
from lhht.auditor_agent import (
    audit_report_from_episode_result,
    has_valid_auditor_control_header,
    parse_audit_report,
)
from lhht.config import ProjectConfigError, _flatten_run_table, load_run_defaults
from lhht.role_prompts import (
    MANAGER_NEXT_ASK,
    MANAGER_NEXT_BLOCKED,
    MANAGER_NEXT_CLI,
    MANAGER_NEXT_DONE,
    MANAGER_NEXT_GUI,
    MANAGER_NEXT_INVALID,
    parse_role_manager_next_step,
)
from lhht.semantic_salvage import (
    DEFAULT_THRESHOLD,
    DEFAULT_TIMEOUT_SECONDS,
    SALVAGE_QUESTION,
    SemifCliScorer,
    salvage_control_value,
    scorer_from_config,
)
from lhht.types import (
    EpisodeResult,
    audit_report_from_dict,
    audit_report_to_dict,
)

MISSED_LINE = "Статус: сделано, всё готово"
LEGAL_VALUES = ["complete", "incomplete", "blocked"]
DESCRIPTIONS = {
    "complete": "The task is complete; every deliverable is done.",
    "incomplete": "The task is incomplete; work remains.",
    "blocked": "The task is blocked; a dependency failed.",
}


class FakeScorer:
    """Deterministic in-process scorer standing in for the subprocess."""

    def __init__(self, probabilities=None, error=None):
        self.probabilities = probabilities
        self.error = error
        self.calls: list[dict] = []

    def score(self, state, question, options):
        self.calls.append({"state": state, "question": question, "options": options})
        if self.error is not None:
            raise self.error
        return self.probabilities


def test_miss_with_top_probability_above_threshold_is_salvaged():
    scorer = FakeScorer(probabilities=[0.05, 0.9, 0.05])
    result = salvage_control_value(MISSED_LINE, LEGAL_VALUES, DESCRIPTIONS, scorer)
    assert result is not None
    assert result.value == "incomplete"
    assert result.probabilities == [0.05, 0.9, 0.05]


def test_salvage_at_exact_threshold_is_salvaged():
    scorer = FakeScorer(probabilities=[0.8, 0.15, 0.05])
    result = salvage_control_value(MISSED_LINE, LEGAL_VALUES, DESCRIPTIONS, scorer)
    assert result is not None
    assert result.value == "complete"


def test_salvage_passes_line_question_and_typed_options_to_the_scorer():
    scorer = FakeScorer(probabilities=[0.9, 0.05, 0.05])
    salvage_control_value(MISSED_LINE, LEGAL_VALUES, DESCRIPTIONS, scorer)
    assert len(scorer.calls) == 1
    call = scorer.calls[0]
    assert call["state"] == MISSED_LINE
    assert call["question"] == SALVAGE_QUESTION
    assert call["options"] == [
        {"id": value, "description": DESCRIPTIONS[value]} for value in LEGAL_VALUES
    ]


def test_salvage_below_threshold_returns_none():
    scorer = FakeScorer(probabilities=[0.7, 0.2, 0.1])
    assert salvage_control_value(MISSED_LINE, LEGAL_VALUES, DESCRIPTIONS, scorer) is None


def test_salvage_honors_a_custom_threshold():
    scorer = FakeScorer(probabilities=[0.65, 0.3, 0.05])
    assert salvage_control_value(MISSED_LINE, LEGAL_VALUES, DESCRIPTIONS, scorer, 0.6) is not None
    assert salvage_control_value(MISSED_LINE, LEGAL_VALUES, DESCRIPTIONS, scorer, 0.7) is None


def test_scorer_returning_none_returns_none():
    scorer = FakeScorer(probabilities=None)
    assert salvage_control_value(MISSED_LINE, LEGAL_VALUES, DESCRIPTIONS, scorer) is None


def test_scorer_error_degrades_to_none_without_propagating():
    scorer = FakeScorer(error=RuntimeError("semif is deliberately strict"))
    assert salvage_control_value(MISSED_LINE, LEGAL_VALUES, DESCRIPTIONS, scorer) is None


def test_scorer_with_malformed_probabilities_returns_none():
    for probabilities in ([0.9, 0.1], [1.0, 0.0, 0.0, 0.0], "0.9"):
        scorer = FakeScorer(probabilities=probabilities)
        assert salvage_control_value(MISSED_LINE, LEGAL_VALUES, DESCRIPTIONS, scorer) is None


def test_tie_breaks_to_the_first_legal_value():
    scorer = FakeScorer(probabilities=[0.5, 0.5, 0.0])
    result = salvage_control_value(MISSED_LINE, LEGAL_VALUES, DESCRIPTIONS, scorer, 0.4)
    assert result is not None
    assert result.value == "complete"


def test_no_scorer_returns_none_without_any_decision():
    assert salvage_control_value(MISSED_LINE, LEGAL_VALUES, DESCRIPTIONS, None) is None


# --- SemifCliScorer against a fake but contract-faithful semif-score CLI ---

FAKE_SCRIPT = """\
import argparse, json, os, sys, time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--mode")
parser.add_argument("--backend")
parser.add_argument("--gguf")
parser.add_argument("--model")
parser.add_argument("--revision")
parser.add_argument("--input", type=Path)
parser.add_argument("--output", type=Path)
args = parser.parse_args()
rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
record = {"argv": sys.argv[1:], "rows": rows, "output_existed_before": args.output.exists()}
Path(os.environ["LHHT_FAKE_RECORD"]).write_text(json.dumps(record), encoding="utf-8")
# Mirror the real parser's cross-checks: --gguf only rides with the llamacpp
# backend, and llamacpp refuses to start without a GGUF checkpoint.
if args.gguf is not None and args.backend != "llamacpp":
    sys.exit(2)
if args.backend == "llamacpp" and args.gguf is None:
    sys.exit(2)
behavior = os.environ.get("LHHT_FAKE_BEHAVIOR", "ok")
if behavior == "exit":
    sys.exit(3)
if behavior == "hang":
    # Release the inherited pipes at the OS level so the parent's post-kill
    # read does not wait for this orphaned sleeper on Windows; closing
    # sys.stdout alone leaves the kernel handle open behind cmd.exe.
    for fd in (1, 2):
        try:
            os.close(fd)
        except OSError:
            pass
    time.sleep(120)
if behavior == "no-output":
    sys.exit(0)
out = []
for row in rows:
    if behavior == "garbage":
        out.append("this line is not json")
        continue
    probabilities = [float(v) for v in os.environ.get("LHHT_FAKE_PROBABILITIES", "0.9,0.05,0.05").split(",")]
    if behavior == "short":
        probabilities = probabilities[:1]
    out.append(json.dumps({"id": row["id"], "option_ids": [o["id"] for o in row["options"]], "probabilities": probabilities}))
with args.output.open("x", encoding="utf-8") as fh:
    for line in out:
        fh.write(line + "\\n")
"""


@pytest.fixture
def fake_semif(tmp_path, monkeypatch):
    script = tmp_path / "fake_semif_score.py"
    script.write_text(FAKE_SCRIPT, encoding="utf-8")
    if sys.platform == "win32":
        wrapper = tmp_path / "semif-score.cmd"
        wrapper.write_text(f'@"{sys.executable}" -X utf8 "{script}" %*\r\n', encoding="ascii")
    else:
        wrapper = tmp_path / "semif-score"
        wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" -X utf8 "{script}" "$@"\n', encoding="ascii")
        wrapper.chmod(0o755)
    record = tmp_path / "record.json"
    monkeypatch.setenv("LHHT_FAKE_RECORD", str(record))
    return SimpleNamespace(command=str(wrapper), record=record)


def scorer_command(fake_semif, **kwargs):
    return SemifCliScorer(
        fake_semif.command, "Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", **kwargs
    )


def cli_options():
    return [{"id": value, "description": DESCRIPTIONS[value]} for value in LEGAL_VALUES]


def read_record(fake_semif):
    return json.loads(fake_semif.record.read_text(encoding="utf-8"))


def test_cli_scorer_reads_probabilities_back(fake_semif, monkeypatch):
    monkeypatch.setenv("LHHT_FAKE_PROBABILITIES", "0.05,0.9,0.05")
    scorer = scorer_command(fake_semif)
    before = set(Path(tempfile.gettempdir()).glob("lhht-semif-*"))
    assert scorer.score(MISSED_LINE, SALVAGE_QUESTION, cli_options()) == [0.05, 0.9, 0.05]
    after = set(Path(tempfile.gettempdir()).glob("lhht-semif-*"))
    assert after == before  # the temp working directory was cleaned up

    record = read_record(fake_semif)
    argv = record["argv"]
    for flag, value in (
        ("--mode", "direct"),
        ("--model", "Qwen/Qwen3.5-4B"),
        ("--revision", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"),
    ):
        assert argv[argv.index(flag) + 1] == value
    assert "--gguf" not in argv
    assert "--backend" not in argv  # the pairing only rides with a gguf
    assert record["output_existed_before"] is False
    (input_flag,) = [argv[i + 1] for i, item in enumerate(argv) if item == "--input"]
    (output_flag,) = [argv[i + 1] for i, item in enumerate(argv) if item == "--output"]
    assert input_flag != output_flag
    row = record["rows"][0]
    assert row["id"] and isinstance(row["id"], str)
    assert row["state"] == MISSED_LINE
    assert row["question"] == SALVAGE_QUESTION
    assert row["options"] == cli_options()


def test_cli_scorer_passes_gguf_when_configured(fake_semif):
    scorer = scorer_command(fake_semif, gguf="models/qwen.gguf")
    # The fake exits non-zero on --gguf without --backend llamacpp, so a
    # successful score here proves the backend flag rode along.
    assert scorer.score(MISSED_LINE, SALVAGE_QUESTION, cli_options()) == [0.9, 0.05, 0.05]
    argv = read_record(fake_semif)["argv"]
    assert argv[argv.index("--gguf") + 1] == "models/qwen.gguf"
    assert argv[argv.index("--backend") + 1] == "llamacpp"


def test_the_fake_semif_rejects_a_split_gguf_backend_pairing(fake_semif, tmp_path):
    """Guard the guard: the fake must keep refusing what semif-score refuses,
    or the pairing test above would silently stop covering the repair."""
    row = {
        "id": "x",
        "state": "s",
        "question": "q",
        "options": [
            {"id": "a", "description": "A"},
            {"id": "b", "description": "B"},
        ],
    }
    input_path = tmp_path / "in.jsonl"
    input_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    for extra in (["--gguf", "model.gguf"], ["--backend", "llamacpp"]):
        completed = subprocess.run(
            [
                fake_semif.command,
                *extra,
                "--mode", "direct",
                "--input", str(input_path),
                "--output", str(tmp_path / "out.jsonl"),
            ],
            capture_output=True,
        )
        assert completed.returncode != 0


def test_cli_scorer_nonzero_exit_returns_none(fake_semif, monkeypatch):
    monkeypatch.setenv("LHHT_FAKE_BEHAVIOR", "exit")
    assert scorer_command(fake_semif).score(MISSED_LINE, SALVAGE_QUESTION, cli_options()) is None


def test_cli_scorer_missing_output_returns_none(fake_semif, monkeypatch):
    monkeypatch.setenv("LHHT_FAKE_BEHAVIOR", "no-output")
    assert scorer_command(fake_semif).score(MISSED_LINE, SALVAGE_QUESTION, cli_options()) is None


def test_cli_scorer_malformed_output_returns_none(fake_semif, monkeypatch):
    monkeypatch.setenv("LHHT_FAKE_BEHAVIOR", "garbage")
    assert scorer_command(fake_semif).score(MISSED_LINE, SALVAGE_QUESTION, cli_options()) is None


def test_cli_scorer_wrong_length_probabilities_return_none(fake_semif, monkeypatch):
    monkeypatch.setenv("LHHT_FAKE_BEHAVIOR", "short")
    assert scorer_command(fake_semif).score(MISSED_LINE, SALVAGE_QUESTION, cli_options()) is None


def test_cli_scorer_timeout_returns_none(fake_semif, monkeypatch):
    monkeypatch.setenv("LHHT_FAKE_BEHAVIOR", "hang")
    scorer = scorer_command(fake_semif, timeout_seconds=1)
    started = time.monotonic()
    assert scorer.score(MISSED_LINE, SALVAGE_QUESTION, cli_options()) is None
    assert time.monotonic() - started < 30


def test_cli_scorer_missing_command_returns_none(tmp_path):
    scorer = SemifCliScorer(str(tmp_path / "no-such-semif-score"), "m", "r")
    assert scorer.score(MISSED_LINE, SALVAGE_QUESTION, cli_options()) is None


def test_cli_scorer_skips_the_subprocess_for_invalid_rows(fake_semif):
    scorer = scorer_command(fake_semif)
    one_option = cli_options()[:1]  # SemIf requires 2-16 options
    assert scorer.score(MISSED_LINE, SALVAGE_QUESTION, one_option) is None
    assert not fake_semif.record.exists()


# --- [run.semif] config table and the scorer construction path ---------------

REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


def test_defaults_without_a_semif_section_stay_byte_for_byte_and_build_no_scorer():
    defaults = _flatten_run_table({"model": "gpt-5.6-sol", "max_rounds": 25})
    assert defaults == {"model": "gpt-5.6-sol", "max_rounds": 25}
    assert scorer_from_config(defaults) is None


def test_a_disabled_semif_section_builds_no_scorer_and_never_runs_it(fake_semif):
    defaults = _flatten_run_table(
        {
            "semif": {
                "enabled": False,
                "command": fake_semif.command,
                "model": "Qwen/Qwen3.5-4B",
                "revision": REVISION,
            }
        }
    )
    assert defaults["semif_enabled"] is False
    assert scorer_from_config(defaults) is None
    assert not fake_semif.record.exists()


def test_a_disabled_semif_section_leaves_every_other_default_untouched():
    base = {
        "model": "gpt-5.6-sol",
        "guard_exclude_paths": ["target"],
        "timeouts": {"manager": 300},
        "roles": {"auditor": {"model": "gpt-5.7"}},
    }
    without = _flatten_run_table(dict(base))
    with_disabled = _flatten_run_table(
        {
            **base,
            "semif": {
                "enabled": False,
                "command": "semif-score",
                "model": "Qwen/Qwen3.5-4B",
                "revision": REVISION,
            },
        }
    )
    assert {
        key: value
        for key, value in with_disabled.items()
        if not key.startswith("semif_")
    } == without


def test_an_enabled_semif_section_flattens_and_builds_a_working_scorer(
    fake_semif, monkeypatch
):
    monkeypatch.setenv("LHHT_FAKE_PROBABILITIES", "0.9,0.05,0.05")
    defaults = _flatten_run_table(
        {
            "semif": {
                "enabled": True,
                "command": fake_semif.command,
                "model": "Qwen/Qwen3.5-4B",
                "revision": REVISION,
                "gguf": "models/qwen.gguf",
                "threshold": 0.9,
                "timeout_seconds": 30,
            }
        }
    )
    assert defaults == {
        "semif_enabled": True,
        "semif_command": fake_semif.command,
        "semif_model": "Qwen/Qwen3.5-4B",
        "semif_revision": REVISION,
        "semif_gguf": "models/qwen.gguf",
        "semif_threshold": 0.9,
        "semif_timeout_seconds": 30,
    }
    scorer = scorer_from_config(defaults)
    assert isinstance(scorer, SemifCliScorer)
    assert scorer.score(MISSED_LINE, SALVAGE_QUESTION, cli_options()) == [0.9, 0.05, 0.05]
    argv = read_record(fake_semif)["argv"]
    assert argv[argv.index("--backend") + 1] == "llamacpp"


def test_configured_timeout_seconds_reach_the_subprocess(fake_semif, monkeypatch):
    monkeypatch.setenv("LHHT_FAKE_BEHAVIOR", "hang")
    defaults = _flatten_run_table(
        {
            "semif": {
                "enabled": True,
                "command": fake_semif.command,
                "model": "Qwen/Qwen3.5-4B",
                "revision": REVISION,
                "timeout_seconds": 1,
            }
        }
    )
    scorer = scorer_from_config(defaults)
    started = time.monotonic()
    assert scorer.score(MISSED_LINE, SALVAGE_QUESTION, cli_options()) is None
    assert time.monotonic() - started < 30


def test_module_constants_pin_the_mandated_defaults():
    assert DEFAULT_THRESHOLD == 0.8
    assert DEFAULT_TIMEOUT_SECONDS == 30


def test_unknown_semif_keys_are_rejected():
    with pytest.raises(ProjectConfigError, match=r"unknown \[run\.semif\] key"):
        _flatten_run_table({"semif": {"enabled": False, "surprise": 1}})


@pytest.mark.parametrize(
    ("table", "message"),
    (
        ({"enabled": "yes"}, "must be true or false"),
        ({"command": ""}, "must be a non-empty string"),
        ({"model": 7}, "must be a non-empty string"),
        ({"threshold": "0.8"}, "must be a number greater than 0 and at most 1"),
        ({"threshold": 0}, "must be a number greater than 0 and at most 1"),
        ({"threshold": 1.5}, "must be a number greater than 0 and at most 1"),
        ({"timeout_seconds": 0}, "must be an integer of at least 1"),
        ({"timeout_seconds": 2.5}, "must be an integer of at least 1"),
    ),
)
def test_bad_semif_values_are_rejected(table, message):
    with pytest.raises(ProjectConfigError, match=message):
        _flatten_run_table({"semif": table})


def test_enabling_without_the_required_keys_is_rejected():
    with pytest.raises(ProjectConfigError, match="run.semif.enabled = true"):
        _flatten_run_table({"semif": {"enabled": True}})
    with pytest.raises(ProjectConfigError, match="run.semif.model"):
        _flatten_run_table(
            {"semif": {"enabled": True, "command": "semif-score", "revision": REVISION}}
        )


def test_load_run_defaults_reads_the_semif_table_end_to_end(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text(
        "[run]\n"
        'model = "gpt-5.6-sol"\n'
        "\n"
        "[run.semif]\n"
        "enabled = true\n"
        'command = "semif-score"\n'
        'model = "Qwen/Qwen3.5-4B"\n'
        f'revision = "{REVISION}"\n',
        encoding="utf-8",
    )
    defaults = load_run_defaults(config)
    assert defaults["model"] == "gpt-5.6-sol"
    assert defaults["semif_enabled"] is True
    assert defaults["semif_command"] == "semif-score"
    assert isinstance(scorer_from_config(defaults), SemifCliScorer)


def test_load_run_defaults_without_semif_builds_no_scorer(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text('[run]\nmodel = "gpt-5.6-sol"\n', encoding="utf-8")
    assert scorer_from_config(load_run_defaults(config)) is None


# --- auditor control-header salvage cascade -----------------------------------
#
# The three _parse_*_control_header() parsers keep the regex as the only fast
# path and attempt salvage only after a miss; the parsed report must record
# where a salvaged verdict came from, with the per-option probabilities.

REGEX_HIT_HEADER = (
    "Статус: завершено\n"
    "Целостность: чисто\n"
    "Аудит контракта: согласован\n"
)
STATUS_MISS_RAW = (
    "Статус: сделано, всё готово\n"
    "Integrity: clean\n"
    "Contract audit: aligned\n"
)
INTEGRITY_MISS_RAW = (
    "Status: complete\n"
    "Целостность: проверена, вопросов нет\n"
    "Contract audit: aligned\n"
)
CONTRACT_MISS_RAW = (
    "Status: complete\n"
    "Integrity: clean\n"
    "Аудит контракта: в целом соответствует требованиям\n"
)
PROSE_WITHOUT_LABELS = (
    "Просто контекст без контрольных строк.\nЕщё строка.\nИ ещё одна строка."
)


@pytest.fixture
def fresh_salvage_state(monkeypatch):
    """Isolate the auditor parsers' process-wide [run.semif] resolution."""
    monkeypatch.setattr(auditor_agent, "_PROJECT_SALVAGE_SETTINGS", {})


def today_invalid_header_fallback(report):
    assert report.status == "blocked"
    assert report.integrity_status == "suspect"
    assert report.contract_audit_status == "unknown"
    assert report.control_salvage == []
    assert "lacks a valid three-line control header" in report.report_text


def test_regex_hit_keeps_today_behavior_and_never_touches_the_scorer(
    fresh_salvage_state,
):
    scorer = FakeScorer(probabilities=None)
    report = parse_audit_report(REGEX_HIT_HEADER, 1, scorer=scorer)
    assert scorer.calls == []
    assert report.status == "complete"
    assert report.integrity_status == "clean"
    assert report.contract_audit_status == "aligned"
    assert report.control_salvage == []


def test_status_miss_with_top_probability_at_or_above_threshold_is_salvaged(
    fresh_salvage_state,
):
    scorer = FakeScorer(probabilities=[0.9, 0.08, 0.02])
    report = parse_audit_report(STATUS_MISS_RAW, 1, scorer=scorer)
    assert report.status == "complete"
    assert report.integrity_status == "clean"  # regex hits survive a salvaged line
    assert report.contract_audit_status == "aligned"
    assert report.control_salvage == [
        {
            "control": "status",
            "value": "complete",
            "probabilities": {"complete": 0.9, "incomplete": 0.08, "blocked": 0.02},
        }
    ]
    # The header check and the status extraction share one memoized decision.
    assert len(scorer.calls) == 1
    assert scorer.calls[0]["state"] == "Статус: сделано, всё готово"
    assert scorer.calls[0]["question"] == SALVAGE_QUESTION
    assert scorer.calls[0]["options"] == [
        {"id": value, "description": auditor_agent._STATUS_VALUE_DESCRIPTIONS[value]}
        for value in ("complete", "incomplete", "blocked")
    ]
    assert "сделано" in report.report_text
    assert "lacks a valid three-line control header" not in report.report_text


def test_integrity_miss_with_top_probability_at_or_above_threshold_is_salvaged(
    fresh_salvage_state,
):
    scorer = FakeScorer(probabilities=[0.9, 0.08, 0.02])
    report = parse_audit_report(INTEGRITY_MISS_RAW, 1, scorer=scorer)
    assert report.status == "complete"
    assert report.integrity_status == "clean"
    assert report.integrity_findings == []
    assert report.contract_audit_status == "aligned"
    assert report.control_salvage == [
        {
            "control": "integrity",
            "value": "clean",
            "probabilities": {"clean": 0.9, "suspect": 0.08, "violation": 0.02},
        }
    ]
    assert len(scorer.calls) == 1
    assert scorer.calls[0]["state"] == "Целостность: проверена, вопросов нет"


def test_contract_audit_miss_with_top_probability_at_or_above_threshold_is_salvaged(
    fresh_salvage_state,
):
    scorer = FakeScorer(probabilities=[0.9, 0.05, 0.03, 0.02])
    report = parse_audit_report(CONTRACT_MISS_RAW, 1, scorer=scorer)
    assert report.status == "complete"
    assert report.integrity_status == "clean"
    assert report.contract_audit_status == "aligned"
    assert report.control_salvage == [
        {
            "control": "contract_audit",
            "value": "aligned",
            "probabilities": {
                "aligned": 0.9,
                "unknown": 0.05,
                "needs_revision": 0.03,
                "invalid": 0.02,
            },
        }
    ]
    assert scorer.calls[0]["state"] == "Аудит контракта: в целом соответствует требованиям"


def test_miss_below_threshold_keeps_today_defaults(fresh_salvage_state):
    scorer = FakeScorer(probabilities=[0.7, 0.2, 0.1])
    report = parse_audit_report(STATUS_MISS_RAW, 1, scorer=scorer)
    assert scorer.calls  # salvage was attempted and simply lost
    today_invalid_header_fallback(report)


@pytest.mark.parametrize(
    "scorer",
    (
        FakeScorer(error=RuntimeError("semif is deliberately strict")),
        FakeScorer(probabilities=None),
    ),
    ids=["scorer-raises", "scorer-returns-none"],
)
def test_scorer_failure_keeps_today_defaults(scorer, fresh_salvage_state):
    today_invalid_header_fallback(parse_audit_report(STATUS_MISS_RAW, 1, scorer=scorer))


def test_prose_without_control_labels_never_calls_the_scorer(fresh_salvage_state):
    scorer = FakeScorer(probabilities=[0.9, 0.05, 0.05])
    report = parse_audit_report(PROSE_WITHOUT_LABELS, 1, scorer=scorer)
    assert scorer.calls == []
    assert report.status == "blocked"
    assert report.control_salvage == []


def test_auditor_salvage_honors_an_explicit_threshold(fresh_salvage_state):
    salvaged = parse_audit_report(
        STATUS_MISS_RAW,
        1,
        scorer=FakeScorer(probabilities=[0.65, 0.3, 0.05]),
        threshold=0.6,
    )
    assert salvaged.status == "complete"
    strict = parse_audit_report(
        STATUS_MISS_RAW, 1, scorer=FakeScorer(probabilities=[0.65, 0.3, 0.05])
    )
    assert strict.status == "blocked"  # 0.65 stays below the 0.8 default


def disabled_semif_defaults():
    return _flatten_run_table(
        {
            "semif": {
                "enabled": False,
                "command": "semif-score",
                "model": "Qwen/Qwen3.5-4B",
                "revision": REVISION,
            }
        }
    )


def test_disabled_config_constructs_no_scorer_and_keeps_defaults(
    monkeypatch, fresh_salvage_state
):
    monkeypatch.setattr(auditor_agent, "load_run_defaults", disabled_semif_defaults)

    def must_not_construct(*args, **kwargs):
        raise AssertionError("a disabled [run.semif] must construct no scorer")

    monkeypatch.setattr(semantic_salvage_module, "SemifCliScorer", must_not_construct)
    today_invalid_header_fallback(parse_audit_report(STATUS_MISS_RAW, 1))
    assert has_valid_auditor_control_header(STATUS_MISS_RAW) is False


def enabled_semif_defaults(**extra):
    return _flatten_run_table(
        {
            "semif": {
                "enabled": True,
                "command": "semif-score",
                "model": "Qwen/Qwen3.5-4B",
                "revision": REVISION,
                **extra,
            }
        }
    )


def test_enabled_config_supplies_the_scorer_to_the_parsers(
    monkeypatch, fresh_salvage_state
):
    monkeypatch.setattr(
        auditor_agent, "load_run_defaults", lambda: enabled_semif_defaults()
    )
    fake = FakeScorer(probabilities=[0.9, 0.08, 0.02])
    monkeypatch.setattr(
        semantic_salvage_module, "SemifCliScorer", lambda *args, **kwargs: fake
    )
    report = parse_audit_report(STATUS_MISS_RAW, 1)
    assert report.status == "complete"
    assert report.control_salvage == [
        {
            "control": "status",
            "value": "complete",
            "probabilities": {"complete": 0.9, "incomplete": 0.08, "blocked": 0.02},
        }
    ]
    assert has_valid_auditor_control_header(STATUS_MISS_RAW) is True


def test_config_threshold_governs_the_configured_scorer(monkeypatch, fresh_salvage_state):
    monkeypatch.setattr(
        auditor_agent, "load_run_defaults", lambda: enabled_semif_defaults(threshold=0.95)
    )
    monkeypatch.setattr(
        semantic_salvage_module,
        "SemifCliScorer",
        lambda *args, **kwargs: FakeScorer(probabilities=[0.9, 0.08, 0.02]),
    )
    today_invalid_header_fallback(parse_audit_report(STATUS_MISS_RAW, 1))


def test_control_header_predicate_stays_regex_only_when_salvage_is_not_configured(
    monkeypatch, fresh_salvage_state
):
    monkeypatch.setattr(auditor_agent, "load_run_defaults", lambda: {})
    assert has_valid_auditor_control_header(REGEX_HIT_HEADER) is True
    assert has_valid_auditor_control_header(STATUS_MISS_RAW) is False


def test_episode_result_report_salvages_and_records_provenance(fresh_salvage_state):
    scorer = FakeScorer(probabilities=[0.9, 0.08, 0.02])
    result = EpisodeResult(status="done", actions_log=STATUS_MISS_RAW, duration_ms=10)
    report = audit_report_from_episode_result(result, 1, scorer=scorer)
    assert report.status == "complete"
    assert report.integrity_status == "clean"
    assert report.contract_audit_status == "aligned"
    assert report.control_salvage == [
        {
            "control": "status",
            "value": "complete",
            "probabilities": {"complete": 0.9, "incomplete": 0.08, "blocked": 0.02},
        }
    ]


def test_control_salvage_provenance_survives_serialization(fresh_salvage_state):
    scorer = FakeScorer(probabilities=[0.9, 0.08, 0.02])
    report = parse_audit_report(STATUS_MISS_RAW, 1, scorer=scorer)
    data = audit_report_to_dict(report)
    assert data["control_salvage"] == report.control_salvage
    assert audit_report_from_dict(data).control_salvage == report.control_salvage
    legacy = audit_report_from_dict({"round_id": "round_1", "status": "blocked"})
    assert legacy.control_salvage == []


# --- manager route salvage cascade ---------------------------------------------
#
# parse_role_manager_next_step() keeps the string sets as the only fast path
# and attempts salvage only after a total miss; a salvaged route replaces
# MANAGER_NEXT_INVALID, and every scorer failure keeps today's fallback.

ROUTE_MISS_LINE = "Следующий шаг: спросить оператора"
ROUTE_MISS_RAW = (
    "Раунд завершён без нарушений.\n"
    f"{ROUTE_MISS_LINE}\n"
    "Продолжаем по плану."
)
ROUTE_PROSE_WITHOUT_LABEL = (
    "Никакого явного маршрута в этом ответе нет.\n"
    "Просто рассуждение о плане."
)
ROUTE_HIT_INPUTS = (
    ("Next: cli", MANAGER_NEXT_CLI),
    ("**Next: done** — всё готово", MANAGER_NEXT_DONE),
    ("Следующий шаг: готово", MANAGER_NEXT_DONE),
    ("`next: ask`", MANAGER_NEXT_ASK),
    ("下一步：GUI任务", MANAGER_NEXT_GUI),
    ("Next: blocked", MANAGER_NEXT_BLOCKED),
)
ROUTE_OPTIONS = [
    {
        "id": value,
        "description": role_prompts._MANAGER_ROUTE_VALUE_DESCRIPTIONS[value],
    }
    for value in ("gui", "cli", "ask", "done", "blocked")
]


@pytest.fixture
def fresh_route_salvage_state(monkeypatch):
    """Isolate the manager route parser's process-wide [run.semif] resolution."""
    monkeypatch.setattr(role_prompts, "_ROUTE_SALVAGE_SETTINGS", {})


@pytest.mark.parametrize(("line", "expected"), ROUTE_HIT_INPUTS)
def test_manager_regex_hit_keeps_today_behavior_and_never_touches_the_scorer(
    line, expected, fresh_route_salvage_state
):
    scorer = FakeScorer(probabilities=None)
    assert parse_role_manager_next_step(line, scorer=scorer) == expected
    assert scorer.calls == []


def test_manager_regex_hit_with_semif_enabled_still_never_constructs_the_scorer(
    monkeypatch, fresh_route_salvage_state
):
    monkeypatch.setattr(
        role_prompts, "load_run_defaults", lambda: enabled_semif_defaults()
    )

    def must_not_construct(*args, **kwargs):
        raise AssertionError("a regex hit must construct no scorer")

    monkeypatch.setattr(semantic_salvage_module, "SemifCliScorer", must_not_construct)
    assert parse_role_manager_next_step("Next: done") == MANAGER_NEXT_DONE


def test_manager_miss_with_top_probability_at_or_above_threshold_is_salvaged(
    fresh_route_salvage_state,
):
    scorer = FakeScorer(probabilities=[0.05, 0.03, 0.9, 0.01, 0.01])
    assert parse_role_manager_next_step(ROUTE_MISS_RAW, scorer=scorer) == MANAGER_NEXT_ASK
    assert len(scorer.calls) == 1
    call = scorer.calls[0]
    assert call["state"] == ROUTE_MISS_LINE  # only the label line is re-judged
    assert call["question"] == SALVAGE_QUESTION
    assert call["options"] == ROUTE_OPTIONS


def test_manager_salvage_at_exact_threshold_is_salvaged(fresh_route_salvage_state):
    scorer = FakeScorer(probabilities=[0.8, 0.05, 0.05, 0.05, 0.05])
    assert parse_role_manager_next_step(ROUTE_MISS_RAW, scorer=scorer) == MANAGER_NEXT_GUI


def test_manager_miss_below_threshold_keeps_today_invalid(fresh_route_salvage_state):
    scorer = FakeScorer(probabilities=[0.5, 0.2, 0.1, 0.1, 0.1])
    assert (
        parse_role_manager_next_step(ROUTE_MISS_RAW, scorer=scorer)
        == MANAGER_NEXT_INVALID
    )
    assert scorer.calls  # salvage was attempted and simply lost


@pytest.mark.parametrize(
    "scorer",
    (
        FakeScorer(error=RuntimeError("semif is deliberately strict")),
        FakeScorer(probabilities=None),
    ),
    ids=["scorer-raises", "scorer-returns-none"],
)
def test_manager_scorer_failure_keeps_today_invalid(scorer, fresh_route_salvage_state):
    assert (
        parse_role_manager_next_step(ROUTE_MISS_RAW, scorer=scorer)
        == MANAGER_NEXT_INVALID
    )


def test_manager_salvage_honors_an_explicit_threshold(fresh_route_salvage_state):
    scorer = FakeScorer(probabilities=[0.65, 0.1, 0.1, 0.1, 0.05])
    assert (
        parse_role_manager_next_step(ROUTE_MISS_RAW, scorer=scorer, threshold=0.6)
        == MANAGER_NEXT_GUI
    )
    strict = parse_role_manager_next_step(
        ROUTE_MISS_RAW, scorer=FakeScorer(probabilities=[0.65, 0.1, 0.1, 0.1, 0.05])
    )
    assert strict == MANAGER_NEXT_INVALID  # 0.65 stays below the 0.8 default


def test_manager_prose_without_a_route_label_never_resolves_the_scorer(
    monkeypatch, fresh_route_salvage_state
):
    monkeypatch.setattr(
        role_prompts, "load_run_defaults", lambda: enabled_semif_defaults()
    )

    def must_not_construct(*args, **kwargs):
        raise AssertionError("prose without a route label must construct no scorer")

    monkeypatch.setattr(semantic_salvage_module, "SemifCliScorer", must_not_construct)
    assert (
        parse_role_manager_next_step(ROUTE_PROSE_WITHOUT_LABEL)
        == MANAGER_NEXT_INVALID
    )


def test_manager_disabled_config_constructs_no_scorer_and_keeps_invalid(
    monkeypatch, fresh_route_salvage_state
):
    monkeypatch.setattr(role_prompts, "load_run_defaults", disabled_semif_defaults)

    def must_not_construct(*args, **kwargs):
        raise AssertionError("a disabled [run.semif] must construct no scorer")

    monkeypatch.setattr(semantic_salvage_module, "SemifCliScorer", must_not_construct)
    assert parse_role_manager_next_step(ROUTE_MISS_RAW) == MANAGER_NEXT_INVALID


def test_manager_absent_config_constructs_no_scorer_and_keeps_invalid(
    monkeypatch, fresh_route_salvage_state
):
    monkeypatch.setattr(role_prompts, "load_run_defaults", lambda: {})

    def must_not_construct(*args, **kwargs):
        raise AssertionError("an absent [run.semif] must construct no scorer")

    monkeypatch.setattr(semantic_salvage_module, "SemifCliScorer", must_not_construct)
    assert parse_role_manager_next_step(ROUTE_MISS_RAW) == MANAGER_NEXT_INVALID


def test_manager_enabled_config_supplies_the_scorer(
    monkeypatch, fresh_route_salvage_state
):
    monkeypatch.setattr(
        role_prompts, "load_run_defaults", lambda: enabled_semif_defaults()
    )
    fake = FakeScorer(probabilities=[0.05, 0.03, 0.9, 0.01, 0.01])
    monkeypatch.setattr(
        semantic_salvage_module, "SemifCliScorer", lambda *args, **kwargs: fake
    )
    assert parse_role_manager_next_step(ROUTE_MISS_RAW) == MANAGER_NEXT_ASK


def test_manager_config_threshold_governs_the_configured_scorer(
    monkeypatch, fresh_route_salvage_state
):
    monkeypatch.setattr(
        role_prompts, "load_run_defaults", lambda: enabled_semif_defaults(threshold=0.95)
    )
    monkeypatch.setattr(
        semantic_salvage_module,
        "SemifCliScorer",
        lambda *args, **kwargs: FakeScorer(probabilities=[0.05, 0.03, 0.9, 0.01, 0.01]),
    )
    assert parse_role_manager_next_step(ROUTE_MISS_RAW) == MANAGER_NEXT_INVALID


# --- regressions for spellings observed in live runs ----------------------------
#
# semif-salvage-1 burned rounds 3-4 on a zh route written without the 任务
# suffix, and the hyphenated RU contract label made a whole header unsalvageable
# (label gate) before 2026-09-22. Both must stay recovered.


def test_manager_route_salvage_recovers_zh_route_without_suffix(
    fresh_route_salvage_state,
):
    scorer = FakeScorer(probabilities=[0.02, 0.96, 0.01, 0.005, 0.005])
    raw = "Текущее состояние: работа идёт.\n\n下一步：cli"
    assert parse_role_manager_next_step(raw, scorer=scorer) == MANAGER_NEXT_CLI


def test_audit_salvage_recovers_hyphenated_ru_contract_label(
    fresh_salvage_state,
):
    scorer = FakeScorer(probabilities=[0.97, 0.01, 0.01, 0.01])
    raw = (
        "Статус: завершено\n"
        "Целостность: чисто\n"
        "Контракт-аудит: в полном соответствии с требованиями\n"
        "\n"
        "## Аудит-факты\n"
        "Проверено напрямую."
    )
    report = parse_audit_report(raw, 3, scorer=scorer)
    assert report.status == "complete"
    assert report.integrity_status == "clean"
    assert report.contract_audit_status == "aligned"
    assert [r["control"] for r in report.control_salvage] == ["contract_audit"]
