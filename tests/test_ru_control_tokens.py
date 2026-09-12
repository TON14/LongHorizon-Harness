"""Русские контрольные токены менеджера и аудитора.

Русскоязычные задачи (ton-graph) тянут язык ответа модели в русский даже
при английском промпте: аудитор пишет «Статус: complete», менеджер —
«Следующий шаг: cli». До нормализации такие раунды сгорали на
next=invalid / «lacks a valid three-line control header».
"""
from __future__ import annotations

import pytest

from lh_harness.auditor_agent import parse_audit_report
from lh_harness.role_prompts import (
    MANAGER_NEXT_ASK,
    MANAGER_NEXT_BLOCKED,
    MANAGER_NEXT_CLI,
    MANAGER_NEXT_DONE,
    MANAGER_NEXT_GUI,
    parse_role_manager_next_step,
)


@pytest.mark.parametrize(
    "line,expected",
    [
        ("Следующий шаг: cli", MANAGER_NEXT_CLI),
        ("Следующий шаг: cli — продолжить сбор фактов", MANAGER_NEXT_CLI),
        ("Далее: done", MANAGER_NEXT_DONE),
        ("Следующий шаг: готово", MANAGER_NEXT_DONE),
        ("Следующий шаг: gui", MANAGER_NEXT_GUI),
        ("Следующий шаг: ask", MANAGER_NEXT_ASK),
        ("Далее: блок", MANAGER_NEXT_BLOCKED),
        ("**Следующий шаг: cli**", MANAGER_NEXT_CLI),
        ("`Следующий шаг: done`", MANAGER_NEXT_DONE),
        ("Next: cli", MANAGER_NEXT_CLI),
        ("下一步: CLI任务", MANAGER_NEXT_CLI),
    ],
)
def test_manager_route_accepts_russian(line: str, expected) -> None:
    assert parse_role_manager_next_step(line) == expected


@pytest.mark.parametrize(
    "line",
    [
        "Следующий шаг пока не выбран",
        "Статус: размышляю",
    ],
)
def test_manager_route_rejects_russian_prose(line: str) -> None:
    assert parse_role_manager_next_step(line) not in (
        MANAGER_NEXT_CLI,
        MANAGER_NEXT_DONE,
        MANAGER_NEXT_GUI,
        MANAGER_NEXT_ASK,
        MANAGER_NEXT_BLOCKED,
    )


def test_audit_report_parses_russian_control_header() -> None:
    raw = (
        "Статус: завершено\n"
        "Целостность: чисто\n"
        "Аудит контракта: согласован\n"
        "\n"
        "## Сводка аудита\n"
        "Проверено напрямую."
    )
    report = parse_audit_report(raw, 1)
    assert report.status == "complete"
    assert report.integrity_status == "clean"
    assert report.contract_audit_status == "aligned"


def test_audit_report_parses_russian_blocked_header() -> None:
    raw = (
        "Статус: незавершено\n"
        "Целостность: подозрительно\n"
        "Аудит контракта: требуется доработка\n"
        "\n"
        "Найден дефект."
    )
    report = parse_audit_report(raw, 2)
    assert report.status == "incomplete"
    assert report.integrity_status == "suspect"
    assert report.contract_audit_status == "needs_revision"


def test_audit_report_russian_english_mixed() -> None:
    raw = (
        "Статус: complete\n"
        "Integrity: clean\n"
        "Аудит контракта: aligned\n"
    )
    report = parse_audit_report(raw, 3)
    assert report.status == "complete"
    assert report.integrity_status == "clean"
    assert report.contract_audit_status == "aligned"
