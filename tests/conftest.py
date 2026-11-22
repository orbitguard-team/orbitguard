"""Общие настройки pytest для OrbitGuard.

Защита от ложного зелёного результата: если ни один тест не прошёл
(например, все пропущены как заглушки), запуск считается проваленным.
"""

import pytest

_STATS = {"passed": 0}


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """Считает прошедшие тесты.

    Args:
        report: Отчёт pytest об одной фазе теста.
    """
    if report.when == "call" and report.passed:
        _STATS["passed"] += 1


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Проваливает запуск, если не прошёл ни один тест.

    Args:
        session: Сессия pytest.
        exitstatus: Код завершения, который pytest собирается вернуть.
    """
    if exitstatus == pytest.ExitCode.OK and _STATS["passed"] == 0:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        if reporter is not None:
            reporter.ensure_newline()
            reporter.write_line(
                "Ни один тест не прошёл: заглушки не дают зелёный результат.",
                red=True,
            )
