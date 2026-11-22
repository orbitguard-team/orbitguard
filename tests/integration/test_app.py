"""Сквозные проверки интерфейса на сценариях OrbitGuard."""

import json
from pathlib import Path

import pytest

from orbitguard.metrics import export_result_json


APP = Path(__file__).resolve().parents[2] / "app.py"


@pytest.fixture
def app():
    """Запускает интерфейс, если установлены зависимости UI.

    Returns:
        streamlit.testing.v1.AppTest: Состояние после первого рендера.
    """
    pytest.importorskip("plotly")
    streamlit_testing = pytest.importorskip("streamlit.testing.v1")
    test = streamlit_testing.AppTest.from_file(str(APP), default_timeout=90)
    return test.run()


@pytest.mark.parametrize(
    ("name", "status", "has_maneuver"),
    (
        ("Безопасный пролёт", "already_safe", False),
        ("Опасное сближение", "feasible", True),
        ("Позднее предупреждение", "no_feasible_candidate", False),
    ),
)
def test_calculated_scenarios(app, name, status, has_maneuver):
    """Рисует графики и экспортирует данные одного запуска."""
    app.selectbox[0].select(name)
    app.button[0].click().run()
    assert not app.exception
    run = app.session_state["run"]
    assert run.result.status.value == status
    assert (run.result.best is not None) is has_maneuver
    assert bool(app.success) is (status != "no_feasible_candidate")
    assert bool(app.warning) is (status == "no_feasible_candidate")
    positions, distances = [
        json.loads(chart.proto.spec) for chart in app.get("plotly_chart")
    ]
    assert positions["layout"]["scene"]["aspectmode"] == "data"
    assert any(trace["name"] == "Импульс" for trace in positions["data"]) is (
        has_maneuver
    )
    threshold = run.scenario.safe_distance_m / 1000
    assert any(
        shape["y0"] == shape["y1"] == threshold
        for shape in distances["layout"]["shapes"]
    )
    assert len(app.get("download_button")) == 1
    report = json.loads(export_result_json(
        run.scenario, run.result, candidates=run.candidates,
        compute_time_s=run.elapsed_s,
    ))
    assert report["metrics"]["status"] == status
    assert report["scenario"]["horizon_s"] == run.scenario.horizon_s
    assert len(report["candidates"]) == len(run.candidates)


def test_settings_clear_old_result_and_report_errors(app):
    """Изменение силы и ошибочного ввода убирает результат."""
    app.selectbox[0].select("Безопасный пролёт")
    app.button[0].click().run()
    assert len(app.get("plotly_chart")) == 2
    j2 = app.checkbox(key="safe_flyby-j2")
    assert j2.value is False
    j2.check().run()
    assert "run" not in app.session_state
    assert not app.get("download_button")
    app.button[0].click().run()
    assert "не поддерживается" in app.error[0].value
    app.checkbox(key="safe_flyby-j2").uncheck()
    app.number_input(key="safe_flyby-budget").set_value(-1)
    app.button[0].click().run()
    assert "Проверьте настройки" in app.error[0].value
    assert not app.get("plotly_chart")
