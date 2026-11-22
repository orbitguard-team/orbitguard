"""Сквозные проверки трёх синтетических сценариев на реальном расчёте."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from orbitguard import models
from orbitguard.maneuver import build_rtn_candidates, search_maneuver
from orbitguard.metrics import export_result_json
from orbitguard.propagation import propagate
from orbitguard.scenarios import load_scenario_file

ROOT = Path(__file__).resolve().parents[2]


def _load(name):
    """Загружает Scenario и независимое аналитическое ожидание.

    Args:
        name (str): Имя файла без расширения.

    Returns:
        tuple: Сценарий и ожидаемый результат.
    """
    return load_scenario_file(ROOT / "scenarios" / f"{name}.json")


def _candidates(expected):
    """Использует записанный пример манёвра, если он есть.

    Args:
        expected (Expectation): Ожидаемый результат сценария.

    Returns:
        tuple: Упорядоченная сетка входных импульсов.
    """
    return () if expected.witness is None else (expected.witness.impulse,)


@pytest.mark.parametrize("name", [
    "safe_flyby", "dangerous_feasible", "late_warning_no_solution"])
def test_three_scenarios_have_calculated_status(name):
    """Статусы вычисляются, а не читаются из expected."""
    scenario, expected = _load(name)
    result = search_maneuver(scenario, candidates=_candidates(expected))
    assert result.status is expected.status
    assert result.before.closest.t_s == pytest.approx(
        expected.closest.t_s, abs=expected.time_tol_s)
    assert result.before.closest.distance_m == pytest.approx(
        expected.closest.distance_m, abs=expected.distance_tol_m)
    assert result.forces == scenario.forces
    if result.status is models.Status.FEASIBLE:
        assert result.after.accepted and result.best is expected.witness.impulse
    elif result.status is models.Status.ALREADY_SAFE:
        assert result.best is None and result.after == result.before
    else:
        assert result.best is result.after is None


def test_selected_maneuver_safe_on_dense_full_horizon():
    """Независимо уточняет каждый минимум на всём горизонте."""
    scenario, expected = _load("dangerous_feasible")
    burn = expected.witness.impulse
    result = search_maneuver(scenario, candidates=(burn,))
    span = (0, scenario.horizon_s)
    sat = propagate(scenario.satellite, scenario.epoch, span, scenario.forces,
                    numerics=scenario.numerics, impulses=(burn,))
    debris = propagate(scenario.debris, scenario.epoch, span, scenario.forces,
                       numerics=scenario.numerics)
    def distance(time):
        """Расстояние по независимо выбранному времени, м.

        Args:
            time (float): Секунды от начала сценария.

        Returns:
            float: Расстояние между объектами, м.
        """
        return float(np.linalg.norm(sat.state_at(float(time))[:3]
                                    - debris.state_at(float(time))[:3]))

    grid = np.arange(span[0], span[1] + 1, 1.0)
    distances = np.array([distance(t) for t in grid])
    local = np.flatnonzero((distances[1:-1] <= distances[:-2])
                          & (distances[1:-1] <= distances[2:])) + 1
    refined = [minimize_scalar(distance, bounds=(grid[i - 1], grid[i + 1]),
                               method="bounded", options={"xatol": 1e-6})
               for i in local]
    assert all(item.success for item in refined)
    independently_found = [(span[0], distances[0]),
                           *((item.x, item.fun) for item in refined),
                           (span[1], distances[-1])]
    required = scenario.safe_distance_m + scenario.numerics.distance_margin_m
    assert len(independently_found) == len(result.after.all_minima)
    for (time, distance_m), recorded in zip(
            independently_found, result.after.all_minima):
        assert time == pytest.approx(recorded.t_s, abs=0.01)
        assert distance_m == pytest.approx(recorded.distance_m, abs=0.1)
    assert min(distance_m for _, distance_m in independently_found) >= required


def test_insufficient_budget_and_late_window_are_explicit():
    """Неверный бюджет и окно возвращают no_feasible_candidate."""
    scenario, expected = _load("dangerous_feasible")
    burn = expected.witness.impulse
    for limits in (replace(scenario.limits, dv_max_mps=0),
                   replace(scenario.limits, t_earliest_s=1)):
        result = search_maneuver(replace(scenario, limits=limits),
                                 candidates=(burn,))
        assert result.status is models.Status.NO_FEASIBLE_CANDIDATE
        assert result.after is None and not result.candidates[0].admissible


def test_budget_and_window_boundaries_are_inclusive():
    """Импульс ровно на пределах Δv и окна остаётся допустимым."""
    scenario, expected = _load("dangerous_feasible")
    burn = expected.witness.impulse
    for limits in (replace(scenario.limits, dv_max_mps=burn.dv_mps),
                   replace(scenario.limits, t_earliest_s=burn.t_s,
                           t_latest_s=burn.t_s)):
        result = search_maneuver(replace(scenario, limits=limits),
                                 candidates=(burn,))
        assert result.status is models.Status.FEASIBLE
        assert result.candidates[0].admissible


def test_late_warning_has_no_solution_in_nonempty_grid():
    """Опасный поздний сценарий отвергает двенадцать проверенных импульсов."""
    scenario, _ = _load("late_warning_no_solution")
    baseline = propagate(
        scenario.satellite, scenario.epoch, (0, scenario.horizon_s),
        scenario.forces, numerics=scenario.numerics)
    candidates = build_rtn_candidates(
        baseline, times_s=(0, scenario.limits.t_latest_s),
        directions_rtn=((1, 0, 0), (-1, 0, 0), (0, 1, 0),
                        (0, -1, 0), (0, 0, 1), (0, 0, -1)),
        magnitudes_mps=(scenario.limits.dv_max_mps,))
    assert len(candidates) == 12
    result = search_maneuver(scenario, candidates=candidates)
    assert result.status is models.Status.NO_FEASIBLE_CANDIDATE
    assert result.after is None and len(result.candidates) == 12
    assert all(not candidate.admissible for candidate in result.candidates)
    assert all(candidate.after is not None for candidate in result.candidates)


def test_tighter_numerics_preserve_status_and_minimum():
    """Ужесточение допусков и шага сохраняет вывод о безопасности."""
    scenario, expected = _load("dangerous_feasible")
    burn = expected.witness.impulse
    base = search_maneuver(scenario, candidates=(burn,))
    n = scenario.numerics
    tighter = replace(n, rtol=n.rtol / 10,
                      position_atol_m=n.position_atol_m / 10,
                      velocity_atol_mps=n.velocity_atol_mps / 10,
                      max_step_s=n.max_step_s / 2)
    refined = search_maneuver(replace(scenario, numerics=tighter),
                              candidates=(burn,))
    assert base.status is refined.status is models.Status.FEASIBLE
    assert base.after.closest.distance_m == pytest.approx(
        refined.after.closest.distance_m, abs=0.1)
    assert base.after.closest.t_s == pytest.approx(
        refined.after.closest.t_s, abs=0.01)


def test_export_records_inputs_without_invented_fuel():
    """Отчёт хранит тот же импульс и не заявляет расход топлива."""
    scenario, expected = _load("dangerous_feasible")
    candidates = _candidates(expected)
    result = search_maneuver(scenario, candidates=candidates)
    report = json.loads(export_result_json(scenario, result,
                                            candidates=candidates))
    assert report["metrics"]["status"] == "feasible"
    assert report["candidates"][0]["dv_eci_mps"] == list(
        candidates[0].dv_eci_mps)
    assert "fuel_kg" not in report["metrics"]
