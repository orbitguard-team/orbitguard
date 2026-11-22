"""Заглушки тестов: Поиск сближения, OG-03 (#3).

Каждая заглушка описывает проверку и пропускается до готовности задачи.
До готовности прогноза — на прямолинейном движении r(t) = r0 + v·t.
"""

import pytest

import stubs


@stubs.pending("OG-03 (#3)")
def test_straight_lines_match_analytic_minimum():
    """Для прямых t* = −(Δr·Δv)/|Δv|² и d* известны точно."""
    pytest.fail(stubs.NOT_WRITTEN)


@stubs.pending("OG-03 (#3)")
def test_minimum_between_grid_points_refined():
    """Минимум между узлами сетки уточняется, а не берётся по сетке."""
    pytest.fail(stubs.NOT_WRITTEN)


@stubs.pending("OG-03 (#3)")
def test_multiple_minima_all_found():
    """Находятся все локальные минимумы на интервале."""
    pytest.fail(stubs.NOT_WRITTEN)


@stubs.pending("OG-03 (#3)")
def test_minimum_at_interval_boundary():
    """Минимум на краю интервала находится и помечается."""
    pytest.fail(stubs.NOT_WRITTEN)


@stubs.pending("OG-03 (#3)")
def test_result_symmetric_in_objects():
    """Перестановка объектов не меняет время и дистанцию."""
    pytest.fail(stubs.NOT_WRITTEN)


@stubs.pending("OG-03 (#3)")
def test_violated_flag_matches_safe_distance():
    """violated = дистанция меньше безопасной."""
    pytest.fail(stubs.NOT_WRITTEN)


@stubs.pending("OG-03 (#3)")
def test_rel_speed_reported():
    """Относительная скорость в момент сближения совпадает с |Δv|."""
    pytest.fail(stubs.NOT_WRITTEN)
