"""Заглушки тестов: Прогноз движения, OG-02 (#2).

Каждая заглушка описывает проверку и пропускается до готовности задачи.
Эталоны: tests/reference_orbits.py, проверки: tests/invariants.py.
"""

import pytest

import stubs


@stubs.pending("OG-02 (#2)")
def test_circular_orbit_period_matches_kepler():
    """Круговая орбита возвращается в начало через 2π·√(a³/μ)."""
    pytest.fail(stubs.NOT_WRITTEN)


@stubs.pending("OG-02 (#2)")
def test_matches_analytic_kepler_ellipse():
    """Без возмущений траектория совпадает с kepler_states_eci на сутках."""
    pytest.fail(stubs.NOT_WRITTEN)


@stubs.pending("OG-02 (#2)")
def test_energy_conserved_two_body():
    """Дрейф энергии меньше допуска из OG-01."""
    pytest.fail(stubs.NOT_WRITTEN)


@stubs.pending("OG-02 (#2)")
def test_angular_momentum_conserved_two_body():
    """Дрейф вектора момента импульса меньше допуска."""
    pytest.fail(stubs.NOT_WRITTEN)


@stubs.pending("OG-02 (#2)")
def test_converges_with_tighter_tolerance():
    """При ужесточении rtol положение через горизонт сходится."""
    pytest.fail(stubs.NOT_WRITTEN)


@stubs.pending("OG-02 (#2)")
def test_forward_then_backward_returns_to_start():
    """Прогноз вперёд и обратно возвращает начальное состояние."""
    pytest.fail(stubs.NOT_WRITTEN)


@stubs.pending("OG-02 (#2)")
def test_rotation_of_initial_state_rotates_trajectory():
    """Поворот начального состояния поворачивает всю траекторию."""
    pytest.fail(stubs.NOT_WRITTEN)


@stubs.pending("OG-02 (#2)")
def test_state_at_matches_grid_points():
    """state_at в узлах сетки совпадает с states_eci."""
    pytest.fail(stubs.NOT_WRITTEN)


@stubs.pending("OG-02 (#2)")
def test_impulse_keeps_position_continuous():
    """В момент импульса положение не прыгает."""
    pytest.fail(stubs.NOT_WRITTEN)


@stubs.pending("OG-02 (#2)")
def test_impulse_changes_velocity_by_dv():
    """Скорость меняется ровно на dv_eci_mps."""
    pytest.fail(stubs.NOT_WRITTEN)


@stubs.pending("OG-02 (#2)")
def test_zero_impulse_matches_no_impulse():
    """Нулевой импульс не меняет траекторию."""
    pytest.fail(stubs.NOT_WRITTEN)


@stubs.pending("OG-02 (#2)")
def test_invalid_state_raises():
    """NaN, нулевой радиус и неверная форма массива дают ValueError."""
    pytest.fail(stubs.NOT_WRITTEN)


@stubs.pending("OG-02 (#2)")
def test_reentry_stops_with_status():
    """Ниже REENTRY_ALTITUDE_M расчёт останавливается со статусом."""
    pytest.fail(stubs.NOT_WRITTEN)


@stubs.pending("OG-02 (#2)")
def test_repeat_run_is_identical():
    """Повторный запуск с теми же входными данными даёт тот же результат."""
    pytest.fail(stubs.NOT_WRITTEN)
