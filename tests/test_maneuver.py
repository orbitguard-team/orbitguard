"""Заглушки тестов: Подбор манёвра, OG-04 (#4).

Каждая заглушка описывает проверку и пропускается до готовности задачи.
"""

import pytest

import stubs


@stubs.pending("OG-04 (#4)")
def test_already_safe_returns_zero_dv():
    """Безопасный сценарий: статус already_safe, манёвра нет."""
    pytest.fail(stubs.NOT_WRITTEN)


@stubs.pending("OG-04 (#4)")
def test_feasible_is_safe_on_whole_horizon():
    """Дистанция безопасна на всём интервале, включая новые сближения."""
    pytest.fail(stubs.NOT_WRITTEN)


@stubs.pending("OG-04 (#4)")
def test_dv_within_limit():
    """|Δv| не больше dv_max_mps."""
    pytest.fail(stubs.NOT_WRITTEN)


@stubs.pending("OG-04 (#4)")
def test_burn_time_within_window():
    """Момент импульса в [t_earliest_s, t_latest_s] и раньше сближения."""
    pytest.fail(stubs.NOT_WRITTEN)


@stubs.pending("OG-04 (#4)")
def test_no_feasible_candidate_status():
    """При слишком малом dv_max — статус no_feasible_candidate."""
    pytest.fail(stubs.NOT_WRITTEN)


@stubs.pending("OG-04 (#4)")
def test_best_has_min_dv_among_feasible():
    """Выбранный манёвр — минимальный Δv среди допустимых кандидатов."""
    pytest.fail(stubs.NOT_WRITTEN)


@stubs.pending("OG-04 (#4)")
def test_tie_breaking_is_deterministic():
    """При равных Δv результат воспроизводим."""
    pytest.fail(stubs.NOT_WRITTEN)


@stubs.pending("OG-04 (#4)")
def test_dv_rtn_and_eci_consistent():
    """dv_rtn_mps и dv_eci_mps — один и тот же вектор в разных осях."""
    pytest.fail(stubs.NOT_WRITTEN)
