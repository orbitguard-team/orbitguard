"""Заглушки тестов: Сценарии, OG-06 (#6).

Каждая заглушка описывает проверку и пропускается до готовности задачи.
"""

import pytest

import stubs


@stubs.pending("OG-06 (#6)")
def test_all_scenarios_load():
    """Все файлы scenarios/*.json загружаются."""
    pytest.fail(stubs.NOT_WRITTEN)


@stubs.pending("OG-06 (#6)")
def test_synthetic_scenarios_have_source():
    """У синтетических сценариев заполнен source."""
    pytest.fail(stubs.NOT_WRITTEN)


@stubs.pending("OG-06 (#6)")
def test_expected_status_matches_result():
    """Фактический статус совпадает с ожидаемым в сценарии."""
    pytest.fail(stubs.NOT_WRITTEN)


@stubs.pending("OG-06 (#6)")
def test_invalid_scenario_raises():
    """Отрицательная дистанция, нет даты или неизвестная сила — ValueError."""
    pytest.fail(stubs.NOT_WRITTEN)


@stubs.pending("OG-06 (#6)")
def test_elements_and_state_inputs_agree():
    """Сценарий в элементах и в положении-скорости дают одно состояние."""
    pytest.fail(stubs.NOT_WRITTEN)
