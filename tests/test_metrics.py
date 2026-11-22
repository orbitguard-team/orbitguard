"""Заглушки тестов: Метрики и экспорт, OG-05 (#5).

Каждая заглушка описывает проверку и пропускается до готовности задачи.
"""

import pytest

import stubs


@stubs.pending("OG-05 (#5)")
def test_summary_has_before_and_after():
    """Есть дистанция и время сближения до и после, Δv и момент импульса."""
    pytest.fail(stubs.NOT_WRITTEN)


@stubs.pending("OG-05 (#5)")
def test_export_roundtrip_reproduces_run():
    """Из экспорта можно повторить расчёт с тем же результатом."""
    pytest.fail(stubs.NOT_WRITTEN)


@stubs.pending("OG-05 (#5)")
def test_units_present_in_export():
    """Все числовые поля экспорта с единицами."""
    pytest.fail(stubs.NOT_WRITTEN)


@stubs.pending("OG-05 (#5)")
def test_fuel_only_with_mass_and_isp():
    """Расход топлива есть только при заданных массе и Isp."""
    pytest.fail(stubs.NOT_WRITTEN)


@stubs.pending("OG-05 (#5)")
def test_forces_recorded_in_export():
    """В экспорте записано, какие силы были включены."""
    pytest.fail(stubs.NOT_WRITTEN)
