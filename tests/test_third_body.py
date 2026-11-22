"""Заглушки тестов: Притяжение Луны и Солнца, OG-15.

Каждая заглушка описывает проверку и пропускается до готовности задачи.
Эталон положений хранится в репозитории, без сети.
"""

import pytest

import stubs


@stubs.pending("OG-15")
def test_sun_position_matches_reference():
    """Положение Солнца совпадает с опубликованным на нескольких датах."""
    pytest.fail(stubs.NOT_WRITTEN)


@stubs.pending("OG-15")
def test_moon_position_matches_reference():
    """Положение Луны совпадает с опубликованным на нескольких датах."""
    pytest.fail(stubs.NOT_WRITTEN)


@stubs.pending("OG-15")
def test_tidal_accel_zero_at_earth_center():
    """Для точки в центре Земли ускорение равно нулю."""
    pytest.fail(stubs.NOT_WRITTEN)


@stubs.pending("OG-15")
def test_leo_accel_order_of_magnitude():
    """На низкой орбите ускорение порядка 1e-7–1e-6 м/с²."""
    pytest.fail(stubs.NOT_WRITTEN)


@stubs.pending("OG-15")
def test_disabled_bodies_have_no_effect():
    """Выключенные Луна и Солнце не влияют на результат."""
    pytest.fail(stubs.NOT_WRITTEN)
