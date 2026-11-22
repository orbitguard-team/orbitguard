"""Заглушки тестов: Сопротивление атмосферы, OG-14.

Каждая заглушка описывает проверку и пропускается до готовности задачи.
Эталон: invariants.circular_decay_rate_mps.
"""

import pytest

import stubs


@stubs.pending("OG-14")
def test_circular_decay_matches_analytic():
    """При постоянной плотности da/dt совпадает с −ρ·B·√(μ·a)."""
    pytest.fail(stubs.NOT_WRITTEN)


@stubs.pending("OG-14")
def test_corotating_object_has_no_drag():
    """Объект, движущийся вместе с атмосферой, не тормозится."""
    pytest.fail(stubs.NOT_WRITTEN)


@stubs.pending("OG-14")
def test_energy_decreases_monotonically():
    """Энергия орбиты монотонно убывает."""
    pytest.fail(stubs.NOT_WRITTEN)


@stubs.pending("OG-14")
def test_density_scale_scales_accel():
    """Множитель плотности линейно масштабирует ускорение."""
    pytest.fail(stubs.NOT_WRITTEN)


@stubs.pending("OG-14")
def test_missing_mass_or_area_reported():
    """Без массы, площади или cd сила не считается и это есть в warnings."""
    pytest.fail(stubs.NOT_WRITTEN)


@stubs.pending("OG-14")
def test_altitude_uses_ellipsoid():
    """Высота над эллипсоидом: на полюсе и экваторе разная при одном радиусе."""
    pytest.fail(stubs.NOT_WRITTEN)
