"""Заглушки тестов: Сплюснутость Земли (J2), OG-13.

Каждая заглушка описывает проверку и пропускается до готовности задачи.
Эталоны: invariants.j2_accel_mps2, invariants.j2_raan_rate_radps.
"""

import pytest

import stubs


@stubs.pending("OG-13")
def test_j2_accel_matches_reference_formula():
    """accel_j2 совпадает с эталонной формулой из PHYSICS.md."""
    pytest.fail(stubs.NOT_WRITTEN)


@stubs.pending("OG-13")
def test_raan_drift_matches_analytic():
    """Средний дрейф узла за сутки совпадает с dΩ/dt."""
    pytest.fail(stubs.NOT_WRITTEN)


@stubs.pending("OG-13")
def test_energy_with_j2_conserved():
    """Энергия с потенциалом J2 сохраняется."""
    pytest.fail(stubs.NOT_WRITTEN)


@stubs.pending("OG-13")
def test_z_angular_momentum_conserved():
    """z-компонента момента импульса сохраняется."""
    pytest.fail(stubs.NOT_WRITTEN)


@stubs.pending("OG-13")
def test_disabled_j2_matches_two_body():
    """С выключенным J2 результат совпадает с двухтельным."""
    pytest.fail(stubs.NOT_WRITTEN)
