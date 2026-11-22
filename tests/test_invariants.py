"""Проверки самих физических инвариантов и эталонных орбит.

Работают уже сейчас, без кода прогноза: эталон — аналитическая
кеплерова орбита. Если эти тесты падают, нельзя доверять проверкам
прогноза, сближения и сил, которые на них опираются.
"""

import numpy as np
import pytest

import invariants
import reference_orbits as ref

ISS_ALTITUDE_M = 420e3
ISS_INCLINATION_RAD = np.radians(51.64)
SECONDS_PER_DAY = 86400.0

# Допуски: эталон аналитический, ошибка — только округление float64.
ROUNDOFF_RTOL = 1e-12
FINITE_DIFF_STEP_M = 1.0
FINITE_DIFF_RTOL = 1e-6


def _iss_like_ellipse(t_s: np.ndarray) -> np.ndarray:
    """Состояния на слегка эллиптической МКС-подобной орбите.

    Args:
        t_s: Моменты времени (N,), с.

    Returns:
        Состояния (N, 6), м и м/с, ECI.
    """
    return ref.kepler_states_eci(
        sma_m=ref.R_EARTH_M + ISS_ALTITUDE_M, ecc=0.01,
        inclination_rad=ISS_INCLINATION_RAD, raan_rad=0.3, argp_rad=1.1,
        mean_anomaly0_rad=0.2, t_s=t_s)


def test_circular_orbit_energy_equals_minus_mu_over_2a():
    """Энергия круговой орбиты равна −μ/(2a)."""
    sma_m = ref.R_EARTH_M + ISS_ALTITUDE_M
    states = ref.kepler_states_eci(sma_m, 0.0, ISS_INCLINATION_RAD, 0.0,
                                   0.0, 0.0, np.linspace(0.0, 6000.0, 50))
    energy = invariants.specific_energy_m2ps2(states, ref.MU_EARTH_M3PS2)
    np.testing.assert_allclose(energy, -ref.MU_EARTH_M3PS2 / (2.0 * sma_m),
                               rtol=ROUNDOFF_RTOL)


def test_energy_and_momentum_constant_on_kepler_ellipse():
    """На точной кеплеровой орбите энергия и момент импульса постоянны."""
    states = _iss_like_ellipse(np.linspace(0.0, SECONDS_PER_DAY, 500))
    energy = invariants.specific_energy_m2ps2(states, ref.MU_EARTH_M3PS2)
    momentum = invariants.specific_angular_momentum_m2ps(states)
    assert invariants.max_relative_drift(energy) < ROUNDOFF_RTOL
    assert invariants.max_relative_drift(momentum) < ROUNDOFF_RTOL


def test_kepler_orbit_returns_after_one_period():
    """Через один период объект возвращается в исходное состояние."""
    sma_m = ref.R_EARTH_M + ISS_ALTITUDE_M
    period_s = ref.orbital_period_s(sma_m)
    states = _iss_like_ellipse(np.array([0.0, period_s]))
    np.testing.assert_allclose(states[1], states[0], rtol=1e-9, atol=1e-6)


def test_iss_like_period_is_about_92_minutes():
    """Период МКС-подобной орбиты — около 92–93 минут."""
    period_min = ref.orbital_period_s(ref.R_EARTH_M + ISS_ALTITUDE_M) / 60.0
    assert 92.0 < period_min < 93.5


def test_drift_detector_catches_perturbation():
    """Проверка дрейфа замечает искусственно добавленное отклонение."""
    states = _iss_like_ellipse(np.linspace(0.0, 6000.0, 100))
    states[-1, 3:] *= 1.0 + 1e-6
    energy = invariants.specific_energy_m2ps2(states, ref.MU_EARTH_M3PS2)
    assert invariants.max_relative_drift(energy) > 1e-7


def test_drift_of_zero_initial_value_raises():
    """Относительный дрейф от нуля не определён."""
    with pytest.raises(ValueError):
        invariants.max_relative_drift(np.array([0.0, 1.0]))


def test_j2_accel_is_minus_gradient_of_j2_potential():
    """Формула ускорения J2 согласована с потенциалом J2."""
    r_eci_m = np.array([[4.0e6, -3.0e6, 4.5e6], [6.8e6, 0.0, 0.0],
                        [1.0e6, 2.0e6, -6.4e6]])
    accel = invariants.j2_accel_mps2(r_eci_m, ref.MU_EARTH_M3PS2, ref.J2,
                                     ref.R_EARTH_GRAV_M)
    gradient = np.zeros_like(r_eci_m)
    for axis in range(3):
        step = np.zeros(3)
        step[axis] = FINITE_DIFF_STEP_M
        plus = invariants.j2_potential_energy_m2ps2(
            r_eci_m + step, ref.MU_EARTH_M3PS2, ref.J2, ref.R_EARTH_GRAV_M)
        minus = invariants.j2_potential_energy_m2ps2(
            r_eci_m - step, ref.MU_EARTH_M3PS2, ref.J2, ref.R_EARTH_GRAV_M)
        gradient[:, axis] = (plus - minus) / (2.0 * FINITE_DIFF_STEP_M)
    np.testing.assert_allclose(accel, -gradient, rtol=FINITE_DIFF_RTOL)


def test_j2_raan_drift_for_iss_like_orbit_is_about_minus_5_deg_per_day():
    """Дрейф узла МКС-подобной орбиты — около −5° в сутки."""
    rate_radps = invariants.j2_raan_rate_radps(
        ref.R_EARTH_M + ISS_ALTITUDE_M, 0.0, ISS_INCLINATION_RAD,
        ref.MU_EARTH_M3PS2, ref.J2, ref.R_EARTH_GRAV_M)
    rate_deg_per_day = np.degrees(rate_radps) * SECONDS_PER_DAY
    assert -5.2 < rate_deg_per_day < -4.8


def test_circular_decay_rate_is_negative_and_scales_with_density():
    """Атмосфера снижает орбиту; скорость снижения пропорциональна ρ."""
    sma_m = ref.R_EARTH_M + ISS_ALTITUDE_M
    slow = invariants.circular_decay_rate_mps(sma_m, 1e-12, 0.01,
                                              ref.MU_EARTH_M3PS2)
    fast = invariants.circular_decay_rate_mps(sma_m, 2e-12, 0.01,
                                              ref.MU_EARTH_M3PS2)
    assert slow < 0.0
    assert fast == pytest.approx(2.0 * slow)
