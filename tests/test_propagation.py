"""Независимые аналитические эталоны и границы контракта OG-02."""

import dataclasses
import datetime
import types

import numpy as np
import pytest

from orbitguard import constants, models, propagation
from orbitguard.api import propagate


RADIUS_M = 7_000_000.0
MU = constants.MU_EARTH_M3_S2
SPEED_MPS = np.sqrt(MU / RADIUS_M)
PERIOD_S = 2 * np.pi * np.sqrt(RADIUS_M**3 / MU)
EPOCH = datetime.datetime(2026, 11, 22, tzinfo=datetime.timezone.utc)


def circular_object():
    """Объект на круговой орбите радиуса RADIUS_M.

    Returns:
        models.SpaceObject: Начальное состояние в ECI.
    """
    return models.SpaceObject(
        "circle", [RADIUS_M, 0, 0, 0, SPEED_MPS, 0]
    )


def analytical_state(t_s):
    """Аналитическое состояние круговой орбиты.

    Args:
        t_s (float): Время от epoch, с.

    Returns:
        np.ndarray: Состояние (6,) в м и м/с.
    """
    angle = SPEED_MPS * t_s / RADIUS_M
    return np.array([
        RADIUS_M * np.cos(angle), RADIUS_M * np.sin(angle), 0,
        -SPEED_MPS * np.sin(angle), SPEED_MPS * np.cos(angle), 0,
    ])


@pytest.fixture(scope="module")
def circle():
    """Численная траектория на один период.

    Returns:
        IntegratedTrajectory: Траектория RK45 круговой орбиты.
    """
    return propagate(
        circular_object(), EPOCH, (0, PERIOD_S), models.ForceModel()
    )


def test_full_period_and_invariants_match_kepler(circle):
    final = circle.state_at(PERIOD_S)
    initial = circular_object().state0
    assert np.linalg.norm(final[:3] - initial[:3]) < 1.0
    assert np.linalg.norm(final[3:] - initial[3:]) < 0.001
    positions, velocities = circle.states[:, :3], circle.states[:, 3:]
    energy = (np.sum(velocities**2, axis=1) / 2
              - MU / np.linalg.norm(positions, axis=1))
    momentum = np.cross(positions, velocities)
    energy_drift = np.max(np.abs((energy - energy[0]) / energy[0]))
    momentum_drift = np.max(np.linalg.norm(
        momentum - momentum[0], axis=1
    )) / np.linalg.norm(momentum[0])
    assert energy_drift < 1e-8
    assert momentum_drift < 1e-8


def test_dense_output_matches_analytic_between_grid_nodes(circle):
    for t_s in (0.123, 123.456, PERIOD_S / 3.7, PERIOD_S - 0.12):
        actual, expected = circle.state_at(t_s), analytical_state(t_s)
        assert np.linalg.norm(actual[:3] - expected[:3]) < 1.0
        assert np.linalg.norm(actual[3:] - expected[3:]) < 0.001
    assert circle.states.shape == (len(circle.t_s), 6)
    assert np.all(np.diff(circle.t_s) > 0)
    assert circle.epoch == EPOCH and circle.frame == "ECI-J2000"
    changed = circle.state_at(0)
    changed[:] = 0
    assert circle.state_at(0)[0] == RADIUS_M


def test_tighter_tolerances_converge_to_analytic_solution(circle):
    tighter = models.Numerics(
        rtol=1e-11, position_atol_m=1e-4,
        velocity_atol_mps=1e-7, max_step_s=5,
    )
    refined = propagate(
        circular_object(), EPOCH, (0, PERIOD_S), models.ForceModel(),
        numerics=tighter,
    )
    sample = np.linspace(0, PERIOD_S, 101)
    base = np.vstack([circle.state_at(float(t)) for t in sample])
    fine = np.vstack([refined.state_at(float(t)) for t in sample])
    exact = np.vstack([analytical_state(t) for t in sample])
    base_error = np.max(np.linalg.norm(base[:, :3] - exact[:, :3], axis=1))
    fine_error = np.max(np.linalg.norm(fine[:, :3] - exact[:, :3], axis=1))
    assert fine_error < base_error
    assert np.max(np.linalg.norm(base[:, :3] - fine[:, :3], axis=1)) < 0.1


def test_elliptical_orbit_reaches_analytic_apogee():
    eccentricity = 0.1
    sma_m = 8_000_000.0
    peri_m = sma_m * (1 - eccentricity)
    apo_m = sma_m * (1 + eccentricity)
    peri_speed = np.sqrt(MU * (2 / peri_m - 1 / sma_m))
    apo_speed = np.sqrt(MU * (2 / apo_m - 1 / sma_m))
    obj = models.SpaceObject("ellipse", [peri_m, 0, 0, 0, peri_speed, 0])
    half_period_s = np.pi * np.sqrt(sma_m**3 / MU)
    orbit = propagate(obj, EPOCH, (0, half_period_s), models.ForceModel())
    state = orbit.state_at(half_period_s)
    assert np.linalg.norm(state[:3] - [-apo_m, 0, 0]) < 1
    assert np.linalg.norm(state[3:] - [0, -apo_speed, 0]) < 0.001


@pytest.mark.parametrize("burn_s", [0.0, 31.234, 100.0])
def test_impulse_has_continuous_position_and_velocity_jump(burn_s):
    burn = models.Impulse(burn_s, [0, 2, 0.5])
    obj = circular_object()
    orbit = propagate(
        obj, EPOCH, (0, 100), models.ForceModel(), impulses=[burn]
    )
    before = orbit.state_at(burn_s, side="left")
    after = orbit.state_at(burn_s, side="right")
    np.testing.assert_array_equal(before[:3], after[:3])
    np.testing.assert_allclose(after[3:] - before[3:], burn.dv_eci_mps)
    assert orbit.t_s[0] == 0 and orbit.t_s[-1] == 100
    np.testing.assert_array_equal(
        orbit.states[np.flatnonzero(orbit.t_s == burn_s)[0]], after
    )
    np.testing.assert_array_equal(obj.state0, circular_object().state0)
    if burn_s > 0:
        np.testing.assert_allclose(
            orbit.state_at(burn_s / 2), analytical_state(burn_s / 2),
            rtol=0, atol=0.001,
        )
    if burn_s < 100:
        # Независимый перезапуск с новым начальным состоянием после импульса.
        restarted = propagate(
            models.SpaceObject("restart", after),
            EPOCH + datetime.timedelta(seconds=burn_s),
            (0, 100 - burn_s), models.ForceModel(),
        )
        np.testing.assert_allclose(
            orbit.state_at(100), restarted.state_at(100 - burn_s),
            rtol=0, atol=0.001,
        )


@pytest.mark.parametrize("t_s", [-1, PERIOD_S + 1, float("nan"), [1, 2]])
def test_no_extrapolation_or_vector_time(circle, t_s):
    with pytest.raises(ValueError):
        circle.state_at(t_s)


def test_unknown_side_is_rejected(circle):
    with pytest.raises(ValueError):
        circle.state_at(0, side="middle")


@pytest.mark.parametrize("interval", [(1, 10), (0, 0), (0, -1), (0, np.inf)])
def test_invalid_intervals_are_rejected(interval):
    with pytest.raises(ValueError):
        propagate(circular_object(), EPOCH, interval, models.ForceModel())


def test_naive_epoch_and_invalid_impulses_are_rejected():
    obj = circular_object()
    with pytest.raises(ValueError):
        propagate(obj, EPOCH.replace(tzinfo=None), (0, 10), models.ForceModel())
    for burns in ([models.Impulse(11, [0, 0, 0])],
                  [models.Impulse(1, [0, 0, 0])] * 2):
        with pytest.raises(ValueError):
            propagate(obj, EPOCH, (0, 10), models.ForceModel(), impulses=burns)


@pytest.mark.parametrize("force", ["j2", "drag", "moon", "sun"])
def test_unimplemented_force_is_not_silently_ignored(force):
    with pytest.raises(models.UnsupportedForceError, match=force):
        propagate(
            circular_object(), EPOCH, (0, 10),
            dataclasses.replace(models.ForceModel(), **{force: True}),
        )


def test_falling_object_stops_at_altitude_boundary():
    floor = constants.EARTH_RADIUS_M + constants.MIN_ALTITUDE_M
    obj = models.SpaceObject("fall", [floor + 100, 0, 0, -100, 0, 0])
    with pytest.raises(models.PropagationError, match="Altitude floor"):
        propagate(obj, EPOCH, (0, 100), models.ForceModel())


def test_solver_failure_is_explicit(monkeypatch):
    def failed_solver(*_args, **_kwargs):
        """Имитирует неуспешный solve_ivp.

        Args:
            *_args: Позиционные аргументы solve_ivp, не используются.
            **_kwargs: Именованные аргументы solve_ivp, не используются.

        Returns:
            types.SimpleNamespace: Результат с success=False.
        """
        return types.SimpleNamespace(success=False, message="test failure")

    monkeypatch.setattr(propagation.integrate, "solve_ivp", failed_solver)
    with pytest.raises(models.PropagationError, match="test failure"):
        propagate(circular_object(), EPOCH, (0, 10), models.ForceModel())


def test_two_objects_can_be_evaluated_on_shared_time_grid():
    obj = circular_object()
    first = propagate(obj, EPOCH, (0, 100), models.ForceModel())
    second = propagate(
        models.SpaceObject("other", obj.state0 + [1000, 0, 0, 0, 0, 0]),
        EPOCH, (0, 100), models.ForceModel(),
    )
    np.testing.assert_array_equal(first.t_s, second.t_s)
    for time_s in (1.23, 45.67):
        assert np.linalg.norm(
            first.state_at(time_s)[:3] - second.state_at(time_s)[:3]
        ) > 0
