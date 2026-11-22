"""Воспроизводимые измерения OG-02; печатаются вычисленные значения."""

import datetime
import json

import numpy as np

from orbitguard import constants, models, propagation


def measure() -> dict[str, float]:
    """Сравнивает численную круговую орбиту с аналитическим решением.

    Returns:
        Измеренные ошибки периода, возврата и дрейфа инвариантов.
    """
    radius_m = 7_000_000.0
    mu = constants.MU_EARTH_M3_S2
    speed_mps = np.sqrt(mu / radius_m)
    period_s = 2 * np.pi * np.sqrt(radius_m**3 / mu)
    obj = models.SpaceObject("circle", [radius_m, 0, 0, 0, speed_mps, 0])
    epoch = datetime.datetime(2026, 11, 22, tzinfo=datetime.timezone.utc)
    orbit = propagation.propagate(
        obj, epoch, (0, period_s), models.ForceModel()
    )
    refined = propagation.propagate(
        obj, epoch, (0, period_s), models.ForceModel(),
        numerics=models.Numerics(
            rtol=1e-11, position_atol_m=1e-4,
            velocity_atol_mps=1e-7, max_step_s=5,
        ),
    )
    positions = orbit.states[:, :3]
    velocities = orbit.states[:, 3:]
    energy = (np.sum(velocities**2, axis=1) / 2
              - mu / np.linalg.norm(positions, axis=1))
    momentum = np.cross(positions, velocities)
    sample_s = np.linspace(0, period_s, 101)
    base = np.vstack([orbit.state_at(float(t)) for t in sample_s])
    fine = np.vstack([refined.state_at(float(t)) for t in sample_s])
    angles = speed_mps * sample_s / radius_m
    exact = np.column_stack((
        radius_m * np.cos(angles), radius_m * np.sin(angles),
        np.zeros_like(angles),
    ))
    return {
        "period_s": float(period_s),
        "return_position_error_m": float(np.linalg.norm(
            orbit.states[-1, :3] - obj.state0[:3])),
        "return_velocity_error_mps": float(np.linalg.norm(
            orbit.states[-1, 3:] - obj.state0[3:])),
        "max_relative_energy_drift": float(np.max(
            np.abs((energy - energy[0]) / energy[0]))),
        "max_relative_momentum_drift": float(np.max(np.linalg.norm(
            momentum - momentum[0], axis=1)) / np.linalg.norm(momentum[0])),
        "base_max_position_error_m": float(np.max(np.linalg.norm(
            base[:, :3] - exact, axis=1))),
        "refined_max_position_error_m": float(np.max(np.linalg.norm(
            fine[:, :3] - exact, axis=1))),
        "refinement_max_position_difference_m": float(np.max(
            np.linalg.norm(base[:, :3] - fine[:, :3], axis=1))),
    }


if __name__ == "__main__":
    print(json.dumps(measure(), indent=2))
