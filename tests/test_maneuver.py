"""Проверки поиска манёвра на аналитическом пересечении круговых орбит."""

from dataclasses import replace
from datetime import datetime, timezone

import numpy as np
import pytest

from orbitguard.constants import MU_EARTH_M3_S2
from orbitguard.maneuver import build_rtn_candidates, search_maneuver
from orbitguard.models import (
    Impulse, ManeuverLimits, Scenario, SpaceObject, Status,
)
from orbitguard.propagation import propagate


def assert_crossing(conjunction, tolerance_s):
    """Проверяет сближение около t=300 с в пределах допуска.

    Args:
        conjunction (ConjunctionResult): Результат поиска сближения.
        tolerance_s (float): Допуск по времени, с.
    """
    closest = conjunction.closest
    assert closest.t_s == pytest.approx(300, abs=tolerance_s)
    # Допуск времени переводится в расстояние через относительную скорость.
    assert closest.distance_m <= closest.rel_speed_mps * tolerance_s + 1e-3


@pytest.fixture
def crossing() -> Scenario:
    """Две круговые орбиты, пересекающиеся в t=300 с.

    Returns:
        Синтетический сценарий в ECI и СИ.
    """
    radius = 7e6
    speed = np.sqrt(MU_EARTH_M3_S2 / radius)
    phase = -300 * speed / radius
    x_dir = np.array([1.0, 0.0, 0.0])

    def object_on_circle(name, inclination):
        """Объект на круговой орбите с заданным наклоном.

        Args:
            name (str): Имя объекта.
            inclination (float): Наклон плоскости, рад.

        Returns:
            SpaceObject: Начальное состояние в ECI.
        """
        y_dir = np.array([0, np.cos(inclination), np.sin(inclination)])
        position = radius * (np.cos(phase) * x_dir + np.sin(phase) * y_dir)
        velocity = speed * (-np.sin(phase) * x_dir + np.cos(phase) * y_dir)
        return SpaceObject(name, np.r_[position, velocity])

    return Scenario(
        name="Круговые орбиты, пересечение в t=300 с",
        epoch=datetime(2026, 11, 22, tzinfo=timezone.utc),
        horizon_s=600,
        satellite=object_on_circle("sat", 0),
        debris=object_on_circle("debris", 0.005),
        safe_distance_m=100,
        limits=ManeuverLimits(5, 0, 400),
    )


def test_chooses_smallest_impulse_then_earliest_time(crossing):
    early = Impulse(0, [0, 1, 0])
    same_time = Impulse(0, [0, -1, 0])
    late = Impulse(20, [0, 1, 0])
    bigger = Impulse(0, [0, 2, 0])
    candidates = (late, early, same_time, bigger)
    result = search_maneuver(crossing, candidates=candidates)
    assert result.status is Status.FEASIBLE
    assert result.best is early
    assert all(item.admissible for item in result.candidates)
    assert all(
        item.impulse is burn
        for item, burn in zip(result.candidates, candidates)
    )
    assert_crossing(result.before, crossing.numerics.closest_time_tol_s)
    assert result.after.closest.distance_m >= 101


def test_applies_impulse_and_converges_on_denser_grid(crossing):
    burn = Impulse(20, [0, 1, 0])
    span = (0, crossing.horizon_s)
    trajectory = propagate(
        crossing.satellite, crossing.epoch, span, crossing.forces,
        numerics=crossing.numerics, impulses=(burn,),
    )
    left = trajectory.state_at(burn.t_s, side="left")
    right = trajectory.state_at(burn.t_s, side="right")
    np.testing.assert_array_equal(right[:3], left[:3])
    np.testing.assert_allclose(right[3:] - left[3:], burn.dv_eci_mps)

    nominal = search_maneuver(crossing, candidates=(burn,))
    numerics = crossing.numerics
    refined = replace(
        numerics, rtol=numerics.rtol / 10, max_step_s=numerics.max_step_s / 2,
        position_atol_m=numerics.position_atol_m / 10,
        velocity_atol_mps=numerics.velocity_atol_mps / 10,
    )
    refined_scenario = replace(crossing, numerics=refined)
    dense = search_maneuver(refined_scenario, candidates=(burn,))
    assert nominal.status is dense.status is Status.FEASIBLE
    assert nominal.after.closest.distance_m == pytest.approx(
        dense.after.closest.distance_m, abs=0.1,
    )
    assert nominal.after.closest.t_s == pytest.approx(
        dense.after.closest.t_s, abs=0.01,
    )


def test_late_impulse_cannot_remove_past_approach(crossing):
    too_late = Impulse(350, [0, 4, 0])
    result = search_maneuver(crossing, candidates=(too_late,))
    assert result.status is Status.NO_FEASIBLE_CANDIDATE
    assert result.best is result.after is None
    candidate = result.candidates[0]
    assert not candidate.admissible
    assert_crossing(candidate.after, crossing.numerics.closest_time_tol_s)


def test_early_impulse_that_keeps_encounter_unsafe_is_rejected(crossing):
    weak = Impulse(20, [0, 0.1, 0])
    result = search_maneuver(crossing, candidates=(weak,))
    assert result.status is Status.NO_FEASIBLE_CANDIDATE
    candidate = result.candidates[0]
    assert candidate.after.closest.t_s > weak.t_s
    assert candidate.after.violated and not candidate.admissible


def test_out_of_limits_and_empty_search(crossing):
    result = search_maneuver(
        crossing, candidates=(Impulse(0, [0, 6, 0]), Impulse(401, [0, 1, 0])),
    )
    assert result.status is Status.NO_FEASIBLE_CANDIDATE
    assert [(item.admissible, item.after) for item in result.candidates] == [
        (False, None), (False, None),
    ]
    assert "Δv" in result.candidates[0].reason
    assert "окна" in result.candidates[1].reason
    empty = search_maneuver(crossing, candidates=())
    assert empty.status is Status.NO_FEASIBLE_CANDIDATE and not empty.candidates


def test_physical_failure_does_not_interrupt_remaining_candidates(crossing):
    scenario = replace(crossing, limits=ManeuverLimits(5500, 0, 400))
    unsafe = Impulse(0, [-5000, 0, 0])
    valid = Impulse(0, [0, 1, 0])
    result = search_maneuver(scenario, candidates=(unsafe, valid))
    assert result.status is Status.FEASIBLE and result.best is valid
    failed = result.candidates[0]
    assert not failed.admissible and failed.after is None
    assert failed.reason.startswith("Сбой распространения:")


def test_failed_baseline_has_distinct_status(crossing):
    falling = SpaceObject("falling", [7e6, 0, 0, -5000, 7546, 0])
    result = search_maneuver(
        replace(crossing, satellite=falling), candidates=(),
    )
    assert result.status is Status.PROPAGATION_FAILED
    assert result.before is result.after is result.best is None
    assert result.message.startswith("Сбой распространения:")


def test_already_safe_requires_configured_margin(crossing):
    distant = replace(crossing, horizon_s=60, limits=ManeuverLimits(5, 0, 60))
    result = search_maneuver(distant, candidates=())
    assert result.status is Status.ALREADY_SAFE
    assert result.before is result.after and result.best is None

    near_threshold = replace(
        distant, safe_distance_m=result.before.closest.distance_m,
    )
    within_margin = search_maneuver(near_threshold, candidates=())
    assert within_margin.status is Status.NO_FEASIBLE_CANDIDATE
    assert not within_margin.before.violated


def test_grid_expands_rtn_axes_in_stable_order(crossing):
    trajectory = propagate(
        crossing.satellite, crossing.epoch, (0, 60), crossing.forces,
    )
    candidates = build_rtn_candidates(
        trajectory, times_s=(0, 20), magnitudes_mps=(1, 2),
        directions_rtn=([2, 0, 0], [0, 0, -1]),
    )
    assert [burn.t_s for burn in candidates] == [0] * 4 + [20] * 4
    assert [burn.dv_mps for burn in candidates] == pytest.approx(
        [1, 2, 1, 2] * 2
    )
    initial_radial = crossing.satellite.state0[:3] / 7e6
    np.testing.assert_allclose(candidates[0].dv_eci_mps, initial_radial)
    np.testing.assert_allclose(candidates[3].dv_eci_mps, [0, 0, -2])
