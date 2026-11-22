"""Проверки OG-03 на аналитических траекториях."""

import math
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import numpy as np

from orbitguard.conjunction import find_closest_approach
from orbitguard.models import Numerics


EPOCH = datetime(2026, 11, 22, tzinfo=timezone.utc)


class AnalyticTrajectory:
    """Траектория, заданная аналитическими функциями положения и скорости."""

    frame = "ECI-J2000"
    epoch = EPOCH

    def __init__(self, times, position, velocity, cross_track=0.0):
        self.t_s = np.asarray(times, dtype=float)
        self.position = position
        self.velocity = velocity
        self.cross_track = cross_track

    def state_at(self, time: float, *, side: str = "right") -> np.ndarray:
        """Состояние в момент времени.

        Args:
            time: Момент, с от epoch.
            side: Сторона импульса.

        Returns:
            Состояние (6,), м и м/с.
        """
        return np.array([
            self.position(time), self.cross_track, 0,
            self.velocity(time, side), 0, 0,
        ])


class ConjunctionTests(unittest.TestCase):
    """Поиск сближения на аналитических траекториях."""

    def setUp(self):
        self.stationary = AnalyticTrajectory(
            [0, 10], lambda _: 0, lambda _t, _side: 0
        )

    def test_approach_between_knots_and_threshold(self):
        moving = AnalyticTrajectory(
            [0, 10], lambda t: 2 * (t - 4.17), lambda _t, _side: 2, 3
        )
        result = find_closest_approach(moving, self.stationary, 4, (0, 10))
        self.assertAlmostEqual(result.closest.t_s, 4.17, delta=1e-3)
        self.assertAlmostEqual(result.closest.distance_m, 3, places=6)
        self.assertEqual(result.closest.rel_speed_mps, 2)
        self.assertTrue(result.violated)
        result = find_closest_approach(moving, self.stationary, 3, (0, 10))
        self.assertFalse(result.violated)

    def test_multiple_minima_and_search_grid_convergence(self):
        trajectory = AnalyticTrajectory(
            [0, 2, 5, 10], lambda t: 4 + math.cos(t),
            lambda t, _side: -math.sin(t),
        )
        numerics = Numerics(max_step_s=2)
        coarse = find_closest_approach(
            trajectory, self.stationary, 1, (0, 10), numerics=numerics
        )
        dense = find_closest_approach(
            trajectory, self.stationary, 1, (0, 10),
            numerics=replace(numerics, max_step_s=1, closest_time_tol_s=1e-4),
        )
        self.assertEqual(len(coarse.all_minima), 4)
        self.assertEqual(len(dense.all_minima), 4)
        for result in (coarse, dense):
            expected_times = (0, math.pi, 3 * math.pi, 10)
            for candidate, expected in zip(result.all_minima, expected_times):
                self.assertAlmostEqual(candidate.t_s, expected, delta=1e-3)
        for coarse_minimum, dense_minimum in zip(coarse.all_minima,
                                                 dense.all_minima):
            self.assertAlmostEqual(coarse_minimum.t_s, dense_minimum.t_s,
                                   delta=0.01)
            self.assertAlmostEqual(coarse_minimum.distance_m,
                                   dense_minimum.distance_m, delta=0.1)

    def test_boundary_minimum_and_constant_distance(self):
        for slope, expected in ((-1, 8), (0, 2), (1, 2)):
            with self.subTest(slope=slope):
                moving = AnalyticTrajectory(
                    [0, 10], lambda t, k=slope: 20 + k * t,
                    lambda _t, _side, k=slope: k,
                )
                result = find_closest_approach(moving, self.stationary, 1,
                                               (2, 8))
                self.assertEqual(result.closest.t_s, expected)
                self.assertEqual([item.t_s for item in result.all_minima],
                                 [2, 8])

    def test_rejects_incompatible_trajectories_and_partial_coverage(self):
        moving = AnalyticTrajectory([2, 12], lambda t: t - 5,
                                    lambda _t, _side: 1)
        for interval in ((0, 10), (2, 12), (8, 2), (2, 2)):
            with self.subTest(interval=interval), self.assertRaises(ValueError):
                find_closest_approach(moving, self.stationary, 1, interval)
        result = find_closest_approach(moving, self.stationary, 1, (2, 10))
        self.assertAlmostEqual(result.closest.t_s, 5, delta=1e-3)
        moving.epoch += timedelta(seconds=1)
        with self.assertRaises(ValueError):
            find_closest_approach(moving, self.stationary, 1, (2, 10))
        moving.epoch = EPOCH
        moving.frame = "ECEF"
        with self.assertRaises(ValueError):
            find_closest_approach(moving, self.stationary, 1, (2, 10))

    def test_impulse_minimum_at_exact_knot(self):
        def velocity(t: float, side: str) -> float:
            """Скорость со скачком в момент импульса 4.2 с.

            Args:
                t: Момент, с.
                side: Сторона импульса.

            Returns:
                Скорость, м/с.
            """
            return -1 if t < 4.2 or (t == 4.2 and side == "left") else 2

        moving = AnalyticTrajectory(
            [0, 4.2, 10],
            lambda t: 2 + (4.2 - t if t < 4.2 else 2 * (t - 4.2)),
            velocity,
        )
        result = find_closest_approach(moving, self.stationary, 3, (0, 10))
        self.assertEqual(result.closest.t_s, 4.2)
        self.assertEqual(result.closest.distance_m, 2)
        self.assertEqual(result.closest.rel_speed_mps, 2)
