"""Метрики и воспроизводимость экспорта для статусов OG-04."""

import json
import unittest
from datetime import datetime, timezone

import numpy as np

from orbitguard.metrics import build_metrics, export_result_json
from orbitguard.models import (
    Approach, ConjunctionResult, ForceModel, Impulse, ManeuverLimits,
    ManeuverResult, Numerics, Scenario, SpaceObject, Status,
)


def conjunction(distance: float, time: float) -> ConjunctionResult:
    approach = Approach(time, distance, 2.0)
    return ConjunctionResult(approach, (approach,), 100.0)


class MetricsTests(unittest.TestCase):
    def test_success_reports_distances_times_and_impulse(self):
        impulse = Impulse(12.0, [3.0, 4.0, 0.0])
        result = ManeuverResult(
            Status.FEASIBLE, impulse, conjunction(30, 61), conjunction(120, 72)
        )
        metrics = build_metrics(result, compute_time_s=0.25)
        self.assertEqual(metrics["status"], "feasible")
        self.assertEqual(metrics["before"]["d_min_m"], 30)
        self.assertEqual(metrics["before"]["t_min_s"], 61)
        self.assertEqual(metrics["after"]["d_min_m"], 120)
        self.assertEqual(metrics["after"]["t_min_s"], 72)
        self.assertEqual(metrics["dv_eci_mps"], [3.0, 4.0, 0.0])
        self.assertEqual(metrics["dv_mps"], 5.0)
        self.assertEqual(metrics["impulse_t_s"], 12.0)
        self.assertEqual(metrics["compute_time_s"], 0.25)

    def test_safe_baseline_has_zero_delta_v_and_no_burn_time(self):
        safe = conjunction(200, 40)
        result = ManeuverResult(Status.ALREADY_SAFE, None, safe, safe)
        metrics = build_metrics(result, compute_time_s=0)
        self.assertEqual(metrics["before"], metrics["after"])
        self.assertEqual(metrics["dv_eci_mps"], [0.0, 0.0, 0.0])
        self.assertEqual(metrics["dv_mps"], 0)
        self.assertIsNone(metrics["impulse_t_s"])

    def test_unsuccessful_results_leave_unavailable_values_empty(self):
        for status, baseline in (
            (Status.NO_FEASIBLE_CANDIDATE, conjunction(30, 61)),
            (Status.PROPAGATION_FAILED, None),
        ):
            with self.subTest(status=status):
                result = ManeuverResult(status, None, baseline, None)
                metrics = build_metrics(result, compute_time_s=1)
                self.assertEqual(metrics["status"], status.value)
                self.assertEqual(metrics["before"] is None, baseline is None)
                for key in ("after", "dv_eci_mps", "dv_mps", "impulse_t_s"):
                    self.assertIsNone(metrics[key])

    def test_rejects_invalid_elapsed_time(self):
        result = ManeuverResult(Status.PROPAGATION_FAILED, None, None, None)
        for elapsed in (-1, float("nan"), float("inf"), True):
            with self.subTest(elapsed=elapsed), self.assertRaises(ValueError):
                build_metrics(result, compute_time_s=elapsed)

    def test_export_rebuilds_ordered_inputs_without_fabricating_fuel(self):
        scenario = Scenario(
            name="Синтетический сценарий",
            epoch=datetime(2026, 11, 22, tzinfo=timezone.utc),
            horizon_s=600,
            satellite=SpaceObject("sat", [7e6, 0, 0, 0, 7546, 0]),
            debris=SpaceObject("debris", [7e6, 1000, 0, 0, 7546, 0]),
            safe_distance_m=100,
            limits=ManeuverLimits(5, 0, 400),
            numerics=Numerics(max_step_s=5),
            source="аналитическая фикстура",
        )
        safe = conjunction(200, 40)
        result = ManeuverResult(Status.ALREADY_SAFE, None, safe, safe)
        candidates = (Impulse(90, [0, 1, 0]), Impulse(12, [3, 4, 0]))
        output = export_result_json(
            scenario, result, candidates=candidates, compute_time_s=np.float64(0.1)
        )
        self.assertIn(scenario.name, output)
        report = json.loads(output)
        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["units"]["velocity"], "m/s")
        self.assertEqual(report["metrics"]["status"], "already_safe")
        self.assertNotIn("fuel_kg", report["metrics"])
        inputs = report["scenario"]
        self.assertEqual(inputs.pop("schema_version"), 1)
        self.assertTrue(inputs["epoch"].endswith("Z"))
        inputs["epoch"] = datetime.fromisoformat(inputs["epoch"])
        for field in ("satellite", "debris"):
            inputs[field] = SpaceObject(**inputs[field])
            np.testing.assert_array_equal(
                inputs[field].state0, getattr(scenario, field).state0
            )
        inputs["forces"] = ForceModel(**inputs["forces"])
        inputs["numerics"] = Numerics(**inputs["numerics"])
        inputs["limits"] = ManeuverLimits(**inputs["limits"])
        rebuilt = Scenario(**inputs)
        for key in (
            "epoch", "frame", "horizon_s", "safe_distance_m",
            "forces", "limits", "numerics", "synthetic", "source",
        ):
            self.assertEqual(getattr(rebuilt, key), getattr(scenario, key))
        for value, expected in zip(report["candidates"], candidates, strict=True):
            restored = Impulse(**value)
            self.assertEqual(restored.t_s, expected.t_s)
            np.testing.assert_array_equal(restored.dv_eci_mps, expected.dv_eci_mps)
