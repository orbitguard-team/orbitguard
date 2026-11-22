"""Проверки форматов метрик и воспроизводимости JSON-отчёта."""

import json
import platform
from dataclasses import asdict
from datetime import datetime, timezone

import numpy as np
import pytest

from orbitguard.metrics import (
    build_metrics, export_json, export_result_json, summarize,
)
from orbitguard.models import (
    Approach, CandidateResult, ConjunctionResult, ForceModel, Impulse,
    ManeuverLimits, ManeuverResult, Numerics, Scenario, SpaceObject, Status,
)


@pytest.fixture
def scenario() -> Scenario:
    """Синтетический сценарий с параметрами объектов.

    Returns:
        Сценарий.
    """
    return Scenario(
        name="Синтетический сценарий",
        epoch=datetime(2026, 11, 22, microsecond=123456, tzinfo=timezone.utc),
        horizon_s=600,
        satellite=SpaceObject("sat", [7e6, 0, 0, 0, 7546.053290107542, 0],
                              mass_kg=200, area_m2=2, cd=2.2),
        debris=SpaceObject("debris", [7e6, 1000, 0, 0, 7546, 0],
                           mass_kg=50, area_m2=0.5, cd=2.1),
        safe_distance_m=100,
        limits=ManeuverLimits(5, 0, 400),
        numerics=Numerics(max_step_s=5, distance_margin_m=3),
        source="аналитическая фикстура",
    )


def conjunction(distance: float, time: float) -> ConjunctionResult:
    """Сближение с одним минимумом.

    Args:
        distance: Расстояние, м.
        time: Момент, с.

    Returns:
        Результат с безопасной дистанцией 100 м.
    """
    approach = Approach(time, distance, 2.0)
    return ConjunctionResult(approach, (approach,), 100.0, 3.0)


@pytest.fixture
def feasible() -> ManeuverResult:
    """Найденный манёвр с одним отклонённым кандидатом.

    Returns:
        Результат подбора.
    """
    chosen = Impulse(12, [3, 4, 0])
    rejected = Impulse(90, [0, 1, 0])
    after = conjunction(120, 72)
    return ManeuverResult(
        Status.FEASIBLE, chosen, conjunction(30, 61), after,
        forces=ForceModel(), candidates=(
            CandidateResult(rejected, False, None, "ошибка интегратора"),
            CandidateResult(chosen, True, after, ""),
        ),
    )


def test_summarize_includes_distances_thresholds_and_impulse(
    scenario, feasible
):
    metrics = build_metrics(feasible, scenario=scenario, compute_time_s=0.25)
    assert metrics["status"] == "feasible"
    assert metrics["before"]["d_min_m"] == 30
    assert metrics["before"]["t_min_s"] == 61
    assert metrics["after"]["d_min_m"] == 120
    assert metrics["after"]["t_min_s"] == 72
    assert metrics["after"]["accepted"] is True
    assert metrics["d_min_gain_m"] == 90
    assert metrics["dv_eci_mps"] == [3.0, 4.0, 0.0]
    assert metrics["dv_mps"] == 5.0
    assert metrics["impulse_t_s"] == 12
    assert metrics["compute_time_s"] == 0.25
    assert metrics["safe_distance_m"] == 100
    assert metrics["distance_margin_m"] == 3
    assert metrics["forces"] == asdict(scenario.forces)


def test_threshold_violation_and_acceptance_use_different_boundaries(scenario):
    before = conjunction(101, 40)
    result = ManeuverResult(
        Status.NO_FEASIBLE_CANDIDATE, None, before, None, ForceModel()
    )
    metrics = summarize(result, scenario=scenario)
    assert metrics["before"]["violated"] is False
    assert metrics["before"]["accepted"] is False
    assert metrics["d_min_gain_m"] is None


def test_safe_baseline_has_zero_delta_v_and_no_burn_time():
    safe = conjunction(200, 40)
    result = ManeuverResult(Status.ALREADY_SAFE, None, safe, safe, ForceModel())
    metrics = summarize(result)
    assert metrics["before"] == metrics["after"]
    assert metrics["dv_eci_mps"] == [0.0, 0.0, 0.0]
    assert metrics["dv_mps"] == 0
    assert metrics["impulse_t_s"] is None
    assert metrics["compute_time_s"] is None
    assert metrics["d_min_gain_m"] == 0


@pytest.mark.parametrize("elapsed", [-1, float("nan"), float("inf"), True])
def test_rejects_invalid_elapsed_time(elapsed):
    result = ManeuverResult(
        Status.PROPAGATION_FAILED, None, None, None, ForceModel()
    )
    with pytest.raises(ValueError, match="compute_time_s"):
        build_metrics(result, compute_time_s=elapsed)


def test_feasible_export_rebuilds_exact_scenario_and_candidate_order(
    scenario, feasible
):
    payload = export_result_json(
        scenario, feasible, compute_time_s=np.float64(0.1), git_commit="abc123",
    )
    report = json.loads(payload)
    assert report["schema_version"] == 1
    assert report["units"]["velocity"] == "m/s"
    assert report["metrics"]["status"] == "feasible"
    assert report["metrics"]["dv_mps"] == 5.0
    assert "fuel_kg" not in report["metrics"]
    assert report["software"]["git_commit"] == "abc123"
    assert report["software"]["python"] == platform.python_version()
    assert report["software"]["numpy"] == np.__version__
    assert report["software"]["orbitguard"]
    assert report["software"]["scipy"]

    inputs = report["scenario"]
    assert inputs.pop("schema_version") == 1
    assert inputs["epoch"] == "2026-11-22T00:00:00.123456Z"
    inputs["epoch"] = datetime.fromisoformat(inputs["epoch"])
    for field in ("satellite", "debris"):
        inputs[field] = SpaceObject(**inputs[field])
    inputs["forces"] = ForceModel(**inputs["forces"])
    inputs["numerics"] = Numerics(**inputs["numerics"])
    inputs["limits"] = ManeuverLimits(**inputs["limits"])
    np.testing.assert_equal(asdict(Scenario(**inputs)), asdict(scenario))

    logged = report["candidate_results"]
    assert report["candidates"] == [entry["impulse"] for entry in logged]
    assert logged[0]["admissible"] is False
    assert logged[0]["after"] is None
    assert logged[0]["reason"] == "ошибка интегратора"
    assert logged[1]["admissible"] is True
    assert logged[1]["after"]["d_min_m"] == 120
    for value, record in zip(report["candidates"], feasible.candidates):
        restored = asdict(Impulse(**value))
        np.testing.assert_equal(restored, asdict(record.impulse))


@pytest.mark.parametrize(
    "mode", ["order", "vector", "time", "missing", "extra"],
)
def test_export_rejects_different_candidate_inputs(scenario, feasible, mode):
    first, second = (item.impulse for item in feasible.candidates)
    alternatives = {
        "order": (second, first),
        "vector": (Impulse(first.t_s, [0, 1, 1]), second),
        "time": (Impulse(first.t_s + 1, first.dv_eci_mps), second),
        "missing": (first,),
        "extra": (first, second, second),
    }
    with pytest.raises(ValueError, match="не совпадают с журналом"):
        export_result_json(scenario, feasible, candidates=alternatives[mode])


def test_export_preserves_unexamined_inputs_on_early_success(scenario):
    safe = conjunction(200, 40)
    result = ManeuverResult(Status.ALREADY_SAFE, None, safe, safe, ForceModel())
    report = json.loads(export_result_json(
        scenario, result, candidates=(Impulse(90, [0, 1, 0]),)
    ))
    assert report["candidates"] == [{"t_s": 90, "dv_eci_mps": [0.0, 1.0, 0.0]}]
    assert report["candidate_results"] == []


def test_failed_propagation_can_be_exported_without_baseline(scenario):
    result = ManeuverResult(
        Status.PROPAGATION_FAILED, None, None, None, ForceModel(),
        message="сбой"
    )
    report = json.loads(export_result_json(scenario, result))
    metrics = report["metrics"]
    assert metrics["status"] == "propagation_failed"
    assert report["message"] == "сбой"
    assert report["candidates"] is None
    assert report["candidate_results"] == []
    for key in ("before", "after", "dv_eci_mps", "dv_mps", "impulse_t_s"):
        assert metrics[key] is None


def test_empty_candidate_search_exports_empty_list(scenario):
    result = ManeuverResult(
        Status.NO_FEASIBLE_CANDIDATE, None, conjunction(30, 61), None,
        ForceModel(),
    )
    report = json.loads(export_result_json(scenario, result))
    assert report["candidates"] == []
    assert report["candidate_results"] == []


def test_export_json_writes_same_report_to_file(scenario, feasible, tmp_path):
    destination = tmp_path / "report.json"
    export_json(scenario, feasible, destination)
    assert destination.read_text(encoding="utf-8") == export_result_json(
        scenario, feasible,
    )
