"""Проверки общих контрактов и валидации входа, без орбитальных расчётов."""

from dataclasses import replace
from datetime import datetime
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from orbitguard.api import search_maneuver
from orbitguard.models import (
    Approach, ConjunctionResult, ForceModel, Impulse, ManeuverLimits,
    ManeuverResult, Numerics, Scenario, SpaceObject, Status,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def scenario() -> Scenario:
    """Синтетический сценарий из examples/contract_example.py.

    Returns:
        Проверенный сценарий.
    """
    spec = importlib.util.spec_from_file_location(
        "example", ROOT / "examples/contract_example.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.make_example()


def test_caller_array_cannot_silently_change_initial_conditions(scenario):
    original = scenario.satellite.state0.copy()
    obj = SpaceObject("test", original)
    original[0] = 0
    assert obj.state0[0] == 7_000_000
    with pytest.raises(ValueError):
        obj.state0[0] = 0


@pytest.mark.parametrize("state", [
    [1, 2, 3], [7e6, 0, 0, 0, float("nan"), 0],
    [6378137, 0, 0, 0, 0, 0], [7e6, 0, 0, 0, float("inf"), 0],
])
def test_invalid_states_are_rejected(state):
    with pytest.raises(ValueError):
        SpaceObject("invalid", state)


@pytest.mark.parametrize("changes", [
    {"epoch": datetime(2026, 11, 22)}, {"frame": "ECEF"},
    {"horizon_s": 0}, {"safe_distance_m": -1},
    {"horizon_s": 100}, {"horizon_s": float("inf")},
    {"forces": ForceModel(drag=True)},
])
def test_inconsistent_scenarios_are_rejected(scenario, changes):
    with pytest.raises(ValueError):
        replace(scenario, **changes)


def test_drag_requires_physical_data_for_each_object(scenario):
    sat = replace(scenario.satellite, mass_kg=10, area_m2=1, cd=2.2)
    with pytest.raises(ValueError):
        replace(scenario, satellite=sat, forces=ForceModel(drag=True))
    debris = replace(scenario.debris, mass_kg=1, area_m2=0.1, cd=2.2)
    with_drag = replace(scenario, satellite=sat, debris=debris,
                        forces=ForceModel(drag=True))
    assert with_drag.forces.drag


@pytest.mark.parametrize("factory", [
    lambda: ManeuverLimits(-1, 0, 1), lambda: ManeuverLimits(1, 2, 1),
    lambda: Numerics(rtol=1), lambda: Numerics(max_step_s=0),
    lambda: ForceModel(j2="false"), lambda: Impulse(0, [0, float("nan"), 0]),
])
def test_invalid_numerics_and_limits(factory):
    with pytest.raises(ValueError):
        factory()


def test_threshold_boundary_and_explicit_no_solution():
    approach = Approach(0, 100, 1)
    result = ConjunctionResult(approach, (approach,), 100)
    assert not result.violated
    assert replace(result, safe_distance_m=101).violated
    failed = ManeuverResult(Status.NO_FEASIBLE_CANDIDATE, None, result, None)
    assert failed.best is None and failed.after is None
    with pytest.raises(ValueError):
        ManeuverResult(Status.FEASIBLE, None, result, result)
    with pytest.raises(ValueError):
        ManeuverResult(Status.ALREADY_SAFE, Impulse(0, [1, 0, 0]), result,
                       result)
    assert Impulse(0, [3, 4, 0]).dv_mps == 5


def test_unimplemented_algorithms_never_report_success(scenario):
    with pytest.raises(NotImplementedError, match="OG-04"):
        search_maneuver(scenario, candidates=[])


def test_json_fixture_matches_python_fixture(scenario):
    data = json.loads(
        (ROOT / "examples/scenario.json").read_text(encoding="utf-8"))
    assert data.pop("schema_version") == 1
    data["epoch"] = datetime.fromisoformat(
        data["epoch"].replace("Z", "+00:00"))
    for name in ("satellite", "debris"):
        data[name] = SpaceObject(**data[name])
    data["forces"] = ForceModel(**data["forces"])
    data["numerics"] = Numerics(**data["numerics"])
    data["limits"] = ManeuverLimits(**data["limits"])
    loaded = Scenario(**data)
    assert loaded.epoch == scenario.epoch
    assert loaded.limits == scenario.limits
    assert loaded.forces == scenario.forces
    np.testing.assert_allclose(loaded.satellite.state0,
                               scenario.satellite.state0, rtol=0, atol=1e-9)
    np.testing.assert_allclose(loaded.debris.state0, scenario.debris.state0,
                               rtol=0, atol=1e-9)
