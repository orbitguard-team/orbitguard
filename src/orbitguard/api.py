"""Общий API: propagate реализован; поиск и загрузчик ждут OG-03/04/06."""

from pathlib import Path
from collections.abc import Sequence

from .models import (
    ConjunctionResult, Impulse, ManeuverResult, Numerics,
    Scenario, Trajectory,
)
from .propagation import propagate


def find_closest_approach(
    a: Trajectory, b: Trajectory, safe_distance_m: float,
    t_span_s: tuple[float, float], *, numerics: Numerics = Numerics(),
) -> ConjunctionResult:
    raise NotImplementedError("OG-03: conjunction search is not implemented")


def search_maneuver(
    scenario: Scenario, *, candidates: Sequence[Impulse],
) -> ManeuverResult:
    raise NotImplementedError("OG-04: maneuver search is not implemented")


def load_scenario(path: str | Path) -> Scenario:
    raise NotImplementedError("OG-06: JSON scenario loader is not implemented")
