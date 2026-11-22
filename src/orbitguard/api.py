"""Общий API: propagate реализован; остальные алгоритмы ждут OG-03/04/06."""

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
    """Заготовка поиска сближения.

    Args:
        a: Траектория первого объекта.
        b: Траектория второго объекта.
        safe_distance_m: Безопасная дистанция, м.
        t_span_s: Интервал поиска, с.
        numerics: Допуски поиска и запас расстояния.

    Raises:
        NotImplementedError: Алгоритм ещё не реализован.
    """
    raise NotImplementedError("OG-03: поиск сближения ещё не реализован")


def search_maneuver(
    scenario: Scenario, *, candidates: Sequence[Impulse],
) -> ManeuverResult:
    """Заготовка поиска манёвра.

    Args:
        scenario: Сценарий в СИ.
        candidates: Упорядоченные импульсы ECI.

    Raises:
        NotImplementedError: Алгоритм ещё не реализован.
    """
    raise NotImplementedError("OG-04: поиск манёвра ещё не реализован")


def load_scenario(path: str | Path) -> Scenario:
    """Заготовка загрузчика JSON.

    Args:
        path: Путь к сценарию.

    Raises:
        NotImplementedError: Алгоритм ещё не реализован.
    """
    raise NotImplementedError("OG-06: загрузчик JSON ещё не реализован")
