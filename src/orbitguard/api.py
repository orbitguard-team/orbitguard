"""Общий API: propagate реализован; поиск и загрузчик ждут OG-03/04/06."""

from pathlib import Path
from collections.abc import Sequence

from .models import (
    ConjunctionResult, Impulse, ManeuverResult, Numerics,
    Scenario, Trajectory,
)
from .propagation import propagate

__all__ = ["find_closest_approach", "load_scenario", "propagate",
           "search_maneuver"]


def find_closest_approach(
    a: Trajectory, b: Trajectory, safe_distance_m: float,
    t_span_s: tuple[float, float], *, numerics: Numerics = Numerics(),
) -> ConjunctionResult:
    """Ищет ближайшее сближение; реализация — OG-03.

    Args:
        a: Траектория первого объекта.
        b: Траектория второго объекта.
        safe_distance_m: Безопасная дистанция, м.
        t_span_s: Интервал поиска, с от epoch.
        numerics: Численные допуски.

    Raises:
        NotImplementedError: Пока поиск не подключён.
    """
    raise NotImplementedError("OG-03: conjunction search is not implemented")


def search_maneuver(
    scenario: Scenario, *, candidates: Sequence[Impulse],
) -> ManeuverResult:
    """Подбирает манёвр; реализация — OG-04.

    Args:
        scenario: Сценарий сближения.
        candidates: Проверяемые импульсы.

    Raises:
        NotImplementedError: Пока подбор не подключён.
    """
    raise NotImplementedError("OG-04: maneuver search is not implemented")


def load_scenario(path: str | Path) -> Scenario:
    """Загружает сценарий из JSON; реализация — OG-06.

    Args:
        path: Путь к файлу сценария.

    Raises:
        NotImplementedError: Пока загрузчик не подключён.
    """
    raise NotImplementedError(
        "OG-06: JSON scenario loader is not implemented")
