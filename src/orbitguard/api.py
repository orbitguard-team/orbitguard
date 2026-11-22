"""Сигнатуры; реализации принадлежат OG-02/03/04/06."""

from datetime import datetime
from pathlib import Path
from collections.abc import Sequence

from .models import (
    ConjunctionResult, ForceModel, Impulse, ManeuverResult, Numerics,
    Scenario, SpaceObject, Trajectory,
)


def propagate(
    obj: SpaceObject, epoch: datetime, t_span_s: tuple[float, float],
    forces: ForceModel, *, numerics: Numerics = Numerics(),
    impulses: Sequence[Impulse] = (),
) -> Trajectory:
    """Заготовка прогноза движения.

    Args:
        obj: Объект в ECI, м и м/с.
        epoch: Начальная дата UTC.
        t_span_s: Интервал (0, T), с.
        forces: Конфигурация сил.
        numerics: Численные допуски.
        impulses: Последовательность импульсов ECI, м/с.

    Raises:
        NotImplementedError: Алгоритм ещё не реализован.
    """
    raise NotImplementedError("OG-02: прогноз ещё не реализован")


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
