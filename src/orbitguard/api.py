"""Публичные сигнатуры. Реализации — в OG-02/03/04/06."""

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
    """Прогноз движения; реализация — OG-02.

    Args:
        obj: Объект с начальным состоянием в epoch.
        epoch: Дата и время начала, UTC.
        t_span_s: Интервал (0, T), с.
        forces: Учитываемые силы.
        numerics: Численные допуски.
        impulses: Импульсы объекта.

    Raises:
        NotImplementedError: Пока прогноз не подключён.
    """
    raise NotImplementedError("OG-02: propagation is not implemented")


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
