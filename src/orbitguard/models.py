"""Валидация входов и результатов; единицы СИ, контракт docs/contracts.md."""

from dataclasses import dataclass, field, fields
from datetime import datetime, timedelta
from enum import Enum
from numbers import Real
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .constants import EARTH_RADIUS_M, MIN_ALTITUDE_M

Array = NDArray[np.float64]


def finite(value: float, name: str, minimum: float = 0.0) -> None:
    """Проверяет конечный скаляр; публичный помощник валидации.

    Args:
        value: Проверяемое число в единицах вызывающей стороны.
        name: Имя поля для сообщения об ошибке.
        minimum: Нижняя допустимая граница в тех же единицах.

    Raises:
        ValueError: Число неконечно, имеет неверный тип или меньше границы.
    """
    if (not isinstance(value, Real) or isinstance(value, (bool, np.bool_))
            or not np.isfinite(value) or value < minimum):
        raise ValueError(f"{name}: нужно конечное число >= {minimum}")


def vector(value: object, size: int, name: str) -> Array:
    """Копирует конечный вектор и запрещает случайную запись.

    Args:
        value: Входной массив в единицах вызывающей стороны.
        size: Число компонентов.
        name: Имя поля для сообщения об ошибке.

    Returns:
        Неизменяемая копия float64 формы (size,).

    Raises:
        ValueError: Неверный тип, форма или неконечные компоненты.
    """
    try:
        array = np.array(value, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}: нужен числовой вектор") from exc
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name}: нужен конечный вектор формы ({size},)")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class SpaceObject:
    """Объект в ECI: state0 (6,) в м и м/с; параметры тела в СИ."""
    name: str
    state0: Array
    mass_kg: float | None = None
    area_m2: float | None = None
    cd: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name: нужна непустая строка")
        object.__setattr__(self, "state0", vector(self.state0, 6, "state0"))
        if np.linalg.norm(self.state0[:3]) < EARTH_RADIUS_M + MIN_ALTITUDE_M:
            raise ValueError("Начальное положение ниже условной границы модели")
        for name in ("mass_kg", "area_m2", "cd"):
            value = getattr(self, name)
            if value is not None:
                finite(value, name)
                if value == 0:
                    raise ValueError(f"{name} должен быть положительным")


@dataclass(frozen=True)
class ForceModel:
    """Включённые возмущения; центральное поле включено всегда."""
    j2: bool = False
    drag: bool = False
    moon: bool = False
    sun: bool = False
    density_scale: float = 1.0

    def __post_init__(self) -> None:
        for name in ("j2", "drag", "moon", "sun"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} должен быть bool")
        finite(self.density_scale, "density_scale")


@dataclass(frozen=True)
class Numerics:
    """Численные допуски в СИ; rtol безразмерен, запас расстояния в метрах."""
    rtol: float = 1e-10
    position_atol_m: float = 1e-3
    velocity_atol_mps: float = 1e-6
    closest_time_tol_s: float = 1e-3
    distance_margin_m: float = 1.0
    max_step_s: float = 10.0

    def __post_init__(self) -> None:
        for item in fields(self):
            name = item.name
            value = getattr(self, name)
            finite(value, name)
            if value == 0:
                raise ValueError(f"{name} должен быть положительным")
        if self.rtol >= 1:
            raise ValueError("rtol должен быть < 1")


@dataclass(frozen=True)
class ManeuverLimits:
    """Бюджет Δv в м/с и допустимое окно импульса в секундах от epoch."""
    dv_max_mps: float
    t_earliest_s: float
    t_latest_s: float

    def __post_init__(self) -> None:
        for item in fields(self):
            name = item.name
            finite(getattr(self, name), name)
        if self.t_latest_s < self.t_earliest_s:
            raise ValueError("Конец окна манёвра раньше начала")


@dataclass(frozen=True)
# Состав сценария задан общим контрактом, поля не являются сервисами.
class Scenario:  # pylint: disable=too-many-instance-attributes
    """Два объекта в одной ECI-системе, UTC-эпоха и параметры расчёта в СИ."""
    name: str
    epoch: datetime
    horizon_s: float
    satellite: SpaceObject
    debris: SpaceObject
    safe_distance_m: float
    limits: ManeuverLimits
    forces: ForceModel = field(default_factory=ForceModel)
    numerics: Numerics = field(default_factory=Numerics)
    synthetic: bool = True
    source: str = ""
    frame: str = "ECI-J2000"

    def __post_init__(self) -> None:
        expected = {
            "name": str, "source": str, "epoch": datetime,
            "satellite": SpaceObject, "debris": SpaceObject,
            "limits": ManeuverLimits, "forces": ForceModel,
            "numerics": Numerics,
        }
        for name, kind in expected.items():
            if not isinstance(getattr(self, name), kind):
                raise ValueError(f"{name}: требуется {kind.__name__}")
        if not self.name.strip():
            raise ValueError("name: нужна непустая строка")
        if self.epoch.tzinfo is None or self.epoch.utcoffset() != timedelta(0):
            raise ValueError("epoch должен быть datetime с часовым поясом UTC")
        if self.frame != "ECI-J2000":
            raise ValueError("Поддерживается только ECI-J2000")
        finite(self.horizon_s, "horizon_s")
        finite(self.safe_distance_m, "safe_distance_m")
        if self.horizon_s == 0 or self.safe_distance_m == 0:
            raise ValueError("Горизонт и дистанция должны быть > 0")
        if self.limits.t_latest_s > self.horizon_s:
            raise ValueError("Окно манёвра выходит за горизонт")
        if not isinstance(self.synthetic, bool):
            raise ValueError("synthetic должен быть bool")
        if self.forces.drag:
            for obj in (self.satellite, self.debris):
                keys = ("mass_kg", "area_m2", "cd")
                if any(getattr(obj, key) is None for key in keys):
                    raise ValueError("Для drag нужны mass_kg, area_m2, cd")


class Trajectory(Protocol):
    """Кусочная траектория в ECI; положения м, скорости м/с, время с."""
    epoch: datetime
    frame: str
    t_s: Array  # строго возрастающая (N,), N >= 2, включая границы
    states: Array  # (N, 6); положения и скорости в СИ

    def state_at(self, t_s: float, *, side: str = "right") -> Array:
        """Состояние (6,) в СИ без экстраполяции.

        Args:
            t_s: Время от epoch, с.
            side: left или right — предел на импульсе.

        Raises:
            ValueError: Время вне интервала или неизвестный side.
        """
        ...


@dataclass(frozen=True)
class ForceContext:
    """Объект, UTC-эпоха и силы для вычисления ускорения в ECI."""
    epoch: datetime
    obj: SpaceObject
    forces: ForceModel
    frame: str = "ECI-J2000"


class Acceleration(Protocol):
    """Интерфейс силы: конечное ускорение (3,) в ECI, м/с²."""
    def __call__(
        self, t_s: float, r_m: Array, v_mps: Array, ctx: ForceContext
    ) -> Array:
        """Конечное ускорение (3,) в ECI, м/с², в момент epoch + t_s."""
        ...


@dataclass(frozen=True)
class Impulse:
    """Мгновенное изменение скорости (3,) в ECI, м/с, в момент t_s."""
    t_s: float
    dv_eci_mps: Array

    def __post_init__(self) -> None:
        finite(self.t_s, "t_s")
        value = vector(self.dv_eci_mps, 3, "dv_eci_mps")
        object.__setattr__(self, "dv_eci_mps", value)

    @property
    def dv_mps(self) -> float:
        """Евклидова норма импульса, м/с."""
        return float(np.linalg.norm(self.dv_eci_mps))


@dataclass(frozen=True)
class Approach:
    """Время с, дистанция м и относительная скорость м/с двух объектов."""
    t_s: float
    distance_m: float
    rel_speed_mps: float

    def __post_init__(self) -> None:
        for item in fields(self):
            name = item.name
            finite(getattr(self, name), name)


@dataclass(frozen=True)
class ConjunctionResult:
    """Упорядоченные минимумы и проверка порога с численным запасом, м."""
    closest: Approach
    all_minima: tuple[Approach, ...]
    safe_distance_m: float
    distance_margin_m: float = 1.0

    def __post_init__(self) -> None:
        finite(self.safe_distance_m, "safe_distance_m")
        finite(self.distance_margin_m, "distance_margin_m")
        if self.distance_margin_m == 0:
            raise ValueError("distance_margin_m должен быть положительным")
        if self.safe_distance_m == 0:
            raise ValueError("safe_distance_m должен быть положительным")
        try:
            minima = tuple(self.all_minima)
        except TypeError as exc:
            raise ValueError("all_minima: нужна последовательность") from exc
        if not minima or not all(isinstance(a, Approach) for a in minima):
            raise ValueError("all_minima: нужны элементы Approach")
        if any(a.t_s >= b.t_s for a, b in zip(minima, minima[1:])):
            raise ValueError("Времена all_minima должны строго возрастать")
        earliest = min(minima, key=lambda a: (a.distance_m, a.t_s))
        if not isinstance(self.closest, Approach) or self.closest != earliest:
            raise ValueError("closest: нужен самый ранний минимум")
        object.__setattr__(self, "all_minima", minima)

    @property
    def violated(self) -> bool:
        """Дистанция строго меньше безопасного порога без запаса."""
        return self.closest.distance_m < self.safe_distance_m

    @property
    def accepted(self) -> bool:
        """Дистанция удовлетворяет порогу с численным запасом."""
        return (self.closest.distance_m - self.safe_distance_m
                >= self.distance_margin_m)


class Status(str, Enum):
    """Явные исходы поиска; отсутствие решения не означает безопасность."""
    ALREADY_SAFE = "already_safe"
    FEASIBLE = "feasible"
    NO_FEASIBLE_CANDIDATE = "no_feasible_candidate"
    PROPAGATION_FAILED = "propagation_failed"


@dataclass(frozen=True)
class CandidateResult:
    """Проверенный импульс и результат; admissible требует запаса расстояния."""
    impulse: Impulse
    admissible: bool
    after: ConjunctionResult | None
    reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.impulse, Impulse):
            raise ValueError("impulse должен быть Impulse")
        if not isinstance(self.admissible, bool):
            raise ValueError("admissible должен быть bool")
        if (self.after is not None
                and not isinstance(self.after, ConjunctionResult)):
            raise ValueError("after должен быть ConjunctionResult или None")
        if self.admissible and (self.after is None or not self.after.accepted):
            raise ValueError("Кандидату нужен запас расстояния")
        if not isinstance(self.reason, str):
            raise ValueError("reason должен быть строкой")


@dataclass(frozen=True)
# Поля результата определены контрактом экспорта и UI.
class ManeuverResult:  # pylint: disable=too-many-instance-attributes
    """Исход поиска с фактическими силами и неизменяемым журналом кандидатов."""
    status: Status
    best: Impulse | None
    before: ConjunctionResult | None
    after: ConjunctionResult | None
    forces: ForceModel
    candidates: tuple[CandidateResult, ...] = ()
    message: str = ""
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # Явные проверки всех разрешённых сочетаний статуса и полей.
        # pylint: disable=too-many-branches
        if not isinstance(self.forces, ForceModel):
            raise ValueError("forces должен быть ForceModel")
        for name in ("before", "after"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, ConjunctionResult):
                raise ValueError(f"{name}: нужен ConjunctionResult или None")
        if self.best is not None and not isinstance(self.best, Impulse):
            raise ValueError("best должен быть Impulse или None")
        for name, kind in (("candidates", CandidateResult), ("warnings", str)):
            value = getattr(self, name)
            if isinstance(value, (str, bytes)):
                raise ValueError(f"{name}: нужна последовательность элементов")
            try:
                items = tuple(value)
            except TypeError as exc:
                raise ValueError(f"{name}: нужна последовательность") from exc
            if not all(isinstance(item, kind) for item in items):
                raise ValueError(f"{name}: неверный тип элемента")
            object.__setattr__(self, name, items)
        if not isinstance(self.message, str):
            raise ValueError("message должен быть строкой")
        if self.before is not None and self.after is not None:
            if (self.before.safe_distance_m != self.after.safe_distance_m
                    or self.before.distance_margin_m
                    != self.after.distance_margin_m):
                raise ValueError("Порог и запас до/после должны совпадать")
        if not isinstance(self.status, Status):
            raise ValueError("status должен быть Status")
        if self.status in (Status.FEASIBLE, Status.ALREADY_SAFE):
            if self.before is None or self.after is None:
                raise ValueError("Успешный результат требует before и after")
        if self.status == Status.FEASIBLE and self.best is None:
            raise ValueError("feasible требует импульс")
        if self.status != Status.FEASIBLE and self.best is not None:
            raise ValueError("Только feasible имеет выбранный импульс")
        if self.status == Status.NO_FEASIBLE_CANDIDATE and self.before is None:
            raise ValueError("Поиск без решения требует исходное сближение")
        failed = (Status.NO_FEASIBLE_CANDIDATE, Status.PROPAGATION_FAILED)
        if self.status in failed:
            if self.after is not None:
                raise ValueError("При неуспешном поиске after должен быть None")
        if self.status in (Status.FEASIBLE, Status.ALREADY_SAFE):
            if not self.after.accepted:
                raise ValueError("Успешный результат требует запаса расстояния")
        if self.status == Status.ALREADY_SAFE and self.before != self.after:
            raise ValueError("already_safe должен сохранять исходное сближение")


class PropagationError(RuntimeError):
    """Сбой интеграции или выход траектории из физической области модели."""


class UnsupportedForceError(ValueError):
    """Запрошенная сила ещё не реализована и не проверена."""
