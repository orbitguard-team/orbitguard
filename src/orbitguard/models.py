"""Shared input validation and result types. See docs/contracts.md."""

from dataclasses import dataclass, field, fields
from datetime import datetime, timedelta
from enum import Enum
from numbers import Real
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .constants import EARTH_RADIUS_M, MIN_ALTITUDE_M

Array = NDArray[np.float64]


def finite(value: float, name: str, minimum: float = 0.0) -> None:
    """Проверяет, что значение — конечное число не меньше минимума.

    Args:
        value: Проверяемое значение.
        name: Имя поля для текста ошибки.
        minimum: Наименьшее допустимое значение.

    Raises:
        ValueError: Если значение не число, бесконечно или меньше минимума.
    """
    if (not isinstance(value, Real) or isinstance(value, (bool, np.bool_))
            or not np.isfinite(value) or value < minimum):
        raise ValueError(f"{name} must be finite and >= {minimum}")


def vector(value: ArrayLike, size: int, name: str) -> Array:
    """Превращает значение в неизменяемый конечный вектор float64.

    Args:
        value: Последовательность чисел.
        size: Ожидаемая длина вектора.
        name: Имя поля для текста ошибки.

    Returns:
        Копия значения формы (size,), защищённая от записи.

    Raises:
        ValueError: Если форма неверна или есть нечисловые значения.
    """
    array = np.array(value, dtype=np.float64, copy=True)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite ({size},) vector")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class SpaceObject:
    """Объект на орбите: начальное состояние и физические параметры.

    Attributes:
        name: Имя объекта.
        state0: Состояние (6,) в момент epoch: м и м/с, ECI.
        mass_kg: Масса, кг; нужна для атмосферы.
        area_m2: Площадь миделя, м²; нужна для атмосферы.
        cd: Коэффициент сопротивления, безразмерный.
    """

    name: str
    state0: Array
    mass_kg: float | None = None
    area_m2: float | None = None
    cd: float | None = None

    def __post_init__(self):
        object.__setattr__(self, "state0", vector(self.state0, 6, "state0"))
        if np.linalg.norm(self.state0[:3]) < EARTH_RADIUS_M + MIN_ALTITUDE_M:
            raise ValueError(
                "Initial position is below the model altitude floor")
        for name in ("mass_kg", "area_m2", "cd"):
            value = getattr(self, name)
            if value is not None:
                finite(value, name)
                if value == 0:
                    raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class ForceModel:
    """Какие силы учитывать в прогнозе движения.

    Attributes:
        j2: Сплюснутость Земли.
        drag: Сопротивление атмосферы.
        moon: Притяжение Луны.
        sun: Притяжение Солнца.
        density_scale: Множитель плотности атмосферы, безразмерный.
    """

    j2: bool = False
    drag: bool = False
    moon: bool = False
    sun: bool = False
    density_scale: float = 1.0

    def __post_init__(self):
        for name in ("j2", "drag", "moon", "sun"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        finite(self.density_scale, "density_scale")


@dataclass(frozen=True)
class Numerics:
    """Численные допуски расчёта.

    Attributes:
        rtol: Относительный допуск интегратора, безразмерный.
        position_atol_m: Абсолютный допуск по положению, м.
        velocity_atol_mps: Абсолютный допуск по скорости, м/с.
        closest_time_tol_s: Точность момента сближения, с.
        distance_margin_m: Запас к безопасной дистанции, м.
        max_step_s: Наибольший шаг интегратора, с.
    """

    rtol: float = 1e-10
    position_atol_m: float = 1e-3
    velocity_atol_mps: float = 1e-6
    closest_time_tol_s: float = 1e-3
    distance_margin_m: float = 1.0
    max_step_s: float = 10.0

    def __post_init__(self):
        for item in fields(self):
            name = item.name
            value = getattr(self, name)
            finite(value, name)
            if value == 0:
                raise ValueError(f"{name} must be positive")
        if self.rtol >= 1:
            raise ValueError("rtol must be < 1")


@dataclass(frozen=True)
class ManeuverLimits:
    """Ограничения на импульсный манёвр.

    Attributes:
        dv_max_mps: Наибольшая величина импульса, м/с.
        t_earliest_s: Самый ранний момент импульса, с от epoch.
        t_latest_s: Самый поздний момент импульса, с от epoch.
    """

    dv_max_mps: float
    t_earliest_s: float
    t_latest_s: float

    def __post_init__(self):
        for item in fields(self):
            finite(getattr(self, item.name), item.name)
        if self.t_latest_s < self.t_earliest_s:
            raise ValueError("Maneuver window is reversed")


@dataclass(frozen=True)
class Scenario:
    """Сценарий сближения двух объектов.

    Attributes:
        name: Имя сценария.
        epoch: Дата и время начала, UTC.
        horizon_s: Длина интервала прогноза, с.
        satellite: Защищаемый спутник.
        debris: Опасный объект.
        safe_distance_m: Безопасная дистанция, м.
        limits: Ограничения на манёвр.
        forces: Учитываемые силы.
        numerics: Численные допуски.
        synthetic: Данные синтетические.
        source: Откуда данные или как построены.
        frame: Система координат.
    """

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

    def __post_init__(self):
        if (self.epoch.tzinfo is None
                or self.epoch.utcoffset() != timedelta(0)):
            raise ValueError("epoch must be timezone-aware UTC")
        if self.frame != "ECI-J2000":
            raise ValueError("Only ECI-J2000 is supported")
        finite(self.horizon_s, "horizon_s")
        finite(self.safe_distance_m, "safe_distance_m")
        if self.horizon_s == 0 or self.safe_distance_m == 0:
            raise ValueError("Horizon and safety distance must be positive")
        if self.limits.t_latest_s > self.horizon_s:
            raise ValueError("Maneuver window exceeds horizon")
        if not isinstance(self.synthetic, bool):
            raise ValueError("synthetic must be boolean")
        if self.forces.drag:
            for obj in (self.satellite, self.debris):
                if any(getattr(obj, key) is None
                       for key in ("mass_kg", "area_m2", "cd")):
                    raise ValueError("Drag requires mass_kg, area_m2 and cd "
                                     "for both objects")


class Trajectory(Protocol):
    """Рассчитанная траектория объекта.

    Attributes:
        epoch: Дата и время начала, UTC.
        frame: Система координат.
        t_s: Сетка времени (N,), строго возрастает, N ≥ 2, с границами.
        states: Состояния (N, 6): положения и скорости в СИ.
    """

    epoch: datetime
    frame: str
    t_s: Array
    states: Array

    def state_at(self, t_s: float, *, side: str = "right") -> Array:
        """Состояние (6,) в м и м/с в момент t_s, без экстраполяции.

        Args:
            t_s: Момент времени, с от epoch.
            side: Какую сторону импульса брать в момент импульса:
                "left" — до, "right" — после.
        """
        ...


@dataclass(frozen=True)
class ForceContext:
    """Данные, общие для всех сил при расчёте ускорения.

    Attributes:
        epoch: Дата и время начала, UTC.
        obj: Объект, для которого считается ускорение.
        forces: Учитываемые силы.
        frame: Система координат.
    """

    epoch: datetime
    obj: SpaceObject
    forces: ForceModel
    frame: str = "ECI-J2000"


class Acceleration(Protocol):
    """Сигнатура функции силы."""

    def __call__(self, t_s: float, r_m: Array, v_mps: Array,
                 ctx: ForceContext) -> Array:
        """Конечное ускорение (3,) в м/с², ECI, в момент epoch + t_s.

        Args:
            t_s: Момент времени, с от epoch.
            r_m: Положение (3,), м, ECI.
            v_mps: Скорость (3,), м/с, ECI.
            ctx: Общие данные расчёта.
        """
        ...


@dataclass(frozen=True)
class Impulse:
    """Мгновенный импульс скорости.

    Attributes:
        t_s: Момент импульса, с от epoch.
        dv_eci_mps: Изменение скорости (3,), м/с, ECI.
    """

    t_s: float
    dv_eci_mps: Array

    def __post_init__(self):
        finite(self.t_s, "t_s")
        object.__setattr__(self, "dv_eci_mps",
                           vector(self.dv_eci_mps, 3, "dv_eci_mps"))

    @property
    def dv_mps(self) -> float:
        """Величина импульса, м/с."""
        return float(np.linalg.norm(self.dv_eci_mps))


@dataclass(frozen=True)
class Approach:
    """Локальный минимум расстояния между объектами.

    Attributes:
        t_s: Момент сближения, с от epoch.
        distance_m: Расстояние в этот момент, м.
        rel_speed_mps: Относительная скорость, м/с.
    """

    t_s: float
    distance_m: float
    rel_speed_mps: float

    def __post_init__(self):
        for item in fields(self):
            finite(getattr(self, item.name), item.name)


@dataclass(frozen=True)
class ConjunctionResult:
    """Результат поиска сближения.

    Attributes:
        closest: Ближайшее сближение на интервале.
        all_minima: Все локальные минимумы расстояния.
        safe_distance_m: Безопасная дистанция, м.
    """

    closest: Approach
    all_minima: tuple[Approach, ...]
    safe_distance_m: float

    def __post_init__(self):
        finite(self.safe_distance_m, "safe_distance_m")
        if self.safe_distance_m == 0:
            raise ValueError("safe_distance_m must be positive")

    @property
    def violated(self) -> bool:
        """Сближение ближе безопасной дистанции."""
        return self.closest.distance_m < self.safe_distance_m


class Status(str, Enum):
    """Исход подбора манёвра."""

    ALREADY_SAFE = "already_safe"
    FEASIBLE = "feasible"
    NO_FEASIBLE_CANDIDATE = "no_feasible_candidate"
    PROPAGATION_FAILED = "propagation_failed"


@dataclass(frozen=True)
class CandidateResult:
    """Проверенный вариант импульса.

    Attributes:
        impulse: Импульс.
        admissible: Вариант допустим.
        after: Сближение после импульса, если расчёт удался.
        reason: Почему вариант отклонён.
    """

    impulse: Impulse
    admissible: bool
    after: ConjunctionResult | None
    reason: str = ""


@dataclass(frozen=True)
class ManeuverResult:
    """Результат подбора манёвра.

    Attributes:
        status: Исход подбора.
        best: Выбранный импульс, если найден.
        before: Сближение без манёвра.
        after: Сближение после выбранного манёвра.
        candidates: Все проверенные варианты.
        message: Пояснение к результату.
    """

    status: Status
    best: Impulse | None
    before: ConjunctionResult | None
    after: ConjunctionResult | None
    candidates: tuple[CandidateResult, ...] = ()
    message: str = ""

    def __post_init__(self):
        if not isinstance(self.status, Status):
            raise ValueError("status must be a Status")
        if self.status in (Status.FEASIBLE, Status.ALREADY_SAFE):
            if self.before is None or self.after is None:
                raise ValueError("Successful result requires before and after")
        if self.status == Status.FEASIBLE and self.best is None:
            raise ValueError("feasible requires an impulse")
        if self.status != Status.FEASIBLE and self.best is not None:
            raise ValueError("Only feasible has a selected impulse")
        if self.status == Status.NO_FEASIBLE_CANDIDATE and self.before is None:
            raise ValueError(
                "Search without a feasible candidate still requires baseline")
        if self.status in (Status.NO_FEASIBLE_CANDIDATE,
                           Status.PROPAGATION_FAILED):
            if self.after is not None:
                raise ValueError(
                    "An unsuccessful search has no selected after result")
        if (self.status in (Status.FEASIBLE, Status.ALREADY_SAFE)
                and self.after.violated):
            raise ValueError("A successful result cannot violate the threshold")
        if self.status == Status.ALREADY_SAFE and self.before != self.after:
            raise ValueError("already_safe must preserve the baseline result")


class PropagationError(RuntimeError):
    """Integration failed or trajectory left the physical model domain."""


class UnsupportedForceError(ValueError):
    """A requested force has not been implemented and validated."""
