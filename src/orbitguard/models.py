"""Shared input validation and result types. See docs/contracts.md."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from numbers import Real
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .constants import EARTH_RADIUS_M, MIN_ALTITUDE_M

Array = NDArray[np.float64]


def finite(value: float, name: str, minimum: float = 0.0) -> None:
    if (not isinstance(value, Real) or isinstance(value, (bool, np.bool_))
            or not np.isfinite(value) or value < minimum):
        raise ValueError(f"{name} must be finite and >= {minimum}")


def vector(value, size: int, name: str) -> Array:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite ({size},) vector")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class SpaceObject:
    name: str
    state0: Array
    mass_kg: float | None = None
    area_m2: float | None = None
    cd: float | None = None

    def __post_init__(self):
        object.__setattr__(self, "state0", vector(self.state0, 6, "state0"))
        if np.linalg.norm(self.state0[:3]) < EARTH_RADIUS_M + MIN_ALTITUDE_M:
            raise ValueError("Initial position is below the model altitude floor")
        for name in ("mass_kg", "area_m2", "cd"):
            value = getattr(self, name)
            if value is not None:
                finite(value, name)
                if value == 0:
                    raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class ForceModel:
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
    rtol: float = 1e-10
    position_atol_m: float = 1e-3
    velocity_atol_mps: float = 1e-6
    closest_time_tol_s: float = 1e-3
    distance_margin_m: float = 1.0
    max_step_s: float = 10.0

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            finite(value, name)
            if value == 0:
                raise ValueError(f"{name} must be positive")
        if self.rtol >= 1:
            raise ValueError("rtol must be < 1")


@dataclass(frozen=True)
class ManeuverLimits:
    dv_max_mps: float
    t_earliest_s: float
    t_latest_s: float

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            finite(getattr(self, name), name)
        if self.t_latest_s < self.t_earliest_s:
            raise ValueError("Maneuver window is reversed")


@dataclass(frozen=True)
class Scenario:
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
        if self.epoch.tzinfo is None or self.epoch.utcoffset() != timedelta(0):
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
                if any(getattr(obj, key) is None for key in ("mass_kg", "area_m2", "cd")):
                    raise ValueError("Drag requires mass_kg, area_m2 and cd for both objects")


class Trajectory(Protocol):
    epoch: datetime
    frame: str
    t_s: Array  # strictly increasing (N,), N >= 2, includes interval boundaries
    states: Array  # (N, 6); positions then velocities in SI

    def state_at(self, t_s: float, *, side: str = "right") -> Array:
        """Return (6,) state; reject extrapolation; side selects impulse limit."""
        ...


@dataclass(frozen=True)
class ForceContext:
    epoch: datetime
    obj: SpaceObject
    forces: ForceModel
    frame: str = "ECI-J2000"


class Acceleration(Protocol):
    def __call__(self, t_s: float, r_m: Array, v_mps: Array, ctx: ForceContext) -> Array:
        """Return a finite (3,) ECI acceleration in m/s² at epoch + t_s."""
        ...


@dataclass(frozen=True)
class Impulse:
    t_s: float
    dv_eci_mps: Array

    def __post_init__(self):
        finite(self.t_s, "t_s")
        object.__setattr__(self, "dv_eci_mps", vector(self.dv_eci_mps, 3, "dv_eci_mps"))

    @property
    def dv_mps(self) -> float:
        return float(np.linalg.norm(self.dv_eci_mps))


@dataclass(frozen=True)
class Approach:
    t_s: float
    distance_m: float
    rel_speed_mps: float

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            finite(getattr(self, name), name)


@dataclass(frozen=True)
class ConjunctionResult:
    closest: Approach
    all_minima: tuple[Approach, ...]
    safe_distance_m: float

    def __post_init__(self):
        finite(self.safe_distance_m, "safe_distance_m")
        if self.safe_distance_m == 0:
            raise ValueError("safe_distance_m must be positive")

    @property
    def violated(self) -> bool:
        return self.closest.distance_m < self.safe_distance_m


class Status(str, Enum):
    ALREADY_SAFE = "already_safe"
    FEASIBLE = "feasible"
    NO_FEASIBLE_CANDIDATE = "no_feasible_candidate"
    PROPAGATION_FAILED = "propagation_failed"


@dataclass(frozen=True)
class CandidateResult:
    impulse: Impulse
    admissible: bool
    after: ConjunctionResult | None
    reason: str = ""


@dataclass(frozen=True)
class ManeuverResult:
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
            raise ValueError("Search without a feasible candidate still requires baseline")
        if self.status in (Status.NO_FEASIBLE_CANDIDATE, Status.PROPAGATION_FAILED):
            if self.after is not None:
                raise ValueError("An unsuccessful search has no selected after result")
        if self.status in (Status.FEASIBLE, Status.ALREADY_SAFE) and self.after.violated:
            raise ValueError("A successful result cannot violate the threshold")
        if self.status == Status.ALREADY_SAFE and self.before != self.after:
            raise ValueError("already_safe must preserve the baseline result")


class PropagationError(RuntimeError):
    """Integration failed or trajectory left the physical model domain."""


class UnsupportedForceError(ValueError):
    """A requested force has not been implemented and validated."""
