"""Двухтельное движение RK45 в ECI-J2000 и единицах СИ."""

import dataclasses
import datetime
from collections.abc import Sequence

import numpy as np
from scipy import integrate

from orbitguard import constants, models


def accel_central(
    t_s: float,
    r_m: models.Array,
    v_mps: models.Array,
    ctx: models.ForceContext,
) -> models.Array:
    """Возвращает ускорение центрального поля, м/с², в ECI.

    Args:
        t_s: Секунды от эпохи; поле стационарно.
        r_m: Положение от центра Земли, м, форма (3,).
        v_mps: Скорость, м/с; центральная сила от неё не зависит.
        ctx: Контекст сил; μ взято из constants.

    Returns:
        Вектор ускорения формы (3,) в м/с².

    Raises:
        PropagationError: Нулевая или нечисленная позиция.
    """
    radius_m = float(np.linalg.norm(r_m))
    if not np.isfinite(radius_m) or radius_m == 0:
        raise models.PropagationError("Invalid radius in central gravity")
    return -(constants.MU_EARTH_M3_S2 / radius_m**2) * (r_m / radius_m)


@dataclasses.dataclass(frozen=True)
class _Segment:
    start_s: float
    end_s: float
    solution: integrate.OdeSolution


@dataclasses.dataclass(frozen=True)
class IntegratedTrajectory:
    """Кусочное dense output; таблица содержит правые пределы импульса."""

    epoch: datetime.datetime
    frame: str
    t_s: models.Array
    states: models.Array
    _segments: tuple[_Segment, ...]
    _impulse: models.Impulse | None
    _before_impulse: models.Array | None
    _after_impulse: models.Array | None

    def state_at(
        self, t_s: float, *, side: str = "right"
    ) -> models.Array:
        """Возвращает состояние (6,) в м и м/с без экстраполяции.

        Args:
            t_s: Скалярное время в секундах от epoch.
            side: left до импульса, right после; вне импульса совпадают.

        Returns:
            Новая копия состояния в ECI-J2000.

        Raises:
            ValueError: Время вне интервала или неизвестный side.
        """
        models.finite(t_s, "t_s")
        if side not in ("left", "right"):
            raise ValueError("side must be left or right")
        if not self.t_s[0] <= t_s <= self.t_s[-1]:
            raise ValueError("Cannot extrapolate trajectory")
        if self._impulse is not None and t_s == self._impulse.t_s:
            state = (self._before_impulse if side == "left"
                     else self._after_impulse)
            return state.copy()
        for segment in self._segments:
            if segment.start_s <= t_s <= segment.end_s:
                return np.asarray(segment.solution(t_s)).copy()
        raise ValueError("Time is not covered by trajectory")


def propagate(
    obj: models.SpaceObject,
    epoch: datetime.datetime,
    t_span_s: tuple[float, float],
    forces: models.ForceModel,
    *,
    numerics: models.Numerics = models.Numerics(),
    impulses: Sequence[models.Impulse] = (),
) -> IntegratedTrajectory:
    """Интегрирует dr/dt=v, dv/dt=−μr/|r|³ методом RK45.

    Args:
        obj: Начальное состояние в ECI, м и м/с.
        epoch: Общая timezone-aware дата UTC для t=0.
        t_span_s: Интервал (0, T), T > 0, секунды.
        forces: Конфигурация; пока только центральное поле.
        numerics: Допуски и максимальный шаг интегратора.
        impulses: Не более одного импульса ECI внутри [0, T].

    Returns:
        Траектория с общей сеткой вывода и кусочным dense output.

    Raises:
        ValueError: Неверная эпоха, интервал или импульс.
        UnsupportedForceError: Запрошена ещё не реализованная сила.
        PropagationError: Сбой интегратора или пересечение границы высоты.
    """
    if (not isinstance(epoch, datetime.datetime)
            or epoch.tzinfo is None
            or epoch.utcoffset() != datetime.timedelta(0)):
        raise ValueError("epoch must be timezone-aware UTC")
    if len(t_span_s) != 2:
        raise ValueError("t_span_s must contain two endpoints")
    start_s, end_s = t_span_s
    models.finite(start_s, "start_s")
    models.finite(end_s, "end_s")
    if start_s != 0 or end_s <= start_s:
        raise ValueError("t_span_s must be (0, T) with T > 0")
    enabled = [name for name in ("j2", "drag", "moon", "sun")
               if getattr(forces, name)]
    if enabled:
        raise models.UnsupportedForceError(
            f"Forces not implemented: {', '.join(enabled)}"
        )
    burns = tuple(impulses)
    if len(burns) > 1:
        raise ValueError("MVP supports at most one impulse")
    burn = burns[0] if burns else None
    if burn is not None and not start_s <= burn.t_s <= end_s:
        raise ValueError("Impulse outside integration interval")

    context = models.ForceContext(epoch, obj, forces)
    floor_m = constants.EARTH_RADIUS_M + constants.MIN_ALTITUDE_M

    def rhs(t_s, state):
        return np.concatenate((
            state[3:], accel_central(t_s, state[:3], state[3:], context)
        ))

    def altitude_boundary(t_s, state):
        return np.linalg.norm(state[:3]) - floor_m

    altitude_boundary.terminal = True
    altitude_boundary.direction = -1
    atol = np.array([numerics.position_atol_m] * 3
                    + [numerics.velocity_atol_mps] * 3)
    segments = []
    state = obj.state0.copy()
    # Повторная проверка защищает от сознательного изменения read-only массива.
    if (not np.all(np.isfinite(state))
            or np.linalg.norm(state[:3]) < floor_m):
        raise ValueError("Initial state is outside the model domain")
    current_s = 0.0
    before = after = None
    boundaries = ([burn.t_s] if burn is not None else []) + [end_s]
    for index, stop_s in enumerate(boundaries):
        if stop_s > current_s:
            try:
                solution = integrate.solve_ivp(
                    rhs, (current_s, stop_s), state, method="RK45",
                    rtol=numerics.rtol, atol=atol,
                    max_step=numerics.max_step_s, dense_output=True,
                    events=altitude_boundary,
                )
            except (ValueError, FloatingPointError, OverflowError) as exc:
                raise models.PropagationError(str(exc)) from exc
            if not solution.success:
                raise models.PropagationError(solution.message)
            if solution.t_events[0].size:
                crossing_s = solution.t_events[0][0]
                raise models.PropagationError(
                    f"Altitude floor reached at t={crossing_s:.9g} s"
                )
            if not np.all(np.isfinite(solution.y)):
                raise models.PropagationError("Non-finite integrated state")
            segments.append(_Segment(current_s, stop_s, solution.sol))
            state = solution.y[:, -1].copy()
            current_s = stop_s
        if burn is not None and index == 0:
            before = state.copy()
            state[3:] += burn.dv_eci_mps
            if not np.all(np.isfinite(state)):
                raise models.PropagationError("Non-finite post-impulse state")
            after = state.copy()

    # Сетка вывода независима от адаптивных шагов RK45. Один и тот же
    # горизонт/шаг дают общую сетку для объектов без импульса.
    count = max(2, int(np.ceil(end_s / numerics.max_step_s)) + 1)
    times = np.linspace(0, end_s, count)
    if burn is not None:
        times = np.unique(np.append(times, burn.t_s))
    times.setflags(write=False)
    trajectory = IntegratedTrajectory(
        epoch, "ECI-J2000", times, np.empty((0, 6)),
        tuple(segments), burn, before, after,
    )
    states = np.vstack([trajectory.state_at(float(t)) for t in times])
    states.setflags(write=False)
    return dataclasses.replace(trajectory, states=states)
