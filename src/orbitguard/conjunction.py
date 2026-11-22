"""Поиск минимальной дистанции между траекториями в общей системе ECI."""

from collections.abc import Callable, Iterator

import numpy as np

from .models import Approach, ConjunctionResult, Numerics, Trajectory, finite


def _search_nodes(
    a: Trajectory, b: Trajectory, start: float, end: float, max_step: float
) -> Iterator[tuple[float, str]]:
    """Перебирает моменты проверки: оба предела узлов и точки внутри.

    Args:
        a: Первая траектория.
        b: Вторая траектория.
        start: Начало интервала, с от epoch.
        end: Конец интервала, с от epoch.
        max_step: Наибольший шаг между точками проверки, с.

    Yields:
        Пары (момент в с, сторона "left" или "right" для импульса).
    """
    knots = np.union1d(a.t_s, b.t_s)
    knots = np.r_[start, knots[(start < knots) & (knots < end)], end]
    yield start, "right"
    for left, right in zip(knots[:-1], knots[1:]):
        count = max(1, int(np.ceil((right - left) / max_step)))
        for point in np.linspace(left, right, count + 1)[1:-1]:
            if left < point < right:
                yield float(point), "right"
        yield float(right), "left"
        if right < end:
            yield float(right), "right"


def _refine_minimum(rate: Callable[[float], float], left: float,
                    right: float, tolerance: float) -> float:
    """Уточняет делением пополам переход от сближения к удалению.

    Args:
        rate: Скорость сближения r_rel · v_rel в момент времени.
        left: Момент, где объекты ещё сближаются, с.
        right: Момент, где объекты уже удаляются, с.
        tolerance: Требуемая точность момента, с.

    Returns:
        Момент минимума расстояния, с.
    """
    while right - left > tolerance:
        middle = left + (right - left) / 2
        if middle in (left, right):
            break
        slope = rate(middle)
        if slope == 0:
            return middle
        if slope < 0:
            left = middle
        else:
            right = middle
    return left + (right - left) / 2


def find_closest_approach(
    a: Trajectory,
    b: Trajectory,
    safe_distance_m: float,
    t_span_s: tuple[float, float],
    *,
    numerics: Numerics = Numerics(),
) -> ConjunctionResult:
    """Возвращает ближайшее сближение, найденные минимумы и границы.

    Args:
        a: Первая траектория в ECI.
        b: Вторая траектория с той же эпохой и системой координат.
        safe_distance_m: Положительный порог безопасности, м.
        t_span_s: Полностью покрываемый обеими траекториями интервал, с.
        numerics: Верхний шаг поиска и допуск уточнения времени.

    Returns:
        Результат в СИ. Для скорости на импульсе берётся правый предел.
        Конечная сетка может пропустить несколько экстремумов между
        соседними точками. Для проверки результата уплотняйте сетку.

    Raises:
        ValueError: Несовместимые траектории, порог или интервал.
    """
    if a.epoch != b.epoch or a.frame != b.frame:
        raise ValueError(
            "Эпоха и система координат траекторий должны совпадать")
    if len(t_span_s) != 2:
        raise ValueError("Интервал поиска должен содержать два момента")
    start, end = t_span_s
    finite(start, "start_s")
    finite(end, "end_s")
    finite(safe_distance_m, "safe_distance_m")
    if safe_distance_m <= 0:
        raise ValueError("Порог безопасности должен быть положительным")
    if not (max(a.t_s[0], b.t_s[0]) <= start < end
            <= min(a.t_s[-1], b.t_s[-1])):
        raise ValueError(
            "Обе траектории должны покрывать весь интервал поиска")

    def relative_state(time: float, side: str = "right") -> np.ndarray:
        """Относительное состояние a − b.

        Args:
            time: Момент, с от epoch.
            side: Сторона импульса.

        Returns:
            Состояние (6,), м и м/с.
        """
        return a.state_at(time, side=side) - b.state_at(time, side=side)

    def rate(time: float, side: str = "right") -> float:
        """Скорость сближения r_rel · v_rel: < 0 — объекты сближаются.

        Args:
            time: Момент, с от epoch.
            side: Сторона импульса.

        Returns:
            r_rel · v_rel, м²/с.
        """
        state = relative_state(time, side)
        return float(state[:3] @ state[3:])

    def approach(time: float) -> Approach:
        """Сближение в заданный момент.

        Args:
            time: Момент, с от epoch.

        Returns:
            Момент, расстояние и относительная скорость.
        """
        state = relative_state(time)
        return Approach(time, float(np.linalg.norm(state[:3])),
                        float(np.linalg.norm(state[3:])))

    times = {float(start), float(end)}
    last_negative = None
    first_zero = None
    for time, side in _search_nodes(a, b, start, end, numerics.max_step_s / 2):
        slope = rate(time, side)
        if slope < 0:
            last_negative = time
            first_zero = None
        elif slope == 0:
            if last_negative is not None and first_zero is None:
                first_zero = time
        elif last_negative is not None:
            if first_zero is not None:
                minimum = first_zero
            else:
                minimum = _refine_minimum(
                    rate, last_negative, time, numerics.closest_time_tol_s
                )
            times.add(minimum)
            last_negative = first_zero = None

    minima = tuple(approach(time) for time in sorted(times))
    closest = min(minima, key=lambda item: (item.distance_m, item.t_s))
    return ConjunctionResult(
        closest, minima, safe_distance_m, numerics.distance_margin_m
    )
