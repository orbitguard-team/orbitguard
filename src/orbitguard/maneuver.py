"""Перебор конечного набора импульсных манёвров в ECI-J2000."""

from collections.abc import Sequence

import numpy as np

from . import models
from .conjunction import find_closest_approach
from .propagation import propagate


def build_rtn_candidates(
    trajectory: models.Trajectory,
    *,
    times_s: Sequence[float],
    magnitudes_mps: Sequence[float],
    directions_rtn: Sequence[models.Array],
) -> tuple[models.Impulse, ...]:
    """Строит произведение заданных времён, направлений RTN и величин.

    Порядок элементов сохраняется внутри каждой оси перебора. Базис RTN
    строится по левому пределу состояния до импульса.

    Args:
        trajectory: Траектория спутника до манёвра.
        times_s: Моменты импульса от эпохи, с.
        magnitudes_mps: Модули импульса, м/с.
        directions_rtn: Ненулевые направления в базисе RTN.

    Returns:
        Импульсы ECI в порядке времени, направления и модуля.

    Raises:
        ValueError: Некорректная сетка или неопределённый базис RTN.
    """
    directions = []
    for direction in directions_rtn:
        vector = models.vector(direction, 3, "direction_rtn")
        length = float(np.linalg.norm(vector))
        if not np.isfinite(length) or length == 0:
            raise ValueError("Норма направления RTN должна быть конечной и > 0")
        directions.append(vector / length)
    for magnitude in magnitudes_mps:
        models.finite(magnitude, "magnitude_mps")

    candidates = []
    for time in times_s:
        models.finite(time, "t_s")
        state = trajectory.state_at(time, side="left")
        position, velocity = state[:3], state[3:]
        moment = np.cross(position, velocity)
        radius = float(np.linalg.norm(position))
        moment_norm = float(np.linalg.norm(moment))
        defined = np.isfinite([radius, moment_norm]).all()
        if not defined or min(radius, moment_norm) == 0:
            raise ValueError("Для этого состояния базис RTN не определён")
        radial = position / radius
        normal = moment / moment_norm
        along_track = np.cross(normal, radial)
        rtn_to_eci = np.column_stack((radial, along_track, normal))
        for direction in directions:
            for magnitude in magnitudes_mps:
                candidates.append(
                    models.Impulse(time, (rtn_to_eci @ direction) * magnitude)
                )
    return tuple(candidates)


def search_maneuver(
    scenario: models.Scenario,
    *,
    candidates: Sequence[models.Impulse],
) -> models.ManeuverResult:
    """Выбирает допустимый импульс с минимальной нормой из переданного списка.

    Оба объекта сравниваются на всём горизонте для каждого кандидата.
    При равной норме выбирается более раннее время, затем первый кандидат.

    Args:
        scenario: Начальные состояния, горизонт, ограничения и допуски.
        candidates: Конечная последовательность входных импульсов в ECI.

    Returns:
        Статус, метрики сближения, выбранный импульс и журнал перебора.

    Raises:
        ValueError: Некорректная конфигурация или неверный тип кандидата.
    """
    candidates = tuple(candidates)
    if any(not isinstance(impulse, models.Impulse) for impulse in candidates):
        raise ValueError("Кандидаты должны быть экземплярами Impulse")

    span = (0.0, scenario.horizon_s)
    numerics = scenario.numerics

    def track(
        obj: models.SpaceObject, impulses: tuple[models.Impulse, ...] = ()
    ) -> models.Trajectory:
        """Траектория объекта на горизонте сценария.

        Args:
            obj: Объект с начальным состоянием в ECI.
            impulses: Импульсы спутника, не более одного.

        Returns:
            Траектория в ECI-J2000.
        """
        return propagate(
            obj, scenario.epoch, span, scenario.forces,
            numerics=numerics, impulses=impulses,
        )

    def approach(satellite: models.Trajectory) -> models.ConjunctionResult:
        """Сближение спутника с мусором на горизонте.

        Args:
            satellite: Траектория спутника.

        Returns:
            Результат с порогом и запасом сценария.
        """
        return find_closest_approach(
            satellite, debris, scenario.safe_distance_m, span, numerics=numerics
        )

    try:
        debris = track(scenario.debris)
        before = approach(track(scenario.satellite))
    except models.PropagationError as exc:
        return models.ManeuverResult(
            models.Status.PROPAGATION_FAILED, None, None, None,
            forces=scenario.forces, message=f"Сбой распространения: {exc}",
        )
    if before.accepted:
        return models.ManeuverResult(
            models.Status.ALREADY_SAFE, None, before, before,
            forces=scenario.forces,
        )

    limits = scenario.limits
    journal = []
    for impulse in candidates:
        if not limits.t_earliest_s <= impulse.t_s <= limits.t_latest_s:
            journal.append(models.CandidateResult(
                impulse, False, None, "Время импульса вне окна манёвра"
            ))
            continue
        if impulse.dv_mps > limits.dv_max_mps:
            journal.append(models.CandidateResult(
                impulse, False, None, "Превышен предел Δv"
            ))
            continue
        try:
            after = approach(track(scenario.satellite, (impulse,)))
        except models.PropagationError as exc:
            journal.append(models.CandidateResult(
                impulse, False, None, f"Сбой распространения: {exc}"
            ))
            continue
        admissible = after.accepted
        reason = (
            "" if admissible
            else "Минимальная дистанция меньше порога с запасом"
        )
        journal.append(
            models.CandidateResult(impulse, admissible, after, reason)
        )

    evaluated = tuple(journal)
    best = min(
        (item for item in evaluated if item.admissible),
        key=lambda item: (item.impulse.dv_mps, item.impulse.t_s),
        default=None,
    )
    if best is None:
        return models.ManeuverResult(
            models.Status.NO_FEASIBLE_CANDIDATE, None, before, None,
            forces=scenario.forces, candidates=evaluated,
        )
    return models.ManeuverResult(
        models.Status.FEASIBLE, best.impulse, before, best.after,
        forces=scenario.forces, candidates=evaluated,
    )
