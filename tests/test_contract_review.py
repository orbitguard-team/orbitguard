"""Регрессии ревью OG-01: типы, журнал и запас безопасной дистанции."""

import dataclasses
import datetime

import pytest

from orbitguard import models


def _scenario():
    """Создаёт синтетический вход без физического прогноза."""
    obj = models.SpaceObject("test", [7e6, 0, 0, 0, 7500, 0])
    return models.Scenario(
        "test", datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc),
        100, obj, obj, 100, models.ManeuverLimits(1, 0, 50),
    )


@pytest.mark.parametrize("changes", [
    {"epoch": "2026-01-01"}, {"satellite": {}}, {"debris": None},
    {"forces": None}, {"limits": (1, 0, 10)}, {"numerics": {}},
    {"name": 123}, {"source": []},
])
def test_invalid_nested_inputs_raise_value_error(changes: dict):
    """Неверные составные поля дают ValueError, а не AttributeError.

    Args:
        changes: Подмена одного поля корректного сценария.
    """
    with pytest.raises(ValueError):
        dataclasses.replace(_scenario(), **changes)


def test_object_name_must_be_string():
    """Число не принимается вместо имени объекта."""
    with pytest.raises(ValueError):
        models.SpaceObject(123, [7e6, 0, 0, 0, 7500, 0])


@pytest.mark.parametrize("distance_m, accepted", [
    (99, False), (100, False), (101, False), (102.5, True), (103, True),
])
def test_success_requires_the_configured_margin(
    distance_m: float, accepted: bool
):
    """Оба успешных статуса требуют именно заданный запас, включая границу.

    Args:
        distance_m: Проверяемая минимальная дистанция, м.
        accepted: Ожидаемая допустимость с запасом 2.5 м.
    """
    approach = models.Approach(0, distance_m, 1)
    result = models.ConjunctionResult(approach, (approach,), 100, 2.5)
    assert result.accepted == accepted
    assert result.violated == (distance_m < 100)
    forces = models.ForceModel(j2=True)
    burn = models.Impulse(0, [0, 1, 0])
    for status, best in ((models.Status.ALREADY_SAFE, None),
                         (models.Status.FEASIBLE, burn)):
        if accepted:
            output = models.ManeuverResult(
                status, best, result, result, forces)
            assert output.forces == forces
        else:
            with pytest.raises(ValueError, match="запаса"):
                models.ManeuverResult(status, best, result, result, forces)
    if not accepted:
        with pytest.raises(ValueError, match="запас"):
            models.CandidateResult(burn, True, result)


def test_forces_are_required_and_validated_for_failed_result():
    """Даже неуспешный расчёт явно сохраняет конфигурацию сил."""
    # Проверяем намеренное отсутствие обязательного аргумента.
    # pylint: disable=no-value-for-parameter
    with pytest.raises(TypeError):
        models.ManeuverResult(models.Status.PROPAGATION_FAILED,
                              None, None, None)
    with pytest.raises(ValueError, match="forces"):
        models.ManeuverResult(models.Status.PROPAGATION_FAILED,
                              None, None, None, None)


def test_minima_and_journal_are_copied_to_tuples():
    """Изменение исходных списков не изменяет frozen-результат."""
    first, last = models.Approach(0, 102, 1), models.Approach(10, 103, 1)
    minima = [first, last]
    result = models.ConjunctionResult(first, minima, 100)
    minima.clear()
    assert result.all_minima == (first, last)
    candidate = models.CandidateResult(models.Impulse(0, [0, 1, 0]),
                                       True, result)
    candidates, warnings = [candidate], ["пример предупреждения"]
    output = models.ManeuverResult(
        models.Status.FEASIBLE, candidate.impulse, result, result,
        models.ForceModel(), candidates, warnings=warnings)
    candidates.clear()
    warnings.clear()
    assert output.candidates == (candidate,)
    assert output.warnings == ("пример предупреждения",)


@pytest.mark.parametrize("case", ["empty", "duplicate", "reverse", "wrong",
                                       "tie", "missing"])
def test_inconsistent_minima_are_rejected(case: str):
    """Нельзя выбрать неверный минимум или неоднозначный порядок времён.

    Args:
        case: Нарушаемое условие списка минимумов.
    """
    first = models.Approach(0, 101, 1)
    last = models.Approach(10, 102, 1)
    closest, minima = {
        "empty": (first, ()),
        "duplicate": (first, (first, first)),
        "reverse": (first, (last, first)),
        "wrong": (last, (first, last)),
        "tie": (dataclasses.replace(last, distance_m=101),
                (first, dataclasses.replace(last, distance_m=101))),
        "missing": (first, (last,)),
    }[case]
    with pytest.raises(ValueError):
        models.ConjunctionResult(closest, minima, 100)


def test_before_after_must_use_same_threshold_and_margin():
    """Успех нельзя получить уменьшением порога или запаса после манёвра."""
    approach = models.Approach(0, 105, 1)
    before = models.ConjunctionResult(approach, (approach,), 100, 2)
    for changes in ({"safe_distance_m": 99}, {"distance_margin_m": 1}):
        with pytest.raises(ValueError, match="совпадать"):
            models.ManeuverResult(
                models.Status.FEASIBLE, models.Impulse(0, [0, 1, 0]),
                before, dataclasses.replace(before, **changes),
                models.ForceModel())
