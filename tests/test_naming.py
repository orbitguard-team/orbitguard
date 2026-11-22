"""Проверка соглашений об именах в структурах данных (docs/DEV_GUIDE.md).

Числовые поля несут единицу или систему координат в суффиксе имени,
структуры неизменяемы. Включается автоматически, когда появится
orbitguard.models (OG-01).
"""

import dataclasses
import inspect
import typing

import numpy as np
import pytest

models = pytest.importorskip(
    "orbitguard.models", reason="Модуля models ещё нет: ждёт OG-01 (#1)")

UNIT_SUFFIXES = ("_m", "_mps", "_mps2", "_s", "_kg", "_m2", "_rad", "_deg",
                 "_kgpm3", "_m3ps2", "_radps", "_m2pkg", "_eci", "_rtn")
DIMENSIONLESS = {"cd", "ecc", "j2", "density_scale"}


def _dataclasses() -> list[type]:
    """Все dataclass-структуры модуля models.

    Returns:
        Классы, объявленные в orbitguard.models.
    """
    return [
        cls for _, cls in inspect.getmembers(models, inspect.isclass)
        if dataclasses.is_dataclass(cls)
        and cls.__module__ == models.__name__
    ]


def _is_numeric(annotation: object) -> bool:
    """Аннотация описывает число или массив.

    Args:
        annotation: Тип поля.

    Returns:
        True для float, int и np.ndarray, в том числе с None.
    """
    args = typing.get_args(annotation) or (annotation,)
    return any(arg in (float, int, np.ndarray) for arg in args)


def test_models_has_dataclasses():
    """В models есть структуры данных."""
    assert _dataclasses()


@pytest.mark.parametrize("cls", _dataclasses(), ids=lambda c: c.__name__)
def test_dataclass_is_frozen(cls: type):
    """Структуры данных неизменяемы.

    Args:
        cls: Проверяемая структура.
    """
    assert cls.__dataclass_params__.frozen, f"{cls.__name__} не frozen"


@pytest.mark.parametrize("cls", _dataclasses(), ids=lambda c: c.__name__)
def test_numeric_fields_have_unit_suffix(cls: type):
    """У числовых полей есть суффикс единицы или системы координат.

    Args:
        cls: Проверяемая структура.
    """
    hints = typing.get_type_hints(cls)
    bad = [
        field.name for field in dataclasses.fields(cls)
        if _is_numeric(hints[field.name])
        and field.name not in DIMENSIONLESS
        and not field.name.endswith(UNIT_SUFFIXES)
    ]
    assert not bad, f"{cls.__name__}: поля без единицы {bad}"
