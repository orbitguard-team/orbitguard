"""Соглашения об именах в структурах orbitguard.models.

Числовые поля несут единицу в суффиксе имени, структуры неизменяемы.
Включается автоматически, когда есть orbitguard.models (OG-01).
"""

import dataclasses
import inspect

import pytest

import naming_rules

models = pytest.importorskip(
    "orbitguard.models", reason="Модуля models ещё нет: ждёт OG-01 (#1)")


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
    """У числовых полей есть суффикс единицы или согласованное имя.

    Args:
        cls: Проверяемая структура.
    """
    bad = naming_rules.fields_without_unit(cls)
    assert not bad, f"{cls.__name__}: поля без единицы {bad}"
