"""Правила имён числовых полей (docs/DEV_GUIDE.md, раздел 3).

Числовое поле — float, int или массив NumPy, в том числе
numpy.typing.NDArray и варианты с None. Его имя должно оканчиваться
суффиксом единицы или системы координат либо быть в списке согласованных
исключений.
"""

import dataclasses
import types
import typing

import numpy as np

UNIT_SUFFIXES = ("_m", "_mps", "_mps2", "_s", "_kg", "_m2", "_rad", "_deg",
                 "_kgpm3", "_m3ps2", "_radps", "_m2pkg", "_eci", "_rtn")

# Согласованные исключения: величины без единицы или со смешанными
# единицами, у которых единицы описаны в docs/contracts.md.
AGREED_NAMES = frozenset({
    "cd",             # коэффициент сопротивления, безразмерный
    "ecc",            # эксцентриситет, безразмерный
    "j2",             # коэффициент J2, безразмерный
    "density_scale",  # множитель плотности, безразмерный
    "rtol",           # относительный допуск интегратора, безразмерный
    "state0",         # состояние (6,): м и м/с, система — поле frame
})

_UNION_TYPES = (typing.Union, types.UnionType)


def is_numeric(annotation: object) -> bool:
    """Проверяет, описывает ли аннотация число или массив.

    Args:
        annotation: Тип поля, например float, NDArray[np.float64]
            или NDArray[np.float64] | None.

    Returns:
        True для float, int, np.ndarray и NDArray, в том числе внутри
        Union и Optional.
    """
    if annotation in (float, int, np.ndarray):
        return True
    origin = typing.get_origin(annotation)
    if origin is np.ndarray:
        return True
    if origin in _UNION_TYPES:
        return any(is_numeric(arg) for arg in typing.get_args(annotation))
    # В новых версиях NumPy NDArray — псевдоним типа (PEP 695):
    # его origin — сам псевдоним, а ndarray лежит в __value__.
    alias_value = getattr(origin if origin is not None else annotation,
                          "__value__", None)
    if alias_value is not None:
        return is_numeric(alias_value)
    return False


def fields_without_unit(cls: type) -> list[str]:
    """Находит числовые поля без суффикса единицы.

    Args:
        cls: Класс-dataclass.

    Returns:
        Имена нарушающих полей.
    """
    hints = typing.get_type_hints(cls)
    return [
        field.name for field in dataclasses.fields(cls)
        if is_numeric(hints[field.name])
        and field.name not in AGREED_NAMES
        and not field.name.endswith(UNIT_SUFFIXES)
    ]
