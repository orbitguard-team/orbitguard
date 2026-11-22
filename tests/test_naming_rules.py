"""Регрессионные проверки самой логики проверки имён.

Работают без orbitguard.models: проверяют naming_rules на примерах.
"""

import dataclasses

import numpy as np
from numpy import typing as npt
import pytest

import naming_rules

Array = npt.NDArray[np.float64]


@dataclasses.dataclass(frozen=True)
class _GoodFields:
    """Все числовые поля названы по правилам."""

    name: str
    miss_distance_m: float
    dv_eci_mps: Array
    t_s: Array | None = None
    rtol: float = 1e-10
    cd: float | None = None
    enabled: bool = True


@dataclasses.dataclass(frozen=True)
class _BadFields:
    """Числовые поля без суффикса единицы."""

    distance: float
    velocity: Array
    times: Array | None = None
    count: int = 0


@pytest.mark.parametrize("annotation", [
    float, int, np.ndarray, Array, Array | None, float | None,
    npt.NDArray[np.int64],
], ids=str)
def test_numeric_annotations_detected(annotation: object):
    """Числа и массивы, в том числе NDArray и Optional, распознаются.

    Args:
        annotation: Проверяемая аннотация.
    """
    assert naming_rules.is_numeric(annotation)


@pytest.mark.parametrize("annotation", [str, bool, bool | None, list[str]],
                         ids=str)
def test_non_numeric_annotations_ignored(annotation: object):
    """Строки, флаги и списки не считаются числовыми полями.

    Args:
        annotation: Проверяемая аннотация.
    """
    assert not naming_rules.is_numeric(annotation)


def test_good_fields_pass():
    """Поля с суффиксами и согласованные имена (rtol, cd) проходят."""
    assert not naming_rules.fields_without_unit(_GoodFields)


def test_bad_fields_detected_including_ndarray():
    """Поля без суффикса находятся, включая NDArray и Optional NDArray."""
    assert naming_rules.fields_without_unit(_BadFields) == [
        "distance", "velocity", "times", "count"]
