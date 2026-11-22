"""Метрики манёвра и JSON-отчёт с входными данными для повторного расчёта."""

import json
import platform
import tomllib
from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import cast

import numpy as np

from . import models


def _closest(
    result: models.ConjunctionResult | None, margin: float | None
) -> dict[str, object] | None:
    if result is None:
        return None
    distance = result.closest.distance_m
    return {
        "d_min_m": distance,
        "t_min_s": result.closest.t_s,
        "rel_speed_mps": result.closest.rel_speed_mps,
        "violated": result.violated,
        "accepted": None if margin is None else (
            distance >= result.safe_distance_m + margin
        ),
    }


def summarize(
    result: models.ManeuverResult,
    *,
    scenario: models.Scenario | None = None,
    compute_time_s: float | None = None,
) -> dict[str, object]:
    """Собирает метрики для интерфейса в единицах СИ.

    Args:
        result: Результат поиска манёвра.
        scenario: Исходный сценарий для порога, запаса и настроек сил.
        compute_time_s: Измеренная вызывающим кодом длительность, в секундах.

    Returns:
        Метрики, статус и конфигурация. Неизвестные значения равны None.
        violated означает d_min ниже порога. accepted учитывает запас.

    Raises:
        ValueError: Длительность отрицательная или нечисленная.
    """
    if compute_time_s is not None:
        models.finite(compute_time_s, "compute_time_s")

    conjunction = result.before or result.after
    threshold = None if conjunction is None else conjunction.safe_distance_m
    margin = getattr(conjunction, "distance_margin_m", None)
    forces = getattr(result, "forces", None)
    if scenario is not None:
        threshold = scenario.safe_distance_m if threshold is None else threshold
        if margin is None:
            margin = scenario.numerics.distance_margin_m
        forces = scenario.forces if forces is None else forces

    impulse = result.best
    if impulse is not None:
        delta_v = impulse.dv_eci_mps.tolist()
        delta_v_norm = impulse.dv_mps
        impulse_time = impulse.t_s
    elif result.status is models.Status.ALREADY_SAFE:
        delta_v, delta_v_norm, impulse_time = [0.0, 0.0, 0.0], 0.0, None
    else:
        delta_v = delta_v_norm = impulse_time = None

    gain = None
    if result.before is not None and result.after is not None:
        before_distance = result.before.closest.distance_m
        gain = result.after.closest.distance_m - before_distance

    return {
        "status": result.status.value,
        "before": _closest(result.before, margin),
        "after": _closest(result.after, margin),
        "safe_distance_m": threshold,
        "distance_margin_m": margin,
        "d_min_gain_m": gain,
        "forces": asdict(forces) if forces is not None else None,
        "dv_eci_mps": delta_v,
        "dv_mps": delta_v_norm,
        "impulse_t_s": impulse_time,
        "compute_time_s": (
            float(compute_time_s) if compute_time_s is not None else None
        ),
    }


def build_metrics(
    result: models.ManeuverResult,
    *,
    compute_time_s: float,
    scenario: models.Scenario | None = None,
) -> dict[str, object]:
    """Собирает метрики с явно переданной длительностью расчёта.

    Args:
        result: Результат поиска манёвра.
        compute_time_s: Измеренная длительность расчёта, в секундах.
        scenario: Исходный сценарий для порога, запаса и настроек сил.

    Returns:
        Метрики в формате summarize.

    Raises:
        ValueError: Длительность отрицательная или нечисленная.
    """
    return summarize(result, scenario=scenario, compute_time_s=compute_time_s)


def _json_value(value: object) -> object:
    if isinstance(value, datetime):
        return value.isoformat().removesuffix("+00:00") + "Z"
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Тип {type(value).__name__} нельзя сохранить в отчёте")


def _input_candidates(
    result: models.ManeuverResult,
    candidates: Sequence[models.Impulse] | None,
) -> tuple[models.Impulse, ...] | None:
    logged = tuple(item.impulse for item in result.candidates)
    complete = result.status in (
        models.Status.FEASIBLE, models.Status.NO_FEASIBLE_CANDIDATE,
    )
    if result.status is models.Status.FEASIBLE and not logged:
        raise ValueError("В результате манёвра отсутствует журнал кандидатов")
    if candidates is None:
        return logged if complete else None

    given = tuple(candidates)
    same_length = len(given) == len(logged)
    same_prefix = len(given) >= len(logged) and all(
        expected.t_s == actual.t_s
        and np.array_equal(expected.dv_eci_mps, actual.dv_eci_mps)
        for expected, actual in zip(logged, given)
    )
    if not same_prefix or (complete and not same_length):
        raise ValueError("Входные импульсы не совпадают с журналом расчёта")
    return given


def _distribution_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _orbitguard_version() -> str | None:
    installed = _distribution_version("orbitguard")
    if installed is not None:
        return installed
    config = Path(__file__).resolve().parents[2] / "pyproject.toml"
    if not config.is_file():
        return None
    with config.open("rb") as stream:
        project = tomllib.load(stream).get("project", {})
    if project.get("name") == "orbitguard":
        return project.get("version")
    return None


def export_result_json(
    scenario: models.Scenario,
    result: models.ManeuverResult,
    *,
    candidates: Sequence[models.Impulse] | None = None,
    compute_time_s: float | None = None,
    git_commit: str | None = None,
) -> str:
    """Формирует воспроизводимый JSON-отчёт v1.

    Args:
        scenario: Полный исходный сценарий.
        result: Результат манёвра и журнал кандидатов.
        candidates: Входные импульсы в исходном порядке. Нужны для полного
            экспорта, если поиск завершился до перебора.
        compute_time_s: Измеренная длительность расчёта, в секундах.
        git_commit: Ревизия расчёта, если её знает вызывающий код.

    Returns:
        JSON-строка с настройками, метриками, журналом и версиями среды.

    Raises:
        ValueError: Входные импульсы расходятся с журналом или значения
            не могут быть представлены конечными JSON-числами.
    """
    inputs = _input_candidates(result, candidates)
    metrics = summarize(
        result, scenario=scenario, compute_time_s=compute_time_s,
    )
    margin = cast(float | None, metrics["distance_margin_m"])
    report = {
        "schema_version": 1,
        "units": {
            "position": "m",
            "velocity": "m/s",
            "time": "s",
            "mass": "kg",
            "area": "m^2",
        },
        "scenario": {"schema_version": 1, **asdict(scenario)},
        "candidates": (
            None if inputs is None else [asdict(item) for item in inputs]
        ),
        "candidate_results": [
            {
                "impulse": asdict(item.impulse),
                "admissible": item.admissible,
                "reason": item.reason,
                "after": _closest(item.after, margin),
            }
            for item in result.candidates
        ],
        "metrics": metrics,
        "message": result.message,
        "software": {
            "orbitguard": _orbitguard_version(),
            "git_commit": git_commit,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": _distribution_version("scipy"),
        },
    }
    return json.dumps(
        report, default=_json_value, ensure_ascii=False, allow_nan=False,
        indent=2,
    ) + "\n"


def export_json(
    scenario: models.Scenario,
    result: models.ManeuverResult,
    path: str | Path,
    *,
    candidates: Sequence[models.Impulse] | None = None,
    compute_time_s: float | None = None,
    git_commit: str | None = None,
) -> None:
    """Сохраняет отчёт в UTF-8 для файлового API.

    Args:
        scenario: Исходный сценарий.
        result: Результат поиска манёвра.
        path: Путь выходного JSON-файла.
        candidates: Полный набор входных импульсов, если он нужен для отчёта.
        compute_time_s: Измеренная длительность расчёта, в секундах.
        git_commit: Ревизия расчёта, если она известна.

    Raises:
        ValueError: Несогласованные импульсы или непредставимые числа.
        OSError: Ошибка записи файла.
    """
    report = export_result_json(
        scenario, result, candidates=candidates,
        compute_time_s=compute_time_s, git_commit=git_commit,
    )
    Path(path).write_text(report, encoding="utf-8")
