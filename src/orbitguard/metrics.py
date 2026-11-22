"""Метрики манёвра и JSON-отчёт с входными данными для повторного расчёта."""

import json
from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime

import numpy as np

from .models import ConjunctionResult, Impulse, ManeuverResult, Scenario, Status, finite


def _closest(result: ConjunctionResult | None) -> dict | None:
    if result is None:
        return None
    return {
        "d_min_m": result.closest.distance_m,
        "t_min_s": result.closest.t_s,
        "rel_speed_mps": result.closest.rel_speed_mps,
        "violated": result.violated,
    }


def build_metrics(result: ManeuverResult, *, compute_time_s: float) -> dict:
    """Собирает метрики результата. Длительность передаёт вызывающий код.

    Времена сближения и импульса отсчитываются от эпохи сценария.
    Вектор импульса задан в той же системе координат, что и сценарий.
    """
    finite(compute_time_s, "compute_time_s")
    impulse = result.best
    if impulse is not None:
        delta_v = impulse.dv_eci_mps.tolist()
        delta_v_norm = impulse.dv_mps
        impulse_time = impulse.t_s
    elif result.status is Status.ALREADY_SAFE:
        delta_v, delta_v_norm, impulse_time = [0.0, 0.0, 0.0], 0.0, None
    else:
        delta_v = delta_v_norm = impulse_time = None

    return {
        "status": result.status.value,
        "before": _closest(result.before),
        "after": _closest(result.after),
        "dv_eci_mps": delta_v,
        "dv_mps": delta_v_norm,
        "impulse_t_s": impulse_time,
        "compute_time_s": float(compute_time_s),
    }


def _json_value(value: object) -> object:
    if isinstance(value, datetime):
        return value.isoformat().removesuffix("+00:00") + "Z"
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Тип {type(value).__name__} нельзя сохранить в отчёте")


def export_result_json(
    scenario: Scenario,
    result: ManeuverResult,
    *,
    candidates: Sequence[Impulse],
    compute_time_s: float,
) -> str:
    """Формирует отчёт v1. Передавайте исходных кандидатов в порядке поиска.

    Полный список кандидатов нужен даже при досрочном завершении поиска.
    Вложенный сценарий использует формат входного сценария v1.
    """
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
        "candidates": [asdict(candidate) for candidate in candidates],
        "metrics": build_metrics(result, compute_time_s=compute_time_s),
        "message": result.message,
    }
    return json.dumps(
        report, default=_json_value, ensure_ascii=False, allow_nan=False, indent=2
    ) + "\n"
