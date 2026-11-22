"""Загрузка и сохранение сценариев OrbitGuard в JSON (OG-06, #6).

Формат — schema_version 1 из docs/contracts.md: все поля Scenario записаны
явно, СИ, ECI-J2000, эпоха в UTC. Дополнительно файл может содержать
необязательный блок expected — ожидаемый результат расчёта и его
обоснование (docs/scenarios.md). Блок не влияет на сам расчёт.

Загрузчик строгий: неизвестные и пропущенные поля, повторные ключи,
NaN/Infinity, неверная версия, система координат или единицы дают
ValueError с указанием файла и поля.
"""

import dataclasses
import datetime
import json
import math
import pathlib
import re
from collections.abc import Mapping
from numbers import Real
from typing import Any

from orbitguard import models

SCHEMA_VERSION = 1
FRAME = "ECI-J2000"
# Каталог сценариев в корне репозитория (установка pip install -e).
SCENARIO_DIR = pathlib.Path(__file__).resolve().parents[2] / "scenarios"

_TOP_KEYS = frozenset({
    "schema_version", "name", "epoch", "frame", "horizon_s", "satellite",
    "debris", "safe_distance_m", "limits", "forces", "numerics",
    "synthetic", "source",
})
_OPTIONAL_TOP_KEYS = frozenset({"expected"})
_EXPECTED_KEYS = frozenset({"status", "closest_approach", "tolerance",
                            "basis"})
_OPTIONAL_EXPECTED_KEYS = frozenset({"witness_impulse", "shift_bound_m"})
_APPROACH_KEYS = frozenset({"t_s", "distance_m", "rel_speed_mps"})
_TOLERANCE_KEYS = frozenset({"t_s", "distance_m"})
_WITNESS_KEYS = frozenset({"t_s", "dv_eci_mps", "min_distance_m"})


def _field_names(cls: type) -> frozenset[str]:
    return frozenset(field.name for field in dataclasses.fields(cls))


@dataclasses.dataclass(frozen=True)
class Witness:
    """Пример допустимого импульса — доказательство статуса feasible.

    Attributes:
        impulse: Импульс в ECI в пределах окна и dv_max.
        min_distance_m: Минимальная дистанция на всём горизонте после
            импульса, м.
    """

    impulse: models.Impulse
    min_distance_m: float


@dataclasses.dataclass(frozen=True)
class Expectation:
    """Ожидаемый результат сценария и его обоснование.

    Attributes:
        status: Ожидаемый статус search_maneuver.
        closest: Ближайшее сближение без манёвра на [0, horizon_s]
            по построению сценария (задача двух тел, аналитически).
        time_tol_s: Допуск сравнения момента сближения, с.
        distance_tol_m: Допуск сравнения дистанции, м.
        basis: Почему ожидается именно этот статус.
        witness: Для feasible — пример допустимого импульса.
        shift_bound_m: Для no_feasible_candidate — верхняя граница
            смещения спутника в момент исходного сближения при любом
            импульсе в пределах окна и dv_max, м.
    """

    status: models.Status
    closest: models.Approach
    time_tol_s: float
    distance_tol_m: float
    basis: str
    witness: Witness | None = None
    shift_bound_m: float | None = None


def _context(path: str | pathlib.Path) -> str:
    return f"Сценарий {pathlib.Path(path).name}"


def _reject_constant(token: str) -> float:
    raise ValueError(f"недопустимое число {token}: нужны конечные значения")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"повторный ключ «{key}»")
        result[key] = value
    return result


def _check_keys(
    data: Any,
    required: frozenset[str],
    where: str,
    optional: frozenset[str] = frozenset(),
) -> Mapping[str, Any]:
    if not isinstance(data, dict):
        raise ValueError(f"{where}: ожидается объект JSON")
    missing = sorted(required - data.keys())
    unknown = sorted(data.keys() - required - optional)
    if missing:
        raise ValueError(f"{where}: нет полей {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{where}: неизвестные поля {', '.join(unknown)}")
    return data


def _number(value: Any, where: str) -> float:
    if (not isinstance(value, Real) or isinstance(value, bool)
            or not math.isfinite(value)):
        raise ValueError(f"{where}: ожидается конечное число")
    return float(value)


def _optional_number(value: Any, where: str) -> float | None:
    return None if value is None else _number(value, where)


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where}: ожидается непустая строка")
    return value


def _vector(value: Any, size: int, where: str) -> list[float]:
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{where}: ожидается список из {size} чисел")
    return [_number(item, f"{where}[{index}]")
            for index, item in enumerate(value)]


def _parse_epoch(value: Any, where: str) -> datetime.datetime:
    text = _text(value, where)
    try:
        epoch = datetime.datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{where}: не дата ISO 8601: {text}") from exc
    if epoch.tzinfo is None or epoch.utcoffset() != datetime.timedelta(0):
        raise ValueError(f"{where}: эпоха должна быть в UTC (суффикс Z)")
    return epoch.astimezone(datetime.timezone.utc)


def format_epoch(epoch: datetime.datetime) -> str:
    """Эпоха в ISO 8601 UTC с суффиксом Z, как в файлах сценариев.

    Args:
        epoch: Дата и время с часовым поясом UTC.

    Returns:
        Строка вида 2026-12-01T00:00:00Z.
    """
    return epoch.astimezone(datetime.timezone.utc).isoformat().replace(
        "+00:00", "Z")


def _parse_object(data: Any, where: str) -> models.SpaceObject:
    data = _check_keys(data, _field_names(models.SpaceObject), where)
    return models.SpaceObject(
        name=_text(data["name"], f"{where}.name"),
        state0=_vector(data["state0"], 6, f"{where}.state0"),
        mass_kg=_optional_number(data["mass_kg"], f"{where}.mass_kg"),
        area_m2=_optional_number(data["area_m2"], f"{where}.area_m2"),
        cd=_optional_number(data["cd"], f"{where}.cd"),
    )


def _parse_forces(data: Any, where: str) -> models.ForceModel:
    data = _check_keys(data, _field_names(models.ForceModel), where)
    flags = {}
    for name in ("j2", "drag", "moon", "sun"):
        if not isinstance(data[name], bool):
            raise ValueError(f"{where}.{name}: ожидается true или false")
        flags[name] = data[name]
    return models.ForceModel(
        density_scale=_number(data["density_scale"],
                              f"{where}.density_scale"),
        **flags,
    )


def _parse_numbers(data: Any, cls: type, where: str) -> Any:
    data = _check_keys(data, _field_names(cls), where)
    return cls(**{name: _number(value, f"{where}.{name}")
                  for name, value in data.items()})


def _parse_scenario(data: Any) -> models.Scenario:
    data = _check_keys(data, _TOP_KEYS, "файл", _OPTIONAL_TOP_KEYS)
    version = data["schema_version"]
    if version != SCHEMA_VERSION or isinstance(version, bool):
        raise ValueError(
            f"schema_version: поддерживается {SCHEMA_VERSION}, "
            f"получено {version!r}")
    if data["frame"] != FRAME:
        raise ValueError(f"frame: поддерживается только {FRAME}")
    if not isinstance(data["synthetic"], bool):
        raise ValueError("synthetic: ожидается true или false")
    return models.Scenario(
        name=_text(data["name"], "name"),
        epoch=_parse_epoch(data["epoch"], "epoch"),
        horizon_s=_number(data["horizon_s"], "horizon_s"),
        satellite=_parse_object(data["satellite"], "satellite"),
        debris=_parse_object(data["debris"], "debris"),
        safe_distance_m=_number(data["safe_distance_m"], "safe_distance_m"),
        limits=_parse_numbers(data["limits"], models.ManeuverLimits,
                              "limits"),
        forces=_parse_forces(data["forces"], "forces"),
        numerics=_parse_numbers(data["numerics"], models.Numerics,
                                "numerics"),
        synthetic=data["synthetic"],
        source=_text(data["source"], "source"),
        frame=data["frame"],
    )


def _parse_status(value: Any) -> models.Status:
    try:
        return models.Status(value)
    except ValueError as exc:
        allowed = ", ".join(status.value for status in models.Status)
        raise ValueError(
            f"expected.status: {value!r} не из {allowed}") from exc


def _parse_expectation(data: Any) -> Expectation:
    where = "expected"
    data = _check_keys(data, _EXPECTED_KEYS, where, _OPTIONAL_EXPECTED_KEYS)
    closest = _check_keys(data["closest_approach"], _APPROACH_KEYS,
                          f"{where}.closest_approach")
    tolerance = _check_keys(data["tolerance"], _TOLERANCE_KEYS,
                            f"{where}.tolerance")
    witness = None
    if "witness_impulse" in data:
        raw = _check_keys(data["witness_impulse"], _WITNESS_KEYS,
                          f"{where}.witness_impulse")
        witness = Witness(
            models.Impulse(
                _number(raw["t_s"], f"{where}.witness_impulse.t_s"),
                _vector(raw["dv_eci_mps"], 3,
                        f"{where}.witness_impulse.dv_eci_mps")),
            _number(raw["min_distance_m"],
                    f"{where}.witness_impulse.min_distance_m"))
    shift_bound_m = None
    if "shift_bound_m" in data:
        shift_bound_m = _number(data["shift_bound_m"],
                                f"{where}.shift_bound_m")
    approach = models.Approach(
        *(_number(closest[name], f"{where}.closest_approach.{name}")
          for name in ("t_s", "distance_m", "rel_speed_mps")))
    return Expectation(
        status=_parse_status(data["status"]),
        closest=approach,
        time_tol_s=_number(tolerance["t_s"], f"{where}.tolerance.t_s"),
        distance_tol_m=_number(tolerance["distance_m"],
                               f"{where}.tolerance.distance_m"),
        basis=_text(data["basis"], f"{where}.basis"),
        witness=witness,
        shift_bound_m=shift_bound_m,
    )


def _check_expectation(scenario: models.Scenario,
                       expected: Expectation) -> None:
    """Проверяет, что ожидание не противоречит параметрам сценария.

    Критерий безопасности — как в contracts.md: d ≥ safe + margin.

    Raises:
        ValueError: Ожидание внутренне противоречиво.
    """
    required_m = (scenario.safe_distance_m
                  + scenario.numerics.distance_margin_m)
    closest_m = expected.closest.distance_m
    limits = scenario.limits
    if expected.time_tol_s <= 0 or expected.distance_tol_m <= 0:
        raise ValueError("expected.tolerance: допуски должны быть > 0")
    if not 0 <= expected.closest.t_s <= scenario.horizon_s:
        raise ValueError("expected.closest_approach.t_s: вне горизонта")
    status = expected.status
    has_witness = expected.witness is not None
    has_bound = expected.shift_bound_m is not None
    if has_witness != (status == models.Status.FEASIBLE):
        raise ValueError(
            "expected.witness_impulse: нужен только для статуса feasible")
    if has_bound != (status == models.Status.NO_FEASIBLE_CANDIDATE):
        raise ValueError("expected.shift_bound_m: нужен только для "
                         "статуса no_feasible_candidate")
    if status == models.Status.ALREADY_SAFE and closest_m < required_m:
        raise ValueError(
            f"expected: already_safe, но дистанция {closest_m:.3f} м "
            f"меньше порога с запасом {required_m:.3f} м")
    if (status in (models.Status.FEASIBLE,
                   models.Status.NO_FEASIBLE_CANDIDATE)
            and closest_m >= required_m):
        raise ValueError(
            f"expected: {status.value}, но сближение {closest_m:.3f} м "
            f"не опасно при пороге с запасом {required_m:.3f} м")
    if has_witness:
        impulse = expected.witness.impulse
        if not limits.t_earliest_s <= impulse.t_s <= limits.t_latest_s:
            raise ValueError("expected.witness_impulse.t_s: вне окна манёвра")
        if impulse.dv_mps > limits.dv_max_mps:
            raise ValueError("expected.witness_impulse: Δv больше dv_max")
        if expected.witness.min_distance_m < required_m:
            raise ValueError("expected.witness_impulse: импульс не даёт "
                             "безопасной дистанции")
    if has_bound and closest_m + expected.shift_bound_m >= required_m:
        raise ValueError(
            "expected.shift_bound_m: оценка не доказывает отсутствие "
            "решения (дистанция + смещение ≥ порога с запасом)")


def _read(path: str | pathlib.Path) -> dict[str, Any]:
    try:
        text = pathlib.Path(path).read_text(encoding="utf-8")
        return json.loads(text, parse_constant=_reject_constant,
                          object_pairs_hook=_unique_object)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{_context(path)}: неверный JSON: {exc}") from exc
    except ValueError as exc:
        raise ValueError(f"{_context(path)}: {exc}") from exc


def load_scenario_file(
    path: str | pathlib.Path,
) -> tuple[models.Scenario, Expectation | None]:
    """Загружает сценарий и, если есть, блок ожидаемого результата.

    Args:
        path: Путь к JSON-файлу сценария.

    Returns:
        Проверенный Scenario и Expectation или None, если блока нет.

    Raises:
        ValueError: Файл не соответствует формату или противоречив.
        OSError: Файл нельзя прочитать.
    """
    data = _read(path)
    try:
        scenario = _parse_scenario(data)
        expected = None
        if "expected" in data:
            expected = _parse_expectation(data["expected"])
            _check_expectation(scenario, expected)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{_context(path)}: {exc}") from exc
    return scenario, expected


def load_scenario(path: str | pathlib.Path) -> models.Scenario:
    """Загружает сценарий из JSON без ручного ввода (контракт OG-01).

    Args:
        path: Путь к JSON-файлу сценария, schema_version 1.

    Returns:
        Проверенный Scenario: СИ, ECI-J2000, эпоха в UTC.

    Raises:
        ValueError: Файл не соответствует формату или противоречив.
        OSError: Файл нельзя прочитать.
    """
    return load_scenario_file(path)[0]


def load_expectation(path: str | pathlib.Path) -> Expectation | None:
    """Возвращает блок expected сценария или None, если его нет.

    Args:
        path: Путь к JSON-файлу сценария.

    Returns:
        Ожидаемый результат с обоснованием.

    Raises:
        ValueError: Файл не соответствует формату или противоречив.
    """
    return load_scenario_file(path)[1]


def scenario_paths(
    directory: str | pathlib.Path = SCENARIO_DIR,
) -> list[pathlib.Path]:
    """Возвращает отсортированный список файлов *.json каталога.

    Args:
        directory: Каталог сценариев; по умолчанию scenarios/ репозитория.

    Returns:
        Пути к файлам сценариев в алфавитном порядке.
    """
    return sorted(pathlib.Path(directory).glob("*.json"))


def _object_to_dict(obj: models.SpaceObject) -> dict[str, Any]:
    return {
        "name": obj.name,
        "state0": [float(value) for value in obj.state0],
        "mass_kg": obj.mass_kg,
        "area_m2": obj.area_m2,
        "cd": obj.cd,
    }


def _expectation_to_dict(expected: Expectation) -> dict[str, Any]:
    result = {
        "status": expected.status.value,
        "closest_approach": {
            "t_s": expected.closest.t_s,
            "distance_m": expected.closest.distance_m,
            "rel_speed_mps": expected.closest.rel_speed_mps,
        },
        "tolerance": {"t_s": expected.time_tol_s,
                      "distance_m": expected.distance_tol_m},
        "basis": expected.basis,
    }
    if expected.witness is not None:
        impulse = expected.witness.impulse
        result["witness_impulse"] = {
            "t_s": float(impulse.t_s),
            "dv_eci_mps": [float(value) for value in impulse.dv_eci_mps],
            "min_distance_m": expected.witness.min_distance_m,
        }
    if expected.shift_bound_m is not None:
        result["shift_bound_m"] = expected.shift_bound_m
    return result


def scenario_to_dict(
    scenario: models.Scenario,
    expected: Expectation | None = None,
) -> dict[str, Any]:
    """Преобразует сценарий в словарь формата JSON schema_version 1.

    Args:
        scenario: Проверенный сценарий.
        expected: Необязательный блок ожидаемого результата.

    Returns:
        Словарь, который load_scenario_file читает обратно без потерь.
    """
    result = {
        "schema_version": SCHEMA_VERSION,
        "name": scenario.name,
        "epoch": format_epoch(scenario.epoch),
        "frame": scenario.frame,
        "horizon_s": float(scenario.horizon_s),
        "satellite": _object_to_dict(scenario.satellite),
        "debris": _object_to_dict(scenario.debris),
        "safe_distance_m": float(scenario.safe_distance_m),
        "limits": dataclasses.asdict(scenario.limits),
        "forces": dataclasses.asdict(scenario.forces),
        "numerics": dataclasses.asdict(scenario.numerics),
        "synthetic": scenario.synthetic,
        "source": scenario.source,
    }
    if expected is not None:
        _check_expectation(scenario, expected)
        result["expected"] = _expectation_to_dict(expected)
    return result


def _collapse_number_lists(text: str) -> str:
    # Векторы печатаются в одну строку: так файл удобнее читать и сравнивать.
    pattern = re.compile(r"\[\s+([-+0-9.eE,\s]+?)\s+\]")
    return pattern.sub(
        lambda match: "[" + ", ".join(
            item.strip() for item in match.group(1).split(",")) + "]",
        text)


def dumps_scenario(
    scenario: models.Scenario,
    expected: Expectation | None = None,
) -> str:
    """Возвращает канонический текст JSON сценария.

    Числа записываются кратчайшим точным представлением float64,
    поэтому повторная загрузка даёт те же значения бит в бит.

    Args:
        scenario: Проверенный сценарий.
        expected: Необязательный блок ожидаемого результата.

    Returns:
        Текст JSON с отступом 2 и переводом строки в конце.
    """
    data = scenario_to_dict(scenario, expected)
    text = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)
    return _collapse_number_lists(text) + "\n"


def save_scenario(
    scenario: models.Scenario,
    path: str | pathlib.Path,
    expected: Expectation | None = None,
) -> None:
    """Сохраняет сценарий в JSON, пригодный для load_scenario.

    Args:
        scenario: Проверенный сценарий.
        path: Куда записать файл.
        expected: Необязательный блок ожидаемого результата.
    """
    pathlib.Path(path).write_text(dumps_scenario(scenario, expected),
                                  encoding="utf-8")
