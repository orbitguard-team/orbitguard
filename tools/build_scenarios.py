"""Генератор синтетических сценариев OG-06 (#6) и проверка их обоснования.

Сценарии строятся обратным расчётом от заданной геометрии сближения:
задаются орбита спутника, момент сближения (TCA), орбита фрагмента и вектор
промаха в плоскости, перпендикулярной относительной скорости. Состояния
переносятся на эпоху сценария аналитически (tools/twobody_reference.py:
задача двух тел, универсальная переменная Кеплера), независимо от
численного интегратора OG-02.

Модель сил сценариев — центральное поле. При включении J2, атмосферы,
Луны или Солнца геометрия изменится: сценарий нужно строить заново с той же
моделью сил, что и в расчёте.

Запуск из корня репозитория:
    python tools/build_scenarios.py           # записать scenarios/*.json
    python tools/build_scenarios.py --check   # сравнить с файлами
    python tools/build_scenarios.py --report  # таблица проверок

Описание и источники параметров — docs/scenarios.md.
"""

import argparse
import dataclasses
import datetime
import math
import pathlib
import sys
from typing import Any

import numpy as np
from scipy import optimize

import twobody_reference as ref
from orbitguard import models, scenarios

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "scenarios"

# Допуски сравнения с численным расчётом: момент — критерий contracts.md;
# дистанция — квадратичный минимум при ошибке момента closest_time_tol_s/2
# и относительной скорости до 15 км/с даёт (v·δt)²/(2d) < 0.2 м при d≥100 м,
# остальное — запас на ошибку интегрирования (≪ 1 мм за 3 ч).
TIME_TOL_S = 0.01
DISTANCE_TOL_M = 1.0
# Совпадение файла и генератора: разные libm и порядок операций дают
# разницу в младших битах. Положения ~7e6 м после переноса на часы
# воспроизводятся до ~1e-7 м, поэтому допуск абсолютный — 1 мкм (и 1 мкм/с),
# физически пренебрежимый по сравнению с допуском сравнения 1 м.
CHECK_REL_TOL = 1e-12
CHECK_ABS_TOL = 1e-6


# ---------------------------------------------------------------------------
# Геометрия сближения
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class OrbitShape:
    """Форма и наклонение орбиты по высотам перигея и апогея над сферой."""

    perigee_alt_m: float
    apogee_alt_m: float
    inclination_deg: float

    @property
    def sma_m(self) -> float:
        """Большая полуось, м."""
        return ref.R_EARTH_M + (self.perigee_alt_m + self.apogee_alt_m) / 2.0

    @property
    def ecc(self) -> float:
        """Эксцентриситет."""
        return (self.apogee_alt_m - self.perigee_alt_m) / (2.0 * self.sma_m)


@dataclasses.dataclass(frozen=True)
class Encounter:
    """Сближение, заданное в момент TCA относительно эпохи EPOCH.

    Attributes:
        tca_s: Момент сближения, с от EPOCH.
        debris_orbit: Орбита фрагмента.
        debris_descending: Фрагмент в TCA движется к югу (иначе к северу).
        debris_outbound: Фрагмент удаляется от перигея (истинная
            аномалия в (0, π)).
        miss_m: Дистанция в TCA, м.
        miss_angle_deg: Угол вектора промаха в плоскости, перпендикулярной
            относительной скорости, от её «радиальной» оси, град.
    """

    tca_s: float
    debris_orbit: OrbitShape
    debris_descending: bool
    debris_outbound: bool
    miss_m: float
    miss_angle_deg: float


def _plane_through_point(unit: np.ndarray, inclination_rad: float,
                         descending: bool) -> tuple[float, float]:
    """Долгота узла и аргумент широты плоскости через точку unit.

    Args:
        unit: Единичный вектор положения (3,), ECI.
        inclination_rad: Наклонение плоскости, рад.
        descending: Движение к югу в этой точке.

    Returns:
        (raan_rad, arg_latitude_rad).

    Raises:
        ValueError: Широта точки больше наклонения.
    """
    sin_u = unit[2] / math.sin(inclination_rad)
    if abs(sin_u) > 1.0:
        raise ValueError("Широта точки недостижима для этого наклонения")
    arg_lat = math.asin(sin_u)
    if descending:
        arg_lat = math.pi - arg_lat
    raan = (math.atan2(unit[1], unit[0])
            - math.atan2(math.sin(arg_lat) * math.cos(inclination_rad),
                         math.cos(arg_lat)))
    return raan, arg_lat


def _miss_offset(sat_tca: np.ndarray, debris_v_mps: np.ndarray,
                 enc: Encounter) -> np.ndarray:
    """Вектор промаха (фрагмент минус спутник) ⟂ относительной скорости.

    Args:
        sat_tca: Состояние спутника в TCA (6,).
        debris_v_mps: Скорость фрагмента в TCA (3,), м/с.
        enc: Параметры сближения.

    Returns:
        Смещение (3,), м, |смещение| = enc.miss_m.
    """
    unit = sat_tca[:3] / np.linalg.norm(sat_tca[:3])
    axis_v = sat_tca[3:] - debris_v_mps
    axis_v = axis_v / np.linalg.norm(axis_v)
    axis_r = unit - (unit @ axis_v) * axis_v
    axis_r = axis_r / np.linalg.norm(axis_r)
    angle = math.radians(enc.miss_angle_deg)
    return enc.miss_m * (math.cos(angle) * axis_r
                         + math.sin(angle) * np.cross(axis_v, axis_r))


def debris_state_at_tca(sat_tca: np.ndarray, enc: Encounter) -> np.ndarray:
    """Строит состояние фрагмента в TCA по состоянию спутника.

    Плоскость орбиты фрагмента проходит через точку спутника с заданным
    наклонением; радиус совпадает за счёт выбора истинной аномалии.
    Затем фрагмент сдвигается на вектор промаха, перпендикулярный
    относительной скорости, поэтому r_rel·v_rel = 0 в TCA точно.

    Args:
        sat_tca: Состояние спутника в TCA (6,).
        enc: Параметры сближения.

    Returns:
        Состояние фрагмента в TCA (6,), м и м/с.

    Raises:
        ValueError: Радиус спутника вне диапазона орбиты фрагмента.
    """
    orbit = enc.debris_orbit
    radius_m = float(np.linalg.norm(sat_tca[:3]))
    inclination = math.radians(orbit.inclination_deg)
    raan, arg_lat = _plane_through_point(
        sat_tca[:3] / radius_m, inclination, enc.debris_descending)
    cos_nu = (orbit.sma_m * (1.0 - orbit.ecc**2) / radius_m - 1.0) / orbit.ecc
    if abs(cos_nu) > 1.0:
        raise ValueError("Радиус спутника вне диапазона орбиты фрагмента")
    true_anomaly = math.acos(cos_nu) * (1.0 if enc.debris_outbound else -1.0)
    debris = ref.elements_to_state(orbit.sma_m, orbit.ecc, inclination,
                                   raan, arg_lat - true_anomaly,
                                   true_anomaly)
    offset = _miss_offset(sat_tca, debris[3:], enc)
    return np.concatenate([sat_tca[:3] + offset, debris[3:]])


# ---------------------------------------------------------------------------
# Сценарии
# ---------------------------------------------------------------------------

EPOCH = datetime.datetime(2026, 12, 1, tzinfo=datetime.timezone.utc)
# Спутник: 6U CubeSat на МКС-подобной орбите. Элементы заданы в момент
# главного сближения MAIN_TCA_S; u = ω + ν = 55° — восходящий участок,
# широта ≈ 40° с.ш.
MAIN_TCA_S = 10_800.0
SAT_ORBIT = OrbitShape(perigee_alt_m=415e3, apogee_alt_m=425e3,
                       inclination_deg=51.64)
SAT_RAAN_DEG = 120.0
SAT_ARGP_DEG = 30.0
SAT_TRUE_ANOMALY_AT_TCA_DEG = 25.0
# Фрагменты облака с наклонением 82.6° (как у Kosmos 1408), снизившиеся
# до высот МКС; эксцентриситет и высоты — синтетические.
DEBRIS_ORBIT = OrbitShape(perigee_alt_m=405e3, apogee_alt_m=470e3,
                          inclination_deg=82.6)

MAIN_ENCOUNTER = Encounter(
    tca_s=MAIN_TCA_S, debris_orbit=DEBRIS_ORBIT, debris_descending=True,
    debris_outbound=True, miss_m=120.0, miss_angle_deg=70.0)
SAFE_ENCOUNTER = Encounter(
    tca_s=12_600.0, debris_orbit=DEBRIS_ORBIT, debris_descending=False,
    debris_outbound=False, miss_m=4_000.0, miss_angle_deg=60.0)

SAFE_DISTANCE_M = 1_000.0
HORIZON_S = 14_400.0
# Манёвр не позже чем за 30 мин до TCA: подготовка и закладка команд.
NOMINAL_LIMITS = models.ManeuverLimits(dv_max_mps=0.5, t_earliest_s=0.0,
                                       t_latest_s=MAIN_TCA_S - 1_800.0)
# Позднее предупреждение: сближение обнаружено за 15 мин до TCA, манёвр
# возможен только в первые 5 мин; dv_max и спутник те же.
LATE_LEAD_S = 900.0
LATE_LIMITS = models.ManeuverLimits(dv_max_mps=0.5, t_earliest_s=0.0,
                                    t_latest_s=300.0)
LATE_HORIZON_S = LATE_LEAD_S + 1_800.0
# Пример допустимого манёвра: +0.1 м/с вдоль орбиты (T в RTN) в t = 0,
# почти два витка до TCA.
WITNESS_T_S = 0.0
WITNESS_DV_RTN_MPS = np.array([0.0, 0.1, 0.0])
# Справочные моменты для таблицы наименьшего Δv вдоль орбиты.
REFERENCE_BURN_TIMES_S = (0.0, 5_400.0, NOMINAL_LIMITS.t_latest_s)
DV_SCAN_STEP_MPS = 0.002
NEAR_TCA_WINDOW_S = 600.0

SATELLITE_NAME = "6U CubeSat (синтетический)"
SATELLITE_PHYSICS = {"mass_kg": 10.0, "area_m2": 0.071, "cd": 2.2}
DEBRIS_PHYSICS = {"mass_kg": 0.4, "area_m2": 0.03, "cd": 2.2}
FORCES = models.ForceModel()
NUMERICS = models.Numerics()

SOURCE_COMMON = (
    "Синтетический сценарий OG-06: построен tools/build_scenarios.py "
    "обратным расчётом от заданной геометрии сближения в модели "
    "центрального поля (задача двух тел); описание и источники "
    "параметров — docs/scenarios.md. ")


def satellite_state_at_epoch() -> np.ndarray:
    """Состояние спутника на EPOCH, перенесённое из MAIN_TCA_S.

    Returns:
        Состояние (6,), м и м/с, ECI.
    """
    at_tca = ref.elements_to_state(
        SAT_ORBIT.sma_m, SAT_ORBIT.ecc,
        math.radians(SAT_ORBIT.inclination_deg), math.radians(SAT_RAAN_DEG),
        math.radians(SAT_ARGP_DEG),
        math.radians(SAT_TRUE_ANOMALY_AT_TCA_DEG))
    return ref.kepler(at_tca, -MAIN_TCA_S)[0]


def debris_state_at_epoch(sat_epoch: np.ndarray,
                          enc: Encounter) -> np.ndarray:
    """Состояние фрагмента на EPOCH для сближения enc.

    Args:
        sat_epoch: Состояние спутника на EPOCH (6,).
        enc: Сближение.

    Returns:
        Состояние фрагмента (6,), м и м/с, ECI.
    """
    sat_tca = ref.kepler(sat_epoch, enc.tca_s)[0]
    return ref.kepler(debris_state_at_tca(sat_tca, enc), -enc.tca_s)[0]


@dataclasses.dataclass(frozen=True)
class Built:
    """Сценарий, ожидание и данные проверок для отчёта."""

    scenario: models.Scenario
    expected: scenarios.Expectation
    minima: tuple[models.Approach, ...]
    bound: ref.ShiftBound | None = None


def _scenario(name: str, epoch: datetime.datetime, horizon_s: float,
              states: tuple[np.ndarray, np.ndarray], debris_name: str,
              limits: models.ManeuverLimits,
              source: str) -> models.Scenario:
    """Собирает сценарий с общими параметрами генератора.

    Args:
        name: Имя сценария.
        epoch: Дата и время начала, UTC.
        horizon_s: Горизонт прогноза, с.
        states: Начальные состояния спутника и мусора.
        debris_name: Имя мусора.
        limits: Ограничения манёвра.
        source: Описание происхождения данных.

    Returns:
        Проверенный сценарий.
    """
    return models.Scenario(
        name=name, epoch=epoch, horizon_s=horizon_s,
        satellite=models.SpaceObject(SATELLITE_NAME, states[0],
                                     **SATELLITE_PHYSICS),
        debris=models.SpaceObject(debris_name, states[1], **DEBRIS_PHYSICS),
        safe_distance_m=SAFE_DISTANCE_M, limits=limits, forces=FORCES,
        numerics=NUMERICS, synthetic=True, source=source)


def _designed_closest(sat0: np.ndarray, deb0: np.ndarray, horizon_s: float,
                      tca_s: float, miss_m: float) -> tuple:
    """Минимумы на горизонте; глобальный обязан совпасть с заданным TCA.

    Args:
        sat0: Начальное состояние спутника (6,).
        deb0: Начальное состояние мусора (6,).
        horizon_s: Горизонт, с.
        tca_s: Заданный момент сближения, с.
        miss_m: Заданная дистанция сближения, м.

    Returns:
        Глобальный минимум и все локальные минимумы.

    Raises:
        ArithmeticError: Глобальный минимум не совпал с заданным.
    """
    minima = ref.approaches(ref.Track(sat0), ref.Track(deb0), horizon_s)
    found = ref.closest(minima)
    if (abs(found.t_s - tca_s) > 1e-6
            or abs(found.distance_m - miss_m) > 1e-6):
        raise ArithmeticError(
            f"Глобальный минимум {found} не совпал с заданным TCA "
            f"{tca_s} с / {miss_m} м")
    return found, tuple(minima)


def _required_m() -> float:
    """Безопасная дистанция с запасом.

    Returns:
        Порог, м.
    """
    return SAFE_DISTANCE_M + NUMERICS.distance_margin_m


def build_safe(sat0: np.ndarray) -> Built:
    """Сближение на 4 км — манёвр не нужен (already_safe).

    Args:
        sat0: Состояние спутника на EPOCH (6,).

    Returns:
        Сценарий safe_flyby с обоснованием.
    """
    enc = SAFE_ENCOUNTER
    deb0 = debris_state_at_epoch(sat0, enc)
    found, minima = _designed_closest(sat0, deb0, HORIZON_S, enc.tca_s,
                                      enc.miss_m)
    scenario = _scenario(
        "safe_flyby", EPOCH, HORIZON_S, (sat0, deb0),
        "Фрагмент i=82.6° №2 (синтетический)", NOMINAL_LIMITS,
        SOURCE_COMMON + "6U CubeSat на МКС-подобной орбите и фрагмент на "
        "наклонении 82.6°; TCA и промах 4 км заданы конструкцией.")
    expected = scenarios.Expectation(
        status=models.Status.ALREADY_SAFE, closest=found,
        time_tol_s=TIME_TOL_S, distance_tol_m=DISTANCE_TOL_M,
        basis=(f"Без манёвра минимальная дистанция на всём горизонте "
               f"{found.distance_m:.1f} м в t = {found.t_s:.1f} с — не "
               f"меньше порога {SAFE_DISTANCE_M:.0f} м с запасом "
               f"{NUMERICS.distance_margin_m:.0f} м. Проверено "
               f"аналитическим переносом обеих орбит: все минимумы "
               f"r_rel·v_rel = 0 на горизонте и его границы."))
    return Built(scenario, expected, minima)


def build_dangerous(sat0: np.ndarray) -> Built:
    """Сближение на 120 м через 3 ч; манёвр 0.1 м/с решает (feasible).

    Args:
        sat0: Состояние спутника на EPOCH (6,).

    Returns:
        Сценарий dangerous_feasible с примером допустимого импульса.
    """
    enc = MAIN_ENCOUNTER
    deb0 = debris_state_at_epoch(sat0, enc)
    found, minima = _designed_closest(sat0, deb0, HORIZON_S, enc.tca_s,
                                      enc.miss_m)
    at_burn = ref.kepler(sat0, WITNESS_T_S)[0]
    dv_eci = ref.rtn_basis(at_burn) @ WITNESS_DV_RTN_MPS
    witness_min = ref.closest(ref.approaches(
        ref.Track(sat0, WITNESS_T_S, dv_eci), ref.Track(deb0), HORIZON_S))
    scenario = _scenario(
        "dangerous_feasible", EPOCH, HORIZON_S, (sat0, deb0),
        "Фрагмент i=82.6° №1 (синтетический)", NOMINAL_LIMITS,
        SOURCE_COMMON + "6U CubeSat на МКС-подобной орбите и фрагмент на "
        "наклонении 82.6° (как облако Kosmos 1408); TCA через 3 ч, "
        "промах 120 м заданы конструкцией.")
    expected = scenarios.Expectation(
        status=models.Status.FEASIBLE, closest=found,
        time_tol_s=TIME_TOL_S, distance_tol_m=DISTANCE_TOL_M,
        basis=(f"Без манёвра сближение {found.distance_m:.1f} м в "
               f"t = {found.t_s:.1f} с — опаснее порога "
               f"{SAFE_DISTANCE_M:.0f} м. Допустимый пример: "
               f"{WITNESS_DV_RTN_MPS[1]:.2f} м/с вдоль орбиты (T в RTN) в "
               f"t = {WITNESS_T_S:.0f} с даёт минимум "
               f"{witness_min.distance_m:.1f} м на всём горизонте "
               f"(аналитический перенос до и после импульса). Это пример, "
               f"а не минимальный Δv."),
        witness=scenarios.Witness(models.Impulse(WITNESS_T_S, dv_eci),
                                  witness_min.distance_m))
    return Built(scenario, expected, minima)


def build_late(sat0: np.ndarray) -> Built:
    """То же сближение, предупреждение за 15 мин — решения нет.

    Args:
        sat0: Состояние спутника на EPOCH (6,).

    Returns:
        Сценарий late_warning_no_solution с границей смещения.
    """
    enc = MAIN_ENCOUNTER
    start_s = enc.tca_s - LATE_LEAD_S
    sat_late = ref.kepler(sat0, start_s)[0]
    deb_late = ref.kepler(debris_state_at_epoch(sat0, enc), start_s)[0]
    found, minima = _designed_closest(sat_late, deb_late, LATE_HORIZON_S,
                                      LATE_LEAD_S, enc.miss_m)
    bound = ref.shift_bound(sat_late, LATE_LEAD_S, LATE_LIMITS)
    bound_m = math.ceil(bound.bound_m)
    scenario = _scenario(
        "late_warning_no_solution",
        EPOCH + datetime.timedelta(seconds=start_s), LATE_HORIZON_S,
        (sat_late, deb_late), "Фрагмент i=82.6° №1 (синтетический)",
        LATE_LIMITS,
        SOURCE_COMMON + "То же сближение, что dangerous_feasible, но "
        "предупреждение получено за 15 мин до TCA; манёвр возможен только "
        "в первые 5 мин.")
    expected = scenarios.Expectation(
        status=models.Status.NO_FEASIBLE_CANDIDATE, closest=found,
        time_tol_s=TIME_TOL_S, distance_tol_m=DISTANCE_TOL_M,
        basis=(f"Без манёвра сближение {found.distance_m:.1f} м. Любой "
               f"импульс |Δv| ≤ {LATE_LIMITS.dv_max_mps} м/с в окне "
               f"[{LATE_LIMITS.t_earliest_s:.0f}, "
               f"{LATE_LIMITS.t_latest_s:.0f}] с смещает спутник в момент "
               f"исходного TCA не более чем на {bound_m} м (матрица "
               f"перехода состояний Φ_rv, перебор момента с шагом "
               f"{ref.BOUND_STEP_S:.0f} с, проверка нелинейности). Значит, "
               f"дистанция в этот момент ≤ {found.distance_m:.0f} + "
               f"{bound_m} < {_required_m():.0f} м: безопасного манёвра в "
               f"заданных пределах нет — для любой сетки кандидатов."),
        shift_bound_m=float(bound_m))
    return Built(scenario, expected, minima, bound=bound)


def min_along_track_dv(sat0: np.ndarray, deb0: np.ndarray, enc: Encounter,
                       burn_t_s: float, sign: float) -> float | None:
    """Наименьший |Δv| вдоль ±T в момент burn_t_s, дающий безопасность.

    Справочная величина для docs/scenarios.md и OG-04: одно направление и
    один момент, а не глобальный оптимум. Перебор по сетке
    DV_SCAN_STEP_MPS около TCA, уточнение методом Брента, округление
    вверх до 0.1 мм/с и проверка на всём горизонте.

    Args:
        sat0: Состояние спутника на эпоху (6,).
        deb0: Состояние фрагмента на эпоху (6,).
        enc: Сближение (используется tca_s).
        burn_t_s: Момент импульса, с.
        sign: +1 — по скорости, −1 — против.

    Returns:
        |Δv|, м/с, или None, если до dv_max решения нет.

    Raises:
        ArithmeticError: Найденный импульс не прошёл проверку горизонта.
    """
    direction = sign * ref.rtn_basis(ref.kepler(sat0, burn_t_s)[0])[:, 1]

    def _margin(dv_mps: float) -> float:
        """Запас дистанции у исходного сближения после импульса.

        Args:
            dv_mps: Величина импульса, м/с.

        Returns:
            Минимальная дистанция минус порог с запасом, м.
        """
        near = ref.segment_minima(
            ref.Track(sat0, burn_t_s, dv_mps * direction), ref.Track(deb0),
            enc.tca_s - NEAR_TCA_WINDOW_S, enc.tca_s + NEAR_TCA_WINDOW_S)
        return ref.closest(near).distance_m - _required_m()

    grid = np.arange(0.0, NOMINAL_LIMITS.dv_max_mps + 1e-12,
                     DV_SCAN_STEP_MPS)
    for low, high in zip(grid[:-1], grid[1:]):
        if _margin(high) >= 0.0:
            found = math.ceil(
                optimize.brentq(_margin, low, high, xtol=1e-6) * 1e4) / 1e4
            full = ref.approaches(
                ref.Track(sat0, burn_t_s, found * direction),
                ref.Track(deb0), HORIZON_S)
            if ref.closest(full).distance_m < _required_m():
                raise ArithmeticError("Проверка на всём горизонте не прошла")
            return found
    return None


def build_all() -> list[Built]:
    """Строит все сценарии: безопасный, опасный, без решения.

    Returns:
        Сценарии с ожиданиями и данными проверок.
    """
    sat0 = satellite_state_at_epoch()
    return [build_safe(sat0), build_dangerous(sat0), build_late(sat0)]


# ---------------------------------------------------------------------------
# Командная строка
# ---------------------------------------------------------------------------

def _path(built: Built) -> pathlib.Path:
    """Путь к файлу сценария.

    Args:
        built: Построенный сценарий.

    Returns:
        Путь в каталоге scenarios/.
    """
    return OUT_DIR / f"{built.scenario.name}.json"


def _close(stored: Any, fresh: Any) -> bool:
    """Рекурсивное сравнение JSON-данных: числа — с допуском.

    Args:
        stored: Данные из файла.
        fresh: Только что построенные данные.

    Returns:
        True, если данные совпадают.
    """
    if isinstance(stored, dict) and isinstance(fresh, dict):
        return (stored.keys() == fresh.keys()
                and all(_close(stored[k], fresh[k]) for k in stored))
    if isinstance(stored, list) and isinstance(fresh, list):
        return (len(stored) == len(fresh)
                and all(_close(a, b) for a, b in zip(stored, fresh)))
    if (isinstance(stored, float) and isinstance(fresh, float)):
        return math.isclose(stored, fresh, rel_tol=CHECK_REL_TOL,
                            abs_tol=CHECK_ABS_TOL)
    return stored == fresh


def check(items: list[Built]) -> list[str]:
    """Сравнивает сохранённые файлы с генератором.

    Args:
        items: Результат build_all().

    Returns:
        Описания расхождений; пустой список — файлы актуальны.
    """
    problems = []
    for built in items:
        path = _path(built)
        if not path.exists():
            problems.append(f"{path.name}: файла нет")
            continue
        stored = scenarios.scenario_to_dict(
            *scenarios.load_scenario_file(path))
        fresh = scenarios.scenario_to_dict(built.scenario, built.expected)
        if not _close(stored, fresh):
            problems.append(f"{path.name}: отличается от генератора")
    return problems


def _describe(built: Built) -> list[str]:
    """Строки отчёта о сценарии.

    Args:
        built: Построенный сценарий.

    Returns:
        Строки для печати.
    """
    sc, exp = built.scenario, built.expected
    lines = [
        f"== {sc.name}: {exp.status.value}",
        f"   эпоха {scenarios.format_epoch(sc.epoch)}, горизонт "
        f"{sc.horizon_s:.0f} с, окно [{sc.limits.t_earliest_s:.0f}, "
        f"{sc.limits.t_latest_s:.0f}] с, dv_max {sc.limits.dv_max_mps} м/с",
    ]
    for obj in (sc.satellite, sc.debris):
        info = ref.describe_orbit(obj.state0)
        lines.append(
            f"   {obj.name}: hp={info['perigee_alt_m'] / 1e3:.2f} км, "
            f"ha={info['apogee_alt_m'] / 1e3:.2f} км, "
            f"i={info['inclination_deg']:.3f}°, T={info['period_s']:.1f} с, "
            f"e={info['ecc']:.5f}")
    sat_tca = ref.kepler(sc.satellite.state0, exp.closest.t_s)[0]
    info = ref.describe_orbit(sat_tca)
    lines.append(
        f"   TCA t={exp.closest.t_s:.3f} с, d={exp.closest.distance_m:.3f} м,"
        f" v_rel={exp.closest.rel_speed_mps:.1f} м/с, широта "
        f"{info['latitude_deg']:.2f}°, высота {info['altitude_m'] / 1e3:.2f} "
        f"км")
    others = sorted(item.distance_m for item in built.minima
                    if item.t_s != exp.closest.t_s)
    lines.append(f"   прочие минимумы и границы: ближайший {others[0]:.0f} м "
                 f"(всего {len(others)})")
    if exp.witness is not None:
        lines.append(f"   пример: t={exp.witness.impulse.t_s} с, |Δv|="
                     f"{exp.witness.impulse.dv_mps:.3f} м/с, d_min="
                     f"{exp.witness.min_distance_m:.1f} м")
    if built.bound is not None:
        bound = built.bound
        lines.append(
            f"   граница смещения {bound.bound_m:.2f} м (σ_max="
            f"{bound.max_sigma_s:.1f} с, Δσ={bound.lipschitz_s:.3f} с, "
            f"нелинейность {bound.nonlinear_m:.2e} м, точный максимум "
            f"{bound.max_actual_shift_m:.2f} м)")
    return lines


def report(items: list[Built]) -> str:
    """Текстовый отчёт: орбиты, сближения, обоснование статусов.

    Args:
        items: Результат build_all().

    Returns:
        Многострочный текст для docs/scenarios.md.
    """
    lines = []
    for built in items:
        lines.extend(_describe(built))
        if built.scenario.name != "dangerous_feasible":
            continue
        lines.append("   наименьший |Δv| вдоль ±T (справочно):")
        for burn_t_s in REFERENCE_BURN_TIMES_S:
            values = [min_along_track_dv(
                built.scenario.satellite.state0,
                built.scenario.debris.state0, MAIN_ENCOUNTER, burn_t_s, sign)
                for sign in (1.0, -1.0)]
            text = ", ".join("нет" if v is None else f"{v:.4f}"
                             for v in values)
            lines.append(f"     t_b={burn_t_s:.0f} с (за "
                         f"{(MAIN_TCA_S - burn_t_s) / 60:.0f} мин): "
                         f"+T/−T = {text} м/с")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Точка входа командной строки.

    Args:
        argv: Аргументы; по умолчанию sys.argv.

    Returns:
        Код завершения: 0 — успех, 1 — файлы расходятся с генератором.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true",
                      help="сравнить scenarios/*.json с генератором")
    mode.add_argument("--report", action="store_true",
                      help="вывести таблицу проверок")
    args = parser.parse_args(argv)
    items = build_all()
    if args.report:
        sys.stdout.write(report(items) + "\n")
        return 0
    if args.check:
        problems = check(items)
        for problem in problems:
            sys.stderr.write(problem + "\n")
        return 1 if problems else 0
    OUT_DIR.mkdir(exist_ok=True)
    for built in items:
        scenarios.save_scenario(built.scenario, _path(built), built.expected)
        sys.stdout.write(f"записан {_path(built).relative_to(ROOT)}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
