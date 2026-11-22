"""Аналитический эталон задачи двух тел для сценариев OG-06 (#6).

Независим от численного интегратора OG-02: перенос по кеплеровой орбите
в универсальной переменной, перевод элементов орбиты, поиск всех
минимумов дистанции и оценка достижимого смещения от импульса через
матрицу перехода состояний. Используется генератором
tools/build_scenarios.py и тестами tests/test_scenarios.py.

Единицы — СИ, система координат — ECI-J2000, μ и R — из orbitguard.constants,
то есть те же, что в расчёте.
"""

import dataclasses
import math
import sys

import numpy as np
from scipy import optimize

from orbitguard import constants, models

MU_M3PS2 = constants.MU_EARTH_M3_S2
R_EARTH_M = constants.EARTH_RADIUS_M

_KEPLER_MAX_ITER = 60
_KEPLER_REL_TOL = 1e-13
# Порог перехода от рядов Тейлора к замкнутым формулам Штумпфа: ниже него
# 1 − cos√z и √z − sin√z теряют точность из-за вычитания близких чисел.
_STUMPFF_SERIES_Z = 0.1
_STUMPFF_SERIES_TERMS = 10
# Шаг поиска смены знака r_rel·v_rel. При относительной скорости
# ~10 км/с между двумя минимумами в этих сценариях — минуты, не секунды.
SCAN_STEP_S = 2.0
# Уточнение момента минимума методом Брента, с.
ROOT_XTOL_S = 1e-9
# Шаг перебора момента импульса при оценке достижимого смещения.
BOUND_STEP_S = 1.0
# Пробное приращение скорости для конечных разностей, м/с.
FD_STEP_MPS = 1e-3


# ---------------------------------------------------------------------------
# Кеплерово движение
# ---------------------------------------------------------------------------

def _stumpff_series(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ряды C = Σ (−z)^k/(2k+2)!, S = Σ (−z)^k/(2k+3)! для малых |z|.

    Args:
        z: Аргумент функций Штумпфа.

    Returns:
        Значения C(z) и S(z).
    """
    term_c = np.full_like(z, 0.5)
    term_s = np.full_like(z, 1.0 / 6.0)
    sum_c = term_c.copy()
    sum_s = term_s.copy()
    for k in range(1, _STUMPFF_SERIES_TERMS):
        term_c = term_c * (-z) / ((2 * k + 1) * (2 * k + 2))
        term_s = term_s * (-z) / ((2 * k + 2) * (2 * k + 3))
        sum_c += term_c
        sum_s += term_s
    return sum_c, sum_s


def _stumpff(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Функции Штумпфа C(z), S(z) без потери точности при малых |z|.

    Args:
        z: Аргумент функций Штумпфа.

    Returns:
        Значения C(z) и S(z).
    """
    c = np.empty_like(z)
    s = np.empty_like(z)
    small = np.abs(z) < _STUMPFF_SERIES_Z
    pos = z >= _STUMPFF_SERIES_Z
    neg = z <= -_STUMPFF_SERIES_Z
    root = np.sqrt(z[pos])
    c[pos] = 2.0 * np.sin(root / 2.0)**2 / z[pos]
    s[pos] = (root - np.sin(root)) / root**3
    root = np.sqrt(-z[neg])
    c[neg] = (np.cosh(root) - 1.0) / (-z[neg])
    s[neg] = (np.sinh(root) - root) / root**3
    c[small], s[small] = _stumpff_series(z[small])
    return c, s


def _universal_anomaly(r0: np.ndarray, vr0: np.ndarray, alpha: np.ndarray,
                       dt: np.ndarray) -> np.ndarray:
    """Решает уравнение Кеплера в универсальной переменной методом Ньютона.

    Args:
        r0: Начальный радиус (N,), м.
        vr0: Начальная радиальная скорость (N,), м/с.
        alpha: 1/a (N,), 1/м.
        dt: Интервал (N,), с.

    Returns:
        Универсальная аномалия χ (N,), √м.

    Raises:
        ArithmeticError: Метод не сошёлся.
    """
    sqrt_mu = math.sqrt(MU_M3PS2)
    chi = sqrt_mu * np.abs(alpha) * dt
    for _ in range(_KEPLER_MAX_ITER):
        z = alpha * chi**2
        c, s = _stumpff(z)
        value = (r0 * vr0 / sqrt_mu * chi**2 * c
                 + (1.0 - alpha * r0) * chi**3 * s
                 + r0 * chi - sqrt_mu * dt)
        slope = (r0 * vr0 / sqrt_mu * chi * (1.0 - z * s)
                 + (1.0 - alpha * r0) * chi**2 * c + r0)
        step = value / slope
        chi = chi - step
        if np.all(np.abs(step)
                  <= _KEPLER_REL_TOL * np.maximum(1.0, np.abs(chi))):
            return chi
    raise ArithmeticError("Уравнение Кеплера не сошлось")


def kepler(states: np.ndarray, dt_s: np.ndarray | float) -> np.ndarray:
    """Переносит состояния по кеплеровой орбите на время dt.

    Универсальная переменная и функции Штумпфа (Curtis, «Orbital
    Mechanics for Engineering Students», алгоритмы 3.3–3.4; Vallado,
    алгоритм 8). Работает для эллипса, параболы и гиперболы, dt < 0 —
    расчёт назад.

    Args:
        states: Состояния (6,) или (N, 6), м и м/с, ECI.
        dt_s: Интервалы, с: скаляр или (N,).

    Returns:
        Состояния (N, 6) в момент t + dt.
    """
    states = np.atleast_2d(np.asarray(states, dtype=np.float64))
    dt = np.atleast_1d(np.asarray(dt_s, dtype=np.float64))
    states, dt = np.broadcast_arrays(states, dt[:, None])
    return _lagrange(states[:, :3], states[:, 3:], dt[:, 0])


def _lagrange(r0_vec: np.ndarray, v0_vec: np.ndarray,
              dt: np.ndarray) -> np.ndarray:
    """Перенос через коэффициенты Лагранжа f, g и их производные.

    Args:
        r0_vec: Положения (N, 3), м.
        v0_vec: Скорости (N, 3), м/с.
        dt: Интервалы (N,), с.

    Returns:
        Состояния (N, 6).
    """
    r0 = np.linalg.norm(r0_vec, axis=1)
    alpha = 2.0 / r0 - np.einsum("ij,ij->i", v0_vec, v0_vec) / MU_M3PS2
    chi = _universal_anomaly(
        r0, np.einsum("ij,ij->i", r0_vec, v0_vec) / r0, alpha, dt)
    sqrt_mu = math.sqrt(MU_M3PS2)
    c, s = _stumpff(alpha * chi**2)
    f = 1.0 - chi**2 / r0 * c
    g = dt - chi**3 / sqrt_mu * s
    r_vec = f[:, None] * r0_vec + g[:, None] * v0_vec
    r = np.linalg.norm(r_vec, axis=1)
    f_dot = sqrt_mu / (r * r0) * (alpha * chi**3 * s - chi)
    g_dot = 1.0 - chi**2 / r * c
    return np.hstack([r_vec, f_dot[:, None] * r0_vec + g_dot[:, None] * v0_vec])


def elements_to_state(
    sma_m: float, ecc: float, inclination_rad: float, raan_rad: float,
    argp_rad: float, true_anomaly_rad: float,
) -> np.ndarray:
    """Переводит классические элементы орбиты в состояние ECI (6,).

    Args:
        sma_m: Большая полуось, м.
        ecc: Эксцентриситет, 0 ≤ e < 1.
        inclination_rad: Наклонение, рад.
        raan_rad: Долгота восходящего узла, рад.
        argp_rad: Аргумент перицентра, рад.
        true_anomaly_rad: Истинная аномалия, рад.

    Returns:
        [x, y, z, vx, vy, vz], м и м/с.
    """
    p_m = sma_m * (1.0 - ecc**2)
    cos_nu, sin_nu = math.cos(true_anomaly_rad), math.sin(true_anomaly_rad)
    r_pf = p_m / (1.0 + ecc * cos_nu) * np.array([cos_nu, sin_nu, 0.0])
    v_pf = math.sqrt(MU_M3PS2 / p_m) * np.array([-sin_nu, ecc + cos_nu, 0.0])
    # Перифокальная система → ECI: R3(Ω)·R1(i)·R3(ω).
    rotation = (_rotation_z(raan_rad) @ _rotation_x(inclination_rad)
                @ _rotation_z(argp_rad))
    return np.concatenate([rotation @ r_pf, rotation @ v_pf])


def _rotation_z(angle_rad: float) -> np.ndarray:
    """Матрица поворота вокруг оси z.

    Args:
        angle_rad: Угол, рад.

    Returns:
        Матрица (3, 3).
    """
    cos, sin = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])


def _rotation_x(angle_rad: float) -> np.ndarray:
    """Матрица поворота вокруг оси x.

    Args:
        angle_rad: Угол, рад.

    Returns:
        Матрица (3, 3).
    """
    cos, sin = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, cos, -sin], [0.0, sin, cos]])


def describe_orbit(state: np.ndarray) -> dict[str, float]:
    """Возвращает понятные характеристики орбиты для отчёта и тестов.

    Высота отсчитывается от сферы радиуса R_EARTH_M, как в контракте.

    Args:
        state: Состояние (6,), м и м/с, ECI.

    Returns:
        sma_m, ecc, inclination_deg, perigee_alt_m, apogee_alt_m,
        altitude_m, latitude_deg, period_s.
    """
    r_vec, v_vec = state[:3], state[3:]
    r = float(np.linalg.norm(r_vec))
    h_vec = np.cross(r_vec, v_vec)
    sma_m = -MU_M3PS2 / (2.0 * (float(v_vec @ v_vec) / 2.0 - MU_M3PS2 / r))
    ecc = float(np.linalg.norm(np.cross(v_vec, h_vec) / MU_M3PS2 - r_vec / r))
    return {
        "sma_m": sma_m,
        "ecc": ecc,
        "inclination_deg": math.degrees(
            math.acos(h_vec[2] / np.linalg.norm(h_vec))),
        "perigee_alt_m": sma_m * (1.0 - ecc) - R_EARTH_M,
        "apogee_alt_m": sma_m * (1.0 + ecc) - R_EARTH_M,
        "altitude_m": r - R_EARTH_M,
        "latitude_deg": math.degrees(math.asin(r_vec[2] / r)),
        "period_s": 2.0 * math.pi * math.sqrt(sma_m**3 / MU_M3PS2),
    }


def rtn_basis(state: np.ndarray) -> np.ndarray:
    """Матрица [R T N] по состоянию до импульса (contracts.md).

    R = r/|r|, N = (r×v)/|r×v|, T = N×R; Δv_ECI = [R T N] Δv_RTN.

    Args:
        state: Состояние (6,), м и м/с, ECI.

    Returns:
        Матрица (3, 3), столбцы — орты R, T, N в ECI.
    """
    radial = state[:3] / np.linalg.norm(state[:3])
    normal = np.cross(state[:3], state[3:])
    normal = normal / np.linalg.norm(normal)
    return np.column_stack([radial, np.cross(normal, radial), normal])


# ---------------------------------------------------------------------------
# Минимумы дистанции (аналитический эталон)
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Track:
    """Кусочно-кеплерова траектория: до и после одного импульса.

    Attributes:
        state0: Состояние на эпоху (6,).
        burn_t_s: Момент импульса, с, или None.
        dv_eci_mps: Импульс (3,), м/с, ECI, или None.
    """

    state0: np.ndarray
    burn_t_s: float | None = None
    dv_eci_mps: np.ndarray | None = None

    def states(self, t_s: np.ndarray | float,
               side: str = "right") -> np.ndarray:
        """Состояния в моменты t_s; в момент импульса side как в OG-01.

        Args:
            t_s: Моменты, с от эпохи: скаляр или (N,).
            side: left — до импульса, right — после.

        Returns:
            Состояния (N, 6).
        """
        t_s = np.atleast_1d(np.asarray(t_s, dtype=np.float64))
        result = kepler(self.state0, t_s)
        if self.burn_t_s is None:
            return result
        after_burn = kepler(self.state0, self.burn_t_s)[0]
        after_burn[3:] += self.dv_eci_mps
        mask = (t_s > self.burn_t_s) | (
            (t_s == self.burn_t_s) & (side == "right"))
        if np.any(mask):
            result[mask] = kepler(after_burn, t_s[mask] - self.burn_t_s)
        return result


def _relative(sat: Track, debris: Track, t_s: np.ndarray | float,
              side: str = "right") -> np.ndarray:
    """Относительное состояние спутника и мусора.

    Args:
        sat: Траектория спутника.
        debris: Траектория мусора.
        t_s: Моменты, с от epoch.
        side: Сторона импульса.

    Returns:
        Состояния (N, 6), м и м/с.
    """
    return sat.states(t_s, side) - debris.states(t_s, side)


def segment_minima(sat: Track, debris: Track, start_s: float,
                    end_s: float) -> list[models.Approach]:
    """Минимумы на отрезке без импульса внутри (импульс — только на краях).

    Args:
        sat: Траектория спутника.
        debris: Траектория фрагмента.
        start_s: Начало отрезка, с; берётся правый предел.
        end_s: Конец отрезка, с; берётся левый предел.

    Returns:
        Границы и уточнённые минимумы в порядке времени.
    """
    count = max(2, int(math.ceil((end_s - start_s) / SCAN_STEP_S)) + 1)
    grid = np.linspace(start_s, end_s, count)
    rel = _relative(sat, debris, grid)
    rel[-1] = _relative(sat, debris, end_s, "left")[0]
    rates = np.einsum("ij,ij->i", rel[:, :3], rel[:, 3:])
    times = {start_s, end_s}
    for k in np.nonzero((rates[:-1] < 0) & (rates[1:] >= 0))[0]:
        if rates[k + 1] == 0:
            times.add(float(grid[k + 1]))
            continue
        side = "left" if k + 1 == count - 1 else "right"

        def _rate(t_s: float, side: str = side) -> float:
            """Скорость сближения r_rel · v_rel.

            Args:
                t_s: Момент, с от epoch.
                side: Сторона импульса.

            Returns:
                r_rel · v_rel, м²/с.
            """
            state = _relative(sat, debris, t_s, side)[0]
            return float(state[:3] @ state[3:])

        times.add(optimize.brentq(_rate, grid[k], grid[k + 1],
                                  xtol=ROOT_XTOL_S,
                                  rtol=4.0 * sys.float_info.epsilon))
    result = []
    for t_s in sorted(times):
        state = _relative(sat, debris, t_s,
                          "left" if t_s == end_s else "right")[0]
        result.append(models.Approach(
            float(t_s), float(np.linalg.norm(state[:3])),
            float(np.linalg.norm(state[3:]))))
    return result


def approaches(sat: Track, debris: Track,
               horizon_s: float) -> list[models.Approach]:
    """Все локальные минимумы дистанции и границы на [0, horizon_s].

    Корни r_rel·v_rel = 0 со сменой знака с «−» на «+» уточняются
    методом Брента до ROOT_XTOL_S; границы и момент импульса — кандидаты.

    Args:
        sat: Траектория спутника.
        debris: Траектория фрагмента.
        horizon_s: Горизонт, с.

    Returns:
        Минимумы в порядке времени.
    """
    if sat.burn_t_s is None or sat.burn_t_s in (0.0, horizon_s):
        return segment_minima(sat, debris, 0.0, horizon_s)
    return (segment_minima(sat, debris, 0.0, sat.burn_t_s)
            + segment_minima(sat, debris, sat.burn_t_s, horizon_s))


def closest(items: list[models.Approach]) -> models.Approach:
    """Ближайшее сближение; при равенстве — самое раннее.

    Args:
        items: Кандидаты.

    Returns:
        Кандидат с наименьшей дистанцией.
    """
    return min(items, key=lambda item: (item.distance_m, item.t_s))


# ---------------------------------------------------------------------------
# Граница достижимого смещения (сценарий без решения)
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class ShiftBound:
    """Оценка смещения спутника в момент TCA от любого допустимого импульса.

    Attributes:
        bound_m: Верхняя граница с запасом, м.
        max_sigma_s: max σ_max(Φ_rv) по окну, м/(м/с).
        lipschitz_s: Наибольшее изменение σ_max между соседними узлами.
        nonlinear_m: Наибольшее отличие точного смещения от линейного, м.
        max_actual_shift_m: Наибольшее точное смещение среди проверенных.
    """

    bound_m: float
    max_sigma_s: float
    lipschitz_s: float
    nonlinear_m: float
    max_actual_shift_m: float


def stm_rv(at_burn: np.ndarray, dt_s: np.ndarray) -> np.ndarray:
    """Блок Φ_rv = ∂r(t+dt)/∂v(t) матрицы перехода, центральные разности.

    Args:
        at_burn: Состояния в моменты импульса (N, 6).
        dt_s: Время от импульса до TCA (N,), с.

    Returns:
        Матрицы (N, 3, 3), м/(м/с).
    """
    stm = np.empty((len(at_burn), 3, 3))
    for axis in range(3):
        delta = np.zeros(6)
        delta[3 + axis] = FD_STEP_MPS
        plus = kepler(at_burn + delta, dt_s)[:, :3]
        minus = kepler(at_burn - delta, dt_s)[:, :3]
        stm[:, :, axis] = (plus - minus) / (2.0 * FD_STEP_MPS)
    return stm


def _nonlinear_check(at_burn: np.ndarray, dt_s: np.ndarray,
                     stm: np.ndarray,
                     dv_list: list[np.ndarray]) -> tuple[float, float]:
    """Точное смещение для заданных импульсов против линейного.

    Args:
        at_burn: Состояния в моменты импульса (N, 6).
        dt_s: Время до TCA (N,), с.
        stm: Φ_rv (N, 3, 3).
        dv_list: Наборы импульсов (N, 3), м/с.

    Returns:
        (наибольшая нелинейная поправка, наибольшее смещение), м.
    """
    nominal = kepler(at_burn, dt_s)[:, :3]
    nonlinear = 0.0
    largest = 0.0
    for dv in dv_list:
        moved = kepler(at_burn + np.hstack([np.zeros_like(dv), dv]), dt_s)
        shift = moved[:, :3] - nominal
        linear = np.einsum("nij,nj->ni", stm, dv)
        nonlinear = max(nonlinear, float(np.max(
            np.linalg.norm(shift - linear, axis=1))))
        largest = max(largest, float(np.max(np.linalg.norm(shift, axis=1))))
    return nonlinear, largest


def shift_bound(sat_state0: np.ndarray, tca_s: float,
                limits: models.ManeuverLimits) -> ShiftBound:
    """Верхняя граница |δr(tca)| для |Δv| ≤ dv_max в окне limits.

    δr(tca) = Φ_rv(tca, t_b)·Δv + O(|Δv|²). Максимум по направлениям —
    наибольшее сингулярное число Φ_rv; максимум по t_b — перебор с шагом
    BOUND_STEP_S плюс наибольший скачок между узлами. Нелинейная часть
    измеряется точным переносом для худших и осевых направлений и
    берётся с двукратным запасом.

    Args:
        sat_state0: Состояние спутника на эпоху сценария (6,).
        tca_s: Момент исходного сближения, с.
        limits: Окно манёвра и dv_max.

    Returns:
        Граница и её составляющие.

    Raises:
        ValueError: Окно манёвра не заканчивается до TCA.
    """
    if limits.t_latest_s >= tca_s:
        raise ValueError("Окно манёвра должно заканчиваться до TCA")
    count = max(2, int(math.ceil(
        (limits.t_latest_s - limits.t_earliest_s) / BOUND_STEP_S)) + 1)
    burn_times = np.linspace(limits.t_earliest_s, limits.t_latest_s, count)
    at_burn = kepler(sat_state0, burn_times)
    stm = stm_rv(at_burn, tca_s - burn_times)
    _, sigma, right = np.linalg.svd(stm)
    max_sigma = float(sigma[:, 0].max())
    lipschitz = float(np.max(np.abs(np.diff(sigma[:, 0]))))
    directions = [right[:, 0, :]] + [
        np.tile(np.eye(3)[axis], (count, 1)) for axis in range(3)]
    nonlinear, largest = _nonlinear_check(
        at_burn, tca_s - burn_times, stm,
        [sign * limits.dv_max_mps * direction
         for direction in directions for sign in (1.0, -1.0)])
    return ShiftBound(
        limits.dv_max_mps * (max_sigma + lipschitz) + 2.0 * nonlinear,
        max_sigma, lipschitz, nonlinear, largest)
