"""Аналитические эталонные орбиты для тестов.

Двухтельная задача решается точно через уравнение Кеплера. Эти орбиты —
независимый эталон для прогноза движения (OG-02), поиска сближения (OG-03)
и проверки самих физических проверок (tests/invariants.py).
Единицы: СИ, углы — радианы, система координат ECI.
"""

import numpy as np

# Константы эталона. Совпадают с docs/DEV_GUIDE.md, раздел 7.
MU_EARTH_M3PS2 = 3.986004418e14  # WGS 84
R_EARTH_M = 6378137.0  # WGS 84
R_EARTH_GRAV_M = 6378136.3  # EGM2008, радиус в паре с J2
J2 = 1.08262668e-3  # EGM2008

_KEPLER_TOL_RAD = 1e-13
_KEPLER_MAX_ITER = 50


def orbital_period_s(sma_m: float, mu_m3ps2: float = MU_EARTH_M3PS2) -> float:
    """Период кеплеровой орбиты.

    Args:
        sma_m: Большая полуось, м.
        mu_m3ps2: Гравитационный параметр, м³/с².

    Returns:
        Период обращения, с.
    """
    return 2.0 * np.pi * np.sqrt(sma_m**3 / mu_m3ps2)


def solve_kepler(mean_anomaly_rad: np.ndarray, ecc: float) -> np.ndarray:
    """Решает уравнение Кеплера M = E − e·sin E методом Ньютона.

    Args:
        mean_anomaly_rad: Средняя аномалия, рад.
        ecc: Эксцентриситет, 0 ≤ e < 1.

    Returns:
        Эксцентрическая аномалия, рад.

    Raises:
        ValueError: Если эксцентриситет вне [0, 1) или метод не сошёлся.
    """
    if not 0.0 <= ecc < 1.0:
        raise ValueError(f"Эксцентриситет должен быть в [0, 1): {ecc}")
    # Приводим к [−π, π): иначе на больших углах точность float64
    # не позволяет достичь допуска.
    mean_anomaly_rad = np.mod(
        np.asarray(mean_anomaly_rad, dtype=float) + np.pi, 2.0 * np.pi) - np.pi
    ecc_anomaly_rad = mean_anomaly_rad.copy()
    for _ in range(_KEPLER_MAX_ITER):
        step = (ecc_anomaly_rad - ecc * np.sin(ecc_anomaly_rad)
                - mean_anomaly_rad) / (1.0 - ecc * np.cos(ecc_anomaly_rad))
        ecc_anomaly_rad = ecc_anomaly_rad - step
        if np.all(np.abs(step) < _KEPLER_TOL_RAD):
            return ecc_anomaly_rad
    raise ValueError("Уравнение Кеплера не сошлось")


def _perifocal_to_eci(inclination_rad: float, raan_rad: float,
                      argp_rad: float) -> np.ndarray:
    """Матрица поворота из перифокальной системы в ECI.

    Args:
        inclination_rad: Наклонение, рад.
        raan_rad: Долгота восходящего узла, рад.
        argp_rad: Аргумент перицентра, рад.

    Returns:
        Матрица (3, 3).
    """
    cos_o, sin_o = np.cos(raan_rad), np.sin(raan_rad)
    cos_i, sin_i = np.cos(inclination_rad), np.sin(inclination_rad)
    cos_w, sin_w = np.cos(argp_rad), np.sin(argp_rad)
    return np.array([
        [cos_o * cos_w - sin_o * sin_w * cos_i,
         -cos_o * sin_w - sin_o * cos_w * cos_i,
         sin_o * sin_i],
        [sin_o * cos_w + cos_o * sin_w * cos_i,
         -sin_o * sin_w + cos_o * cos_w * cos_i,
         -cos_o * sin_i],
        [sin_w * sin_i, cos_w * sin_i, cos_i],
    ])


def kepler_states_eci(
    sma_m: float,
    ecc: float,
    inclination_rad: float,
    raan_rad: float,
    argp_rad: float,
    mean_anomaly0_rad: float,
    t_s: np.ndarray,
    mu_m3ps2: float = MU_EARTH_M3PS2,
) -> np.ndarray:
    """Точные состояния на кеплеровой орбите в заданные моменты.

    Args:
        sma_m: Большая полуось, м.
        ecc: Эксцентриситет, 0 ≤ e < 1.
        inclination_rad: Наклонение, рад.
        raan_rad: Долгота восходящего узла, рад.
        argp_rad: Аргумент перицентра, рад.
        mean_anomaly0_rad: Средняя аномалия в момент t = 0, рад.
        t_s: Моменты времени (N,), с от даты начала.
        mu_m3ps2: Гравитационный параметр, м³/с².

    Returns:
        Состояния (N, 6): [x, y, z, vx, vy, vz] в м и м/с, ECI.
    """
    t_s = np.atleast_1d(np.asarray(t_s, dtype=float))
    mean_motion_radps = np.sqrt(mu_m3ps2 / sma_m**3)
    ecc_anomaly_rad = solve_kepler(
        mean_anomaly0_rad + mean_motion_radps * t_s, ecc)
    states_pf = _perifocal_states(sma_m, ecc, ecc_anomaly_rad, mu_m3ps2)
    rotation = _perifocal_to_eci(inclination_rad, raan_rad, argp_rad)
    return np.hstack([states_pf[:, :3] @ rotation.T,
                      states_pf[:, 3:] @ rotation.T])


def _perifocal_states(sma_m: float, ecc: float, ecc_anomaly_rad: np.ndarray,
                      mu_m3ps2: float) -> np.ndarray:
    """Состояния в перифокальной системе по эксцентрической аномалии.

    Args:
        sma_m: Большая полуось, м.
        ecc: Эксцентриситет.
        ecc_anomaly_rad: Эксцентрическая аномалия (N,), рад.
        mu_m3ps2: Гравитационный параметр, м³/с².

    Returns:
        Состояния (N, 6), м и м/с; ось x — на перицентр.
    """
    true_anomaly_rad = 2.0 * np.arctan2(
        np.sqrt(1.0 + ecc) * np.sin(ecc_anomaly_rad / 2.0),
        np.sqrt(1.0 - ecc) * np.cos(ecc_anomaly_rad / 2.0))
    radius_m = sma_m * (1.0 - ecc * np.cos(ecc_anomaly_rad))
    speed_scale_mps = np.sqrt(mu_m3ps2 / (sma_m * (1.0 - ecc**2)))
    zeros = np.zeros_like(true_anomaly_rad)
    return np.stack([
        radius_m * np.cos(true_anomaly_rad),
        radius_m * np.sin(true_anomaly_rad),
        zeros,
        -speed_scale_mps * np.sin(true_anomaly_rad),
        speed_scale_mps * (ecc + np.cos(true_anomaly_rad)),
        zeros,
    ], axis=1)
