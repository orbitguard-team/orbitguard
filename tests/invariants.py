"""Физические инварианты и эталонные формулы для тестов.

Функции работают с массивами состояний (N, 6) в СИ и ECI и не зависят
от реализации прогноза — их можно применять к любому результату расчёта.
Формулы — docs/PHYSICS.md.
"""

import numpy as np


def _split(states_eci: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Делит состояния на положения и скорости.

    Args:
        states_eci: Состояния (6,) или (N, 6), м и м/с.

    Returns:
        Положения (N, 3), м, и скорости (N, 3), м/с.
    """
    states_eci = np.atleast_2d(states_eci)
    return states_eci[:, :3], states_eci[:, 3:]


def specific_energy_m2ps2(states_eci: np.ndarray,
                          mu_m3ps2: float) -> np.ndarray:
    """Удельная механическая энергия в центральном поле: v²/2 − μ/r.

    Args:
        states_eci: Состояния (N, 6), м и м/с.
        mu_m3ps2: Гравитационный параметр, м³/с².

    Returns:
        Энергия (N,), м²/с².
    """
    r_m, v_mps = _split(states_eci)
    return (0.5 * np.sum(v_mps**2, axis=1)
            - mu_m3ps2 / np.linalg.norm(r_m, axis=1))


def specific_angular_momentum_m2ps(states_eci: np.ndarray) -> np.ndarray:
    """Удельный момент импульса h = r × v.

    Args:
        states_eci: Состояния (N, 6), м и м/с.

    Returns:
        Векторы момента (N, 3), м²/с.
    """
    r_m, v_mps = _split(states_eci)
    return np.cross(r_m, v_mps)


def j2_potential_energy_m2ps2(r_eci_m: np.ndarray, mu_m3ps2: float,
                              j2: float, r_ref_m: float) -> np.ndarray:
    """Добавка J2 к удельной потенциальной энергии.

    V_J2 = μ·J2·R²·(3z²/r² − 1) / (2r³); ускорение J2 равно −∇V_J2.

    Args:
        r_eci_m: Положения (N, 3), м.
        mu_m3ps2: Гравитационный параметр, м³/с².
        j2: Коэффициент J2.
        r_ref_m: Радиус, с которым задан J2, м.

    Returns:
        Потенциальная энергия (N,), м²/с².
    """
    r_eci_m = np.atleast_2d(r_eci_m)
    radius_m = np.linalg.norm(r_eci_m, axis=1)
    sin_lat_sq = (r_eci_m[:, 2] / radius_m)**2
    return (mu_m3ps2 * j2 * r_ref_m**2 * (3.0 * sin_lat_sq - 1.0)
            / (2.0 * radius_m**3))


def j2_accel_mps2(r_eci_m: np.ndarray, mu_m3ps2: float, j2: float,
                  r_ref_m: float) -> np.ndarray:
    """Ускорение от J2 — эталонная формула из docs/PHYSICS.md.

    Args:
        r_eci_m: Положения (N, 3), м.
        mu_m3ps2: Гравитационный параметр, м³/с².
        j2: Коэффициент J2.
        r_ref_m: Радиус, с которым задан J2, м.

    Returns:
        Ускорения (N, 3), м/с².
    """
    r_eci_m = np.atleast_2d(r_eci_m)
    radius_m = np.linalg.norm(r_eci_m, axis=1)
    z_ratio_sq = (r_eci_m[:, 2] / radius_m)**2
    factor = -1.5 * j2 * mu_m3ps2 * r_ref_m**2 / radius_m**5
    return np.stack([
        factor * r_eci_m[:, 0] * (1.0 - 5.0 * z_ratio_sq),
        factor * r_eci_m[:, 1] * (1.0 - 5.0 * z_ratio_sq),
        factor * r_eci_m[:, 2] * (3.0 - 5.0 * z_ratio_sq),
    ], axis=1)


def specific_energy_j2_m2ps2(states_eci: np.ndarray, mu_m3ps2: float,
                             j2: float, r_ref_m: float) -> np.ndarray:
    """Удельная энергия с учётом J2 — сохраняется при движении с J2.

    Args:
        states_eci: Состояния (N, 6), м и м/с.
        mu_m3ps2: Гравитационный параметр, м³/с².
        j2: Коэффициент J2.
        r_ref_m: Радиус, с которым задан J2, м.

    Returns:
        Энергия (N,), м²/с².
    """
    r_m, _ = _split(states_eci)
    return (specific_energy_m2ps2(states_eci, mu_m3ps2)
            + j2_potential_energy_m2ps2(r_m, mu_m3ps2, j2, r_ref_m))


def j2_raan_rate_radps(sma_m: float, ecc: float, inclination_rad: float,
                       mu_m3ps2: float, j2: float, r_ref_m: float) -> float:
    """Вековой дрейф долготы восходящего узла из-за J2.

    dΩ/dt = −1.5·n·J2·(R/p)²·cos i.

    Args:
        sma_m: Большая полуось, м.
        ecc: Эксцентриситет.
        inclination_rad: Наклонение, рад.
        mu_m3ps2: Гравитационный параметр, м³/с².
        j2: Коэффициент J2.
        r_ref_m: Радиус, с которым задан J2, м.

    Returns:
        Скорость дрейфа узла, рад/с.
    """
    mean_motion_radps = np.sqrt(mu_m3ps2 / sma_m**3)
    semi_latus_m = sma_m * (1.0 - ecc**2)
    return float(-1.5 * mean_motion_radps * j2 * (r_ref_m / semi_latus_m)**2
                 * np.cos(inclination_rad))


def circular_decay_rate_mps(sma_m: float, density_kgpm3: float,
                            ballistic_m2pkg: float, mu_m3ps2: float) -> float:
    """Скорость снижения круговой орбиты при постоянной плотности.

    da/dt = −ρ·B·√(μ·a), где B = Cd·A/m.

    Args:
        sma_m: Большая полуось, м.
        density_kgpm3: Плотность атмосферы, кг/м³.
        ballistic_m2pkg: Баллистический коэффициент Cd·A/m, м²/кг.
        mu_m3ps2: Гравитационный параметр, м³/с².

    Returns:
        da/dt, м/с (отрицательная).
    """
    return float(-density_kgpm3 * ballistic_m2pkg * np.sqrt(mu_m3ps2 * sma_m))


def max_relative_drift(values: np.ndarray) -> float:
    """Наибольшее относительное отклонение величины от начального значения.

    Args:
        values: Скаляры (N,) или векторы (N, k) во времени.

    Returns:
        max |x(t) − x(0)| / |x(0)|.

    Raises:
        ValueError: Если начальное значение равно нулю.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    initial_norm = np.linalg.norm(values[0])
    if initial_norm == 0.0:
        raise ValueError("Начальное значение равно нулю")
    return float(np.max(np.linalg.norm(values - values[0], axis=1))
                 / initial_norm)
