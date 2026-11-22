"""Тесты сценариев OG-06 (#6): формат, происхождение и ожидаемые статусы.

Эталоны — аналитика задачи двух тел (уравнение Кеплера, линейная теория
Клохесси — Уилтшира) и независимый численный расчёт OG-02/OG-03, а не
повтор вывода генератора. Описание — docs/scenarios.md.
"""

import importlib.util
import json
import math
import pathlib
import sys
from collections.abc import Callable

import numpy as np
import pytest

from orbitguard import constants, models, scenarios

ROOT = pathlib.Path(__file__).resolve().parents[1]
MU_M3PS2 = constants.MU_EARTH_M3_S2

REQUIRED_STATUSES = {
    models.Status.ALREADY_SAFE,
    models.Status.FEASIBLE,
    models.Status.NO_FEASIBLE_CANDIDATE,
}
# Точность аналитического переноса: float64 на нескольких витках.
KEPLER_POSITION_TOL_M = 1e-5
KEPLER_VELOCITY_TOL_MPS = 1e-8
INVARIANT_REL_TOL = 1e-12
# МКС-подобная орбита по BACKLOG OG-06: ≈420 км, наклонение 51.6°.
ISS_ALTITUDE_RANGE_M = (400e3, 440e3)
ISS_INCLINATION_DEG = 51.6
ISS_INCLINATION_TOL_DEG = 0.1
# Совпадение линейной матрицы перехода с теорией Клохесси — Уилтшира:
# поправки порядка e ≈ 7e-4 и (Δv/v)·nτ ≪ 1%.
CW_REL_TOL = 0.01


# Генератор и аналитический эталон — скрипты в tools/, не часть пакета.
sys.path.insert(0, str(ROOT / "tools"))
# pylint: disable=wrong-import-position,wrong-import-order
import build_scenarios as build
import twobody_reference as ref
# pylint: enable=wrong-import-position,wrong-import-order


def _load_example():
    path = ROOT / "examples" / "contract_example.py"
    spec = importlib.util.spec_from_file_location("contract_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _files() -> list[pathlib.Path]:
    return scenarios.scenario_paths(ROOT / "scenarios")


def _loaded() -> dict[str, tuple]:
    return {path.stem: scenarios.load_scenario_file(path)
            for path in _files()}


def _required_m(scenario: models.Scenario) -> float:
    return scenario.safe_distance_m + scenario.numerics.distance_margin_m


# ---------------------------------------------------------------------------
# Файлы сценариев
# ---------------------------------------------------------------------------

def test_all_scenarios_load():
    """Все scenarios/*.json загружаются, имя совпадает с файлом."""
    loaded = _loaded()
    assert len(loaded) >= 3
    for stem, (scenario, expected) in loaded.items():
        assert scenario.name == stem
        assert expected is not None, f"{stem}: нет блока expected"


def test_required_statuses_present():
    """Есть безопасный, опасный с решением и без решения в пределах."""
    statuses = {expected.status for _, expected in _loaded().values()}
    assert REQUIRED_STATUSES <= statuses


def test_synthetic_scenarios_have_source():
    """Синтетические данные помечены и ссылаются на генератор."""
    for scenario, _ in _loaded().values():
        assert scenario.synthetic is True
        assert "tools/build_scenarios.py" in scenario.source
        assert "docs/scenarios.md" in scenario.source


def test_files_are_canonical_and_roundtrip(tmp_path: pathlib.Path):
    """Сохранение после загрузки даёт тот же текст бит в бит.

    Args:
        tmp_path: Временный каталог pytest.
    """
    for path in _files():
        scenario, expected = scenarios.load_scenario_file(path)
        copy_path = tmp_path / path.name
        scenarios.save_scenario(scenario, copy_path, expected)
        assert copy_path.read_text(encoding="utf-8") == path.read_text(
            encoding="utf-8")
        again, expected_again = scenarios.load_scenario_file(copy_path)
        assert (scenarios.scenario_to_dict(again, expected_again)
                == scenarios.scenario_to_dict(scenario, expected))


def test_stored_states_match_generator():
    """Файлы воспроизводятся генератором из документированных параметров."""
    assert not build.check(build.build_all())


def test_iss_like_satellite_orbit():
    """Спутник на МКС-подобной орбите: ≈420 км, наклонение 51.6°."""
    for scenario, _ in _loaded().values():
        info = ref.describe_orbit(scenario.satellite.state0)
        low, high = ISS_ALTITUDE_RANGE_M
        assert low <= info["perigee_alt_m"] <= info["apogee_alt_m"] <= high
        assert (abs(info["inclination_deg"] - ISS_INCLINATION_DEG)
                <= ISS_INCLINATION_TOL_DEG)


def test_forces_are_supported_baseline():
    """Сценарии построены и проверены только для центрального поля."""
    for scenario, _ in _loaded().values():
        assert scenario.forces == models.ForceModel(
            j2=False, drag=False, moon=False, sun=False, density_scale=1.0)


def test_contract_example_loads():
    """Пример OG-01 examples/scenario.json читается загрузчиком."""
    path = ROOT / "examples" / "scenario.json"
    if not path.exists():
        pytest.skip("Нет examples/scenario.json из OG-01 (PR #16)")
    example = _load_example()
    scenario, expected = scenarios.load_scenario_file(path)
    assert expected is None
    assert (scenarios.scenario_to_dict(scenario)
            == scenarios.scenario_to_dict(example.make_example()))


# ---------------------------------------------------------------------------
# Строгость загрузчика
# ---------------------------------------------------------------------------

def _valid() -> dict:
    path = ROOT / "scenarios" / "dangerous_feasible.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _set(path: str, value):
    def mutate(data: dict) -> None:
        *parents, last = path.split(".")
        target = data
        for key in parents:
            target = target[key]
        if value is _DELETE:
            del target[last]
        else:
            target[last] = value
    return mutate


_DELETE = object()


@pytest.mark.parametrize("mutate, message", [
    (_set("extra", 1), "неизвестные поля extra"),
    (_set("satellite.state0_eci", [0.0] * 6), "неизвестные поля"),
    (_set("horizon_s", _DELETE), "нет полей horizon_s"),
    (_set("satellite.mass_kg", _DELETE), "нет полей mass_kg"),
    (_set("schema_version", 2), "schema_version"),
    (_set("schema_version", True), "schema_version"),
    (_set("frame", "TEME"), "frame"),
    (_set("epoch", "2026-12-01T00:00:00"), "UTC"),
    (_set("epoch", "2026-12-01T03:00:00+03:00"), "UTC"),
    (_set("epoch", "1 декабря"), "ISO 8601"),
    (_set("safe_distance_m", -1000.0), "safe_distance_m"),
    (_set("horizon_s", "14400"), "конечное число"),
    (_set("horizon_s", True), "конечное число"),
    (_set("satellite.state0", [1.0, 2.0]), "6 чисел"),
    (_set("satellite.state0", [7e6, 0, 0, 0, 7546, 0, 0]), "6 чисел"),
    (_set("forces.j2", 1), "true или false"),
    (_set("synthetic", "yes"), "synthetic"),
    (_set("source", "  "), "source"),
    (_set("limits.t_latest_s", 20_000.0), "horizon"),
    (_set("expected.status", "safe"), "expected.status"),
    (_set("expected.witness_impulse", _DELETE), "witness_impulse"),
    (_set("expected.shift_bound_m", 10.0), "shift_bound_m"),
    (_set("expected.witness_impulse.t_s", 9_500.0), "окна"),
    (_set("expected.witness_impulse.dv_eci_mps", [0.6, 0.0, 0.0]),
     "dv_max"),
    (_set("expected.witness_impulse.min_distance_m", 900.0), "безопасной"),
    (_set("expected.closest_approach.distance_m", 5000.0), "не опасно"),
    (_set("expected.tolerance.t_s", 0.0), "допуски"),
])
def test_invalid_scenario_raises(tmp_path: pathlib.Path,
                                 mutate: Callable[[dict], None],
                                 message: str):
    """Неверные поля, единицы, эпоха и противоречия — ValueError.

    Args:
        tmp_path: Временный каталог pytest.
        mutate: Порча корректного сценария.
        message: Ожидаемый фрагмент текста ошибки.
    """
    data = _valid()
    mutate(data)
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        scenarios.load_scenario(path)


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity", "1e400"])
def test_non_finite_numbers_rejected(tmp_path: pathlib.Path, token: str):
    """NaN, бесконечность и переполнение не попадают в расчёт.

    Args:
        tmp_path: Временный каталог pytest.
        token: Неконечное число в тексте JSON.
    """
    text = json.dumps(_valid()).replace('"horizon_s": 14400.0',
                                        f'"horizon_s": {token}')
    path = tmp_path / "bad.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        scenarios.load_scenario(path)


def test_duplicate_keys_rejected(tmp_path: pathlib.Path):
    """Повторный ключ не перезаписывает значение молча.

    Args:
        tmp_path: Временный каталог pytest.
    """
    text = json.dumps(_valid()).replace(
        '"horizon_s": 14400.0', '"horizon_s": 14400.0, "horizon_s": 1.0')
    path = tmp_path / "bad.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="повторный ключ"):
        scenarios.load_scenario(path)


def test_already_safe_requires_margin(tmp_path: pathlib.Path):
    """already_safe ровно на пороге без запаса отклоняется.

    Args:
        tmp_path: Временный каталог pytest.
    """
    data = json.loads((ROOT / "scenarios" / "safe_flyby.json").read_text(
        encoding="utf-8"))
    data["expected"]["closest_approach"]["distance_m"] = (
        data["safe_distance_m"])
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="already_safe"):
        scenarios.load_scenario(path)


def test_scenario_without_expectation_loads(tmp_path: pathlib.Path):
    """Блок expected необязателен: пользовательский файл без него читается.

    Args:
        tmp_path: Временный каталог pytest.
    """
    data = _valid()
    del data["expected"]
    path = tmp_path / "user.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    scenario, expected = scenarios.load_scenario_file(path)
    assert expected is None
    assert scenario.name == "dangerous_feasible"


# ---------------------------------------------------------------------------
# Аналитический эталон генератора
# ---------------------------------------------------------------------------

def _state_by_kepler_equation(elements: dict, dt_s: float) -> np.ndarray:
    """Независимый перенос: средняя аномалия → уравнение Кеплера."""
    sma, ecc = elements["sma_m"], elements["ecc"]
    nu0 = elements["true_anomaly_rad"]
    ecc_anomaly0 = 2.0 * math.atan2(math.sqrt(1 - ecc) * math.sin(nu0 / 2),
                                    math.sqrt(1 + ecc) * math.cos(nu0 / 2))
    mean0 = ecc_anomaly0 - ecc * math.sin(ecc_anomaly0)
    mean = mean0 + math.sqrt(MU_M3PS2 / sma**3) * dt_s
    ecc_anomaly = mean
    for _ in range(50):
        ecc_anomaly -= ((ecc_anomaly - ecc * math.sin(ecc_anomaly) - mean)
                        / (1 - ecc * math.cos(ecc_anomaly)))
    nu = 2.0 * math.atan2(math.sqrt(1 + ecc) * math.sin(ecc_anomaly / 2),
                          math.sqrt(1 - ecc) * math.cos(ecc_anomaly / 2))
    return ref.elements_to_state(
        sma, ecc, elements["inclination_rad"], elements["raan_rad"],
        elements["argp_rad"], nu)


ELLIPSE = {"sma_m": 7_500e3, "ecc": 0.1, "inclination_rad": 0.5,
           "raan_rad": 1.0, "argp_rad": 2.0, "true_anomaly_rad": 0.3}


def test_elements_to_state_invariants():
    """r = p/(1+e·cos ν), |h| = √(μp), энергия −μ/2a, наклонение."""
    state = ref.elements_to_state(*ELLIPSE.values())
    sma, ecc = ELLIPSE["sma_m"], ELLIPSE["ecc"]
    p_m = sma * (1 - ecc**2)
    r_vec, v_vec = state[:3], state[3:]
    h_vec = np.cross(r_vec, v_vec)
    assert np.linalg.norm(r_vec) == pytest.approx(
        p_m / (1 + ecc * math.cos(ELLIPSE["true_anomaly_rad"])),
        rel=INVARIANT_REL_TOL)
    assert np.linalg.norm(h_vec) == pytest.approx(
        math.sqrt(MU_M3PS2 * p_m), rel=INVARIANT_REL_TOL)
    assert (v_vec @ v_vec / 2 - MU_M3PS2 / np.linalg.norm(r_vec)
            == pytest.approx(-MU_M3PS2 / (2 * sma), rel=INVARIANT_REL_TOL))
    assert math.acos(h_vec[2] / np.linalg.norm(h_vec)) == pytest.approx(
        ELLIPSE["inclination_rad"], rel=INVARIANT_REL_TOL)


@pytest.mark.parametrize("dt_s", [0.0, 1.0, -1.0, 600.0, -4_321.5,
                                  20_000.0, -55_555.0])
def test_kepler_matches_kepler_equation(dt_s: float):
    """Универсальная переменная совпадает с уравнением Кеплера.

    Args:
        dt_s: Интервал переноса, с; отрицательный — назад.
    """
    state0 = ref.elements_to_state(*ELLIPSE.values())
    got = ref.kepler(state0, dt_s)[0]
    want = _state_by_kepler_equation(ELLIPSE, dt_s)
    assert np.max(np.abs(got[:3] - want[:3])) < KEPLER_POSITION_TOL_M
    assert np.max(np.abs(got[3:] - want[3:])) < KEPLER_VELOCITY_TOL_MPS


def test_kepler_period_returns_and_reverses():
    """Через период — в исходную точку; вперёд и назад — тождество."""
    for scenario, _ in _loaded().values():
        state0 = scenario.satellite.state0
        period_s = ref.describe_orbit(state0)["period_s"]
        back = ref.kepler(ref.kepler(state0, 7_777.7)[0], -7_777.7)[0]
        full = ref.kepler(state0, period_s)[0]
        for got in (back, full):
            assert np.max(np.abs(got[:3] - state0[:3])) < KEPLER_POSITION_TOL_M
            assert (np.max(np.abs(got[3:] - state0[3:]))
                    < KEPLER_VELOCITY_TOL_MPS)


def test_designed_tca_is_global_minimum():
    """Заданное сближение — глобальный минимум на горизонте.

    Остальные локальные минимумы и границы дальше порога с запасом,
    поэтому манёвр решает одну опасную встречу.
    """
    for name, (scenario, expected) in _loaded().items():
        minima = ref.approaches(ref.Track(scenario.satellite.state0),
                                  ref.Track(scenario.debris.state0),
                                  scenario.horizon_s)
        found = ref.closest(minima)
        assert found.t_s == pytest.approx(expected.closest.t_s, abs=1e-6)
        assert found.distance_m == pytest.approx(
            expected.closest.distance_m, abs=1e-6)
        assert found.rel_speed_mps == pytest.approx(
            expected.closest.rel_speed_mps, abs=1e-6)
        others = [item for item in minima if item.t_s != found.t_s]
        assert all(item.distance_m >= _required_m(scenario)
                   for item in others), name


def test_encounters_are_high_speed_crossings():
    """Встречи «под углом», как типично в LEO: 10–15 км/с."""
    for scenario, expected in _loaded().values():
        assert 10e3 <= expected.closest.rel_speed_mps <= 15e3, scenario.name


def test_feasible_witness_is_admissible():
    """Пример импульса в пределах окна и dv_max даёт безопасную дистанцию."""
    scenario, expected = _loaded()["dangerous_feasible"]
    impulse = expected.witness.impulse
    sat = ref.Track(scenario.satellite.state0, impulse.t_s,
                      impulse.dv_eci_mps)
    minima = ref.approaches(sat, ref.Track(scenario.debris.state0),
                              scenario.horizon_s)
    assert ref.closest(minima).distance_m == pytest.approx(
        expected.witness.min_distance_m, abs=1e-6)
    assert expected.witness.min_distance_m >= _required_m(scenario)
    at_burn = ref.kepler(scenario.satellite.state0, impulse.t_s)[0]
    dv_rtn = ref.rtn_basis(at_burn).T @ impulse.dv_eci_mps
    np.testing.assert_allclose(dv_rtn, [0.0, 0.1, 0.0], atol=1e-15)


def _cw_sigma_max(mean_motion_radps: float, tau_s: float) -> float:
    """σ_max матрицы Φ_rv Клохесси — Уилтшира (R, T, N)."""
    nt = mean_motion_radps * tau_s
    sin, cos = math.sin(nt), math.cos(nt)
    stm = np.array([
        [sin, 2 * (1 - cos), 0.0],
        [2 * (cos - 1), 4 * sin - 3 * nt, 0.0],
        [0.0, 0.0, sin],
    ]) / mean_motion_radps
    return float(np.linalg.svd(stm, compute_uv=False)[0])


def test_no_solution_bound_proves_infeasibility():
    """Ни один импульс в пределах не уводит дальше порога.

    Граница из файла ≥ dv_max·σ_max(Φ_rv), σ_max сверено с линейной теорией
    Клохесси — Уилтшира, и d + граница < порог + запас.
    """
    scenario, expected = _loaded()["late_warning_no_solution"]
    tca_s = expected.closest.t_s
    bound = ref.shift_bound(scenario.satellite.state0, tca_s,
                              scenario.limits)
    assert bound.bound_m <= expected.shift_bound_m
    assert (expected.closest.distance_m + expected.shift_bound_m
            < _required_m(scenario))
    mean_motion = math.sqrt(
        MU_M3PS2 / ref.describe_orbit(scenario.satellite.state0)["sma_m"]**3)
    worst_tau_s = tca_s - scenario.limits.t_earliest_s
    assert bound.max_sigma_s == pytest.approx(
        _cw_sigma_max(mean_motion, worst_tau_s), rel=CW_REL_TOL)


def test_no_solution_random_impulses_stay_unsafe():
    """Случайные допустимые импульсы (seed 0) не дают безопасности."""
    scenario, _ = _loaded()["late_warning_no_solution"]
    rng = np.random.default_rng(0)
    limits = scenario.limits
    deb = ref.Track(scenario.debris.state0)
    for _ in range(20):
        direction = rng.normal(size=3)
        dv = limits.dv_max_mps * direction / np.linalg.norm(direction)
        t_s = rng.uniform(limits.t_earliest_s, limits.t_latest_s)
        sat = ref.Track(scenario.satellite.state0, t_s, dv)
        minima = ref.approaches(sat, deb, scenario.horizon_s)
        assert ref.closest(minima).distance_m < _required_m(scenario)


# ---------------------------------------------------------------------------
# Независимый численный расчёт OG-02 и OG-03
# ---------------------------------------------------------------------------

def _numeric(scenario: models.Scenario, impulses=()):
    propagation = pytest.importorskip(
        "orbitguard.propagation", reason="ждёт OG-02 (#2)")
    conjunction = pytest.importorskip(
        "orbitguard.conjunction", reason="ждёт OG-03 (#3)")
    span = (0.0, scenario.horizon_s)
    sat = propagation.propagate(scenario.satellite, scenario.epoch, span,
                                scenario.forces, numerics=scenario.numerics,
                                impulses=impulses)
    deb = propagation.propagate(scenario.debris, scenario.epoch, span,
                                scenario.forces, numerics=scenario.numerics)
    return conjunction.find_closest_approach(
        sat, deb, scenario.safe_distance_m, span, numerics=scenario.numerics)


@pytest.mark.parametrize("name", ["safe_flyby", "dangerous_feasible",
                                  "late_warning_no_solution"])
def test_numerical_closest_approach_matches_design(name: str):
    """RK45 + поиск сближения подтверждают момент и дистанцию.

    Args:
        name: Имя сценария.
    """
    scenario, expected = _loaded()[name]
    result = _numeric(scenario)
    assert result.closest.t_s == pytest.approx(
        expected.closest.t_s, abs=expected.time_tol_s)
    assert result.closest.distance_m == pytest.approx(
        expected.closest.distance_m, abs=expected.distance_tol_m)
    safe = result.closest.distance_m >= _required_m(scenario)
    assert safe == (expected.status == models.Status.ALREADY_SAFE)


def test_numerical_witness_is_safe():
    """Численно после примера манёвра дистанция безопасна на всём горизонте."""
    scenario, expected = _loaded()["dangerous_feasible"]
    result = _numeric(scenario, impulses=(expected.witness.impulse,))
    assert result.closest.distance_m == pytest.approx(
        expected.witness.min_distance_m, abs=expected.distance_tol_m)
    assert result.closest.distance_m >= _required_m(scenario)


def test_numerical_worst_impulses_stay_unsafe():
    """Численно худшие импульсы (наибольшее смещение) не спасают."""
    scenario, _ = _loaded()["late_warning_no_solution"]
    limits = scenario.limits
    for t_s in (limits.t_earliest_s, limits.t_latest_s):
        at_burn = ref.kepler(scenario.satellite.state0, t_s)[0]
        basis = ref.rtn_basis(at_burn)
        for axis in range(3):
            for sign in (1.0, -1.0):
                dv = sign * limits.dv_max_mps * basis[:, axis]
                result = _numeric(scenario, (models.Impulse(t_s, dv),))
                assert result.closest.distance_m < _required_m(scenario)


@pytest.mark.parametrize("name", ["safe_flyby", "dangerous_feasible",
                                  "late_warning_no_solution"])
def test_expected_status_matches_result(name: str):
    """search_maneuver (OG-04) возвращает ожидаемый статус.

    Кандидаты: для feasible — пример из файла; для остальных — импульсы
    dv_max по ±R, ±T, ±N на краях окна.

    Args:
        name: Имя сценария.
    """
    maneuver = pytest.importorskip("orbitguard.maneuver",
                                   reason="ждёт OG-04 (#4)")
    scenario, expected = _loaded()[name]
    limits = scenario.limits
    candidates = []
    if expected.witness is not None:
        candidates.append(expected.witness.impulse)
    for t_s in (limits.t_earliest_s, limits.t_latest_s):
        basis = ref.rtn_basis(
            ref.kepler(scenario.satellite.state0, t_s)[0])
        for axis in range(3):
            for sign in (1.0, -1.0):
                candidates.append(models.Impulse(
                    t_s, sign * limits.dv_max_mps * basis[:, axis]))
    if expected.status == models.Status.ALREADY_SAFE:
        candidates = []
    result = maneuver.search_maneuver(scenario,
                                      candidates=tuple(candidates))
    assert result.status == expected.status
