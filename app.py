"""Интерфейс расчёта сближений и импульсных манёвров OrbitGuard."""

from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from orbitguard import models
from orbitguard.maneuver import build_rtn_candidates, search_maneuver
from orbitguard.metrics import export_result_json, summarize
from orbitguard.propagation import propagate
from orbitguard.scenarios import load_scenario, scenario_paths


SCENARIO_DIR = Path(__file__).resolve().parent / "scenarios"
SCENARIO_LABELS = {
    "safe_flyby": "Безопасный пролёт",
    "dangerous_feasible": "Опасное сближение",
    "late_warning_no_solution": "Позднее предупреждение",
}
# Пока достаточно небольшой воспроизводимой сетки вдоль орбиты.
TIME_SAMPLES = 3
BUDGET_FRACTIONS = np.array([0.2, 0.5, 1.0])
DIRECTIONS_RTN = ([0, 1, 0], [0, -1, 0])
METERS_PER_KM = 1000


@dataclass(frozen=True)
class Run:
    """Сохраняет данные одного запуска для диаграмм, метрик и экспорта."""

    scenario: models.Scenario
    result: models.ManeuverResult
    candidates: tuple[models.Impulse, ...]
    elapsed_s: float
    debris: models.Trajectory | None
    before: models.Trajectory | None
    after: models.Trajectory | None


def calculate(scenario: models.Scenario) -> Run:
    """Рассчитывает результат и траектории для отображения.

    Args:
        scenario: Исходные состояния и параметры расчёта в СИ.

    Returns:
        Результат запуска, траектории, кандидаты и длительность.
    """
    start = perf_counter()
    span = (0.0, scenario.horizon_s)

    def track(
        obj: models.SpaceObject, impulses: tuple[models.Impulse, ...] = (),
    ) -> models.Trajectory:
        """Распространяет объект на общей шкале времени сценария.

        Args:
            obj: Объект с исходным состоянием в СИ.
            impulses: Импульсы спутника, не более одного.

        Returns:
            Траектория объекта в ECI-J2000.
        """
        return propagate(
            obj, scenario.epoch, span, scenario.forces,
            numerics=scenario.numerics, impulses=impulses,
        )

    limits = scenario.limits
    candidates = ()
    try:
        # Базовая траектория нужна для перевода направлений RTN в ECI.
        before = track(scenario.satellite)
    except models.PropagationError as exc:
        result = models.ManeuverResult(
            models.Status.PROPAGATION_FAILED, None, None, None,
            forces=scenario.forces, message=f"Сбой распространения: {exc}",
        )
        return Run(
            scenario, result, candidates, perf_counter() - start,
            None, None, None,
        )

    if limits.dv_max_mps > 0:
        times = np.unique(np.linspace(
            limits.t_earliest_s, limits.t_latest_s, TIME_SAMPLES,
        ))
        candidates = build_rtn_candidates(
            before, times_s=times.tolist(),
            magnitudes_mps=(limits.dv_max_mps * BUDGET_FRACTIONS).tolist(),
            directions_rtn=DIRECTIONS_RTN,
        )
    result = search_maneuver(scenario, candidates=candidates)
    if result.before is None:
        return Run(
            scenario, result, candidates, perf_counter() - start,
            None, None, None,
        )

    debris = track(scenario.debris)
    after = track(scenario.satellite, (result.best,)) if result.best else None
    return Run(
        scenario, result, candidates, perf_counter() - start,
        debris, before, after,
    )


def _add_path(
    figure: go.Figure, trajectory: models.Trajectory, name: str, color: str,
) -> None:
    """Рисует траекторию в километрах.

    Args:
        figure: Пространственный график.
        trajectory: Траектория в СИ.
        name: Название ряда.
        color: Цвет линии.
    """
    x, y, z = (trajectory.states[:, :3] / METERS_PER_KM).T
    figure.add_scatter3d(
        x=x, y=y, z=z, mode="lines", name=name, line={"color": color},
    )


def _add_marker(
    figure: go.Figure, trajectory: models.Trajectory, time_s: float,
    name: str,
) -> None:
    """Рисует точку на траектории в момент расчётного события.

    Args:
        figure: Пространственный график.
        trajectory: Траектория в СИ.
        time_s: Момент события в секундах от эпохи.
        name: Название маркера.
    """
    x, y, z = trajectory.state_at(time_s)[:3] / METERS_PER_KM
    figure.add_scatter3d(
        x=[x], y=[y], z=[z], mode="markers", name=name,
        text=[f"t = {time_s:.3f} с"],
        hovertemplate="%{text}<extra>%{fullData.name}</extra>",
    )


def _add_distance(
    figure: go.Figure, satellite: models.Trajectory, debris: models.Trajectory,
    approach: models.Approach, name: str, color: str,
) -> None:
    """Сравнивает положения объектов в одни и те же моменты времени.

    Args:
        figure: График расстояния.
        satellite: Траектория спутника в СИ.
        debris: Траектория второго объекта в СИ.
        approach: Рассчитанный минимум расстояния.
        name: Название ряда.
        color: Цвет линии и маркера.
    """
    times = np.union1d(satellite.t_s, debris.t_s)
    times = np.union1d(times, [approach.t_s])
    distances = [
        np.linalg.norm(
            satellite.state_at(float(time))[:3]
            - debris.state_at(float(time))[:3]
        ) / METERS_PER_KM
        for time in times
    ]
    figure.add_scatter(
        x=times, y=distances, mode="lines", name=name, line={"color": color},
    )
    figure.add_scatter(
        x=[approach.t_s], y=[approach.distance_m / METERS_PER_KM],
        mode="markers", name=f"Минимум · {name.lower()}",
        marker={"color": color, "size": 9},
    )


def show_charts(run: Run) -> None:
    """Строит графики рассчитанных траекторий и расстояния.

    Args:
        run: Результат одного запуска и его траектории.
    """
    result = run.result
    if run.before is None or run.debris is None or result.before is None:
        return
    positions = go.Figure()
    distances = go.Figure()
    _add_path(positions, run.debris, run.scenario.debris.name, "#f59e0b")
    _add_path(positions, run.before, "Спутник до манёвра", "#38bdf8")
    _add_marker(
        positions, run.before, result.before.closest.t_s, "Сближение до",
    )
    _add_distance(
        distances, run.before, run.debris, result.before.closest,
        "До манёвра", "#38bdf8",
    )
    if run.after is not None and result.after is not None and result.best:
        _add_path(positions, run.after, "Спутник после манёвра", "#34d399")
        _add_marker(
            positions, run.after, result.after.closest.t_s, "Сближение после",
        )
        _add_marker(positions, run.after, result.best.t_s, "Импульс")
        _add_distance(
            distances, run.after, run.debris, result.after.closest,
            "После манёвра", "#34d399",
        )
        distances.add_vline(
            x=result.best.t_s, line_dash="dot", annotation_text="Импульс",
        )

    positions.update_layout(scene={
        "xaxis_title": "X, км", "yaxis_title": "Y, км",
        "zaxis_title": "Z, км", "aspectmode": "data",
    })
    distances.add_hline(
        y=run.scenario.safe_distance_m / METERS_PER_KM,
        line_dash="dash", annotation_text="Порог",
    )
    distances.update_layout(
        xaxis_title="Время от эпохи, с", yaxis_title="Расстояние, км",
    )
    st.subheader("Траектории")
    st.plotly_chart(positions)
    st.subheader("Расстояние между объектами")
    st.plotly_chart(distances)


def clear_run() -> None:
    """Сбрасывает результат, когда пользователь редактирует вход."""
    st.session_state.pop("run", None)


def show_result(run: Run) -> None:
    """Показывает исход расчёта и действие скачивания отчёта.

    Args:
        run: Результат одного запуска с его исходными параметрами.
    """
    result = run.result
    status = models.Status
    if result.status is status.ALREADY_SAFE:
        st.success(
            "На заданном горизонте исходная дистанция достигает порога "
            "с учётом запаса."
        )
    elif result.status is status.FEASIBLE:
        st.success("Найден допустимый импульс среди проверенных кандидатов.")
    elif result.status is status.NO_FEASIBLE_CANDIDATE:
        st.warning("Среди проверенных кандидатов допустимый манёвр не найден.")
    else:
        st.error(result.message or "Не удалось рассчитать траектории.")

    metrics = summarize(
        result, scenario=run.scenario, compute_time_s=run.elapsed_s,
    )
    st.caption(
        f"Эпоха UTC: {run.scenario.epoch.isoformat()} · "
        f"{run.scenario.frame} · Горизонт: {run.scenario.horizon_s:g} с · "
        f"Расчёт: {run.elapsed_s:.2f} с"
    )
    columns = st.columns(3)
    for column, side, label in zip(
        columns, ("before", "after"), ("Минимум до", "Минимум после"),
    ):
        approach = metrics[side]
        if approach is not None:
            column.metric(label, f"{approach['d_min_m']:.2f} м")
            column.caption(f"Время от эпохи: {approach['t_min_s']:.3f} с")
    delta_v = metrics["dv_mps"]
    if delta_v is not None:
        columns[2].metric("Δv", f"{delta_v:.4f} м/с")
        if metrics["impulse_t_s"] is not None:
            columns[2].caption(
                f"Время импульса: {metrics['impulse_t_s']:.3f} с",
            )

    show_charts(run)
    st.download_button(
        "Скачать JSON-отчёт",
        export_result_json(
            run.scenario, result, candidates=run.candidates,
            compute_time_s=run.elapsed_s,
        ),
        file_name=f"{run.scenario.name}-result.json",
        mime="application/json",
    )


def settings(scenario: models.Scenario) -> tuple[dict, bool]:
    """Собирает настройки с отдельным состоянием для каждого сценария.

    Args:
        scenario: Сценарий со значениями по умолчанию.

    Returns:
        Настройки и признак нажатия кнопки запуска.
    """
    prefix = scenario.name
    limits = scenario.limits

    def number(label: str, value: float, step: float, key: str) -> float:
        """Создаёт числовое поле со сбросом предыдущего результата.

        Args:
            label: Подпись поля с единицами.
            value: Исходное значение.
            step: Шаг редактирования.
            key: Ключ поля в пределах сценария.

        Returns:
            Текущее числовое значение.
        """
        return st.number_input(
            label, value=float(value), step=step, key=f"{prefix}-{key}",
            on_change=clear_run, format="%.3f",
        )

    threshold = number(
        "Безопасная дистанция, м", scenario.safe_distance_m, 100.0, "distance",
    )
    horizon = number("Горизонт, с", scenario.horizon_s, 60.0, "horizon")
    budget = number("Максимальный Δv, м/с", limits.dv_max_mps, 0.05, "budget")
    with st.expander("Окно манёвра"):
        earliest = number("Начало, с", limits.t_earliest_s, 60.0, "start")
        latest = number("Конец, с", limits.t_latest_s, 60.0, "end")
    with st.expander("Дополнительные силы"):
        forces = {
            key: st.checkbox(
                label, value=getattr(scenario.forces, key),
                key=f"{prefix}-{key}", on_change=clear_run,
            )
            for key, label in (
                ("j2", "Несферичность Земли (J2)"),
                ("drag", "Атмосферное сопротивление"),
                ("moon", "Притяжение Луны"), ("sun", "Притяжение Солнца"),
            )
        }
        st.caption("Текущий расчёт поддерживает центральную гравитацию.")
    return {
        "safe_distance_m": threshold, "horizon_s": horizon,
        "limits": (budget, earliest, latest), "forces": forces,
    }, st.button("Рассчитать", type="primary")


def main() -> None:
    """Запускает интерфейс Streamlit."""
    st.set_page_config(page_title="OrbitGuard", layout="wide")
    st.title("OrbitGuard")
    paths = scenario_paths(SCENARIO_DIR)
    if not paths:
        st.error("В папке scenarios нет доступных сценариев.")
        return

    with st.sidebar:
        selected = st.selectbox(
            "Сценарий", paths, on_change=clear_run,
            format_func=lambda path: SCENARIO_LABELS.get(path.stem, path.stem),
        )
        try:
            scenario = load_scenario(selected)
        except (ValueError, OSError) as exc:
            st.error(f"Не удалось загрузить сценарий: {exc}")
            return
        values, submitted = settings(scenario)

    if scenario.synthetic:
        st.info(
            "Синтетические начальные данные. Графики строятся расчётом RK45."
        )
    if submitted:
        clear_run()
        try:
            scenario = replace(
                scenario, safe_distance_m=values["safe_distance_m"],
                horizon_s=values["horizon_s"],
                limits=models.ManeuverLimits(*values["limits"]),
                forces=replace(scenario.forces, **values["forces"]),
            )
            with st.spinner("Рассчитываем траектории и проверяем манёвры…"):
                st.session_state["run"] = calculate(scenario)
        except models.UnsupportedForceError as exc:
            st.error(f"Выбранная сила пока не поддерживается: {exc}")
        except ValueError as exc:
            st.error(f"Проверьте настройки сценария: {exc}")
        except models.PropagationError as exc:
            st.error(f"Сбой распространения: {exc}")

    if "run" in st.session_state:
        show_result(st.session_state["run"])
    elif not submitted:
        st.caption("Выберите сценарий и нажмите «Рассчитать».")


if __name__ == "__main__":
    main()
