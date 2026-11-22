"""Streamlit interface prototype for OG-07."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st


SCENARIOS = {
    "Дальний пролёт": (15, 0.0),
    "Близкий пролёт": (1, 0.4),
    "Почти совпадающие пути": (0.2, 1.0),
}


def demo_data(scenario, horizon_s):
    """Generate illustrative coordinates in km on a shared time grid."""
    offset_km, phase = SCENARIOS[scenario]
    t_s = np.linspace(0, horizon_s, 401)
    angle = np.linspace(-0.8, 0.8, len(t_s))
    first = np.column_stack(
        (7000 * np.cos(angle), 7000 * np.sin(angle), np.zeros_like(angle))
    )
    delta = np.column_stack(
        (
            np.full_like(angle, offset_km),
            30 * (angle - phase / 5),
            4 * (angle - phase / 5),
        )
    )
    second = first + delta
    distance_km = np.linalg.norm(delta, axis=1)
    return t_s, first, second, distance_km


def trajectory_chart(t_s, first, second, closest, impulse):
    figure = go.Figure()
    for name, points in (("Спутник", first), ("Второй объект", second)):
        figure.add_scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode="lines",
            name=name,
        )
    for name, index, color in (
        ("Сближение", closest, "#ff9843"),
        ("Пример точки импульса", impulse, "#a78bfa"),
    ):
        x, y, z = first[index]
        figure.add_scatter3d(
            x=[x],
            y=[y],
            z=[z],
            mode="markers",
            marker={"size": 6, "color": color},
            name=name,
            text=[f"t = {t_s[index]:.1f} с"],
            hovertemplate="%{text}<extra>%{fullData.name}</extra>",
        )
    figure.update_layout(
        scene={
            "xaxis_title": "X, км",
            "yaxis_title": "Y, км",
            "zaxis_title": "Z, км",
            "aspectmode": "data",
        },
        legend={"orientation": "h"},
    )
    return figure


def distance_chart(t_s, distance_km, threshold_km, closest, impulse):
    figure = go.Figure()
    figure.add_scatter(x=t_s, y=distance_km, mode="lines", name="Расстояние")
    figure.add_scatter(
        x=[t_s[closest]],
        y=[distance_km[closest]],
        mode="markers",
        marker={"color": "#ff9843", "size": 9},
        name="Сближение",
    )
    figure.add_hline(y=threshold_km, line_dash="dash", annotation_text="Порог")
    figure.add_vline(x=t_s[impulse], line_dash="dot", annotation_text="Пример импульса")
    figure.update_layout(xaxis_title="Время от начала, с", yaxis_title="Расстояние, км")
    return figure


def main():
    st.set_page_config(page_title="OrbitGuard", layout="wide")
    st.title("OrbitGuard")
    st.info(
        "Демонстрационные данные. Траектории и точка импульса условны. "
        "Орбитальный расчёт и подбор манёвра появятся в OG-08."
    )

    with st.sidebar:
        st.header("Настройки сценария")
        scenario = st.selectbox("Сценарий", tuple(SCENARIOS))
        threshold_km = st.number_input("Порог сближения, км", value=2.0, step=0.1)
        horizon_s = st.number_input("Горизонт, с", value=3600, step=60)
        max_delta_v = st.number_input("Максимальный Δv, м/с", value=5.0, step=0.1)
        st.caption("Лимит Δv будет использован при подборе манёвра в OG-08.")
        submitted = st.button("Построить демонстрацию", type="primary")

    settings = (scenario, threshold_km, horizon_s, max_delta_v)
    if submitted:
        st.session_state.pop("demo_settings", None)
        if not all(np.isfinite(value) and value > 0 for value in settings[1:]):
            st.error("Порог, горизонт и лимит Δv должны быть конечными числами больше нуля.")
            return
        with st.spinner("Подготавливаем демонстрационные данные…"):
            st.session_state["demo_data"] = demo_data(scenario, horizon_s)
            st.session_state["demo_settings"] = settings

    if st.session_state.get("demo_settings") != settings:
        st.caption("Выберите настройки и нажмите «Построить демонстрацию».")
        return

    t_s, first, second, distance_km = st.session_state["demo_data"]
    closest = int(np.argmin(distance_km))
    impulse = len(t_s) // 4
    st.subheader("Демонстрационные траектории")
    st.plotly_chart(trajectory_chart(t_s, first, second, closest, impulse))
    st.subheader("Расстояние между объектами")
    st.plotly_chart(distance_chart(t_s, distance_km, threshold_km, closest, impulse))


if __name__ == "__main__":
    main()
