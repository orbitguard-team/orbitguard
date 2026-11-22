"""Синтетические начальные условия без прогноза безопасности."""

from datetime import datetime, timezone
import numpy as np

from orbitguard.constants import MU_EARTH_M3_S2
from orbitguard.models import ManeuverLimits, Scenario, SpaceObject


def make_example() -> Scenario:
    """Создаёт пример входа.

    Returns:
        Валидированный синтетический сценарий в ECI и СИ.
    """
    radius_m = 7_000_000.0
    speed_mps = float(np.sqrt(MU_EARTH_M3_S2 / radius_m))
    return Scenario(
        name="contract_fixture",
        epoch=datetime(2026, 11, 22, tzinfo=timezone.utc),
        horizon_s=600.0,
        satellite=SpaceObject("satellite", [radius_m, 0, 0, 0, speed_mps, 0]),
        debris=SpaceObject("debris", [radius_m, 1000, 0, 0, speed_mps, 0]),
        safe_distance_m=100.0,
        limits=ManeuverLimits(5.0, 0.0, 300.0),
        source=("Synthetic contract fixture; "
                "no safety outcome has been computed"),
    )


if __name__ == "__main__":
    scenario = make_example()
    print(f"Вход проверен: {scenario.name}; {scenario.frame}; "
          f"горизонт={scenario.horizon_s} с")
    print("Расчёт траектории и оценка сближения не выполнялись.")
