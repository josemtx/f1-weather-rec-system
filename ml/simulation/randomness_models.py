"""Fuentes de aleatoriedad para la simulacion Monte Carlo (ver plan, seccion 6).

Cada funcion aisla UNA fuente de incertidumbre para que monte_carlo.py pueda
componerlas sin acoplarse a como se estima cada una.
"""

import pandas as pd

RAIN_THRESHOLD_MM = 1.0
DEFAULT_PIT_STD = 0.8  # segundos, usado cuando el equipo no tiene historial (rookie season, etc.)

_SAFETY_CAR_PROXY = {"street": 0.7, "hybrid": 0.5, "permanent": 0.3}


def climate_variability(db, circuit_short_name: str, month: int) -> dict:
    """Desviacion estandar historica (todas las fuentes/anios) de cada
    variable climatica para ese circuito+mes -- define cuanto puede
    desviarse el clima real del valor puntual usado como feature."""
    rows = list(
        db["climate_historical"].find(
            {"circuit_short_name": circuit_short_name},
            {"_id": 0, "date": 1, "temp_2m_avg": 1, "precipitation_mm": 1, "humidity_relative": 1, "wind_speed_10m": 1},
        )
    )
    if not rows:
        return {"temp_2m_avg": 2.0, "precipitation_mm": 0.5, "humidity_relative": 5.0, "wind_speed_10m": 1.0}

    df = pd.DataFrame(rows)
    df["month"] = pd.to_datetime(df["date"]).dt.month
    month_df = df[df["month"] == month]
    if len(month_df) < 3:
        month_df = df  # respaldo: no hay suficiente historial de ese mes especifico

    return {
        "temp_2m_avg": float(month_df["temp_2m_avg"].std(ddof=0)) or 1.0,
        "precipitation_mm": float(month_df["precipitation_mm"].std(ddof=0)) or 0.2,
        "humidity_relative": float(month_df["humidity_relative"].std(ddof=0)) or 3.0,
        "wind_speed_10m": float(month_df["wind_speed_10m"].std(ddof=0)) or 0.5,
    }


def rain_probability(db, circuit_short_name: str, month: int) -> float:
    rows = list(
        db["climate_historical"].find(
            {"circuit_short_name": circuit_short_name},
            {"_id": 0, "date": 1, "precipitation_mm": 1},
        )
    )
    if not rows:
        return 0.15  # prior generico si no hay historial

    df = pd.DataFrame(rows)
    df["month"] = pd.to_datetime(df["date"]).dt.month
    month_df = df[df["month"] == month]
    if len(month_df) < 3:
        month_df = df

    return float((month_df["precipitation_mm"] > RAIN_THRESHOLD_MM).mean())


def safety_car_shock_scale(circuit_type: str) -> float:
    """Cuanto se amplia la varianza del resultado por probabilidad de SC/VSC.
    Proxy documentado por tipo de circuito -- ver limitacion en el plan
    (pendiente ingesta de race_control)."""
    return _SAFETY_CAR_PROXY.get(circuit_type, 0.5)
