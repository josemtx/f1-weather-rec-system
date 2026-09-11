"""Estimacion de temperatura de asfalto para carreras futuras.

Ningun servicio meteorologico da temperatura de pista: es especifica del
motorsport. Pero OpenF1 publica, para cada sesion de 2023+, temperatura de
aire Y de asfalto medidas por sensores -- 22.134 lecturas emparejadas. Con
ellas se ajusta la relacion aire -> asfalto y se aplica sobre el pronostico.

El ajuste es POR CIRCUITO, no global, porque el asfalto de cada trazado se
calienta de forma distinta (color, composicion, exposicion al sol, hora de
la sesion). Medido sobre los propios datos:

    ajuste global              -> error medio 5.35 C
    global + humedad           -> error medio 4.60 C
    por circuito               -> error medio 2.69 C

Un circuito nuevo (Madrid en 2026) no tiene historial, asi que cae al ajuste
global con humedad. Es menos preciso y esta asumido: mejor una estimacion
con error conocido que dejar la feature vacia.
"""

import logging

import numpy as np
import pandas as pd

from ml.features.feature_defs._openf1_join import sessions_lookup

logger = logging.getLogger(__name__)

MIN_READINGS_PER_CIRCUIT = 200
ALL_SESSIONS = ["Race", "Qualifying", "Sprint", "Sprint Qualifying"]


def _fit(frame: pd.DataFrame) -> np.ndarray | None:
    """Minimos cuadrados de track_temp ~ air_temp + humedad + constante."""
    if len(frame) < 10:
        return None
    x = np.column_stack([
        frame["air_temperature"].to_numpy(dtype=float),
        frame["humidity"].to_numpy(dtype=float),
        np.ones(len(frame)),
    ])
    y = frame["track_temperature"].to_numpy(dtype=float)
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    return coef


def fit_air_to_track(db) -> dict:
    """{'global': coef, 'por_circuito': {circuito: coef}} o {} si no hay datos."""
    rows = list(db["weather"].find({}, {"_id": 0}))
    if not rows:
        return {}

    weather = pd.DataFrame(rows)
    sessions = sessions_lookup(db, ALL_SESSIONS)
    if sessions.empty:
        return {}

    merged = weather.merge(sessions, on="session_key", how="inner")
    if merged.empty:
        return {}

    global_coef = _fit(merged)
    if global_coef is None:
        return {}

    per_circuit = {}
    for circuit, group in merged.groupby("circuit_short_name"):
        if len(group) < MIN_READINGS_PER_CIRCUIT:
            continue
        coef = _fit(group)
        if coef is not None:
            per_circuit[circuit] = coef

    logger.info(
        f"Relacion aire->asfalto ajustada con {len(merged)} lecturas "
        f"({len(per_circuit)} circuitos con ajuste propio)"
    )
    return {"global": global_coef, "por_circuito": per_circuit}


def estimate_track_temp(
    model: dict, circuit_short_name: pd.Series, air_temp: pd.Series, humidity: pd.Series
) -> pd.Series:
    """Estimacion vectorizada; NaN donde falten aire o humedad."""
    if not model:
        return pd.Series(np.nan, index=air_temp.index)

    air = pd.to_numeric(air_temp, errors="coerce")
    hum = pd.to_numeric(humidity, errors="coerce")
    result = pd.Series(np.nan, index=air.index, dtype="float64")

    for idx in air.index:
        a, h = air.get(idx), hum.get(idx)
        if pd.isna(a) or pd.isna(h):
            continue
        coef = model["por_circuito"].get(circuit_short_name.get(idx), model["global"])
        result.at[idx] = float(coef[0] * a + coef[1] * h + coef[2])

    return result
