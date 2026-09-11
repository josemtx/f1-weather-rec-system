"""Categoria D: priors de circuito (~5 features).

circuit_overtaking_difficulty es estatico (de circuits.json, ya en context_df).
Los demas son historicos "as-of race date" -- IMPORTANTE: a diferencia de
driver/team form, estos agregan sobre TODOS los pilotos de una misma carrera,
que comparten la misma fecha. Un shift(1) fila-a-fila filtraria entre pilotos
de la MISMA carrera (fuga). Por eso se agrega primero a nivel carrera (una
fila por circuito+fecha) y se desplaza ahi, antes de unir de vuelta a nivel
piloto.

circuit_pit_loss_time_estimate y circuit_typical_num_stops dependen de las
colecciones `pit`/`stints` (OpenF1) -- si estan vacias (p.ej. bloqueo temporal
de OpenF1), estas columnas quedan en NaN, que XGBoost maneja nativamente.
"""

import pandas as pd

from ml.features.anti_leakage import expanding_mean_shifted
from ml.features.feature_defs._openf1_join import sessions_lookup

# Proxy documentado (ver plan, seccion D): sin ingesta de race_control todavia,
# se aproxima la propension a safety car por tipo de circuito.
_SAFETY_CAR_PROXY = {"street": 0.7, "hybrid": 0.5, "permanent": 0.3}


def _race_level_shifted_mean(df: pd.DataFrame, value_col: str) -> pd.Series:
    race_level = (
        df.groupby(["circuit_short_name", "race_date"], as_index=False)[value_col]
        .mean()
        .sort_values(["circuit_short_name", "race_date"])
    )
    race_level["_shifted"] = expanding_mean_shifted(
        race_level, "circuit_short_name", "race_date", value_col
    )
    merged = df.merge(
        race_level[["circuit_short_name", "race_date", "_shifted"]],
        on=["circuit_short_name", "race_date"],
        how="left",
    )
    return merged["_shifted"]


def _pit_loss_estimate(db, context_df: pd.DataFrame) -> pd.Series:
    # sessions_lookup filtra a Race (sessions incluye tambien Qualifying/
    # Sprint/Sprint Qualifying) y traduce los nombres de circuito de OpenF1
    # al canonico de circuits.json -- sin eso el join falla en 7 circuitos.
    sessions_df = sessions_lookup(db, ["Race"])
    if sessions_df.empty:
        return pd.Series([float("nan")] * len(context_df), index=context_df.index)

    sessions_df = sessions_df.rename(columns={"race_date": "date_start"})
    pit = list(db["pit"].find({}, {"_id": 0, "session_key": 1, "pit_duration": 1}))
    if not pit:
        return pd.Series([float("nan")] * len(context_df), index=context_df.index)

    pit_df = pd.DataFrame(pit).merge(sessions_df, on="session_key", how="left")
    pit_df["race_date"] = pd.to_datetime(pit_df["date_start"]).dt.date

    race_level = pit_df.groupby(["circuit_short_name", "race_date"], as_index=False)["pit_duration"].mean()
    race_level["race_date"] = pd.to_datetime(race_level["race_date"])
    race_level = race_level.sort_values(["circuit_short_name", "race_date"])
    race_level["_shifted"] = expanding_mean_shifted(race_level, "circuit_short_name", "race_date", "pit_duration")

    merged = context_df.merge(
        race_level[["circuit_short_name", "race_date", "_shifted"]],
        on=["circuit_short_name", "race_date"],
        how="left",
    )
    return merged["_shifted"]


def _typical_num_stops(db, context_df: pd.DataFrame) -> pd.Series:
    # Mismo motivo que _pit_loss_estimate: filtrado a Race + traduccion de
    # nombres de circuito, ambos resueltos por sessions_lookup.
    sessions_df = sessions_lookup(db, ["Race"])
    if sessions_df.empty:
        return pd.Series([float("nan")] * len(context_df), index=context_df.index)

    sessions_df = sessions_df.rename(columns={"race_date": "date_start"})
    stints = list(db["stints"].find({}, {"_id": 0, "session_key": 1, "driver_number": 1, "stint_number": 1}))
    if not stints:
        return pd.Series([float("nan")] * len(context_df), index=context_df.index)

    stints_df = pd.DataFrame(stints)
    n_stops = stints_df.groupby(["session_key", "driver_number"])["stint_number"].max().reset_index()
    n_stops = n_stops.merge(sessions_df, on="session_key", how="left")
    n_stops["race_date"] = pd.to_datetime(n_stops["date_start"]).dt.date

    race_level = n_stops.groupby(["circuit_short_name", "race_date"], as_index=False)["stint_number"].mean()
    race_level["race_date"] = pd.to_datetime(race_level["race_date"])
    race_level = race_level.sort_values(["circuit_short_name", "race_date"])
    race_level["_shifted"] = expanding_mean_shifted(race_level, "circuit_short_name", "race_date", "stint_number")

    merged = context_df.merge(
        race_level[["circuit_short_name", "race_date", "_shifted"]],
        on=["circuit_short_name", "race_date"],
        how="left",
    )
    return merged["_shifted"]


def build_circuit_priors(db, context_df: pd.DataFrame) -> pd.DataFrame:
    df = context_df.copy()
    df["_dnf"] = ~df["finished"]

    out = pd.DataFrame(index=context_df.index)
    out["circuit_avg_dnf_rate_historical"] = _race_level_shifted_mean(df, "_dnf")
    out["circuit_avg_safety_car_proxy"] = context_df["circuit_type"].map(_SAFETY_CAR_PROXY)
    out["circuit_pit_loss_time_estimate"] = _pit_loss_estimate(db, context_df)
    out["circuit_typical_num_stops"] = _typical_num_stops(db, context_df)

    return out
