"""Categoria H: clasificacion (~9 features).

driver_quali_position y driver_quali_gap_to_pole_pct son datos de ESTA
carrera (no historicos) -- legitimos como feature porque la clasificacion
ocurre el sabado, antes de la carrera del domingo, igual que grid_position
en context.py. El resto (avg_last5/10, historial en circuito) sigue la
regla anti-fuga de siempre.

Clima de clasificacion: NASA POWER ya cubre el anio calendario completo por
circuito (no solo domingos), asi que no hace falta ingesta adicional -- solo
unir climate_historical por `qualifying_date` en vez de `race_date`.
"""

import pandas as pd

from ml.features.anti_leakage import expanding_mean_shifted, rolling_mean_shifted
from ml.features.feature_defs.climate import coalesce_climate
from ml.features.feature_defs._openf1_join import with_driver_code_and_race_key

RAIN_THRESHOLD_MM = 1.0


def _quali_results(db) -> pd.DataFrame:
    rows = list(db["qualifying_results"].find({}, {"_id": 0}))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["qualifying_date"] = pd.to_datetime(df["qualifying_date"])

    pole_time = df.groupby(["year", "round"])["best_seconds"].transform("min")
    df["quali_gap_to_pole_pct"] = (df["best_seconds"] - pole_time) / pole_time * 100
    return df


def _quali_climate(db, quali_df: pd.DataFrame, context_df: pd.DataFrame) -> pd.DataFrame:
    raw = list(db["climate_historical"].find({}, {"_id": 0}))
    if not raw or quali_df.empty:
        return pd.DataFrame(columns=["year", "round", "quali_day_temp_avg", "quali_day_precipitation_mm", "quali_day_rain_flag"])

    coalesced = coalesce_climate(pd.DataFrame(raw))
    coalesced["date"] = pd.to_datetime(coalesced["date"])

    circuit_by_round = context_df[["year", "round", "circuit_short_name"]].drop_duplicates()
    quali_with_circuit = quali_df[["year", "round", "qualifying_date"]].drop_duplicates().merge(
        circuit_by_round, on=["year", "round"], how="left"
    )

    merged = quali_with_circuit.merge(
        coalesced, left_on=["circuit_short_name", "qualifying_date"], right_on=["circuit_short_name", "date"], how="left"
    )
    merged["quali_day_rain_flag"] = (merged["precipitation_mm"] > RAIN_THRESHOLD_MM).astype(float)

    return merged[["year", "round", "temp_2m_avg", "precipitation_mm", "quali_day_rain_flag"]].rename(
        columns={"temp_2m_avg": "quali_day_temp_avg", "precipitation_mm": "quali_day_precipitation_mm"}
    )


def _quali_laps_and_tyres(db) -> pd.DataFrame:
    """Datos REALES de OpenF1 (sesion Qualifying, 2023+): numero de vueltas
    lanzadas y que fraccion se corrio en blando -- indicador de agresividad
    de programa/evolucion de pista, no disponible via Jolpica (que solo da
    el mejor tiempo por segmento, no cuantas vueltas hizo falta para lograrlo).

    OJO: with_driver_code_and_race_key devuelve `race_date` = fecha REAL de
    la sesion (sabado para Qualifying), no la fecha de carrera de context_df
    -- por eso se une por circuito+anio (`year`, derivado de esa misma fecha,
    nunca cruza el limite de anio entre sabado y domingo), no por fecha exacta.
    """
    laps_raw = pd.DataFrame(
        list(db["laps"].find({}, {"_id": 0, "session_key": 1, "driver_number": 1, "lap_duration": 1, "is_pit_out_lap": 1}))
    )
    stints_raw = pd.DataFrame(
        list(db["stints"].find({}, {"_id": 0, "session_key": 1, "driver_number": 1, "compound": 1, "lap_start": 1, "lap_end": 1}))
    )
    if laps_raw.empty or stints_raw.empty:
        return pd.DataFrame(columns=["driver_code", "circuit_short_name", "year", "n_quali_laps", "quali_soft_tyre_share"])

    laps = with_driver_code_and_race_key(db, laps_raw, session_names=["Qualifying"])
    stints = with_driver_code_and_race_key(db, stints_raw, session_names=["Qualifying"])
    if laps.empty or stints.empty:
        return pd.DataFrame(columns=["driver_code", "circuit_short_name", "year", "n_quali_laps", "quali_soft_tyre_share"])

    laps["year"] = laps["race_date"].dt.year
    stints["year"] = stints["race_date"].dt.year

    key = ["driver_code", "circuit_short_name", "year"]
    laps = laps[laps["is_pit_out_lap"] != True]  # noqa: E712
    n_laps = laps.groupby(key).size().reset_index(name="n_quali_laps")

    stints["laps_used"] = stints["lap_end"] - stints["lap_start"] + 1
    soft_laps = stints[stints["compound"] == "SOFT"].groupby(key)["laps_used"].sum()
    total_laps = stints.groupby(key)["laps_used"].sum()
    soft_share = (soft_laps / total_laps).reset_index(name="quali_soft_tyre_share")

    return n_laps.merge(soft_share, on=key, how="outer")


def build_qualifying(db, context_df: pd.DataFrame) -> pd.DataFrame:
    quali = _quali_results(db)
    if quali.empty:
        cols = [
            "driver_quali_position", "grid_penalty_positions", "quali_gap_to_pole_pct",
            "driver_avg_quali_position_last5", "driver_avg_quali_position_last10",
            "driver_circuit_quali_history_avg_position",
            "quali_day_temp_avg", "quali_day_precipitation_mm", "quali_day_rain_flag",
            "conditions_delta_quali_to_race", "conditions_rain_changed",
        ]
        return pd.DataFrame(index=context_df.index, columns=cols)

    df = context_df.merge(
        quali[["year", "round", "driver_code", "qualifying_position", "quali_gap_to_pole_pct"]],
        on=["year", "round", "driver_code"], how="left",
    )
    df["_driver_circuit_key"] = df["driver_code"] + "|" + df["circuit_short_name"].fillna("UNK")

    quali_climate = _quali_climate(db, quali, context_df)
    df = df.merge(quali_climate, on=["year", "round"], how="left")

    quali_laps_tyres = _quali_laps_and_tyres(db)
    df = df.merge(quali_laps_tyres, on=["driver_code", "circuit_short_name", "year"], how="left")

    out = pd.DataFrame(index=context_df.index)
    out["driver_quali_position"] = df["qualifying_position"]
    out["grid_penalty_positions"] = df["grid_position"] - df["qualifying_position"]
    out["quali_gap_to_pole_pct"] = df["quali_gap_to_pole_pct"]
    out["driver_avg_quali_position_last5"] = rolling_mean_shifted(
        df.assign(qualifying_position=df["qualifying_position"]), "driver_code", "race_date", "qualifying_position", 5
    )
    out["driver_avg_quali_position_last10"] = rolling_mean_shifted(
        df.assign(qualifying_position=df["qualifying_position"]), "driver_code", "race_date", "qualifying_position", 10
    )
    out["driver_circuit_quali_history_avg_position"] = expanding_mean_shifted(
        df.assign(qualifying_position=df["qualifying_position"]), "_driver_circuit_key", "race_date", "qualifying_position"
    )
    out["quali_day_temp_avg"] = df["quali_day_temp_avg"]
    out["quali_day_precipitation_mm"] = df["quali_day_precipitation_mm"]
    out["quali_day_rain_flag"] = df["quali_day_rain_flag"]
    out["n_quali_laps"] = df["n_quali_laps"]
    out["quali_soft_tyre_share"] = df["quali_soft_tyre_share"]
    # conditions_delta_quali_to_race y conditions_rain_changed se calculan en
    # build_features.py, DESPUES de unir esta categoria con clima (E) -- ambas
    # mitades (clima de carrera y de clasificacion) hacen falta a la vez.

    return out
