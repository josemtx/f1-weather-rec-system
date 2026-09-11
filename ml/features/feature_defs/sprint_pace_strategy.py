"""Extension de categoria I: ritmo/estrategia REAL de sprint (OpenF1 laps/stints
de la sesion 'Sprint', no solo el resultado final de Jolpica). Solo fines de
semana con sprint (2021+ resultado, 2023+ con datos OpenF1 reales de vueltas).

Igual que sprint_form.py, el rolling se hace sobre la SECUENCIA DE SPRINTS de
cada piloto, no sobre todas las carreras -- NaN fuera de fines de semana sprint.
"""

import pandas as pd

from ml.features.anti_leakage import rolling_mean_shifted
from ml.features.feature_defs._openf1_join import with_driver_code_and_race_key

MAX_REASONABLE_LAP_SECONDS = 180


def _sprint_race_level(db) -> pd.DataFrame:
    """OJO: with_driver_code_and_race_key devuelve `race_date` = fecha REAL
    de la sesion Sprint (viernes/sabado segun calendario), no la fecha de
    carrera de context_df -- se une por circuito+anio (derivado de esa misma
    fecha, nunca cruza el limite de anio), no por fecha exacta."""
    stints_raw = pd.DataFrame(
        list(db["stints"].find(
            {}, {"_id": 0, "session_key": 1, "driver_number": 1, "stint_number": 1, "lap_start": 1, "lap_end": 1}
        ))
    )
    laps_raw = pd.DataFrame(
        list(db["laps"].find(
            {}, {"_id": 0, "session_key": 1, "driver_number": 1, "lap_duration": 1, "is_pit_out_lap": 1}
        ))
    )
    if stints_raw.empty or laps_raw.empty:
        return pd.DataFrame(columns=["driver_code", "circuit_short_name", "year", "n_stints", "pace_percentile"])

    stints = with_driver_code_and_race_key(db, stints_raw, session_names=["Sprint"])
    laps = with_driver_code_and_race_key(db, laps_raw, session_names=["Sprint"])
    if stints.empty or laps.empty:
        return pd.DataFrame(columns=["driver_code", "circuit_short_name", "year", "n_stints", "pace_percentile"])

    stints["year"] = stints["race_date"].dt.year
    laps["year"] = laps["race_date"].dt.year

    key = ["driver_code", "circuit_short_name", "year"]
    n_stints = stints.groupby(key).size().reset_index(name="n_stints")

    laps = laps[(laps["is_pit_out_lap"] != True) & laps["lap_duration"].notna() & (laps["lap_duration"] < MAX_REASONABLE_LAP_SECONDS)]  # noqa: E712
    median_pace = laps.groupby(key)["lap_duration"].median().reset_index(name="driver_median_pace")
    session_key = ["circuit_short_name", "year"]
    median_pace["pace_percentile"] = 1.0 - median_pace.groupby(session_key)["driver_median_pace"].rank(pct=True)

    return n_stints.merge(median_pace[key + ["pace_percentile"]], on=key, how="outer")


def build_sprint_pace_strategy(db, context_df: pd.DataFrame) -> pd.DataFrame:
    race_level = _sprint_race_level(db)
    if race_level.empty:
        cols = ["driver_avg_sprint_num_stops_last5", "driver_avg_sprint_pace_percentile_last5"]
        return pd.DataFrame(index=context_df.index, columns=cols)

    df = context_df.merge(race_level, on=["driver_code", "circuit_short_name", "year"], how="left")

    out = pd.DataFrame(index=context_df.index)
    out["driver_avg_sprint_num_stops_last5"] = rolling_mean_shifted(df, "driver_code", "race_date", "n_stints", 5)
    out["driver_avg_sprint_pace_percentile_last5"] = rolling_mean_shifted(
        df, "driver_code", "race_date", "pace_percentile", 5
    )

    return out
