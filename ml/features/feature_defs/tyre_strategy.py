"""Categoria F: neumaticos/estrategia (~5 features). Solo 2023+ (OpenF1
laps/stints/pit) -- NaN en 2018-2022, que XGBoost maneja nativamente.
"""

import numpy as np
import pandas as pd

from ml.features.anti_leakage import rolling_mean_shifted
from ml.features.feature_defs._openf1_join import with_driver_code_and_race_key


def _stint_level_metrics(db) -> pd.DataFrame:
    stints = pd.DataFrame(
        list(db["stints"].find(
            {}, {"_id": 0, "session_key": 1, "driver_number": 1, "stint_number": 1,
                 "compound": 1, "lap_start": 1, "lap_end": 1}
        ))
    )
    if stints.empty:
        return pd.DataFrame()
    return with_driver_code_and_race_key(db, stints)


def _race_level_strategy(db) -> pd.DataFrame:
    stints = _stint_level_metrics(db)
    if stints.empty:
        return pd.DataFrame(columns=["driver_code", "circuit_short_name", "race_date", "n_stints", "preferred_compound_share"])

    stints["laps_used"] = stints["lap_end"] - stints["lap_start"] + 1
    key = ["driver_code", "circuit_short_name", "race_date"]

    n_stints = stints.groupby(key).size().reset_index(name="n_stints")

    compound_laps = stints.groupby(key + ["compound"])["laps_used"].sum().reset_index()
    total_laps = stints.groupby(key)["laps_used"].sum().reset_index(name="total_laps")
    compound_laps = compound_laps.merge(total_laps, on=key)
    compound_laps["share"] = compound_laps["laps_used"] / compound_laps["total_laps"]
    preferred = compound_laps.groupby(key)["share"].max().reset_index(name="preferred_compound_share")

    return n_stints.merge(preferred, on=key, how="outer")


def _degradation_rate(db) -> pd.DataFrame:
    """Pendiente de lap_duration vs indice de vuelta dentro del stint, por
    carrera+piloto (media de todos sus stints). Vueltas de entrada/salida de
    boxes se excluyen: distorsionan el ritmo sin reflejar degradacion real."""
    stints = _stint_level_metrics(db)
    laps_raw = pd.DataFrame(
        list(db["laps"].find(
            {}, {"_id": 0, "session_key": 1, "driver_number": 1, "lap_number": 1,
                 "lap_duration": 1, "is_pit_out_lap": 1}
        ))
    )
    if stints.empty or laps_raw.empty:
        return pd.DataFrame(columns=["driver_code", "circuit_short_name", "race_date", "tyre_degradation_rate"])

    laps = with_driver_code_and_race_key(db, laps_raw)
    laps = laps[(laps["is_pit_out_lap"] != True) & laps["lap_duration"].notna()]  # noqa: E712

    merged = laps.merge(
        stints[["session_key", "driver_number", "stint_number", "lap_start", "lap_end"]],
        on=["session_key", "driver_number"], how="inner",
    )
    merged = merged[(merged["lap_number"] >= merged["lap_start"]) & (merged["lap_number"] <= merged["lap_end"])]
    merged["in_stint_index"] = merged["lap_number"] - merged["lap_start"]

    def _slope(group: pd.DataFrame) -> float:
        if len(group) < 3:
            return np.nan
        return float(np.polyfit(group["in_stint_index"], group["lap_duration"], 1)[0])

    key = ["driver_code", "circuit_short_name", "race_date", "session_key", "driver_number", "stint_number"]
    per_stint = merged.groupby(key, group_keys=False).apply(_slope, include_groups=False)
    per_stint = per_stint.reset_index(name="slope")

    race_key = ["driver_code", "circuit_short_name", "race_date"]
    return per_stint.groupby(race_key)["slope"].mean().reset_index(name="tyre_degradation_rate")


def _pit_duration_by_team(db, context_df: pd.DataFrame) -> pd.DataFrame:
    pit_raw = pd.DataFrame(list(db["pit"].find({}, {"_id": 0, "session_key": 1, "driver_number": 1, "pit_duration": 1})))
    if pit_raw.empty:
        return pd.DataFrame(columns=["constructor_id", "circuit_short_name", "race_date", "pit_duration_mean", "pit_duration_std"])

    pit = with_driver_code_and_race_key(db, pit_raw)
    driver_team = context_df[["driver_code", "circuit_short_name", "race_date", "constructor_id"]].drop_duplicates()
    pit = pit.merge(driver_team, on=["driver_code", "circuit_short_name", "race_date"], how="inner")

    key = ["constructor_id", "circuit_short_name", "race_date"]
    agg = pit.groupby(key)["pit_duration"].agg(["mean", "std"]).reset_index()
    agg.columns = key + ["pit_duration_mean", "pit_duration_std"]
    return agg


def build_tyre_strategy(db, context_df: pd.DataFrame) -> pd.DataFrame:
    df = context_df.copy()

    strategy = _race_level_strategy(db)
    df = df.merge(strategy, on=["driver_code", "circuit_short_name", "race_date"], how="left")

    degradation = _degradation_rate(db)
    df = df.merge(degradation, on=["driver_code", "circuit_short_name", "race_date"], how="left")

    pit_team = _pit_duration_by_team(db, context_df)
    df = df.merge(pit_team, on=["constructor_id", "circuit_short_name", "race_date"], how="left")

    out = pd.DataFrame(index=context_df.index)
    out["driver_avg_num_stops_last5"] = rolling_mean_shifted(df, "driver_code", "race_date", "n_stints", 5)
    out["driver_preferred_compound_share_last5"] = rolling_mean_shifted(
        df, "driver_code", "race_date", "preferred_compound_share", 5
    )
    out["driver_tyre_degradation_rate_last5"] = rolling_mean_shifted(
        df, "driver_code", "race_date", "tyre_degradation_rate", 5
    )
    out["team_avg_pit_stop_duration_last5"] = rolling_mean_shifted(
        df, "constructor_id", "race_date", "pit_duration_mean", 5
    )
    out["team_pit_stop_duration_stddev_last5"] = rolling_mean_shifted(
        df, "constructor_id", "race_date", "pit_duration_std", 5
    )

    return out
