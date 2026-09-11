"""Categoria G: ritmo dinamico (~3 features, solo 2023+).

driver_quali_gap_to_pole_pct del plan original se omite: OpenF1 solo se
ingesta con session_name=Race (ver AppFormula1.java), no hay sesiones de
clasificacion -- no hay vueltas de quali que dar. Documentado como limitacion
conocida (ver ml/docs/ARCHITECTURE.md) en vez de aproximarlo con datos que no
representan lo mismo.
"""

import numpy as np
import pandas as pd

from ml.features.anti_leakage import rolling_mean_shifted
from ml.features.feature_defs._openf1_join import with_driver_code_and_race_key

MAX_REASONABLE_LAP_SECONDS = 180  # descarta vueltas bajo safety car/VSC extremo como outliers de ritmo


def _clean_race_laps(db) -> pd.DataFrame:
    laps_raw = pd.DataFrame(
        list(db["laps"].find(
            {}, {"_id": 0, "session_key": 1, "driver_number": 1, "lap_number": 1,
                 "lap_duration": 1, "is_pit_out_lap": 1}
        ))
    )
    if laps_raw.empty:
        return pd.DataFrame()

    laps = with_driver_code_and_race_key(db, laps_raw)
    laps = laps[
        (laps["is_pit_out_lap"] != True)  # noqa: E712
        & laps["lap_duration"].notna()
        & (laps["lap_duration"] < MAX_REASONABLE_LAP_SECONDS)
    ]
    return laps


def _pace_percentile_and_consistency(laps: pd.DataFrame) -> pd.DataFrame:
    key = ["driver_code", "circuit_short_name", "race_date"]
    per_driver = laps.groupby(key)["lap_duration"].agg(["median", "std"]).reset_index()
    per_driver.columns = key + ["driver_median_pace", "driver_pace_stddev"]

    session_key = ["circuit_short_name", "race_date"]
    per_driver["pace_percentile"] = per_driver.groupby(session_key)["driver_median_pace"].rank(pct=True)
    # Percentil invertido: 1.0 = el mas rapido del campo (menor tiempo), no el mas lento.
    per_driver["pace_percentile"] = 1.0 - per_driver["pace_percentile"]

    return per_driver


def _team_pace_trend(db, laps: pd.DataFrame, context_df: pd.DataFrame) -> pd.DataFrame:
    driver_team = context_df[["driver_code", "circuit_short_name", "race_date", "constructor_id"]].drop_duplicates()
    team_laps = laps.merge(driver_team, on=["driver_code", "circuit_short_name", "race_date"], how="inner")

    key = ["constructor_id", "circuit_short_name", "race_date"]

    def _slope(group: pd.DataFrame) -> float:
        if len(group) < 5:
            return np.nan
        return float(np.polyfit(group["lap_number"], group["lap_duration"], 1)[0])

    return team_laps.groupby(key, group_keys=False).apply(_slope, include_groups=False).reset_index(name="team_pace_trend")


def build_dynamic_pace(db, context_df: pd.DataFrame) -> pd.DataFrame:
    laps = _clean_race_laps(db)
    if laps.empty:
        cols = ["driver_avg_pace_percentile_last5", "driver_pace_consistency_stddev_last5", "team_race_pace_trend_last5"]
        return pd.DataFrame(index=context_df.index, columns=cols)

    pace = _pace_percentile_and_consistency(laps)
    team_trend = _team_pace_trend(db, laps, context_df)

    df = context_df.merge(pace, on=["driver_code", "circuit_short_name", "race_date"], how="left")
    df = df.merge(team_trend, on=["constructor_id", "circuit_short_name", "race_date"], how="left")

    out = pd.DataFrame(index=context_df.index)
    out["driver_avg_pace_percentile_last5"] = rolling_mean_shifted(df, "driver_code", "race_date", "pace_percentile", 5)
    out["driver_pace_consistency_stddev_last5"] = rolling_mean_shifted(
        df, "driver_code", "race_date", "driver_pace_stddev", 5
    )
    out["team_race_pace_trend_last5"] = rolling_mean_shifted(df, "constructor_id", "race_date", "team_pace_trend", 5)

    return out
