"""Categoria B: forma historica del piloto (~10 features), sin fuga temporal.

Todas las columnas usan los helpers de anti_leakage.py (shift(1) antes de
rolling/expanding) sobre `context_df` ya ordenado por fecha de carrera.
"""

import numpy as np
import pandas as pd

from ml.features.anti_leakage import (
    expanding_count_shifted,
    expanding_mean_shifted,
    expanding_sum_shifted,
    rolling_mean_shifted,
    rolling_rate_shifted,
    rolling_sum_shifted,
)

RAN_FULL_DISTANCE_STATUS = r"^(Finished|Lapped|\+\d+ Laps?)$"


def _build_h2h_flags(df: pd.DataFrame) -> pd.Series:
    """Para cada fila, True si el piloto termino por delante de su companero
    de equipo en esa misma carrera (DNF trata como peor que cualquier finish)."""
    df = df.copy()
    df["_effective_pos"] = df["finish_position"].fillna(999)

    def _flag_group(group: pd.DataFrame) -> pd.Series:
        if len(group) != 2:
            return pd.Series([np.nan] * len(group), index=group.index)
        a, b = group.index
        beat = pd.Series(index=group.index, dtype=float)
        beat[a] = float(group.loc[a, "_effective_pos"] < group.loc[b, "_effective_pos"])
        beat[b] = float(group.loc[b, "_effective_pos"] < group.loc[a, "_effective_pos"])
        return beat

    return df.groupby(["year", "round", "constructor_id"], group_keys=False).apply(_flag_group)


def build_driver_form(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_driver_circuit_key"] = df["driver_code"] + "|" + df["circuit_short_name"].fillna("UNK")
    df["_driver_season_key"] = df["driver_code"] + "|" + df["year"].astype(str)
    df["_podium"] = df["finish_position"].fillna(99) <= 3
    df["_dnf"] = ~df["finished"]
    df["_grid_to_finish_delta"] = df["grid_position"] - df["finish_position"].fillna(df["grid_position"] + 10)
    df["_win"] = df["finish_position"].fillna(99) == 1
    df["_h2h_beat_teammate"] = _build_h2h_flags(df)
    # Ritmo sin fiabilidad: un abandono clasifica P20 y hunde la media de
    # llegadas aunque el piloto ruede en el podio cuando termina. Se anula
    # (NaN) y la media movil lo ignora; la fiabilidad ya vive en
    # driver_dnf_rate_last10. Es el ancla pre-clasificacion del regresor.
    # "Corrio la distancia" se decide por el estado y no por `finished`:
    # quien se retira con el 90% recorrido queda clasificado (finished=True,
    # status "Retired"), y eso tampoco es ritmo.
    ran_full_distance = df["status"].astype(str).str.match(RAN_FULL_DISTANCE_STATUS)
    df["_finish_if_ran"] = df["finish_position"].where(ran_full_distance)

    out = pd.DataFrame(index=df.index)
    out["driver_avg_finish_last5"] = rolling_mean_shifted(df, "driver_code", "race_date", "finish_position", 5)
    out["driver_avg_finish_when_finished_last5"] = rolling_mean_shifted(
        df, "driver_code", "race_date", "_finish_if_ran", 5
    )
    out["driver_avg_finish_last10"] = rolling_mean_shifted(df, "driver_code", "race_date", "finish_position", 10)
    out["driver_avg_finish_season_to_date"] = expanding_mean_shifted(
        df, "_driver_season_key", "race_date", "finish_position"
    )
    out["driver_podium_rate_last10"] = rolling_rate_shifted(df, "driver_code", "race_date", "_podium", 10)
    out["driver_dnf_rate_last10"] = rolling_rate_shifted(df, "driver_code", "race_date", "_dnf", 10)
    out["driver_points_last5"] = rolling_sum_shifted(df, "driver_code", "race_date", "points", 5)
    out["driver_avg_grid_to_finish_delta_last10"] = rolling_mean_shifted(
        df, "driver_code", "race_date", "_grid_to_finish_delta", 10
    )
    out["driver_circuit_history_avg_finish"] = expanding_mean_shifted(
        df, "_driver_circuit_key", "race_date", "finish_position"
    )
    out["driver_teammate_h2h_rate_last10"] = rolling_rate_shifted(
        df, "driver_code", "race_date", "_h2h_beat_teammate", 10
    )
    out["driver_career_wins_to_date"] = expanding_sum_shifted(df, "driver_code", "race_date", "_win")
    out["driver_n_prior_races"] = expanding_count_shifted(df, "driver_code", "race_date")

    return out
