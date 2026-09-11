"""Categoria C: forma historica del equipo/constructor (~6 features)."""

import pandas as pd

from ml.features.anti_leakage import (
    expanding_mean_shifted,
    rolling_mean_shifted,
    rolling_rate_shifted,
    rolling_sum_shifted,
)


def build_team_form(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_team_circuit_key"] = df["constructor_id"] + "|" + df["circuit_short_name"].fillna("UNK")
    df["_podium"] = df["finish_position"].fillna(99) <= 3
    df["_dnf"] = ~df["finished"]

    out = pd.DataFrame(index=df.index)
    out["team_avg_finish_last5"] = rolling_mean_shifted(df, "constructor_id", "race_date", "finish_position", 5)
    out["team_avg_finish_last10"] = rolling_mean_shifted(df, "constructor_id", "race_date", "finish_position", 10)
    out["team_podium_rate_last10"] = rolling_rate_shifted(df, "constructor_id", "race_date", "_podium", 10)
    out["team_dnf_rate_last10"] = rolling_rate_shifted(df, "constructor_id", "race_date", "_dnf", 10)
    out["team_points_last5"] = rolling_sum_shifted(df, "constructor_id", "race_date", "points", 5)
    out["team_circuit_history_avg_finish"] = expanding_mean_shifted(
        df, "_team_circuit_key", "race_date", "finish_position"
    )

    return out
