"""Categoria A: features de contexto (~8) -- base de la matriz de features.

Fuente: historical_results (Jolpica, grid+resultado, todos los anios) join
circuits (jolpica_circuit_id -> circuit_short_name/tipo/altitud/longitud).
Esta es la unica fuente de grid_position (ver plan: decision de usar Jolpica
para TODOS los anios en vez de mezclar OpenF1 2023+/Jolpica 2018-2022).
"""

import pandas as pd


def build_context(db) -> pd.DataFrame:
    results = list(
        db["historical_results"].find(
            {},
            {
                "_id": 0,
                "year": 1,
                "round": 1,
                "circuit_id": 1,
                "race_date": 1,
                "driver_code": 1,
                "driver_id": 1,
                "driver_full_name": 1,
                "constructor_id": 1,
                "constructor_name": 1,
                "grid_position": 1,
                "finish_position": 1,
                "points": 1,
                "status": 1,
                "finished": 1,
            },
        )
    )
    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    circuits = list(
        db["circuits"].find(
            {},
            {
                "_id": 0,
                "jolpica_circuit_id": 1,
                "circuit_short_name": 1,
                "circuit_type": 1,
                "altitude_m": 1,
                "length_km": 1,
                "overtaking_difficulty": 1,
                "country": 1,
            },
        )
    )
    circuits_df = pd.DataFrame(circuits).rename(columns={"jolpica_circuit_id": "circuit_id"})

    df = df.merge(circuits_df, on="circuit_id", how="left")

    unmatched = df[df["circuit_short_name"].isna()]["circuit_id"].unique()
    if len(unmatched) > 0:
        # No abortamos: se documenta y esas filas quedan con context NaN
        # (mejor que perder el resto de la temporada por un circuito nuevo).
        df.attrs["unmatched_circuit_ids"] = list(unmatched)

    df["is_street_circuit"] = df["circuit_type"] == "street"
    df["is_hybrid_circuit"] = df["circuit_type"] == "hybrid"

    # Proxy pendiente de verificar contra cobertura real de OpenF1 (ver Dia 1:
    # bloqueado por sesion en vivo). Una vez laps/stints tengan datos reales,
    # recalcular por session_key en vez de por anio.
    df["era_data_tier"] = df["year"].apply(lambda y: "openf1_full" if y >= 2023 else "jolpica_basic")

    df["race_date"] = pd.to_datetime(df["race_date"])

    return df
