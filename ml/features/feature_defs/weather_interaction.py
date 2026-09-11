"""Categoria J: interaccion piloto x clima.

Motivacion medida, no supuesta: la ablacion por regimenes de informacion
(ml/training/ablation.py) mostro que quitar TODAS las features de clima no
empeoraba el modelo (MAE 3.890 vs 3.886). La hipotesis es que la pregunta
estaba mal planteada: el clima no actua como efecto directo ("llueve ->
peor") sino como interaccion ("llueve Y este piloto rinde bien mojado").

Un arbol puede aprender esa interaccion por si solo, pero necesita muchos
ejemplos de cada combinacion piloto x condicion, y las carreras con lluvia
son ~5% del total. Estas features se la dan ya calculada.

La feature clave es el DELTA, no la media absoluta: que un piloto acabe 8o
de media en mojado solo dice que su coche es mediano. Que acabe 8o en mojado
cuando su media en seco es 12o dice que EL piloto mejora bajo lluvia.
"""

import pandas as pd

from ml.features.anti_leakage import (
    conditional_expanding_count_shifted,
    conditional_expanding_mean_shifted,
)

HOT_DELTA_THRESHOLD_C = 3.0


def build_weather_interaction(df: pd.DataFrame, climate_df: pd.DataFrame) -> pd.DataFrame:
    """df: context_df.  climate_df: salida de build_climate (mismo indice)."""
    work = df.copy()
    work["_finish"] = pd.to_numeric(work["finish_position"], errors="coerce")
    work["_was_wet"] = climate_df["climate_rain_probability_flag"].fillna(0) > 0
    work["_was_dry"] = ~work["_was_wet"]

    # "Calurosa" es relativo al propio circuito y epoca del anio: 30C en
    # Bahrein es normal, en Spa es una anomalia. Por eso se usa el delta
    # frente a la norma estacional y no la temperatura absoluta.
    temp_delta = climate_df["climate_temp_delta_vs_norm"]
    work["_was_hot"] = temp_delta > HOT_DELTA_THRESHOLD_C

    out = pd.DataFrame(index=df.index)

    wet_avg = conditional_expanding_mean_shifted(work, "driver_code", "race_date", "_finish", "_was_wet")
    dry_avg = conditional_expanding_mean_shifted(work, "driver_code", "race_date", "_finish", "_was_dry")
    hot_avg = conditional_expanding_mean_shifted(work, "driver_code", "race_date", "_finish", "_was_hot")

    out["driver_avg_finish_in_wet"] = wet_avg
    out["driver_avg_finish_in_dry"] = dry_avg
    # Negativo = mejora bajo lluvia respecto a su propio nivel en seco.
    out["driver_wet_skill_delta"] = wet_avg - dry_avg
    out["driver_n_wet_races"] = conditional_expanding_count_shifted(
        work, "driver_code", "race_date", "_was_wet"
    )

    out["driver_avg_finish_in_hot"] = hot_avg
    out["driver_hot_skill_delta"] = hot_avg - dry_avg

    team_wet = conditional_expanding_mean_shifted(work, "constructor_id", "race_date", "_finish", "_was_wet")
    team_dry = conditional_expanding_mean_shifted(work, "constructor_id", "race_date", "_finish", "_was_dry")
    out["team_avg_finish_in_wet"] = team_wet
    out["team_wet_skill_delta"] = team_wet - team_dry

    return out
