"""Orquestador de feature engineering -> ml/data/processed/features_<version>.parquet

Une context (A) + driver_form (B) + team_form (C) + circuit_priors (D) +
climate (E) + tyre_strategy (F) + dynamic_pace (G) + qualifying (H) +
sprint_form (I) en una sola matriz, una fila por (year, round, driver_code).
F/G dependen de OpenF1 (laps/stints/pit), que solo cubre 2023+; I depende de
fines de semana con sprint (2021+) -- ambos quedan en NaN donde no aplican,
que XGBoost maneja nativamente (ver era_data_tier en context.py).

Uso: python -m ml.features.build_features
"""

import logging
from pathlib import Path

import pandas as pd

from ml.common.db import get_db
from ml.common.logging_conf import configure
from ml.features.feature_defs.circuit_priors import build_circuit_priors
from ml.features.feature_defs.climate import build_climate
from ml.features.feature_defs.context import build_context
from ml.features.feature_defs.driver_form import build_driver_form
from ml.features.feature_defs.dynamic_pace import build_dynamic_pace
from ml.features.feature_defs.qualifying import build_qualifying
from ml.features.feature_defs.sprint_form import build_sprint_form
from ml.features.feature_defs.sprint_pace_strategy import build_sprint_pace_strategy
from ml.features.feature_defs.team_form import build_team_form
from ml.features.feature_defs.tyre_strategy import build_tyre_strategy
from ml.features.feature_defs.weather_interaction import build_weather_interaction
from ml.features.upcoming import build_stub_rows

logger = logging.getLogger(__name__)

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
VERSION = "v3"


def build_feature_matrix(db, upcoming: tuple[int, int] | None = None) -> pd.DataFrame:
    logger.info("Construyendo features de contexto (A)...")
    context_df = build_context(db)
    if context_df.empty:
        raise RuntimeError("historical_results esta vacia; ejecuta ml.jolpica.ingest_jolpica primero.")

    if upcoming is not None:
        # Las filas fantasma se anaden ANTES de calcular nada: asi todas las
        # categorias de features se calculan igual para ellas que para una
        # carrera real, y el shift(1) del anti-fuga las deja ver solo el
        # pasado. Ver ml/features/upcoming.py.
        year, round_number = upcoming
        stub = build_stub_rows(db, year, round_number)
        context_df = pd.concat([context_df, stub], ignore_index=True)
        logger.info(f"Anadidas {len(stub)} filas fantasma para {year} R{round_number}")

    logger.info(f"Contexto: {len(context_df)} filas (carrera x piloto)")

    logger.info("Construyendo forma de piloto (B)...")
    driver_df = build_driver_form(context_df)

    logger.info("Construyendo forma de equipo (C)...")
    team_df = build_team_form(context_df)

    logger.info("Construyendo priors de circuito (D)...")
    circuit_df = build_circuit_priors(db, context_df)

    logger.info("Construyendo clima (E)...")
    climate_df = build_climate(db, context_df)

    logger.info("Construyendo estrategia de neumaticos (F)...")
    tyre_df = build_tyre_strategy(db, context_df)

    logger.info("Construyendo ritmo dinamico (G)...")
    pace_df = build_dynamic_pace(db, context_df)

    logger.info("Construyendo clasificacion (H)...")
    quali_df = build_qualifying(db, context_df)

    logger.info("Construyendo forma en sprints (I)...")
    sprint_df = build_sprint_form(db, context_df)
    sprint_pace_df = build_sprint_pace_strategy(db, context_df)

    logger.info("Construyendo interaccion piloto x clima (J)...")
    weather_int_df = build_weather_interaction(context_df, climate_df)

    # Delta de condiciones quali->carrera: necesita clima de ambas sesiones a
    # la vez, se calcula aqui en vez de dentro de una sola categoria.
    quali_df["conditions_delta_quali_to_race"] = (
        climate_df["race_day_temp_avg"] - quali_df["quali_day_temp_avg"]
    )
    quali_df["conditions_rain_changed"] = (
        climate_df["climate_rain_probability_flag"] != quali_df["quali_day_rain_flag"]
    ).astype(float)

    features = pd.concat(
        [context_df, driver_df, team_df, circuit_df, climate_df, tyre_df, pace_df, quali_df, sprint_df,
         sprint_pace_df, weather_int_df],
        axis=1,
    )

    if "unmatched_circuit_ids" in context_df.attrs:
        logger.warning(
            f"circuit_id sin match en circuits.json (features de circuito en NaN para esas filas): "
            f"{context_df.attrs['unmatched_circuit_ids']}"
        )

    return features


def save_feature_matrix(features: pd.DataFrame) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"features_{VERSION}.parquet"
    features.to_parquet(out_path, index=False)
    logger.info(f"Guardado {out_path} ({len(features)} filas, {len(features.columns)} columnas)")
    return out_path


def main():
    db = get_db()
    features = build_feature_matrix(db)
    save_feature_matrix(features)

    logger.info("Resumen por anio:")
    logger.info(features.groupby("year").size().to_string())

    null_rates = features.isna().mean().sort_values(ascending=False)
    logger.info("Top 15 columnas con mas NaN (esperado en F/G hasta que OpenF1 este poblado):")
    logger.info(null_rates.head(15).to_string())


if __name__ == "__main__":
    configure()
    main()
