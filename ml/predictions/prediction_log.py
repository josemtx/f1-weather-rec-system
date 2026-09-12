"""Registro persistente de predicciones: que dijimos, cuando y sabiendo que.

Sin esto el sistema solo sabe responder "esto predigo ahora". Con esto puede
responder "esto predije, esto paso, y asi de bien acierto", que es lo unico
que convierte un track record en algo comprobable por un tercero.

El resultado real NO se copia aqui: se une desde historical_results al
leerlo. Guardar una copia del desenlace solo crearia una segunda verdad que
puede quedarse desactualizada.
"""

import logging

import pandas as pd
import pymongo

logger = logging.getLogger(__name__)

COLLECTION = "predictions"

# Que sabiamos en el momento de predecir. Se deduce de los datos existentes,
# no se pasa a mano: asi no se puede etiquetar mal una prediccion.
REGIME_PRE_QUALI = "pre_quali"
REGIME_POST_SPRINT = "post_sprint"
REGIME_POST_QUALI = "post_quali"


def ensure_indexes(db) -> None:
    db[COLLECTION].create_index(
        [
            ("year", pymongo.ASCENDING),
            ("round", pymongo.ASCENDING),
            ("driver_code", pymongo.ASCENDING),
            ("information_regime", pymongo.ASCENDING),
        ],
        unique=True,
    )


def detect_information_regime(db, year: int, round_number: int) -> str:
    """Que informacion del fin de semana existe ya.

    La clasificacion es sabado y el sprint viernes/sabado: ambos ocurren
    antes de la carrera, asi que usarlos no es trampa. Lo que define el
    regimen es simplemente hasta donde ha llegado el fin de semana.
    """
    key = {"year": year, "round": round_number}
    if db["qualifying_results"].count_documents(key) > 0:
        return REGIME_POST_QUALI
    if db["sprint_results"].count_documents(key) > 0:
        return REGIME_POST_SPRINT
    return REGIME_PRE_QUALI


def race_already_run(db, year: int, round_number: int) -> bool:
    return db["historical_results"].count_documents({"year": year, "round": round_number}) > 0


def save_predictions(
    db,
    year: int,
    round_number: int,
    circuit_short_name: str,
    predictions: pd.DataFrame,
    model_version: str,
    climate_source: str,
    n_simulations: int,
    seed: int,
    allow_after_race: bool = False,
) -> int:
    """Guarda una prediccion por piloto. Idempotente por (carrera, piloto, regimen).

    Se niega a escribir si la carrera ya se disputo: una prediccion generada
    despues de la carrera usaria clima observado en vez de pronostico y
    parrilla real en vez de estimada. Seguiria sin haber fuga temporal, pero
    ya no seria lo que de verdad predijimos, y el track record dejaria de
    significar nada. `allow_after_race` existe solo para rehacer el historico
    a proposito, dejando constancia.
    """
    if race_already_run(db, year, round_number) and not allow_after_race:
        raise ValueError(
            f"{year} R{round_number} ya tiene resultados: registrar una prediccion ahora "
            f"falsearia el track record. Usa allow_after_race=True si es a proposito."
        )

    ensure_indexes(db)
    regime = detect_information_regime(db, year, round_number)
    now = pd.Timestamp.now("UTC").isoformat()

    col = db[COLLECTION]
    saved = 0
    for _, row in predictions.iterrows():
        doc = {
            "year": year,
            "round": round_number,
            "circuit_short_name": circuit_short_name,
            "driver_code": row["driver_code"],
            "information_regime": regime,
            "predicted_at": now,
            "model_version": model_version,
            "climate_source": climate_source,
            "n_simulations": n_simulations,
            "seed": seed,
            "p_win": float(row["p_win"]),
            "p_podium": float(row["p_podium"]),
            "p_points": float(row["p_points"]),
            "p_dnf": float(row["p_dnf"]),
            "mean_finish_position": float(row["mean_finish_position"]),
            "median_finish_position": float(row["median_finish_position"]),
            "p10_finish_position": float(row["p10_finish_position"]),
            "p90_finish_position": float(row["p90_finish_position"]),
        }
        col.replace_one(
            {
                "year": year, "round": round_number,
                "driver_code": doc["driver_code"], "information_regime": regime,
            },
            doc, upsert=True,
        )
        saved += 1

    logger.info(f"Guardadas {saved} predicciones de {year} R{round_number} (regimen: {regime})")
    return saved


def load_predictions_with_outcome(db, year: int | None = None) -> pd.DataFrame:
    """Predicciones unidas con lo que realmente paso.

    Las carreras aun no disputadas salen con resultado NaN, que es lo
    correcto: siguen siendo predicciones pendientes de resolver.
    """
    query = {"year": year} if year else {}
    preds = list(db[COLLECTION].find(query, {"_id": 0}))
    if not preds:
        return pd.DataFrame()

    df = pd.DataFrame(preds)

    results = list(db["historical_results"].find(
        query, {"_id": 0, "year": 1, "round": 1, "driver_code": 1,
                "finish_position": 1, "grid_position": 1, "finished": 1}
    ))
    if results:
        outcome = pd.DataFrame(results).rename(columns={
            "finish_position": "actual_finish_position",
            "grid_position": "actual_grid_position",
            "finished": "actual_finished",
        })
        df = df.merge(outcome, on=["year", "round", "driver_code"], how="left")
    else:
        df["actual_finish_position"] = pd.NA
        df["actual_grid_position"] = pd.NA
        df["actual_finished"] = pd.NA

    return df
