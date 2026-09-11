"""Cuanto vale cada bloque de informacion: ablacion por regimen de informacion.

Responde a la pregunta practica: "¿cuanto peor predigo la carrera de manana
si aun no se ha corrido la clasificacion?". Se mide sobre 2025, donde SI
conocemos el resultado, borrando features para imitar cada momento del fin
de semana:

  post-quali  -> todo disponible (sabado noche). Es el backtest habitual.
  pre-quali   -> sin parrilla ni tiempos de clasificacion (viernes)
  solo-forma  -> ademas sin clima, para aislar cuanto aporta el clima

El modelo se REENTRENA en cada regimen: si en produccion no vas a tener
clasificacion, tampoco debes entrenar con ella (un modelo que aprendio a
apoyarse en la pole y luego no la recibe rinde peor que uno que nunca
conto con ella).

Uso: python -m ml.training.ablation
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, roc_auc_score

from ml.common.logging_conf import configure
from ml.training.train import (
    CATEGORICAL_COLS,
    FEATURES_PATH,
    TRAIN_YEARS,
    VAL_YEAR,
    _feature_cols,
    _prepare,
)

logger = logging.getLogger(__name__)

OUT_PATH = Path(__file__).resolve().parents[1] / "models" / "ablation_information_regimes.json"

# Features que solo existen una vez disputada la clasificacion del sabado.
QUALI_FEATURES = [
    "grid_position", "driver_quali_position", "quali_gap_to_pole_pct",
    "grid_penalty_positions", "n_quali_laps", "quali_soft_tyre_share",
    "quali_day_temp_avg", "quali_day_precipitation_mm", "quali_day_rain_flag",
    "conditions_delta_quali_to_race", "conditions_rain_changed",
]

CLIMATE_FEATURES = [
    # Clima directo de la carrera (categoria E)
    "race_day_temp_avg", "race_day_temp_max", "race_day_precipitation_mm",
    "race_day_humidity", "race_day_wind_speed", "climate_rain_probability_flag",
    "circuit_seasonal_climate_norm_temp", "climate_temp_delta_vs_norm",
    "race_day_track_temp", "race_day_wet_fraction",
    # Interaccion piloto x clima (categoria J) -- si no se ocultan tambien,
    # "sin clima" seguiria teniendo informacion climatica y la ablacion
    # mediria mal.
    "driver_avg_finish_in_wet", "driver_avg_finish_in_dry", "driver_wet_skill_delta",
    "driver_n_wet_races", "driver_avg_finish_in_hot", "driver_hot_skill_delta",
    "team_avg_finish_in_wet", "team_wet_skill_delta",
]

REGIMES = {
    "post_quali": [],
    "pre_quali": QUALI_FEATURES,
    "solo_forma": QUALI_FEATURES + CLIMATE_FEATURES,
}


def _evaluate_regime(train: pd.DataFrame, val: pd.DataFrame, feature_cols: list[str], blanked: list[str]) -> dict:
    """Entrena y evalua ocultando (NaN) las features de `blanked`."""
    to_blank = [c for c in blanked if c in feature_cols]
    train_x, val_x = train[feature_cols].copy(), val[feature_cols].copy()
    for col in to_blank:
        train_x[col] = np.nan
        val_x[col] = np.nan

    reg = xgb.XGBRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        tree_method="hist", enable_categorical=True, random_state=42,
    )
    reg.fit(train_x, train["y_finish_position"])
    pred = reg.predict(val_x)

    result = {
        "features_ocultas": len(to_blank),
        "mae": float(mean_absolute_error(val["y_finish_position"], pred)),
        "residual_std": float(np.std(val["y_finish_position"].to_numpy() - pred)),
    }

    for target in ["podium", "points", "dnf"]:
        clf = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            tree_method="hist", enable_categorical=True, random_state=42,
        )
        clf.fit(train_x, train[f"y_{target}"])
        proba = clf.predict_proba(val_x)[:, 1]
        y_true = val[f"y_{target}"]
        result[f"auc_{target}"] = float(roc_auc_score(y_true, proba)) if y_true.nunique() > 1 else None

    # Aciertos de podio: de los 3 primeros predichos por carrera, cuantos
    # estuvieron de verdad en el podio. Mas legible que el AUC.
    val_eval = val[["year", "round", "y_finish_position"]].copy()
    val_eval["pred"] = pred
    hits, races = 0, 0
    for _, grp in val_eval.groupby(["year", "round"]):
        pred_top3 = set(grp.nsmallest(3, "pred").index)
        real_top3 = set(grp.nsmallest(3, "y_finish_position").index)
        hits += len(pred_top3 & real_top3)
        races += 1
    result["aciertos_podio_por_carrera"] = round(hits / races, 3) if races else None

    return result


def main():
    raw = pd.read_parquet(FEATURES_PATH)
    df = _prepare(raw[raw["year"] <= VAL_YEAR])
    feature_cols = _feature_cols(df)

    train = df[df["year"].isin(TRAIN_YEARS)]
    val = df[df["year"] == VAL_YEAR]
    logger.info(f"Ablacion sobre {len(val)} filas de {VAL_YEAR} (entrenando con {len(train)})")

    results = {}
    for name, blanked in REGIMES.items():
        results[name] = _evaluate_regime(train, val, feature_cols, blanked)
        r = results[name]
        logger.info(
            f"[{name:11s}] MAE={r['mae']:.3f}  podio_AUC={r['auc_podium']:.3f}  "
            f"puntos_AUC={r['auc_points']:.3f}  aciertos_podio={r['aciertos_podio_por_carrera']}/3"
        )

    base = results["post_quali"]["mae"]
    logger.info("=== Coste de no tener clasificacion ===")
    for name, r in results.items():
        logger.info(f"  {name:11s}: MAE {r['mae']:.3f}  ({r['mae'] - base:+.3f} vs post-quali)")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info(f"Guardado en {OUT_PATH}")


if __name__ == "__main__":
    configure()
    main()
