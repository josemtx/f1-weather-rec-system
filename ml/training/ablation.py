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
conto con ella). El regresor es el anclado de train.py: post-quali frente a
la parrilla, los otros dos frente a la forma reciente, y se reporta el MAE
del ancla sola para saber cuanto aporta el modelo sobre ella.

Uso: python -m ml.training.ablation [ruta_salida.json]
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ml.common.logging_conf import configure
from ml.features.build_features import FEATURES_PATH
from ml.training.anchor import QUALI_FEATURES, REGIME_POST_QUALI, REGIME_PRE_QUALI
from ml.training.train import (
    TARGETS,
    TRAIN_YEARS,
    VAL_YEAR,
    _auc,
    _feature_cols,
    _prepare,
    fit_classifier,
    fit_regressor,
    predict_position,
    regressor_metrics,
)

logger = logging.getLogger(__name__)

OUT_PATH = Path(__file__).resolve().parents[1] / "models" / "ablation_information_regimes.json"

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

# (regimen del regresor anclado, features ocultas ademas de las del regimen)
REGIMES = {
    "post_quali": (REGIME_POST_QUALI, []),
    "pre_quali": (REGIME_PRE_QUALI, QUALI_FEATURES),
    "solo_forma": (REGIME_PRE_QUALI, QUALI_FEATURES + CLIMATE_FEATURES),
}


def _evaluate_regime(train: pd.DataFrame, val: pd.DataFrame, feature_cols: list[str], regime: str, blanked: list[str]) -> dict:
    """Entrena y evalua ocultando (NaN) las features de `blanked`."""
    to_blank = [c for c in blanked if c in feature_cols]
    train_b, val_b = train.copy(), val.copy()
    for col in to_blank:
        train_b[col] = np.nan
        val_b[col] = np.nan

    reg = fit_regressor(train_b, feature_cols, regime)
    pred = predict_position(reg, val_b, feature_cols, regime)
    m = regressor_metrics(val_b, pred, regime)

    result = {
        "features_ocultas": len(to_blank),
        "mae": m["mae_val_2025"],
        "mae_ancla": m["mae_ancla_val_2025"],
        "mae_incluyendo_dnf": m["mae_val_2025_incluyendo_dnf"],
        "residual_std": m["residual_std_val_2025"],
    }

    for target in TARGETS:
        clf = fit_classifier(train_b, feature_cols, target)
        result[f"auc_{target}"] = _auc(val_b[f"y_{target}"], clf.predict_proba(val_b[feature_cols])[:, 1])

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


def main(out_path: Path = OUT_PATH):
    raw = pd.read_parquet(FEATURES_PATH)
    df = _prepare(raw[raw["year"] <= VAL_YEAR])
    feature_cols = _feature_cols(df)

    train = df[df["year"].isin(TRAIN_YEARS)]
    val = df[df["year"] == VAL_YEAR]
    logger.info(f"Ablacion sobre {len(val)} filas de {VAL_YEAR} (entrenando con {len(train)})")

    results = {}
    for name, (regime, blanked) in REGIMES.items():
        results[name] = _evaluate_regime(train, val, feature_cols, regime, blanked)
        r = results[name]
        logger.info(
            f"[{name:11s}] MAE={r['mae']:.3f} (ancla {r['mae_ancla']:.3f})  podio_AUC={r['auc_podium']:.3f}  "
            f"puntos_AUC={r['auc_points']:.3f}  aciertos_podio={r['aciertos_podio_por_carrera']}/3"
        )

    base = results["post_quali"]["mae"]
    logger.info("=== Coste de no tener clasificacion (MAE finalizadores) ===")
    for name, r in results.items():
        logger.info(f"  {name:11s}: MAE {r['mae']:.3f}  ({r['mae'] - base:+.3f} vs post-quali)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info(f"Guardado en {out_path}")


if __name__ == "__main__":
    configure()
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else OUT_PATH)
