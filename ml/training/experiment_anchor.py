"""Experimento: ¿como conseguir que el regresor supere a la parrilla?

residual_diagnosis.py mostro que el modelo post-clasificacion no mejora a
"predecir = parrilla" (2024: 3.47 vs 2.91; 2025: 3.36 vs 3.34), y que el
pre-clasificacion vale lo que la media movil del piloto. Tres cambios
baratos, cruzados entre si, sobre ambos folds walk-forward y ambos regimenes:

  anchor    : predecir la DIFERENCIA respecto a un ancla (parrilla post-quali,
              forma reciente pre-quali) en vez de la posicion absoluta. El
              suelo del modelo pasa a ser el ancla.
  finishers : entrenar solo con quienes terminaron. El DNF lo pone el
              simulador con su propio dado; meterlo aqui sesga +1 puesto a todos.
  objective : error absoluto (la metrica real) en vez de cuadratico.

La metrica principal es el MAE sobre finalizadores, que es lo que el
simulador usa; se reporta tambien sobre todas las filas por comparabilidad.

Uso: python -m ml.training.experiment_anchor
"""

import itertools
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

from ml.common.logging_conf import configure
from ml.training.anchor import QUALI_FEATURES, anchor_for
from ml.training.train import FEATURES_PATH, VAL_YEAR, _feature_cols, _prepare

logger = logging.getLogger(__name__)

OUT_PATH = Path(__file__).resolve().parents[1] / "models" / "experiment_anchor.json"

FOLDS = [("2023->2024", [2023], 2024), ("2023+24->2025", [2023, 2024], 2025)]
REGIMES = {"post_quali": [], "pre_quali": QUALI_FEATURES}
N_DRIVERS = 20


def _fit_predict(train, val, feature_cols, blanked, anchor, finishers, objective) -> np.ndarray:
    train_x, val_x = train[feature_cols].copy(), val[feature_cols].copy()
    for col in blanked:
        if col in feature_cols:
            train_x[col] = np.nan
            val_x[col] = np.nan

    y_train = train["y_finish_position"].copy()
    if anchor is not None:
        y_train = y_train - anchor.loc[train.index]
    if finishers:
        keep = train["y_dnf"] == 0
        train_x, y_train = train_x[keep], y_train[keep]

    reg = xgb.XGBRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        tree_method="hist", enable_categorical=True, random_state=42,
        objective=f"reg:{objective}",
    )
    reg.fit(train_x, y_train)
    pred = reg.predict(val_x)
    if anchor is not None:
        pred = pred + anchor.loc[val.index].to_numpy()
    return np.clip(pred, 1, N_DRIVERS)


def _scores(val: pd.DataFrame, pred: np.ndarray) -> dict:
    fin = (val["y_dnf"] == 0).to_numpy()
    y = val["y_finish_position"].to_numpy()
    return {
        "mae_finalizadores": float(mean_absolute_error(y[fin], pred[fin])),
        "mae_todas": float(mean_absolute_error(y, pred)),
    }


def main():
    raw = pd.read_parquet(FEATURES_PATH)
    df = _prepare(raw[raw["year"] <= VAL_YEAR])
    feature_cols = _feature_cols(df)
    anchors = {r: anchor_for(df, r) for r in REGIMES}

    variants = list(itertools.product([False, True], [False, True], ["squarederror", "absoluteerror"]))
    results = {}

    for regime, blanked in REGIMES.items():
        results[regime] = {}
        for fold_name, train_years, test_year in FOLDS:
            train = df[df["year"].isin(train_years)]
            val = df[df["year"] == test_year]
            fold = {"ancla_sola": _scores(val, anchors[regime].loc[val.index].to_numpy())}
            for use_anchor, finishers, objective in variants:
                name = f"{'anchor' if use_anchor else 'abs'}|{'fin' if finishers else 'all'}|{objective[:3]}"
                pred = _fit_predict(
                    train, val, feature_cols, blanked,
                    anchors[regime] if use_anchor else None, finishers, objective,
                )
                fold[name] = _scores(val, pred)
            results[regime][fold_name] = fold

    _log(results)
    OUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info(f"Guardado en {OUT_PATH}")


def _log(results: dict):
    for regime, folds in results.items():
        logger.info(f"=== {regime} ===  (MAE finalizadores / MAE todas)")
        names = list(next(iter(folds.values())).keys())
        header = f"{'variante':22s}" + "".join(f"{f:>22s}" for f in folds)
        logger.info(header)
        for name in names:
            row = f"{name:22s}"
            for fold in folds.values():
                s = fold[name]
                row += f"{s['mae_finalizadores']:>11.3f} /{s['mae_todas']:>8.3f}"
            logger.info(row)
        # Media entre folds sobre finalizadores, ordenada
        avg = {n: np.mean([f[n]["mae_finalizadores"] for f in folds.values()]) for n in names}
        best = sorted(avg.items(), key=lambda kv: kv[1])[:4]
        logger.info("  Mejores por media de MAE finalizadores: " + ", ".join(f"{n}={v:.3f}" for n, v in best))


if __name__ == "__main__":
    configure()
    main()
