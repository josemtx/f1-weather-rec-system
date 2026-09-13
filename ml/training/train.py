"""Entrenamiento XGBoost: regresor de posicion final (anclado, por regimen)
+ clasificadores podium/points/dnf calibrados. Ver ml/docs/ARCHITECTURE.md.

Dos versiones con roles distintos salen de cada ejecucion:
  - evaluation : train 2023-24, val 2025. Sus metricas son creibles porque
                 nunca vio 2025. Con solo tres temporadas completas, un split
                 train/val/test de tres vias dejaba 440 filas de entrenamiento
                 y el clasificador de DNF no aprendia (AUC ~0.51), asi que 2025
                 se usa a la vez para calibrar y para reportar -- simplificacion
                 aceptada y documentada. El backtest walk-forward
                 (2023->2024, 2023+24->2025) compensa verificando que la senal
                 es estable entre folds.
  - production : misma receta con todo (2023-25). Predice mejor (conoce a los
                 pilotos recientes) pero no tiene metricas propias: hereda la
                 calibracion y la incertidumbre del modelo de evaluacion.

Receta del regresor (decidida el 2026-09-12 con residual_diagnosis.py sobre
ambos folds; el modelo que predecia posicion absoluta no superaba a la parrilla):
  - predice la DIFERENCIA respecto a un ancla (ml/training/anchor.py), con un
    regresor por regimen (post_quali / pre_quali);
  - se entrena SOLO con quienes terminaron: el abandono lo pone el simulador
    con su propio dado, y meterlo aqui sesgaba +1 puesto a todos los demas;
  - objetivo de error absoluto, que es la metrica real.
  Efecto medido (MAE finalizadores, media de folds): post_quali 3.01 -> 2.55
  (parrilla sola 2.74); pre_quali 3.47 -> 2.97 (ritmo reciente solo 2.95).

Uso: python -m ml.training.train [--tag NOMBRE] [--no-register]
  --tag          sufijo del nombre de version (evita pisar una existente)
  --no-register  guarda artefactos y metricas sin mover los punteros
                 evaluation/production del registry
"""

import argparse
import json
import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.inspection import permutation_importance
from sklearn.metrics import brier_score_loss, mean_absolute_error, roc_auc_score

from ml.common.logging_conf import configure
from ml.features.build_features import CATEGORICAL_COLS, FEATURES_PATH
from ml.training.anchor import (
    N_DRIVERS,
    REGIME_POST_QUALI,
    REGIME_PRE_QUALI,
    anchor_for,
    blank_quali_features,
)
from ml.training.model_registry import (
    MODELS_ROOT,
    apply_calibration,
    save_calibration,
    save_version_metadata,
    save_xgb_model,
    write_registry,
)

logger = logging.getLogger(__name__)

DROP_COLS = {
    "race_date", "driver_id", "driver_full_name", "constructor_name",
    "circuit_id", "country", "finish_position", "points", "status", "finished",
    # Metadatos de procedencia del clima, no features: en entrenamiento
    # climate_source es constante ("observed") y forecast_rain_probability
    # siempre NaN. Existen para el simulador y el dashboard.
    "climate_source", "forecast_rain_probability", "position_text",
}
TARGETS = ("podium", "points", "dnf")

# 2018-2022 esta en el parquet solo como calentamiento de las medias moviles
# de 2023 (el rolling mira hacia atras); nunca entra al modelo.
TRAIN_YEARS = [2023, 2024]
VAL_YEAR = 2025
WALK_FORWARD_FOLDS = [("2023->2024", [2023], 2024), ("2023+2024->2025", [2023, 2024], 2025)]

REGRESSOR_TARGET = "delta_from_anchor"
REGRESSOR_FILES = {
    REGIME_POST_QUALI: "finish_position_regressor",
    REGIME_PRE_QUALI: "finish_position_regressor_pre_quali",
}
REGRESSOR_METRIC_KEYS = {
    REGIME_POST_QUALI: "finish_position",
    REGIME_PRE_QUALI: "finish_position_pre_quali",
}


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["y_podium"] = (df["finish_position"] <= 3).astype(int)
    df["y_points"] = (df["points"] > 0).astype(int)
    df["y_dnf"] = (~df["finished"]).astype(int)
    df["y_finish_position"] = df["finish_position"].astype(float)
    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype("category")
    return df


def _feature_cols(df: pd.DataFrame) -> list[str]:
    exclude = DROP_COLS | {"y_podium", "y_points", "y_dnf", "y_finish_position"}
    return [c for c in df.columns if c not in exclude]


def fit_regressor(train: pd.DataFrame, feature_cols: list[str], regime: str, n_estimators: int = 300) -> xgb.XGBRegressor:
    finishers = train[train["y_dnf"] == 0]
    X = blank_quali_features(finishers[feature_cols], regime)
    y = finishers["y_finish_position"] - anchor_for(finishers, regime)
    reg = xgb.XGBRegressor(
        objective="reg:absoluteerror",
        n_estimators=n_estimators, max_depth=5, learning_rate=0.05,
        tree_method="hist", enable_categorical=True, random_state=42,
    )
    reg.fit(X, y)
    return reg


def fit_classifier(train: pd.DataFrame, feature_cols: list[str], target: str, n_estimators: int = 300) -> xgb.XGBClassifier:
    clf = xgb.XGBClassifier(
        n_estimators=n_estimators, max_depth=4, learning_rate=0.05,
        tree_method="hist", enable_categorical=True, eval_metric="logloss", random_state=42,
    )
    clf.fit(train[feature_cols], train[f"y_{target}"])
    return clf


def predict_position(reg: xgb.XGBRegressor, df: pd.DataFrame, feature_cols: list[str], regime: str) -> np.ndarray:
    X = blank_quali_features(df[feature_cols], regime)
    return np.clip(reg.predict(X) + anchor_for(df, regime).to_numpy(), 1, N_DRIVERS)


def regressor_metrics(val: pd.DataFrame, pred: np.ndarray, regime: str) -> dict:
    """MAE sobre finalizadores (lo que consume el simulador) y frente al ancla.

    La desviacion del error se mide SOLO sobre quienes terminaron: el error
    de un piloto que abandona desde 3o y queda clasificado 18o es enorme,
    pero esa varianza ya la modela el simulador tirando el dado del DNF
    aparte. Incluirla aqui seria contarla dos veces.
    """
    y = val["y_finish_position"].to_numpy()
    fin = val["y_dnf"].to_numpy() == 0
    anchor = anchor_for(val, regime).to_numpy()
    return {
        "mae_val_2025": float(mean_absolute_error(y[fin], pred[fin])),
        "mae_ancla_val_2025": float(mean_absolute_error(y[fin], anchor[fin])),
        "mae_val_2025_incluyendo_dnf": float(mean_absolute_error(y, pred)),
        "residual_std_val_2025": float(np.std(y[fin] - pred[fin])),
    }


def _auc(y_true: pd.Series, proba: np.ndarray) -> float | None:
    return float(roc_auc_score(y_true, proba)) if y_true.nunique() > 1 else None


def _fit_platt(clf: xgb.XGBClassifier, val: pd.DataFrame, feature_cols: list[str], target: str) -> tuple[float, float]:
    """Platt scaling (a, b) sobre val; se re-aplica en inferencia con apply_calibration."""
    calibrator = CalibratedClassifierCV(FrozenEstimator(clf), method="sigmoid")
    calibrator.fit(val[feature_cols], val[f"y_{target}"])
    cal = calibrator.calibrated_classifiers_[0].calibrators[0]
    return float(cal.a_), float(cal.b_)


def _classifier_metrics(clf, train, val, feature_cols, target, a, b) -> dict:
    y_train, y_val = train[f"y_{target}"], val[f"y_{target}"]
    raw = clf.predict_proba(val[feature_cols])[:, 1]
    return {
        "brier_raw_val": float(brier_score_loss(y_val, raw)),
        "brier_calibrated_val": float(brier_score_loss(y_val, apply_calibration(raw, a, b))),
        "auc_val": _auc(y_val, raw),
        "positive_rate_train": float(y_train.mean()),
        "positive_rate_val": float(y_val.mean()),
        "calibration_a": a,
        "calibration_b": b,
    }


def _walk_forward_backtest(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """Senal cruda (sin calibrar) en folds temporales independientes."""
    results = {}
    for label, train_years, test_year in WALK_FORWARD_FOLDS:
        train_fold, test_fold = df[df["year"].isin(train_years)], df[df["year"] == test_year]
        if train_fold.empty or test_fold.empty:
            continue
        m = {}
        for regime in (REGIME_POST_QUALI, REGIME_PRE_QUALI):
            reg = fit_regressor(train_fold, feature_cols, regime, n_estimators=200)
            rm = regressor_metrics(test_fold, predict_position(reg, test_fold, feature_cols, regime), regime)
            m[f"mae_{regime}"] = rm["mae_val_2025"]
            m[f"mae_ancla_{regime}"] = rm["mae_ancla_val_2025"]
        for target in TARGETS:
            clf = fit_classifier(train_fold, feature_cols, target, n_estimators=200)
            m[f"auc_{target}"] = _auc(test_fold[f"y_{target}"], clf.predict_proba(test_fold[feature_cols])[:, 1])
        logger.info(
            f"[walk-forward {label}] MAE post={m['mae_post_quali']:.3f} (ancla {m['mae_ancla_post_quali']:.3f}) "
            f"pre={m['mae_pre_quali']:.3f} (ancla {m['mae_ancla_pre_quali']:.3f}) "
            + " ".join(f"AUC_{t}={m[f'auc_{t}']:.3f}" for t in TARGETS if m[f"auc_{t}"] is not None)
        )
        results[label] = m
    return results


def _shap_summary(model, X: pd.DataFrame, top_n: int = 15) -> dict:
    values = np.abs(shap.TreeExplainer(model).shap_values(X)).mean(axis=0)
    return {name: float(v) for name, v in sorted(zip(X.columns, values), key=lambda kv: -kv[1])[:top_n]}


def _permutation_importance(model, X: pd.DataFrame, y: pd.Series, top_n: int = 15) -> dict:
    result = permutation_importance(model, X, y, n_repeats=5, random_state=42, scoring="neg_mean_absolute_error")
    return {name: float(v) for name, v in sorted(zip(X.columns, result.importances_mean), key=lambda kv: -kv[1])[:top_n]}


def _save_categories(train: pd.DataFrame, version_dir: Path) -> None:
    # Categorias vistas en ENTRENAMIENTO: un piloto debutante o un circuito
    # nuevo se mapea a NaN en inferencia (RaceSimulator._prep_categoricals)
    # en vez de hacer fallar a XGBoost, que rechaza categorias desconocidas.
    categories = {col: sorted(train[col].dropna().unique().tolist()) for col in CATEGORICAL_COLS}
    version_dir.mkdir(parents=True, exist_ok=True)
    (version_dir / "categories.json").write_text(json.dumps(categories, indent=2), encoding="utf-8")


def _train_evaluation(df: pd.DataFrame, feature_cols: list[str], version_name: str) -> dict:
    train, val = df[df["year"].isin(TRAIN_YEARS)], df[df["year"] == VAL_YEAR]
    assert train["year"].max() < val["year"].min(), "Guard de split temporal violado (train/val)"
    logger.info(f"=== Modelo de evaluacion: train={len(train)} ({TRAIN_YEARS}), val={len(val)} ({VAL_YEAR}) ===")
    version_dir = MODELS_ROOT / version_name
    metrics = {"version": version_name, "n_features": len(feature_cols), "regressor": {"target": REGRESSOR_TARGET}}

    for regime, file_name in REGRESSOR_FILES.items():
        reg = fit_regressor(train, feature_cols, regime)
        save_xgb_model(reg, version_dir, file_name)
        m = regressor_metrics(val, predict_position(reg, val, feature_cols, regime), regime)
        logger.info(f"[{regime}] MAE finalizadores val: {m['mae_val_2025']:.3f} (ancla sola {m['mae_ancla_val_2025']:.3f}) "
                    f"| desviacion del error {m['residual_std_val_2025']:.3f}")
        if regime == REGIME_POST_QUALI:
            val_fin = val[val["y_dnf"] == 0]
            y_delta = val_fin["y_finish_position"] - anchor_for(val_fin, regime)
            m["shap_top15"] = _shap_summary(reg, val_fin[feature_cols])
            m["permutation_importance_top15"] = _permutation_importance(reg, val_fin[feature_cols], y_delta)
        metrics[REGRESSOR_METRIC_KEYS[regime]] = m

    for target in TARGETS:
        clf = fit_classifier(train, feature_cols, target)
        a, b = _fit_platt(clf, val, feature_cols, target)
        save_xgb_model(clf, version_dir, f"{target}_classifier")
        save_calibration(a, b, version_dir, f"{target}_classifier")
        metrics[target] = _classifier_metrics(clf, train, val, feature_cols, target, a, b)
        logger.info(f"[{target}] Brier crudo={metrics[target]['brier_raw_val']:.4f} "
                    f"calibrado={metrics[target]['brier_calibrated_val']:.4f} AUC={metrics[target]['auc_val']}")

    _save_categories(train, version_dir)
    logger.info("=== Backtest walk-forward (verificacion de estabilidad, sin calibrar) ===")
    metrics["walk_forward_backtest"] = _walk_forward_backtest(df, feature_cols)
    return metrics


def _train_production(df: pd.DataFrame, feature_cols: list[str], eval_metrics: dict, version_name: str) -> dict:
    years = TRAIN_YEARS + [VAL_YEAR]
    train = df[df["year"].isin(years)]
    logger.info(f"=== Modelo de produccion: {len(train)} filas ({years}) ===")
    version_dir = MODELS_ROOT / version_name

    for regime, file_name in REGRESSOR_FILES.items():
        save_xgb_model(fit_regressor(train, feature_cols, regime), version_dir, file_name)
    for target in TARGETS:
        save_xgb_model(fit_classifier(train, feature_cols, target), version_dir, f"{target}_classifier")
        inherited = eval_metrics[target]
        save_calibration(inherited["calibration_a"], inherited["calibration_b"], version_dir, f"{target}_classifier")
    _save_categories(train, version_dir)

    post, pre = REGRESSOR_METRIC_KEYS[REGIME_POST_QUALI], REGRESSOR_METRIC_KEYS[REGIME_PRE_QUALI]
    residual_keys = ("residual_std_val_2025", "mae_val_2025", "mae_ancla_val_2025")
    return {
        "version": version_name,
        "rol": "production",
        "entrenado_con": years,
        "n_filas_entrenamiento": len(train),
        "calibracion_heredada_de": eval_metrics["version"],
        "regressor": {"target": REGRESSOR_TARGET},
        "nota": "Sin metricas propias por diseno: ha visto todos los datos. Su calidad estimada "
                "es la del modelo de evaluacion, misma receta dejando 2025 fuera.",
        "metricas_estimadas_del_modelo_de_evaluacion": {
            "mae": eval_metrics[post]["mae_val_2025"],
            "mae_ancla": eval_metrics[post]["mae_ancla_val_2025"],
            "mae_pre_quali": eval_metrics[pre]["mae_val_2025"],
            "residual_std": eval_metrics[post]["residual_std_val_2025"],
            **{f"auc_{t}": eval_metrics[t]["auc_val"] for t in TARGETS},
        },
        # El simulador lee residual_std de aqui, por regimen: la unica medida
        # honesta del error es la del modelo de evaluacion.
        post: {k: eval_metrics[post][k] for k in residual_keys},
        pre: {k: eval_metrics[pre][k] for k in residual_keys},
    }


def main(tag: str | None = None, register: bool = True) -> None:
    logger.info(f"Cargando {FEATURES_PATH}")
    raw = pd.read_parquet(FEATURES_PATH)
    df = _prepare(raw[raw["year"] <= VAL_YEAR])  # la temporada en curso nunca entra en metricas
    feature_cols = _feature_cols(df)
    logger.info(f"Features usadas: {len(feature_cols)}")

    prefix = date.today().isoformat() + (f"_{tag}" if tag else "")
    eval_name, prod_name = f"{prefix}_evaluation", f"{prefix}_production"

    eval_metrics = _train_evaluation(df, feature_cols, eval_name)
    prod_metrics = _train_production(df, feature_cols, eval_metrics, prod_name)

    if register:
        write_registry(eval_name, eval_metrics, feature_cols, role="evaluation")
        write_registry(prod_name, prod_metrics, feature_cols, role="production")
        logger.info(f"registry.json -> evaluation={eval_name}, production={prod_name}")
    else:
        save_version_metadata(eval_name, eval_metrics, feature_cols)
        save_version_metadata(prod_name, prod_metrics, feature_cols)
        logger.info(f"Guardados {eval_name} y {prod_name}; registry.json NO tocado (--no-register)")

    logger.info("Top SHAP del regresor post_quali (deltas frente a parrilla):")
    for name, value in list(eval_metrics[REGRESSOR_METRIC_KEYS[REGIME_POST_QUALI]]["shap_top15"].items())[:5]:
        logger.info(f"  {name}: {value:.4f}")


if __name__ == "__main__":
    configure()
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default=None)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()
    main(tag=args.tag, register=not args.no_register)
