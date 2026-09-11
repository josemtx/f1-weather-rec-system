"""Entrenamiento XGBoost, modelo B (2023-2026, ver ml/docs/ARCHITECTURE.md):
regresor de posicion final + clasificadores podium/points/dnf.

Con solo 3 temporadas completas en el rango 2023-2025 (2026 en curso, excluido
de metricas), un split train/val/test de 3 vias dejaba apenas 440 filas de
entrenamiento (1 temporada) -- insuficiente para que el clasificador de DNF
aprendiera (AUC cayo a ~0.51, practicamente azar; ver decision 2026-09-11).

Se opto por train=2023+2024 (919 filas), val=2025 usado a la vez para
calibracion Y como metrica reportada (simplificacion aceptada por falta de
una 4a temporada completa -- documentado, no oculto). Para no perder rigor
por no tener un test set separado, se anade un backtest walk-forward
(_walk_forward_backtest): fold1 train=2023/test=2024, fold2
train=2023+2024/test=2025, sin calibrar, solo para verificar que el poder
predictivo (MAE/AUC) es estable entre folds y no un artefacto de una sola
particion.

Uso: python -m ml.training.train
"""

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
from ml.training.model_registry import (
    MODELS_ROOT,
    apply_calibration,
    save_calibration,
    save_xgb_model,
    write_registry,
)

logger = logging.getLogger(__name__)

FEATURES_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "features_v3.parquet"

DROP_COLS = {
    "race_date", "driver_id", "driver_full_name", "constructor_name",
    "circuit_id", "country", "finish_position", "points", "status", "finished",
    # Metadatos de procedencia del clima, no features: en entrenamiento
    # climate_source es constante ("observed") y forecast_rain_probability
    # siempre NaN. Existen para el simulador y para poder mostrar en el
    # dashboard si una prediccion salio de datos reales o de un pronostico.
    "climate_source", "forecast_rain_probability", "position_text",
}
CATEGORICAL_COLS = ["circuit_short_name", "circuit_type", "overtaking_difficulty", "era_data_tier", "driver_code", "constructor_id"]

# 2018-2022 se mantiene en el parquet como "calentamiento" de las medias
# moviles de las primeras filas de 2023 (ya resuelto en build_features.py,
# el rolling ve hacia atras aunque esas filas no entren al modelo), pero
# nunca es year seleccionado aqui.
TRAIN_YEARS = [2023, 2024]
VAL_YEAR = 2025


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


def _temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df["year"].isin(TRAIN_YEARS)]
    val = df[df["year"] == VAL_YEAR]

    assert train["year"].max() < val["year"].min(), "Guard de split temporal violado (train/val)"

    logger.info(f"Split temporal: train={len(train)} (2023+2024), val={len(val)} (2025)")
    return train, val


def _train_regressor(train, val, feature_cols):
    model = xgb.XGBRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        tree_method="hist", enable_categorical=True,
        early_stopping_rounds=20, eval_metric="mae",
        random_state=42,
    )
    model.fit(
        train[feature_cols], train["y_finish_position"],
        eval_set=[(val[feature_cols], val["y_finish_position"])],
        verbose=False,
    )
    val_pred = model.predict(val[feature_cols])
    mae = mean_absolute_error(val["y_finish_position"], val_pred)

    # Desviacion tipica del error de prediccion: la incertidumbre del PROPIO
    # MODELO, distinta de la del mundo. Sin ella el Monte Carlo simula un
    # universo donde el modelo acierta siempre y da probabilidades absurdas.
    #
    # Se mide SOLO sobre quienes terminaron: el error de un piloto que
    # abandona desde 3o y queda clasificado 18o es enorme, pero esa varianza
    # ya la modela el simulador tirando el dado del DNF aparte. Incluirla
    # aqui seria contarla dos veces y ensanchar de mas la distribucion.
    residuals_all = val["y_finish_position"].to_numpy() - val_pred
    finished_mask = val["y_dnf"].to_numpy() == 0
    residual_std_finishers = float(np.std(residuals_all[finished_mask]))

    logger.info(
        f"[finish_position] MAE en val (2025): {mae:.3f} posiciones | "
        f"desviacion del error: {residual_std_finishers:.3f} (solo finalizadores) "
        f"vs {float(np.std(residuals_all)):.3f} (todas, infla por DNF)"
    )
    return model, {
        "mae_val_2025": mae,
        "residual_std_val_2025": residual_std_finishers,
        "residual_std_incluyendo_dnf": float(np.std(residuals_all)),
        "best_iteration": int(model.best_iteration),
    }


def _train_calibrated_classifier(target_name: str, train, val, feature_cols):
    y_train = train[f"y_{target_name}"]
    y_val = val[f"y_{target_name}"]

    base = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        tree_method="hist", enable_categorical=True,
        eval_metric="logloss",
        random_state=42,
    )
    base.fit(train[feature_cols], y_train)

    raw_val_proba = base.predict_proba(val[feature_cols])[:, 1]
    calibrator = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid")
    calibrator.fit(val[feature_cols], y_val)
    cal = calibrator.calibrated_classifiers_[0].calibrators[0]
    a, b = float(cal.a_), float(cal.b_)

    calibrated_val_proba = apply_calibration(raw_val_proba, a, b)

    # NOTA: val (2025) se usa a la vez para ajustar la calibracion y para
    # reportar metricas -- simplificacion aceptada por no tener una 4a
    # temporada completa para un test separado (ver docstring del modulo).
    # El backtest walk-forward (_walk_forward_backtest) compensa esto
    # verificando estabilidad entre folds independientes.
    metrics = {
        "brier_raw_val": float(brier_score_loss(y_val, raw_val_proba)),
        "brier_calibrated_val": float(brier_score_loss(y_val, calibrated_val_proba)),
        "auc_val": float(roc_auc_score(y_val, raw_val_proba)) if y_val.nunique() > 1 else None,
        "positive_rate_train": float(y_train.mean()),
        "positive_rate_val": float(y_val.mean()),
        "calibration_a": a,
        "calibration_b": b,
    }
    logger.info(
        f"[{target_name}] Brier crudo={metrics['brier_raw_val']:.4f} "
        f"calibrado={metrics['brier_calibrated_val']:.4f} AUC={metrics['auc_val']}"
    )
    return base, a, b, metrics


def _walk_forward_backtest(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """Verifica que el poder predictivo es estable entre folds temporales
    independientes, sin calibrar (solo senal cruda) -- compensa no tener un
    test set separado del modelo principal."""
    folds = [
        ("2023->2024", [2023], 2024),
        ("2023+2024->2025", [2023, 2024], 2025),
    ]
    results = {}
    for label, train_years, test_year in folds:
        train_fold = df[df["year"].isin(train_years)]
        test_fold = df[df["year"] == test_year]
        if train_fold.empty or test_fold.empty:
            continue

        reg = xgb.XGBRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            tree_method="hist", enable_categorical=True, random_state=42,
        )
        reg.fit(train_fold[feature_cols], train_fold["y_finish_position"])
        mae = mean_absolute_error(test_fold["y_finish_position"], reg.predict(test_fold[feature_cols]))

        fold_metrics = {"mae_finish_position": float(mae)}
        for target in ["podium", "points", "dnf"]:
            y_train = train_fold[f"y_{target}"]
            y_test = test_fold[f"y_{target}"]
            clf = xgb.XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                tree_method="hist", enable_categorical=True, random_state=42,
            )
            clf.fit(train_fold[feature_cols], y_train)
            proba = clf.predict_proba(test_fold[feature_cols])[:, 1]
            fold_metrics[f"auc_{target}"] = float(roc_auc_score(y_test, proba)) if y_test.nunique() > 1 else None

        logger.info(f"[walk-forward {label}] MAE={mae:.3f} " + " ".join(
            f"AUC_{k.split('_')[1]}={v:.3f}" for k, v in fold_metrics.items() if k.startswith("auc_") and v is not None
        ))
        results[label] = fold_metrics

    return results


def _shap_summary(model, X_sample: pd.DataFrame, top_n: int = 15) -> dict:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    mean_abs = np.abs(shap_values).mean(axis=0)
    ranked = sorted(zip(X_sample.columns, mean_abs), key=lambda x: -x[1])[:top_n]
    return {name: float(val) for name, val in ranked}


def _permutation_importance_check(model, X_val, y_val, top_n: int = 15) -> dict:
    # El modelo necesita TODAS las columnas de entrenamiento (incluidas las
    # categoricas) para predecir; permutation_importance las baraja una a una
    # y reporta el impacto en el error, categoricas incluidas.
    result = permutation_importance(
        model, X_val, y_val, n_repeats=5, random_state=42, scoring="neg_mean_absolute_error"
    )
    ranked = sorted(zip(X_val.columns, result.importances_mean), key=lambda x: -x[1])[:top_n]
    return {name: float(val) for name, val in ranked}


def _train_production_model(df: pd.DataFrame, feature_cols: list[str], eval_metrics: dict, eval_categories: dict):
    """Misma receta, entrenada con TODO lo disponible (2023-2025).

    El modelo de evaluacion deja 2025 fuera para que sus metricas sean
    creibles. Pero para predecir de verdad interesa lo contrario: cuanto mas
    reciente sea lo que ha visto, mejor. Entrenar tambien con 2025 hace, por
    ejemplo, que conozca a Antonelli y Bortoleto, que hoy caen a NaN por ser
    categorias no vistas.

    La calibracion (a, b) se hereda del modelo de evaluacion: se ajusto sobre
    2025 con la misma receta, y aqui no queda ningun conjunto sin ver sobre
    el que ajustarla honestamente. Queda documentado en metrics.json.
    """
    years = list(TRAIN_YEARS) + [VAL_YEAR]
    train_all = df[df["year"].isin(years)]
    logger.info(f"=== Modelo de produccion: entrenando con {len(train_all)} filas ({years}) ===")

    version_name = f"{date.today().isoformat()}_production"
    version_dir = MODELS_ROOT / version_name
    version_dir.mkdir(parents=True, exist_ok=True)

    reg = xgb.XGBRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        tree_method="hist", enable_categorical=True, random_state=42,
    )
    reg.fit(train_all[feature_cols], train_all["y_finish_position"])
    save_xgb_model(reg, version_dir, "finish_position_regressor")

    for target in ["podium", "points", "dnf"]:
        clf = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            tree_method="hist", enable_categorical=True, random_state=42,
            eval_metric="logloss",
        )
        clf.fit(train_all[feature_cols], train_all[f"y_{target}"])
        save_xgb_model(clf, version_dir, f"{target}_classifier")

        inherited = eval_metrics.get(target, {})
        save_calibration(
            inherited.get("calibration_a", 0.0), inherited.get("calibration_b", 0.0),
            version_dir, f"{target}_classifier",
        )

    categories = {col: sorted(train_all[col].dropna().unique().tolist()) for col in CATEGORICAL_COLS}
    (version_dir / "categories.json").write_text(json.dumps(categories, indent=2), encoding="utf-8")

    prod_metrics = {
        "version": version_name,
        "rol": "production",
        "entrenado_con": years,
        "n_filas_entrenamiento": len(train_all),
        "calibracion_heredada_de": eval_metrics.get("version"),
        "nota": (
            "Este modelo no tiene metricas propias por diseno: ha visto todos los datos "
            "disponibles. Su calidad estimada es la del modelo de evaluacion, que uso la "
            "misma receta dejando 2025 fuera."
        ),
        "metricas_estimadas_del_modelo_de_evaluacion": {
            "mae": eval_metrics.get("finish_position", {}).get("mae_val_2025"),
            "residual_std": eval_metrics.get("finish_position", {}).get("residual_std_val_2025"),
            "auc_podium": eval_metrics.get("podium", {}).get("auc_val"),
            "auc_points": eval_metrics.get("points", {}).get("auc_val"),
            "auc_dnf": eval_metrics.get("dnf", {}).get("auc_val"),
        },
        # El simulador lee residual_std de aqui: debe ser el del modelo de
        # evaluacion, ya que es la unica medida honesta del error.
        "finish_position": {
            "residual_std_val_2025": eval_metrics.get("finish_position", {}).get("residual_std_val_2025"),
            "mae_val_2025": eval_metrics.get("finish_position", {}).get("mae_val_2025"),
        },
    }
    write_registry(version_name, prod_metrics, feature_cols, role="production")
    logger.info(f"Modelo de produccion guardado en {version_dir}")


def main():
    logger.info(f"Cargando {FEATURES_PATH}")
    raw = pd.read_parquet(FEATURES_PATH)
    df = _prepare(raw[raw["year"] <= 2025])  # 2026 excluido de metricas (temporada en curso)
    feature_cols = _feature_cols(df)
    logger.info(f"Features usadas: {len(feature_cols)}")

    train, val = _temporal_split(df)

    version_name = f"{date.today().isoformat()}_modelB"
    version_dir = MODELS_ROOT / version_name

    all_metrics = {"version": version_name, "n_features": len(feature_cols)}

    reg_model, reg_metrics = _train_regressor(train, val, feature_cols)
    save_xgb_model(reg_model, version_dir, "finish_position_regressor")
    all_metrics["finish_position"] = reg_metrics

    logger.info("SHAP (finish_position) sobre val 2025...")
    all_metrics["finish_position"]["shap_top15"] = _shap_summary(reg_model, val[feature_cols])
    logger.info("Permutation importance (finish_position) sobre val 2025, segunda opinion...")
    all_metrics["finish_position"]["permutation_importance_top15"] = _permutation_importance_check(
        reg_model, val[feature_cols], val["y_finish_position"]
    )

    for target in ["podium", "points", "dnf"]:
        model, a, b, metrics = _train_calibrated_classifier(target, train, val, feature_cols)
        save_xgb_model(model, version_dir, f"{target}_classifier")
        save_calibration(a, b, version_dir, f"{target}_classifier")
        all_metrics[target] = metrics

    # Categorias vistas en ENTRENAMIENTO (no en todo el dataset): al predecir
    # una carrera futura pueden aparecer pilotos o circuitos que el modelo no
    # vio nunca (debutantes como Lindblad en 2026, circuitos nuevos como
    # Madrid). XGBoost rechaza categorias desconocidas, asi que en inferencia
    # se mapean a NaN y el modelo se apoya en la forma reciente del piloto en
    # lugar de en su identidad. Ver RaceSimulator._prep_categoricals.
    categories = {col: sorted(train[col].dropna().unique().tolist()) for col in CATEGORICAL_COLS}
    version_dir.mkdir(parents=True, exist_ok=True)
    (version_dir / "categories.json").write_text(json.dumps(categories, indent=2), encoding="utf-8")

    logger.info("=== Backtest walk-forward (verificacion de estabilidad, sin calibrar) ===")
    all_metrics["walk_forward_backtest"] = _walk_forward_backtest(df, feature_cols)

    _train_production_model(df, feature_cols, all_metrics, categories)

    write_registry(version_name, all_metrics, feature_cols, role="evaluation")
    logger.info(f"Artefactos guardados en {version_dir}, registry.json -> evaluation={version_name}")

    logger.info("=== Verificacion cualitativa: Verstappen 2023 (train) deberia dominar SHAP en forma reciente ===")
    top_shap = list(all_metrics["finish_position"]["shap_top15"].items())[:5]
    for name, val_ in top_shap:
        logger.info(f"  {name}: {val_:.4f}")


if __name__ == "__main__":
    configure()
    main()
