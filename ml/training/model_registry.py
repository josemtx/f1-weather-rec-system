"""Guardado/carga de artefactos de modelo SIN pickle (portabilidad/seguridad).

XGBoost se guarda en su formato nativo .json. La calibracion sigmoid
(Platt scaling) de CalibratedClassifierCV se reduce a dos escalares (a, b)
que se guardan como JSON plano y se re-aplican manualmente en inferencia con
la misma formula que usa sklearn internamente (verificado: 1/(1+exp(a*p+b))
sobre la probabilidad cruda de XGBoost, diff=0.0 frente a sklearn):
ver ml/training/train.py donde se ajusta y valida.
"""

import json
import math
from pathlib import Path

import numpy as np
import xgboost as xgb

MODELS_ROOT = Path(__file__).resolve().parents[1] / "models"


def save_xgb_model(model, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.json"
    model.save_model(str(path))
    return path


def load_xgb_regressor(path: Path) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor()
    model.load_model(str(path))
    return model


def load_xgb_classifier(path: Path) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier()
    model.load_model(str(path))
    return model


def save_calibration(a: float, b: float, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}_calibration.json"
    path.write_text(json.dumps({"a": a, "b": b}), encoding="utf-8")
    return path


def load_calibration(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def apply_calibration(raw_proba: np.ndarray, a: float, b: float) -> np.ndarray:
    """Misma formula que sklearn CalibratedClassifierCV(method='sigmoid')
    aplicaria sobre predict_proba de un clasificador sin decision_function
    (caso de XGBClassifier). Verificado bit-a-bit contra sklearn en pruebas."""
    return 1.0 / (1.0 + np.exp(a * raw_proba + b))


def write_registry(
    version_dir_name: str, metrics: dict, feature_list: list[str], role: str = "production"
) -> Path:
    """Registra una version bajo un rol.

    Se distinguen dos roles porque cumplen funciones incompatibles:
      - `evaluation`: entrenado dejando 2025 fuera. Sus metricas son creibles
        precisamente porque nunca vio ese anio.
      - `production`: misma receta entrenada con TODO lo disponible (2025
        incluido). Predice mejor, pero no se le pueden medir las notas sobre
        datos que ya ha visto: su calidad estimada es la del modelo de
        evaluacion.
    """
    registry_path = MODELS_ROOT / "registry.json"
    registry = {}
    if registry_path.exists():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry[role] = version_dir_name
    registry.setdefault("history", [])
    if version_dir_name not in registry["history"]:
        registry["history"].append(version_dir_name)
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")

    version_dir = MODELS_ROOT / version_dir_name
    (version_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    (version_dir / "feature_list.json").write_text(json.dumps(feature_list, indent=2), encoding="utf-8")
    return registry_path


def get_production_dir() -> Path:
    registry_path = MODELS_ROOT / "registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    return MODELS_ROOT / registry["production"]
