# ml_blueprint.py
#
# Blueprint Flask para el subsistema predictivo -- se monta en
# python-app/server.py junto a los endpoints existentes (consulta1/2/3),
# sin modificarlos. Reutiliza el mismo estilo de logging/manejo de errores.

import json
import logging
from pathlib import Path

import pandas as pd
from flask import Blueprint, jsonify

from ml.common.db import get_db
from ml.simulation.monte_carlo import RaceSimulator
from ml.training.model_registry import MODELS_ROOT, get_production_dir

logger = logging.getLogger(__name__)

ml_bp = Blueprint("ml", __name__, url_prefix="/ml")

FEATURES_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "features_v3.parquet"

_features_cache: pd.DataFrame | None = None
_simulator_cache: RaceSimulator | None = None


def _get_features() -> pd.DataFrame:
    global _features_cache
    if _features_cache is None:
        _features_cache = pd.read_parquet(FEATURES_PATH)
    return _features_cache


def _get_simulator() -> RaceSimulator:
    global _simulator_cache
    if _simulator_cache is None:
        _simulator_cache = RaceSimulator(get_db())
    return _simulator_cache


@ml_bp.route("/model_info", methods=["GET"])
def model_info():
    logger.info("Ejecutando /ml/model_info")
    try:
        registry = json.loads((MODELS_ROOT / "registry.json").read_text(encoding="utf-8"))
        prod_dir = get_production_dir()
        metrics = json.loads((prod_dir / "metrics.json").read_text(encoding="utf-8"))
        return jsonify({
            "production_version": registry["production"],
            "history": registry.get("history", []),
            "n_features": metrics.get("n_features"),
            "finish_position_mae_val": metrics.get("finish_position", {}).get("mae_val_2025"),
            "podium_auc_val": metrics.get("podium", {}).get("auc_val"),
            "points_auc_val": metrics.get("points", {}).get("auc_val"),
            "dnf_auc_val": metrics.get("dnf", {}).get("auc_val"),
            "walk_forward_backtest": metrics.get("walk_forward_backtest"),
        })
    except Exception as e:
        logger.exception(f"Error en /ml/model_info: {e}")
        return jsonify({"error": "Error interno"}), 500


@ml_bp.route("/races", methods=["GET"])
def list_races():
    logger.info("Ejecutando /ml/races")
    try:
        df = _get_features()
        races = (
            df[["year", "round", "circuit_short_name"]]
            .drop_duplicates()
            .sort_values(["year", "round"])
        )
        return jsonify(races.to_dict(orient="records"))
    except Exception as e:
        logger.exception(f"Error en /ml/races: {e}")
        return jsonify({"error": "Error interno"}), 500


@ml_bp.route("/predict/<int:year>/<int:round_number>", methods=["GET"])
def predict_race(year: int, round_number: int):
    logger.info(f"Ejecutando /ml/predict/{year}/{round_number}")
    try:
        df = _get_features()
        race = df[(df["year"] == year) & (df["round"] == round_number)].copy()
        if race.empty:
            return jsonify({"error": f"No hay datos para year={year} round={round_number}"}), 404

        simulator = _get_simulator()
        result = simulator.simulate_race(race, n_simulations=2000, seed=42)

        response = {
            "year": year,
            "round": round_number,
            "circuit_short_name": race["circuit_short_name"].iloc[0],
            "actual_results": race[["driver_code", "grid_position", "finish_position"]]
            .sort_values("finish_position")
            .to_dict(orient="records"),
            "predictions": result.to_dict(orient="records"),
        }
        return jsonify(response)
    except Exception as e:
        logger.exception(f"Error en /ml/predict/{year}/{round_number}: {e}")
        return jsonify({"error": "Error interno"}), 500
