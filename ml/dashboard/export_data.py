"""Exporta a JSON todo lo que consume el dashboard estatico.

El dashboard es una pagina estatica para que cualquiera pueda verla desde
GitHub Pages sin clonar el repo, instalar dependencias ni levantar MongoDB.
Eso obliga a precalcular aqui lo que normalmente se haria en vivo.

El sandbox what-if es el caso claro: una simulacion en vivo necesitaria
Python detras, asi que se precalcula una rejilla de escenarios
(temperatura x probabilidad de lluvia) y los deslizadores se mueven por
ella. Es una rejilla, no una simulacion en vivo, y el dashboard lo dice.

Uso: python -m ml.dashboard.export_data
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ml.common.db import get_db
from ml.common.logging_conf import configure
from ml.dashboard.circuit_art import get_circuit_svg_markup
from ml.dashboard.nerd_data import build_nerd_data_payload
from ml.features.build_features import build_feature_matrix
from ml.features.upcoming import get_next_race
from ml.predictions.prediction_log import detect_information_regime, load_predictions_with_outcome
from ml.simulation.monte_carlo import RaceSimulator
from ml.simulation.randomness_models import rain_probability
from ml.training.model_registry import MODELS_ROOT, get_production_dir

logger = logging.getLogger(__name__)

OUT_PATH = Path(__file__).resolve().parent / "web" / "data.json"

TEMP_OFFSETS = [-10, -5, 0, 5, 10]
RAIN_PROBS = [0.0, 0.25, 0.5, 0.75, 1.0]
SANDBOX_SIMS = 2000
NEXT_RACE_SIMS = 5000
BACKTEST_SIMS = 1500


def _clean(value):
    """NaN/NaT no son JSON validos."""
    if value is None:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if np.isnan(value) else round(float(value), 4)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _drivers_payload(predictions: pd.DataFrame, teams: dict) -> list[dict]:
    out = []
    for _, row in predictions.iterrows():
        code = row["driver_code"]
        out.append({
            "code": _clean(code),
            "team": teams.get(code),
            "p_win": _clean(row["p_win"]),
            "p_podium": _clean(row["p_podium"]),
            "p_points": _clean(row["p_points"]),
            "p_dnf": _clean(row["p_dnf"]),
            "mean_position": _clean(row["mean_finish_position"]),
            "p10": _clean(row["p10_finish_position"]),
            "p90": _clean(row["p90_finish_position"]),
            # Rango util: donde acaba SI TERMINA. El total mezcla abandonos y
            # se vuelve ilegible ("entre P1 y P18" no informa de nada).
            "p10_fin": _clean(row.get("p10_if_finishes")),
            "p90_fin": _clean(row.get("p90_if_finishes")),
        })
    return out


def _apply_scenario(race: pd.DataFrame, temp_offset: float, rain_prob: float) -> pd.DataFrame:
    """Reescribe el clima de la carrera para un escenario hipotetico."""
    scenario = race.copy()
    for col in ["race_day_temp_avg", "race_day_temp_max"]:
        if col in scenario.columns:
            scenario[col] = pd.to_numeric(scenario[col], errors="coerce") + temp_offset
    if "race_day_track_temp" in scenario.columns:
        # El asfalto se mueve mas que el aire: la pendiente ajustada sobre los
        # sensores reales ronda 1.3 C de pista por cada C de aire.
        scenario["race_day_track_temp"] = (
            pd.to_numeric(scenario["race_day_track_temp"], errors="coerce") + temp_offset * 1.3
        )
    if "forecast_rain_probability" in scenario.columns:
        scenario["forecast_rain_probability"] = rain_prob
    if "climate_rain_probability_flag" in scenario.columns:
        scenario["climate_rain_probability_flag"] = 1.0 if rain_prob >= 0.5 else 0.0
    if "race_day_precipitation_mm" in scenario.columns:
        scenario["race_day_precipitation_mm"] = round(rain_prob * 6.0, 2)
    return scenario


def build_next_race_payload(db, simulator: RaceSimulator) -> tuple[dict, dict, dict]:
    nxt = get_next_race(db)
    if nxt is None:
        return {}, {}, {}

    year, round_number = int(nxt["year"]), int(nxt["round"])
    features = build_feature_matrix(db, upcoming=(year, round_number))
    race = features[(features["year"] == year) & (features["round"] == round_number)].copy()

    circuit_doc = db["circuits"].find_one({"jolpica_circuit_id": nxt["circuit_id"]}, {"_id": 0}) or {}
    teams = dict(zip(race["driver_code"], race["constructor_name"]))
    row = race.iloc[0]

    predictions = simulator.simulate_race(race, n_simulations=NEXT_RACE_SIMS, seed=42)

    race_date = nxt["race_date"]
    payload = {
        "year": year,
        "round": round_number,
        "circuit": row["circuit_short_name"],
        "circuit_name": circuit_doc.get("circuit_name"),
        "country": circuit_doc.get("country"),
        "circuit_type": circuit_doc.get("circuit_type"),
        "race_date": race_date.strftime("%Y-%m-%d") if hasattr(race_date, "strftime") else str(race_date),
        "race_time": nxt.get("race_time"),
        "qualifying_date": nxt.get("qualifying_date"),
        "has_sprint": bool(nxt.get("sprint_date")),
        "regime": detect_information_regime(db, year, round_number),
        "model_version": simulator.model_dir.name,
        "circuit_svg": get_circuit_svg_markup(row["circuit_short_name"]),
        "weather": {
            "source": _clean(row.get("climate_source")),
            "temp": _clean(row.get("race_day_temp_avg")),
            "track_temp": _clean(row.get("race_day_track_temp")),
            "humidity": _clean(row.get("race_day_humidity")),
            "wind": _clean(row.get("race_day_wind_speed")),
            "rain_mm": _clean(row.get("race_day_precipitation_mm")),
            # Sin pronostico, la misma climatologia del circuito que usa el simulador.
            "rain_prob": _clean(row.get("forecast_rain_probability")) if pd.notna(row.get("forecast_rain_probability"))
            else _clean(rain_probability(db, row["circuit_short_name"], race_date.month)),
        },
        "drivers": _drivers_payload(predictions, teams),
    }

    logger.info(f"Escenarios del sandbox: {len(TEMP_OFFSETS)}x{len(RAIN_PROBS)} combinaciones")
    grid = {}
    for offset in TEMP_OFFSETS:
        for rain in RAIN_PROBS:
            scenario = _apply_scenario(race, offset, rain)
            result = simulator.simulate_race(scenario, n_simulations=SANDBOX_SIMS, seed=42)
            grid[f"{offset}|{rain}"] = _drivers_payload(result, teams)

    sandbox = {
        "temp_offsets": TEMP_OFFSETS,
        "rain_probs": RAIN_PROBS,
        "base_temp": _clean(row.get("race_day_temp_avg")),
        "grid": grid,
    }

    nerd_data = build_nerd_data_payload(race)
    return payload, sandbox, nerd_data


def build_backtest_payload(db, year: int = 2025) -> dict:
    """Que habriamos predicho en cada carrera de 2025, con el modelo que NUNCA
    vio ese anio, comparado con lo que paso de verdad."""
    registry = json.loads((MODELS_ROOT / "registry.json").read_text(encoding="utf-8"))
    eval_dir = MODELS_ROOT / registry["evaluation"]
    simulator = RaceSimulator(db, model_dir=eval_dir)

    features = build_feature_matrix(db)
    season = features[features["year"] == year]
    races = []

    for round_number in sorted(season["round"].unique()):
        race = season[season["round"] == round_number].copy()
        if race.empty or race["finish_position"].isna().all():
            continue
        predictions = simulator.simulate_race(race, n_simulations=BACKTEST_SIMS, seed=42)
        actual = dict(zip(race["driver_code"], race["finish_position"]))
        grid = dict(zip(race["driver_code"], race["grid_position"]))
        finished = dict(zip(race["driver_code"], race["finished"]))

        drivers = []
        for _, p in predictions.iterrows():
            code = p["driver_code"]
            drivers.append({
                "code": code,
                "p_win": _clean(p["p_win"]),
                "p_podium": _clean(p["p_podium"]),
                "predicted_mean": _clean(p["mean_finish_position"]),
                "actual": _clean(actual.get(code)),
                # La parrilla es el baseline honesto: lo que acertarias sin modelo.
                "grid": _clean(grid.get(code)),
                "finished": _clean(finished.get(code)),
            })
        races.append({
            "year": int(year),
            "round": int(round_number),
            "circuit": race["circuit_short_name"].iloc[0],
            "drivers": drivers,
        })

    logger.info(f"Backtest de {year}: {len(races)} carreras")
    return {"model_version": registry["evaluation"], "races": races}


def _calibration_in_use() -> dict:
    """Cobertura real del intervalo P10-P90 con la escala en uso, por regimen
    (salida de ml/training/calibrate_simulation.py)."""
    out = {}
    for regime in ("post_quali", "pre_quali"):
        path = MODELS_ROOT / f"simulation_calibration_{regime}.json"
        if not path.exists():
            continue
        cal = json.loads(path.read_text(encoding="utf-8"))
        row = next((r for r in cal["resultados"] if abs(r["escala"] - cal["en_uso"]) < 1e-9), None)
        if row:
            out[regime] = {"scale": cal["en_uso"], "coverage": row["cobertura"],
                           "width": row["anchura_media"], "within3": row["aciertos_mas_menos_3"]}
    return out


def build_model_payload() -> dict:
    registry = json.loads((MODELS_ROOT / "registry.json").read_text(encoding="utf-8"))
    eval_metrics = json.loads((MODELS_ROOT / registry["evaluation"] / "metrics.json").read_text(encoding="utf-8"))

    ablation_path = MODELS_ROOT / "ablation_information_regimes.json"
    ablation = json.loads(ablation_path.read_text(encoding="utf-8")) if ablation_path.exists() else {}

    shap = eval_metrics.get("finish_position", {}).get("shap_top15", {})
    post = eval_metrics.get("finish_position", {})
    pre = eval_metrics.get("finish_position_pre_quali", {})
    return {
        "evaluation_version": registry["evaluation"],
        "production_version": registry.get("production"),
        "mae": _clean(post.get("mae_val_2025")),
        "mae_anchor": _clean(post.get("mae_ancla_val_2025")),
        "mae_pre_quali": _clean(pre.get("mae_val_2025")),
        "mae_anchor_pre_quali": _clean(pre.get("mae_ancla_val_2025")),
        "calibration": _calibration_in_use(),
        "residual_std": _clean(post.get("residual_std_val_2025")),
        "auc_podium": _clean(eval_metrics.get("podium", {}).get("auc_val")),
        "auc_points": _clean(eval_metrics.get("points", {}).get("auc_val")),
        "auc_dnf": _clean(eval_metrics.get("dnf", {}).get("auc_val")),
        "n_features": eval_metrics.get("n_features"),
        "walk_forward": eval_metrics.get("walk_forward_backtest", {}),
        "ablation": ablation,
        "shap_top": [{"feature": k, "value": _clean(v)} for k, v in list(shap.items())[:12]],
    }


def build_track_record_payload(db) -> list[dict]:
    df = load_predictions_with_outcome(db)
    if df.empty:
        return []
    return [
        {
            "year": int(r["year"]), "round": int(r["round"]), "circuit": r["circuit_short_name"],
            "driver": r["driver_code"], "regime": r["information_regime"],
            "predicted_at": r["predicted_at"], "climate_source": r["climate_source"],
            "p_win": _clean(r["p_win"]), "p_podium": _clean(r["p_podium"]),
            "predicted_mean": _clean(r["mean_finish_position"]),
            "actual": _clean(r.get("actual_finish_position")),
        }
        for _, r in df.iterrows()
    ]


def main() -> None:
    db = get_db()
    simulator = RaceSimulator(db, model_dir=get_production_dir())

    logger.info("Construyendo proxima carrera, sandbox y nerd data...")
    next_race, sandbox, nerd_data = build_next_race_payload(db, simulator)

    logger.info("Construyendo backtest de 2025...")
    backtest = build_backtest_payload(db)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "next_race": next_race,
        "sandbox": sandbox,
        "nerd_data": nerd_data,
        "model": build_model_payload(),
        "backtest": backtest,
        "track_record": build_track_record_payload(db),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # allow_nan=False a proposito: NaN no es JSON valido y el navegador solo
    # dice "Unexpected token N". Mejor que reviente aqui, con el traceback
    # senalando el campo, que exportar un fichero que el dashboard no puede leer.
    OUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, allow_nan=False), encoding="utf-8"
    )
    logger.info(f"Guardado {OUT_PATH} ({OUT_PATH.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    configure()
    main()
