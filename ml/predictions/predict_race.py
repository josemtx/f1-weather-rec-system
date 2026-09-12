"""Genera y registra la prediccion de una carrera.

Uso:
    python -m ml.predictions.predict_race              # proxima carrera
    python -m ml.predictions.predict_race 2026 14      # una concreta

El regimen de informacion (pre_quali / post_sprint / post_quali) no se pasa
por parametro: se deduce de que datos del fin de semana existen ya. Basta
con volver a lanzarlo tras la clasificacion para tener la segunda prediccion.
"""

import logging
import sys

import pandas as pd

from ml.common.db import get_db
from ml.common.logging_conf import configure
from ml.features.build_features import build_feature_matrix
from ml.features.upcoming import get_next_race
from ml.predictions.prediction_log import (
    detect_information_regime,
    race_already_run,
    save_predictions,
)
from ml.simulation.monte_carlo import RaceSimulator

logger = logging.getLogger(__name__)

N_SIMULATIONS = 5000
SEED = 42


def predict_and_store(db, year: int, round_number: int, n_simulations: int = N_SIMULATIONS) -> pd.DataFrame:
    if race_already_run(db, year, round_number):
        raise ValueError(f"{year} R{round_number} ya se disputo; no tiene sentido predecirla.")

    regime = detect_information_regime(db, year, round_number)
    logger.info(f"Prediciendo {year} R{round_number} | regimen de informacion: {regime}")

    features = build_feature_matrix(db, upcoming=(year, round_number))
    race = features[(features["year"] == year) & (features["round"] == round_number)].copy()
    if race.empty:
        raise ValueError(f"No se pudieron construir features para {year} R{round_number}")

    simulator = RaceSimulator(db)
    predictions = simulator.simulate_race(race, n_simulations=n_simulations, seed=SEED)

    circuit = race["circuit_short_name"].iloc[0]
    climate_source = race["climate_source"].iloc[0]

    save_predictions(
        db, year, round_number, circuit, predictions,
        model_version=simulator.model_dir.name,
        climate_source=climate_source,
        n_simulations=n_simulations,
        seed=SEED,
    )

    _report(race, predictions, regime, circuit)
    return predictions


def _report(race: pd.DataFrame, predictions: pd.DataFrame, regime: str, circuit: str) -> None:
    row = race.iloc[0]
    print()
    print("=" * 78)
    print(f"  {int(row['year'])} RONDA {int(row['round'])} - {circuit.upper()}")
    print(f"  Regimen: {regime}   |   Clima: {row['climate_source']}")
    temp = row.get("race_day_temp_avg")
    track = row.get("race_day_track_temp")
    if pd.notna(temp):
        track_txt = f", asfalto {track:.0f}C (estimado)" if pd.notna(track) else ""
        print(f"  {temp:.1f}C aire{track_txt}, lluvia {row.get('race_day_precipitation_mm', 0):.1f}mm")
    print("=" * 78)
    cols = ["driver_code", "p_win", "p_podium", "p_points", "p_dnf", "mean_finish_position"]
    shown = predictions[cols].head(10).copy()
    for c in ["p_win", "p_podium", "p_points", "p_dnf"]:
        shown[c] = (shown[c] * 100).round(1).astype(str) + "%"
    shown["mean_finish_position"] = shown["mean_finish_position"].round(2)
    print(shown.to_string(index=False))
    print()


def main() -> None:
    db = get_db()

    if len(sys.argv) >= 3:
        year, round_number = int(sys.argv[1]), int(sys.argv[2])
    else:
        nxt = get_next_race(db)
        if nxt is None:
            logger.error("No hay ninguna carrera futura en el calendario.")
            return
        year, round_number = int(nxt["year"]), int(nxt["round"])

    predict_and_store(db, year, round_number)


if __name__ == "__main__":
    configure()
    main()
