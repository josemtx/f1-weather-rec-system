"""Comprueba que los intervalos del simulador dicen la verdad.

Si el intervalo P10-P90 esta bien calibrado, debe contener el 80% de los
resultados reales. Ni mas ni menos: cubrir el 95% significa que el simulador
es tan prudente que no informa de nada, y cubrir el 60% que promete una
precision que no tiene.

Barre varias escalas del ruido residual sobre la temporada 2025 -- predicha
por el modelo que nunca la vio -- y muestra cual acierta el 80%. El valor
elegido vive en RESIDUAL_SCALE, por regimen (ml/simulation/monte_carlo.py); este script
existe para que esa constante sea verificable y no un numero magico.

Uso: python -m ml.training.calibrate_simulation [version_del_modelo] [pre_quali]
     (por defecto, la version `evaluation` del registry, en regimen post_quali;
     con `pre_quali` se ocultan las features de clasificacion de cada carrera
     para calibrar la incertidumbre del regresor pre-clasificacion)
"""

import json
import logging
import sys

import numpy as np
import pandas as pd

from ml.common.db import get_db
from ml.common.logging_conf import configure
from ml.features.build_features import build_feature_matrix
from ml.simulation.monte_carlo import RESIDUAL_SCALE, RaceSimulator
from ml.training.anchor import REGIME_POST_QUALI, REGIME_PRE_QUALI, blank_quali_features
from ml.training.model_registry import MODELS_ROOT

logger = logging.getLogger(__name__)

SCALES = [1.0, 0.8, 0.65, 0.5, 0.4]
TARGET_COVERAGE = 0.80
SIMS = 1500
VAL_YEAR = 2025


def evaluate_scale(simulator: RaceSimulator, races: list[pd.DataFrame], scale: float, regime: str) -> dict:
    std_attr = "residual_std" if regime == REGIME_POST_QUALI else "residual_std_pre_quali"
    original = getattr(simulator, std_attr)
    try:
        setattr(simulator, std_attr, original * (scale / RESIDUAL_SCALE[regime]))
        covered, widths, near = [], [], []
        for race in races:
            pred = simulator.simulate_race(race, n_simulations=SIMS, seed=42)
            actual = dict(zip(race["driver_code"], race["finish_position"]))
            for _, p in pred.iterrows():
                a = actual.get(p["driver_code"])
                if a is None or pd.isna(a):
                    continue
                covered.append(p["p10_finish_position"] <= a <= p["p90_finish_position"])
                widths.append(p["p90_finish_position"] - p["p10_finish_position"])
                near.append(abs(p["mean_finish_position"] - a) <= 3)
        return {
            "escala": scale,
            "cobertura": float(np.mean(covered)),
            "anchura_media": float(np.mean(widths)),
            "aciertos_mas_menos_3": float(np.mean(near)),
            "n": len(covered),
        }
    finally:
        setattr(simulator, std_attr, original)


def main(model_dir_name: str | None = None, regime: str = REGIME_POST_QUALI) -> None:
    db = get_db()
    if model_dir_name is None:
        registry = json.loads((MODELS_ROOT / "registry.json").read_text(encoding="utf-8"))
        model_dir_name = registry["evaluation"]
    simulator = RaceSimulator(db, model_dir=MODELS_ROOT / model_dir_name)

    features = build_feature_matrix(db)
    season = features[features["year"] == VAL_YEAR]
    races = [season[season["round"] == r].copy() for r in sorted(season["round"].unique())]
    races = [r for r in races if not r["finish_position"].isna().all()]
    if regime == REGIME_PRE_QUALI:
        races = [blank_quali_features(r, regime) for r in races]
    logger.info(f"Calibrando sobre {len(races)} carreras de {VAL_YEAR} en regimen {regime}")

    results = [evaluate_scale(simulator, races, s, regime) for s in SCALES]

    print()
    print(f"{'escala':>7} {'cobertura':>10} {'anchura':>9} {'+-3 puestos':>12}")
    for r in results:
        marca = "  <- en uso" if abs(r["escala"] - RESIDUAL_SCALE[regime]) < 1e-9 else ""
        print(f"{r['escala']:>7.2f} {r['cobertura']*100:>9.1f}% {r['anchura_media']:>8.1f} "
              f"{r['aciertos_mas_menos_3']*100:>11.1f}%{marca}")

    mejor = min(results, key=lambda r: abs(r["cobertura"] - TARGET_COVERAGE))
    print()
    print(f"Escala que mas se acerca al {TARGET_COVERAGE:.0%}: {mejor['escala']:.2f} "
          f"(cobertura {mejor['cobertura']*100:.1f}%)")
    if abs(mejor["escala"] - RESIDUAL_SCALE[regime]) > 1e-9:
        print(f"AVISO: RESIDUAL_SCALE[{regime}] vale {RESIDUAL_SCALE[regime]}; conviene revisarlo.")

    out = MODELS_ROOT / f"simulation_calibration_{regime}.json"
    out.write_text(json.dumps({"objetivo": TARGET_COVERAGE, "en_uso": RESIDUAL_SCALE[regime], "modelo": model_dir_name,
                               "regimen": regime, "resultados": results}, indent=2), encoding="utf-8")
    logger.info(f"Guardado {out}")


if __name__ == "__main__":
    configure()
    main(
        sys.argv[1] if len(sys.argv) > 1 else None,
        REGIME_PRE_QUALI if len(sys.argv) > 2 and sys.argv[2] == REGIME_PRE_QUALI else REGIME_POST_QUALI,
    )
