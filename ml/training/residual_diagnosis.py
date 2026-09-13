"""De donde sale el MAE: descomposicion del error del regresor sobre 2025.

Antes de tocar el modelo conviene saber que parte del error es atacable por
cada palanca. Se entrena la receta de train.py (regimen post_quali, train
2023-24, val 2025) y se descompone el error absoluto:

  - por si el piloto termino o abandono (cuanto MAE es "culpa" del DNF)
  - por tramo de posicion real (cabeza / puntos / pelotón / cola)
  - sesgo con signo por tramo (¿comprime hacia el centro?)
  - orden vs magnitud: MAE de la prediccion cruda frente a la prediccion
    convertida en ranking 1..N dentro de cada carrera
  - baselines ingenuos: parrilla, media movil del piloto

Uso: python -m ml.training.residual_diagnosis
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error

from ml.common.logging_conf import configure
from ml.features.build_features import FEATURES_PATH
from ml.training.anchor import REGIME_POST_QUALI
from ml.training.train import TRAIN_YEARS, VAL_YEAR, _feature_cols, _prepare, fit_regressor, predict_position

logger = logging.getLogger(__name__)

OUT_PATH = Path(__file__).resolve().parents[1] / "models" / "residual_diagnosis.json"

POSITION_BUCKETS = [(1, 3, "P1-3"), (4, 10, "P4-10"), (11, 15, "P11-15"), (16, 20, "P16-20")]


def _rank_within_race(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby(["year", "round"])[col].rank(method="first")


def _mae(y: pd.Series, p: pd.Series | np.ndarray) -> float:
    return float(mean_absolute_error(y, p))


def _by_group(df: pd.DataFrame, mask: pd.Series, label: str) -> dict:
    sub = df[mask]
    return {
        "grupo": label,
        "n": int(len(sub)),
        "mae": _mae(sub["y"], sub["pred"]),
        "sesgo": float((sub["pred"] - sub["y"]).mean()),
        "share_error_total": float(sub["abs_err"].sum() / df["abs_err"].sum()),
    }


def main():
    raw = pd.read_parquet(FEATURES_PATH)
    df = _prepare(raw[raw["year"] <= VAL_YEAR])
    feature_cols = _feature_cols(df)
    train = df[df["year"].isin(TRAIN_YEARS)]
    val = df[df["year"] == VAL_YEAR].copy()

    val["pred"] = predict_position(fit_regressor(train, feature_cols, REGIME_POST_QUALI), val, feature_cols, REGIME_POST_QUALI)
    val["y"] = val["y_finish_position"]
    val["abs_err"] = (val["pred"] - val["y"]).abs()
    val["pred_rank"] = _rank_within_race(val, "pred")
    val["dnf"] = val["y_dnf"] == 1

    out: dict = {"n_val": int(len(val)), "n_carreras": int(val["round"].nunique())}

    # 1. Global, y que pasaria si los DNF se predijeran perfectamente
    out["mae_total"] = _mae(val["y"], val["pred"])
    out["mae_ranking_en_carrera"] = _mae(val["y"], val["pred_rank"])
    out["spearman_medio_por_carrera"] = float(np.mean([
        spearmanr(g["y"], g["pred"]).correlation for _, g in val.groupby(["year", "round"])
    ]))

    # 2. Terminaron vs abandonaron
    out["por_resultado"] = [
        _by_group(val, ~val["dnf"], "termino"),
        _by_group(val, val["dnf"], "abandono"),
    ]
    finishers = val[~val["dnf"]]
    out["mae_si_dnf_perfecto"] = float(finishers["abs_err"].sum() / len(val))

    # 3. Por tramo de posicion real, con sesgo (positivo = predice peor puesto)
    out["por_tramo_real"] = [
        _by_group(val, val["y"].between(lo, hi), label) for lo, hi, label in POSITION_BUCKETS
    ]
    out["por_tramo_real_solo_finalizadores"] = [
        _by_group(finishers, finishers["y"].between(lo, hi), label) for lo, hi, label in POSITION_BUCKETS
    ]

    # 4. Compresion: rango de la prediccion frente al real
    out["compresion"] = {
        "std_real": float(val["y"].std()),
        "std_pred": float(val["pred"].std()),
        "pred_min_medio_por_carrera": float(val.groupby("round")["pred"].min().mean()),
        "pred_max_medio_por_carrera": float(val.groupby("round")["pred"].max().mean()),
    }

    # 5. Baselines ingenuos sobre las mismas filas
    grid = val["grid_position"].astype(float)
    grid_rank = _rank_within_race(val.assign(_g=grid.fillna(20)), "_g")
    roll = val["driver_avg_finish_last5"].astype(float)
    baselines = {
        "parrilla": _mae(val["y"], grid.fillna(20)),
        "parrilla_como_ranking": _mae(val["y"], grid_rank),
        "media_movil_5_piloto": _mae(val["y"], roll.fillna(roll.mean())),
        "media_movil_5_como_ranking": _mae(val["y"], _rank_within_race(val.assign(_r=roll.fillna(99)), "_r")),
    }
    fin_mask = ~val["dnf"]
    baselines["parrilla_solo_finalizadores"] = _mae(val.loc[fin_mask, "y"], grid[fin_mask].fillna(20))
    baselines["modelo_solo_finalizadores"] = _mae(val.loc[fin_mask, "y"], val.loc[fin_mask, "pred"])
    out["baselines"] = baselines

    # 6. Carreras peores
    per_race = val.groupby(["round", "circuit_short_name"], observed=True).agg(
        mae=("abs_err", "mean"), n_dnf=("dnf", "sum"), wet=("race_day_wet_fraction", "max"),
    ).reset_index().sort_values("mae", ascending=False)
    out["peores_carreras"] = per_race.head(6).to_dict(orient="records")
    out["mejores_carreras"] = per_race.tail(3).to_dict(orient="records")
    out["corr_mae_carrera_vs_n_dnf"] = float(per_race["mae"].corr(per_race["n_dnf"]))

    # 7. Mayores errores individuales
    worst = val.nlargest(8, "abs_err")[["round", "circuit_short_name", "driver_code", "grid_position", "y", "pred", "status"]]
    out["peores_predicciones"] = worst.round(1).to_dict(orient="records")

    _log(out)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    logger.info(f"Guardado en {OUT_PATH}")


def _log(out: dict):
    logger.info(f"=== Diagnostico de residuos, {VAL_YEAR} ({out['n_val']} filas, {out['n_carreras']} carreras) ===")
    logger.info(f"MAE total {out['mae_total']:.3f} | como ranking en carrera {out['mae_ranking_en_carrera']:.3f} | "
                f"Spearman medio {out['spearman_medio_por_carrera']:.3f}")
    logger.info(f"MAE si los DNF se predijeran perfectamente: {out['mae_si_dnf_perfecto']:.3f}")
    for g in out["por_resultado"]:
        logger.info(f"  {g['grupo']:9s} n={g['n']:3d} MAE={g['mae']:.2f} sesgo={g['sesgo']:+.2f} share={g['share_error_total']:.0%}")
    logger.info("Por tramo real (todos / solo finalizadores):")
    for a, b in zip(out["por_tramo_real"], out["por_tramo_real_solo_finalizadores"]):
        logger.info(f"  {a['grupo']:7s} MAE={a['mae']:.2f} sesgo={a['sesgo']:+.2f} share={a['share_error_total']:.0%}"
                    f"   | fin: MAE={b['mae']:.2f} sesgo={b['sesgo']:+.2f}")
    c = out["compresion"]
    logger.info(f"Compresion: std real {c['std_real']:.2f} vs pred {c['std_pred']:.2f}; "
                f"pred media por carrera va de {c['pred_min_medio_por_carrera']:.1f} a {c['pred_max_medio_por_carrera']:.1f}")
    logger.info("Baselines:")
    for k, v in out["baselines"].items():
        logger.info(f"  {k:32s} {v:.3f}")
    logger.info(f"Correlacion MAE por carrera vs n DNF: {out['corr_mae_carrera_vs_n_dnf']:.2f}")
    logger.info("Peores carreras:")
    for r in out["peores_carreras"]:
        logger.info(f"  R{r['round']:2d} {r['circuit_short_name']:12s} MAE={r['mae']:.2f} dnf={r['n_dnf']} wet={r['wet']}")
    logger.info("Peores predicciones individuales:")
    for r in out["peores_predicciones"]:
        logger.info(f"  R{r['round']:2d} {r['circuit_short_name']:12s} {r['driver_code']} grid={r['grid_position']} "
                    f"real={r['y']} pred={r['pred']} ({r['status']})")


if __name__ == "__main__":
    configure()
    main()
