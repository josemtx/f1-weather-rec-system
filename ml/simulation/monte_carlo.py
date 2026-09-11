"""Simulador Monte Carlo de carrera (ver plan, seccion 6).

Consume los modelos XGBoost ya entrenados (regresor de posicion final +
clasificador DNF calibrado) y compone las fuentes de ruido de
randomness_models.py para generar una distribucion de resultado por piloto,
no una unica prediccion puntual.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ml.simulation.randomness_models import (
    climate_variability,
    rain_probability,
    safety_car_shock_scale,
    sample_dnf_flags,
    sample_pit_duration_jitter,
)
from ml.training.model_registry import (
    apply_calibration,
    get_production_dir,
    load_calibration,
    load_xgb_classifier,
    load_xgb_regressor,
)

logger = logging.getLogger(__name__)

CATEGORICAL_COLS = ["circuit_short_name", "circuit_type", "overtaking_difficulty", "era_data_tier", "driver_code", "constructor_id"]

# Respaldo si el modelo no trae metrics.json (modelos antiguos): orden de
# magnitud del error tipico observado, ~3-4 posiciones.
DEFAULT_RESIDUAL_STD = 3.5


class RaceSimulator:
    def __init__(self, db, model_dir: Path | None = None):
        self.db = db
        model_dir = model_dir or get_production_dir()
        self.model_dir = model_dir
        self.feature_cols: list[str] = json.loads((model_dir / "feature_list.json").read_text())

        self.reg = load_xgb_regressor(model_dir / "finish_position_regressor.json")
        self.dnf_clf = load_xgb_classifier(model_dir / "dnf_classifier.json")
        self.dnf_cal = load_calibration(model_dir / "dnf_classifier_calibration.json")

        categories_path = model_dir / "categories.json"
        self.categories: dict[str, list] = (
            json.loads(categories_path.read_text(encoding="utf-8")) if categories_path.exists() else {}
        )

        # Incertidumbre del modelo, medida en validacion. Sin esto el
        # simulador solo modela la variabilidad del MUNDO (clima, fiabilidad)
        # y asume que su propia prediccion es exacta -- daba resultados como
        # "si no abandona, gana el 100% de las veces".
        metrics_path = model_dir / "metrics.json"
        self.residual_std = DEFAULT_RESIDUAL_STD
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.residual_std = float(
                metrics.get("finish_position", {}).get("residual_std_val_2025", DEFAULT_RESIDUAL_STD)
            )
        logger.info(f"Incertidumbre del modelo (desviacion del error): {self.residual_std:.2f} posiciones")

        logger.info(f"RaceSimulator cargado desde {model_dir} ({len(self.feature_cols)} features)")

    def _prep_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fija las categorias a las vistas en entrenamiento.

        Un valor no visto (piloto debutante, circuito nuevo) se convierte en
        NaN en vez de hacer fallar a XGBoost, que rechaza categorias
        desconocidas. El modelo entonces predice por la forma reciente del
        piloto y no por su identidad, que es lo correcto: de un debutante no
        se sabe nada como efecto fijo, pero si como rendimiento reciente.
        """
        df = df.copy()
        for col in CATEGORICAL_COLS:
            if col not in df.columns:
                continue
            known = self.categories.get(col)
            if known:
                unseen = set(df[col].dropna().unique()) - set(known)
                if unseen:
                    logger.info(f"Categorias no vistas en entrenamiento en '{col}' -> NaN: {sorted(unseen)}")
                df[col] = pd.Categorical(df[col], categories=known)
            else:
                df[col] = df[col].astype("category")
        return df

    def simulate_race(
        self, race_features: pd.DataFrame, n_simulations: int = 2000, seed: int = 42
    ) -> pd.DataFrame:
        """race_features: una fila por piloto con al menos self.feature_cols +
        driver_code + circuit_short_name + circuit_type + race_date. Devuelve
        un DataFrame con probabilidades agregadas por piloto tras N simulaciones."""
        race_features = self._prep_categoricals(race_features).reset_index(drop=True)
        n_drivers = len(race_features)
        rng = np.random.default_rng(seed)

        circuit_short_name = race_features["circuit_short_name"].iloc[0]
        circuit_type = race_features["circuit_type"].iloc[0]
        month = pd.to_datetime(race_features["race_date"].iloc[0]).month

        climate_std = climate_variability(self.db, circuit_short_name, month)
        sc_scale = safety_car_shock_scale(circuit_type)

        # Probabilidad de lluvia: si la fila trae la del propio pronostico
        # (`pop` de OpenWeatherMap, carreras futuras), se usa esa -- es una
        # estimacion del dia concreto, no la frecuencia historica del mes.
        # Si no, se cae a la climatologia del circuito.
        forecast_rain_p = None
        if "forecast_rain_probability" in race_features.columns:
            values = pd.to_numeric(race_features["forecast_rain_probability"], errors="coerce").dropna()
            if not values.empty:
                forecast_rain_p = float(values.max())

        if forecast_rain_p is not None:
            rain_p = forecast_rain_p
            logger.info(f"Lluvia muestreada del pronostico: p={rain_p:.2f}")
        else:
            rain_p = rain_probability(self.db, circuit_short_name, month)

        X_base = race_features[self.feature_cols]
        base_dnf_raw = self.dnf_clf.predict_proba(X_base)[:, 1]
        base_dnf_proba = apply_calibration(base_dnf_raw, self.dnf_cal["a"], self.dnf_cal["b"])

        driver_codes = race_features["driver_code"].values
        rank_accum = {code: [] for code in driver_codes}
        dnf_accum = {code: [] for code in driver_codes}

        pit_std = (
            race_features["team_pit_stop_duration_stddev_last5"].values
            if "team_pit_stop_duration_stddev_last5" in race_features.columns
            else np.full(n_drivers, np.nan)
        )

        for _ in range(n_simulations):
            X = X_base.copy()

            temp_noise = rng.normal(0, climate_std["temp_2m_avg"])
            if "race_day_temp_avg" in X.columns:
                X["race_day_temp_avg"] = X["race_day_temp_avg"] + temp_noise
            if "race_day_temp_max" in X.columns:
                X["race_day_temp_max"] = X["race_day_temp_max"] + temp_noise

            is_rain_sim = rng.random() < rain_p
            if is_rain_sim and "race_day_precipitation_mm" in X.columns:
                X["race_day_precipitation_mm"] = np.maximum(
                    X["race_day_precipitation_mm"], rng.exponential(3.0)
                )
                if "climate_rain_probability_flag" in X.columns:
                    X["climate_rain_probability_flag"] = 1.0

            if "race_day_humidity" in X.columns:
                X["race_day_humidity"] = X["race_day_humidity"] + rng.normal(0, climate_std["humidity_relative"])
            if "race_day_wind_speed" in X.columns:
                X["race_day_wind_speed"] = np.maximum(
                    0.0, X["race_day_wind_speed"] + rng.normal(0, climate_std["wind_speed_10m"])
                )

            if "team_avg_pit_stop_duration_last5" in X.columns:
                jitter = np.array([sample_pit_duration_jitter(s, rng, 1)[0] for s in pit_std])
                X["team_avg_pit_stop_duration_last5"] = X["team_avg_pit_stop_duration_last5"] + jitter

            dnf_this_sim = sample_dnf_flags(base_dnf_proba, rng)
            if is_rain_sim:
                # Shock adicional de fiabilidad: lluvia + circuitos propensos a
                # SC/VSC elevan el riesgo de incidente (simplificacion
                # documentada, no un modelo de SC por vuelta -- ver plan).
                extra_dnf = rng.random(n_drivers) < (0.05 * sc_scale)
                dnf_this_sim = dnf_this_sim | extra_dnf

            scores = self.reg.predict(X)

            # Ruido residual: representa lo que el modelo NO sabe. Calibrado
            # con la dispersion real de su error en validacion, de modo que
            # la distribucion de resultados simulados sea tan ancha como de
            # verdad lo es su capacidad predictiva.
            scores = scores + rng.normal(0.0, self.residual_std, size=n_drivers)

            worst_score = max(float(np.max(scores)), n_drivers) + 10.0
            scores_sim = np.where(dnf_this_sim, worst_score + rng.random(n_drivers), scores)

            order = np.argsort(scores_sim)
            ranks = np.empty(n_drivers, dtype=int)
            ranks[order] = np.arange(1, n_drivers + 1)

            for i, code in enumerate(driver_codes):
                rank_accum[code].append(int(ranks[i]))
                dnf_accum[code].append(bool(dnf_this_sim[i]))

        results = []
        for code in driver_codes:
            ranks_arr = np.array(rank_accum[code])
            dnf_arr = np.array(dnf_accum[code])
            results.append({
                "driver_code": code,
                "p_win": float((ranks_arr == 1).mean()),
                "p_podium": float((ranks_arr <= 3).mean()),
                "p_points": float((ranks_arr <= 10).mean()),
                "p_dnf": float(dnf_arr.mean()),
                "mean_finish_position": float(ranks_arr.mean()),
                "median_finish_position": float(np.median(ranks_arr)),
                "p10_finish_position": float(np.percentile(ranks_arr, 10)),
                "p90_finish_position": float(np.percentile(ranks_arr, 90)),
            })

        return pd.DataFrame(results).sort_values("mean_finish_position").reset_index(drop=True)
