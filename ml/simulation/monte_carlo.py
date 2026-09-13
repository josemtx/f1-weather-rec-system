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

from ml.features.build_features import CATEGORICAL_COLS
from ml.simulation.randomness_models import (
    DEFAULT_PIT_STD,
    climate_variability,
    rain_probability,
    safety_car_shock_scale,
)
from ml.training.anchor import (
    REGIME_POST_QUALI,
    REGIME_PRE_QUALI,
    anchor_for,
    blank_quali_features,
    fill_grid_from_quali,
    regime_for,
)
from ml.training.model_registry import (
    apply_calibration,
    get_production_dir,
    load_calibration,
    load_xgb_classifier,
    load_xgb_regressor,
)

logger = logging.getLogger(__name__)

# Respaldo si el modelo no trae metrics.json (modelos antiguos): orden de
# magnitud del error tipico observado, ~3-4 posiciones.
DEFAULT_RESIDUAL_STD = 3.5

# El ruido inyectado NO es el error medido en bruto, sino una fraccion.
#
# Motivo: al ruido se le suma el efecto de REORDENAR. El error del modelo ya
# incluye que su orden es imperfecto; si ademas se perturba el score y se
# vuelve a ordenar, la dispersion se cuenta dos veces. Medido sobre las 479
# predicciones de 2025 con el regresor anclado (el intervalo P10-P90 deberia
# cubrir el 80% de los resultados reales), por regimen:
#
#   post_quali (sigma 3.45)        pre_quali (sigma 4.01)
#     escala  cobertura  +-3         escala  cobertura  +-3
#       1.00    86.2%   53.4%          1.00    82.7%   44.3%
#       0.65    82.5%   59.1%          0.80    80.8%   45.3%   <- calibrado
#       0.50    80.2%   60.3%  <-      0.65    78.3%   46.1%
#       0.40    77.7%   60.1%          0.50    74.9%   47.4%   <- demasiado seguro
#
# Reejecutable con ml/training/calibrate_simulation.py [version] [pre_quali].
RESIDUAL_SCALE = {REGIME_POST_QUALI: 0.5, REGIME_PRE_QUALI: 0.8}


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
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
        self.residual_std = float(
            metrics.get("finish_position", {}).get("residual_std_val_2025", DEFAULT_RESIDUAL_STD)
        )

        # Regresor anclado (ml/training/anchor.py): predice deltas frente a
        # la parrilla (o a la forma reciente si aun no hay clasificacion), con
        # un regresor y una incertidumbre por regimen. Los modelos anteriores
        # al 2026-09-12 predicen la posicion absoluta con un solo regresor.
        self.anchored = metrics.get("regressor", {}).get("target") == "delta_from_anchor"
        self.reg_pre_quali = None
        self.residual_std_pre_quali = self.residual_std
        if self.anchored:
            self.reg_pre_quali = load_xgb_regressor(model_dir / "finish_position_regressor_pre_quali.json")
            self.residual_std_pre_quali = float(
                metrics.get("finish_position_pre_quali", {}).get("residual_std_val_2025", self.residual_std)
            )
        logger.info(
            f"Incertidumbre del modelo (desviacion del error): {self.residual_std:.2f} posiciones"
            + (f" post_quali / {self.residual_std_pre_quali:.2f} pre_quali" if self.anchored else "")
        )

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
                # Se anulan explicitamente antes de tipar: dejar que
                # pd.Categorical las descarte por si solo esta deprecado y
                # pasara a lanzar excepcion, ademas de ocultar la intencion.
                df[col] = df[col].where(df[col].isin(known))
                df[col] = pd.Categorical(df[col], categories=known)
            else:
                df[col] = df[col].astype("category")
        return df

    def _predict_scores(
        self, race_features: pd.DataFrame, X: pd.DataFrame, n_simulations: int
    ) -> tuple[np.ndarray, float]:
        """Posicion esperada por fila de X (parrilla apilada n_simulations veces)
        y la sigma del ruido residual a inyectar (incertidumbre del regresor
        que la produjo, ya escalada)."""
        if not self.anchored:
            return self.reg.predict(X), self.residual_std * RESIDUAL_SCALE[REGIME_POST_QUALI]

        regime = regime_for(race_features)
        if regime == REGIME_POST_QUALI:
            reg, residual_std = self.reg, self.residual_std
        else:
            reg, residual_std = self.reg_pre_quali, self.residual_std_pre_quali
        anchor = np.tile(anchor_for(race_features, regime).to_numpy(dtype=float), n_simulations)
        return reg.predict(blank_quali_features(X, regime)) + anchor, residual_std * RESIDUAL_SCALE[regime]

    def _rain_probability(self, race_features: pd.DataFrame, circuit_short_name: str, month: int) -> float:
        """La del pronostico del dia (`pop` de OpenWeatherMap, carreras futuras)
        si la fila la trae; si no, la climatologia del circuito para ese mes."""
        if "forecast_rain_probability" in race_features.columns:
            values = pd.to_numeric(race_features["forecast_rain_probability"], errors="coerce").dropna()
            if not values.empty:
                logger.info(f"Lluvia muestreada del pronostico: p={float(values.max()):.2f}")
                return float(values.max())
        return rain_probability(self.db, circuit_short_name, month)

    @staticmethod
    def _perturb(X: pd.DataFrame, rng, n_simulations: int, n_drivers: int,
                 climate_std: dict, rain_p: float, pit_std: np.ndarray) -> np.ndarray:
        """Perturba in situ la parrilla apilada con las fuentes de aleatoriedad del
        mundo y devuelve, por fila, si esa simulacion es una carrera con lluvia.

        Clima compartido por todos los pilotos de una misma simulacion (es la
        misma carrera); jitter de boxes por piloto y simulacion, con la
        variabilidad historica de SU equipo."""
        def per_sim(values):
            return np.repeat(values, n_drivers)

        temp_noise = per_sim(rng.normal(0, climate_std["temp_2m_avg"], size=n_simulations))
        for col in ("race_day_temp_avg", "race_day_temp_max"):
            if col in X.columns:
                X[col] = X[col].to_numpy(dtype=float) + temp_noise

        rain_rows = per_sim(rng.random(n_simulations) < rain_p)
        if "race_day_precipitation_mm" in X.columns:
            rain_mm = per_sim(rng.exponential(3.0, size=n_simulations))
            current = X["race_day_precipitation_mm"].to_numpy(dtype=float)
            X["race_day_precipitation_mm"] = np.where(rain_rows, np.maximum(current, rain_mm), current)
            if "climate_rain_probability_flag" in X.columns:
                flag = X["climate_rain_probability_flag"].to_numpy(dtype=float)
                X["climate_rain_probability_flag"] = np.where(rain_rows, 1.0, flag)

        if "race_day_humidity" in X.columns:
            hum_noise = per_sim(rng.normal(0, climate_std["humidity_relative"], size=n_simulations))
            X["race_day_humidity"] = X["race_day_humidity"].to_numpy(dtype=float) + hum_noise
        if "race_day_wind_speed" in X.columns:
            wind_noise = per_sim(rng.normal(0, climate_std["wind_speed_10m"], size=n_simulations))
            X["race_day_wind_speed"] = np.maximum(0.0, X["race_day_wind_speed"].to_numpy(dtype=float) + wind_noise)

        if "team_avg_pit_stop_duration_last5" in X.columns:
            std_per_row = np.tile(np.array([s if s and s > 0 else DEFAULT_PIT_STD for s in pit_std], dtype=float), n_simulations)
            X["team_avg_pit_stop_duration_last5"] = (
                X["team_avg_pit_stop_duration_last5"].to_numpy(dtype=float) + rng.normal(0.0, std_per_row)
            )
        return rain_rows

    @staticmethod
    def _summarise(driver_codes: np.ndarray, ranks: np.ndarray, dnf_matrix: np.ndarray) -> pd.DataFrame:
        results = []
        for i, code in enumerate(driver_codes):
            ranks_arr, dnf_arr = ranks[:, i], dnf_matrix[:, i]
            # Rango CONDICIONADO A TERMINAR. Mezclar abandonos dentro del
            # intervalo lo vuelve inutil: un 13% de abandono arrastra el P90 al
            # fondo de la parrilla y "entre P1 y P18" no dice nada. Separar
            # "donde acaba si termina" de "probabilidad de no terminar"
            # informa mucho mas con los mismos datos.
            finished = ranks_arr[~dnf_arr]
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
                "median_if_finishes": float(np.median(finished)) if finished.size else float("nan"),
                "p10_if_finishes": float(np.percentile(finished, 10)) if finished.size else float("nan"),
                "p90_if_finishes": float(np.percentile(finished, 90)) if finished.size else float("nan"),
            })
        return pd.DataFrame(results).sort_values("mean_finish_position").reset_index(drop=True)

    def simulate_race(self, race_features: pd.DataFrame, n_simulations: int = 2000, seed: int = 42) -> pd.DataFrame:
        """race_features: una fila por piloto con al menos self.feature_cols +
        driver_code + circuit_short_name + circuit_type + race_date. Devuelve
        un DataFrame con probabilidades agregadas por piloto tras N simulaciones."""
        race_features = race_features.reset_index(drop=True)
        # Los codigos se capturan ANTES de fijar categorias: un debutante se
        # convierte en NaN para el modelo, pero el resultado debe seguir
        # identificandolo por su nombre.
        driver_codes = race_features["driver_code"].to_numpy()
        race_features = fill_grid_from_quali(self._prep_categoricals(race_features))
        n_drivers = len(race_features)
        total = n_simulations * n_drivers
        rng = np.random.default_rng(seed)

        circuit_short_name = race_features["circuit_short_name"].iloc[0]
        month = pd.to_datetime(race_features["race_date"].iloc[0]).month
        climate_std = climate_variability(self.db, circuit_short_name, month)
        sc_scale = safety_car_shock_scale(race_features["circuit_type"].iloc[0])
        rain_p = self._rain_probability(race_features, circuit_short_name, month)

        X_base = race_features[self.feature_cols]
        base_dnf_proba = apply_calibration(self.dnf_clf.predict_proba(X_base)[:, 1], self.dnf_cal["a"], self.dnf_cal["b"])
        pit_std = (
            race_features["team_pit_stop_duration_stddev_last5"].to_numpy()
            if "team_pit_stop_duration_stddev_last5" in race_features.columns
            else np.full(n_drivers, np.nan)
        )

        # Todas las simulaciones en UNA llamada al modelo: se apilan
        # n_simulations copias de la parrilla (piloto cicla rapido, simulacion
        # lento: fila i -> sim i // n_drivers) y se perturban vectorizadas.
        # En bucle eran ~80 s; asi ~2 s, lo que hace viable la rejilla del sandbox.
        X = X_base.iloc[np.tile(np.arange(n_drivers), n_simulations)].reset_index(drop=True)
        rain_rows = self._perturb(X, rng, n_simulations, n_drivers, climate_std, rain_p, pit_std)

        dnf_flat = rng.random(total) < np.tile(base_dnf_proba, n_simulations)
        # Shock adicional de fiabilidad bajo lluvia en circuitos propensos a
        # SC/VSC (simplificacion documentada, no un modelo de SC por vuelta).
        dnf_flat |= (rng.random(total) < (0.05 * sc_scale)) & rain_rows

        scores, residual_sigma = self._predict_scores(race_features, X, n_simulations)
        # Ruido residual: lo que el modelo NO sabe, calibrado con la
        # dispersion real de su error en validacion.
        scores = (scores + rng.normal(0.0, residual_sigma, size=total)).reshape(n_simulations, n_drivers)
        dnf_matrix = dnf_flat.reshape(n_simulations, n_drivers)

        worst = np.maximum(scores.max(axis=1, keepdims=True), n_drivers) + 10.0
        scores = np.where(dnf_matrix, worst + rng.random((n_simulations, n_drivers)), scores)
        order = np.argsort(scores, axis=1)
        ranks = np.empty_like(order)
        np.put_along_axis(ranks, order, np.arange(1, n_drivers + 1), axis=1)

        return self._summarise(driver_codes, ranks, dnf_matrix)
