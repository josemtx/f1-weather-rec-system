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
    DEFAULT_PIT_STD,
    climate_variability,
    rain_probability,
    safety_car_shock_scale,
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

# El ruido inyectado NO es el error medido en bruto, sino la mitad.
#
# Motivo: al ruido se le suma el efecto de REORDENAR. El error del modelo ya
# incluye que su orden es imperfecto; si ademas se perturba el score y se
# vuelve a ordenar, la dispersion se cuenta dos veces. Medido sobre las 479
# predicciones de 2025 (el intervalo P10-P90 deberia cubrir el 80% de los
# resultados reales):
#
#     escala   sigma   cobertura   anchura   aciertos +-3
#       1.00    3.46      88.1%      13.0        50.3%     <- inflado
#       0.65    2.25      84.1%      11.9        54.5%
#       0.50    1.73      80.8%      11.4        56.6%     <- calibrado
#       0.40    1.39      77.7%      10.9        56.6%     <- demasiado seguro
#
# Reejecutable con ml/training/calibrate_simulation.py.
RESIDUAL_SCALE = 0.5


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
                # Se anulan explicitamente antes de tipar: dejar que
                # pd.Categorical las descarte por si solo esta deprecado y
                # pasara a lanzar excepcion, ademas de ocultar la intencion.
                df[col] = df[col].where(df[col].isin(known))
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
        race_features = race_features.reset_index(drop=True)
        # Los codigos se capturan ANTES de fijar categorias: _prep_categoricals
        # convierte en NaN las no vistas en entrenamiento, y driver_code es una
        # de ellas. El modelo debe recibir NaN (no conoce a un debutante), pero
        # el resultado tiene que seguir identificandolo por su nombre.
        driver_codes = race_features["driver_code"].to_numpy()
        race_features = self._prep_categoricals(race_features)
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

        pit_std = (
            race_features["team_pit_stop_duration_stddev_last5"].values
            if "team_pit_stop_duration_stddev_last5" in race_features.columns
            else np.full(n_drivers, np.nan)
        )

        # Todas las simulaciones se resuelven en UNA llamada al modelo en vez
        # de una por simulacion: se apilan n_simulations copias de la parrilla
        # y se perturban de forma vectorizada. La version en bucle hacia 5000
        # predicciones de 22 filas y tardaba ~80s; asi son ~2s, que es lo que
        # hace viable precalcular la rejilla de escenarios del sandbox.
        #
        # Orden de las filas: el indice de piloto cicla rapido y el de
        # simulacion lento, asi que fila i -> sim = i // n_drivers.
        row_index = np.tile(np.arange(n_drivers), n_simulations)
        X = X_base.iloc[row_index].reset_index(drop=True)
        total = n_simulations * n_drivers

        def per_sim(values):
            """Un valor por simulacion, repetido para todos sus pilotos."""
            return np.repeat(values, n_drivers)

        # Clima: compartido por todos los pilotos de una misma simulacion,
        # porque es la misma carrera.
        temp_noise = per_sim(rng.normal(0, climate_std["temp_2m_avg"], size=n_simulations))
        if "race_day_temp_avg" in X.columns:
            X["race_day_temp_avg"] = X["race_day_temp_avg"].to_numpy(dtype=float) + temp_noise
        if "race_day_temp_max" in X.columns:
            X["race_day_temp_max"] = X["race_day_temp_max"].to_numpy(dtype=float) + temp_noise

        is_rain_sim = rng.random(n_simulations) < rain_p
        rain_rows = per_sim(is_rain_sim)
        if "race_day_precipitation_mm" in X.columns:
            rain_mm = per_sim(rng.exponential(3.0, size=n_simulations))
            current = X["race_day_precipitation_mm"].to_numpy(dtype=float)
            X["race_day_precipitation_mm"] = np.where(
                rain_rows, np.maximum(current, rain_mm), current
            )
            if "climate_rain_probability_flag" in X.columns:
                flag = X["climate_rain_probability_flag"].to_numpy(dtype=float)
                X["climate_rain_probability_flag"] = np.where(rain_rows, 1.0, flag)

        if "race_day_humidity" in X.columns:
            hum_noise = per_sim(rng.normal(0, climate_std["humidity_relative"], size=n_simulations))
            X["race_day_humidity"] = X["race_day_humidity"].to_numpy(dtype=float) + hum_noise
        if "race_day_wind_speed" in X.columns:
            wind_noise = per_sim(rng.normal(0, climate_std["wind_speed_10m"], size=n_simulations))
            X["race_day_wind_speed"] = np.maximum(
                0.0, X["race_day_wind_speed"].to_numpy(dtype=float) + wind_noise
            )

        # Estrategia: el jitter de boxes es por piloto y simulacion, con la
        # variabilidad historica de SU equipo.
        if "team_avg_pit_stop_duration_last5" in X.columns:
            std_per_row = np.tile(
                np.array([s if s and s > 0 else DEFAULT_PIT_STD for s in pit_std], dtype=float),
                n_simulations,
            )
            jitter = rng.normal(0.0, std_per_row)
            X["team_avg_pit_stop_duration_last5"] = (
                X["team_avg_pit_stop_duration_last5"].to_numpy(dtype=float) + jitter
            )

        dnf_flat = rng.random(total) < np.tile(base_dnf_proba, n_simulations)
        # Shock adicional de fiabilidad bajo lluvia en circuitos propensos a
        # SC/VSC (simplificacion documentada, no un modelo de SC por vuelta).
        extra_dnf = (rng.random(total) < (0.05 * sc_scale)) & rain_rows
        dnf_flat = dnf_flat | extra_dnf

        scores = self.reg.predict(X)
        # Ruido residual: lo que el modelo NO sabe, calibrado con la
        # dispersion real de su error en validacion.
        scores = scores + rng.normal(0.0, self.residual_std * RESIDUAL_SCALE, size=total)

        scores = scores.reshape(n_simulations, n_drivers)
        dnf_matrix = dnf_flat.reshape(n_simulations, n_drivers)

        worst = np.maximum(scores.max(axis=1, keepdims=True), n_drivers) + 10.0
        scores = np.where(dnf_matrix, worst + rng.random((n_simulations, n_drivers)), scores)

        order = np.argsort(scores, axis=1)
        ranks = np.empty_like(order)
        np.put_along_axis(ranks, order, np.arange(1, n_drivers + 1), axis=1)

        results = []
        for i, code in enumerate(driver_codes):
            ranks_arr = ranks[:, i]
            dnf_arr = dnf_matrix[:, i]

            # Rango CONDICIONADO A TERMINAR. Mezclar abandonos dentro del
            # intervalo lo vuelve inutil: un piloto con 13% de abandono
            # arrastra el percentil 90 al fondo de la parrilla, y el
            # resultado ("entre P1 y P18") no dice nada. Separar las dos
            # preguntas -- donde acaba si termina, y que probabilidad hay de
            # que no termine -- informa mucho mas con los mismos datos.
            finished = ranks_arr[~dnf_arr]
            if finished.size:
                p10_fin = float(np.percentile(finished, 10))
                p90_fin = float(np.percentile(finished, 90))
                median_fin = float(np.median(finished))
            else:
                p10_fin = p90_fin = median_fin = float("nan")

            results.append({
                "p10_if_finishes": p10_fin,
                "p90_if_finishes": p90_fin,
                "median_if_finishes": median_fin,
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
