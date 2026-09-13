"""Ancla del regresor: la mejor estimacion "gratis" de la posicion final.

El regresor no predice la posicion absoluta sino la DIFERENCIA respecto a un
ancla. Motivo medido (residual_diagnosis.py, 2026-09-12): con
~900 filas los arboles no aprenden bien la identidad parrilla -> llegada, y
el modelo que predecia posicion absoluta no mejoraba a "predecir = parrilla"
(2024: 3.47 vs 2.91; 2025: 3.36 vs 3.34). Regalandole el ancla, el suelo del
modelo pasa a ser el ancla y solo tiene que aprender quien gana o pierde
puestos respecto a ella.

Dos regimenes, dos anclas y dos regresores (uno que aprendio deltas frente a
la parrilla no sirve cuando la parrilla no existe):

  post_quali : parrilla. Para una carrera futura la parrilla oficial no
               existe hasta el domingo, asi que se usa la posicion de
               clasificacion (difiere solo por sanciones). Salida desde el
               pit lane (grid 0) o sin tiempo -> ultimo.
  pre_quali  : ritmo reciente del piloto (media de sus 5 ultimas llegadas
               SOLO cuando termino: un abandono clasifica P20 y no dice nada
               de su ritmo; la fiabilidad la pone el dado del DNF aparte),
               la media con abandonos si nunca termino, la de su equipo si
               es debutante, mitad de parrilla si nada.
"""

import numpy as np
import pandas as pd

N_DRIVERS = 20
REGIME_POST_QUALI = "post_quali"
REGIME_PRE_QUALI = "pre_quali"

# Features que solo existen una vez disputada la clasificacion del sabado.
# Se ponen en blanco para entrenar/predecir en regimen pre_quali.
QUALI_FEATURES = [
    "grid_position", "driver_quali_position", "quali_gap_to_pole_pct",
    "grid_penalty_positions", "n_quali_laps", "quali_soft_tyre_share",
    "quali_day_temp_avg", "quali_day_precipitation_mm", "quali_day_rain_flag",
    "conditions_delta_quali_to_race", "conditions_rain_changed",
]


def _quali_info(df: pd.DataFrame) -> pd.Series:
    """Parrilla o, en su defecto, posicion de clasificacion. NaN = no hay."""
    grid = pd.to_numeric(df["grid_position"], errors="coerce")
    grid = grid.where(grid != 0, N_DRIVERS)
    quali = pd.to_numeric(df["driver_quali_position"], errors="coerce")
    return grid.fillna(quali)


def regime_for(df: pd.DataFrame) -> str:
    """El regimen lo deciden los datos: hay clasificacion o no la hay."""
    return REGIME_POST_QUALI if _quali_info(df).notna().any() else REGIME_PRE_QUALI


def anchor_for(df: pd.DataFrame, regime: str) -> pd.Series:
    if regime == REGIME_POST_QUALI:
        return _quali_info(df).fillna(N_DRIVERS)
    pace = pd.to_numeric(df["driver_avg_finish_when_finished_last5"], errors="coerce")
    form = pd.to_numeric(df["driver_avg_finish_last5"], errors="coerce")
    team = pd.to_numeric(df["team_avg_finish_last5"], errors="coerce")
    return pace.fillna(form).fillna(team).fillna(N_DRIVERS / 2)


def fill_grid_from_quali(df: pd.DataFrame) -> pd.DataFrame:
    """Parrilla estimada para una carrera futura: la posicion de clasificacion.

    Los modelos se entrenan con `grid_position` siempre presente, pero para
    una carrera que aun no se ha corrido la parrilla oficial no existe hasta
    el domingo y la fila fantasma la trae en NaN. Un NaN que el modelo nunca
    vio en entrenamiento cae por la rama por defecto de cada arbol: medido
    sobre 2025, una prediccion post-clasificacion con la parrilla en NaN
    rinde como si no hubiera clasificacion (MAE 3.79 vs 2.49). Rellenarla
    con la clasificacion (difieren solo por sanciones) devuelve 2.46.
    Pre-clasificacion ambas son NaN y se queda como esta.
    """
    if df["grid_position"].notna().all():
        return df
    df = df.copy()
    missing = df["grid_position"].isna()
    df.loc[missing, "grid_position"] = pd.to_numeric(df.loc[missing, "driver_quali_position"], errors="coerce")
    if "grid_penalty_positions" in df.columns:
        df.loc[missing & df["grid_position"].notna(), "grid_penalty_positions"] = 0.0
    return df


def blank_quali_features(X: pd.DataFrame, regime: str) -> pd.DataFrame:
    if regime == REGIME_POST_QUALI:
        return X
    X = X.copy()
    for col in QUALI_FEATURES:
        if col in X.columns:
            X[col] = np.nan
    return X
