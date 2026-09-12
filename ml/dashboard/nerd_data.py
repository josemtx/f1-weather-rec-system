"""Construye el payload de la pestana "Nerd Data": el feature matrix crudo,
sin resumir, tal cual lo ve el modelo -- para el lector que quiere ver todos
los numeros, no solo la conclusion.

Cada feature se etiqueta a mano una vez aqui (unica fuente de verdad), en
vez de que el JS del dashboard tenga que adivinar formato/unidad a partir
del nombre de columna.
"""

import numpy as np
import pandas as pd

# (columna, etiqueta, formato) -- formato: "pct" (0-1 -> %), "num" (decimal),
# "int", "text". Agrupadas en el mismo orden que las categorias A-J del plan.
FEATURE_GROUPS: list[tuple[str, list[tuple[str, str, str]]]] = [
    ("Circuito", [
        ("circuit_type", "Tipo de circuito", "text"),
        ("overtaking_difficulty", "Dificultad de adelantamiento", "text"),
        ("altitude_m", "Altitud", "int_m"),
        ("length_km", "Longitud de vuelta", "num_km"),
        ("circuit_avg_dnf_rate_historical", "Tasa histórica de abandono", "pct"),
        ("circuit_avg_safety_car_proxy", "Propensión a safety car (proxy)", "pct"),
        ("circuit_pit_loss_time_estimate", "Pérdida de tiempo en boxes", "num_s"),
        ("circuit_typical_num_stops", "Nº de paradas habitual", "num"),
    ]),
    ("Forma del piloto", [
        ("driver_avg_finish_last5", "Media de posición, últimas 5", "num"),
        ("driver_avg_finish_last10", "Media de posición, últimas 10", "num"),
        ("driver_avg_finish_season_to_date", "Media de posición esta temporada", "num"),
        ("driver_podium_rate_last10", "Tasa de podio, últimas 10", "pct"),
        ("driver_dnf_rate_last10", "Tasa de abandono, últimas 10", "pct"),
        ("driver_points_last5", "Puntos, últimas 5", "num"),
        ("driver_avg_grid_to_finish_delta_last10", "Ganancia media parrilla→meta", "num"),
        ("driver_circuit_history_avg_finish", "Media histórica en este circuito", "num"),
        ("driver_teammate_h2h_rate_last10", "Cara a cara vs compañero, últimas 10", "pct"),
        ("driver_career_wins_to_date", "Victorias en carrera", "int"),
        ("driver_n_prior_races", "Carreras previas conocidas", "int"),
    ]),
    ("Forma del equipo", [
        ("team_avg_finish_last5", "Media de posición, últimas 5", "num"),
        ("team_avg_finish_last10", "Media de posición, últimas 10", "num"),
        ("team_podium_rate_last10", "Tasa de podio, últimas 10", "pct"),
        ("team_dnf_rate_last10", "Tasa de abandono, últimas 10", "pct"),
        ("team_points_last5", "Puntos, últimas 5", "num"),
        ("team_circuit_history_avg_finish", "Media histórica en este circuito", "num"),
    ]),
    ("Clima de carrera", [
        ("climate_source", "Fuente del dato", "text"),
        ("race_day_temp_avg", "Temperatura del aire", "num_c"),
        ("race_day_temp_max", "Temperatura máxima", "num_c"),
        ("race_day_track_temp", "Temperatura de asfalto", "num_c"),
        ("race_day_humidity", "Humedad relativa", "pct100"),
        ("race_day_wind_speed", "Viento", "num_ms"),
        ("race_day_precipitation_mm", "Precipitación", "num_mm"),
        ("race_day_wet_fraction", "Fracción de sesión en mojado (sensores)", "pct"),
        ("forecast_rain_probability", "Probabilidad de lluvia (pronóstico)", "pct"),
        ("circuit_seasonal_climate_norm_temp", "Norma estacional del circuito", "num_c"),
        ("climate_temp_delta_vs_norm", "Desviación vs. norma estacional", "num_c_signed"),
    ]),
    ("Neumáticos y estrategia", [
        ("driver_avg_num_stops_last5", "Paradas medias, últimas 5", "num"),
        ("driver_preferred_compound_share_last5", "Cuota del compuesto preferido", "pct"),
        ("driver_tyre_degradation_rate_last5", "Tasa de degradación", "num_sig"),
        ("team_avg_pit_stop_duration_last5", "Duración media de parada", "num_s"),
        ("team_pit_stop_duration_stddev_last5", "Variabilidad de parada", "num_s"),
    ]),
    ("Ritmo de carrera", [
        ("driver_avg_pace_percentile_last5", "Percentil de ritmo vs. campo", "pct"),
        ("driver_pace_consistency_stddev_last5", "Consistencia (desv. típica)", "num_s"),
        ("team_race_pace_trend_last5", "Tendencia de ritmo del equipo", "num_sig"),
    ]),
    ("Clasificación", [
        ("driver_quali_position", "Posición de clasificación", "int"),
        ("grid_penalty_positions", "Penalización de parrilla", "int_signed"),
        ("quali_gap_to_pole_pct", "Gap a la pole", "pct_signed"),
        ("driver_avg_quali_position_last5", "Media clasificación, últimas 5", "num"),
        ("driver_avg_quali_position_last10", "Media clasificación, últimas 10", "num"),
        ("driver_circuit_quali_history_avg_position", "Media histórica en este circuito", "num"),
        ("n_quali_laps", "Vueltas lanzadas en Q", "int"),
        ("quali_soft_tyre_share", "Cuota de blandos en Q", "pct"),
        ("quali_day_temp_avg", "Temperatura en clasificación", "num_c"),
        ("quali_day_precipitation_mm", "Precipitación en clasificación", "num_mm"),
        ("conditions_delta_quali_to_race", "Cambio de temperatura Q→carrera", "num_c_signed"),
        ("conditions_rain_changed", "¿Cambió mojado↔seco Q→carrera?", "bool"),
    ]),
    ("Sprint", [
        ("weekend_has_sprint", "Fin de semana con sprint", "bool"),
        ("driver_sprint_position_this_weekend", "Posición en el sprint (este finde)", "int"),
        ("driver_avg_sprint_finish_last5", "Media en sprint, últimos 5", "num"),
        ("driver_sprint_podium_rate_last5", "Tasa de podio en sprint", "pct"),
        ("team_avg_sprint_finish_last5", "Media del equipo en sprint", "num"),
        ("driver_avg_sprint_num_stops_last5", "Paradas medias en sprint", "num"),
        ("driver_avg_sprint_pace_percentile_last5", "Percentil de ritmo en sprint", "pct"),
    ]),
    ("Interacción con el clima", [
        ("driver_avg_finish_in_wet", "Media de posición en mojado", "num"),
        ("driver_avg_finish_in_dry", "Media de posición en seco", "num"),
        ("driver_wet_skill_delta", "Delta de habilidad en mojado", "num_signed"),
        ("driver_n_wet_races", "Carreras en mojado (histórico)", "int"),
        ("driver_avg_finish_in_hot", "Media de posición con calor", "num"),
        ("driver_hot_skill_delta", "Delta de habilidad con calor", "num_signed"),
        ("team_avg_finish_in_wet", "Media del equipo en mojado", "num"),
        ("team_wet_skill_delta", "Delta de habilidad del equipo en mojado", "num_signed"),
    ]),
]


def _fmt(value, kind: str):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if kind in ("text",):
        return str(value)
    if kind == "bool":
        return bool(value)
    if kind in ("int", "int_m"):
        return int(round(value))
    if kind == "int_signed":
        return int(round(value))
    if kind in ("num", "num_signed", "num_sig", "num_km", "num_s", "num_ms", "num_mm", "num_c", "num_c_signed"):
        return round(float(value), 3)
    if kind in ("pct", "pct_signed"):
        return round(float(value) * 100, 2)
    if kind == "pct100":
        return round(float(value), 1)
    return value


def build_nerd_data_payload(race: pd.DataFrame) -> dict:
    """race: filas de features de UNA carrera (una por piloto). Devuelve
    {groups: [...], drivers: {code: {feature: value_formateado}}}."""
    groups_meta = [
        {"group": g, "features": [{"key": k, "label": lbl, "kind": kind} for k, lbl, kind in feats]}
        for g, feats in FEATURE_GROUPS
    ]

    drivers = {}
    for _, row in race.iterrows():
        code = row["driver_code"]
        values = {}
        for _, feats in FEATURE_GROUPS:
            for key, _, kind in feats:
                values[key] = _fmt(row.get(key), kind)
        drivers[code] = values

    return {"groups": groups_meta, "drivers": drivers}
