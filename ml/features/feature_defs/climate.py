"""Categoria E: clima.

Dos mecanismos distintos que conviene no confundir:

1. COALESCENCIA DE FUENTES OBSERVADAS (`coalesce_climate`)
   climate_historical guarda NASA POWER y ERA5 como filas independientes
   (`source` forma parte del indice unico), NUNCA se sobreescriben entre si.
   Aqui, y solo aqui, se combinan: ERA5 gana cuando ambas existen para el
   mismo circuito+fecha. Ver ml/tests/test_climate_coalesce.py.

2. CADENA DE RESPALDO POR HORIZONTE (`build_climate`)
   Una carrera futura no tiene clima observado. El orden de preferencia es:

     observado (climate_historical)  ->  carreras ya disputadas
     pronostico (forecast_data)      ->  proximos ~5 dias, OpenWeatherMap
     climatologia (norma del circuito en ese mes)  ->  mas alla del pronostico

   La columna `climate_source` deja constancia de cual se uso en cada fila:
   no es una feature del modelo (en entrenamiento seria constante), sirve
   para poder decir en el dashboard "esto se predijo con pronostico, no con
   datos reales".
"""

import logging

import pandas as pd

from ml.features.feature_defs._openf1_join import sessions_lookup
from ml.features.feature_defs.track_temp_model import estimate_track_temp, fit_air_to_track

logger = logging.getLogger(__name__)

RAIN_THRESHOLD_MM = 1.0
_SOURCE_PRIORITY = {"era5": 0, "nasa_power": 1}

# Ventana alrededor de la hora de carrera para promediar el pronostico, que
# viene en tramos de 3 horas. +-3h cubre el tramo anterior y el posterior.
FORECAST_WINDOW_HOURS = 3
DEFAULT_RACE_TIME = "14:00:00Z"

CLIMATE_COLS = [
    "race_day_temp_avg", "race_day_temp_max", "race_day_precipitation_mm",
    "race_day_humidity", "race_day_wind_speed", "climate_rain_probability_flag",
    "circuit_seasonal_climate_norm_temp", "climate_temp_delta_vs_norm",
    "race_day_track_temp", "race_day_wet_fraction",
    "forecast_rain_probability", "climate_source",
]


def coalesce_climate(raw_climate_df: pd.DataFrame) -> pd.DataFrame:
    """Una fila por circuito+fecha, con ERA5 preferido sobre NASA POWER."""
    if raw_climate_df.empty:
        return raw_climate_df

    df = raw_climate_df.copy()
    df["_priority"] = df["source"].map(_SOURCE_PRIORITY).fillna(99)
    df = df.sort_values(["circuit_short_name", "date", "_priority"])
    coalesced = df.drop_duplicates(subset=["circuit_short_name", "date"], keep="first")
    return coalesced.drop(columns=["_priority"])


def session_sensor_weather(db, session_names: list[str] = ("Race",)) -> pd.DataFrame:
    """Clima medido por los sensores de pista de OpenF1, agregado por sesion.

    Es la mejor fuente disponible para 2023+ y supera al reanalisis (NASA
    POWER / ERA5) en tres cosas que importan:

      - Se mide EN el circuito y DURANTE la sesion, no es una media diaria
        de una celda de rejilla. La precipitacion diaria puede registrar un
        chaparron de madrugada mientras la carrera de las 15:00 fue en seco.
      - Incluye temperatura de ASFALTO, la variable que de verdad gobierna
        el comportamiento del neumatico, y que ningun servicio meteorologico
        proporciona.
      - `rainfall` es 0/1 por lectura, asi que su media da la FRACCION de
        sesion en mojado: una carrera que empieza seca y acaba pasada por
        agua deja de ser un simple si/no.
    """
    rows = list(db["weather"].find({}, {"_id": 0}))
    if not rows:
        return pd.DataFrame()

    weather = pd.DataFrame(rows)
    sessions = sessions_lookup(db, session_names)
    if sessions.empty:
        return pd.DataFrame()

    merged = weather.merge(sessions, on="session_key", how="inner")
    if merged.empty:
        return pd.DataFrame()

    agg = merged.groupby(["circuit_short_name", "race_date"]).agg(
        sensor_air_temp=("air_temperature", "mean"),
        sensor_air_temp_max=("air_temperature", "max"),
        sensor_track_temp=("track_temperature", "mean"),
        sensor_track_temp_max=("track_temperature", "max"),
        sensor_humidity=("humidity", "mean"),
        sensor_wind_speed=("wind_speed", "mean"),
        sensor_wet_fraction=("rainfall", "mean"),
    ).reset_index()
    return agg


def _race_datetimes(db, context_df: pd.DataFrame) -> pd.Series:
    """Momento exacto de cada carrera (UTC). La hora importa: Las Vegas y
    Singapur son nocturnas, asi que promediar el dia entero falsearia el
    clima de carrera."""
    schedule = list(db["race_schedule"].find({}, {"_id": 0, "year": 1, "round": 1, "race_time": 1}))
    if not schedule:
        times = pd.Series(DEFAULT_RACE_TIME, index=context_df.index)
    else:
        sched_df = pd.DataFrame(schedule)
        merged = context_df[["year", "round"]].merge(sched_df, on=["year", "round"], how="left")
        times = merged["race_time"].fillna(DEFAULT_RACE_TIME)
        times.index = context_df.index

    date_str = context_df["race_date"].dt.strftime("%Y-%m-%d")
    return pd.to_datetime(date_str + " " + times.str.replace("Z", "", regex=False), errors="coerce")


def _forecast_for_races(db, context_df: pd.DataFrame, race_datetimes: pd.Series) -> pd.DataFrame:
    """Promedia los tramos de pronostico cercanos a la hora de carrera.
    Devuelve un DataFrame alineado al indice de context_df (NaN donde no hay
    pronostico disponible, p.ej. carreras a mas de 5 dias vista)."""
    rows = list(db["forecast_data"].find({}, {"_id": 0}))
    empty = pd.DataFrame(index=context_df.index, columns=[
        "temp_2m_avg", "temp_2m_max", "precipitation_mm",
        "humidity_relative", "wind_speed_10m", "precipitation_probability",
    ], dtype="float64")
    if not rows:
        return empty

    fc = pd.DataFrame(rows)
    fc["datetime"] = pd.to_datetime(fc["datetime"])

    window = pd.Timedelta(hours=FORECAST_WINDOW_HOURS)
    result = empty.copy()

    # Solo se buscan las filas que realmente pueden tener pronostico (fecha
    # dentro del rango descargado): evita recorrer 3.744 filas historicas.
    fc_min, fc_max = fc["datetime"].min() - window, fc["datetime"].max() + window
    candidates = context_df.index[(race_datetimes >= fc_min) & (race_datetimes <= fc_max)]

    for idx in candidates:
        circuit = context_df.at[idx, "circuit_short_name"]
        target = race_datetimes.at[idx]
        if pd.isna(target):
            continue
        near = fc[
            (fc["circuit_short_name"] == circuit)
            & (fc["datetime"] >= target - window)
            & (fc["datetime"] <= target + window)
        ]
        if near.empty:
            continue
        result.loc[idx, "temp_2m_avg"] = near["temp_2m_avg"].mean()
        result.loc[idx, "temp_2m_max"] = near["temp_2m_max"].max()
        result.loc[idx, "precipitation_mm"] = near["precipitation_mm"].sum()
        result.loc[idx, "humidity_relative"] = near["humidity_relative"].mean()
        result.loc[idx, "wind_speed_10m"] = near["wind_speed_10m"].mean()
        # La probabilidad de lluvia del propio modelo meteorologico: es la
        # incertidumbre que consume el simulador Monte Carlo (enfoque C).
        result.loc[idx, "precipitation_probability"] = near["precipitation_probability"].max()

    return result


def build_climate(db, context_df: pd.DataFrame) -> pd.DataFrame:
    raw = list(db["climate_historical"].find({}, {"_id": 0}))
    if not raw:
        return pd.DataFrame(index=context_df.index, columns=CLIMATE_COLS)

    coalesced = coalesce_climate(pd.DataFrame(raw))
    coalesced["date"] = pd.to_datetime(coalesced["date"])

    ctx = context_df.copy()
    ctx["_date_str"] = ctx["race_date"].dt.normalize()
    ctx["_month"] = ctx["race_date"].dt.month
    merged = ctx.merge(
        coalesced,
        left_on=["circuit_short_name", "_date_str"],
        right_on=["circuit_short_name", "date"],
        how="left",
    )

    # Norma estacional: media historica de temp_2m_avg para ese circuito+mes.
    # No hay riesgo de fuga: es clima observado, independiente del desenlace
    # de cualquier carrera, y conocido de antemano por climatologia.
    coalesced["month"] = coalesced["date"].dt.month
    norms = (
        coalesced.groupby(["circuit_short_name", "month"])["temp_2m_avg"]
        .mean().rename("circuit_seasonal_climate_norm_temp")
    )
    merged = merged.merge(norms, left_on=["circuit_short_name", "_month"], right_index=True, how="left")

    out = pd.DataFrame(index=context_df.index)
    out["race_day_temp_avg"] = merged["temp_2m_avg"].values
    out["race_day_temp_max"] = merged["temp_2m_max"].values if "temp_2m_max" in merged else float("nan")
    out["race_day_precipitation_mm"] = merged["precipitation_mm"].values
    out["race_day_humidity"] = merged["humidity_relative"].values
    out["race_day_wind_speed"] = merged["wind_speed_10m"].values
    out["circuit_seasonal_climate_norm_temp"] = merged["circuit_seasonal_climate_norm_temp"].values
    out["forecast_rain_probability"] = float("nan")
    out["race_day_track_temp"] = float("nan")
    out["race_day_wet_fraction"] = float("nan")
    out["climate_source"] = pd.Series(
        ["reanalysis"] * len(context_df), index=context_df.index
    ).where(out["race_day_temp_avg"].notna(), other=None)

    # --- Fuente preferente: sensores de pista (2023+) ---
    # Sustituye al reanalisis donde existe: se midio en el circuito durante
    # la sesion, no es una media diaria de rejilla.
    sensors = session_sensor_weather(db, ["Race"])
    if not sensors.empty:
        sensor_merged = context_df[["circuit_short_name"]].copy()
        sensor_merged["race_date"] = context_df["race_date"].dt.normalize()
        sensor_merged = sensor_merged.merge(sensors, on=["circuit_short_name", "race_date"], how="left")

        has_sensor = sensor_merged["sensor_air_temp"].notna().to_numpy()
        if has_sensor.any():
            out.loc[has_sensor, "race_day_temp_avg"] = sensor_merged.loc[has_sensor, "sensor_air_temp"].values
            out.loc[has_sensor, "race_day_temp_max"] = sensor_merged.loc[has_sensor, "sensor_air_temp_max"].values
            out.loc[has_sensor, "race_day_humidity"] = sensor_merged.loc[has_sensor, "sensor_humidity"].values
            out.loc[has_sensor, "race_day_wind_speed"] = sensor_merged.loc[has_sensor, "sensor_wind_speed"].values
            out.loc[has_sensor, "race_day_track_temp"] = sensor_merged.loc[has_sensor, "sensor_track_temp"].values
            out.loc[has_sensor, "race_day_wet_fraction"] = sensor_merged.loc[has_sensor, "sensor_wet_fraction"].values
            out.loc[has_sensor, "climate_source"] = "sensors"
            logger.info(f"Clima por sensores de pista en {int(has_sensor.sum())} filas")

    # --- Respaldo 1: pronostico (proximos ~5 dias) ---
    missing = out["race_day_temp_avg"].isna()
    if missing.any():
        race_dts = _race_datetimes(db, context_df)
        forecast = _forecast_for_races(db, context_df, race_dts)
        use_forecast = missing & forecast["temp_2m_avg"].notna()
        if use_forecast.any():
            out.loc[use_forecast, "race_day_temp_avg"] = forecast.loc[use_forecast, "temp_2m_avg"]
            out.loc[use_forecast, "race_day_temp_max"] = forecast.loc[use_forecast, "temp_2m_max"]
            out.loc[use_forecast, "race_day_precipitation_mm"] = forecast.loc[use_forecast, "precipitation_mm"]
            out.loc[use_forecast, "race_day_humidity"] = forecast.loc[use_forecast, "humidity_relative"]
            out.loc[use_forecast, "race_day_wind_speed"] = forecast.loc[use_forecast, "wind_speed_10m"]
            out.loc[use_forecast, "forecast_rain_probability"] = forecast.loc[use_forecast, "precipitation_probability"]
            out.loc[use_forecast, "climate_source"] = "forecast"

            # El pronostico no trae temperatura de asfalto (ningun servicio
            # meteorologico la da): se estima desde aire+humedad con la
            # relacion aprendida de los sensores de pista. Ver track_temp_model.
            track_model = fit_air_to_track(db)
            if track_model:
                estimated = estimate_track_temp(
                    track_model,
                    context_df.loc[use_forecast, "circuit_short_name"],
                    out.loc[use_forecast, "race_day_temp_avg"],
                    out.loc[use_forecast, "race_day_humidity"],
                )
                out.loc[use_forecast, "race_day_track_temp"] = estimated
                logger.info("Temperatura de asfalto estimada para las filas con pronostico")

            logger.info(f"Clima por pronostico en {int(use_forecast.sum())} filas")

    # --- Respaldo 2: climatologia (mas alla del horizonte de pronostico) ---
    still_missing = out["race_day_temp_avg"].isna() & out["circuit_seasonal_climate_norm_temp"].notna()
    if still_missing.any():
        out.loc[still_missing, "race_day_temp_avg"] = out.loc[still_missing, "circuit_seasonal_climate_norm_temp"]
        out.loc[still_missing, "climate_source"] = "climatology"
        logger.info(f"Clima por climatologia (sin pronostico aun) en {int(still_missing.sum())} filas")

    # Bandera de lluvia: cuando hay sensores se usa si de verdad se corrio en
    # mojado (fraccion de sesion con lluvia medida en pista). Solo cuando no
    # los hay se recurre a la precipitacion diaria del reanalisis, que es un
    # indicador mucho mas ruidoso: puede marcar como "lluviosa" una carrera
    # seca en la que llovio de madrugada.
    rain_from_sensors = out["race_day_wet_fraction"] > 0
    rain_from_daily = out["race_day_precipitation_mm"] > RAIN_THRESHOLD_MM
    out["climate_rain_probability_flag"] = (
        rain_from_sensors.where(out["race_day_wet_fraction"].notna(), rain_from_daily).astype(float)
    )
    out["climate_temp_delta_vs_norm"] = out["race_day_temp_avg"] - out["circuit_seasonal_climate_norm_temp"]

    return out
