"""Cliente HTTP para NASA POWER (climatologia diaria historica, gratuita, sin API key).

Un unico request por circuito cubre todo el rango 2018-2026 (mucho mas
eficiente que un request por fecha de carrera). fill_value -999 indica dato
no disponible (tipicamente fechas futuras respecto a hoy) y se filtra fuera.
"""

import logging
import time

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
PARAMETERS = "T2M,T2M_MAX,T2M_MIN,RH2M,WS10M,PRECTOTCORR,PS,CLOUD_AMT"
MAX_RETRIES = 5
REQUEST_DELAY_SECONDS = 1.0
FILL_VALUE = -999.0


def get_daily_climate(lat: float, lon: float, start: str, end: str) -> dict[str, dict]:
    """Devuelve {yyyymmdd: {t2m_avg, t2m_max, t2m_min, humidity, wind_speed,
    precipitation_mm, pressure, cloud_cover}, ...} filtrando fill_value.
    """
    params = {
        "parameters": PARAMETERS,
        "community": "RE",
        "longitude": lon,
        "latitude": lat,
        "start": start,
        "end": end,
        "format": "JSON",
    }

    attempt = 1
    while True:
        response = requests.get(BASE_URL, params=params, timeout=60)
        if response.status_code == 429:
            if attempt >= MAX_RETRIES:
                raise IOError(f"NASA POWER sigue devolviendo 429 tras {MAX_RETRIES} intentos")
            wait = 2.0 * attempt
            logger.warning(f"Rate limit (429) en NASA POWER, reintento {attempt}/{MAX_RETRIES} en {wait}s")
            time.sleep(wait)
            attempt += 1
            continue
        response.raise_for_status()
        break

    data = response.json()
    params_by_var = data["properties"]["parameter"]
    time.sleep(REQUEST_DELAY_SECONDS)

    dates = params_by_var["T2M"].keys()
    result = {}
    for date in dates:
        t2m = params_by_var["T2M"][date]
        if t2m == FILL_VALUE:
            continue
        result[date] = {
            "temp_2m_avg": t2m,
            "temp_2m_max": params_by_var["T2M_MAX"][date],
            "temp_2m_min": params_by_var["T2M_MIN"][date],
            "humidity_relative": params_by_var["RH2M"][date],
            "wind_speed_10m": params_by_var["WS10M"][date],
            "precipitation_mm": params_by_var["PRECTOTCORR"][date],
            "surface_pressure": params_by_var["PS"][date],
            "cloud_cover": params_by_var["CLOUD_AMT"][date],
        }
    return result
