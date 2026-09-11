"""Cliente HTTP para Jolpica-F1 (sucesor gratuito de Ergast, compatible en schema).

Fuente unica y consistente de grid_position y clasificacion final para TODOS
los anios 2018-2026 (OpenF1 no ingesta datos de clasificacion/grid). Mismo
patron de resiliencia que Formula1Service.java: reintento con backoff en 429,
tope de reintentos, y llamador responsable de degradar con gracia por temporada.
"""

import logging
import time

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://api.jolpi.ca/ergast/f1"
MAX_RETRIES = 8
REQUEST_DELAY_SECONDS = 0.6
MAX_BACKOFF_SECONDS = 20.0


def _get(path: str, params: dict | None = None) -> dict:
    url = f"{BASE_URL}/{path}"
    attempt = 1
    while True:
        response = requests.get(url, params=params, timeout=15)
        if response.status_code == 429:
            if attempt >= MAX_RETRIES:
                raise IOError(f"Jolpica sigue devolviendo 429 tras {MAX_RETRIES} intentos para: {path}")
            # Respeta Retry-After si el servidor lo manda; si no, backoff
            # exponencial con techo -- el lineal (1,2,3,4,5s) resulto
            # insuficiente en corridas largas (huecos reales en 2024-2026,
            # ver ml/docs/ARCHITECTURE.md).
            retry_after = response.headers.get("Retry-After")
            wait = float(retry_after) if retry_after else min(2.0 ** attempt, MAX_BACKOFF_SECONDS)
            logger.warning(f"Rate limit (429) en '{path}', reintento {attempt}/{MAX_RETRIES} en {wait}s")
            time.sleep(wait)
            attempt += 1
            continue
        response.raise_for_status()
        return response.json()


def get_season_schedule(year: int) -> list[dict]:
    """Devuelve [{round, circuit_id, race_date, qualifying_date, sprint_date}, ...].

    qualifying_date/sprint_date vienen de los sub-objetos Qualifying/Sprint del
    calendario (fechas REALES de sabado/viernes, distintas de race_date) --
    necesarias para poder pedir el clima de esas sesiones, no solo el de carrera.
    sprint_date es None en fines de semana sin sprint (la mayoria antes de 2021).
    """
    data = _get(f"{year}.json", params={"limit": 40})
    races = data["MRData"]["RaceTable"]["Races"]
    time.sleep(REQUEST_DELAY_SECONDS)
    return [
        {
            "round": int(r["round"]),
            "circuit_id": r["Circuit"]["circuitId"],
            "race_date": r["date"],
            # La HORA importa para el clima: Las Vegas y Singapur son
            # nocturnas, Bahrein al atardecer. Sin ella no se puede elegir
            # el tramo correcto del pronostico (que viene cada 3 horas).
            "race_time": r.get("time"),
            "qualifying_date": r.get("Qualifying", {}).get("date"),
            "qualifying_time": r.get("Qualifying", {}).get("time"),
            "sprint_date": r.get("Sprint", {}).get("date"),
            "sprint_time": r.get("Sprint", {}).get("time"),
        }
        for r in races
    ]


def get_race_results(year: int, round_number: int) -> list[dict]:
    """Devuelve resultados crudos (Ergast schema) de una carrera."""
    data = _get(f"{year}/{round_number}/results.json", params={"limit": 40})
    races = data["MRData"]["RaceTable"]["Races"]
    time.sleep(REQUEST_DELAY_SECONDS)
    if not races:
        return []
    return races[0]["Results"]


def get_qualifying_results(year: int, round_number: int) -> list[dict]:
    """Devuelve resultados crudos de clasificacion (position, Q1/Q2/Q3)."""
    data = _get(f"{year}/{round_number}/qualifying.json", params={"limit": 40})
    races = data["MRData"]["RaceTable"]["Races"]
    time.sleep(REQUEST_DELAY_SECONDS)
    if not races:
        return []
    return races[0]["QualifyingResults"]


def get_sprint_results(year: int, round_number: int) -> list[dict]:
    """Devuelve resultados crudos de sprint. Lista vacia si ese fin de semana
    no tuvo sprint (no es un error, la mayoria de rondas no lo tienen)."""
    data = _get(f"{year}/{round_number}/sprint.json", params={"limit": 40})
    races = data["MRData"]["RaceTable"]["Races"]
    time.sleep(REQUEST_DELAY_SECONDS)
    if not races:
        return []
    return races[0].get("SprintResults", [])
