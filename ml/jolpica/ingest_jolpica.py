"""Ingesta idempotente Jolpica-F1: resultados de carrera, clasificacion,
sprint y calendario de sesiones -> historical_results / qualifying_results /
sprint_results / race_schedule.

Uso: python -m ml.jolpica.ingest_jolpica [anio ...]
     python -m ml.jolpica.ingest_jolpica 2024 2025 2026   # solo esos anios,
     util para rellenar huecos dejados por 429 sin re-correr todo el rango.
"""

import logging
import re
import sys

import pymongo

from ml.common.db import get_db
from ml.common.logging_conf import configure
from ml.jolpica.jolpica_client import (
    get_qualifying_results,
    get_race_results,
    get_season_schedule,
    get_sprint_results,
)

logger = logging.getLogger(__name__)

YEARS = list(range(2018, 2027))
_TIME_RE = re.compile(r"^(?:(\d+):)?(\d+\.\d+)$")


def _parse_time_to_seconds(value: str | None) -> float | None:
    """'1:29.708' o '29.708' -> segundos. None si no hay tiempo (eliminado
    antes de esa sesion, p.ej. sin Q3 tras quedar fuera en Q1)."""
    if not value:
        return None
    m = _TIME_RE.match(value)
    if not m:
        return None
    minutes = int(m.group(1)) if m.group(1) else 0
    return minutes * 60 + float(m.group(2))


def _race_doc(year: int, round_number: int, circuit_id: str, race_date: str, result: dict) -> dict:
    driver = result["Driver"]
    constructor = result["Constructor"]
    status = result.get("status", "Unknown")
    # positionText es numerico para cualquier piloto CLASIFICADO (incluye
    # "Lapped" -- termino la carrera, solo una vuelta por detras del lider)
    # y una letra (R=retirado, D=descalificado, W=retirado antes de empezar,
    # N=no clasificado) para quien NO termino. Parsear el texto libre de
    # `status` es fragil: "Lapped" no empieza por "+" y quedaba mal marcado
    # como DNF (encontrado via DNF rate 18.8% train vs 31% test -- ver
    # ml/docs/ARCHITECTURE.md).
    finished = result["positionText"].isdigit()
    return {
        "year": year,
        "round": round_number,
        "circuit_id": circuit_id,
        "race_date": race_date,
        "driver_code": driver.get("code") or driver["driverId"],
        "driver_id": driver["driverId"],
        "driver_full_name": f"{driver['givenName']} {driver['familyName']}",
        "constructor_id": constructor["constructorId"],
        "constructor_name": constructor["name"],
        "grid_position": int(result["grid"]),
        "finish_position": int(result["position"]) if result["position"].isdigit() else None,
        "position_text": result["positionText"],
        "points": float(result["points"]),
        "status": status,
        "finished": finished,
        "laps_completed": int(result["laps"]),
    }


def _qualifying_doc(year: int, round_number: int, circuit_id: str, qualifying_date: str | None, result: dict) -> dict:
    driver = result["Driver"]
    q1 = _parse_time_to_seconds(result.get("Q1"))
    q2 = _parse_time_to_seconds(result.get("Q2"))
    q3 = _parse_time_to_seconds(result.get("Q3"))
    best = min(t for t in (q1, q2, q3) if t is not None) if any(t is not None for t in (q1, q2, q3)) else None
    return {
        "year": year,
        "round": round_number,
        "circuit_id": circuit_id,
        "qualifying_date": qualifying_date,
        "driver_code": driver.get("code") or driver["driverId"],
        "constructor_id": result["Constructor"]["constructorId"],
        "qualifying_position": int(result["position"]),
        "q1_seconds": q1,
        "q2_seconds": q2,
        "q3_seconds": q3,
        "best_seconds": best,
    }


def _sprint_doc(year: int, round_number: int, circuit_id: str, sprint_date: str | None, result: dict) -> dict:
    driver = result["Driver"]
    return {
        "year": year,
        "round": round_number,
        "circuit_id": circuit_id,
        "sprint_date": sprint_date,
        "driver_code": driver.get("code") or driver["driverId"],
        "constructor_id": result["Constructor"]["constructorId"],
        "sprint_grid": int(result["grid"]),
        "sprint_position": int(result["position"]) if result["position"].isdigit() else None,
        "sprint_points": float(result["points"]),
        "sprint_status": result.get("status", "Unknown"),
    }


def ingest(years: list[int] | None = None) -> tuple[int, int]:
    years = years or YEARS
    db = get_db()

    race_col = db["historical_results"]
    race_col.create_index(
        [("year", pymongo.ASCENDING), ("round", pymongo.ASCENDING), ("driver_code", pymongo.ASCENDING)],
        unique=True,
    )
    quali_col = db["qualifying_results"]
    quali_col.create_index(
        [("year", pymongo.ASCENDING), ("round", pymongo.ASCENDING), ("driver_code", pymongo.ASCENDING)],
        unique=True,
    )
    sprint_col = db["sprint_results"]
    sprint_col.create_index(
        [("year", pymongo.ASCENDING), ("round", pymongo.ASCENDING), ("driver_code", pymongo.ASCENDING)],
        unique=True,
    )
    schedule_col = db["race_schedule"]
    schedule_col.create_index([("year", pymongo.ASCENDING), ("round", pymongo.ASCENDING)], unique=True)

    total_saved = 0
    seasons_failed = 0

    for year in years:
        try:
            schedule = get_season_schedule(year)
        except Exception as e:
            logger.warning(f"Sin calendario para year={year}: {e}")
            seasons_failed += 1
            continue

        if not schedule:
            logger.warning(f"Calendario vacio para year={year}, se omite")
            continue

        year_saved = 0
        quali_saved = 0
        sprint_saved = 0

        for race in schedule:
            round_number = race["round"]
            circuit_id = race["circuit_id"]

            schedule_col.replace_one(
                {"year": year, "round": round_number},
                {
                    "year": year, "round": round_number, "circuit_id": circuit_id,
                    "race_date": race["race_date"], "race_time": race.get("race_time"),
                    "qualifying_date": race["qualifying_date"],
                    "qualifying_time": race.get("qualifying_time"),
                    "sprint_date": race["sprint_date"], "sprint_time": race.get("sprint_time"),
                },
                upsert=True,
            )

            try:
                results = get_race_results(year, round_number)
                for result in results:
                    doc = _race_doc(year, round_number, circuit_id, race["race_date"], result)
                    race_col.replace_one(
                        {"year": doc["year"], "round": doc["round"], "driver_code": doc["driver_code"]},
                        doc, upsert=True,
                    )
                    year_saved += 1
            except Exception as e:
                logger.warning(f"Sin resultados de carrera para year={year} round={round_number}: {e}")

            try:
                quali_results = get_qualifying_results(year, round_number)
                for result in quali_results:
                    doc = _qualifying_doc(year, round_number, circuit_id, race["qualifying_date"], result)
                    quali_col.replace_one(
                        {"year": doc["year"], "round": doc["round"], "driver_code": doc["driver_code"]},
                        doc, upsert=True,
                    )
                    quali_saved += 1
            except Exception as e:
                logger.warning(f"Sin resultados de clasificacion para year={year} round={round_number}: {e}")

            if race["sprint_date"]:
                try:
                    sprint_results = get_sprint_results(year, round_number)
                    for result in sprint_results:
                        doc = _sprint_doc(year, round_number, circuit_id, race["sprint_date"], result)
                        sprint_col.replace_one(
                            {"year": doc["year"], "round": doc["round"], "driver_code": doc["driver_code"]},
                            doc, upsert=True,
                        )
                        sprint_saved += 1
                except Exception as e:
                    logger.warning(f"Sin resultados de sprint para year={year} round={round_number}: {e}")

        logger.info(
            f"Year {year}: {len(schedule)} carreras, {year_saved} resultados, "
            f"{quali_saved} clasificaciones, {sprint_saved} sprints guardados"
        )
        total_saved += year_saved

    logger.info(f"Ingesta Jolpica finalizada. Resultados guardados: {total_saved}, temporadas fallidas: {seasons_failed}")
    return total_saved, seasons_failed


def ingest_schedule_only(years: list[int] | None = None) -> int:
    """Refresca solo `race_schedule` (1 peticion por anio, segundos).

    Util porque el calendario cambia dentro de una temporada (Imola 2023 se
    cancelo por inundaciones) y porque permite anadir campos nuevos sin
    re-descargar todos los resultados.
    """
    years = years or YEARS
    db = get_db()
    schedule_col = db["race_schedule"]
    schedule_col.create_index([("year", pymongo.ASCENDING), ("round", pymongo.ASCENDING)], unique=True)

    total = 0
    for year in years:
        try:
            schedule = get_season_schedule(year)
        except Exception as e:
            logger.warning(f"Sin calendario para year={year}: {e}")
            continue
        for race in schedule:
            schedule_col.replace_one(
                {"year": year, "round": race["round"]},
                {
                    "year": year, "round": race["round"], "circuit_id": race["circuit_id"],
                    "race_date": race["race_date"], "race_time": race.get("race_time"),
                    "qualifying_date": race["qualifying_date"],
                    "qualifying_time": race.get("qualifying_time"),
                    "sprint_date": race["sprint_date"], "sprint_time": race.get("sprint_time"),
                },
                upsert=True,
            )
            total += 1
        logger.info(f"Year {year}: {len(schedule)} rondas de calendario actualizadas")

    logger.info(f"Calendario actualizado. Rondas: {total}")
    return total


if __name__ == "__main__":
    configure()
    args = sys.argv[1:]
    if args and args[0] == "--schedule-only":
        ingest_schedule_only([int(a) for a in args[1:]] or None)
    else:
        ingest([int(a) for a in args] if args else None)
