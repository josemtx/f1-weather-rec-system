"""Ingesta idempotente de clima historico diario (NASA POWER) -> climate_historical.

Un request por circuito cubre todo 2018-2026 de una vez. Requiere que
`circuits` ya este cargada (ml.ingestion.load_circuits).

Uso: python -m ml.ingestion.ingest_climate
"""

import logging

import pymongo

from ml.common.db import get_db
from ml.common.logging_conf import configure
from ml.ingestion.nasa_power_client import get_daily_climate

logger = logging.getLogger(__name__)

START_DATE = "20180101"
END_DATE = "20261231"
SOURCE = "nasa_power"


def _fmt_date(yyyymmdd: str) -> str:
    return f"{yyyymmdd[0:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:8]}"


def ingest() -> tuple[int, int]:
    db = get_db()
    circuits_col = db["circuits"]
    climate_col = db["climate_historical"]
    climate_col.create_index(
        [
            ("circuit_short_name", pymongo.ASCENDING),
            ("date", pymongo.ASCENDING),
            ("source", pymongo.ASCENDING),
        ],
        unique=True,
    )

    circuits = list(circuits_col.find({}))
    if not circuits:
        logger.error("La coleccion circuits esta vacia; ejecuta ml.ingestion.load_circuits primero.")
        return 0, 0

    total_saved = 0
    circuits_failed = 0

    for circuit in circuits:
        name = circuit["circuit_short_name"]
        try:
            daily = get_daily_climate(circuit["lat"], circuit["lon"], START_DATE, END_DATE)
        except Exception as e:
            logger.warning(f"Sin clima NASA POWER para circuito={name}: {e}")
            circuits_failed += 1
            continue

        saved_here = 0
        for yyyymmdd, metrics in daily.items():
            doc = {
                "circuit_short_name": name,
                "date": _fmt_date(yyyymmdd),
                "source": SOURCE,
                **metrics,
            }
            climate_col.replace_one(
                {"circuit_short_name": name, "date": doc["date"], "source": SOURCE},
                doc,
                upsert=True,
            )
            saved_here += 1

        logger.info(f"Circuito {name}: {saved_here} dias de clima guardados")
        total_saved += saved_here

    logger.info(
        f"Ingesta NASA POWER finalizada. Dias guardados: {total_saved}, circuitos fallidos: {circuits_failed}"
    )
    return total_saved, circuits_failed


if __name__ == "__main__":
    configure()
    ingest()
