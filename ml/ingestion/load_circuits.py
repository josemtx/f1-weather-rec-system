"""Carga idempotente de ml/config/circuits.json en la coleccion `circuits`.

Uso: python -m ml.ingestion.load_circuits
"""

import json
import logging
from pathlib import Path

import pymongo

from ml.common.db import get_db
from ml.common.logging_conf import configure

logger = logging.getLogger(__name__)

CIRCUITS_JSON = Path(__file__).resolve().parents[1] / "config" / "circuits.json"


def load_circuits() -> int:
    db = get_db()
    col = db["circuits"]
    col.create_index([("circuit_short_name", pymongo.ASCENDING)], unique=True)

    data = json.loads(CIRCUITS_JSON.read_text(encoding="utf-8"))
    circuits = data["circuits"]

    upserted = 0
    for circuit in circuits:
        col.replace_one(
            {"circuit_short_name": circuit["circuit_short_name"]},
            circuit,
            upsert=True,
        )
        upserted += 1

    logger.info(f"Circuitos cargados/actualizados: {upserted}")
    return upserted


if __name__ == "__main__":
    configure()
    load_circuits()
