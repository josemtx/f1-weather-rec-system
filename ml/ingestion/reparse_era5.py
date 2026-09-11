"""Re-extrae climate_historical (source=era5) desde los .nc YA DESCARGADOS en
ml/data/raw/era5/, sin volver a pedir nada a Copernicus CDS.

Se uso porque _extract_daily_means se corrigio (temp_2m_max faltaba en las
primeras corridas de ingest_era5.py) despues de que el job en background ya
hubiera descargado varios circuitos con el codigo viejo -- en vez de perder
esa cola real de CDS, se relee el disco con la version corregida y se
sobreescribe (upsert) climate_historical.

Uso: python -m ml.ingestion.reparse_era5
"""

import logging
from pathlib import Path

import pymongo

from ml.common.db import get_db
from ml.common.logging_conf import configure
from ml.ingestion.ingest_era5 import RAW_DIR, SOURCE, _extract_daily_means, _race_dates_by_circuit

logger = logging.getLogger(__name__)


def reparse() -> int:
    db = get_db()
    climate_col = db["climate_historical"]
    climate_col.create_index(
        [
            ("circuit_short_name", pymongo.ASCENDING),
            ("date", pymongo.ASCENDING),
            ("source", pymongo.ASCENDING),
        ],
        unique=True,
    )

    dates_by_circuit = _race_dates_by_circuit(db)
    total_saved = 0

    if not RAW_DIR.exists():
        logger.warning(f"{RAW_DIR} no existe todavia, nada que reprocesar")
        return 0

    for circuit_dir in RAW_DIR.iterdir():
        if not circuit_dir.is_dir():
            continue
        circuit_short_name = circuit_dir.name.replace("_", " ")
        valid_dates = dates_by_circuit.get(circuit_short_name)
        if not valid_dates:
            # el nombre con "_" a veces no revierte exacto (p.ej. "Ciudad de Mexico");
            # se intenta tambien buscando por coincidencia de espacio/guion bajo.
            valid_dates = next(
                (v for k, v in dates_by_circuit.items() if k.replace(" ", "_") == circuit_dir.name), None
            )
        if not valid_dates:
            logger.warning(f"Sin fechas de carrera conocidas para {circuit_dir.name}, se omite")
            continue

        saved_here = 0
        for nc_file in circuit_dir.glob("*.nc"):
            try:
                daily = _extract_daily_means(nc_file)
            except Exception as e:
                logger.warning(f"No se pudo reprocesar {nc_file}: {e}")
                continue
            for date, metrics in daily.items():
                if date not in valid_dates:
                    continue
                doc = {"circuit_short_name": circuit_short_name, "date": date, "source": SOURCE, **metrics}
                climate_col.replace_one(
                    {"circuit_short_name": circuit_short_name, "date": date, "source": SOURCE},
                    doc,
                    upsert=True,
                )
                saved_here += 1

        if saved_here:
            logger.info(f"Reprocesado {circuit_short_name}: {saved_here} dias actualizados (con temp_2m_max)")
        total_saved += saved_here

    logger.info(f"Reproceso ERA5 finalizado. Dias actualizados: {total_saved}")
    return total_saved


if __name__ == "__main__":
    configure()
    reparse()
