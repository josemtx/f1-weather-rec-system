"""Ingesta ERA5-Land (Copernicus CDS) -> climate_historical (source=era5).

Enriquecimiento OPCIONAL y desacoplado: si no hay credenciales CDS en .env
(CDS_URL, CDS_KEY), se registra el motivo y se termina sin error -- el resto
del pipeline (NASA POWER) ya cubre climate_historical para todos los anios.

Pensado para lanzarse una vez y dejarse correr en background durante el
sprint (la cola de CDS puede tardar minutos u horas por peticion); procesa
un circuito a la vez y continua aunque uno falle o tarde demasiado.

Uso: python -m ml.ingestion.ingest_era5
"""

import logging
import zipfile
from pathlib import Path

import pymongo
import xarray as xr

from ml.common.db import get_db
from ml.common.logging_conf import configure
from ml.ingestion.era5_client import credentials_available, submit_and_download

logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).resolve().parents[2] / "ml" / "data" / "raw" / "era5"
SOURCE = "era5"


def _race_dates_by_circuit(db) -> dict[str, list[str]]:
    circuits = {c["jolpica_circuit_id"]: c["circuit_short_name"] for c in db["circuits"].find({})}
    dates_by_short_name: dict[str, set[str]] = {}
    for result in db["historical_results"].find({}, {"circuit_id": 1, "race_date": 1}):
        short_name = circuits.get(result["circuit_id"])
        if not short_name:
            continue
        dates_by_short_name.setdefault(short_name, set()).add(result["race_date"])
    return {k: sorted(v) for k, v in dates_by_short_name.items()}


def _kelvin_to_celsius(k):
    return k - 273.15


def _resolve_actual_netcdf(nc_path: Path) -> Path:
    """El API de CDS a veces devuelve un .zip (con un unico .nc dentro) aunque
    se pida data_format=netcdf directamente. Se detecta y extrae de forma
    transparente; si ya es netCDF puro, se usa tal cual."""
    if not zipfile.is_zipfile(nc_path):
        return nc_path

    extract_dir = nc_path.parent / f"_{nc_path.stem}_extracted"
    with zipfile.ZipFile(nc_path) as z:
        names = [n for n in z.namelist() if n.endswith(".nc")]
        if not names:
            raise ValueError(f"{nc_path} es un zip sin ningun .nc dentro: {z.namelist()}")
        extract_dir.mkdir(parents=True, exist_ok=True)
        z.extract(names[0], extract_dir)
        return extract_dir / names[0]


def _extract_daily_means(nc_path: Path) -> dict[str, dict]:
    actual_path = _resolve_actual_netcdf(nc_path)
    ds = xr.open_dataset(actual_path)
    df = ds.to_dataframe().reset_index()
    df["date"] = df["valid_time"].dt.strftime("%Y-%m-%d") if "valid_time" in df.columns else df["time"].dt.strftime("%Y-%m-%d")

    out = {}
    for date, group in df.groupby("date"):
        t2m_c = _kelvin_to_celsius(group["t2m"]).mean()
        d2m_c = _kelvin_to_celsius(group["d2m"]).mean()
        # Aproximacion de humedad relativa desde temp/dewpoint (formula de Magnus).
        humidity = 100 * (
            2.71828 ** ((17.625 * d2m_c) / (243.04 + d2m_c))
            / 2.71828 ** ((17.625 * t2m_c) / (243.04 + t2m_c))
        )
        wind_speed = ((group["u10"] ** 2 + group["v10"] ** 2) ** 0.5).mean()
        out[date] = {
            "temp_2m_avg": float(t2m_c),
            # Maximo entre las horas UTC muestreadas (HOURS en era5_client.py),
            # no el maximo real del dia -- mismo caveat que aplicaria a NASA
            # POWER si su T2M_MAX no fuese ya diario. Se mantiene por paridad
            # de columnas con NASA POWER para que la coalescencia (climate.py)
            # no deje NaN en race_day_temp_max al preferir ERA5.
            "temp_2m_max": float(_kelvin_to_celsius(group["t2m"]).max()),
            "humidity_relative": float(humidity),
            "wind_speed_10m": float(wind_speed),
            "surface_pressure": float(group["sp"].mean() / 100.0),
            "precipitation_mm": float(group["tp"].mean() * 1000.0),
        }
    ds.close()
    return out


def ingest() -> tuple[int, int]:
    if not credentials_available():
        logger.warning(
            "CDS_URL/CDS_KEY no configuradas en .env -- se omite ERA5. "
            "Registrate en https://cds.climate.copernicus.eu/user, acepta la licencia "
            "de 'ERA5-Land hourly data from 1950 to present', y anade CDS_URL y CDS_KEY al .env. "
            "climate_historical ya tiene cobertura completa via NASA POWER."
        )
        return 0, 0

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

    circuits_by_name = {c["circuit_short_name"]: c for c in db["circuits"].find({})}
    dates_by_circuit = _race_dates_by_circuit(db)
    if not dates_by_circuit:
        logger.error("No hay historical_results todavia; ejecuta ml.jolpica.ingest_jolpica primero.")
        return 0, 0

    total_saved = 0
    circuits_failed = 0

    for circuit_short_name, dates in dates_by_circuit.items():
        circuit = circuits_by_name.get(circuit_short_name)
        if not circuit:
            continue
        try:
            circuit_dir = RAW_DIR / circuit_short_name.replace(" ", "_")
            submit_and_download(circuit_short_name, circuit["lat"], circuit["lon"], dates, circuit_dir)

            saved_here = 0
            for nc_file in circuit_dir.glob("*.nc"):
                daily = _extract_daily_means(nc_file)
                for date, metrics in daily.items():
                    if date not in dates:
                        continue
                    doc = {
                        "circuit_short_name": circuit_short_name,
                        "date": date,
                        "source": SOURCE,
                        **metrics,
                    }
                    climate_col.replace_one(
                        {"circuit_short_name": circuit_short_name, "date": date, "source": SOURCE},
                        doc,
                        upsert=True,
                    )
                    saved_here += 1

            logger.info(f"Circuito {circuit_short_name}: {saved_here} dias ERA5 guardados")
            total_saved += saved_here
        except Exception as e:
            logger.warning(f"Fallo ERA5 para circuito={circuit_short_name}: {e}")
            circuits_failed += 1
            continue

    logger.info(f"Ingesta ERA5 finalizada. Dias guardados: {total_saved}, circuitos fallidos: {circuits_failed}")
    return total_saved, circuits_failed


if __name__ == "__main__":
    configure()
    ingest()
