"""Cliente para Copernicus Climate Data Store (ERA5-Land hourly reanalysis).

A diferencia de NASA POWER (rapido, punto unico, sin key), CDS requiere
cuenta+API key y encola las peticiones (puede tardar de minutos a horas).
Para minimizar tiempo de cola pedimos SOLO las fechas de carrera exactas
por circuito (conocidas via historical_results de Jolpica), no el rango
completo 2018-2026 -- una peticion de ~1-9 dias por circuito en vez de ~3200.

Requiere en .env: CDS_URL (normalmente https://cds.climate.copernicus.eu/api)
y CDS_KEY (tu Personal Access Token de https://cds.climate.copernicus.eu/user).
Sin ellas, ingest_era5.py se salta con instrucciones claras en el log en vez
de fallar -- ERA5 es un enriquecimiento opcional, nunca bloqueante (ver
climate_historical, que coexiste con nasa_power via el campo `source`).
"""

import logging
import os
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

logger = logging.getLogger(__name__)

VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_pressure",
    "total_precipitation",
]

# Horas UTC candidatas para cubrir sesiones de clasificacion/carrera en cualquier
# huso horario del calendario F1; se promedian tras la descarga.
HOURS = [f"{h:02d}:00" for h in range(10, 19)]


def credentials_available() -> bool:
    return bool(os.getenv("CDS_URL") and os.getenv("CDS_KEY"))


def submit_and_download(
    circuit_short_name: str,
    lat: float,
    lon: float,
    dates: list[str],
    out_dir: Path,
) -> Path:
    """Encola y descarga un unico NetCDF con las fechas de carrera de un circuito.

    dates: lista de 'YYYY-MM-DD'. Bloquea hasta que CDS complete la peticion
    (puede ser lento) -- pensado para ejecutarse en background, no en el flujo
    interactivo principal del sprint.
    """
    import cdsapi

    by_year = defaultdict(lambda: defaultdict(set))
    for date in dates:
        year, month, day = date.split("-")
        by_year[year][month].add(day)

    client = cdsapi.Client(url=os.environ["CDS_URL"], key=os.environ["CDS_KEY"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{circuit_short_name.replace(' ', '_')}.nc"

    # Caja pequena (~0.2 grados) centrada en el circuito; ERA5-Land es grid 0.1x0.1.
    area = [lat + 0.1, lon - 0.1, lat - 0.1, lon + 0.1]  # N, W, S, E

    for year, months in by_year.items():
        for month, days in months.items():
            request = {
                "variable": VARIABLES,
                "year": year,
                "month": month,
                "day": sorted(days),
                "time": HOURS,
                "area": area,
                "data_format": "netcdf",
            }
            year_month_path = out_dir / f"{circuit_short_name.replace(' ', '_')}_{year}_{month}.nc"
            logger.info(f"Solicitando ERA5-Land para {circuit_short_name} {year}-{month} (dias={sorted(days)})")
            client.retrieve("reanalysis-era5-land", request, str(year_month_path))

    return out_path
