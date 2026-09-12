"""Extrae el trazado REAL de un circuito a partir de telemetria GPS de OpenF1.

No es un dibujo ni una foto de stock: es la posicion x/y del coche muestreada
varias veces por segundo durante su vuelta mas rapida y limpia, la misma
fuente que usa el propio broadcast de F1 para el mapa en vivo. Solo funciona
para circuitos con sesiones 2023+ (cobertura real de OpenF1, ver
ml/docs/ARCHITECTURE.md).

El pipeline:
  1. Coger CUALQUIER sesion ya disputada de ese circuito (memoria: `sessions`)
  2. De esa sesion, la vuelta mas rapida SIN ser vuelta de entrada/salida de
     boxes (asi el trazado no incluye el carril de boxes)
  3. Pedir a OpenF1 los puntos GPS (x, y) durante esa ventana de tiempo
  4. Normalizar a un cuadro 0-1000 preservando el aspecto, listo para SVG

El resultado (pocos cientos de puntos) se guarda en Mongo, no los 15-20k
puntos GPS crudos por sesion -- esos no hace falta conservarlos.

Uso: python -m ml.ingestion.circuit_trace [circuit_short_name ...]
     Sin argumentos: lo intenta para todos los circuitos con sesiones 2023+.
"""

import logging
import sys
import time
from datetime import datetime, timedelta, timezone

import requests

from ml.common.db import get_db
from ml.common.logging_conf import configure

logger = logging.getLogger(__name__)

BASE_URL = "https://api.openf1.org/v1"
TARGET_POINTS = 260  # suficientes para un trazado suave, pocos para el HTML
VIEWBOX = 1000


def _get(path: str, params: dict) -> list:
    for attempt in range(1, 4):
        r = requests.get(f"{BASE_URL}/{path}", params=params, timeout=30)
        if r.status_code == 429:
            time.sleep(2.0 * attempt)
            continue
        r.raise_for_status()
        data = r.json()
        return data if isinstance(data, list) else []
    raise IOError(f"OpenF1 sigue devolviendo 429 para {path}")


def _fastest_clean_lap(session_key: int) -> dict | None:
    """La vuelta mas rapida de la sesion que no sea entrada/salida de boxes,
    de entre TODOS los pilotos (para no depender de que uno concreto haya
    hecho la vuelta rapida)."""
    laps = _get("laps", {"session_key": session_key})
    clean = [
        l for l in laps
        if l.get("lap_duration") and not l.get("is_pit_out_lap") and l.get("date_start")
    ]
    if not clean:
        return None
    return min(clean, key=lambda l: l["lap_duration"])


def extract_trace(session_key: int) -> list[tuple[float, float]] | None:
    lap = _fastest_clean_lap(session_key)
    if lap is None:
        return None

    start = datetime.fromisoformat(lap["date_start"].replace("Z", "+00:00"))
    end = start + timedelta(seconds=lap["lap_duration"] + 0.5)

    # OJO con el nombre del parametro: es "date>" y "date<" (el operador va
    # DENTRO del nombre), no "date>=" -- ese detalle causaba un 404 silencioso
    # en toda sesion probada durante el desarrollo.
    points = _get("location", {
        "session_key": session_key,
        "driver_number": lap["driver_number"],
        "date>": start.strftime("%Y-%m-%dT%H:%M:%S"),
        "date<": end.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    if len(points) < 20:
        return None

    return [(p["x"], p["y"]) for p in points if "x" in p and "y" in p]


def normalize_trace(points: list[tuple[float, float]]) -> list[list[float]]:
    """Escala a un cuadro VIEWBOXxVIEWBOX preservando el aspecto real del
    circuito (nunca lo distorsiona a un cuadrado), con margen del 6%, y
    reduce a ~TARGET_POINTS por muestreo uniforme."""
    step = max(1, len(points) // TARGET_POINTS)
    sampled = points[::step]

    xs = [p[0] for p in sampled]
    ys = [p[1] for p in sampled]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    span = max(maxx - minx, maxy - miny) or 1.0

    margin = VIEWBOX * 0.06
    scale = (VIEWBOX - 2 * margin) / span
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2

    # y se invierte: en SVG crece hacia abajo, en la telemetria hacia "norte".
    return [
        [round(margin + (VIEWBOX - 2 * margin) / 2 + (x - cx) * scale, 1),
         round(margin + (VIEWBOX - 2 * margin) / 2 - (y - cy) * scale, 1)]
        for x, y in sampled
    ]


def build_trace_for_circuit(db, circuit_short_name: str) -> dict | None:
    # Se consulta OpenF1 EN VIVO, no la coleccion `sessions` cacheada: esa
    # solo tiene los 4 tipos que ingiere AppFormula1.java (Race/Qualifying/
    # Sprint/Sprint Qualifying) y nunca los libres -- que son justo los que
    # tienen telemetria completa y ya se han disputado el jueves/viernes,
    # antes que la propia clasificacion.
    try:
        all_sessions = _get("sessions", {"circuit_short_name": circuit_short_name})
    except Exception as e:
        logger.warning(f"No se pudo listar sesiones de {circuit_short_name}: {e}")
        return None

    # Mas reciente primero, pero se prueban varias: la mas reciente puede ser
    # una sesion todavia futura (programada, sin vueltas registradas).
    candidates = sorted(
        ({"session_key": s["session_key"], "year": s["year"]} for s in all_sessions if not s.get("is_cancelled")),
        key=lambda s: s["session_key"], reverse=True,
    )[:10]
    if not candidates:
        logger.warning(f"Sin sesiones OpenF1 para {circuit_short_name} (normal si es de antes de 2023)")
        return None

    for session in candidates:
        try:
            raw = extract_trace(session["session_key"])
        except Exception as e:
            logger.warning(f"{circuit_short_name} session_key={session['session_key']}: {e}")
            continue
        if raw:
            points = normalize_trace(raw)
            return {
                "circuit_short_name": circuit_short_name,
                "points": points,
                "source_session_key": session["session_key"],
                "source_year": session["year"],
                "n_raw_points": len(raw),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }

    logger.warning(f"Ninguna de las {len(candidates)} sesiones probadas tenia vuelta limpia utilizable para {circuit_short_name}")
    return None


def main() -> None:
    db = get_db()
    db["circuit_traces"].create_index("circuit_short_name", unique=True)

    requested = sys.argv[1:]
    circuits = requested or db["sessions"].distinct("circuit_short_name")

    saved, skipped = 0, 0
    for name in circuits:
        trace = build_trace_for_circuit(db, name)
        if trace is None:
            skipped += 1
            continue
        db["circuit_traces"].replace_one(
            {"circuit_short_name": name}, trace, upsert=True
        )
        logger.info(f"{name}: trazado guardado ({len(trace['points'])} puntos, "
                    f"vuelta de {trace['source_year']})")
        saved += 1

    logger.info(f"Trazados guardados: {saved}, sin datos suficientes: {skipped}")


if __name__ == "__main__":
    configure()
    main()
