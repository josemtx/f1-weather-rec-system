"""Mapeo (session_key, driver_number) -> driver_code (Jolpica) via nombre completo.

OpenF1 identifica pilotos por driver_number+full_name ("Sergio PEREZ"); Jolpica
usa driver_code ("PER") + driver_full_name ("Sergio Perez"). No hay una clave
compartida directa, asi que se empareja por nombre normalizado (mayusculas,
sin acentos) dentro del mismo anio. 4 excepciones conocidas (nombre corto,
orden apellido-nombre, acentos que el terminal no siempre preserva) se
resuelven con un mapa explicito en vez de heuristicas fragiles.
"""

import unicodedata

import pandas as pd

_MANUAL_OVERRIDES = {
    # OpenF1 full_name normalizado -> Jolpica full_name normalizado
    "KIMI ANTONELLI": "ANDREA KIMI ANTONELLI",
    "ZHOU GUANYU": "GUANYU ZHOU",
}


def _normalize(name: str) -> str:
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    return name.upper().strip()


def build_driver_number_to_code(db) -> pd.DataFrame:
    """Devuelve DataFrame [session_key, driver_number, driver_code]."""
    drivers = list(db["drivers"].find({}, {"_id": 0, "driver_number": 1, "full_name": 1, "session_key": 1}))
    if not drivers:
        return pd.DataFrame(columns=["session_key", "driver_number", "driver_code"])

    sessions = {s["session_key"]: s["year"] for s in db["sessions"].find({}, {"_id": 0, "session_key": 1, "year": 1})}

    results = list(db["historical_results"].find({}, {"_id": 0, "year": 1, "driver_code": 1, "driver_full_name": 1}))
    jolpica_by_year: dict[int, dict[str, str]] = {}
    for r in results:
        norm = _normalize(r["driver_full_name"])
        jolpica_by_year.setdefault(r["year"], {})[norm] = r["driver_code"]

    rows = []
    unmatched = set()
    for d in drivers:
        year = sessions.get(d["session_key"])
        if year is None:
            continue
        norm = _normalize(d["full_name"])
        norm = _MANUAL_OVERRIDES.get(norm, norm)
        code = jolpica_by_year.get(year, {}).get(norm)
        if code is None:
            unmatched.add((year, d["full_name"]))
            continue
        rows.append({"session_key": d["session_key"], "driver_number": d["driver_number"], "driver_code": code})

    if unmatched:
        import logging
        logging.getLogger(__name__).warning(f"Pilotos sin mapeo driver_number->driver_code: {sorted(unmatched)}")

    return pd.DataFrame(rows)
