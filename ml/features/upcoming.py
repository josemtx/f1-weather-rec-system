"""Filas "fantasma" para carreras que aun no se han disputado.

La matriz de features se construye desde `historical_results`, que por
definicion solo contiene carreras ya corridas (tienen `finish_position`).
Para predecir una carrera futura se genera una fila por piloto con todo lo
que SI se sabe (circuito, fecha, piloto, equipo) y el resultado vacio, y se
anade al final del historico.

A partir de ahi no hace falta nada mas: el pipeline de features se ejecuta
sin cambios y el `shift(1)` del anti-fuga hace el resto -- cada fila fantasma
ve todo el pasado y nada de si misma, exactamente igual que una fila real.
No hay una segunda implementacion de las features que pueda divergir.

Lo que queda NaN de forma legitima:
  - antes de la clasificacion: quali_position, gap_to_pole, vueltas de quali
  - siempre: neumaticos/ritmo de ESA carrera (no ha ocurrido)
XGBoost maneja NaN de forma nativa, asi que la prediccion sale igual, solo
con mas incertidumbre.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def get_next_race(db, after_date: pd.Timestamp | None = None) -> dict | None:
    """Primera ronda del calendario cuya fecha es posterior a `after_date`
    (por defecto, hoy) y que todavia no tiene resultados."""
    after_date = after_date or pd.Timestamp.today().normalize()

    schedule = list(db["race_schedule"].find({}, {"_id": 0}))
    if not schedule:
        return None

    sched = pd.DataFrame(schedule)
    sched["race_date"] = pd.to_datetime(sched["race_date"])
    future = sched[sched["race_date"] >= after_date].sort_values("race_date")
    if future.empty:
        return None

    for _, race in future.iterrows():
        already_run = db["historical_results"].count_documents(
            {"year": int(race["year"]), "round": int(race["round"])}
        )
        if already_run == 0:
            return race.to_dict()
    return None


def _lineup_from_last_race(db, year: int) -> pd.DataFrame:
    """Parrilla esperada: la de la ultima carrera disputada.

    Asuncion explicita y con fallo conocido: no capta pilotos sustitutos ni
    debuts anunciados. Es la mejor estimacion disponible sin una fuente de
    inscripciones oficial.
    """
    last_round = max(db["historical_results"].distinct("round", {"year": year}), default=None)
    if last_round is None:
        return pd.DataFrame()

    rows = list(db["historical_results"].find(
        {"year": year, "round": last_round},
        {"_id": 0, "driver_code": 1, "driver_id": 1, "driver_full_name": 1,
         "constructor_id": 1, "constructor_name": 1},
    ))
    logger.info(f"Parrilla tomada de {year} ronda {last_round}: {len(rows)} pilotos")
    return pd.DataFrame(rows)


def build_stub_rows(db, year: int, round_number: int) -> pd.DataFrame:
    """Una fila por piloto para la carrera (year, round), con el mismo
    esquema que produce build_context() pero sin resultado."""
    race = db["race_schedule"].find_one({"year": year, "round": round_number}, {"_id": 0})
    if race is None:
        raise ValueError(f"La ronda {year} R{round_number} no esta en race_schedule")

    lineup = _lineup_from_last_race(db, year)
    if lineup.empty:
        raise ValueError(f"No hay ninguna carrera disputada en {year} de la que deducir la parrilla")

    circuit = db["circuits"].find_one({"jolpica_circuit_id": race["circuit_id"]}, {"_id": 0})
    if circuit is None:
        raise ValueError(
            f"Circuito '{race['circuit_id']}' no esta en circuits.json -- anadelo antes de predecir"
        )

    stub = lineup.copy()
    stub["year"] = year
    stub["round"] = round_number
    stub["circuit_id"] = race["circuit_id"]
    stub["race_date"] = pd.to_datetime(race["race_date"])

    stub["circuit_short_name"] = circuit["circuit_short_name"]
    stub["circuit_type"] = circuit["circuit_type"]
    stub["altitude_m"] = circuit["altitude_m"]
    stub["length_km"] = circuit["length_km"]
    stub["overtaking_difficulty"] = circuit["overtaking_difficulty"]
    stub["country"] = circuit["country"]
    stub["is_street_circuit"] = circuit["circuit_type"] == "street"
    stub["is_hybrid_circuit"] = circuit["circuit_type"] == "hybrid"
    stub["era_data_tier"] = "openf1_full" if year >= 2023 else "jolpica_basic"

    # Lo que define que esta carrera aun no ha ocurrido. Se usa NaN (float)
    # y no pd.NA en los campos numericos: pd.NA convertiria la columna a
    # dtype object y romperia los rollings del anti-fuga.
    stub["grid_position"] = float("nan")
    stub["finish_position"] = float("nan")
    stub["points"] = float("nan")
    stub["status"] = None
    stub["position_text"] = None
    stub["finished"] = pd.NA  # booleano nullable: `~finished` propaga NA sin romper

    return stub
