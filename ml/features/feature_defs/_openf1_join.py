"""Helper compartido por tyre_strategy.py, dynamic_pace.py y las futuras
features de quali/sprint: mapea filas de laps/stints/pit (identificadas por
session_key+driver_number) al espacio de claves de context_df
(circuit_short_name+fecha+driver_code), pasando por sessions
(session_key->circuit+fecha+tipo) y driver_mapping (driver_number->driver_code).

IMPORTANTE: `sessions` ahora contiene 4 tipos de sesion por fin de semana
(Race/Qualifying/Sprint Qualifying/Sprint, ver AppFormula1.java). Todo
consumidor de with_driver_code_and_race_key() DEBE fijar session_names
explicitamente -- unir sin filtrar mezclaria vueltas de clasificacion con
vueltas de carrera bajo la misma clave.
"""

import logging

import pandas as pd

from ml.features.feature_defs.driver_mapping import build_driver_number_to_code

logger = logging.getLogger(__name__)


def openf1_to_canonical_circuit(db) -> dict[str, str]:
    """{nombre OpenF1 -> nombre canonico de circuits.json}.

    OpenF1 llama a varios circuitos de forma distinta ("Catalunya" por
    "Barcelona", "Spa-Francorchamps" por "Spa", "Monte Carlo" sin guion...).
    Sin traducir, el join contra context_df falla en silencio y ~30% de las
    filas de 2023+ pierden sus features de OpenF1 sin dejar rastro (las que
    son medias moviles ni siquiera salen NaN: se calculan sobre una ventana
    incompleta). Ver `openf1_names` en ml/config/circuits.json.
    """
    mapping = {}
    for circuit in db["circuits"].find({}, {"_id": 0, "circuit_short_name": 1, "openf1_names": 1}):
        canonical = circuit["circuit_short_name"]
        mapping[canonical] = canonical
        for alias in circuit.get("openf1_names", []):
            mapping[alias] = canonical
    return mapping


def sessions_lookup(db, session_names: list[str] = ("Race",)) -> pd.DataFrame:
    sessions = list(
        db["sessions"].find(
            {"session_name": {"$in": list(session_names)}},
            {"_id": 0, "session_key": 1, "circuit_short_name": 1, "date_start": 1},
        )
    )
    df = pd.DataFrame(sessions)
    if df.empty:
        return df

    mapping = openf1_to_canonical_circuit(db)
    unknown = set(df["circuit_short_name"]) - set(mapping)
    if unknown:
        logger.warning(
            f"Circuitos de OpenF1 sin equivalente en circuits.json (sus features quedaran "
            f"sin unir): {sorted(unknown)}. Anadelos o define su alias en openf1_names."
        )
    df["circuit_short_name"] = df["circuit_short_name"].map(lambda c: mapping.get(c, c))

    # tz-naive para poder unir con context_df (Jolpica no trae tz); OpenF1
    # date_start viene en UTC, aqui solo importa el dia calendario.
    df["race_date"] = pd.to_datetime(df["date_start"]).dt.tz_localize(None).dt.normalize()
    return df[["session_key", "circuit_short_name", "race_date"]]


def with_driver_code_and_race_key(
    db, records_df: pd.DataFrame, session_names: list[str] = ("Race",)
) -> pd.DataFrame:
    """records_df debe tener session_key y driver_number. Devuelve con
    driver_code, circuit_short_name, race_date anadidos (filas sin mapeo se
    descartan). session_names restringe a que tipo(s) de sesion pertenecen
    los session_key esperados -- por defecto solo Race, para no mezclar con
    Qualifying/Sprint sin querer."""
    if records_df.empty:
        return records_df

    mapping = build_driver_number_to_code(db)
    sessions = sessions_lookup(db, session_names)
    if mapping.empty or sessions.empty:
        return pd.DataFrame()

    merged = records_df.merge(mapping, on=["session_key", "driver_number"], how="inner")
    merged = merged.merge(sessions, on="session_key", how="inner")
    return merged
