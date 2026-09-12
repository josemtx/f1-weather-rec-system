"""El registro de predicciones solo vale si no se puede contaminar a posteriori.

Estos tests usan un doble de MongoDB en memoria: comprueban la logica de
integridad, no la base de datos.
"""

import pandas as pd
import pytest

from ml.predictions.prediction_log import (
    REGIME_POST_QUALI,
    REGIME_POST_SPRINT,
    REGIME_PRE_QUALI,
    detect_information_regime,
    race_already_run,
    save_predictions,
)


class _FakeCollection:
    def __init__(self, docs=None):
        self.docs = list(docs or [])
        self.replaced = []

    def count_documents(self, query):
        return sum(1 for d in self.docs if all(d.get(k) == v for k, v in query.items()))

    def create_index(self, *args, **kwargs):
        return None

    def replace_one(self, query, doc, upsert=False):
        self.replaced.append(doc)


class _FakeDB:
    def __init__(self, **collections):
        self.collections = {name: _FakeCollection(docs) for name, docs in collections.items()}

    def __getitem__(self, name):
        return self.collections.setdefault(name, _FakeCollection())


def _prediction_frame():
    return pd.DataFrame([{
        "driver_code": "VER", "p_win": 0.4, "p_podium": 0.7, "p_points": 0.9, "p_dnf": 0.1,
        "mean_finish_position": 3.2, "median_finish_position": 3.0,
        "p10_finish_position": 1.0, "p90_finish_position": 8.0,
    }])


def test_regimen_es_pre_quali_sin_datos_del_fin_de_semana():
    db = _FakeDB(qualifying_results=[], sprint_results=[])
    assert detect_information_regime(db, 2026, 14) == REGIME_PRE_QUALI


def test_regimen_pasa_a_post_quali_cuando_hay_clasificacion():
    db = _FakeDB(qualifying_results=[{"year": 2026, "round": 14}], sprint_results=[])
    assert detect_information_regime(db, 2026, 14) == REGIME_POST_QUALI


def test_regimen_post_sprint_si_hay_sprint_pero_no_clasificacion():
    db = _FakeDB(qualifying_results=[], sprint_results=[{"year": 2026, "round": 14}])
    assert detect_information_regime(db, 2026, 14) == REGIME_POST_SPRINT


def test_el_regimen_distingue_entre_carreras():
    """Una clasificacion de OTRA ronda no debe contar como la de esta."""
    db = _FakeDB(qualifying_results=[{"year": 2026, "round": 13}], sprint_results=[])
    assert detect_information_regime(db, 2026, 14) == REGIME_PRE_QUALI


def test_no_se_puede_registrar_una_prediccion_de_carrera_ya_disputada():
    db = _FakeDB(historical_results=[{"year": 2026, "round": 13}])
    assert race_already_run(db, 2026, 13)
    with pytest.raises(ValueError, match="ya tiene resultados"):
        save_predictions(db, 2026, 13, "Monza", _prediction_frame(), "m", "forecast", 100, 42)


def test_se_puede_forzar_a_proposito_para_rehacer_historico():
    db = _FakeDB(historical_results=[{"year": 2026, "round": 13}])
    saved = save_predictions(
        db, 2026, 13, "Monza", _prediction_frame(), "m", "sensors", 100, 42,
        allow_after_race=True,
    )
    assert saved == 1


def test_la_prediccion_guardada_conserva_procedencia():
    db = _FakeDB(historical_results=[], qualifying_results=[], sprint_results=[])
    save_predictions(db, 2026, 14, "Madring", _prediction_frame(), "modelo-x", "forecast", 5000, 42)
    doc = db["predictions"].replaced[0]
    assert doc["model_version"] == "modelo-x"
    assert doc["climate_source"] == "forecast"
    assert doc["information_regime"] == REGIME_PRE_QUALI
    assert doc["n_simulations"] == 5000
    assert doc["predicted_at"]
