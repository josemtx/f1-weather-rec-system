"""El orquestador del fin de semana solo actua cuando toca, y cada regimen se registra una vez."""

from datetime import date

import pandas as pd

from ml.automation import weekend
from ml.tests.test_prediction_log import _FakeDB


def _next_race(monkeypatch, race_date: str):
    monkeypatch.setattr(weekend, "get_next_race", lambda db: {"year": 2026, "round": 15, "race_date": pd.Timestamp(race_date)})


def test_fuera_de_semana_de_carrera_no_predice(monkeypatch):
    _next_race(monkeypatch, "2026-09-26")
    calls = []
    monkeypatch.setattr(weekend, "predict_and_store", lambda db, y, r: calls.append((y, r)))
    assert weekend.step_predict_if_due(_FakeDB(), date(2026, 9, 14), days_ahead=3) == (False, False)
    assert calls == []


def test_en_semana_de_carrera_registra_una_vez_por_regimen(monkeypatch):
    _next_race(monkeypatch, "2026-09-26")
    calls = []
    monkeypatch.setattr(weekend, "predict_and_store", lambda db, y, r: calls.append((y, r)))
    db = _FakeDB(qualifying_results=[], sprint_results=[], predictions=[])

    assert weekend.step_predict_if_due(db, date(2026, 9, 25), days_ahead=3) == (True, True)
    assert calls == [(2026, 15)]

    # La pasada siguiente, mismo regimen: ya registrada, no repite.
    db["predictions"].docs.append({"year": 2026, "round": 15, "information_regime": "pre_quali"})
    assert weekend.step_predict_if_due(db, date(2026, 9, 25), days_ahead=3) == (True, False)
    assert calls == [(2026, 15)]

    # Entra la clasificacion: regimen nuevo, se registra otra vez.
    db["qualifying_results"].docs.append({"year": 2026, "round": 15})
    assert weekend.step_predict_if_due(db, date(2026, 9, 25), days_ahead=3) == (True, True)
    assert calls == [(2026, 15), (2026, 15)]


def test_sin_calendario_no_hace_nada(monkeypatch):
    monkeypatch.setattr(weekend, "get_next_race", lambda db: None)
    assert weekend.step_predict_if_due(_FakeDB(), date(2026, 9, 25), days_ahead=3) == (False, False)


def test_jolpica_caido_no_rompe_la_pasada(monkeypatch):
    def boom(years):
        raise ConnectionError("timeout")
    monkeypatch.setattr(weekend, "ingest_jolpica", boom)
    assert weekend.step_ingest_jolpica(_FakeDB(historical_results=[]), 2026) is False


def test_detecta_resultado_nuevo(monkeypatch):
    db = _FakeDB(historical_results=[])

    def fake_ingest(years):
        db["historical_results"].docs.append({"year": 2026, "round": 15})
        return 22, 0
    monkeypatch.setattr(weekend, "ingest_jolpica", fake_ingest)
    assert weekend.step_ingest_jolpica(db, 2026) is True
