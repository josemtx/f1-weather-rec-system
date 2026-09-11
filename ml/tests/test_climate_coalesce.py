"""ERA5 debe ganar sobre NASA POWER cuando ambos existen para circuito+fecha;
si solo hay NASA POWER, debe usarse sin problema (ver plan, seccion E)."""

import pandas as pd

from ml.features.feature_defs.climate import coalesce_climate


def test_era5_wins_when_both_sources_present():
    raw = pd.DataFrame(
        [
            {"circuit_short_name": "Sakhir", "date": "2023-03-05", "source": "nasa_power", "temp_2m_avg": 20.0},
            {"circuit_short_name": "Sakhir", "date": "2023-03-05", "source": "era5", "temp_2m_avg": 23.5},
        ]
    )
    result = coalesce_climate(raw)
    assert len(result) == 1
    assert result.iloc[0]["source"] == "era5"
    assert result.iloc[0]["temp_2m_avg"] == 23.5


def test_nasa_power_used_when_era5_missing():
    raw = pd.DataFrame(
        [{"circuit_short_name": "Sakhir", "date": "2023-03-05", "source": "nasa_power", "temp_2m_avg": 20.0}]
    )
    result = coalesce_climate(raw)
    assert len(result) == 1
    assert result.iloc[0]["source"] == "nasa_power"


def test_multiple_circuits_and_dates_independent():
    raw = pd.DataFrame(
        [
            {"circuit_short_name": "Sakhir", "date": "2023-03-05", "source": "nasa_power", "temp_2m_avg": 20.0},
            {"circuit_short_name": "Sakhir", "date": "2023-03-05", "source": "era5", "temp_2m_avg": 23.5},
            {"circuit_short_name": "Monza", "date": "2023-09-03", "source": "nasa_power", "temp_2m_avg": 28.0},
        ]
    )
    result = coalesce_climate(raw)
    assert len(result) == 2
    monza_row = result[result["circuit_short_name"] == "Monza"].iloc[0]
    assert monza_row["source"] == "nasa_power"
    assert monza_row["temp_2m_avg"] == 28.0
