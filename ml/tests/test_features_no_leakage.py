"""Verifica que los helpers de forma historica (ml/features/anti_leakage.py)
nunca usan datos de la carrera actual o futuras -- la regla anti-fuga del plan.
"""

import pandas as pd
import pytest

from ml.features.anti_leakage import (
    expanding_count_shifted,
    expanding_mean_shifted,
    rolling_mean_shifted,
)

SAMPLE = pd.DataFrame(
    {
        "driver_code": ["VER", "VER", "VER", "VER", "HAM", "HAM"],
        "race_date": [
            "2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01",
            "2023-01-01", "2023-02-01",
        ],
        "finish_position": [1, 2, 1, 3, 5, 4],
    }
)


def test_first_race_per_driver_has_no_history():
    result = rolling_mean_shifted(SAMPLE, "driver_code", "race_date", "finish_position", 2)
    first_ver = result[SAMPLE["driver_code"] == "VER"].iloc[0]
    first_ham = result[SAMPLE["driver_code"] == "HAM"].iloc[0]
    assert pd.isna(first_ver)
    assert pd.isna(first_ham)


def test_rolling_mean_excludes_current_race():
    result = rolling_mean_shifted(SAMPLE, "driver_code", "race_date", "finish_position", 2)
    # VER 2023-02-01: solo debe ver la carrera de 2023-01-01 (posicion 1)
    assert result.iloc[1] == pytest.approx(1.0)
    # VER 2023-03-01: media de 2023-01-01 y 2023-02-01 (1, 2)
    assert result.iloc[2] == pytest.approx(1.5)
    # Si incluyera la carrera actual (posicion 1 el 2023-03-01), daria 4/3 != 1.5
    assert result.iloc[2] != pytest.approx((1 + 2 + 1) / 3)


def test_expanding_mean_grows_only_with_past_races():
    result = expanding_mean_shifted(SAMPLE, "driver_code", "race_date", "finish_position")
    ver_rows = result[SAMPLE["driver_code"] == "VER"].reset_index(drop=True)
    assert pd.isna(ver_rows[0])
    assert ver_rows[1] == pytest.approx(1.0)
    assert ver_rows[2] == pytest.approx(1.5)
    assert ver_rows[3] == pytest.approx((1 + 2 + 1) / 3)


def test_expanding_count_is_prior_race_count_not_total():
    result = expanding_count_shifted(SAMPLE, "driver_code", "race_date")
    ver_rows = result[SAMPLE["driver_code"] == "VER"].reset_index(drop=True)
    assert list(ver_rows) == [0, 1, 2, 3]


def test_groups_are_independent():
    result = rolling_mean_shifted(SAMPLE, "driver_code", "race_date", "finish_position", 10)
    # HAM 2023-02-01 no debe verse afectado por el historial de VER
    ham_second = result[SAMPLE["driver_code"] == "HAM"].iloc[1]
    assert ham_second == pytest.approx(5.0)


def test_out_of_order_rows_still_respect_temporal_order():
    shuffled = SAMPLE.sample(frac=1, random_state=42).reset_index(drop=True)
    result = rolling_mean_shifted(shuffled, "driver_code", "race_date", "finish_position", 2)
    row_0401 = shuffled[shuffled["race_date"] == "2023-04-01"].index[0]
    assert result.loc[row_0401] == pytest.approx(1.5)
