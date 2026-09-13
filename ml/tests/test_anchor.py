import numpy as np
import pandas as pd

from ml.training.anchor import (
    REGIME_POST_QUALI,
    REGIME_PRE_QUALI,
    anchor_for,
    fill_grid_from_quali,
    regime_for,
)


def _race(grid, quali, pace=(3.0, 8.0, np.nan), form=(4.0, 9.0, 12.0), team=(5.0, 5.0, 11.0)):
    return pd.DataFrame({
        "grid_position": grid,
        "driver_quali_position": quali,
        "grid_penalty_positions": [np.nan] * 3,
        "driver_avg_finish_when_finished_last5": pace,
        "driver_avg_finish_last5": form,
        "team_avg_finish_last5": team,
    })


def test_regimen_lo_deciden_los_datos():
    assert regime_for(_race([np.nan] * 3, [np.nan] * 3)) == REGIME_PRE_QUALI
    assert regime_for(_race([np.nan] * 3, [1.0, 2.0, np.nan])) == REGIME_POST_QUALI
    assert regime_for(_race([1.0, 2.0, 3.0], [np.nan] * 3)) == REGIME_POST_QUALI


def test_ancla_post_quali_prefiere_parrilla_y_cae_a_clasificacion():
    race = _race([np.nan, 0.0, 5.0], [2.0, 4.0, 6.0])
    # sin parrilla -> clasificacion; pit lane (0) -> ultimo; parrilla conocida gana
    assert anchor_for(race, REGIME_POST_QUALI).tolist() == [2.0, 20.0, 5.0]


def test_ancla_pre_quali_usa_ritmo_y_cae_a_forma_y_equipo():
    race = _race([np.nan] * 3, [np.nan] * 3, pace=(3.0, np.nan, np.nan), form=(4.0, 9.0, np.nan), team=(5.0, 5.0, 11.0))
    assert anchor_for(race, REGIME_PRE_QUALI).tolist() == [3.0, 9.0, 11.0]


def test_relleno_de_parrilla_desde_clasificacion_solo_donde_falta():
    race = _race([np.nan, 7.0, np.nan], [2.0, 4.0, np.nan])
    race["grid_penalty_positions"] = [np.nan, 3.0, np.nan]
    out = fill_grid_from_quali(race)
    assert out["grid_position"].tolist()[:2] == [2.0, 7.0]
    assert np.isnan(out["grid_position"].iloc[2])
    assert out["grid_penalty_positions"].tolist()[:2] == [0.0, 3.0]
    assert np.isnan(out["grid_penalty_positions"].iloc[2])
    # el original no se muta
    assert np.isnan(race["grid_position"].iloc[0])


def test_relleno_no_toca_una_carrera_ya_corrida():
    race = _race([1.0, 2.0, 3.0], [np.nan] * 3)
    assert fill_grid_from_quali(race) is race
