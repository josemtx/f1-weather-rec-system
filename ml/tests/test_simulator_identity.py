"""Un piloto desconocido para el modelo sigue teniendo nombre.

`driver_code` es a la vez una feature categorica y la etiqueta con la que se
identifica cada prediccion. _prep_categoricals convierte en NaN las
categorias no vistas en entrenamiento, que es lo correcto para el MODELO
(de un debutante no se sabe nada como efecto fijo) pero destruiria la
etiqueta si la salida se leyera despues de esa conversion.

Ese bug llego a produccion: la prediccion de Lindblad se guardo en MongoDB
con driver_code NaN y el dashboard no pudo ni leer el JSON. Estos tests
fijan las dos mitades del contrato.
"""

import numpy as np
import pandas as pd

from ml.simulation.monte_carlo import RaceSimulator


class _Stub:
    """Permite ejercitar _prep_categoricals sin cargar modelos de disco."""

    def __init__(self, categories):
        self.categories = categories

    prep = RaceSimulator._prep_categoricals


def _frame():
    return pd.DataFrame({
        "driver_code": ["VER", "HAM", "LIN"],
        "constructor_id": ["red_bull", "ferrari", "cadillac"],
        "circuit_short_name": ["Monza", "Monza", "Monza"],
        "p_win": [0.4, 0.3, 0.01],
    })


def test_prep_anula_categorias_no_vistas():
    """El MODELO no debe recibir una categoria que nunca vio."""
    stub = _Stub({"driver_code": ["VER", "HAM"], "constructor_id": ["red_bull", "ferrari"]})
    out = stub.prep(_frame())

    assert out["driver_code"].iloc[0] == "VER"
    assert pd.isna(out["driver_code"].iloc[2]), "LIN no estaba en entrenamiento: debe llegar como NaN"
    assert pd.isna(out["constructor_id"].iloc[2]), "cadillac tampoco estaba"


def test_prep_no_muta_el_frame_original():
    """La captura de etiquetas depende de que el frame de entrada siga intacto."""
    original = _frame()
    _Stub({"driver_code": ["VER", "HAM"]}).prep(original)

    assert list(original["driver_code"]) == ["VER", "HAM", "LIN"], (
        "prep debe trabajar sobre una copia: si mutara el original, "
        "los codigos se perderian aunque se capturasen antes"
    )


def test_las_etiquetas_se_capturan_antes_de_anular():
    """Reproduce el orden correcto de simulate_race."""
    race = _frame()
    codes = race["driver_code"].to_numpy()          # antes
    prepped = _Stub({"driver_code": ["VER", "HAM"]}).prep(race)   # despues

    assert list(codes) == ["VER", "HAM", "LIN"]
    assert pd.isna(prepped["driver_code"].iloc[2])
    assert not any(isinstance(c, float) and np.isnan(c) for c in codes)
