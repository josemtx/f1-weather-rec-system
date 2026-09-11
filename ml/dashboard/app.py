"""Dashboard Streamlit: predictor de carreras + reporte del modelo.

Consume directamente RaceSimulator y los artefactos de ml/models/ (no pasa
por la API Flask, evita depender de un servidor corriendo aparte).

Uso: streamlit run ml/dashboard/app.py  (desde la raiz del repo)
"""

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.common.db import get_db
from ml.simulation.monte_carlo import RaceSimulator
from ml.training.model_registry import MODELS_ROOT, get_production_dir

st.set_page_config(page_title="F1 Predictor", layout="wide")

FEATURES_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "features_v3.parquet"


@st.cache_data
def load_features() -> pd.DataFrame:
    return pd.read_parquet(FEATURES_PATH)


@st.cache_resource
def load_simulator() -> RaceSimulator:
    return RaceSimulator(get_db())


@st.cache_data
def load_metrics() -> dict:
    prod_dir = get_production_dir()
    return json.loads((prod_dir / "metrics.json").read_text(encoding="utf-8"))


@st.cache_data
def load_registry() -> dict:
    return json.loads((MODELS_ROOT / "registry.json").read_text(encoding="utf-8"))


def page_predictor():
    st.title("Predictor de carreras (2023-2026)")
    st.caption(
        "Backtesting: predicciones generadas con datos conocidos ANTES de cada carrera "
        "(sin fuga temporal), comparadas contra el resultado real. No es predicción de "
        "carreras futuras -- esa pieza (conexión con pronóstico) sigue pendiente."
    )

    df = load_features()
    races = df[["year", "round", "circuit_short_name"]].drop_duplicates().sort_values(["year", "round"])
    races["label"] = races["year"].astype(str) + " R" + races["round"].astype(str) + " - " + races["circuit_short_name"]

    default_idx = len(races) - 1
    selected_label = st.selectbox("Carrera", races["label"], index=default_idx)
    selected = races[races["label"] == selected_label].iloc[0]

    race = df[(df["year"] == selected["year"]) & (df["round"] == selected["round"])].copy()

    n_sims = st.slider("Número de simulaciones Monte Carlo", 500, 5000, 2000, step=500)

    if st.button("Simular carrera", type="primary"):
        with st.spinner("Simulando..."):
            simulator = load_simulator()
            result = simulator.simulate_race(race, n_simulations=n_sims, seed=42)

        actual = race[["driver_code", "grid_position", "finish_position"]].sort_values("finish_position")
        merged = result.merge(actual, on="driver_code", how="left")

        col1, col2 = st.columns([3, 2])
        with col1:
            st.subheader("Predicción vs. resultado real")
            display_cols = ["driver_code", "grid_position", "finish_position", "p_win", "p_podium", "p_points", "p_dnf", "mean_finish_position"]
            st.dataframe(
                merged[display_cols].style.format({
                    "p_win": "{:.1%}", "p_podium": "{:.1%}", "p_points": "{:.1%}",
                    "p_dnf": "{:.1%}", "mean_finish_position": "{:.2f}",
                }),
                use_container_width=True, hide_index=True,
            )

        with col2:
            st.subheader("Probabilidad de podio")
            fig = px.bar(
                merged.sort_values("p_podium", ascending=True).tail(10),
                x="p_podium", y="driver_code", orientation="h",
                labels={"p_podium": "P(podio)", "driver_code": ""},
            )
            st.plotly_chart(fig, use_container_width=True)

        top3_pred = merged.sort_values("mean_finish_position").head(3)["driver_code"].tolist()
        top3_real = actual.head(3)["driver_code"].tolist()
        hits = len(set(top3_pred) & set(top3_real))
        st.metric("Aciertos en el podio predicho (posición media) vs real", f"{hits}/3")


def page_model_report():
    st.title("Reporte del modelo")

    registry = load_registry()
    metrics = load_metrics()

    st.subheader(f"Versión en producción: `{registry['production']}`")
    st.caption(f"Historial de versiones: {', '.join(registry.get('history', []))}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MAE posición final (val 2025)", f"{metrics['finish_position']['mae_val_2025']:.2f}")
    col2.metric("AUC podio (val 2025)", f"{metrics['podium']['auc_val']:.3f}")
    col3.metric("AUC puntos (val 2025)", f"{metrics['points']['auc_val']:.3f}")
    col4.metric("AUC DNF (val 2025)", f"{metrics['dnf']['auc_val']:.3f}")

    st.subheader("Backtest walk-forward (estabilidad entre folds)")
    wf = metrics.get("walk_forward_backtest", {})
    if wf:
        wf_df = pd.DataFrame(wf).T
        wf_df.index.name = "fold"
        st.dataframe(wf_df, use_container_width=True)
    else:
        st.info("Sin datos de walk-forward en este modelo.")

    st.subheader("Importancia de features (SHAP, posición final)")
    shap_top = metrics["finish_position"].get("shap_top15", {})
    if shap_top:
        shap_df = pd.DataFrame(list(shap_top.items()), columns=["feature", "shap_value"]).sort_values("shap_value")
        fig = px.bar(shap_df, x="shap_value", y="feature", orientation="h")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Permutation importance (segunda opinión, val 2025)")
    perm_top = metrics["finish_position"].get("permutation_importance_top15", {})
    if perm_top:
        perm_df = pd.DataFrame(list(perm_top.items()), columns=["feature", "importance"]).sort_values("importance")
        fig2 = px.bar(perm_df, x="importance", y="feature", orientation="h")
        st.plotly_chart(fig2, use_container_width=True)


def main():
    st.sidebar.title("F1 Predictor")
    page = st.sidebar.radio("Página", ["Predictor de carreras", "Reporte del modelo"])

    if page == "Predictor de carreras":
        page_predictor()
    else:
        page_model_report()


if __name__ == "__main__":
    main()
