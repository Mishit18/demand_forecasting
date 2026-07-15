from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_DIR / "outputs"


def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(OUTPUTS_DIR / name)


def main() -> None:
    st.set_page_config(page_title="Rossmann Forecasting", layout="wide")
    st.title("Rossmann Demand Forecasting and Inventory Optimization")
    st.caption("Forecast accuracy, safety stock, reorder points, and bias diagnostics.")

    scorecard = load_csv("kpi_scorecard.csv")
    kpis = dict(zip(scorecard["metric"], scorecard["value"]))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ensemble MAPE", f"{float(kpis['ensemble_mape']):.2f}%")
    c2.metric("Naive MAPE", f"{float(kpis['naive_mape']):.2f}%")
    c3.metric("Safety Stock Reduction", f"{float(kpis['safety_stock_reduction']):.1f}%")
    c4.metric("Rs. Freed / Store / Year", f"Rs. {float(kpis['working_capital_freed_per_store']):,.0f}")

    st.subheader("Model Performance")
    results = load_csv("results_table.csv")
    st.dataframe(results, use_container_width=True, hide_index=True)
    st.image(str(OUTPUTS_DIR / "plot_04_model_comparison.png"), caption="Holdout MAPE by model")

    left, right = st.columns(2)
    with left:
        st.subheader("Forecast Fit")
        st.image(str(OUTPUTS_DIR / "plot_07_actual_vs_predicted.png"), caption="Actual vs predicted across representative stores")
        st.image(str(OUTPUTS_DIR / "plot_12_predicted_vs_actual.png"), caption="Predicted vs actual scatter")
    with right:
        st.subheader("Drivers and Bias")
        st.image(str(OUTPUTS_DIR / "plot_05_shap_summary.png"), caption="SHAP feature impact")
        st.image(str(OUTPUTS_DIR / "plot_10_bias_heatmap.png"), caption="Bias by store type and month")

    st.subheader("Inventory Policy")
    service = st.selectbox("Service level", [0.90, 0.95, 0.99], index=1)
    lead = st.selectbox("Lead time days", [3, 7, 14], index=1)
    safety_stock = load_csv("safety_stock_table.csv")
    filtered = safety_stock[(safety_stock["Service Level"] == service) & (safety_stock["Lead Time Days"] == lead)]
    st.dataframe(filtered, use_container_width=True, hide_index=True)
    st.image(str(OUTPUTS_DIR / "plot_09_safety_stock_lines.png"), caption="Safety stock by service level")

    st.subheader("Top Reorder Points")
    st.dataframe(load_csv("reorder_points_top10.csv"), use_container_width=True, hide_index=True)

    with st.expander("Model registry"):
        registry = json.loads((PROJECT_DIR / "config" / "model_registry.json").read_text(encoding="utf-8"))
        st.json(registry)


if __name__ == "__main__":
    main()
