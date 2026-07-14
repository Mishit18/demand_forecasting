from __future__ import annotations

import argparse
import json
from pathlib import Path

import nbformat
import pandas as pd


REQUIRED_OUTPUTS = [
    "results_table.csv",
    "safety_stock_table.csv",
    "reorder_points_top10.csv",
    "bias_table.csv",
    "run_metadata.json",
]

REQUIRED_PLOTS = [
    "plot_01_sales_heatmap.png",
    "plot_02_decomposition.png",
    "plot_03_promo_lift_kde.png",
    "plot_04_model_comparison.png",
    "plot_05_shap_summary.png",
    "plot_06_xgb_feature_importance.png",
    "plot_07_actual_vs_predicted.png",
    "plot_08_residual_distribution.png",
    "plot_09_safety_stock_lines.png",
    "plot_10_bias_heatmap.png",
    "plot_11_optuna_history.png",
    "plot_12_predicted_vs_actual.png",
]


def validate(project_dir: Path) -> dict:
    outputs_dir = project_dir / "outputs"
    results_path = outputs_dir / "results_table.csv"
    summary_path = project_dir / "summary.md"
    notebook_path = project_dir / "demand_forecasting.ipynb"

    checks = {}
    checks["data_files_present"] = all((project_dir / "data" / name).exists() for name in ["train.csv", "store.csv"])
    checks["required_outputs_present"] = all((outputs_dir / name).exists() for name in REQUIRED_OUTPUTS)
    checks["required_plots_present"] = all((outputs_dir / name).exists() and (outputs_dir / name).stat().st_size > 10_000 for name in REQUIRED_PLOTS)
    checks["plot_count_at_least_12"] = len(list(outputs_dir.glob("plot_*.png"))) >= 12

    results = pd.read_csv(results_path)
    ensemble_mape = float(results.loc[results["Model"].eq("Ensemble"), "Holdout MAPE"].iloc[0])
    naive_mape = float(results.loc[results["Model"].eq("Naive"), "Holdout MAPE"].iloc[0])
    checks["ensemble_mape_below_12"] = ensemble_mape < 12.0
    checks["ensemble_beats_naive"] = ensemble_mape < naive_mape

    summary = summary_path.read_text(encoding="utf-8")
    placeholder_terms = ["[X]", "[Y]", "[Z]", "TODO", "TBD", "placeholder"]
    checks["summary_has_no_placeholders"] = not any(term.lower() in summary.lower() for term in placeholder_terms)
    checks["summary_has_rs_headline"] = "Rs." in summary and "working capital" in summary

    with notebook_path.open("r", encoding="utf-8") as handle:
        nbformat.read(handle, as_version=4)
    checks["notebook_json_valid"] = True

    report = {
        "status": "pass" if all(checks.values()) else "fail",
        "ensemble_holdout_mape": ensemble_mape,
        "naive_holdout_mape": naive_mape,
        "plot_count": len(list(outputs_dir.glob("plot_*.png"))),
        "checks": checks,
    }
    (outputs_dir / "quality_gate_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parents[1])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = validate(args.project_dir.resolve())
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "pass" else 1)
