from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]


def test_ensemble_clears_accuracy_gate():
    results = pd.read_csv(PROJECT_DIR / "outputs" / "results_table.csv")
    ensemble = float(results.loc[results["Model"].eq("Ensemble"), "Holdout MAPE"].iloc[0])
    naive = float(results.loc[results["Model"].eq("Naive"), "Holdout MAPE"].iloc[0])

    assert ensemble < 12.0
    assert ensemble < naive


def test_required_plots_exist():
    plots = list((PROJECT_DIR / "outputs").glob("plot_*.png"))
    assert len(plots) >= 12
    assert all(plot.stat().st_size > 10_000 for plot in plots)


def test_summary_is_fully_filled():
    summary = (PROJECT_DIR / "summary.md").read_text(encoding="utf-8")
    forbidden = ["[X]", "[Y]", "[Z]", "TODO", "TBD", "placeholder"]

    assert not any(term.lower() in summary.lower() for term in forbidden)
    assert "8.94%" in summary
    assert "Rs. 76,843" in summary


def test_kpi_scorecard_is_present_and_consistent():
    scorecard = pd.read_csv(PROJECT_DIR / "outputs" / "kpi_scorecard.csv")
    metrics = dict(zip(scorecard["metric"], scorecard["value"]))

    assert float(metrics["ensemble_mape"]) < 12.0
    assert metrics["quality_gate_status"] == "pass"
    assert int(float(metrics["plot_count"])) >= 12
