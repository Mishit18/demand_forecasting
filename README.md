# Rossmann Demand Forecasting and Safety Stock

Production-oriented demand forecasting project for the Rossmann Store Sales dataset.

## Run

Place `train.csv` and `store.csv` in `data/`, then run:

```powershell
python src/run_pipeline.py --project-dir .
```

The notebook `demand_forecasting.ipynb` calls the same tested pipeline and writes all artifacts to `outputs/`.

## Outputs

- `outputs/results_table.csv`
- `outputs/plot_01_sales_heatmap.png` through `outputs/plot_12_predicted_vs_actual.png`
- `summary.md`

The pipeline uses causal lag and rolling features, a date-based six-week holdout, baseline models, SARIMAX on high-volume stores, Optuna-tuned XGBoost and LightGBM, an optimized ensemble, safety-stock economics, and forecast-bias diagnostics.
