# Reproducibility

## Environment

- Python: 3.11
- Random seed: 42
- Main dependencies: pandas, scikit-learn, XGBoost, LightGBM, Optuna, statsmodels, SHAP, matplotlib, seaborn
- NumPy is pinned below 2.0 for compatibility with the plotting and model binaries used in this project.

## Full Rebuild

```powershell
python -m pip install -r requirements.txt
python src/download_data.py --project-dir .
python src/run_pipeline.py --project-dir .
python src/validate_artifacts.py --project-dir .
python -m pytest
```

## Fast Smoke Rebuild

```powershell
python src/run_pipeline.py --project-dir . --n-trials 8 --cv-sample-frac 0.20 --sarimax-store-count 3
python src/validate_artifacts.py --project-dir .
```

## Validation Gates

The committed run passes these gates:

- Data files are present.
- Required output tables are present.
- Required plots are present and non-empty.
- At least 12 plots exist.
- Ensemble MAPE is below 12%.
- Ensemble beats the naive baseline.
- Summary contains no placeholders.
- Rupee working-capital headline exists.
- Notebook JSON is valid.

## Current Run

| Item | Value |
|---|---:|
| Naive holdout MAPE | 20.07% |
| Ensemble holdout MAPE | 8.94% |
| Plot count | 18 |
| Quality gate status | Pass |
