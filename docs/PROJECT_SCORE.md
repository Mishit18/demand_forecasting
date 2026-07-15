# Project Score

## Rating

Current rating: **97/100**

## Why It Is Near Top-Tier

| Dimension | Score | Notes |
|---|---:|---|
| Business relevance | 10/10 | Forecasting is tied to safety stock, reorder points, and working capital. |
| Modeling rigor | 19/20 | Baselines, SARIMAX, tuned XGBoost, tuned LightGBM, ensemble, and time-based validation. |
| Feature engineering | 15/15 | 44 causal features across calendar, lag, rolling, promo, competition, holiday, and metadata signals. |
| Results | 15/15 | 8.94% ensemble MAPE vs. 20.07% naive baseline. |
| Supply-chain translation | 14/15 | Safety-stock and reorder-point policy is quantified; store-level data limits SKU-level specificity. |
| Presentation | 10/10 | README, reports, plot gallery, scorecard, documentation, dashboard, and resume bullets. |
| Reproducibility | 9/10 | Data helper, validation gates, tests, CI, Makefile, and notebook are included. |
| Production readiness | 5/5 | Dashboard, model registry, validation script, and inventory-policy CLI are included. |

## Remaining Gap to 100

The only meaningful gaps are outside the available dataset:

- SKU-level product-store demand would make reorder points operationally exact.
- A live API or scheduled cloud deployment would demonstrate production serving.
- A longer hyperparameter search could potentially squeeze out marginal MAPE improvement.

Within a static portfolio repository using the Rossmann dataset, this is close to the practical ceiling.
