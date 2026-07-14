# Portfolio Case Study

## Context

Retail demand forecasting is valuable only when forecast accuracy changes an operating decision. This project links model performance directly to safety stock, reorder points, and working-capital impact.

## Analytical Approach

The dataset contains daily sales for 1,115 Rossmann stores. Closed and zero-sales days were removed because they do not represent forecastable demand. The final model-ready table contains 781,898 rows after lag warmup.

Features were engineered with strict causal constraints. Sales lags and rolling statistics use shifted historical sales, and the customer count is excluded because it would not be available before the forecast day.

## Model Development

The project benchmarks four operational baselines, SARIMAX, XGBoost, LightGBM, and an optimized ensemble. XGBoost and LightGBM are tuned with Optuna and walk-forward validation. The ensemble is optimized on a validation fold and evaluated once on the final six-week holdout.

## Results

| Model | Holdout MAPE |
|---|---:|
| Naive | 20.07% |
| XGBoost | 9.11% |
| LightGBM | 8.98% |
| Ensemble | 8.94% |

The ensemble improves MAPE by 11.12 percentage points compared with the naive baseline.

## Business Impact

Forecast-error sigma falls from 2,022 units under the naive forecast to 845 units under the ensemble. At a 95% service level and seven-day lead time, this translates into a 58.2% reduction in required safety stock.

Using Rs. 75 average unit value and a 20% annual holding-cost rate, the estimated working-capital benefit is Rs. 76,843 per store per year.

## Bias Diagnosis

The largest systematic segment bias appears in StoreType d during month 6, where the model over-forecasts by 124 units on average, equal to 1.82% of average sales. A segment-level multiplicative correction reduces mean absolute forecast error by 1.0%.

## Strategic Takeaway

The project demonstrates a complete strategy-and-operations workflow: forecast demand, validate rigorously, quantify error reduction, translate the improvement into an inventory policy, and identify where the operating process still needs monitoring.
