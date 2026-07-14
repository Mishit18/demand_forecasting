# Model Card

## Objective

Forecast daily store-level demand and translate forecast-error reduction into safety-stock and reorder-point recommendations.

## Validation Design

- Split: date-based final six-week holdout
- Cross-validation: walk-forward time-series validation for tuned tree models
- Metric: MAPE on positive-demand days
- Accuracy gate: ensemble holdout MAPE below 12%

## Models Benchmarked

| Method | Role |
|---|---|
| Naive lag-1 | Operational baseline |
| Seven-day rolling mean | Smoothing baseline |
| Same day last week | Weekly seasonality baseline |
| Store median | Store-level demand baseline |
| SARIMAX | Statistical time-series benchmark |
| XGBoost | Tuned nonlinear tabular model |
| LightGBM | Tuned gradient boosting model |
| Ensemble | Weighted blend optimized on validation fold |

## Final Performance

| Model | Holdout MAPE |
|---|---:|
| Naive | 20.07% |
| XGBoost | 9.11% |
| LightGBM | 8.98% |
| Ensemble | 8.94% |

## Feature Families

- Calendar: weekday, month, week, quarter, day of year
- Demand memory: daily lags, weekly lags, rolling means, rolling medians, rolling volatility, EWM
- Promotions: current promo flag, promo streak, days since/until promo
- Holidays: state and school holiday indicators, holiday distance features
- Store metadata: store type, assortment, Promo2, competition age and distance

## Operational Use

Forecast errors are converted into safety stock:

```text
Safety Stock = Z(service level) * sigma(forecast error) * sqrt(lead time)
```

The final ensemble reduces forecast-error volatility enough to lower safety-stock requirements by 58.2% at a 95% service level and seven-day lead time.

## Monitoring Recommendations

- Track MAPE and bias by store type and calendar month.
- Retrain after promotion policy changes or major seasonality shifts.
- Alert when rolling four-week bias exceeds 2% of average sales in any store segment.
- Recalibrate safety stock when forecast-error sigma shifts by more than 10%.
