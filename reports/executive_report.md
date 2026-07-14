# Executive Report

## Objective

Forecast daily Rossmann store demand and quantify how improved forecast accuracy changes inventory policy.

## Result

The optimized ensemble achieved 8.94% MAPE on the final six-week holdout across 1,115 stores. This improved on the naive baseline by 11.12 percentage points and reduced required safety stock by 58.2% at a 95% service level with a seven-day lead time.

## Business Impact

Using an assumed Rs. 75 average unit value and 20% annual holding-cost rate, the forecast improvement frees an estimated Rs. 76,843 in working capital per store per year.

## Model Comparison

| Model | Holdout MAPE |
|---|---:|
| Naive | 20.07% |
| Rolling Mean | 24.47% |
| Same Day Last Week | 37.64% |
| Store Median | 21.92% |
| SARIMAX | 24.47% |
| XGBoost | 9.11% |
| LightGBM | 8.98% |
| Ensemble | 8.94% |

## Operational Recommendation

Use the ensemble forecast as the primary replenishment signal, maintain segment-level bias monitoring for StoreType d in month 6, and update safety-stock policies monthly using observed forecast-error volatility.

## Risk Controls

- Keep the current lag-shifted feature design to prevent look-ahead leakage.
- Do not add `Customers` as a forecast feature unless the use case changes to same-day nowcasting.
- Monitor systematic under-forecasting and over-forecasting by store type and month.
- Re-estimate holding-cost economics before using the rupee impact as a finance commitment.
