# Interview Guide

## One-Minute Pitch

I built an end-to-end demand forecasting and inventory optimization project on 844k Rossmann store-day records across 1,115 stores. The pipeline uses causal lag, rolling, calendar, promo, holiday, competition, and store metadata features, then benchmarks operational baselines, SARIMAX, XGBoost, LightGBM, and an optimized ensemble. The final ensemble achieved 8.94% MAPE on a six-week holdout, down from a 20.07% naive baseline. I translated the error reduction into a 58.2% safety-stock reduction at a 95% service level, equivalent to Rs. 76,843 working capital freed per store per year under explicit inventory-cost assumptions.

## Decisions Worth Discussing

- Used a date-based holdout instead of random split because this is a forecasting problem.
- Excluded `Customers` from features because it would not be available before demand occurs.
- Generated lag and rolling features with shifted sales to prevent look-ahead leakage.
- Compared ML models against operational baselines before trusting model complexity.
- Converted model accuracy into safety stock and reorder points to connect analytics to business decisions.

## Likely Questions

### Why MAPE?

MAPE is easy to explain to operations teams and normalizes forecast error across stores of different demand levels. I filtered to positive-demand open days, so division-by-zero issues are avoided.

### Why did SARIMAX underperform?

SARIMAX is fitted on store-level univariate dynamics and cannot exploit cross-store metadata, promo interactions, and nonlinear rolling features as effectively as gradient boosting.

### Why did the ensemble only slightly beat LightGBM?

LightGBM was already the strongest individual model. The blend still improved holdout MAPE because XGBoost captured some complementary patterns, but the optimized weights correctly assigned most weight to LightGBM.

### How would this become production-ready?

Add scheduled retraining, feature-store snapshots, data freshness checks, forecast monitoring, bias alerts, and automated recalculation of safety stock by store segment.
