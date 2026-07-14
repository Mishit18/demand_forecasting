# Rossmann Demand Forecasting: Strategy and Operations Summary

## Dataset Overview

- Clean forecastable demand rows: 844,338
- Model-ready rows after causal lag warmup: 781,898
- Stores modeled: 1,115
- Date range: 2013-01-01 to 2015-07-31
- Time split: training 2013-02-26 to 2015-06-19, holdout 2015-06-20 to 2015-07-31
- Engineered features: 44

## Model Results

| Model              | CV MAPE Mean   | CV MAPE Std   | Holdout MAPE   |
|:-------------------|:---------------|:--------------|:---------------|
| Naive              |                |               | 20.07%         |
| Rolling Mean       |                |               | 24.47%         |
| Same Day Last Week |                |               | 37.64%         |
| Store Median       |                |               | 21.92%         |
| SARIMAX            | 13.97%         | 6.31%         | 24.47%         |
| XGBoost            | 11.30%         | 1.40%         | 9.11%          |
| LightGBM           | 11.15%         | 1.12%         | 8.98%          |
| Ensemble           |                |               | 8.94%          |

Best ensemble weights: {
  "Naive": 0.0,
  "XGBoost": 0.20832519844287758,
  "LightGBM": 0.7916748015571224
}

## Safety Stock and Working Capital Impact

Improving MAPE from 20.07% (naive) to 8.94% (ensemble) reduces required safety stock by 58.2% at a 95% service level and seven-day lead time, freeing Rs. 76,843 in working capital per store per year under a 20% annual holding-cost assumption.

## Bias Findings

The largest systematic segment bias is StoreType d in month 6, with mean bias of -124 units (-1.82% of average sales). StoreType-level multiplicative correction factors reduce mean absolute forecast error by 1.0%.

## EDA Highlights

- Daily sales mean Rs./units proxy: 6,956; median: 6,369; std: 3,104; skewness: 1.59; kurtosis: 4.85
- Promo median lift: 40.1% with Mann-Whitney p-value 0.00e+00
- StoreType effect is statistically significant with ANOVA p-value 0.00e+00

## Resume-Ready Bullets

- Built 8-model demand forecasting benchmark on 844,338 Rossmann retail records; achieved 8.94% MAPE on six-week holdout across 1,115 stores.
- Engineered 44 lag, rolling, calendar, promo and store features with strict causal constraints; Optuna-tuned XGBoost achieved 9.11% MAPE vs. 20.07% naive baseline.
- Translated 11.12pp MAPE improvement into 58.2% safety-stock reduction at 95% service level; estimated Rs. 76,843 working capital freed per store annually.
- Diagnosed systematic forecast bias in StoreType d month 6; correction reduced mean forecast error by 1.0%, improving reorder-point accuracy.
