# Operations Playbook

## Forecast Cadence

Run forecasts daily after promotion and holiday calendars are finalized. The forecast horizon should match replenishment lead times used by the supply planning team.

## Inventory Policy

Use the ensemble forecast as the demand signal and calculate reorder point as:

```text
Reorder Point = average_daily_demand * lead_time_days + safety_stock
```

Safety stock should be recalculated from recent forecast-error volatility:

```text
Safety Stock = Z(service level) * sigma(forecast error) * sqrt(lead time)
```

## Recommended Service Levels

| Segment | Suggested Service Level | Rationale |
|---|---:|---|
| High-volume stores | 95% to 99% | Stockouts have larger revenue impact |
| Medium-volume stores | 95% | Balanced cost and availability |
| Low-volume stores | 90% to 95% | Lower carrying-cost tolerance |

## Monitoring

Track these metrics weekly:

- MAPE by store type
- Mean forecast bias by store type and month
- Forecast-error sigma by store
- Safety-stock change by service level and lead time
- Stores with persistent under-forecasting

## Alert Rules

| Alert | Trigger |
|---|---|
| Accuracy degradation | Four-week MAPE rises above 12% |
| Bias drift | Segment bias exceeds 2% of average sales |
| Inventory volatility | Forecast-error sigma shifts by more than 10% |
| Promotion miss | Promo-week MAPE exceeds non-promo MAPE by more than 5pp |

## Implementation Roadmap

1. Run daily batch forecast.
2. Publish forecast and confidence bands to planning dashboard.
3. Recompute safety stock weekly using trailing forecast errors.
4. Review bias heatmap monthly with store operations.
5. Retrain after major promo-policy, assortment, or competitive changes.
