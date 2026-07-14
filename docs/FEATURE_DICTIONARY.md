# Feature Dictionary

The feature set is designed around information that would be available before the forecasted sales day.

## Calendar

| Feature | Description |
|---|---|
| `day_of_week` | Day index from 0 to 6 |
| `month` | Calendar month |
| `week_of_year` | ISO week |
| `year` | Calendar year |
| `is_weekend` | Weekend indicator |
| `is_month_start` | First three days of month |
| `is_month_end` | Day 28 or later |
| `quarter` | Calendar quarter |
| `day_of_year` | Annual seasonality index |

## Demand Memory

| Feature | Description |
|---|---|
| `sales_lag_1`, `sales_lag_2`, `sales_lag_3` | Recent demand history |
| `sales_lag_7`, `sales_lag_14`, `sales_lag_21` | Weekly seasonal memory |
| `sales_lag_28`, `sales_lag_56` | Monthly seasonal memory |
| `rolling_mean_7`, `rolling_mean_14`, `rolling_mean_28` | Shifted rolling demand level |
| `rolling_std_7`, `rolling_std_28` | Shifted demand volatility |
| `rolling_median_7`, `rolling_median_28` | Robust shifted demand level |
| `rolling_max_7`, `rolling_min_7` | Recent shifted demand range |
| `exponentially_weighted_mean_7`, `ewm_mean_28` | Recency-weighted demand level |

## Promotions

| Feature | Description |
|---|---|
| `promo_active` | Current promotion flag |
| `promo_streak` | Consecutive active promotion days |
| `days_since_last_promo` | Recency of promotion activity |
| `days_until_next_promo` | Known future promotion distance |
| `promo2_active` | Store-level Promo2 participation |
| `promo2_weeks_active` | Promo2 maturity in weeks |

## Competition and Store Metadata

| Feature | Description |
|---|---|
| `competition_open_months` | Months since nearby competitor opened |
| `log_competition_distance` | Log-transformed competitor distance |
| `store_type` | Encoded store type |
| `assortment` | Encoded assortment type |
| `Store` | Store identifier used as a stable categorical signal |

## Holidays

| Feature | Description |
|---|---|
| `is_state_holiday` | Public holiday indicator |
| `is_school_holiday` | School holiday indicator |
| `holiday_type` | Encoded holiday category |
| `days_since_last_holiday` | Holiday recency |
| `days_to_next_holiday` | Known future holiday distance |

## Leakage Control

`Customers` is excluded because customer count is observed during the trading day. Including it would turn the task into same-day nowcasting instead of pre-day demand forecasting.
