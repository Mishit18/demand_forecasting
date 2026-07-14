# Data Card

## Dataset

Rossmann Store Sales dataset containing daily sales observations, promotion flags, holiday flags, and store metadata.

## Files

- `data/train.csv`: historical daily store sales
- `data/store.csv`: store-level metadata including store type, assortment, competition distance, and Promo2 fields

## Scope

- Rows after filtering closed and zero-sales days: 844,338
- Stores: 1,115
- Date range: 2013-01-01 to 2015-07-31
- Modeling window after lag warmup: 781,898 rows

## Cleaning Rules

- Kept only `Open == 1` and `Sales > 0`
- Filled missing `CompetitionDistance` with `999999` to represent no known nearby competitor
- Filled missing competition and Promo2 start fields with `0`
- Normalized `StateHoliday` into numeric holiday types
- Removed early lag-warmup rows per store after causal feature generation

## Modeling Notes

The `Customers` column is deliberately excluded from the feature matrix. It is observed during the day and would not be known at forecast-generation time, so using it would overstate real forecasting performance.

## Limitations

- The dataset is store-level, not SKU-level. Reorder-point examples therefore represent store-demand units rather than product-specific replenishment policies.
- Working-capital conversion uses explicit assumptions: Rs. 75 average unit value and 20% annual holding-cost rate.
