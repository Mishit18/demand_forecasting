from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_DIR / "src"))

from run_pipeline import add_features, mape  # noqa: E402


def tiny_clean_frame() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=80, freq="D")
    return pd.DataFrame(
        {
            "Store": [1] * len(dates),
            "Date": dates,
            "Sales": np.arange(100, 100 + len(dates), dtype=float),
            "DayOfWeek": dates.dayofweek + 1,
            "Open": 1,
            "Promo": [1 if i % 10 in [0, 1] else 0 for i in range(len(dates))],
            "StateHoliday": [0] * len(dates),
            "SchoolHoliday": [1 if i % 15 == 0 else 0 for i in range(len(dates))],
            "StoreType": ["a"] * len(dates),
            "Assortment": ["a"] * len(dates),
            "CompetitionDistance": [500.0] * len(dates),
            "CompetitionOpenSinceMonth": [1] * len(dates),
            "CompetitionOpenSinceYear": [2023] * len(dates),
            "Promo2": [0] * len(dates),
            "Promo2SinceWeek": [0] * len(dates),
            "Promo2SinceYear": [0] * len(dates),
            "PromoInterval": [""] * len(dates),
        }
    )


def test_lag_features_are_causal():
    featured, feature_cols = add_features(tiny_clean_frame())
    row = featured.iloc[0]
    original = tiny_clean_frame().set_index("Date")

    assert len(feature_cols) >= 40
    assert row["sales_lag_1"] == original.loc[row["Date"] - pd.Timedelta(days=1), "Sales"]
    assert row["sales_lag_7"] == original.loc[row["Date"] - pd.Timedelta(days=7), "Sales"]
    assert "Customers" not in feature_cols


def test_mape_ignores_zero_actuals():
    assert round(mape(np.array([0, 100]), np.array([999, 90])), 2) == 10.0
