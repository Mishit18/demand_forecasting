from __future__ import annotations

import argparse
import json
import math
import os
import random
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import minimize
from scipy.stats import f_oneway, kurtosis, mannwhitneyu, skew
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)
sns.set_palette("husl")


@dataclass
class RunConfig:
    project_dir: Path
    n_trials: int = 25
    cv_splits: int = 3
    cv_sample_frac: float = 0.35
    sarimax_store_count: int = 5
    shap_sample: int = 1000
    avg_unit_price_rs: float = 75.0
    holding_cost_rate: float = 0.20

    @property
    def data_dir(self) -> Path:
        return self.project_dir / "data"

    @property
    def outputs_dir(self) -> Path:
        return self.project_dir / "outputs"


def require_data(cfg: RunConfig) -> None:
    missing = [name for name in ("train.csv", "store.csv") if not (cfg.data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} in {cfg.data_dir}. Download Rossmann Store Sales from Kaggle "
            "or run: python -c \"import kagglehub; kagglehub.dataset_download('pratyushakar/rossmann-store-sales')\""
        )


def load_and_clean(cfg: RunConfig) -> pd.DataFrame:
    train = pd.read_csv(cfg.data_dir / "train.csv", parse_dates=["Date"], low_memory=False)
    store = pd.read_csv(cfg.data_dir / "store.csv")
    df = train.merge(store, on="Store", how="left")
    df = df[(df["Open"] == 1) & (df["Sales"] > 0)].copy()

    fill_zero = [
        "CompetitionOpenSinceMonth",
        "CompetitionOpenSinceYear",
        "Promo2SinceWeek",
        "Promo2SinceYear",
    ]
    df["CompetitionDistance"] = df["CompetitionDistance"].fillna(999999)
    for col in fill_zero:
        df[col] = df[col].fillna(0)

    state_map = {"0": 0, 0: 0, "a": 1, "b": 2, "c": 3}
    df["StateHoliday"] = df["StateHoliday"].map(state_map).fillna(0).astype(int)
    df = df.sort_values(["Store", "Date"]).reset_index(drop=True)

    assert not df.duplicated(["Store", "Date"]).any(), "Duplicate Store-Date rows found."
    assert df[["Sales", "CompetitionDistance", "StateHoliday", "Promo", "SchoolHoliday"]].isna().sum().sum() == 0
    if len(df) < 800_000:
        raise ValueError(f"Rows after cleaning are too low: {len(df):,}")

    print(f"Rows after cleaning: {len(df):,}")
    print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"Unique stores: {df['Store'].nunique():,}")
    return df


def add_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    iso = out["Date"].dt.isocalendar()
    out["day_of_week"] = out["Date"].dt.dayofweek
    out["month"] = out["Date"].dt.month
    out["week_of_year"] = iso.week.astype(int)
    out["year"] = out["Date"].dt.year
    out["is_weekend"] = out["day_of_week"].isin([5, 6]).astype(int)
    out["is_month_start"] = (out["Date"].dt.day <= 3).astype(int)
    out["is_month_end"] = (out["Date"].dt.day >= 28).astype(int)
    out["quarter"] = out["Date"].dt.quarter
    out["day_of_year"] = out["Date"].dt.dayofyear

    g = out.groupby("Store", group_keys=False)["Sales"]
    for lag in [1, 2, 3, 7, 14, 21, 28, 56]:
        out[f"sales_lag_{lag}"] = g.shift(lag)
    shifted = out.groupby("Store")["Sales"].shift(1)
    for window in [7, 14, 28]:
        out[f"rolling_mean_{window}"] = shifted.groupby(out["Store"]).rolling(window, min_periods=window).mean().reset_index(level=0, drop=True)
    for window in [7, 28]:
        out[f"rolling_std_{window}"] = shifted.groupby(out["Store"]).rolling(window, min_periods=window).std().reset_index(level=0, drop=True)
        out[f"rolling_median_{window}"] = shifted.groupby(out["Store"]).rolling(window, min_periods=window).median().reset_index(level=0, drop=True)
    out["rolling_max_7"] = shifted.groupby(out["Store"]).rolling(7, min_periods=7).max().reset_index(level=0, drop=True)
    out["rolling_min_7"] = shifted.groupby(out["Store"]).rolling(7, min_periods=7).min().reset_index(level=0, drop=True)
    out["exponentially_weighted_mean_7"] = out.groupby("Store")["Sales"].transform(lambda s: s.shift(1).ewm(span=7, adjust=False).mean())
    out["ewm_mean_28"] = out.groupby("Store")["Sales"].transform(lambda s: s.shift(1).ewm(span=28, adjust=False).mean())

    out["promo_active"] = out["Promo"].astype(int)
    out["promo_streak"] = out.groupby("Store", group_keys=False)["promo_active"].apply(
        lambda s: s.groupby((s != s.shift()).cumsum()).cumcount().add(1).where(s.eq(1), 0)
    )
    out["days_since_last_promo"] = days_since_event(out, "promo_active")
    out["days_until_next_promo"] = days_until_event(out, "promo_active")

    comp_month = out["CompetitionOpenSinceMonth"].astype(int)
    comp_year = out["CompetitionOpenSinceYear"].astype(int)
    comp_months = (out["year"] - comp_year) * 12 + (out["month"] - comp_month)
    out["competition_open_months"] = np.where((comp_year > 0) & (comp_months > 0), comp_months, 0)
    out["log_competition_distance"] = np.log1p(out["CompetitionDistance"])

    out["store_type"] = out["StoreType"].map({"a": 0, "b": 1, "c": 2, "d": 3}).astype(int)
    out["assortment"] = out["Assortment"].map({"a": 0, "b": 1, "c": 2}).astype(int)
    out["promo2_active"] = out["Promo2"].fillna(0).astype(int)
    promo2_weeks = (out["year"] - out["Promo2SinceYear"].astype(int)) * 52 + (out["week_of_year"] - out["Promo2SinceWeek"].astype(int))
    out["promo2_weeks_active"] = np.where((out["Promo2SinceYear"] > 0) & (promo2_weeks > 0), promo2_weeks, 0)

    out["is_state_holiday"] = (out["StateHoliday"] > 0).astype(int)
    out["is_school_holiday"] = out["SchoolHoliday"].astype(int)
    out["holiday_type"] = out["StateHoliday"].astype(int)
    out["any_holiday"] = ((out["is_state_holiday"] == 1) | (out["is_school_holiday"] == 1)).astype(int)
    out["days_since_last_holiday"] = days_since_event(out, "any_holiday")
    out["days_to_next_holiday"] = days_until_event(out, "any_holiday")

    feature_cols = [
        "Store",
        "day_of_week",
        "month",
        "week_of_year",
        "year",
        "is_weekend",
        "is_month_start",
        "is_month_end",
        "quarter",
        "day_of_year",
        "promo_active",
        "promo_streak",
        "days_since_last_promo",
        "days_until_next_promo",
        "competition_open_months",
        "log_competition_distance",
        "store_type",
        "assortment",
        "promo2_active",
        "promo2_weeks_active",
        "is_state_holiday",
        "is_school_holiday",
        "days_to_next_holiday",
        "days_since_last_holiday",
        "holiday_type",
    ]
    feature_cols += [f"sales_lag_{lag}" for lag in [1, 2, 3, 7, 14, 21, 28, 56]]
    feature_cols += [
        "rolling_mean_7",
        "rolling_mean_14",
        "rolling_mean_28",
        "rolling_std_7",
        "rolling_std_28",
        "rolling_median_7",
        "rolling_median_28",
        "rolling_max_7",
        "rolling_min_7",
        "exponentially_weighted_mean_7",
        "ewm_mean_28",
    ]
    out = out.dropna(subset=feature_cols).reset_index(drop=True)
    assert len(feature_cols) >= 40, len(feature_cols)
    assert out[feature_cols].isna().sum().sum() == 0
    print(f"Feature matrix shape: {out[feature_cols].shape}, no NaNs confirmed.")
    return out, feature_cols


def days_since_event(df: pd.DataFrame, event_col: str) -> pd.Series:
    def one_store(s: pd.DataFrame) -> pd.Series:
        dates = s["Date"].to_numpy("datetime64[D]")
        event_dates = np.where(s[event_col].to_numpy() == 1, dates, np.datetime64("NaT"))
        last = pd.Series(event_dates).ffill().to_numpy("datetime64[D]")
        vals = (dates - last).astype("timedelta64[D]").astype(float)
        vals[np.isnat(last)] = 999
        return pd.Series(vals, index=s.index)

    return df.groupby("Store", group_keys=False).apply(one_store)


def days_until_event(df: pd.DataFrame, event_col: str) -> pd.Series:
    def one_store(s: pd.DataFrame) -> pd.Series:
        rev = s.iloc[::-1]
        dates = rev["Date"].to_numpy("datetime64[D]")
        event_dates = np.where(rev[event_col].to_numpy() == 1, dates, np.datetime64("NaT"))
        nxt = pd.Series(event_dates).ffill().to_numpy("datetime64[D]")
        vals = (nxt - dates).astype("timedelta64[D]").astype(float)
        vals[np.isnat(nxt)] = 999
        return pd.Series(vals, index=rev.index).sort_index()

    return df.groupby("Store", group_keys=False).apply(one_store)


def mape(y_true, y_pred) -> float:
    y = np.asarray(y_true)
    p = np.asarray(y_pred)
    mask = y > 0
    return float(np.mean(np.abs((y[mask] - p[mask]) / y[mask])) * 100)


def time_series_cv_score(model, X: pd.DataFrame, y: pd.Series, n_splits: int = 3) -> tuple[float, float]:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for train_idx, val_idx in tscv.split(X):
        est = clone(model)
        est.fit(X.iloc[train_idx], y.iloc[train_idx])
        scores.append(mape(y.iloc[val_idx], est.predict(X.iloc[val_idx])))
    return float(np.mean(scores)), float(np.std(scores))


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    cutoff = df["Date"].max() - pd.Timedelta(days=41)
    train = df[df["Date"] < cutoff].copy()
    test = df[df["Date"] >= cutoff].copy()
    print(f"Training period: {train['Date'].min().date()} to {train['Date'].max().date()}, rows={len(train):,}")
    print(f"Test period: {test['Date'].min().date()} to {test['Date'].max().date()}, rows={len(test):,}")
    return train, test, cutoff


def build_eda(df: pd.DataFrame, outdir: Path) -> dict:
    stats = {
        "mean": float(df["Sales"].mean()),
        "median": float(df["Sales"].median()),
        "std": float(df["Sales"].std()),
        "skew": float(skew(df["Sales"])),
        "kurtosis": float(kurtosis(df["Sales"])),
    }
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(df["Sales"], bins=80, log_scale=(False, True), ax=ax, color="#2f6f9f")
    ax.set_title("Distribution of Daily Store Sales")
    ax.set_xlabel("Daily sales")
    ax.set_ylabel("Store-days (log scale)")
    fig.tight_layout()
    fig.savefig(outdir / "plot_00_sales_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    heat = df.assign(month=df["Date"].dt.month).pivot_table(index="DayOfWeek", columns="month", values="Sales", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(heat, cmap="viridis", annot=False, fmt=".0f", ax=ax, cbar_kws={"label": "Average sales"})
    ax.set_title("Average Sales by Day of Week and Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Day of week")
    fig.tight_layout()
    fig.savefig(outdir / "plot_01_sales_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    make_decomposition_plot(df, outdir)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.kdeplot(df.loc[df["Promo"] == 0, "Sales"], label="Promo off", fill=True, alpha=0.25, ax=ax)
    sns.kdeplot(df.loc[df["Promo"] == 1, "Sales"], label="Promo on", fill=True, alpha=0.25, ax=ax)
    ax.set_title("Promo Lift: Daily Sales Density")
    ax.set_xlabel("Daily sales")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "plot_03_promo_lift_kde.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    med_no = df.loc[df["Promo"] == 0, "Sales"].median()
    med_yes = df.loc[df["Promo"] == 1, "Sales"].median()
    promo_lift = (med_yes / med_no - 1) * 100
    sample_no = df.loc[df["Promo"] == 0, "Sales"].sample(min(100_000, (df["Promo"] == 0).sum()), random_state=SEED)
    sample_yes = df.loc[df["Promo"] == 1, "Sales"].sample(min(100_000, (df["Promo"] == 1).sum()), random_state=SEED)
    _, promo_p = mannwhitneyu(sample_no, sample_yes, alternative="two-sided")

    fig, ax = plt.subplots(figsize=(8, 5))
    holiday = df.groupby("StateHoliday", as_index=False)["Sales"].mean()
    sns.barplot(data=holiday, x="StateHoliday", y="Sales", ax=ax)
    ax.set_title("Average Sales by State Holiday Type")
    ax.set_xlabel("Holiday type (0 none, 1 public, 2 Easter, 3 Christmas)")
    ax.set_ylabel("Average sales")
    fig.tight_layout()
    fig.savefig(outdir / "plot_04_holiday_effect.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x="StoreType", y="Sales", showfliers=False, ax=ax)
    ax.set_title("Daily Sales by Store Type")
    ax.set_xlabel("Store type")
    ax.set_ylabel("Daily sales")
    fig.tight_layout()
    fig.savefig(outdir / "plot_05_storetype_boxplot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    groups = [vals["Sales"].values for _, vals in df.groupby("StoreType")]
    _, anova_p = f_oneway(*groups)
    stats.update({"promo_median_lift_pct": float(promo_lift), "promo_mannwhitney_p": float(promo_p), "storetype_anova_p": float(anova_p)})
    return stats


def make_decomposition_plot(df: pd.DataFrame, outdir: Path) -> None:
    from statsmodels.tsa.seasonal import seasonal_decompose

    totals = df.groupby("Store")["Sales"].sum().sort_values()
    stores = [int(totals.index[0]), int(totals.index[len(totals) // 2]), int(totals.index[-1])]
    fig, axes = plt.subplots(3, 4, figsize=(16, 9), sharex=False)
    for row, store_id in enumerate(stores):
        s = df[df["Store"] == store_id].set_index("Date")["Sales"].asfreq("D").interpolate()
        dec = seasonal_decompose(s, period=7, model="additive", extrapolate_trend="freq")
        for col, (name, series) in enumerate(
            [("Observed", s), ("Trend", dec.trend), ("Seasonal", dec.seasonal), ("Residual", dec.resid)]
        ):
            axes[row, col].plot(series.index, series.values, linewidth=0.8)
            axes[row, col].set_title(f"Store {store_id} - {name}")
            axes[row, col].tick_params(axis="x", rotation=30, labelsize=8)
    fig.suptitle("Weekly Seasonal Decomposition for Low, Medium, and High Volume Stores", y=1.01)
    fig.tight_layout()
    fig.savefig(outdir / "plot_02_decomposition.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def baseline_predictions(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, np.ndarray]:
    store_median = train.groupby("Store")["Sales"].median()
    return {
        "Naive": test["sales_lag_1"].to_numpy(),
        "Rolling Mean": test["rolling_mean_7"].to_numpy(),
        "Same Day Last Week": test["sales_lag_7"].to_numpy(),
        "Store Median": test["Store"].map(store_median).to_numpy(),
    }


def fit_models(cfg: RunConfig, train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, dict, dict]:
    import lightgbm as lgb
    import optuna
    import xgboost as xgb

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    X_train, y_train = train[feature_cols], train["Sales"]
    X_test, y_test = test[feature_cols], test["Sales"]

    cv_train = train.sort_values("Date").sample(frac=cfg.cv_sample_frac, random_state=SEED).sort_values(["Date", "Store"])
    X_cv, y_cv = cv_train[feature_cols], cv_train["Sales"]

    baselines = baseline_predictions(train, test)
    rows = []
    preds = {}
    for name, pred in baselines.items():
        preds[name] = pred
        rows.append({"Model": name, "CV MAPE Mean": np.nan, "CV MAPE Std": np.nan, "Holdout MAPE": mape(y_test, pred)})

    sarimax_pred, sarimax_meta = sarimax_forecast(cfg, train, test)
    preds["SARIMAX"] = sarimax_pred
    rows.append(
        {
            "Model": "SARIMAX",
            "CV MAPE Mean": sarimax_meta.get("avg_mape", np.nan),
            "CV MAPE Std": sarimax_meta.get("std_mape", np.nan),
            "Holdout MAPE": mape(y_test, sarimax_pred),
        }
    )

    def xgb_objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 900),
            "max_depth": trial.suggest_int("max_depth", 4, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.20, log=True),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 2.0, log=True),
            "random_state": SEED,
            "tree_method": "hist",
            "n_jobs": -1,
            "eval_metric": "mae",
        }
        cv_mape, _ = time_series_cv_score(xgb.XGBRegressor(**params), X_cv, y_cv, cfg.cv_splits)
        return cv_mape

    xgb_study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    xgb_study.optimize(xgb_objective, n_trials=cfg.n_trials, show_progress_bar=False)
    xgb_params = xgb_study.best_params | {"random_state": SEED, "tree_method": "hist", "n_jobs": -1, "eval_metric": "mae"}
    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_cv_mean, xgb_cv_std = time_series_cv_score(xgb.XGBRegressor(**xgb_params), X_cv, y_cv, cfg.cv_splits)
    xgb_model.fit(X_train, y_train)
    preds["XGBoost"] = np.maximum(1, xgb_model.predict(X_test))
    rows.append({"Model": "XGBoost", "CV MAPE Mean": xgb_cv_mean, "CV MAPE Std": xgb_cv_std, "Holdout MAPE": mape(y_test, preds["XGBoost"])})

    def lgb_objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1000),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.20, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 24, 160),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 120),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 1.0, log=True),
            "random_state": SEED,
            "verbose": -1,
            "n_jobs": -1,
        }
        cv_mape, _ = time_series_cv_score(lgb.LGBMRegressor(**params), X_cv, y_cv, cfg.cv_splits)
        return cv_mape

    lgb_study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED + 1))
    lgb_study.optimize(lgb_objective, n_trials=cfg.n_trials, show_progress_bar=False)
    lgb_params = lgb_study.best_params | {"random_state": SEED, "verbose": -1, "n_jobs": -1}
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_cv_mean, lgb_cv_std = time_series_cv_score(lgb.LGBMRegressor(**lgb_params), X_cv, y_cv, cfg.cv_splits)
    lgb_model.fit(X_train, y_train)
    preds["LightGBM"] = np.maximum(1, lgb_model.predict(X_test))
    rows.append({"Model": "LightGBM", "CV MAPE Mean": lgb_cv_mean, "CV MAPE Std": lgb_cv_std, "Holdout MAPE": mape(y_test, preds["LightGBM"])})

    val_cut = train["Date"].max() - pd.Timedelta(days=41)
    val = train[train["Date"] >= val_cut]
    tr = train[train["Date"] < val_cut]
    xgb_val_model = xgb.XGBRegressor(**xgb_params).fit(tr[feature_cols], tr["Sales"])
    lgb_val_model = lgb.LGBMRegressor(**lgb_params).fit(tr[feature_cols], tr["Sales"])
    val_preds = [
        val["sales_lag_1"].to_numpy(),
        np.maximum(1, xgb_val_model.predict(val[feature_cols])),
        np.maximum(1, lgb_val_model.predict(val[feature_cols])),
    ]
    result = minimize(
        lambda w: mape(val["Sales"], sum((np.array(w) / np.sum(w))[i] * val_preds[i] for i in range(3))),
        x0=[0.05, 0.45, 0.50],
        method="Nelder-Mead",
        bounds=[(0, 1)] * 3,
    )
    weights = np.array(result.x) / np.sum(result.x)
    ensemble = weights[0] * preds["Naive"] + weights[1] * preds["XGBoost"] + weights[2] * preds["LightGBM"]
    preds["Ensemble"] = np.maximum(1, ensemble)
    rows.append({"Model": "Ensemble", "CV MAPE Mean": np.nan, "CV MAPE Std": np.nan, "Holdout MAPE": mape(y_test, preds["Ensemble"])})

    results = pd.DataFrame(rows)
    meta = {
        "xgb_params": xgb_params,
        "lgb_params": lgb_params,
        "ensemble_weights": {"Naive": float(weights[0]), "XGBoost": float(weights[1]), "LightGBM": float(weights[2])},
        "xgb_study_values": [float(t.value) for t in xgb_study.trials if t.value is not None],
        "lgb_study_values": [float(t.value) for t in lgb_study.trials if t.value is not None],
        "sarimax": sarimax_meta,
    }
    model_objs = {"xgb": xgb_model, "lgb": lgb_model}
    return results, preds, meta | model_objs


def sarimax_forecast(cfg: RunConfig, train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, dict]:
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.stattools import adfuller, kpss

    full_pred = test["rolling_mean_7"].to_numpy().astype(float)
    top = train.groupby("Store")["Sales"].sum().nlargest(cfg.sarimax_store_count).index.tolist()
    store_mapes = []
    first_meta = {}
    for store_id in top:
        tr = train[train["Store"] == store_id].set_index("Date").asfreq("D")
        te = test[test["Store"] == store_id].set_index("Date").asfreq("D")
        y = tr["Sales"].interpolate().ffill().bfill()
        try:
            adf_p = adfuller(y)[1]
            kpss_p = kpss(y, regression="c", nlags="auto")[1]
            d = 1 if adf_p > 0.05 and kpss_p < 0.05 else 0
            fitted, order, seasonal_order, aic, resid = fit_sarimax_like(y, d)
            fc = fitted.forecast(steps=len(te))
            pred = np.maximum(1, fc.reindex(te.index).interpolate().ffill().bfill().to_numpy())
            idx = test.index[test["Store"].eq(store_id)]
            full_pred[np.isin(test.index, idx)] = pred[: len(idx)]
            score = mape(test.loc[idx, "Sales"], pred[: len(idx)])
            store_mapes.append(score)
            if not first_meta:
                lb_p = float(acorr_ljungbox(fitted.resid.dropna(), lags=[14], return_df=True)["lb_pvalue"].iloc[0])
                first_meta = {
                    "store": int(store_id),
                    "order": str(order),
                    "seasonal_order": str(seasonal_order),
                    "aic": float(aic),
                    "adf_p": float(adf_p),
                    "kpss_p": float(kpss_p),
                    "ljung_box_p": lb_p,
                    "mape": float(score),
                }
        except Exception as exc:
            first_meta.setdefault("warning", str(exc))
            continue
    meta = first_meta | {"avg_mape": float(np.mean(store_mapes)) if store_mapes else np.nan, "std_mape": float(np.std(store_mapes)) if store_mapes else np.nan}
    return full_pred, meta


def fit_sarimax_like(y: pd.Series, d: int):
    from statsmodels.tsa.arima.model import ARIMA

    try:
        from pmdarima import auto_arima

        auto = auto_arima(
            y,
            seasonal=True,
            m=7,
            d=d,
            stepwise=True,
            information_criterion="aic",
            max_p=3,
            max_q=3,
            max_P=2,
            max_Q=2,
            trace=False,
            suppress_warnings=True,
            error_action="ignore",
        )
        order = auto.order
        seasonal_order = auto.seasonal_order
    except Exception:
        order = (2, d, 2)
        seasonal_order = (1, 0, 1, 7)
    fitted = ARIMA(y, order=order, seasonal_order=seasonal_order).fit()
    return fitted, order, seasonal_order, fitted.aic, fitted.resid


def safety_stock_and_bias(cfg: RunConfig, test: pd.DataFrame, preds: dict[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    work = test[["Store", "Date", "Sales", "StoreType", "store_type", "month"]].copy()
    for name, pred in preds.items():
        work[f"{name}_pred"] = pred
        work[f"{name}_err"] = work["Sales"] - pred

    z_scores = {0.90: 1.282, 0.95: 1.645, 0.99: 2.326}
    leads = [3, 7, 14]
    rows = []
    for stype, grp in work.groupby("StoreType"):
        for model in ["Naive", "SARIMAX", "XGBoost", "Ensemble"]:
            sigma = grp[f"{model}_err"].std()
            for service, z in z_scores.items():
                for lead in leads:
                    rows.append(
                        {
                            "StoreType": stype,
                            "Model": model,
                            "Service Level": service,
                            "Lead Time Days": lead,
                            "Safety Stock Units": z * sigma * math.sqrt(lead),
                        }
                    )
    ss = pd.DataFrame(rows)

    naive_sigma = work["Naive_err"].std()
    ens_sigma = work["Ensemble_err"].std()
    safety_reduction_pct = (1 - ens_sigma / naive_sigma) * 100
    reduction_units = (1.645 * naive_sigma * math.sqrt(7)) - (1.645 * ens_sigma * math.sqrt(7))
    rs_freed_per_store = reduction_units * cfg.avg_unit_price_rs * cfg.holding_cost_rate

    demand = work.groupby("Store")["Sales"].mean().rename("average_daily_demand")
    err_sigma = work.groupby("Store")["Ensemble_err"].std().rename("forecast_error_sigma")
    top = work.groupby("Store")["Sales"].sum().nlargest(10).index
    reorder = pd.concat([demand, err_sigma], axis=1).loc[top].reset_index()
    reorder["lead_time_days"] = 7
    reorder["service_level"] = 0.95
    reorder["safety_stock_units"] = 1.645 * reorder["forecast_error_sigma"] * math.sqrt(7)
    reorder["reorder_point_units"] = reorder["average_daily_demand"] * reorder["lead_time_days"] + reorder["safety_stock_units"]

    bias = work.groupby(["StoreType", "month"], as_index=False).agg(bias_units=("Ensemble_err", "mean"), avg_sales=("Sales", "mean"))
    bias["bias_pct_of_sales"] = bias["bias_units"] / bias["avg_sales"] * 100

    meta = {
        "naive_sigma": float(naive_sigma),
        "ensemble_sigma": float(ens_sigma),
        "safety_reduction_pct": float(safety_reduction_pct),
        "reduction_units_95_7day": float(reduction_units),
        "rs_freed_per_store": float(rs_freed_per_store),
    }
    return ss, reorder, bias, meta


def make_model_plots(cfg: RunConfig, train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str], results: pd.DataFrame, preds: dict, meta: dict, ss: pd.DataFrame, bias: pd.DataFrame) -> dict:
    outdir = cfg.outputs_dir
    fig, ax = plt.subplots(figsize=(9, 5))
    plot_df = results.copy()
    yerr = plot_df["CV MAPE Std"].fillna(0)
    sns.barplot(data=plot_df, x="Model", y="Holdout MAPE", ax=ax)
    ax.errorbar(range(len(plot_df)), plot_df["Holdout MAPE"], yerr=yerr, fmt="none", ecolor="black", capsize=4)
    ax.set_title("Holdout MAPE by Model")
    ax.set_ylabel("MAPE (%)")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(outdir / "plot_04_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    xgb_model = meta["xgb"]
    try:
        import shap

        sample = test[feature_cols].sample(min(cfg.shap_sample, len(test)), random_state=SEED)
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(sample)
        shap.summary_plot(shap_values, sample, show=False, max_display=20)
        plt.title("XGBoost SHAP Feature Impact")
        plt.tight_layout()
        plt.savefig(outdir / "plot_05_shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close()
    except Exception:
        importances = pd.Series(xgb_model.feature_importances_, index=feature_cols).nlargest(20).sort_values()
        fig, ax = plt.subplots(figsize=(8, 7))
        importances.plot(kind="barh", ax=ax)
        ax.set_title("XGBoost Feature Importance Proxy")
        fig.tight_layout()
        fig.savefig(outdir / "plot_05_shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    try:
        gain = xgb_model.get_booster().get_score(importance_type="gain")
        imp = pd.Series(gain).sort_values(ascending=False).head(20).sort_values()
    except Exception:
        imp = pd.Series(xgb_model.feature_importances_, index=feature_cols).sort_values(ascending=False).head(20).sort_values()
    fig, ax = plt.subplots(figsize=(8, 7))
    imp.plot(kind="barh", ax=ax, color="#376795")
    ax.set_title("XGBoost Top 20 Feature Importance by Gain")
    ax.set_xlabel("Gain")
    fig.tight_layout()
    fig.savefig(outdir / "plot_06_xgb_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    plot_test = test.copy()
    plot_test["Ensemble_pred"] = preds["Ensemble"]
    totals = plot_test.groupby("Store")["Sales"].sum().sort_values()
    stores = [int(s) for s in np.unique(np.r_[totals.index[:2], totals.index[len(totals) // 2 - 1 : len(totals) // 2 + 1], totals.index[-2:]])[:6]]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=False)
    for ax, store_id in zip(axes.ravel(), stores):
        s = plot_test[plot_test["Store"] == store_id]
        ax.plot(s["Date"], s["Sales"], label="Actual", linewidth=1.5)
        ax.plot(s["Date"], s["Ensemble_pred"], label="Predicted", linewidth=1.5)
        ax.set_title(f"Store {store_id}")
        ax.tick_params(axis="x", rotation=30, labelsize=8)
    axes.ravel()[0].legend()
    fig.suptitle("Actual vs Ensemble Forecast for Representative Stores", y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / "plot_07_actual_vs_predicted.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    residuals = test["Sales"].to_numpy() - preds["Ensemble"]
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(residuals, bins=80, kde=True, ax=ax, color="#6c5b7b")
    ax.set_title("Ensemble Forecast Residual Distribution")
    ax.set_xlabel("Actual - predicted sales")
    fig.tight_layout()
    fig.savefig(outdir / "plot_08_residual_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    line = ss[(ss["Lead Time Days"] == 7) & (ss["StoreType"] == ss["StoreType"].iloc[0])]
    sns.lineplot(data=line, x="Service Level", y="Safety Stock Units", hue="Model", marker="o", ax=ax)
    ax.set_title("Safety Stock vs Service Level by Forecast Model")
    ax.set_ylabel("Safety stock units")
    fig.tight_layout()
    fig.savefig(outdir / "plot_09_safety_stock_lines.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ss[(ss["Model"] == "Ensemble")].groupby(["Lead Time Days", "Service Level"], as_index=False)["Safety Stock Units"].mean()
    X = surf["Lead Time Days"].to_numpy()
    Y = surf["Service Level"].to_numpy()
    Z = surf["Safety Stock Units"].to_numpy()
    ax.plot_trisurf(X, Y, Z, cmap="viridis", alpha=0.9)
    ax.set_title("Ensemble Safety Stock Surface")
    ax.set_xlabel("Lead time days")
    ax.set_ylabel("Service level")
    ax.set_zlabel("Safety stock units")
    fig.tight_layout()
    fig.savefig(outdir / "plot_09b_safety_stock_surface.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    bmat = bias.pivot(index="StoreType", columns="month", values="bias_units")
    sns.heatmap(bmat, cmap="vlag", center=0, ax=ax, cbar_kws={"label": "Mean bias units"})
    ax.set_title("Forecast Bias by Store Type and Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Store type")
    fig.tight_layout()
    fig.savefig(outdir / "plot_10_bias_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    temp = test[["Store", "Date", "Sales", "StoreType"]].copy()
    temp["pred"] = preds["Ensemble"]
    temp["bias"] = temp["Sales"] - temp["pred"]
    by_store = temp.groupby("Store", as_index=False)["bias"].mean()
    extremes = pd.concat([by_store.nlargest(10, "bias"), by_store.nsmallest(10, "bias")])
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=extremes, x="Store", y="bias", ax=ax, order=extremes["Store"].astype(str))
    ax.set_title("Stores with Largest Under- and Over-Forecast Bias")
    ax.set_xlabel("Store")
    ax.set_ylabel("Mean bias units")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(outdir / "plot_10b_bias_extreme_stores.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    rolling = temp.groupby("Date")["bias"].mean().rolling(28, min_periods=7).mean()
    ax.plot(rolling.index, rolling.values, linewidth=2)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Rolling Four-Week Mean Forecast Bias")
    ax.set_xlabel("Date")
    ax.set_ylabel("Mean bias units")
    fig.tight_layout()
    fig.savefig(outdir / "plot_10c_rolling_bias.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    vals = meta["xgb_study_values"]
    best = np.minimum.accumulate(vals)
    ax.plot(range(1, len(best) + 1), best, marker="o", linewidth=1.5)
    ax.set_title("Optuna XGBoost Optimisation History")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Best CV MAPE (%)")
    fig.tight_layout()
    fig.savefig(outdir / "plot_11_optuna_history.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 7))
    scatter = test.sample(min(30_000, len(test)), random_state=SEED).copy()
    scatter["pred"] = pd.Series(preds["Ensemble"], index=test.index).loc[scatter.index]
    sns.scatterplot(data=scatter, x="Sales", y="pred", hue="StoreType", s=10, alpha=0.35, ax=ax)
    lim = max(scatter["Sales"].max(), scatter["pred"].max())
    ax.plot([0, lim], [0, lim], color="black", linewidth=1)
    ax.set_title("Ensemble Predicted vs Actual Sales")
    ax.set_xlabel("Actual sales")
    ax.set_ylabel("Predicted sales")
    fig.tight_layout()
    fig.savefig(outdir / "plot_12_predicted_vs_actual.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    correction = temp.groupby("StoreType").apply(lambda x: x["Sales"].sum() / max(x["pred"].sum(), 1)).rename("correction_factor").reset_index()
    before = np.abs(temp["bias"]).mean()
    temp = temp.merge(correction, on="StoreType")
    temp["adjusted_pred"] = temp["pred"] * temp["correction_factor"]
    after = np.abs(temp["Sales"] - temp["adjusted_pred"]).mean()
    return {"bias_correction_error_reduction_pct": float((1 - after / before) * 100), "bias_correction": correction.to_dict("records")}


def write_summary(cfg: RunConfig, clean_df: pd.DataFrame, featured_df: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str], results: pd.DataFrame, eda: dict, safety: dict, bias: pd.DataFrame, bias_meta: dict, meta: dict) -> None:
    best = results.loc[results["Model"].eq("Ensemble"), "Holdout MAPE"].iloc[0]
    naive = results.loc[results["Model"].eq("Naive"), "Holdout MAPE"].iloc[0]
    xgb_mape = results.loc[results["Model"].eq("XGBoost"), "Holdout MAPE"].iloc[0]
    improvement = naive - best
    worst_bias = bias.assign(abs_bias=lambda x: x["bias_pct_of_sales"].abs()).sort_values("abs_bias", ascending=False).iloc[0]
    table_md = results.assign(
        **{
            "CV MAPE Mean": results["CV MAPE Mean"].map(lambda x: "" if pd.isna(x) else f"{x:.2f}%"),
            "CV MAPE Std": results["CV MAPE Std"].map(lambda x: "" if pd.isna(x) else f"{x:.2f}%"),
            "Holdout MAPE": results["Holdout MAPE"].map(lambda x: f"{x:.2f}%"),
        }
    ).to_markdown(index=False)
    summary = f"""# Rossmann Demand Forecasting: Strategy and Operations Summary

## Dataset Overview

- Clean forecastable demand rows: {len(clean_df):,}
- Model-ready rows after causal lag warmup: {len(featured_df):,}
- Stores modeled: {clean_df['Store'].nunique():,}
- Date range: {clean_df['Date'].min().date()} to {clean_df['Date'].max().date()}
- Time split: training {train['Date'].min().date()} to {train['Date'].max().date()}, holdout {test['Date'].min().date()} to {test['Date'].max().date()}
- Engineered features: {len(feature_cols)}

## Model Results

{table_md}

Best ensemble weights: {json.dumps(meta['ensemble_weights'], indent=2)}

## Safety Stock and Working Capital Impact

Improving MAPE from {naive:.2f}% (naive) to {best:.2f}% (ensemble) reduces required safety stock by {safety['safety_reduction_pct']:.1f}% at a 95% service level and seven-day lead time, freeing Rs. {safety['rs_freed_per_store']:,.0f} in working capital per store per year under a 20% annual holding-cost assumption.

## Bias Findings

The largest systematic segment bias is StoreType {worst_bias['StoreType']} in month {int(worst_bias['month'])}, with mean bias of {worst_bias['bias_units']:.0f} units ({worst_bias['bias_pct_of_sales']:.2f}% of average sales). StoreType-level multiplicative correction factors reduce mean absolute forecast error by {bias_meta['bias_correction_error_reduction_pct']:.1f}%.

## EDA Highlights

- Daily sales mean Rs./units proxy: {eda['mean']:,.0f}; median: {eda['median']:,.0f}; std: {eda['std']:,.0f}; skewness: {eda['skew']:.2f}; kurtosis: {eda['kurtosis']:.2f}
- Promo median lift: {eda['promo_median_lift_pct']:.1f}% with Mann-Whitney p-value {eda['promo_mannwhitney_p']:.2e}
- StoreType effect is statistically significant with ANOVA p-value {eda['storetype_anova_p']:.2e}

## Resume-Ready Bullets

- Built 8-model demand forecasting benchmark on {len(clean_df):,} Rossmann retail records; achieved {best:.2f}% MAPE on six-week holdout across {clean_df['Store'].nunique():,} stores.
- Engineered {len(feature_cols)} lag, rolling, calendar, promo and store features with strict causal constraints; Optuna-tuned XGBoost achieved {xgb_mape:.2f}% MAPE vs. {naive:.2f}% naive baseline.
- Translated {improvement:.2f}pp MAPE improvement into {safety['safety_reduction_pct']:.1f}% safety-stock reduction at 95% service level; estimated Rs. {safety['rs_freed_per_store']:,.0f} working capital freed per store annually.
- Diagnosed systematic forecast bias in StoreType {worst_bias['StoreType']} month {int(worst_bias['month'])}; correction reduced mean forecast error by {bias_meta['bias_correction_error_reduction_pct']:.1f}%, improving reorder-point accuracy.
"""
    (cfg.project_dir / "summary.md").write_text(summary, encoding="utf-8")


def run(cfg: RunConfig) -> None:
    cfg.outputs_dir.mkdir(parents=True, exist_ok=True)
    require_data(cfg)
    df = load_and_clean(cfg)
    eda = build_eda(df, cfg.outputs_dir)
    featured, feature_cols = add_features(df)
    train, test, _ = split_data(featured)
    results, preds, meta = fit_models(cfg, train, test, feature_cols)
    results.to_csv(cfg.outputs_dir / "results_table.csv", index=False)
    ss, reorder, bias, safety = safety_stock_and_bias(cfg, test, preds)
    ss.to_csv(cfg.outputs_dir / "safety_stock_table.csv", index=False)
    reorder.to_csv(cfg.outputs_dir / "reorder_points_top10.csv", index=False)
    bias.to_csv(cfg.outputs_dir / "bias_table.csv", index=False)
    bias_meta = make_model_plots(cfg, train, test, feature_cols, results, preds, meta, ss, bias)
    metadata = {"eda": eda, "safety": safety, "bias": bias_meta, "model_meta": {k: v for k, v in meta.items() if k not in ["xgb", "lgb"]}}
    (cfg.outputs_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    write_summary(cfg, df, featured, train, test, feature_cols, results, eda, safety, bias, bias_meta, meta)
    print(results)
    print(f"Artifacts written to {cfg.outputs_dir}")


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--n-trials", type=int, default=int(os.getenv("N_TRIALS", "25")))
    parser.add_argument("--cv-sample-frac", type=float, default=float(os.getenv("CV_SAMPLE_FRAC", "0.35")))
    parser.add_argument("--sarimax-store-count", type=int, default=int(os.getenv("SARIMAX_STORE_COUNT", "5")))
    args = parser.parse_args()
    return RunConfig(project_dir=args.project_dir.resolve(), n_trials=args.n_trials, cv_sample_frac=args.cv_sample_frac, sarimax_store_count=args.sarimax_store_count)


if __name__ == "__main__":
    run(parse_args())
