"""
Daily net revenue forecasting pipeline
Inputs:
  - daily_net_revenue.csv (Date, NetRevenue)
  - sales_transactions.csv (transaction-level source used for reconciliation)
Outputs are written to ./forecast_outputs by default.

Run:
  python forecasting_pipeline.py --daily daily_net_revenue.csv --transactions sales_transactions.csv
"""
from __future__ import annotations
import argparse
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import acf

RANDOM_STATE = 42
LAGS = (1, 2, 3, 7, 14, 21, 28)
ROLLING_WINDOWS = (7, 14, 28)
WEEKLY_MIN_HISTORY = 3 * 7
WEEKLY_ACF_THRESHOLD = 0.20
FINAL_TEST_DAYS = 28
VALIDATION_HORIZON = 14
N_VALIDATION_FOLDS = 3
FORECAST_HORIZON = 7


def load_and_prepare(daily_path: Path, transactions_path: Path | None):
    """Load daily target, enforce one observation per calendar day, and audit transactions."""
    raw = pd.read_csv(daily_path)
    required = {"Date", "NetRevenue"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"{daily_path} is missing required columns: {sorted(missing)}")

    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce").dt.normalize()
    raw["NetRevenue"] = pd.to_numeric(raw["NetRevenue"], errors="coerce")
    raw = raw.dropna(subset=["Date", "NetRevenue"])
    if raw.empty:
        raise ValueError("No valid Date/NetRevenue rows found.")
    if raw["Date"].duplicated().any():
        # Aggregate duplicate daily records rather than arbitrarily keeping one.
        raw = raw.groupby("Date", as_index=False)["NetRevenue"].sum()

    # Reindex to a complete daily calendar. Missing dates are zero-sales days,
    # not missing target observations, for this retail revenue series.
    calendar = pd.date_range(raw.Date.min(), raw.Date.max(), freq="D")
    y = (raw.set_index("Date")["NetRevenue"]
             .sort_index()
             .reindex(calendar, fill_value=0.0)
             .astype(float))
    y.index.name = "Date"
    y.name = "NetRevenue"

    audit = pd.DataFrame()
    if transactions_path and transactions_path.exists():
        tx = pd.read_csv(transactions_path)
        if "Date" not in tx.columns or "Revenue" not in tx.columns:
            raise ValueError("Transaction file must contain Date and Revenue columns.")
        tx["Date"] = pd.to_datetime(tx["Date"], errors="coerce").dt.normalize()
        tx["Revenue"] = pd.to_numeric(tx["Revenue"], errors="coerce")
        tx = tx.dropna(subset=["Date", "Revenue"])
        tx_daily = tx.groupby("Date")["Revenue"].sum().rename("TransactionRevenue")
        audit = y.rename("DailyFileRevenue").to_frame().join(tx_daily, how="outer")
        audit["Difference"] = audit["DailyFileRevenue"] - audit["TransactionRevenue"]
        audit["DailyFileRevenue"] = audit["DailyFileRevenue"].fillna(0.0)
        audit["TransactionRevenue"] = audit["TransactionRevenue"].fillna(0.0)
        audit["Difference"] = audit["DailyFileRevenue"] - audit["TransactionRevenue"]
        audit.index.name = "Date"
        audit = audit.reset_index()
    return y, audit


def make_features(history: pd.Series, date: pd.Timestamp) -> pd.DataFrame:
    """Create features using observations strictly before the forecast date."""
    values = history.astype(float)
    row = {
        "dayofweek": date.dayofweek,
        "day": date.day,
        "month": date.month,
        "quarter": date.quarter,
        "year": date.year,
        "dayofyear": date.dayofyear,
        "weekofyear": int(date.isocalendar().week),
        "is_weekend": int(date.dayofweek >= 5),
        "is_month_start": int(date.is_month_start),
        "is_month_end": int(date.is_month_end),
    }
    for lag in LAGS:
        row[f"lag_{lag}"] = float(values.iloc[-lag]) if len(values) >= lag else np.nan
    for window in ROLLING_WINDOWS:
        # The newest value in history is yesterday relative to the date forecast.
        tail = values.iloc[-window:]
        row[f"rolling_mean_{window}"] = float(tail.mean()) if len(tail) else np.nan
        row[f"rolling_std_{window}"] = float(tail.std(ddof=0)) if len(tail) else np.nan
        row[f"rolling_median_{window}"] = float(tail.median()) if len(tail) else np.nan
    return pd.DataFrame([row], index=[date])


def build_training_matrix(y: pd.Series):
    rows, targets = [], []
    for i in range(max(LAGS), len(y)):
        date = y.index[i]
        rows.append(make_features(y.iloc[:i], date))
        targets.append(y.iloc[i])
    X = pd.concat(rows)
    target = pd.Series(targets, index=X.index, name="target")
    return X, target


def weekly_seasonality_supported(train: pd.Series) -> tuple[bool, float]:
    """Conservative weekly diagnostic: enough history and meaningful lag-7 ACF."""
    if len(train) < WEEKLY_MIN_HISTORY:
        return False, np.nan
    vals = train.astype(float).values
    if np.nanstd(vals) == 0:
        return False, 0.0
    try:
        lag7_acf = float(acf(vals, nlags=7, fft=True, missing="drop")[7])
    except Exception:
        return False, np.nan
    return bool(np.isfinite(lag7_acf) and lag7_acf >= WEEKLY_ACF_THRESHOLD), lag7_acf


def forecast_naive(train: pd.Series, horizon: int) -> np.ndarray:
    return np.repeat(float(train.iloc[-1]), horizon)


def forecast_seasonal_naive(train: pd.Series, horizon: int, season: int = 7) -> np.ndarray:
    if len(train) < season:
        return forecast_naive(train, horizon)
    pattern = train.iloc[-season:].to_numpy(dtype=float)
    return np.array([pattern[i % season] for i in range(horizon)], dtype=float)


def forecast_exponential_smoothing(train: pd.Series, horizon: int):
    weekly, lag7 = weekly_seasonality_supported(train)
    kwargs = {"trend": "add", "initialization_method": "estimated"}
    if weekly:
        kwargs.update({"seasonal": "add", "seasonal_periods": 7})
    else:
        kwargs.update({"seasonal": None})
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ExponentialSmoothing(train.astype(float), **kwargs).fit(optimized=True)
        return np.asarray(model.forecast(horizon), dtype=float), ("weekly_additive" if weekly else "trend_only"), lag7
    except Exception:
        # A robust trend-only fallback avoids dropping the model from comparisons.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ExponentialSmoothing(
                train.astype(float), trend="add", seasonal=None,
                initialization_method="estimated"
            ).fit(optimized=True)
        return np.asarray(model.forecast(horizon), dtype=float), "trend_only_fallback", lag7


def forecast_hgb(train: pd.Series, horizon: int) -> np.ndarray:
    X, target = build_training_matrix(train)
    model = HistGradientBoostingRegressor(
        max_iter=250, learning_rate=0.05, max_leaf_nodes=15,
        l2_regularization=1.0, random_state=RANDOM_STATE
    )
    model.fit(X, target)
    history = train.copy()
    predictions = []
    for step in range(horizon):
        date = train.index[-1] + pd.Timedelta(days=step + 1)
        x_next = make_features(history, date).reindex(columns=X.columns)
        pred = float(model.predict(x_next)[0])
        predictions.append(pred)
        history.loc[date] = pred
    return np.asarray(predictions, dtype=float)


def forecast_all(train: pd.Series, horizon: int):
    es_pred, es_variant, lag7 = forecast_exponential_smoothing(train, horizon)
    return {
        "Naive": forecast_naive(train, horizon),
        "SeasonalNaive_7": forecast_seasonal_naive(train, horizon, 7),
        f"ExponentialSmoothing_{es_variant}": es_pred,
        "HistGradientBoosting": forecast_hgb(train, horizon),
    }, lag7


def metric_row(model_name: str, actual: np.ndarray, predicted: np.ndarray, period: str):
    return {
        "Period": period,
        "Model": model_name,
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "N_Forecasts": int(len(actual)),
    }


def chronological_validation(y: pd.Series):
    """Three expanding-window, 14-day rolling-origin validation folds."""
    test_start = len(y) - FINAL_TEST_DAYS
    first_val_start = test_start - N_VALIDATION_FOLDS * VALIDATION_HORIZON
    if first_val_start < max(90, max(LAGS) + 1):
        raise ValueError("Not enough history for the requested validation and final test windows.")
    rows, details = [], []
    for fold in range(N_VALIDATION_FOLDS):
        start = first_val_start + fold * VALIDATION_HORIZON
        end = start + VALIDATION_HORIZON
        train = y.iloc[:start]
        actual = y.iloc[start:end].to_numpy(dtype=float)
        preds, lag7 = forecast_all(train, len(actual))
        for model_name, pred in preds.items():
            rows.append(metric_row(model_name, actual, pred, f"Validation_Fold_{fold+1}"))
            for j, (date, obs, estimate) in enumerate(zip(y.index[start:end], actual, pred)):
                details.append({
                    "Period": f"Validation_Fold_{fold+1}", "Model": model_name,
                    "Date": date, "Actual": obs, "Forecast": float(estimate),
                    "Error": float(obs-estimate), "AbsError": float(abs(obs-estimate)),
                    "SquaredError": float((obs-estimate)**2),
                    "TrainEnd": train.index[-1], "Lag7_ACF": lag7
                })
    return pd.DataFrame(rows), pd.DataFrame(details)


def run_pipeline(daily_path: Path, transactions_path: Path | None, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    y, audit = load_and_prepare(daily_path, transactions_path)
    y.rename("NetRevenue").to_csv(output_dir / "prepared_daily_series.csv", header=True)
    if not audit.empty:
        audit.to_csv(output_dir / "transaction_reconciliation.csv", index=False)

    # Validation model comparison (expanding chronological windows).
    val_metrics, val_detail = chronological_validation(y)
    val_metrics.to_csv(output_dir / "validation_model_comparison.csv", index=False)
    val_detail.to_csv(output_dir / "validation_forecasts.csv", index=False)

    # Final test is untouched during model selection: the last 28 days.
    test_start = len(y) - FINAL_TEST_DAYS
    train_test, actual_test = y.iloc[:test_start], y.iloc[test_start:]
    test_preds, test_lag7 = forecast_all(train_test, len(actual_test))
    test_rows, test_details = [], []
    for name, pred in test_preds.items():
        test_rows.append(metric_row(name, actual_test.to_numpy(), pred, "Final_Test"))
        for date, obs, estimate in zip(actual_test.index, actual_test.to_numpy(), pred):
            test_details.append({
                "Period": "Final_Test", "Model": name, "Date": date,
                "Actual": float(obs), "Forecast": float(estimate),
                "Error": float(obs-estimate), "AbsError": float(abs(obs-estimate)),
                "SquaredError": float((obs-estimate)**2),
                "TrainEnd": train_test.index[-1], "Lag7_ACF": test_lag7
            })
    test_metrics = pd.DataFrame(test_rows)
    test_detail = pd.DataFrame(test_details)
    test_metrics.to_csv(output_dir / "final_test_model_comparison.csv", index=False)
    test_detail.to_csv(output_dir / "final_test_forecasts.csv", index=False)

    # Refit on all observed data and forecast the next seven calendar days.
    future_preds, full_lag7 = forecast_all(y, FORECAST_HORIZON)
    forecast_rows = []
    future_dates = pd.date_range(y.index[-1] + pd.Timedelta(days=1), periods=FORECAST_HORIZON, freq="D")
    for name, pred in future_preds.items():
        for date, estimate in zip(future_dates, pred):
            forecast_rows.append({
                "Date": date, "Model": name, "Forecast_NetRevenue": float(estimate),
                "TrainingDataThrough": y.index[-1], "Weekly_ACF_Lag7": full_lag7
            })
    forecast_table = pd.DataFrame(forecast_rows)
    forecast_table.to_csv(output_dir / "seven_day_forecast.csv", index=False)

    # Also export a compact wide format for easy reporting.
    wide = forecast_table.pivot(index="Date", columns="Model", values="Forecast_NetRevenue").reset_index()
    wide.to_csv(output_dir / "seven_day_forecast_wide.csv", index=False)

    # Aggregate validation metrics by model; folds are equally sized.
    validation_summary = (val_metrics.groupby("Model", as_index=False)
                          .agg(Validation_MAE=("MAE", "mean"),
                               Validation_RMSE=("RMSE", "mean")))
    comparison = validation_summary.merge(
        test_metrics[["Model", "MAE", "RMSE"]].rename(
            columns={"MAE": "Final_Test_MAE", "RMSE": "Final_Test_RMSE"}
        ), on="Model", how="outer"
    ).sort_values("Validation_MAE")
    comparison.to_csv(output_dir / "model_comparison_summary.csv", index=False)

    print("\nPrepared daily series:", len(y), "days", y.index.min().date(), "to", y.index.max().date())
    print("\nValidation summary (mean across chronological folds):")
    print(validation_summary.sort_values("Validation_MAE").to_string(index=False))
    print("\nFinal test metrics (last 28 days):")
    print(test_metrics.sort_values("MAE").to_string(index=False))
    print("\n7-day forecast (all models):")
    print(wide.to_string(index=False))
    if not audit.empty:
        print("\nTransaction reconciliation:")
        print("Rows:", len(audit), "| max absolute difference:",
              float(audit["Difference"].abs().max()))
    print("\nOutputs saved to:", output_dir.resolve())
    return comparison, forecast_table, test_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--daily", type=Path, default=Path("daily_net_revenue.csv"))
    parser.add_argument("--transactions", type=Path, default=Path("sales_transactions.csv"))
    parser.add_argument("--output", type=Path, default=Path("forecast_outputs"))
    args = parser.parse_args()
    tx_path = args.transactions if args.transactions.exists() else None
    run_pipeline(args.daily, tx_path, args.output)


if __name__ == "__main__":
    main()
