"""
Rolling-origin backtesting framework for the CFL pipeline.

For each product the function ``rolling_backtest`` trains every model on
data up to period t and evaluates the one-step-ahead forecast for period
t+1, using only features that were observable at time t (strict
no-look-ahead policy enforced by the feature-engineering lag shifts).

The competition accuracy metric is used throughout:

    accuracy = 1 - |forecast - actual| / actual
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd

from forecasting_pipeline.feature_engineering import (
    FY_QUARTERS_ORDERED,
    get_feature_columns,
)
from forecasting_pipeline.models import BaseModel, get_default_models


# ─────────────────────────── accuracy metric ─────────────────────────────────

def accuracy_score(forecast: float, actual: float) -> float:
    """Competition accuracy = 1 - |forecast - actual| / actual.

    Returns NaN when actual == 0 to avoid division by zero.
    Clamped to [0, 1] so that extreme over-forecasting does not give
    negative values (the competition floor is 0).
    """
    if actual == 0 or not np.isfinite(actual):
        return np.nan
    raw = 1.0 - abs(forecast - actual) / abs(actual)
    return float(np.clip(raw, 0.0, 1.0))


# ─────────────────────────── per-product backtesting ─────────────────────────

def rolling_backtest(
    product_df: pd.DataFrame,
    feature_cols: list[str],
    models: Optional[list[BaseModel]] = None,
    min_train_size: int = 4,
) -> pd.DataFrame:
    """Run rolling-origin backtesting for a single product.

    Parameters
    ----------
    product_df   : rows for one product, sorted by ``quarter_idx``,
                   including a final row for the target quarter
                   (``actual_units = NaN``).
    feature_cols : feature column names for ML models.
    models       : list of model instances; defaults to ``get_default_models()``.
    min_train_size: minimum number of training observations before we start
                    evaluating.

    Returns
    -------
    pd.DataFrame
        Columns: quarter, actual_units, <model_name>_pred, accuracy_<model_name>
    """
    if models is None:
        models = get_default_models()

    df = product_df.sort_values("quarter_idx").reset_index(drop=True)

    # Rows with known actuals only (skip target-quarter rows)
    hist = df[df["actual_units"].notna()].reset_index(drop=True)
    n = len(hist)

    results = []
    for t in range(min_train_size, n):
        train = hist.iloc[:t]
        val_row = hist.iloc[t]

        actual = float(val_row["actual_units"])
        record: dict[str, object] = {
            "quarter":      val_row["quarter"],
            "actual_units": actual,
        }

        y_train = train["actual_units"].values.astype(float)
        X_train = train[feature_cols].values.astype(float)
        X_val   = val_row[feature_cols].values.reshape(1, -1).astype(float)

        for model in models:
            try:
                if model.name in ("holt_winters", "arima", "naive_seasonal"):
                    model.fit(y_train, y_train)
                    pred = float(model.predict(X_val)[0])
                else:
                    valid_train = train.dropna(subset=["actual_units"])
                    if len(valid_train) < 2:
                        pred = float(np.nanmean(y_train))
                    else:
                        y_tr = valid_train["actual_units"].values.astype(float)
                        X_tr = valid_train[feature_cols].values.astype(float)
                        model.fit(X_tr, y_tr)
                        pred = float(model.predict(X_val)[0])
            except Exception:
                pred = float(np.nanmean(y_train))

            pred = max(0.0, pred)
            record[f"{model.name}_pred"] = pred
            record[f"accuracy_{model.name}"] = accuracy_score(pred, actual)

        results.append(record)

    return pd.DataFrame(results)


# ───────────────────────── portfolio-level backtest ──────────────────────────

def portfolio_backtest(
    feature_df: pd.DataFrame,
    models: Optional[list[BaseModel]] = None,
    min_train_size: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run rolling-origin backtesting for all products.

    Returns
    -------
    detail_df  : per-product, per-quarter backtest results
    summary_df : per-product mean accuracy across backtest windows,
                 plus an 'optimal_weights' dict column for the ensemble.
    """
    if models is None:
        models = get_default_models()

    feature_cols = get_feature_columns(feature_df)
    all_results  = []
    summary_rows  = []

    for product, grp in feature_df.groupby("product"):
        grp_sorted = grp.sort_values("quarter_idx")
        bt = rolling_backtest(
            grp_sorted, feature_cols, models=models,
            min_train_size=min_train_size
        )
        if bt.empty:
            continue

        bt["product"] = product
        all_results.append(bt)

        model_names = [m.name for m in models]
        acc_cols = [f"accuracy_{m}" for m in model_names]
        mean_accs = {}
        for col in acc_cols:
            if col in bt.columns:
                mean_accs[col.replace("accuracy_", "")] = float(
                    bt[col].dropna().mean()
                )

        # Weights proportional to mean accuracy (softmax-style)
        if mean_accs:
            raw_weights = np.array(list(mean_accs.values()), dtype=float)
            raw_weights = np.where(np.isfinite(raw_weights), raw_weights, 0.0)
            raw_weights = np.clip(raw_weights, 0.0, None)
            total = raw_weights.sum()
            weights = raw_weights / total if total > 0 else np.ones_like(raw_weights) / len(raw_weights)
            optimal_weights = dict(zip(mean_accs.keys(), weights.tolist()))
        else:
            optimal_weights = {}

        row = {"product": product, "optimal_weights": optimal_weights}
        for k, v in mean_accs.items():
            row[f"mean_accuracy_{k}"] = v
        summary_rows.append(row)

    detail_df  = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows)
    return detail_df, summary_df
