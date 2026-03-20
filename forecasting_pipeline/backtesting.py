"""
Rolling-origin backtesting framework for the CFL pipeline.

For each product the function ``rolling_backtest`` trains every model on
data up to period t and evaluates the one-step-ahead forecast for period
t+1, using only features that were observable at time t (strict
no-look-ahead policy enforced by the feature-engineering lag shifts).

The competition accuracy metric is used throughout:

    accuracy = 1 - |forecast - actual| / actual

Recent folds are weighted more heavily (``recency_decay`` parameter) so
that the weight computation reflects current behaviour rather than a
uniform average over the full backtest horizon.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

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


# ──────────────── direct accuracy-metric weight optimisation ─────────────────

def _optimize_weights_for_accuracy(
    bt: pd.DataFrame,
    model_names: list[str],
    fold_weights: np.ndarray,
) -> dict[str, float]:
    """Find ensemble weights that *directly* maximise the competition accuracy.

    Uses L-BFGS-B with box constraints (w ∈ [0,1]) and an equality
    constraint (sum = 1).  Falls back to equal weights on any failure.

    Parameters
    ----------
    bt           : backtest results DataFrame with ``<model>_pred`` columns
                   and an ``actual_units`` column.
    model_names  : ordered list of model names (same order as weight vector).
    fold_weights : per-fold recency weight array (len == len(bt)).
    """
    pred_cols = [f"{m}_pred" for m in model_names]
    valid = bt.dropna(subset=["actual_units"]).copy()
    if len(valid) < 2 or not all(c in valid.columns for c in pred_cols):
        n = len(model_names)
        return {m: 1.0 / n for m in model_names}

    actuals = valid["actual_units"].values.astype(float)
    preds   = np.column_stack(
        [valid[c].fillna(0).values.astype(float) for c in pred_cols]
    )
    fw = fold_weights[: len(valid)]  # align fold weights

    def neg_accuracy(w: np.ndarray) -> float:
        w = np.maximum(w, 0.0)
        total_w = w.sum()
        if total_w == 0:
            return 1e9  # penalise zero-weight configurations
        w = w / total_w
        ensemble = preds @ w
        acc = np.array(
            [accuracy_score(float(p), float(a)) for p, a in zip(ensemble, actuals)]
        )
        finite_mask = np.isfinite(acc)
        if not finite_mask.any():
            return 1e9
        fw_masked = fw[finite_mask]
        fw_sum = fw_masked.sum()
        if fw_sum == 0:
            return 1e9
        return -float(np.dot(acc[finite_mask], fw_masked) / fw_sum)

    n = len(model_names)
    w0 = np.ones(n) / n
    bounds = [(0.0, 1.0)] * n
    try:
        result = minimize(
            neg_accuracy,
            w0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-9},
        )
        w = np.maximum(result.x, 0.0)
        total = w.sum()
        if total > 0:
            w = w / total
        else:
            w = np.ones(n) / n
    except Exception:
        w = np.ones(n) / n
    return dict(zip(model_names, w.tolist()))


# ─────────────────────── model confidence from backtest ──────────────────────

def _compute_model_confidence(
    bt: pd.DataFrame,
    model_names: list[str],
    fold_weights: np.ndarray,
) -> float:
    """Return a confidence score in (0, 1) based on backtest accuracy stability.

    High confidence ⟹ low fold-to-fold variance in ensemble accuracy.
    Low confidence  ⟹ highly variable accuracy across folds.

    The score is computed as  1 / (1 + CV)  where CV is the coefficient
    of variation of the per-fold mean accuracy.
    """
    acc_cols = [f"accuracy_{m}" for m in model_names if f"accuracy_{m}" in bt.columns]
    if not acc_cols:
        return 0.5

    valid = bt.dropna(subset=["actual_units"])
    if len(valid) < 2:
        return 0.5

    per_fold_accs = np.nanmean(
        np.column_stack([valid[c].values.astype(float) for c in acc_cols]),
        axis=1,
    )
    finite_mask = np.isfinite(per_fold_accs)
    if finite_mask.sum() < 2:
        return 0.5

    fw = fold_weights[: len(valid)]
    fw_masked = fw[finite_mask]
    mean_acc = float(np.average(per_fold_accs[finite_mask], weights=fw_masked))
    std_acc  = float(np.std(per_fold_accs[finite_mask]))

    if mean_acc <= 0:
        return 0.5

    cv = std_acc / mean_acc
    confidence = 1.0 / (1.0 + cv)
    return float(np.clip(confidence, 0.1, 0.9))


# ─────────────────────────── per-product backtesting ─────────────────────────

def rolling_backtest(
    product_df: pd.DataFrame,
    feature_cols: list[str],
    models: Optional[list[BaseModel]] = None,
    min_train_size: int = 4,
    recency_decay: float = 0.85,
) -> pd.DataFrame:
    """Run rolling-origin backtesting for a single product.

    Parameters
    ----------
    product_df    : rows for one product, sorted by ``quarter_idx``,
                    including a final row for the target quarter
                    (``actual_units = NaN``).
    feature_cols  : feature column names for ML models.
    models        : list of model instances; defaults to ``get_default_models()``.
    min_train_size: minimum number of training observations before we start
                    evaluating.
    recency_decay : exponential decay applied to fold weights – the most
                    recent fold has weight 1.0, earlier folds have weight
                    ``recency_decay^k`` where k is the lag from the last fold.

    Returns
    -------
    pd.DataFrame
        Columns: quarter, actual_units, fold_weight,
                 <model_name>_pred, accuracy_<model_name>, error_<model_name>
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
                    model.fit(y_train, y_train, feature_names=feature_cols)
                    pred = float(model.predict(X_val)[0])
                else:
                    valid_train = train.dropna(subset=["actual_units"])
                    if len(valid_train) < 2:
                        pred = float(np.nanmean(y_train))
                    else:
                        y_tr = valid_train["actual_units"].values.astype(float)
                        X_tr = valid_train[feature_cols].values.astype(float)
                        model.fit(X_tr, y_tr, feature_names=feature_cols)
                        pred = float(model.predict(X_val)[0])
            except Exception:
                pred = float(np.nanmean(y_train))

            pred = max(0.0, pred)
            record[f"{model.name}_pred"] = pred
            record[f"accuracy_{model.name}"] = accuracy_score(pred, actual)
            # Signed error (positive = over-forecast, negative = under-forecast)
            record[f"error_{model.name}"] = pred - actual

        results.append(record)

    bt = pd.DataFrame(results)
    if bt.empty:
        return bt

    # Assign recency weights: most-recent fold = 1.0, earlier folds decay
    n_folds = len(bt)
    bt["fold_weight"] = [recency_decay ** (n_folds - 1 - i) for i in range(n_folds)]
    return bt


# ───────────────────────── portfolio-level backtest ──────────────────────────

def portfolio_backtest(
    feature_df: pd.DataFrame,
    models: Optional[list[BaseModel]] = None,
    min_train_size: int = 4,
    recency_decay: float = 0.85,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run rolling-origin backtesting for all products.

    Parameters
    ----------
    recency_decay : fold-weighting decay passed to ``rolling_backtest``.

    Returns
    -------
    detail_df  : per-product, per-quarter backtest results (with fold_weight)
    summary_df : per-product mean accuracy, bias per model, optimal weights,
                 best model name.
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
            min_train_size=min_train_size, recency_decay=recency_decay,
        )
        if bt.empty:
            continue

        bt["product"] = product
        all_results.append(bt)

        model_names = [m.name for m in models]
        fold_w = bt["fold_weight"].values

        mean_accs: dict[str, float] = {}
        mean_bias: dict[str, float] = {}

        for mname in model_names:
            acc_col = f"accuracy_{mname}"
            err_col = f"error_{mname}"

            if acc_col in bt.columns:
                acc_vals = bt[acc_col].values.astype(float)
                finite_mask = np.isfinite(acc_vals)
                if finite_mask.any():
                    w = fold_w[finite_mask]
                    mean_accs[mname] = float(
                        np.average(acc_vals[finite_mask], weights=w)
                    )
                else:
                    mean_accs[mname] = 0.0

            if err_col in bt.columns:
                err_vals = bt[err_col].values.astype(float)
                finite_mask = np.isfinite(err_vals)
                if finite_mask.any():
                    w = fold_w[finite_mask]
                    mean_bias[mname] = float(
                        np.average(err_vals[finite_mask], weights=w)
                    )
                else:
                    mean_bias[mname] = 0.0

        # ── Product-wise model selection: optimise weights for accuracy ───────
        # Replaces squared-softmax heuristic with direct accuracy maximisation.
        best_model = max(mean_accs, key=mean_accs.get) if mean_accs else None

        if mean_accs:
            optimal_weights = _optimize_weights_for_accuracy(
                bt, model_names, fold_w
            )
        else:
            optimal_weights = {}

        # ── Model confidence from backtest variance ──────────────────────────
        model_confidence = _compute_model_confidence(bt, model_names, fold_w)

        row: dict[str, object] = {
            "product":          product,
            "optimal_weights":  optimal_weights,
            "best_model":       best_model,
            "bias":             mean_bias,
            "model_confidence": model_confidence,
        }
        for k, v in mean_accs.items():
            row[f"mean_accuracy_{k}"] = v
        for k, v in mean_bias.items():
            row[f"bias_{k}"] = v
        summary_rows.append(row)

    detail_df  = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows)
    return detail_df, summary_df
