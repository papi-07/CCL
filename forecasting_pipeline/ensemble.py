"""
Weighted ensemble that blends predictions from all constituent models.

Ensemble weights are derived from the rolling-origin backtest accuracy
scores (higher accuracy ⟹ higher weight, sharpened via softmax squaring).
When backtesting data is insufficient, equal weights are used.

Additional capabilities
-----------------------
* Expert forecast blending: DP, Marketing, and Data Science forecasts are
  combined via credibility-weighted averaging and included as an extra
  ensemble signal.
* Bias correction: per-product per-model bias estimated from backtesting is
  subtracted from predictions before blending.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from forecasting_pipeline.models import BaseModel


# ──────────────────────────── expert blending ────────────────────────────────

_EXPERT_COLS = ("dp_forecast", "mktg_forecast", "ds_forecast")


def compute_expert_weights(
    feat_df: pd.DataFrame,
    expert_cols: tuple[str, ...] = _EXPERT_COLS,
) -> dict[str, dict[str, float]]:
    """Compute per-product expert credibility weights.

    Since we only have FY26Q2 expert forecasts (not historical per-quarter
    expert forecasts), we use *proximity to the naive seasonal baseline*
    (lag-4) as a credibility proxy:

        weight_raw[expert] = 1 / (1 + |expert_fc - naive| / naive)

    Experts whose FY26Q2 forecast aligns more closely with same-quarter-last-
    year history receive higher weight. Weights are normalised to sum to 1.

    Parameters
    ----------
    feat_df     : full feature matrix (must contain ``actual_units_lag4``).
    expert_cols : names of the expert forecast columns.

    Returns
    -------
    dict mapping product → {expert_col: weight}
    """
    result: dict[str, dict[str, float]] = {}
    target_rows = feat_df[feat_df["quarter"] == "FY26Q2"].copy()
    n_experts = len(expert_cols)

    for _, row in target_rows.iterrows():
        product = str(row["product"])
        naive = float(row.get("actual_units_lag4", np.nan))

        if not np.isfinite(naive) or naive <= 0:
            result[product] = {col: 1.0 / n_experts for col in expert_cols}
            continue

        weights_raw: dict[str, float] = {}
        for col in expert_cols:
            val = row.get(col, np.nan)
            if pd.notna(val) and np.isfinite(float(val)) and float(val) > 0:
                rel_err = abs(float(val) - naive) / naive
                weights_raw[col] = 1.0 / (1.0 + rel_err)
            else:
                weights_raw[col] = 0.0

        total = sum(weights_raw.values())
        if total > 0:
            result[product] = {k: v / total for k, v in weights_raw.items()}
        else:
            result[product] = {col: 1.0 / n_experts for col in expert_cols}

    return result


def blend_expert_forecasts(
    row: pd.Series,
    expert_weights: dict[str, float],
    expert_cols: tuple[str, ...] = _EXPERT_COLS,
) -> Optional[float]:
    """Return the credibility-weighted average of available expert forecasts.

    Returns ``None`` if no expert values are available.
    """
    weighted_sum = 0.0
    total_w = 0.0
    for col in expert_cols:
        val = row.get(col, np.nan)
        if pd.notna(val) and np.isfinite(float(val)) and float(val) > 0:
            w = expert_weights.get(col, 1.0 / len(expert_cols))
            weighted_sum += w * float(val)
            total_w += w
    if total_w == 0:
        return None
    return weighted_sum / total_w


# ──────────────────────────── core ensemble logic ────────────────────────────

def ensemble_predict(
    predictions: dict[str, float],
    weights: dict[str, float],
) -> float:
    """Return the weighted average of model predictions.

    Parameters
    ----------
    predictions : {model_name: predicted_value}
    weights     : {model_name: weight} – need not sum to 1 (normalised here)

    Returns
    -------
    float : ensemble forecast (weighted mean)
    """
    total_weight = 0.0
    weighted_sum = 0.0
    for name, pred in predictions.items():
        w = weights.get(name, 1.0)
        if np.isfinite(pred) and np.isfinite(w) and w > 0:
            weighted_sum += w * pred
            total_weight += w

    if total_weight == 0:
        finite_preds = [v for v in predictions.values() if np.isfinite(v)]
        return float(np.mean(finite_preds)) if finite_preds else 0.0

    return weighted_sum / total_weight


def build_ensemble_forecast(
    product: str,
    train_df: pd.DataFrame,
    pred_row: pd.Series,
    feature_cols: list[str],
    models: list[BaseModel],
    weights: dict[str, float],
    bias_correction: Optional[dict[str, float]] = None,
    expert_pred: Optional[float] = None,
    expert_weight: float = 0.15,
) -> dict[str, object]:
    """Train all models on full history and return the blended forecast.

    Parameters
    ----------
    product         : product name (for labelling)
    train_df        : all historical rows with known ``actual_units``
    pred_row        : the feature row for the target quarter
    feature_cols    : ML feature column names
    models          : list of model instances
    weights         : {model_name: weight} from backtesting
    bias_correction : optional {model_name: bias} – subtracted before blending.
                      Bias = mean(pred - actual) from backtest, so subtracting
                      it removes systematic over-/under-forecasting.
    expert_pred     : optional weighted expert forecast to include in ensemble
    expert_weight   : fraction of ensemble weight given to expert forecast
                      (remaining ``1 - expert_weight`` distributed to models)

    Returns
    -------
    dict with keys: product, <model>_pred, ensemble_forecast,
                    weights_used, feature_importances, bias_applied
    """
    y_train = train_df["actual_units"].values.astype(float)
    X_train = train_df[feature_cols].values.astype(float)
    X_pred  = pred_row[feature_cols].values.reshape(1, -1).astype(float)

    predictions: dict[str, float] = {}
    feature_importances: dict[str, Optional[np.ndarray]] = {}

    for model in models:
        try:
            if model.name in ("holt_winters", "arima", "naive_seasonal"):
                model.fit(y_train, y_train, feature_names=feature_cols)
                pred = float(model.predict(X_pred)[0])
            else:
                valid = train_df.dropna(subset=["actual_units"])
                if len(valid) < 2:
                    pred = float(np.nanmean(y_train))
                else:
                    y_tr = valid["actual_units"].values.astype(float)
                    X_tr = valid[feature_cols].values.astype(float)
                    model.fit(X_tr, y_tr, feature_names=feature_cols)
                    pred = float(model.predict(X_pred)[0])
                # Capture feature importances
                if hasattr(model, "feature_importances_") and model.feature_importances_ is not None:
                    feature_importances[model.name] = model.feature_importances_
        except Exception:
            pred = float(np.nanmean(y_train))

        # Apply per-model bias correction before clamping
        if bias_correction and model.name in bias_correction:
            pred = pred - bias_correction[model.name]

        predictions[model.name] = max(0.0, pred)

    # ── Include expert blend as an additional signal ─────────────────────────
    blended_weights = {k: w * (1.0 - expert_weight) for k, w in weights.items()}
    if expert_pred is not None and np.isfinite(expert_pred) and expert_pred >= 0:
        predictions["expert_blend"] = expert_pred
        blended_weights["expert_blend"] = expert_weight
    else:
        # Re-normalise model weights to 1.0 when no expert forecast available
        blended_weights = weights

    ensemble = ensemble_predict(predictions, blended_weights)
    ensemble = max(0.0, ensemble)

    result: dict[str, object] = {
        "product":           product,
        "ensemble_forecast": ensemble,
        "weights_used":      blended_weights,
        "feature_importances": feature_importances,
        "bias_applied":      bias_correction or {},
    }
    result.update({f"{k}_pred": v for k, v in predictions.items()})
    return result
