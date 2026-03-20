"""
Weighted ensemble that blends predictions from all constituent models.

Ensemble weights are derived from the rolling-origin backtest accuracy
scores (higher accuracy ⟹ higher weight).  When backtesting data is
insufficient, equal weights are used.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from forecasting_pipeline.models import BaseModel


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
) -> dict[str, object]:
    """Train all models on full history and return the blended forecast.

    Parameters
    ----------
    product      : product name (for labelling)
    train_df     : all historical rows with known ``actual_units``
    pred_row     : the feature row for the target quarter
    feature_cols : ML feature column names
    models       : list of model instances
    weights      : {model_name: weight} from backtesting

    Returns
    -------
    dict with keys: product, <model>_pred, ensemble_forecast,
                    weights_used, feature_importances
    """
    y_train = train_df["actual_units"].values.astype(float)
    X_train = train_df[feature_cols].values.astype(float)
    X_pred  = pred_row[feature_cols].values.reshape(1, -1).astype(float)

    predictions: dict[str, float] = {}
    feature_importances: dict[str, Optional[np.ndarray]] = {}

    for model in models:
        try:
            if model.name in ("holt_winters", "arima", "naive_seasonal"):
                model.fit(y_train, y_train)
                pred = float(model.predict(X_pred)[0])
            else:
                valid = train_df.dropna(subset=["actual_units"])
                if len(valid) < 2:
                    pred = float(np.nanmean(y_train))
                else:
                    y_tr = valid["actual_units"].values.astype(float)
                    X_tr = valid[feature_cols].values.astype(float)
                    model.fit(X_tr, y_tr)
                    pred = float(model.predict(X_pred)[0])
                # Capture feature importances
                if hasattr(model, "feature_importances_") and model.feature_importances_ is not None:
                    feature_importances[model.name] = model.feature_importances_
        except Exception:
            pred = float(np.nanmean(y_train))

        predictions[model.name] = max(0.0, pred)

    ensemble = ensemble_predict(predictions, weights)
    ensemble = max(0.0, ensemble)

    result: dict[str, object] = {
        "product": product,
        "ensemble_forecast": ensemble,
        "weights_used": weights,
        "feature_importances": feature_importances,
    }
    result.update({f"{k}_pred": v for k, v in predictions.items()})
    return result
