"""
Post-processing of raw model forecasts.

Steps applied:
1. Clip negative values to zero.
2. IQR-based outlier detection: cap predictions that deviate wildly from
   the recent historical distribution.
3. Round to the nearest integer (products are counted in whole units).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def clip_negatives(forecast: float) -> float:
    """Ensure forecast is non-negative."""
    return max(0.0, float(forecast))


def _iqr_bounds(series: np.ndarray, k: float = 2.0) -> tuple[float, float]:
    """Return (lower, upper) IQR-based bounds."""
    q1 = float(np.nanpercentile(series, 25))
    q3 = float(np.nanpercentile(series, 75))
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def cap_outliers(
    forecast: float,
    historical_units: np.ndarray,
    iqr_multiplier: float = 2.0,
) -> float:
    """Cap forecast at IQR-based bounds derived from the historical series."""
    finite_hist = historical_units[np.isfinite(historical_units)]
    if len(finite_hist) < 4:
        return forecast

    lo, hi = _iqr_bounds(finite_hist, k=iqr_multiplier)
    # Never cap below 0
    lo = max(0.0, lo)
    return float(np.clip(forecast, lo, hi))


def smooth_jump(
    forecast: float,
    last_actual: float,
    max_change_ratio: float = 3.0,
) -> float:
    """Prevent unrealistic single-quarter jumps.

    If the forecast is more than ``max_change_ratio`` times the last
    observed value, it is clipped.
    """
    if not np.isfinite(last_actual) or last_actual <= 0:
        return forecast
    upper = last_actual * max_change_ratio
    lower = last_actual / max_change_ratio
    return float(np.clip(forecast, lower, upper))


def postprocess(
    forecast: float,
    historical_units: np.ndarray,
    last_actual: float,
    iqr_multiplier: float = 2.0,
    max_change_ratio: float = 3.0,
    apply_jump_smoothing: bool = True,
    round_to_int: bool = True,
) -> float:
    """Apply the full post-processing chain.

    Parameters
    ----------
    forecast          : raw model ensemble forecast
    historical_units  : array of historical actual_units for the product
    last_actual       : most-recent known actual (for jump smoothing)
    iqr_multiplier    : k in the IQR bound formula
    max_change_ratio  : max allowed single-quarter growth / decline factor
    apply_jump_smoothing: whether to apply smooth_jump
    round_to_int      : round to nearest integer (True by default)

    Returns
    -------
    float : post-processed forecast
    """
    fc = clip_negatives(forecast)
    fc = cap_outliers(fc, historical_units, iqr_multiplier)
    if apply_jump_smoothing:
        fc = smooth_jump(fc, last_actual, max_change_ratio)
    if round_to_int:
        fc = float(round(fc))
    return fc
