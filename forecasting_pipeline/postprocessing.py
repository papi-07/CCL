"""
Post-processing of raw model forecasts.

Steps applied:
1. Clip negative values to zero.
2. IQR-based outlier detection: cap predictions that deviate wildly from
   the recent historical distribution.
3. Lifecycle-aware jump smoothing: NPI-Ramp products allow larger growth;
   Decline products are more aggressively capped on the upside.
4. Anchor constraint: limit predictions within 0.7×–1.3× of last actual
   (skipped for NPI-Ramp products to allow natural ramp-up).
5. Round to the nearest integer (products are counted in whole units).
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


def anchor_constraint(
    forecast: float,
    last_actual: float,
    lower_factor: float = 0.7,
    upper_factor: float = 1.3,
) -> float:
    """Limit the forecast within [lower_factor × last_actual, upper_factor × last_actual].

    Prevents large single-quarter swings for stable/declining products.
    No-op when ``last_actual`` is zero or non-finite (constraint cannot be applied).

    Parameters
    ----------
    forecast      : candidate forecast value
    last_actual   : most-recent known actual
    lower_factor  : minimum ratio relative to last actual (default 0.7)
    upper_factor  : maximum ratio relative to last actual (default 1.3)
    """
    if not np.isfinite(last_actual) or last_actual <= 0:
        return forecast
    lower = last_actual * lower_factor
    upper = last_actual * upper_factor
    return float(np.clip(forecast, lower, upper))


def _lifecycle_change_ratio(
    life_cycle: str,
    ts_class: str,
    base_ratio: float,
) -> tuple[float, bool]:
    """Return (max_change_ratio, apply_jump_smoothing) based on lifecycle stage.

    Rules
    -----
    NPI-Ramp   : new/ramping product – allow larger upward swings (×4.0).
    Decline    : declining product   – limit upside to ×2.0.
    Sustaining : default.
    intermittent: skip jump smoothing entirely (sporadic by nature).
    volatile   : widen ratio to ×3.5.
    """
    lc = str(life_cycle).strip()
    ts = str(ts_class).strip()

    # Intermittent demand – jump smoothing is not appropriate
    if ts == "intermittent":
        return base_ratio, False

    if lc == "NPI-Ramp":
        ratio = max(base_ratio, 4.0)
    elif lc == "Decline":
        ratio = min(base_ratio, 2.0)
    elif ts == "volatile":
        ratio = max(base_ratio, 3.5)
    else:
        ratio = base_ratio

    return ratio, True


def postprocess(
    forecast: float,
    historical_units: np.ndarray,
    last_actual: float,
    iqr_multiplier: float = 2.0,
    max_change_ratio: float = 3.0,
    apply_jump_smoothing: bool = True,
    apply_anchor: bool = True,
    anchor_lower: float = 0.7,
    anchor_upper: float = 1.3,
    round_to_int: bool = True,
    life_cycle: str = "Sustaining",
    ts_class: str = "stable",
) -> float:
    """Apply the full post-processing chain with lifecycle-aware smoothing.

    Parameters
    ----------
    forecast          : raw model ensemble forecast
    historical_units  : array of historical actual_units for the product
    last_actual       : most-recent known actual (for jump smoothing / anchor)
    iqr_multiplier    : k in the IQR bound formula
    max_change_ratio  : default max allowed single-quarter growth/decline factor;
                        may be overridden by lifecycle/ts_class rules.
    apply_jump_smoothing: whether to apply smooth_jump (also overridden by
                        lifecycle rules for intermittent products).
    apply_anchor      : whether to apply the anchor constraint.  Automatically
                        disabled for NPI-Ramp and intermittent products.
    anchor_lower      : lower factor for anchor constraint (default 0.7)
    anchor_upper      : upper factor for anchor constraint (default 1.3)
    round_to_int      : round to nearest integer (True by default)
    life_cycle        : product lifecycle stage ("Sustaining", "NPI-Ramp",
                        "Decline", …)
    ts_class          : time-series classification ("stable", "volatile",
                        "intermittent", "new")

    Returns
    -------
    float : post-processed forecast
    """
    # Derive lifecycle-aware parameters
    ratio, do_jump = _lifecycle_change_ratio(life_cycle, ts_class, max_change_ratio)
    if not apply_jump_smoothing:
        do_jump = False

    # Anchor constraint is skipped for NPI-Ramp (natural ramp-up) and
    # intermittent products (sporadic by nature).
    do_anchor = (
        apply_anchor
        and str(life_cycle).strip() != "NPI-Ramp"
        and str(ts_class).strip() != "intermittent"
    )

    fc = clip_negatives(forecast)
    fc = cap_outliers(fc, historical_units, iqr_multiplier)
    if do_jump:
        fc = smooth_jump(fc, last_actual, ratio)
    if do_anchor:
        fc = anchor_constraint(fc, last_actual, anchor_lower, anchor_upper)
    if round_to_int:
        fc = float(round(fc))
    return fc
