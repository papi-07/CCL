"""
Feature engineering for the CFL forecasting pipeline.

The main entry-point is ``build_feature_matrix`` which:

1. Merges Actual Bookings with auxiliary tables (SCMS, VMS, Big Deal).
2. Classifies each product time-series (stable / volatile / intermittent / new).
3. Creates lag features, rolling statistics, growth rates, seasonality dummies,
   VMS/SCMS ratio features, and expert-forecast deviation features.

All operations are strictly time-ordered to prevent any look-ahead leakage.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd

from forecasting_pipeline.data_loader import _FY_TO_CAL, _CAL_TO_FY, _fy_to_cal


# ──────────────────────────────── constants ──────────────────────────────────

# Ordered list of all Cisco FY quarters present in the Actuals sheet
FY_QUARTERS_ORDERED = [
    "FY23Q2", "FY23Q3", "FY23Q4",
    "FY24Q1", "FY24Q2", "FY24Q3", "FY24Q4",
    "FY25Q1", "FY25Q2", "FY25Q3", "FY25Q4",
    "FY26Q1",
]

# Corresponding calendar quarters (auxiliary data)
CAL_QUARTERS_ORDERED = [_FY_TO_CAL[q] for q in FY_QUARTERS_ORDERED]

# The target quarter we need to predict
TARGET_FY_QUARTER  = "FY26Q2"
TARGET_CAL_QUARTER = _FY_TO_CAL[TARGET_FY_QUARTER]   # "2026Q2"


# ─────────────────────────── product classification ──────────────────────────

def classify_products(actuals: pd.DataFrame) -> pd.DataFrame:
    """Classify each product's time-series behaviour.

    Categories
    ----------
    new         : fewer than 4 non-NaN observations
    intermittent: coefficient of variation > 1.2 or >40 % zeros
    volatile    : coefficient of variation between 0.5 and 1.2
    stable      : otherwise (low CV, regular demand)

    Returns the actuals DataFrame with an added ``ts_class`` column.
    """
    product_stats = (
        actuals.groupby("product")["actual_units"]
        .agg(
            n_obs=lambda x: x.notna().sum(),
            mean=lambda x: x.mean(),
            std=lambda x: x.std(ddof=0),
            zero_frac=lambda x: (x.fillna(0) == 0).mean(),
        )
        .reset_index()
    )
    product_stats["cv"] = product_stats["std"] / product_stats["mean"].replace(0, np.nan)

    def _classify(row: pd.Series) -> str:
        if row["n_obs"] < 4:
            return "new"
        if row["zero_frac"] > 0.4 or row["cv"] > 1.2:
            return "intermittent"
        if row["cv"] > 0.5:
            return "volatile"
        return "stable"

    product_stats["ts_class"] = product_stats.apply(_classify, axis=1)
    actuals = actuals.merge(
        product_stats[["product", "ts_class"]], on="product", how="left"
    )
    return actuals


# ──────────────────────── auxiliary signal aggregation ───────────────────────

def _aggregate_scms(scms: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    """Aggregate SCMS data to product-quarter level and join onto actuals."""
    keep = scms[["product", "cal_quarter", "scms_total"]].copy()
    actuals = actuals.merge(keep, on=["product", "cal_quarter"], how="left")
    return actuals


def _aggregate_vms(vms: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    """Aggregate VMS data to product-quarter level and join onto actuals."""
    keep = vms[["product", "cal_quarter", "vms_total"]].copy()
    actuals = actuals.merge(keep, on=["product", "cal_quarter"], how="left")
    return actuals


def _aggregate_big_deal(
    big_deal: pd.DataFrame, actuals: pd.DataFrame
) -> pd.DataFrame:
    """Join big-deal / avg-deal data onto actuals."""
    keep = big_deal[
        ["product", "cal_quarter", "big_deals", "avg_deals"]
    ].copy()
    actuals = actuals.merge(keep, on=["product", "cal_quarter"], how="left")
    return actuals


# ─────────────────────────────── lag builder ─────────────────────────────────

def _add_lag_features(
    df: pd.DataFrame,
    col: str,
    lags: tuple[int, ...] = (1, 2, 3, 4),
    group_col: str = "product",
) -> pd.DataFrame:
    """Add lag columns for *col* within each group, sorted by quarter index."""
    df = df.sort_values([group_col, "quarter_idx"])
    for lag in lags:
        df[f"{col}_lag{lag}"] = df.groupby(group_col)[col].shift(lag)
    return df


def _add_rolling_features(
    df: pd.DataFrame,
    col: str,
    windows: tuple[int, ...] = (2, 4),
    group_col: str = "product",
) -> pd.DataFrame:
    """Add rolling mean features for *col* within each group."""
    df = df.sort_values([group_col, "quarter_idx"])
    for w in windows:
        df[f"{col}_roll{w}"] = (
            df.groupby(group_col)[col]
            .transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
        )
    return df


# ───────────────────────────── main builder ──────────────────────────────────

def build_feature_matrix(
    actuals: pd.DataFrame,
    big_deal: pd.DataFrame,
    scms: pd.DataFrame,
    vms: pd.DataFrame,
    target_quarter: Optional[str] = None,
) -> pd.DataFrame:
    """Build the full feature matrix.

    Parameters
    ----------
    actuals, big_deal, scms, vms : raw tables from ``data_loader.load_all``
    target_quarter : optional FY quarter string for which to also create a
        prediction row (no actuals needed; defaults to TARGET_FY_QUARTER).

    Returns
    -------
    pd.DataFrame
        One row per (product, quarter) with all features and the target column
        ``actual_units``.  The target quarter row has ``actual_units = NaN``.
    """
    if target_quarter is None:
        target_quarter = TARGET_FY_QUARTER

    # ── 1. Add calendar quarter mapping ─────────────────────────────────────
    actuals = actuals.copy()
    actuals["cal_quarter"] = actuals["quarter"].map(_FY_TO_CAL)

    # ── 2. Classify products ─────────────────────────────────────────────────
    actuals = classify_products(actuals)

    # ── 3. Add quarter index for sorting ────────────────────────────────────
    q_to_idx = {q: i for i, q in enumerate(FY_QUARTERS_ORDERED)}
    actuals["quarter_idx"] = actuals["quarter"].map(q_to_idx)

    # ── 4. Append blank target-quarter rows ──────────────────────────────────
    target_rows = (
        actuals[["product", "cost_rank", "life_cycle", "ts_class"]]
        .drop_duplicates()
        .copy()
    )
    target_rows["quarter"] = target_quarter
    target_rows["cal_quarter"] = _FY_TO_CAL.get(target_quarter, target_quarter)
    target_rows["quarter_idx"] = len(FY_QUARTERS_ORDERED)  # after all history
    target_rows["actual_units"] = np.nan

    # Copy the expert forecasts into the target row
    expert_cols = ["dp_forecast", "mktg_forecast", "ds_forecast"]
    for col in expert_cols:
        last_vals = (
            actuals.dropna(subset=[col])
            .groupby("product")[col]
            .last()
        )
        target_rows[col] = target_rows["product"].map(last_vals)

    actuals = pd.concat([actuals, target_rows], ignore_index=True)
    actuals = actuals.sort_values(["product", "quarter_idx"]).reset_index(
        drop=True
    )

    # ── 5. Merge auxiliary signals ──────────────────────────────────────────
    actuals = _aggregate_scms(scms, actuals)
    actuals = _aggregate_vms(vms, actuals)
    actuals = _aggregate_big_deal(big_deal, actuals)

    # ── 6. Lag features on target ────────────────────────────────────────────
    actuals = _add_lag_features(actuals, "actual_units", lags=(1, 2, 3, 4))
    actuals = _add_rolling_features(actuals, "actual_units", windows=(2, 4))

    # ── 7. Lag features on SCMS / VMS ────────────────────────────────────────
    for sig in ("scms_total", "vms_total"):
        if sig in actuals.columns:
            actuals = _add_lag_features(actuals, sig, lags=(1, 2))
            actuals = _add_rolling_features(actuals, sig, windows=(2,))

    # ── 8. Growth-rate features ──────────────────────────────────────────────
    actuals = actuals.sort_values(["product", "quarter_idx"])
    for lag in (1, 2):
        lag_col = f"actual_units_lag{lag}"
        actuals[f"qoq_growth_lag{lag}"] = (
            actuals["actual_units_lag1"] - actuals[lag_col]
        ) / actuals[lag_col].replace(0, np.nan)

    # ── 9. Seasonality encoding ──────────────────────────────────────────────
    # Cisco FY Q1=Aug-Oct, Q2=Nov-Jan, Q3=Feb-Apr, Q4=May-Jul
    actuals["fy_quarter_num"] = actuals["quarter"].str[-1].astype(float)
    for q in range(1, 5):
        actuals[f"is_Q{q}"] = (actuals["fy_quarter_num"] == q).astype(int)

    # ── 10. Trend index ─────────────────────────────────────────────────────
    actuals["trend"] = actuals["quarter_idx"]

    # ── 11. VMS / SCMS ratio ────────────────────────────────────────────────
    if "vms_total" in actuals.columns and "scms_total" in actuals.columns:
        actuals["vms_scms_ratio"] = actuals["vms_total"] / actuals[
            "scms_total"
        ].replace(0, np.nan)

    # ── 12. Big-deal ratio ───────────────────────────────────────────────────
    if "big_deals" in actuals.columns and "avg_deals" in actuals.columns:
        actuals["big_deal_frac"] = actuals["big_deals"] / (
            actuals["big_deals"] + actuals["avg_deals"]
        ).replace(0, np.nan)

    # ── 13. Expert forecast deviation from lag-1 ─────────────────────────────
    for col in expert_cols:
        if col in actuals.columns:
            actuals[f"{col}_dev"] = (
                actuals[col] - actuals["actual_units_lag1"]
            ) / actuals["actual_units_lag1"].replace(0, np.nan)

    # ── 14. Life-cycle encoding ──────────────────────────────────────────────
    lc_map = {"Sustaining": 0, "NPI-Ramp": 1, "Decline": -1}
    actuals["life_cycle_enc"] = actuals["life_cycle"].map(lc_map).fillna(0)

    return actuals.reset_index(drop=True)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of numeric feature columns usable by ML models."""
    exclude = {
        "cost_rank", "product", "life_cycle", "ts_class",
        "quarter", "cal_quarter", "actual_units",
    }
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric if c not in exclude]
