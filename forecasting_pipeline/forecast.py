"""
Main forecasting orchestrator for the CFL Phase 1 pipeline.

Usage
-----
    from forecasting_pipeline.forecast import run_pipeline
    results = run_pipeline()
    print(results[["product", "fy26q2_forecast"]])
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from forecasting_pipeline.data_loader import load_all, _DATA_FILE
from forecasting_pipeline.feature_engineering import (
    build_feature_matrix,
    get_feature_columns,
    TARGET_FY_QUARTER,
)
from forecasting_pipeline.models import get_default_models, BaseModel
from forecasting_pipeline.backtesting import portfolio_backtest
from forecasting_pipeline.ensemble import (
    build_ensemble_forecast,
    compute_expert_weights,
    blend_expert_forecasts,
)
from forecasting_pipeline.postprocessing import postprocess


def run_pipeline(
    data_file: str = _DATA_FILE,
    output_csv: Optional[str] = "fy26q2_forecasts.csv",
    models: Optional[list[BaseModel]] = None,
    min_train_size: int = 4,
    recency_decay: float = 0.85,
    expert_weight: float = 0.15,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run the end-to-end forecasting pipeline.

    Parameters
    ----------
    data_file     : path to CFL_External Data Pack_Phase1.xlsx
    output_csv    : where to save the forecast CSV (None to skip)
    models        : model list; defaults to ``get_default_models()``
    min_train_size: minimum quarters before backtesting starts
    recency_decay : fold-weighting decay for backtesting (recent folds
                    receive proportionally higher weight)
    expert_weight : fraction of final ensemble weight given to the
                    credibility-blended expert forecast (DP / Marketing / DS)
    verbose       : print progress messages

    Returns
    -------
    pd.DataFrame with columns:
        cost_rank, product, life_cycle, ts_class,
        fy26q2_forecast, <model>_pred, ensemble_raw,
        dp_forecast, mktg_forecast, ds_forecast,
        expert_mean, expert_blend_pred,
        best_model, optimal_weights, bias_applied,
        expert_weights_used, feature_importances
    """
    # ── 0. Load data ─────────────────────────────────────────────────────────
    if verbose:
        print("[1/7] Loading data …")
    tables = load_all(data_file)

    # ── 1. Feature engineering ────────────────────────────────────────────────
    if verbose:
        print("[2/7] Building feature matrix …")
    feat_df = build_feature_matrix(
        actuals=tables["actuals"],
        big_deal=tables["big_deal"],
        scms=tables["scms"],
        vms=tables["vms"],
    )

    feature_cols = get_feature_columns(feat_df)
    if verbose:
        print(f"      {len(feature_cols)} feature columns, "
              f"{feat_df['product'].nunique()} products")

    # ── 2. Instantiate models ─────────────────────────────────────────────────
    if models is None:
        models = get_default_models()
    if verbose:
        print(f"[3/7] Models: {[m.name for m in models]}")

    # ── 3. Portfolio backtesting ──────────────────────────────────────────────
    if verbose:
        print("[4/7] Running rolling-origin backtesting …")

    # Use separate model instances for backtesting to avoid state bleed
    bt_models = [type(m)() for m in models]
    _, summary_df = portfolio_backtest(
        feat_df, models=bt_models,
        min_train_size=min_train_size,
        recency_decay=recency_decay,
    )

    if verbose and not summary_df.empty:
        model_names = [m.name for m in models]
        acc_cols = [f"mean_accuracy_{m}" for m in model_names
                    if f"mean_accuracy_{m}" in summary_df.columns]
        if acc_cols:
            print("      Portfolio mean accuracy by model (recency-weighted):")
            for col in acc_cols:
                print(f"        {col}: {summary_df[col].mean():.4f}")

    # Build per-product weight and bias lookups
    weight_lookup: dict[str, dict[str, float]] = {}
    bias_lookup:   dict[str, dict[str, float]] = {}
    best_model_lookup: dict[str, str] = {}

    if not summary_df.empty:
        for _, row in summary_df.iterrows():
            p = row["product"]
            if isinstance(row.get("optimal_weights"), dict):
                weight_lookup[p] = row["optimal_weights"]
            if isinstance(row.get("bias"), dict):
                bias_lookup[p] = row["bias"]
            if pd.notna(row.get("best_model")):
                best_model_lookup[p] = row["best_model"]

    default_weights = {m.name: 1.0 / len(models) for m in models}

    # ── 4. Expert forecast credibility weights ────────────────────────────────
    if verbose:
        print("[5/7] Computing expert forecast credibility weights …")
    per_product_expert_weights = compute_expert_weights(feat_df)

    # ── 5. Generate final FY26Q2 forecasts ───────────────────────────────────
    if verbose:
        print("[6/7] Generating FY26Q2 forecasts …")

    hist_df   = feat_df[feat_df["actual_units"].notna()].copy()
    target_df = feat_df[feat_df["quarter"] == TARGET_FY_QUARTER].copy()

    forecast_records = []
    for _, tgt_row in target_df.iterrows():
        product = tgt_row["product"]

        product_hist = hist_df[hist_df["product"] == product].sort_values(
            "quarter_idx"
        )

        if product_hist.empty:
            continue

        weights        = weight_lookup.get(product, default_weights)
        bias_corr      = bias_lookup.get(product, {})
        best_model     = best_model_lookup.get(product, "")
        exp_weights    = per_product_expert_weights.get(product, {})

        # Weighted expert forecast
        expert_pred = blend_expert_forecasts(tgt_row, exp_weights)

        result = build_ensemble_forecast(
            product=product,
            train_df=product_hist,
            pred_row=tgt_row,
            feature_cols=feature_cols,
            models=models,
            weights=weights,
            bias_correction=bias_corr,
            expert_pred=expert_pred,
            expert_weight=expert_weight,
        )

        # Post-processing with lifecycle awareness
        life_cycle = str(tgt_row.get("life_cycle", "Sustaining"))
        ts_class   = str(tgt_row.get("ts_class", "stable"))
        hist_units = product_hist["actual_units"].values.astype(float)
        last_actual = float(
            product_hist["actual_units"].dropna().iloc[-1]
            if not product_hist["actual_units"].dropna().empty
            else 0.0
        )
        raw_forecast = result["ensemble_forecast"]
        final = postprocess(
            forecast=raw_forecast,
            historical_units=hist_units,
            last_actual=last_actual,
            life_cycle=life_cycle,
            ts_class=ts_class,
        )

        # Expert average (unweighted, for reference)
        expert_vals = [
            tgt_row.get("dp_forecast"),
            tgt_row.get("mktg_forecast"),
            tgt_row.get("ds_forecast"),
        ]
        valid_experts = [
            float(v) for v in expert_vals
            if v is not None and np.isfinite(float(v))
        ]
        expert_mean = float(np.mean(valid_experts)) if valid_experts else np.nan

        # Serialise feature importances as top-10 feature names → score
        fi_dict = result.get("feature_importances", {})
        fi_summary: dict[str, object] = {}
        for mname, fi in fi_dict.items():
            if fi is not None and hasattr(fi, "__len__") and len(fi) == len(feature_cols):
                idx = np.argsort(fi)[::-1][:10]
                fi_summary[mname] = {
                    feature_cols[i]: float(fi[i]) for i in idx
                }

        record = {
            "cost_rank":            tgt_row.get("cost_rank"),
            "product":              product,
            "life_cycle":           life_cycle,
            "ts_class":             ts_class,
            "fy26q2_forecast":      final,
            "ensemble_raw":         raw_forecast,
            "dp_forecast":          tgt_row.get("dp_forecast"),
            "mktg_forecast":        tgt_row.get("mktg_forecast"),
            "ds_forecast":          tgt_row.get("ds_forecast"),
            "expert_mean":          expert_mean,
            "expert_blend_pred":    expert_pred if expert_pred is not None else np.nan,
            "best_model":           best_model,
            "optimal_weights":      str(result.get("weights_used", {})),
            "bias_applied":         str(bias_corr),
            "expert_weights_used":  str(exp_weights),
            "feature_importances":  str(fi_summary),
        }
        for m in models:
            record[f"{m.name}_pred"] = result.get(f"{m.name}_pred", np.nan)
        # Include expert blend prediction column
        record["expert_blend_model_pred"] = result.get("expert_blend_pred", np.nan)

        forecast_records.append(record)

    forecast_df = pd.DataFrame(forecast_records).sort_values(
        "cost_rank"
    ).reset_index(drop=True)

    # ── 6. Save output ────────────────────────────────────────────────────────
    if output_csv:
        out_path = Path(output_csv)
        forecast_df.to_csv(out_path, index=False)
        if verbose:
            print(f"[7/7] Forecast saved to {out_path.resolve()}")
    elif verbose:
        print("[7/7] Done (output_csv=None, no file written)")

    return forecast_df
