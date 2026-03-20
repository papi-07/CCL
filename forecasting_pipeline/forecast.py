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
    compute_expert_accuracy_estimate,
)
from forecasting_pipeline.postprocessing import postprocess

# Model confidence below this threshold triggers the last-actual fallback.
_LOW_CONF_THRESHOLD = 0.20

# Dampening factor for the Q2 seasonal index.
# Moves the seasonal factor halfway between 1.0 and the raw Q2/average ratio
# to avoid overcorrecting when the model already captures seasonality.
_SEASONAL_DAMPENING = 0.5


def run_pipeline(
    data_file: str = _DATA_FILE,
    output_csv: Optional[str] = "fy26q2_forecasts.csv",
    models: Optional[list[BaseModel]] = None,
    min_train_size: int = 4,
    recency_decay: float = 0.85,
    expert_weight: float = 0.15,
    expert_override_threshold: float = 0.1,
    portfolio_calibration: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run the end-to-end forecasting pipeline.

    Parameters
    ----------
    data_file                 : path to CFL_External Data Pack_Phase1.xlsx
    output_csv                : where to save the forecast CSV (None to skip)
    models                    : model list; defaults to ``get_default_models()``
    min_train_size            : minimum quarters before backtesting starts
    recency_decay             : fold-weighting decay for backtesting (recent
                                folds receive proportionally higher weight)
    expert_weight             : *base* fraction of final ensemble weight given
                                to the credibility-blended expert forecast.
                                Actual weight is adjusted per-product by
                                model confidence.
    expert_override_threshold : margin by which expert accuracy estimate must
                                beat model accuracy to trigger expert override.
    portfolio_calibration     : if True, scale all non-NPI forecasts so the
                                portfolio total matches the last-2-quarter
                                historical average (±20 % cap).
    verbose                   : print progress messages

    Returns
    -------
    pd.DataFrame with columns:
        cost_rank, product, life_cycle, ts_class,
        fy26q2_forecast, <model>_pred, ensemble_raw,
        dp_forecast, mktg_forecast, ds_forecast,
        expert_mean, expert_blend_pred,
        best_model, optimal_weights, bias_applied,
        expert_weights_used, feature_importances,
        expert_override_applied, disagreement_blend,
        model_confidence, calibration_factor,
        seasonal_factor, global_bias_correction,
        low_conf_fallback
    """
    # ── 0. Load data ─────────────────────────────────────────────────────────
    if verbose:
        print("[1/8] Loading data …")
    tables = load_all(data_file)

    # ── 1. Feature engineering ────────────────────────────────────────────────
    if verbose:
        print("[2/8] Building feature matrix …")
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
        print(f"[3/8] Models: {[m.name for m in models]}")

    # ── 3. Portfolio backtesting ──────────────────────────────────────────────
    if verbose:
        print("[4/8] Running rolling-origin backtesting …")

    # Use separate model instances for backtesting to avoid state bleed
    bt_models = [type(m)() for m in models]
    detail_df, summary_df = portfolio_backtest(
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

    # ── Global bias correction ────────────────────────────────────────────────
    # Compute portfolio-level mean residual bias from the backtest error columns.
    # This catches any systematic over/under-forecasting at the portfolio level
    # that was not captured by the per-product per-model bias corrections.
    global_bias = 0.0
    if not detail_df.empty:
        error_cols = [c for c in detail_df.columns if c.startswith("error_")]
        if error_cols:
            all_errors = detail_df[error_cols].values.astype(float).flatten()
            finite_errors = all_errors[np.isfinite(all_errors)]
            if len(finite_errors) > 0:
                global_bias = float(np.mean(finite_errors))
    if verbose:
        print(f"      Global bias correction: {global_bias:.4f}")

    # Build per-product weight, bias, confidence, and accuracy lookups
    weight_lookup:     dict[str, dict[str, float]] = {}
    bias_lookup:       dict[str, dict[str, float]] = {}
    best_model_lookup: dict[str, str]              = {}
    confidence_lookup: dict[str, float]            = {}
    mean_acc_lookup:   dict[str, float]            = {}

    if not summary_df.empty:
        for _, row in summary_df.iterrows():
            p = row["product"]
            if isinstance(row.get("optimal_weights"), dict):
                weight_lookup[p] = row["optimal_weights"]
            if isinstance(row.get("bias"), dict):
                bias_lookup[p] = row["bias"]
            if pd.notna(row.get("best_model")):
                best_model_lookup[p] = row["best_model"]
            if pd.notna(row.get("model_confidence")):
                confidence_lookup[p] = float(row["model_confidence"])
            # Mean accuracy across all models for this product
            acc_vals = [
                float(row[f"mean_accuracy_{m.name}"])
                for m in models
                if f"mean_accuracy_{m.name}" in row and pd.notna(row[f"mean_accuracy_{m.name}"])
            ]
            if acc_vals:
                mean_acc_lookup[p] = float(np.mean(acc_vals))

    default_weights = {m.name: 1.0 / len(models) for m in models}

    # ── 4. Expert forecast credibility weights ────────────────────────────────
    if verbose:
        print("[5/8] Computing expert forecast credibility weights …")
    per_product_expert_weights = compute_expert_weights(feat_df)

    # ── 5. Generate final FY26Q2 forecasts ───────────────────────────────────
    if verbose:
        print("[6/8] Generating FY26Q2 forecasts …")

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
        model_conf     = confidence_lookup.get(product, 0.5)

        # Weighted expert forecast
        expert_pred = blend_expert_forecasts(tgt_row, exp_weights)

        # Expert accuracy estimate (for expert override decision)
        expert_acc  = compute_expert_accuracy_estimate(tgt_row)
        model_acc   = mean_acc_lookup.get(product, 0.5)

        # ── Intermittent demand: use median of last non-zero actuals ──────────
        life_cycle  = str(tgt_row.get("life_cycle", "Sustaining"))
        ts_class    = str(tgt_row.get("ts_class", "stable"))
        hist_units  = product_hist["actual_units"].values.astype(float)

        actuals_series = product_hist["actual_units"].dropna()
        last_actual = float(actuals_series.iloc[-1]) if not actuals_series.empty else 0.0
        prev_actual = float(actuals_series.iloc[-2]) if len(actuals_series) >= 2 else np.nan

        # ── Per-product Q2 seasonal factor ───────────────────────────────────
        # Compute the ratio of mean Q2 actuals to the overall mean.
        # Apply with 50 % dampening to avoid overcorrection (the ML models
        # already capture some seasonality via is_Q2 and lag-4 features).
        q2_hist = product_hist[
            product_hist["quarter"].str.endswith("Q2")
        ]["actual_units"].dropna()
        seasonal_factor = 1.0
        if len(q2_hist) >= 1 and len(actuals_series) >= 2:
            all_mean = float(actuals_series.mean())
            if all_mean > 0:
                q2_mean = float(q2_hist.mean())
                q2_ratio = q2_mean / all_mean
                # Dampen: move halfway between 1.0 and the raw ratio
                seasonal_factor = 1.0 + _SEASONAL_DAMPENING * (q2_ratio - 1.0)

        # ── Intermittent demand: use median of last non-zero actuals ──────────
        _use_intermittent = False
        if ts_class == "intermittent":
            non_zero = hist_units[hist_units > 0]
            if len(non_zero) >= 2:
                intermittent_median = float(
                    np.median(non_zero[-min(6, len(non_zero)):])
                )
                raw_forecast  = intermittent_median
                final         = float(round(intermittent_median))
                result = {
                    "product":                product,
                    "ensemble_forecast":      raw_forecast,
                    "weights_used":           {"intermittent_median": 1.0},
                    "feature_importances":    {},
                    "bias_applied":           {},
                    "expert_override_applied": False,
                    "disagreement_blend":     False,
                }
                for m in models:
                    result[f"{m.name}_pred"] = np.nan
                _use_intermittent = True

        _low_conf_fallback = False
        if not _use_intermittent:
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
                model_confidence=model_conf,
                model_mean_accuracy=model_acc,
                expert_accuracy_estimate=expert_acc,
                expert_override_threshold=expert_override_threshold,
            )

            raw_forecast = result["ensemble_forecast"]

            # ── Global bias correction ────────────────────────────────────────
            # Subtract the portfolio-level systematic residual bias.
            raw_forecast = max(0.0, raw_forecast - global_bias)

            # ── Q2 seasonal adjustment ────────────────────────────────────────
            raw_forecast = max(0.0, raw_forecast * seasonal_factor)

            final = postprocess(
                forecast=raw_forecast,
                historical_units=hist_units,
                last_actual=last_actual,
                prev_actual=prev_actual,
                life_cycle=life_cycle,
                ts_class=ts_class,
            )

            # ── Low-confidence fallback ───────────────────────────────────────
            # When the model is highly uncertain (backtest variance too high),
            # fall back to the last observed actual as a safe baseline.
            if model_conf < _LOW_CONF_THRESHOLD and last_actual > 0:
                final = float(round(last_actual))
                _low_conf_fallback = True

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
            "cost_rank":              tgt_row.get("cost_rank"),
            "product":                product,
            "life_cycle":             life_cycle,
            "ts_class":               ts_class,
            "fy26q2_forecast":        final,
            "ensemble_raw":           raw_forecast,
            "dp_forecast":            tgt_row.get("dp_forecast"),
            "mktg_forecast":          tgt_row.get("mktg_forecast"),
            "ds_forecast":            tgt_row.get("ds_forecast"),
            "expert_mean":            expert_mean,
            "expert_blend_pred":      expert_pred if expert_pred is not None else np.nan,
            "best_model":             best_model,
            "optimal_weights":        str(result.get("weights_used", {})),
            "bias_applied":           str(bias_corr),
            "expert_weights_used":    str(exp_weights),
            "feature_importances":    str(fi_summary),
            "expert_override_applied": result.get("expert_override_applied", False),
            "disagreement_blend":     result.get("disagreement_blend", False),
            "model_confidence":       model_conf,
            "calibration_factor":     1.0,   # filled after portfolio calibration
            "seasonal_factor":        seasonal_factor,
            "global_bias_correction": global_bias,
            "low_conf_fallback":      _low_conf_fallback,
        }
        for m in models:
            record[f"{m.name}_pred"] = result.get(f"{m.name}_pred", np.nan)
        # Include expert blend prediction column
        record["expert_blend_model_pred"] = result.get("expert_blend_pred", np.nan)

        forecast_records.append(record)

    forecast_df = pd.DataFrame(forecast_records).sort_values(
        "cost_rank"
    ).reset_index(drop=True)

    # ── 6. Portfolio calibration ──────────────────────────────────────────────
    # Scale non-NPI forecasts so portfolio total ≈ last-2-quarter average.
    # Cap the calibration factor to ±20 % to avoid extreme adjustments.
    if portfolio_calibration and not forecast_df.empty:
        if verbose:
            print("[7/8] Applying portfolio calibration …")
        cal_mask = forecast_df["life_cycle"] != "NPI-Ramp"
        cal_products = set(forecast_df.loc[cal_mask, "product"].tolist())

        q_sorted  = sorted(hist_df["quarter_idx"].unique())
        last_2_q  = q_sorted[-2:] if len(q_sorted) >= 2 else q_sorted

        recent_totals = []
        for q in last_2_q:
            q_sum = float(
                hist_df[
                    (hist_df["quarter_idx"] == q)
                    & (hist_df["product"].isin(cal_products))
                ]["actual_units"].sum()
            )
            if q_sum > 0:
                recent_totals.append(q_sum)

        avg_recent = float(np.mean(recent_totals)) if recent_totals else 0.0
        fc_total   = float(forecast_df.loc[cal_mask, "fy26q2_forecast"].sum())

        if avg_recent > 0 and fc_total > 0:
            cal_factor = float(np.clip(avg_recent / fc_total, 0.80, 1.20))
            forecast_df.loc[cal_mask, "fy26q2_forecast"] = (
                forecast_df.loc[cal_mask, "fy26q2_forecast"] * cal_factor
            ).round()
            forecast_df.loc[cal_mask, "calibration_factor"] = cal_factor
            if verbose:
                print(f"      Calibration factor: {cal_factor:.4f} "
                      f"(forecast_total={fc_total:,.0f}, "
                      f"recent_avg={avg_recent:,.0f})")

    # ── 7. Save output ────────────────────────────────────────────────────────
    if output_csv:
        out_path = Path(output_csv)
        forecast_df.to_csv(out_path, index=False)
        if verbose:
            print(f"[8/8] Forecast saved to {out_path.resolve()}")
    elif verbose:
        print("[8/8] Done (output_csv=None, no file written)")

    return forecast_df
