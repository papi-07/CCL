#!/usr/bin/env python3
"""
Cisco Forecast League Phase 1 – main entry point.

Run from the repository root:

    python main.py

The pipeline will:
  1. Load data from CFL - Training materials & Data Pack/CFL_External Data Pack_Phase1.xlsx
  2. Engineer features
  3. Run rolling-origin backtesting
  4. Produce an ensemble forecast for FY26Q2
  5. Post-process and save results to fy26q2_forecasts.csv
"""

import sys
import warnings

warnings.filterwarnings("ignore")

from forecasting_pipeline.forecast import run_pipeline


def main() -> None:
    print("=" * 60)
    print("  Cisco Forecast League – FY26Q2 Forecasting Pipeline")
    print("=" * 60)

    results = run_pipeline(verbose=True)

    print("\n── FY26 Q2 Forecast Summary ──────────────────────────────")
    display_cols = [
        "cost_rank", "product", "ts_class", "best_model",
        "fy26q2_forecast", "expert_mean", "expert_blend_pred",
        "dp_forecast", "mktg_forecast", "ds_forecast",
    ]
    display_cols = [c for c in display_cols if c in results.columns]
    print(results[display_cols].to_string(index=False))
    print(f"\nTotal portfolio forecast: {results['fy26q2_forecast'].sum():,.0f} units")
    print("=" * 60)


if __name__ == "__main__":
    main()
