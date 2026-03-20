"""
Data loading and cleaning for the CFL Phase 1 data pack.

All time periods are standardised to a canonical quarter string "FY<yy>Q<n>"
(Cisco fiscal year notation) for the Actual Bookings table, and to calendar
"<YYYY>Q<n>" strings for the auxiliary tables (Big Deal, SCMS, VMS).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


_DATA_FILE = (
    "CFL - Training materials & Data Pack/CFL_External Data Pack_Phase1.xlsx"
)

# Mapping from Cisco fiscal quarter (FY-prefixed) to the 4-digit year notation
# used by the auxiliary datasets (SCMS, VMS, Big Deal).
# Both notations refer to the same Cisco fiscal quarter, e.g.
#   "FY23Q2"  ≡  "2023Q2"  (November 2022 – January 2023)
def _fy_to_cal(fy_quarter: str) -> str:
    """Convert "FY23Q2" → "2023Q2" (strip leading "FY", expand 2-digit year)."""
    q = fy_quarter.upper().lstrip("FY")   # "23Q2"
    year_part = q[:2]                      # "23"
    q_part    = q[2:]                      # "Q2"
    return f"20{year_part}{q_part}"        # "2023Q2"


_FY_TO_CAL: dict[str, str] = {
    q: _fy_to_cal(q)
    for q in [
        "FY23Q2", "FY23Q3", "FY23Q4",
        "FY24Q1", "FY24Q2", "FY24Q3", "FY24Q4",
        "FY25Q1", "FY25Q2", "FY25Q3", "FY25Q4",
        "FY26Q1", "FY26Q2",
    ]
}

_CAL_TO_FY: dict[str, str] = {v: k for k, v in _FY_TO_CAL.items()}


def _strip(v: object) -> str:
    """Return str(v) with internal whitespace collapsed."""
    return " ".join(str(v).split())


def load_actuals(data_file: str = _DATA_FILE) -> pd.DataFrame:
    """Load the 'Data Pack - Actual Bookings' sheet.

    Sheet layout (header=None reading):
        Row 0 : column-group headers (Cost Rank, ACTUAL UNITS, Forecasted Units…)
        Row 1 : sub-headers (expert-forecast team names, target quarter label)
        Row 2 : quarter labels (FY23 Q2 … FY26 Q2)
        Row 3 : first product row
        …
        Row 32: last product row (30 products total)

    Returns a long-format DataFrame with columns:
        cost_rank, product, life_cycle, quarter (FY notation, e.g. "FY23Q2"),
        actual_units, dp_forecast, mktg_forecast, ds_forecast
    """
    raw = pd.read_excel(data_file, sheet_name="Data Pack - Actual Bookings",
                        header=None)

    # Row 2 contains the quarter labels starting at column index 3
    # Columns 3-14 = 12 historical quarters (FY23 Q2 → FY26 Q1)
    # Column 15   = FY26 Q2 (target – blank in the data rows)
    # Columns 16-18 = expert forecasts (DP, Marketing, Data Science)
    quarter_row = raw.iloc[2].tolist()

    actual_quarters: list[str] = []
    for v in quarter_row[3:15]:          # 12 historical quarters
        if pd.notna(v):
            actual_quarters.append(_strip(v).replace(" ", ""))

    # Product data rows: row 3 to first blank cost-rank row
    data_rows = raw.iloc[3:33].reset_index(drop=True)

    records = []
    for _, row in data_rows.iterrows():
        cost_rank = row.iloc[0]
        if pd.isna(cost_rank):
            continue
        product    = _strip(row.iloc[1])
        life_cycle = _strip(row.iloc[2])

        # Expert forecasts (columns 16, 17, 18)
        dp_fc   = row.iloc[16]
        mktg_fc = row.iloc[17]
        ds_fc   = row.iloc[18]

        for i, q in enumerate(actual_quarters):
            val = row.iloc[3 + i]
            records.append({
                "cost_rank":     int(cost_rank),
                "product":       product,
                "life_cycle":    life_cycle,
                "quarter":       q,
                "actual_units":  float(val) if pd.notna(val) else np.nan,
                "dp_forecast":   float(dp_fc)   if pd.notna(dp_fc)   else np.nan,
                "mktg_forecast": float(mktg_fc) if pd.notna(mktg_fc) else np.nan,
                "ds_forecast":   float(ds_fc)   if pd.notna(ds_fc)   else np.nan,
            })

    df = pd.DataFrame(records)
    df["cal_quarter"] = df["quarter"].map(_FY_TO_CAL)
    return df


def load_big_deal(data_file: str = _DATA_FILE) -> pd.DataFrame:
    """Load the 'Big Deal' sheet.

    Sheet layout (header=None):
        Row 0 : section headers (Cost Rank, PLID Masked, MFG Book Units, Big Deals, Avg Deals)
        Row 1 : quarter labels (2024Q2 … 2026Q1) repeated for each section
        Row 2+: product data

    Returns a long-format DataFrame with columns:
        product, cal_quarter, mfg_book_units, big_deals, avg_deals
    """
    raw = pd.read_excel(data_file, sheet_name="Big Deal", header=None)

    # Quarter labels are in row 1, columns 2-9 (8 quarters)
    quarter_labels = [_strip(v) for v in raw.iloc[1, 2:10].tolist()]

    records = []
    for _, row in raw.iloc[2:].iterrows():
        cost_rank = row.iloc[0]
        if pd.isna(cost_rank):
            continue
        product = _strip(row.iloc[1])
        for i, q in enumerate(quarter_labels):
            records.append({
                "product":        product,
                "cal_quarter":    q,
                "mfg_book_units": _safe_float(row.iloc[2 + i]),
                "big_deals":      _safe_float(row.iloc[10 + i]),
                "avg_deals":      _safe_float(row.iloc[18 + i]),
            })

    return pd.DataFrame(records)


def load_scms(data_file: str = _DATA_FILE) -> pd.DataFrame:
    """Load the 'SCMS' sheet and aggregate by product + quarter.

    Sheet layout (header=None):
        Row 0 : column headers
        Row 1 : blank separator
        Row 2 : quarter labels (2023Q1 … 2026Q1, 13 quarters) at columns 3-15
        Row 3+: product × segment data

    Returns a long-format DataFrame with columns:
        product, cal_quarter, scms_<segment>, scms_total
    """
    raw = pd.read_excel(data_file, sheet_name="SCMS", header=None)

    # Quarter labels in row 2, columns 3 through 15 (13 quarters)
    quarter_labels = [_strip(v) for v in raw.iloc[2, 3:16].tolist()]

    records = []
    for _, row in raw.iloc[3:].iterrows():
        cost_rank = row.iloc[0]
        if pd.isna(cost_rank):
            continue
        product  = _strip(row.iloc[1])
        segment  = _strip(row.iloc[2])
        for i, q in enumerate(quarter_labels):
            records.append({
                "product":     product,
                "cal_quarter": q,
                "segment":     segment,
                "scms_units":  _safe_float(row.iloc[3 + i]),
            })

    df = pd.DataFrame(records)

    # Pivot segment columns
    pivot = df.pivot_table(
        index=["product", "cal_quarter"],
        columns="segment",
        values="scms_units",
        aggfunc="sum",
    ).reset_index()
    pivot.columns.name = None
    pivot.columns = [
        c if c in ("product", "cal_quarter")
        else f"scms_{c.lower().replace(' ', '_').replace('/', '_')}"
        for c in pivot.columns
    ]

    # Total across all segments
    scms_cols = [c for c in pivot.columns if c.startswith("scms_")]
    pivot["scms_total"] = pivot[scms_cols].sum(axis=1)
    return pivot


def load_vms(data_file: str = _DATA_FILE) -> pd.DataFrame:
    """Load the 'VMS' sheet and aggregate by product + quarter.

    Sheet layout (header=None):
        Row 0 : column headers
        Row 1 : partial labels (Vms Top Name in col 2)
        Row 2 : quarter labels (2023Q1 … 2026Q1, 13 quarters) at columns 3-15
        Row 3+: product × vertical data

    Returns a long-format DataFrame with columns:
        product, cal_quarter, vms_<vertical>, vms_total
    """
    raw = pd.read_excel(data_file, sheet_name="VMS", header=None)

    # Quarter labels in row 2, columns 3 through 15 (13 quarters)
    quarter_labels = [_strip(v) for v in raw.iloc[2, 3:16].tolist()]

    records = []
    for _, row in raw.iloc[3:].iterrows():
        cost_rank = row.iloc[0]
        if pd.isna(cost_rank):
            continue
        product  = _strip(row.iloc[1])
        vertical = _strip(row.iloc[2])
        for i, q in enumerate(quarter_labels):
            records.append({
                "product":     product,
                "cal_quarter": q,
                "vertical":    vertical,
                "vms_units":   _safe_float(row.iloc[3 + i]),
            })

    df = pd.DataFrame(records)

    pivot = df.pivot_table(
        index=["product", "cal_quarter"],
        columns="vertical",
        values="vms_units",
        aggfunc="sum",
    ).reset_index()
    pivot.columns.name = None
    pivot.columns = [
        c if c in ("product", "cal_quarter")
        else (
            "vms_"
            + c.lower()
            .replace(" ", "_")
            .replace("/", "_")
            .replace("-", "_")
            .replace("&", "_")
            .replace(",", "")
        )
        for c in pivot.columns
    ]

    vms_cols = [c for c in pivot.columns if c.startswith("vms_")]
    pivot["vms_total"] = pivot[vms_cols].sum(axis=1)
    return pivot


def load_all(data_file: str = _DATA_FILE) -> dict[str, pd.DataFrame]:
    """Load every table and return a dict keyed by table name."""
    return {
        "actuals":   load_actuals(data_file),
        "big_deal":  load_big_deal(data_file),
        "scms":      load_scms(data_file),
        "vms":       load_vms(data_file),
    }


# ─────────────────────────────── helpers ────────────────────────────────────

def _safe_float(v: object) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan
