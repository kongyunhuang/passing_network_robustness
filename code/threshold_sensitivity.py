# -*- coding: utf-8 -*-
"""
threshold_sensitivity.py — SCGC filter threshold sensitivity analysis (S5)
==========================================================================
Test whether the 30% SCGC_Ratio threshold choice affects conclusions.
For each threshold in [0, 0.10, 0.20, 0.30, 0.40, 0.50], compute
bivariate Pearson r and partial correlations between 8 structural
metrics and the attack percolation threshold (fc_Attack_Strong).

Input:  data/match_summary.csv
Output: Prints Table S4a (bivariate Pearson r) and Table S4b (partial
        correlation) to console.

Usage:
    python code/threshold_sensitivity.py

Last modified: 2026-03-27
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats
from sklearn.linear_model import LinearRegression

# ── Paths ─────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INPUT_CSV = DATA_DIR / "match_summary.csv"

# ── Metric definitions ───────────────────────────────────────────
METRICS = [
    "AvgDegree",
    "Clustering",
    "Assortativity",
    "Reciprocity",
    "FlowHierarchy",
    "OutDegreeCV",
    "Mu11_Mu",
    "CyclicTriadFraction",
]

DISPLAY_LABELS = {
    "AvgDegree":           "⟨k⟩",
    "Clustering":          "C",
    "Assortativity":       "r",
    "Reciprocity":         "ρ↔",
    "FlowHierarchy":       "h",
    "OutDegreeCV":         "CV_out",
    "Mu11_Mu":             "μ₁₁/μ",
    "CyclicTriadFraction": "f_cyc",
}

TARGET = "fc_Attack_Strong"
THRESHOLDS = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50]


# ── Partial correlation via residualisation ──────────────────────
def partial_corr(df, x_col, y_col, control_cols):
    """Partial correlation via residualisation."""
    X_ctrl = df[control_cols].values
    # Regress x on controls
    reg_x = LinearRegression().fit(X_ctrl, df[x_col].values)
    res_x = df[x_col].values - reg_x.predict(X_ctrl)
    # Regress y on controls
    reg_y = LinearRegression().fit(X_ctrl, df[y_col].values)
    res_y = df[y_col].values - reg_y.predict(X_ctrl)
    r, p = scipy.stats.pearsonr(res_x, res_y)
    return r, p


# ── Main ─────────────────────────────────────────────────────────
def main():
    if not INPUT_CSV.exists():
        print(f"ERROR: {INPUT_CSV} not found.", file=sys.stderr)
        sys.exit(1)

    df_all = pd.read_csv(INPUT_CSV)
    df_l2 = df_all[df_all["Scale"] == "L2"].copy()
    print(f"Total L2 matches: {len(df_l2)}")

    # ── Table S4a: bivariate Pearson r ───────────────────────────
    print("\n" + "=" * 90)
    print("Table S4a — Bivariate Pearson r between structural metrics and fc_Attack_Strong")
    print("=" * 90)

    header = f"{'Metric':<10}"
    for th in THRESHOLDS:
        label = f"≥{th:.0%}"
        header += f"  {label:>14}"
    print(header)
    print("-" * 90)

    bivariate_results = {}
    for th in THRESHOLDS:
        mask = (df_l2["SCGC_Ratio"] >= th) & (df_l2["fc_Error_Strong"] > 0)
        sub = df_l2.loc[mask, METRICS + [TARGET]].dropna()
        bivariate_results[th] = {"N": len(sub), "r": {}, "p": {}}
        for m in METRICS:
            r, p = scipy.stats.pearsonr(sub[m], sub[TARGET])
            bivariate_results[th]["r"][m] = r
            bivariate_results[th]["p"][m] = p

    # Print N row
    n_row = f"{'N':<10}"
    for th in THRESHOLDS:
        n_row += f"  {bivariate_results[th]['N']:>14d}"
    print(n_row)
    print("-" * 90)

    for m in METRICS:
        label = DISPLAY_LABELS[m]
        row = f"{label:<10}"
        for th in THRESHOLDS:
            r = bivariate_results[th]["r"][m]
            p = bivariate_results[th]["p"][m]
            stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
            cell = f"{r:+.3f}{stars}"
            row += f"  {cell:>14}"
        print(row)

    print("\n* p<0.05, ** p<0.01, *** p<0.001")

    # ── Table S4b: partial correlations ──────────────────────────
    print("\n" + "=" * 90)
    print("Table S4b — Partial correlation (controlling for remaining 7 metrics)")
    print("=" * 90)

    header = f"{'Metric':<10}"
    for th in THRESHOLDS:
        label = f"≥{th:.0%}"
        header += f"  {label:>14}"
    print(header)
    print("-" * 90)

    partial_results = {}
    for th in THRESHOLDS:
        mask = (df_l2["SCGC_Ratio"] >= th) & (df_l2["fc_Error_Strong"] > 0)
        sub = df_l2.loc[mask, METRICS + [TARGET]].dropna()
        partial_results[th] = {"N": len(sub), "r": {}, "p": {}}
        for m in METRICS:
            controls = [x for x in METRICS if x != m]
            r, p = partial_corr(sub, m, TARGET, controls)
            partial_results[th]["r"][m] = r
            partial_results[th]["p"][m] = p

    # Print N row
    n_row = f"{'N':<10}"
    for th in THRESHOLDS:
        n_row += f"  {partial_results[th]['N']:>14d}"
    print(n_row)
    print("-" * 90)

    for m in METRICS:
        label = DISPLAY_LABELS[m]
        row = f"{label:<10}"
        for th in THRESHOLDS:
            r = partial_results[th]["r"][m]
            p = partial_results[th]["p"][m]
            stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
            cell = f"{r:+.3f}{stars}"
            row += f"  {cell:>14}"
        print(row)

    print("\n* p<0.05, ** p<0.01, *** p<0.001")


if __name__ == "__main__":
    main()
