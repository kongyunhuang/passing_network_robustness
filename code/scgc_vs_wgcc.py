# -*- coding: utf-8 -*-
"""
scgc_vs_wgcc.py — Comparison of SCGC and WGCC order parameters (S6)
====================================================================
Compare SCGC (strong) and WGCC (weak) percolation order parameters.
Report mean +/- SD of percolation thresholds, zero-SCGC/WGCC counts,
and Pearson correlation between SCGC- and WGCC-based thresholds.

Input:  data/match_summary.csv
Output: Prints Table S5 to console.

Usage:
    python code/scgc_vs_wgcc.py

Last modified: 2026-03-27
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats

# ── Paths ─────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INPUT_CSV = DATA_DIR / "match_summary.csv"


def main():
    if not INPUT_CSV.exists():
        print(f"ERROR: {INPUT_CSV} not found.", file=sys.stderr)
        sys.exit(1)

    df_all = pd.read_csv(INPUT_CSV)
    df = df_all[df_all["Scale"] == "L2"].copy()
    N = len(df)
    print(f"Total L2 matches: N = {N}")

    # ── Summary statistics ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("Table S5 — Comparison of SCGC and WGCC order parameters (L2)")
    print("=" * 70)

    cols = {
        "fc_Error_SCGC":  "fc_Error_Strong",
        "fc_Error_WGCC":  "fc_Error_Weak",
        "fc_Attack_SCGC": "fc_Attack_Strong",
        "fc_Attack_WGCC": "fc_Attack_Weak",
    }

    print(f"\n{'Measure':<20} {'Mean':>8} {'SD':>8} {'Zero count':>12} {'Zero %':>8}")
    print("-" * 60)
    for label, col in cols.items():
        mean = df[col].mean()
        sd = df[col].std()
        n_zero = (df[col] == 0).sum()
        pct_zero = n_zero / N * 100
        print(f"{label:<20} {mean:>8.4f} {sd:>8.4f} {n_zero:>12d} {pct_zero:>7.1f}%")

    # ── Correlations between SCGC and WGCC ───────────────────────
    print("\n" + "-" * 60)
    print("Pearson correlation between SCGC and WGCC thresholds")
    print("-" * 60)

    # Attack: filter rows where both > 0
    mask_atk = (df["fc_Attack_Strong"] > 0) & (df["fc_Attack_Weak"] > 0)
    sub_atk = df.loc[mask_atk]
    r_atk, p_atk = scipy.stats.pearsonr(
        sub_atk["fc_Attack_Strong"], sub_atk["fc_Attack_Weak"]
    )
    print(f"Attack:  r = {r_atk:.3f}, p = {p_atk:.2e}, N = {len(sub_atk)}")

    # Error: filter rows where both > 0
    mask_err = (df["fc_Error_Strong"] > 0) & (df["fc_Error_Weak"] > 0)
    sub_err = df.loc[mask_err]
    r_err, p_err = scipy.stats.pearsonr(
        sub_err["fc_Error_Strong"], sub_err["fc_Error_Weak"]
    )
    print(f"Error:   r = {r_err:.3f}, p = {p_err:.2e}, N = {len(sub_err)}")


if __name__ == "__main__":
    main()
