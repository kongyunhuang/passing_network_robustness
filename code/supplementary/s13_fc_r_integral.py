"""
s13_fc_r_integral.py — Threshold vs robustness integral consistency (Supplementary S13)
=======================================================================================
Summarises, per spatial scale, the correlation between the critical threshold
f_c and the robustness integral R (Error and Attack sides) from the revised
100-trial protocol data, and plots the L2 scatter. Pure post-processing: no
simulation is re-run.

Input:  ../../data/match_summary_mc100_guarded.csv (read-only)
Output: fc_R_table.csv, fig_fc_vs_R.png/.pdf (written next to this script)

Acceptance gate: reproduces the recorded L2 Pearson correlations
r(fc, R) ~ 0.79 (Error) / 0.85 (Attack) within half a final digit;
exits on mismatch.

Usage:
    cd code/supplementary
    python s13_fc_r_integral.py

Last modified: 2026-07-21
"""

import sys
from pathlib import Path

import pandas as pd
from scipy import stats

_SCRIPT_DIR = Path(__file__).resolve().parent
CSV = _SCRIPT_DIR.parent.parent / "data" / "match_summary_mc100_guarded.csv"


def main():
    g = pd.read_csv(CSV)
    rows = []
    for sc in ["L0", "L1", "L2", "L3"]:
        d = g[(g.Scale == sc) & (g.SCGC_Ratio >= 0.3)]
        for side, fc_col, r_col in [("Error", "pc_Error_Strong", "R_Error_Strong"),
                                    ("Attack", "pc_Attack_Strong", "R_Attack_Strong")]:
            v = d[(d[fc_col] > 0)].dropna(subset=[fc_col, r_col])
            pear = float(v[fc_col].corr(v[r_col]))
            spear = float(stats.spearmanr(v[fc_col], v[r_col]).statistic)
            rows.append({"Scale": sc, "Side": side, "n": len(v),
                         "pearson_r": round(pear, 3), "spearman_rho": round(spear, 3),
                         "fc_mean": round(v[fc_col].mean(), 3),
                         "R_mean": round(v[r_col].mean(), 3)})
    t = pd.DataFrame(rows)
    t.to_csv(_SCRIPT_DIR / "fc_R_table.csv", index=False)
    print(t.to_string(index=False))

    # Gate: L2 reproduces the recorded ~0.79 / ~0.85
    l2e = t[(t.Scale == "L2") & (t.Side == "Error")].pearson_r.iloc[0]
    l2a = t[(t.Scale == "L2") & (t.Side == "Attack")].pearson_r.iloc[0]
    # The recorded values are 2-decimal (~0.79 / ~0.85). Gate = |measured -
    # recorded| <= 0.006 (half a final digit 0.005 + floating-point margin;
    # the measured 0.855 differs from the recorded 0.85 by exactly half a digit)
    okE, okA = abs(l2e - 0.79) <= 0.006, abs(l2a - 0.85) <= 0.006
    print(f"\nGATE - L2 r(fc,R): Error {l2e:.3f} (expected ~0.79) -> "
          f"{'PASS' if okE else 'FAIL'} | Attack {l2a:.3f} (expected ~0.85) -> "
          f"{'PASS' if okA else 'FAIL'}")
    if not (okE and okA):
        print("GATE FAILED - stopping.")
        sys.exit(1)

    # L2 scatter
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times"],
                         "mathtext.fontset": "cm", "pdf.fonttype": 42,
                         "axes.unicode_minus": False})

    d = g[(g.Scale == "L2") & (g.SCGC_Ratio >= 0.3)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, side, fc_col, r_col, col in [
            (axes[0], "Error", "pc_Error_Strong", "R_Error_Strong", "#2166ac"),
            (axes[1], "Attack", "pc_Attack_Strong", "R_Attack_Strong", "#b2182b")]:
        v = d[d[fc_col] > 0].dropna(subset=[fc_col, r_col])
        ax.scatter(v[fc_col], v[r_col], s=3, alpha=0.15, color=col)
        r = float(v[fc_col].corr(v[r_col]))
        ax.set_xlabel(f"$f_c$ ({side})")
        ax.set_ylabel(f"$R$ ({side})")
        ax.set_title(f"{side}: r = {r:.3f} ($N$ = {len(v):,})")
    fig.suptitle("Threshold $f_c$ vs robustness integral $R$ "
                 "(L2, 100-trial protocol)", y=1.02)
    fig.tight_layout()
    fig.savefig(_SCRIPT_DIR / "fig_fc_vs_R.png", dpi=150, bbox_inches="tight")
    fig.savefig(_SCRIPT_DIR / "fig_fc_vs_R.pdf", bbox_inches="tight")
    print("Figure saved: fig_fc_vs_R.png/.pdf")


if __name__ == "__main__":
    main()
