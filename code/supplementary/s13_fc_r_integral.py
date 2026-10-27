"""
s13_fc_r_integral.py  Pseudo-critical threshold versus robustness integral (Fig. S9)
===================================================================================
Correlates, per scale and for random failure and targeted attack, the
pseudo-critical threshold f_c with the robustness integral R over the networks
with SCGC ratio >= 0.3 and a positive threshold on the respective side, and
plots the L2 scatter. No simulation is run.

Input  data/match_summary_mc100_guarded.csv
Output  figures/figS9.pdf, figures/figS9.png, results/figS9_fc_R_table.csv

Usage
  python code/supplementary/s13_fc_r_integral.py

Last modified 2026-10-27
"""

import sys
from pathlib import Path

import pandas as pd
from scipy import stats

BASE_DIR = Path(__file__).resolve().parent.parent.parent
FIG_DIR = BASE_DIR / "figures"
RESULTS_DIR = BASE_DIR / "results"
# Published analysis (100 Monte Carlo trials at all scales, restricted threshold estimator)
CSV = BASE_DIR / "data" / "match_summary_mc100_guarded.csv"

# Supplementary Section S13, Pearson r(f_c, R) per (scale, side), and the L2 sample sizes
GATE_R = {("L1", "Error"): 0.765, ("L1", "Attack"): 0.826,
          ("L2", "Error"): 0.793, ("L2", "Attack"): 0.855,
          ("L3", "Error"): 0.681, ("L3", "Attack"): 0.756}
GATE_N = {("L2", "Error"): 4330, ("L2", "Attack"): 4368}


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
                         "R_mean": round(v[r_col].mean(), 3),
                         "pearson_r_full": pear})
    t = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t.drop(columns="pearson_r_full").to_csv(RESULTS_DIR / "figS9_fc_R_table.csv", index=False)
    print(t.drop(columns="pearson_r_full").to_string(index=False))
    print("Saved results/figS9_fc_R_table.csv\n")

    # Acceptance check against Supplementary Section S13
    by = t.set_index(["Scale", "Side"])
    failed = False
    for key, exp in GATE_R.items():
        got = by.loc[key, "pearson_r_full"]
        ok = abs(got - exp) <= 0.0005
        failed |= not ok
        print(f"  r(f_c, R) {key[0]} {key[1]:<6} {got:.4f} vs {exp:.3f}  {'PASS' if ok else 'FAIL'}")
    for key, exp in GATE_N.items():
        got = by.loc[key, "n"]
        ok = got == exp
        failed |= not ok
        print(f"  N         {key[0]} {key[1]:<6} {got} vs {exp}  {'PASS' if ok else 'FAIL'}")
    if failed:
        print("GATE FAILED  computed values differ from Supplementary Section S13")
        sys.exit(1)
    print("GATE PASS  all values match Supplementary Section S13\n")

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
        ax.set_xlabel(rf"$f_{{c,\mathrm{{L2}}}}^{{\mathrm{{{side}}}}}$")
        ax.set_ylabel(rf"$R^{{\mathrm{{{side}}}}}$")
        ax.set_title(f"{side}: r = {r:.3f} ($N$ = {len(v):,})")
    fig.suptitle(r"Threshold $f_{c,\mathrm{L2}}$ vs robustness integral $R$ "
                 "(100-trial protocol)", y=1.02)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "figS9.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIG_DIR / "figS9.pdf", bbox_inches="tight")
    print("Saved figures/figS9.pdf, figures/figS9.png")


if __name__ == "__main__":
    main()
