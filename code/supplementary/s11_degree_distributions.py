"""
s11_degree_distributions.py  Degree distributions of empirical and synthetic networks (Fig. S7)
==============================================================================================
Plots the in- and out-degree complementary cumulative distribution functions
(CCDF) of 200 pooled empirical L2 networks against the four calibrated
synthetic models (ER, BA, WS, RGG, 100 realisations each) from the shipped
degree data, and checks the degree statistics quoted in Section S11.
No simulation is run.

Input  data/supplementary/degrees_empirical.csv, degrees_synthetic.csv, ks_distances.csv
Output  figures/figS7.pdf, figures/figS7.png

Usage
  python code/supplementary/s11_degree_distributions.py

Last modified 2026-10-27
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times"],
                     "mathtext.fontset": "cm", "pdf.fonttype": 42,
                     "axes.unicode_minus": False})

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "supplementary"
FIG_DIR = BASE_DIR / "figures"
MODEL_TYPES = ["ER", "BA", "WS", "RGG"]

# Supplementary Section S11. Two-sample KS distances (in-degree, out-degree),
# coefficients of variation, and the largest degree in the pooled samples.
KS_S11 = {"ER": (0.156, 0.161), "BA": (0.086, 0.087),
          "WS": (0.235, 0.238), "RGG": (0.419, 0.423)}
CV_EMPIRICAL = (0.97, 0.98)               # in-degree, out-degree
CV_MODEL = {"ER": 0.63, "WS": 0.47}       # in- and out-degree pooled
MAX_DEGREE = {"Empirical": 20, "BA": 31}


def ccdf(values):
    v = np.sort(np.asarray(values))
    n = len(v)
    x = np.unique(v)
    y = 1.0 - np.searchsorted(v, x, side="left") / n
    return x, y


def cv(v):
    v = np.asarray(v, float)
    return v.std() / v.mean()


def acceptance_check(emp_deg, syn_deg, ks):
    checks = []
    for model, (exp_in, exp_out) in KS_S11.items():
        syn = syn_deg[syn_deg.Model == model]
        for col, exp in [("k_in", exp_in), ("k_out", exp_out)]:
            got = ks_2samp(emp_deg[col], syn[col]).statistic
            checks.append((f"KS {model} {col} from degree data", got, exp, 0.0005))
            checks.append((f"KS {model} {col} in ks_distances.csv",
                           ks.loc[model, col.replace("k_", "KS_")], exp, 0.0005))
    checks.append(("CV empirical k_in", cv(emp_deg.k_in), CV_EMPIRICAL[0], 0.005))
    checks.append(("CV empirical k_out", cv(emp_deg.k_out), CV_EMPIRICAL[1], 0.005))
    for model, exp in CV_MODEL.items():
        syn = syn_deg[syn_deg.Model == model]
        checks.append((f"CV {model} k_in and k_out pooled",
                       cv(np.r_[syn.k_in, syn.k_out]), exp, 0.005))
    checks.append(("largest empirical degree",
                   max(emp_deg.k_in.max(), emp_deg.k_out.max()), MAX_DEGREE["Empirical"], 0))
    ba = syn_deg[syn_deg.Model == "BA"]
    checks.append(("largest BA degree", max(ba.k_in.max(), ba.k_out.max()), MAX_DEGREE["BA"], 0))
    failed = False
    for label, got, exp, tol in checks:
        ok = abs(got - exp) <= tol + 1e-12
        failed |= not ok
        print(f"  {label:<36} {got:8.4f} vs {exp:6.3f}  {'PASS' if ok else 'FAIL'}")
    if failed:
        print("GATE FAILED  computed values differ from Supplementary Section S11")
        sys.exit(1)
    print(f"GATE PASS  all {len(checks)} values match Supplementary Section S11 "
          f"(empirical rows {len(emp_deg)}, synthetic rows {len(syn_deg)})")


def main():
    emp_deg = pd.read_csv(DATA_DIR / "degrees_empirical.csv")
    syn_deg = pd.read_csv(DATA_DIR / "degrees_synthetic.csv")
    ks = pd.read_csv(DATA_DIR / "ks_distances.csv").set_index("Model")
    acceptance_check(emp_deg, syn_deg, ks)

    colors = {"Empirical": "#000000", "ER": "#4daf4a", "BA": "#e41a1c",
              "WS": "#984ea3", "RGG": "#ff7f00"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, col, lab in [(axes[0], "k_in", "In-degree"),
                         (axes[1], "k_out", "Out-degree")]:
        x, y = ccdf(emp_deg[col])
        ax.loglog(x[x > 0], y[x > 0], "o-", ms=4, lw=1.8,
                  color=colors["Empirical"], label="Empirical L2", zorder=5)
        for model in MODEL_TYPES:
            xs, ys = ccdf(syn_deg[syn_deg.Model == model][col])
            ax.loglog(xs[xs > 0], ys[xs > 0], "s--", ms=3, lw=1.2,
                      color=colors[model], label=model, alpha=0.85)
        ax.set_xlabel(f"{lab} $k$")
        ax.set_ylabel(r"CCDF  $P(K \geq k)$")
        ax.set_title(lab)
        ax.legend(fontsize=8)
    fig.suptitle("Degree distributions: empirical L2 (200-network pooled) vs "
                 "calibrated synthetic models (100 realisations each)", y=1.02)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "figS7.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIG_DIR / "figS7.pdf", bbox_inches="tight")
    print("Saved figures/figS7.pdf, figures/figS7.png")


if __name__ == "__main__":
    main()
