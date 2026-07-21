"""
s11_degree_distributions.py — Degree distributions vs synthetic models (Supplementary S11)
==========================================================================================
Plots the in- and out-degree complementary cumulative distribution functions
(CCDF) of the pooled empirical L2 networks (200 sampled networks) against the
four calibrated synthetic model families (ER, BA, WS, RGG; 100 realisations
each), from the archived degree data. Pure re-plot: no simulation is re-run.

Input:  ../../data/supplementary/degrees_empirical.csv,
        degrees_synthetic.csv, ks_distances.csv (all read-only)
Output: fig_degree_distribution.pdf / .png (written next to this script)

Acceptance gate: re-reads ks_distances.csv and checks the BA Kolmogorov-
Smirnov distances KS_in = 0.086 / KS_out = 0.087 quoted in the supplement
(exits on mismatch).

Usage:
    cd code/supplementary
    python s11_degree_distributions.py

Last modified: 2026-07-21
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times"],
                     "mathtext.fontset": "cm", "pdf.fonttype": 42,
                     "axes.unicode_minus": False})

_SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = _SCRIPT_DIR.parent.parent / "data" / "supplementary"
MODEL_TYPES = ["ER", "BA", "WS", "RGG"]


def ccdf(values):
    v = np.sort(np.asarray(values))
    n = len(v)
    x = np.unique(v)
    y = 1.0 - np.searchsorted(v, x, side="left") / n
    return x, y


def main():
    emp_deg = pd.read_csv(DATA_DIR / "degrees_empirical.csv")
    syn_deg = pd.read_csv(DATA_DIR / "degrees_synthetic.csv")
    ks = pd.read_csv(DATA_DIR / "ks_distances.csv").set_index("Model")
    assert abs(ks.loc["BA", "KS_in"] - 0.086) < 1e-9, ks.loc["BA", "KS_in"]
    assert abs(ks.loc["BA", "KS_out"] - 0.087) < 1e-9, ks.loc["BA", "KS_out"]
    print(f"GATE PASS: BA KS_in={ks.loc['BA', 'KS_in']}, KS_out={ks.loc['BA', 'KS_out']}, "
          f"empirical rows={len(emp_deg)}, synthetic rows={len(syn_deg)}")

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
    fig.savefig(_SCRIPT_DIR / "fig_degree_distribution.png", dpi=150, bbox_inches="tight")
    fig.savefig(_SCRIPT_DIR / "fig_degree_distribution.pdf", bbox_inches="tight")
    print("fig_degree_distribution.pdf/.png saved")


if __name__ == "__main__":
    main()
