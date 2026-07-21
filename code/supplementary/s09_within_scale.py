"""
s09_within_scale.py — Within-scale bias vs network size (Supplementary S9)
==========================================================================
Bins the theory-minus-simulation bias of the percolation threshold by network
size N within each spatial scale (resolution held fixed), addressing the
finite-size question. Valid set per scale: rows with a defined theoretical
threshold (fc theory > 0); bias = theoretical threshold - simulated
random-removal threshold (revised 100-trial protocol values). Panels show
bin means with 95% CI against the bin-median node count, with the
within-scale Spearman correlation annotated. Pure post-processing: no
simulation is re-run.

Input:  ../../data/match_summary_mc100_guarded.csv (read-only)
Output: finite_size_bias.csv + fig_finite_size_bias.png/.pdf
        (written next to this script)

Usage:
    cd code/supplementary
    python s09_within_scale.py

Last modified: 2026-07-21
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps

_SCRIPT_DIR = Path(__file__).resolve().parent
GD = pd.read_csv(_SCRIPT_DIR.parent.parent / "data" / "match_summary_mc100_guarded.csv")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 12, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300, "savefig.bbox": "tight", "figure.facecolor": "white",
    "savefig.facecolor": "white", "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
})
BLUE = "#1B4F72"

rows = []
# Shared y axis: the cross-scale ordering (largest deviation at L1, smallest
# at L3) is the main argument; independently scaled panels would render the
# L2 +/-0.06 range as a steep trend and hide the L1 >> L2/L3 magnitude contrast
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)
for ax, scale in zip(axes, ["L1", "L2", "L3"]):
    d = GD[(GD.Scale == scale) & (GD.pc_Theory > 0)].copy()
    d["bias"] = d.pc_Theory - d.pc_Error_Strong
    rho, pval = sps.spearmanr(d.Nodes, d.bias)
    q = pd.qcut(d.Nodes, 6, duplicates="drop")
    g = d.groupby(q, observed=True).agg(
        n=("bias", "size"), Nmed=("Nodes", "median"),
        mean=("bias", "mean"), sd=("bias", "std"))
    g["ci95"] = 1.96 * g["sd"] / np.sqrt(g["n"])
    for binlab, r in g.iterrows():
        rows.append(dict(scale=scale, bin=str(binlab), n=int(r.n), N_median=r.Nmed,
                         bias_mean=r["mean"], bias_ci95=r.ci95,
                         rho_scale=rho, p_scale=pval, n_scale=len(d)))
    ax.axhline(0, color="#999999", lw=0.9, ls="--", zorder=1)
    ax.errorbar(g.Nmed, g["mean"], yerr=g.ci95, fmt="o-", color=BLUE,
                lw=1.6, ms=5.5, capsize=3, zorder=3)
    ax.set_title(f"{scale}  ($N$ = {len(d):,})")
    ax.set_xlabel("Network size $N$ (bin median)")
    p_str = "$P$ < 0.001" if pval < 1e-3 else f"$P$ = {pval:.3f}"
    ax.text(0.04, 0.96, f"$\\rho_s$ = {rho:+.2f}\n{p_str}",
            transform=ax.transAxes, va="top", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      alpha=0.85, edgecolor="none"))
axes[0].set_ylabel(r"Bias $= f_c^{\mathrm{dMR}} - f_c^{\mathrm{Error}}$")
plt.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(_SCRIPT_DIR / f"fig_finite_size_bias.{ext}")
plt.close(fig)

df = pd.DataFrame(rows)
df.to_csv(_SCRIPT_DIR / "finite_size_bias.csv", index=False)
print(df.groupby("scale").agg(n=("n", "sum"), bias_min=("bias_mean", "min"),
                              bias_max=("bias_mean", "max"),
                              rho=("rho_scale", "first")).round(3).to_string())
print("Saved: finite_size_bias.csv + fig_finite_size_bias.png/pdf")
