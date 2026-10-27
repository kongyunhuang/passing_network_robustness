"""
s09_within_scale.py  Within-scale network-size check of the dMR bias (Fig. S5)
=============================================================================
Bins the signed bias of the directed MR benchmark (benchmark minus simulated
random-failure threshold) into six equal-count node-count bins within each of
L1, L2 and L3, using the networks with a positive benchmark threshold (the
Table 3 sample). Panels show bin means with 95% confidence intervals against
the bin-median node count, with the within-scale Spearman correlation.
No simulation is run.

Input  data/match_summary_mc100_guarded.csv
Output  figures/figS5.pdf, figures/figS5.png, results/figS5_finite_size_bias.csv

Usage
  python code/supplementary/s09_within_scale.py

Last modified 2026-10-27
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps

BASE_DIR = Path(__file__).resolve().parent.parent.parent
FIG_DIR = BASE_DIR / "figures"
RESULTS_DIR = BASE_DIR / "results"
# Published analysis (100 Monte Carlo trials at all scales, restricted threshold estimator)
GD = pd.read_csv(BASE_DIR / "data" / "match_summary_mc100_guarded.csv")

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
# Shared y axis, so that the magnitude contrast between L1 and L2/L3 stays visible
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
axes[0].set_ylabel(r"Bias $= f_{c,\ell}^{\mathrm{dMR}} - f_{c,\ell}^{\mathrm{Error}}$")
plt.tight_layout()
FIG_DIR.mkdir(parents=True, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(FIG_DIR / f"figS5.{ext}")
plt.close(fig)

df = pd.DataFrame(rows)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
df.to_csv(RESULTS_DIR / "figS5_finite_size_bias.csv", index=False)
print(df.groupby("scale").agg(n=("n", "sum"), bias_min=("bias_mean", "min"),
                              bias_max=("bias_mean", "max"),
                              rho=("rho_scale", "first")).round(3).to_string())
print("Saved figures/figS5.pdf, figures/figS5.png, results/figS5_finite_size_bias.csv\n")

# Acceptance check against Supplementary Section S9 and the Fig. S5 caption
by = df.groupby("scale")
first_bin = by.bias_mean.first()     # bins are stored from smallest to largest size
last_bin = by.bias_mean.last()
checks = [
    ("L1 networks", by.n.sum()["L1"], 1305, 0),
    ("L2 networks", by.n.sum()["L2"], 4325, 0),
    ("L3 networks", by.n.sum()["L3"], 4554, 0),
    ("L1 most negative bin mean", by.bias_mean.min()["L1"], -0.20, 0.005),
    ("L1 least negative bin mean", by.bias_mean.max()["L1"], -0.13, 0.005),
    ("L2 smallest-size bin mean", first_bin["L2"], -0.063, 0.0005),
    ("L2 largest-size bin mean", last_bin["L2"], +0.035, 0.0005),
    ("L2 Spearman rho", by.rho_scale.first()["L2"], 0.363, 0.0005),
]
failed = False
for label, got, exp, tol in checks:
    ok = abs(got - exp) <= tol + 1e-12
    failed |= not ok
    print(f"  {label:<28} {got:+10.4f} vs {exp:+.3f}  {'PASS' if ok else 'FAIL'}")
for scale, bound in [("L2", 0.07), ("L3", 0.03)]:
    got = by.bias_mean.apply(lambda s: s.abs().max())[scale]
    ok = got <= bound
    failed |= not ok
    print(f"  {scale + ' max |bin mean|':<28} {got:+10.4f} <= {bound:.2f}  {'PASS' if ok else 'FAIL'}")
if failed:
    print("GATE FAILED  computed values differ from Supplementary Section S9")
    sys.exit(1)
print("GATE PASS  all values match Supplementary Section S9")
