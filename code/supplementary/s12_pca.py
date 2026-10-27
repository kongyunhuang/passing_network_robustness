"""
s12_pca.py  Principal component analysis of the eight structural metrics (Fig. S8)
=================================================================================
Principal component analysis of the eight standardised directed structural
metrics of Table 1 on the primary L2 sample (N = 4,330, SCGC ratio >= 0.3 and
a positive random-failure threshold). Reports variance explained, loadings and
the correlation of each component with the targeted-attack threshold. The
acceptance check first reproduces the zero-order and partial correlations of
Section 3.2 of the main text, then the PCA values of Section S12.

Input  data/match_summary_mc100_guarded.csv
Output  figures/figS8.pdf, figures/figS8.png,
        results/figS8_pca_variance.csv, results/figS8_pca_loadings.csv, results/figS8_pc_vs_fc.csv

Usage
  python code/supplementary/s12_pca.py

Last modified 2026-10-27
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
FIG_DIR = BASE_DIR / "figures"
RESULTS_DIR = BASE_DIR / "results"
# Published analysis (100 Monte Carlo trials at all scales, restricted threshold estimator)
MATCH_SUMMARY = BASE_DIR / "data" / "match_summary_mc100_guarded.csv"

METRICS = {
    "AvgDegree": "Mean degree $\\langle k \\rangle$",
    "Clustering": "Clustering $C$",
    "Assortativity": "Assortativity $r$",
    "Reciprocity": "Reciprocity $\\rho$",
    "FlowHierarchy": "Flow hierarchy $h$",
    "OutDegreeCV": "Out-degree CV",
    "Mu11_Mu": "MR $\\mu_{11}/\\mu$",
    "CyclicTriadFraction": "Cyclic triads $f_{cyc}$",
}
FC_COL = "pc_Attack_Strong"

# Published values (main-text Section 3.2 and Fig. 5)
GATE_ZERO = {"AvgDegree": 0.785, "Mu11_Mu": 0.782, "FlowHierarchy": -0.766,
             "Clustering": 0.667, "Reciprocity": 0.627,
             "CyclicTriadFraction": 0.579, "OutDegreeCV": 0.237,
             "Assortativity": 0.094}
GATE_PARTIAL = {"FlowHierarchy": -0.311, "Mu11_Mu": 0.104,
                "AvgDegree": 0.031, "Clustering": -0.040, "Reciprocity": -0.080,
                "Assortativity": 0.041, "OutDegreeCV": 0.037, "CyclicTriadFraction": -0.028}
# Supplementary Section S12, variance explained in percent (PC1, PC2, PC3, PC8),
# correlation of PC1 with the attack threshold, and the bound on |r| for PC2 and PC3
GATE_VARIANCE = {"PC1": 62.8, "PC2": 14.3, "PC3": 10.5, "PC8": 0.1}
GATE_PC1_R = 0.777
GATE_PC23_ABS_R = 0.08


def partial_corr(df, x, y, covars):
    """Partial correlation (linear residual method), corr(resid(x|covars), resid(y|covars))."""
    Z = np.column_stack([np.ones(len(df))] + [df[c].to_numpy(float) for c in covars])
    def resid(col):
        v = df[col].to_numpy(float)
        beta, *_ = np.linalg.lstsq(Z, v, rcond=None)
        return v - Z @ beta
    rx, ry = resid(x), resid(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    df = pd.read_csv(MATCH_SUMMARY)
    l2 = df[(df.Scale == "L2") & (df.SCGC_Ratio >= 0.3) & (df.pc_Error_Strong > 0)].copy()
    n = len(l2)
    print(f"L2 sample  n = {n} (expected 4,330)")
    if n != 4330:
        print("GATE FAILED  sample size differs from the published 4,330")
        sys.exit(1)

    cols = list(METRICS.keys())
    l2 = l2.dropna(subset=cols + [FC_COL])
    print(f"after dropping missing values  n = {len(l2)}")

    # ---------- Acceptance check, main-text correlations ----------
    print("\nAcceptance check against the Section 3.2 correlation coefficients")
    ok_all = True
    for c in cols:
        got = round(float(l2[c].corr(l2[FC_COL])), 3)
        exp = GATE_ZERO[c]
        ok = got == exp
        ok_all &= ok
        print(f"  zero-order {c:<22} {got:+.3f} vs {exp:+.3f}  {'PASS' if ok else 'FAIL'}")
    for c in cols:
        others = [x for x in cols if x != c]
        got = round(partial_corr(l2, c, FC_COL, others), 3)
        exp = GATE_PARTIAL[c]
        ok = got == exp
        ok_all &= ok
        print(f"  partial    {c:<22} {got:+.3f} vs {exp:+.3f}  {'PASS' if ok else 'FAIL'}")
    if not ok_all:
        print("GATE FAILED  stopping")
        sys.exit(1)
    print("GATE PASS  main-text correlations\n")

    # ---------- PCA ----------
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    X = StandardScaler().fit_transform(l2[cols].to_numpy(float))
    pca = PCA()
    scores = pca.fit_transform(X)

    var = pd.DataFrame({
        "PC": [f"PC{i+1}" for i in range(len(cols))],
        "var_explained": np.round(pca.explained_variance_ratio_, 4),
        "cumulative": np.round(np.cumsum(pca.explained_variance_ratio_), 4),
    })
    load = pd.DataFrame(pca.components_.T, index=cols,
                        columns=[f"PC{i+1}" for i in range(len(cols))]).round(3)

    # Correlation of each principal component with the attack threshold
    fc = l2[FC_COL].to_numpy(float)
    r_pc = [float(np.corrcoef(scores[:, i], fc)[0, 1]) for i in range(len(cols))]
    pc_fc = pd.DataFrame({
        "PC": [f"PC{i+1}" for i in range(len(cols))],
        "r_with_fcAttack": np.round(r_pc, 3),
        "var_explained": np.round(pca.explained_variance_ratio_, 3),
    })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    var.to_csv(RESULTS_DIR / "figS8_pca_variance.csv", index=False)
    load.to_csv(RESULTS_DIR / "figS8_pca_loadings.csv")
    pc_fc.to_csv(RESULTS_DIR / "figS8_pc_vs_fc.csv", index=False)

    print("Variance explained")
    print(var.to_string(index=False))
    print("\nLoadings (first 3 PCs)")
    print(load[["PC1", "PC2", "PC3"]].to_string())
    print("\nCorrelation of each PC with the attack threshold")
    print(pc_fc.to_string(index=False))

    # ---------- Acceptance check, Section S12 ----------
    print("\nAcceptance check against Supplementary Section S12")
    failed = False
    for pc, exp in GATE_VARIANCE.items():
        got = 100 * pca.explained_variance_ratio_[int(pc[2:]) - 1]
        ok = abs(got - exp) <= 0.05
        failed |= not ok
        print(f"  variance explained {pc} (%)   {got:7.3f} vs {exp:5.1f}  {'PASS' if ok else 'FAIL'}")
    ok = abs(r_pc[0] - GATE_PC1_R) <= 0.0005
    failed |= not ok
    print(f"  r(PC1, attack threshold)      {r_pc[0]:+7.4f} vs {GATE_PC1_R:+.3f}  {'PASS' if ok else 'FAIL'}")
    for i in (1, 2):
        ok = abs(r_pc[i]) <= GATE_PC23_ABS_R + 0.005
        failed |= not ok
        print(f"  |r(PC{i+1}, attack threshold)|    {abs(r_pc[i]):7.4f} <= {GATE_PC23_ABS_R:.2f}  "
              f"{'PASS' if ok else 'FAIL'}")
    if failed:
        print("GATE FAILED  PCA values differ from Supplementary Section S12")
        sys.exit(1)
    print("GATE PASS  Section S12 values")

    # ---------- Figure, scree plot and loadings heatmap ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times"],
                         "mathtext.fontset": "cm", "pdf.fonttype": 42,
                         "axes.unicode_minus": False})

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6),
                             gridspec_kw={"width_ratios": [1, 1.5]})
    ax = axes[0]
    xs = np.arange(1, len(cols) + 1)
    ax.bar(xs, pca.explained_variance_ratio_, color="#2166ac", alpha=0.85)
    ax.plot(xs, np.cumsum(pca.explained_variance_ratio_), "o-", color="#b2182b",
            label="cumulative")
    ax.axhline(0.8, ls="--", lw=0.8, color="grey")
    ax.axhline(0.9, ls=":", lw=0.8, color="grey")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Variance explained")
    ax.set_title("Scree plot")
    ax.legend(fontsize=8)

    ax = axes[1]
    im = ax.imshow(load.to_numpy(), cmap="RdBu_r", vmin=-0.7, vmax=0.7, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f"PC{i+1}" for i in range(len(cols))], fontsize=8)
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels([METRICS[c] for c in cols], fontsize=8)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = load.iloc[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(v) > 0.45 else "black")
    ax.set_title("Loadings")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f"PCA of eight directed structural metrics (L2, N = {len(l2):,})", y=1.02)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "figS8.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIG_DIR / "figS8.pdf", bbox_inches="tight")
    print("\nSaved figures/figS8.pdf, figures/figS8.png")


if __name__ == "__main__":
    main()
