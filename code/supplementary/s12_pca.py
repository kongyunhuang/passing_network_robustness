"""
s12_pca.py — PCA of the eight directed structural metrics (Supplementary S12)
=============================================================================
Principal component analysis of the eight directed structural metrics of
Table 1 on the L2 valid set (n = 4,330 under the revised-protocol filter:
SCGC ratio >= 0.3 and defined random-removal threshold). Reports variance
explained, loadings, and the post-hoc correlation of each principal
component with the targeted-attack threshold, cross-validating the partial
correlation and VIF analysis of the main text. Pure post-processing: no
simulation is re-run.

Input:  ../../data/match_summary_mc100_guarded.csv (read-only)
Output: pca_variance.csv, pca_loadings.csv, pc_vs_fc.csv,
        fig_pca_metrics.png/.pdf (written next to this script)

Acceptance gate: reproduces all published zero-order and partial correlation
coefficients between the eight metrics and the attack threshold (3 decimal
places); exits on mismatch.

Usage:
    cd code/supplementary
    python s12_pca.py

Last modified: 2026-07-21
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
MATCH_SUMMARY = _SCRIPT_DIR.parent.parent / "data" / "match_summary_mc100_guarded.csv"

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

# Published values (main-text Section 3.2 and the partial-correlation figure)
GATE_ZERO = {"AvgDegree": 0.785, "Mu11_Mu": 0.782, "FlowHierarchy": -0.766,
             "Clustering": 0.667, "Reciprocity": 0.627,
             "CyclicTriadFraction": 0.579, "OutDegreeCV": 0.237,
             "Assortativity": 0.094}
GATE_PARTIAL = {"FlowHierarchy": -0.311, "Mu11_Mu": 0.104,
                "AvgDegree": 0.031, "Clustering": -0.040, "Reciprocity": -0.080,
                "Assortativity": 0.041, "OutDegreeCV": 0.037, "CyclicTriadFraction": -0.028}


def partial_corr(df, x, y, covars):
    """Partial correlation (linear residual method): corr(resid(x|covars), resid(y|covars))."""
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
    print(f"L2 valid set: n={n} (revised-protocol filter, expected 4,330)")
    assert n == 4330, "filter criteria drifted"

    cols = list(METRICS.keys())
    l2 = l2.dropna(subset=cols + [FC_COL])
    print(f"after dropping missing values: n={len(l2)}")

    # ---------- Acceptance gate ----------
    print("\nGATE - reproduce published Section 3.2 correlation coefficients")
    ok_all = True
    for c in cols:
        got = round(float(l2[c].corr(l2[FC_COL])), 3)
        exp = GATE_ZERO[c]
        ok = got == exp
        ok_all &= ok
        print(f"  zero-order {c:<22} {got:+.3f} vs {exp:+.3f} -> {'PASS' if ok else 'FAIL'}")
    for c in cols:
        others = [x for x in cols if x != c]
        got = round(partial_corr(l2, c, FC_COL, others), 3)
        exp = GATE_PARTIAL[c]
        ok = got == exp
        ok_all &= ok
        print(f"  partial    {c:<22} {got:+.3f} vs {exp:+.3f} -> {'PASS' if ok else 'FAIL'}")
    if not ok_all:
        print("GATE FAILED - stopping.")
        sys.exit(1)
    print("GATE ALL PASS\n")

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
    var.to_csv(_SCRIPT_DIR / "pca_variance.csv", index=False)

    load = pd.DataFrame(pca.components_.T, index=cols,
                        columns=[f"PC{i+1}" for i in range(len(cols))]).round(3)
    load.to_csv(_SCRIPT_DIR / "pca_loadings.csv")

    # PC vs f_c: revised-protocol values as the primary reading
    guarded = pd.read_csv(MATCH_SUMMARY,
                          usecols=["MatchID", "TeamID", "Scale", "Venue",
                                   "pc_Attack_Strong"])
    guarded = guarded.rename(columns={"pc_Attack_Strong": "fcA_guarded"})
    l2m = l2.merge(guarded, on=["MatchID", "TeamID", "Scale", "Venue"],
                   how="left", validate="one_to_one")
    assert len(l2m) == len(l2), "merge changed the row count"
    fc_submitted = l2m[FC_COL].to_numpy(float)
    fc_guard = l2m["fcA_guarded"].to_numpy(float)
    mask = ~np.isnan(fc_guard)
    print(f"restricted-protocol attack threshold available: {mask.sum()}/{len(l2m)}")

    def corr(a, b):
        return round(float(np.corrcoef(a, b)[0, 1]), 3)

    pc_fc = pd.DataFrame({
        "PC": [f"PC{i+1}" for i in range(len(cols))],
        "r_with_fcAttack_guarded": [corr(scores[mask, i], fc_guard[mask])
                                    for i in range(len(cols))],
        "r_with_fcAttack_submitted": [corr(scores[:, i], fc_submitted)
                                   for i in range(len(cols))],
        "var_explained": np.round(pca.explained_variance_ratio_, 3),
    })
    pc_fc.to_csv(_SCRIPT_DIR / "pc_vs_fc.csv", index=False)

    print("Variance explained:")
    print(var.to_string(index=False))
    print("\nLoadings (first 3 PCs):")
    print(load[["PC1", "PC2", "PC3"]].to_string())
    print("\nCorrelation of each PC with fc_Attack:")
    print(pc_fc.to_string(index=False))

    # ---------- Figure: scree + loadings heatmap ----------
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
    fig.savefig(_SCRIPT_DIR / "fig_pca_metrics.png", dpi=150, bbox_inches="tight")
    fig.savefig(_SCRIPT_DIR / "fig_pca_metrics.pdf", bbox_inches="tight")
    print("\nFigure saved: fig_pca_metrics.png/.pdf")


if __name__ == "__main__":
    main()
