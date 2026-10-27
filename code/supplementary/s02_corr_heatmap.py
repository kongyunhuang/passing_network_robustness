"""
s02_corr_heatmap.py  Correlation heatmap of the eight directed structural metrics (Fig. S1)
==========================================================================================
Lower-triangular Pearson correlations of the eight structural metrics, L2
single-match networks with SCGC ratio of at least 0.3 and a positive
random-failure threshold. The published figure uses the 4,343 networks that
pass under the first setting of Supplementary Table S8; the published analysis
leaves 4,330 (caption, Section S2), on which the Section S2 values are checked.

Input  data/match_summary.csv, data/match_summary_mc100_guarded.csv
Output  figures/figS1.png, figures/figS1.pdf, results/figS1_correlation_matrix.csv

Usage
  python code/supplementary/s02_corr_heatmap.py

Last modified 2026-10-27
"""

import sys

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ============================================================================
# Paths
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent.parent
FIRST_SETTING = BASE_DIR / 'data' / 'match_summary.csv'
# Published analysis (100 Monte Carlo trials at all scales, restricted
# threshold estimator)
PUBLISHED = BASE_DIR / 'data' / 'match_summary_mc100_guarded.csv'
OUTPUT_DIR = BASE_DIR / 'figures'
RESULTS_DIR = BASE_DIR / 'results'

# ============================================================================
# Publication style
# ============================================================================
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'figure.dpi': 300,
    'mathtext.fontset': 'cm',
})

METRICS = {
    'AvgDegree': r'$\langle k \rangle$',
    'Clustering': r'$C$',
    'Assortativity': r'$r$',
    'Reciprocity': r'$\rho_{\leftrightarrow}$',
    'FlowHierarchy': r'$h$',
    'OutDegreeCV': r'$\mathrm{CV_{out}}$',
    'Mu11_Mu': r'$\mu_{11}/\mu$',
    'CyclicTriadFraction': r'$f_{\mathrm{cyc}}$',
}
SHORT = {'AvgDegree': 'k', 'Clustering': 'C', 'Assortativity': 'r', 'Reciprocity': 'rho',
         'FlowHierarchy': 'h', 'OutDegreeCV': 'CV', 'Mu11_Mu': 'mu', 'CyclicTriadFraction': 'fcyc'}

# Annotations of the published Fig. S1, lower triangle row by row
FIG_S1 = {
    ('Clustering', 'AvgDegree'): 0.86,
    ('Assortativity', 'AvgDegree'): 0.10, ('Assortativity', 'Clustering'): 0.03,
    ('Reciprocity', 'AvgDegree'): 0.79, ('Reciprocity', 'Clustering'): 0.75,
    ('Reciprocity', 'Assortativity'): 0.16,
    ('FlowHierarchy', 'AvgDegree'): -0.87, ('FlowHierarchy', 'Clustering'): -0.76,
    ('FlowHierarchy', 'Assortativity'): -0.04, ('FlowHierarchy', 'Reciprocity'): -0.72,
    ('OutDegreeCV', 'AvgDegree'): 0.21, ('OutDegreeCV', 'Clustering'): 0.26,
    ('OutDegreeCV', 'Assortativity'): 0.16, ('OutDegreeCV', 'Reciprocity'): 0.32,
    ('OutDegreeCV', 'FlowHierarchy'): -0.11,
    ('Mu11_Mu', 'AvgDegree'): 0.97, ('Mu11_Mu', 'Clustering'): 0.86,
    ('Mu11_Mu', 'Assortativity'): 0.13, ('Mu11_Mu', 'Reciprocity'): 0.82,
    ('Mu11_Mu', 'FlowHierarchy'): -0.83, ('Mu11_Mu', 'OutDegreeCV'): 0.40,
    ('CyclicTriadFraction', 'AvgDegree'): 0.73, ('CyclicTriadFraction', 'Clustering'): 0.77,
    ('CyclicTriadFraction', 'Assortativity'): 0.25, ('CyclicTriadFraction', 'Reciprocity'): 0.53,
    ('CyclicTriadFraction', 'FlowHierarchy'): -0.59, ('CyclicTriadFraction', 'OutDegreeCV'): 0.34,
    ('CyclicTriadFraction', 'Mu11_Mu'): 0.77,
}
N_FIGURE, N_TEXT = 4343, 4330
CORE = ['AvgDegree', 'Clustering', 'Reciprocity', 'Mu11_Mu']


def l2_sample(df, error_col):
    """L2 networks with SCGC ratio of at least 0.3 and a positive random-failure threshold."""
    return df[(df['Scale'] == 'L2') & (df['SCGC_Ratio'] >= 0.3) & (df[error_col] > 0)].copy()


def text_checks(c):
    """Values printed in Supplementary Section S2."""
    def rng(row, cols):
        v = [c.loc[row, x] for x in cols]
        return min(v), max(v)
    h_lo, h_hi = rng('FlowHierarchy', CORE)
    f_lo, f_hi = rng('CyclicTriadFraction', CORE)
    cv_lo, cv_hi = rng('OutDegreeCV', CORE)
    r_max = max(abs(c.loc['Assortativity', x]) for x in METRICS if x != 'Assortativity')
    return [
        ('S2 r(<k>, mu11/mu)', c.loc['AvgDegree', 'Mu11_Mu'], 0.97),
        ('S2 r(<k>, C)', c.loc['AvgDegree', 'Clustering'], 0.86),
        ('S2 r(<k>, rho)', c.loc['AvgDegree', 'Reciprocity'], 0.79),
        ('S2 r(C, mu11/mu)', c.loc['Clustering', 'Mu11_Mu'], 0.86),
        ('S2 r(rho, mu11/mu)', c.loc['Reciprocity', 'Mu11_Mu'], 0.82),
        ('S2 h with core group, weakest', h_hi, -0.72),
        ('S2 h with core group, strongest', h_lo, -0.87),
        ('S2 f_cyc with core group, lowest', f_lo, 0.53),
        ('S2 f_cyc with core group, highest', f_hi, 0.77),
        ('S2 CV_out with core group, lowest', cv_lo, 0.21),
        ('S2 CV_out with core group, highest', cv_hi, 0.40),
        ('S2 largest |r| of assortativity', r_max, 0.25),
        ('S2 r^2(<k>, mu11/mu)', c.loc['AvgDegree', 'Mu11_Mu'] ** 2, 0.94),
    ]


def main():
    cols = list(METRICS.keys())
    # The structural metrics are identical in both files. 13 of the 4,343
    # networks of the figure sample have no positive random-failure threshold
    # in the published analysis, which leaves the 4,330 networks of Section S2.
    fig_df = l2_sample(pd.read_csv(FIRST_SETTING), 'fc_Error_Strong')
    text_df = l2_sample(pd.read_csv(PUBLISHED), 'pc_Error_Strong')
    print(f"Figure sample (first setting of Table S8)  N = {len(fig_df)}")
    print(f"Section S2 sample (published analysis)  N = {len(text_df)}")

    corr_text = text_df[cols].corr()
    corr = fig_df[cols].corr()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    corr.to_csv(RESULTS_DIR / 'figS1_correlation_matrix.csv', float_format='%.6f')

    print("\nCorrelation matrix, figure sample:")
    print(corr.rename(index=SHORT, columns=SHORT).round(3).to_string())
    print(f"\nCorrelation matrix, N = {len(text_df)}:")
    print(corr_text.rename(index=SHORT, columns=SHORT).round(3).to_string())
    diff = [(a, b) for (a, b) in FIG_S1 if round(corr.loc[a, b], 2) != round(corr_text.loc[a, b], 2)]
    print("\nCells that differ at two decimals between the two samples: "
          + (", ".join(f"{SHORT[a]}-{SHORT[b]} {corr.loc[a, b]:.2f} vs {corr_text.loc[a, b]:.2f}"
                       for a, b in diff) or "none"))

    # Acceptance check
    checks = [('Figure sample N', len(fig_df), N_FIGURE, len(fig_df) == N_FIGURE),
              ('Section S2 sample N', len(text_df), N_TEXT, len(text_df) == N_TEXT)]
    for (a, b), exp in FIG_S1.items():
        got = corr.loc[a, b]
        checks.append((f"Fig. S1 {SHORT[a]}-{SHORT[b]}", got, exp, abs(got - exp) <= 0.005 + 1e-12))
    for label, got, exp in text_checks(corr_text):
        checks.append((label, got, exp, abs(got - exp) <= 0.005 + 1e-12))

    corr.index = [METRICS[c] for c in corr.index]
    corr.columns = [METRICS[c] for c in corr.columns]

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 6.5))

    # Mask upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    # Custom diverging colormap
    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmin=-1, vmax=1,
        center=0,
        annot=True,
        fmt='.2f',
        square=True,
        linewidths=0.8,
        linecolor='white',
        cbar_kws={'shrink': 0.8, 'label': 'Pearson r'},
        annot_kws={'size': 10},
        ax=ax,
    )

    ax.set_title('')

    # Rotate labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / 'figS1.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / 'figS1.pdf', bbox_inches='tight')
    plt.close(fig)
    print("\nSaved figures/figS1.png, figures/figS1.pdf, results/figS1_correlation_matrix.csv\n")

    print("Acceptance check against Fig. S1 and Supplementary Section S2")
    for label, got, exp, ok in checks:
        print(f"  {label:<40} {got:10.4g} vs {exp:<7g} {'PASS' if ok else 'FAIL'}")
    failed = [c for c in checks if not c[3]]
    if failed:
        print(f"GATE FAILED  {len(failed)} of {len(checks)} values differ from the published values")
        for label, got, exp, _ in failed:
            print(f"  {label}  computed {got:.4f}, published {exp}")
        sys.exit(1)
    print(f"GATE PASS  all {len(checks)} values match")


if __name__ == '__main__':
    main()
