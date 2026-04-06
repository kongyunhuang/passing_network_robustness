"""
fig_corr_heatmap.py — Correlation matrix heatmap of 8 directed structural metrics (Fig. S1)
===========================================================================================
Generates a lower-triangular correlation matrix heatmap using match-level
L2 data (N~4343), consistent with Section 3.2 of the main text.

Input:  data/match_summary.csv
Output: figures/figS1.png, figures/figS1.pdf

Usage:
  python code/fig_corr_heatmap.py

Last modified: 2026-03-27
"""

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
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / 'data' / 'match_summary.csv'
OUTPUT_DIR = BASE_DIR / 'figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Publication style
# ============================================================================
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 11, 'axes.labelsize': 12,
    'figure.dpi': 300, 'mathtext.fontset': 'cm',
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
})


def main():
    df = pd.read_csv(DATA_PATH)

    # Filter L2 + SCGC filter (consistent with main text)
    df_l2 = df[(df['Scale'] == 'L2') & (df['SCGC_Ratio'] >= 0.3) & (df['fc_Error_Strong'] > 0)].copy()

    metrics = {
        'AvgDegree': r'$\langle k \rangle$',
        'Clustering': r'$C$',
        'Assortativity': r'$r$',
        'Reciprocity': r'$\rho_{\leftrightarrow}$',
        'FlowHierarchy': r'$h$',
        'OutDegreeCV': r'$\mathrm{CV_{out}}$',
        'Mu11_Mu': r'$\mu_{11}/\mu$',
        'CyclicTriadFraction': r'$f_{\mathrm{cyc}}$',
    }

    print(f"Match-level N = {len(df_l2)}")

    corr = df_l2[list(metrics.keys())].corr()
    corr.index = [metrics[c] for c in corr.index]
    corr.columns = [metrics[c] for c in corr.columns]

    print("\nCorrelation matrix:")
    print(corr.round(3).to_string())

    fig, ax = plt.subplots(figsize=(8, 6.5))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    sns.heatmap(
        corr, mask=mask, cmap=cmap,
        vmin=-1, vmax=1, center=0,
        annot=True, fmt='.2f', square=True,
        linewidths=0.8, linecolor='white',
        cbar_kws={'shrink': 0.8, 'label': 'Pearson r'},
        annot_kws={'size': 10}, ax=ax,
    )

    ax.set_title('')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()
    for fmt in ['png', 'pdf']:
        fig.savefig(OUTPUT_DIR / f'figS1.{fmt}', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {OUTPUT_DIR / 'figS1.png'}")


if __name__ == '__main__':
    main()
