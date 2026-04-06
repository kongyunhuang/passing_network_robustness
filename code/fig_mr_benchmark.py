"""
fig_mr_benchmark.py — Directed Molloy-Reed benchmark density scatter plot (Fig. 5)
==================================================================================
Generates a hexbin density scatter plot of dMR predicted threshold vs simulated
threshold at L2 scale, including the y=x reference line and statistical
annotations (r, R^2, Bias, MAE).

Input:  data/match_summary.csv
Output: figures/fig5.png, figures/fig5.pdf

Usage:
  python code/fig_mr_benchmark.py

Last modified: 2026-03-27
"""

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
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
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 12,
    'axes.titlesize': 13, 'axes.titleweight': 'bold',
    'axes.labelsize': 13, 'axes.labelweight': 'medium',
    'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 1.2, 'axes.edgecolor': '#333333',
    'axes.unicode_minus': False, 'axes.grid': False,
    'figure.dpi': 150, 'figure.facecolor': 'white',
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'savefig.facecolor': 'white', 'savefig.pad_inches': 0.15,
    'mathtext.fontset': 'cm',
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.width': 1.0, 'ytick.major.width': 1.0,
    'xtick.major.size': 5, 'ytick.major.size': 5,
})


def main():
    df = pd.read_csv(DATA_PATH)
    MR_COL = 'fc_dMR'
    df_valid = df[df[MR_COL].notna() & (df[MR_COL] > 0)].copy()
    df_l2 = df_valid[df_valid['Scale'] == 'L2']
    n_l2 = len(df_l2)

    fig, ax = plt.subplots(figsize=(7, 6))

    # Density scatter (hexbin)
    hb = ax.hexbin(df_l2[MR_COL], df_l2['fc_Error_Strong'],
                   gridsize=35, cmap='Blues', mincnt=1, zorder=2)
    cb = plt.colorbar(hb, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label('Count', fontsize=12)

    # Identity line
    lims = [0, 1.05]
    ax.plot(lims, lims, '--', color='#333333', linewidth=1.2, alpha=0.7, zorder=10)

    # Stats
    r_val, p_val = sp_stats.pearsonr(df_l2[MR_COL], df_l2['fc_Error_Strong'])
    residuals = df_l2[MR_COL] - df_l2['fc_Error_Strong']
    bias = residuals.mean()
    mae = residuals.abs().mean()
    r2 = r_val ** 2

    # Annotation
    ax.text(0.05, 0.95,
            f'r = {r_val:.3f}\n$R^2$ = {r2:.3f}\nBias = {bias:+.3f}\nMAE = {mae:.3f}',
            transform=ax.transAxes, fontsize=13, va='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8, edgecolor='none'))

    ax.set_xlabel(r'$f_c^{\mathrm{dMR}}$', fontsize=14)
    ax.set_ylabel(r'$f_c^{\mathrm{Error}}$', fontsize=14)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal')
    ax.tick_params(axis='both', labelsize=12)

    plt.tight_layout()
    for fmt in ['png', 'pdf']:
        fig.savefig(OUTPUT_DIR / f'fig5.{fmt}')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'fig5.png'}")


if __name__ == '__main__':
    main()
