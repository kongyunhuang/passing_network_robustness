"""
fig_rai_scatter.py — RAI scatter plot (Fig. 6)
==============================================
Positions 48 teams on the fc_Error vs fc_Attack plane, with 1-sigma
ellipses and mean markers for ER/BA/WS/RGG synthetic baselines overlaid.
Diagonal = RAI=0; below the diagonal = robust yet fragile.

Input:  data/match_summary.csv, data/synthetic_baselines.csv
Output: figures/fig6.png, figures/fig6.pdf

Usage:
  python code/fig_rai_scatter.py

Last modified: 2026-03-27
"""

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from pathlib import Path

# ============================================================================
# Paths
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Publication style
# ============================================================================
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 12, 'axes.titlesize': 14, 'axes.titleweight': 'bold',
    'axes.spines.top': False, 'axes.spines.right': False,
    'mathtext.fontset': 'cm',
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
})


def confidence_ellipse(x, y, ax, n_std=1.0, **kwargs):
    cov = np.cov(x, y)
    p = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]) if cov[0, 0] * cov[1, 1] > 0 else 0
    ellipse = Ellipse((0, 0), width=np.sqrt(1 + p) * 2, height=np.sqrt(1 - p) * 2, **kwargs)
    transf = transforms.Affine2D().rotate_deg(45) \
        .scale(np.sqrt(cov[0, 0]) * n_std, np.sqrt(cov[1, 1]) * n_std) \
        .translate(np.mean(x), np.mean(y))
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


# Team labels with manual offsets
TEAM_LABELS = {
    'Manchester City': (12, 8),
    'Barcelona': (12, -4),
    'Real Madrid': (-52, 8),
    'Liverpool': (5, 18),
    'Chelsea': (15, -15),
    'Brighton & Hove Albion': (-22, 20),
    'Arsenal': (-50, 12),
    'Girona': (12, -12),
    'Sheffield United': (-12, -22),
    'Luton Town': (12, -18),
    'Nottingham Forest': (15, -18),
    'Granada': (12, -15),
    'Espanyol': (15, -8),
}


def main():
    ms = pd.read_csv(DATA_DIR / 'match_summary.csv')
    l2 = ms[(ms['Scale'] == 'L2') & (ms['SCGC_Ratio'] >= 0.3)
            & (ms['fc_Error_Strong'] > 0) & (ms['fc_Attack_Strong'] > 0)]
    ta = l2.groupby(['Team', 'League']).agg(
        {'fc_Error_Strong': 'mean', 'fc_Attack_Strong': 'mean'}).reset_index()
    ta['RAI'] = (ta['fc_Error_Strong'] - ta['fc_Attack_Strong']) / ta['fc_Error_Strong']
    ta['ShortName'] = ta['Team'].str.replace('Manchester ', 'Man ') \
        .str.replace('Sheffield ', 'Sheff ') \
        .str.replace('Brighton & Hove Albion', 'Brighton') \
        .str.replace('Wolverhampton Wanderers', 'Wolves') \
        .str.replace('Nottingham Forest', 'Nott. Forest') \
        .str.replace('AFC Bournemouth', 'Bournemouth') \
        .str.replace('Deportivo Alaves', 'Alaves') \
        .str.replace('Real Valladolid', 'Valladolid') \
        .str.replace('Atletico Madrid', 'Atletico') \
        .str.replace('Rayo Vallecano', 'Rayo')

    syn = pd.read_csv(DATA_DIR / 'synthetic_baselines.csv')
    syn_rename = {'fc_Attack': 'fc_Attack_Strong', 'fc_Error': 'fc_Error_Strong'}
    syn.rename(columns={k: v for k, v in syn_rename.items() if k in syn.columns}, inplace=True)
    syn = syn[(syn['fc_Error_Strong'] > 0) & (syn['fc_Attack_Strong'] > 0)]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 0.9], [0, 0.9], 'k--', lw=1, alpha=0.3, zorder=1)

    # Synthetic: ellipse + mean
    syn_cfg = {
        'ER':  {'color': '#2980B9', 'marker': 's'},
        'BA':  {'color': '#C0392B', 'marker': '^'},
        'WS':  {'color': '#E69F00', 'marker': 'D'},
        'RGG': {'color': '#85C1E9', 'marker': 'v'},
    }
    for model, cfg in syn_cfg.items():
        d = syn[syn['Model'] == model]
        confidence_ellipse(d['fc_Error_Strong'], d['fc_Attack_Strong'], ax, n_std=1.0,
                           facecolor=cfg['color'], alpha=0.12, edgecolor=cfg['color'],
                           linewidth=1.5, zorder=2)
        me, ma = d['fc_Error_Strong'].mean(), d['fc_Attack_Strong'].mean()
        rai = (me - ma) / me if me > 0 else 0
        ax.scatter(me, ma, c=cfg['color'], marker=cfg['marker'], s=180,
                   edgecolors='black', linewidth=1.5, zorder=7,
                   label=f'{model} (RAI={rai:.2f})')

    # Football: 48 team points
    for league, color, marker in [('Premier League', '#1B4F72', 'o'), ('La Liga', '#C0392B', 's')]:
        d = ta[ta['League'] == league]
        ax.scatter(d['fc_Error_Strong'], d['fc_Attack_Strong'],
                   c=color, marker=marker, s=55, alpha=0.8,
                   edgecolors='white', linewidth=0.5, zorder=5,
                   label=f'{league} (n={len(d)})')

    # Football mean + ellipse
    fb_me = ta['fc_Error_Strong'].mean()
    fb_ma = ta['fc_Attack_Strong'].mean()
    fb_rai = (fb_me - fb_ma) / fb_me
    confidence_ellipse(ta['fc_Error_Strong'], ta['fc_Attack_Strong'], ax, n_std=1.0,
                       facecolor='#9E9E9E', alpha=0.20, edgecolor='#333',
                       linewidth=1.5, zorder=2)
    ax.scatter(fb_me, fb_ma, c='#333', marker='o', s=200,
               edgecolors='black', linewidth=2, zorder=8,
               label=f'Football mean (RAI={fb_rai:.2f})')

    # Team labels with leader lines
    for team, offset in TEAM_LABELS.items():
        row = ta[ta['Team'] == team]
        if len(row) == 0:
            continue
        short = row['ShortName'].values[0]
        color = '#1B4F72' if row['League'].values[0] == 'Premier League' else '#C0392B'
        ax.annotate(short,
                    xy=(row['fc_Error_Strong'].values[0], row['fc_Attack_Strong'].values[0]),
                    xytext=offset, textcoords='offset points',
                    fontsize=9, color=color, fontweight='bold',
                    arrowprops=dict(arrowstyle='-', color=color, alpha=0.5, lw=0.8))

    ax.set_xlabel(r'$f_c^{Error}$', fontsize=18)
    ax.set_ylabel(r'$f_c^{Attack}$', fontsize=18)
    ax.set_xlim(0.05, 0.82)
    ax.set_ylim(0.05, 0.65)
    ax.grid(True, alpha=0.15)
    ax.legend(loc='upper left', fontsize=10, markerscale=0.8, framealpha=0.9)

    plt.tight_layout()
    for fmt in ['png', 'pdf']:
        fig.savefig(OUTPUT_DIR / f'fig6.{fmt}', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'fig6.png'}")


if __name__ == '__main__':
    main()
