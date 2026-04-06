"""
fig_position_analysis.py — Position degree hierarchy and mediation path visualisation (Fig. S2)
==============================================================================================
Generates a three-panel figure:
(a) Position degree hierarchy violin plot (GK/DEF/MID/FWD)
(b) Degree disparity -> Mean degree (L2) scatter plot
(c) Mean degree (L2) -> fc(Attack) scatter plot

Input:
  - data/position_node_records.csv (full node records)
  - data/position_network_records.csv (network-level position statistics)
  - data/batch_summary.csv (L2 team aggregation)

Output: figures/figS2.png, figures/figS2.pdf

Usage:
  python code/fig_position_analysis.py

Last modified: 2026-03-27
"""

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
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
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 13, 'axes.labelsize': 14, 'axes.titlesize': 14,
    'xtick.labelsize': 12, 'ytick.labelsize': 12,
    'figure.dpi': 300, 'mathtext.fontset': 'cm',
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
})


def main():
    nodes = pd.read_csv(DATA_DIR / 'position_node_records.csv')
    nets = pd.read_csv(DATA_DIR / 'position_network_records.csv')
    batch = pd.read_csv(DATA_DIR / 'batch_summary.csv')

    # Fix league name
    nets['league'] = nets['league'].replace({'EPL': 'Premier League'})

    # Aggregate position network data to team level
    team_pos = nets.groupby(['team_name', 'league']).agg(
        degree_disparity=('degree_disparity', 'mean'),
    ).reset_index()

    # Aggregate batch L2 to team level
    batch_l2 = batch[batch['Scale'] == 'L2'].groupby(['Team', 'League']).agg(
        fc_Attack=('fc_Attack_Strong', 'mean'),
        AvgDegree=('AvgDegree', 'mean'),
        Mu11_Mu=('Mu11_Mu', 'mean'),
    ).reset_index()

    merged = team_pos.merge(batch_l2, left_on=['team_name', 'league'],
                            right_on=['Team', 'League'], how='inner')
    print(f"Merged teams: {len(merged)}")
    print(f"EPL: {(merged['league'] == 'Premier League').sum()}, "
          f"La Liga: {(merged['league'] == 'La Liga').sum()}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel (a): Violin plot
    pos_order = ['Goalkeeper', 'Defender', 'Midfielder', 'Forward']
    pos_labels = ['GK', 'DEF', 'MID', 'FWD']
    pos_colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']

    ax = axes[0]
    for i, (pos, label, color) in enumerate(zip(pos_order, pos_labels, pos_colors)):
        data = nodes[nodes['position_4'] == pos]['degree_total']
        vp = ax.violinplot([data.values], positions=[i], showmeans=True,
                           showmedians=True, showextrema=False)
        for body in vp['bodies']:
            body.set_facecolor(color)
            body.set_alpha(0.7)
        vp['cmeans'].set_color('black')
        vp['cmedians'].set_color('gray')
        ax.text(i, data.mean() + 0.3, f'{data.mean():.2f}', ha='center',
                fontsize=12, fontweight='bold')

    ax.set_xticks(range(4))
    ax.set_xticklabels(pos_labels)
    ax.set_ylabel('Directed Degree (in + out)')
    ax.set_title('(a) Position Degree Hierarchy', fontsize=13)
    ax.set_ylim(0, 15)

    def add_bracket(ax, x1, x2, y, text):
        ax.plot([x1, x1, x2, x2], [y, y + 0.15, y + 0.15, y], 'k-', lw=1)
        ax.text((x1 + x2) / 2, y + 0.2, text, ha='center', fontsize=8)

    add_bracket(ax, 0, 1, 6.5, '***')
    add_bracket(ax, 1, 2, 5.5, '***')
    add_bracket(ax, 2, 3, 4.8, '***')

    # Color map for leagues
    cmap = {'Premier League': '#7B68EE', 'La Liga': '#FFA500'}
    colors = merged['league'].map(cmap)

    # Panel (b): degree_disparity -> AvgDegree
    ax = axes[1]
    x, y = merged['degree_disparity'], merged['AvgDegree']
    ax.scatter(x, y, c=colors, s=50, alpha=0.7, edgecolors='white', linewidth=0.5)
    slope, intercept, r, p, se = stats.linregress(x, y)
    xline = np.linspace(x.min(), x.max(), 100)
    ax.plot(xline, slope * xline + intercept, 'k--', lw=1.5)
    ax.set_xlabel('Back-Front Degree Disparity')
    ax.set_ylabel(r'Mean Degree $\langle k \rangle$ (L2)')
    ax.set_title('(b) Path a: Disparity -> Density', fontsize=13)
    p_str = 'P < 0.001' if p < 0.001 else f'P = {p:.3f}'
    ax.text(0.05, 0.92, f'r = {r:.3f}\n{p_str}', transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Panel (c): AvgDegree -> fc_Attack
    ax = axes[2]
    x2, y2 = merged['AvgDegree'], merged['fc_Attack']
    ax.scatter(x2, y2, c=colors, s=50, alpha=0.7, edgecolors='white', linewidth=0.5)
    slope2, intercept2, r2, p2, se2 = stats.linregress(x2, y2)
    xline2 = np.linspace(x2.min(), x2.max(), 100)
    ax.plot(xline2, slope2 * xline2 + intercept2, 'k--', lw=1.5)
    ax.set_xlabel(r'Mean Degree $\langle k \rangle$ (L2)')
    ax.set_ylabel(r'$f_c^{\mathrm{Attack}}$')
    ax.set_title('(c) Path b: Density -> Robustness', fontsize=13)
    p_str2 = 'P < 0.001' if p2 < 0.001 else f'P = {p2:.3f}'
    ax.text(0.05, 0.92, f'r = {r2:.3f}\n{p_str2}', transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#7B68EE', markersize=8, label='EPL'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFA500', markersize=8, label='La Liga'),
    ]
    axes[2].legend(handles=legend_elements, loc='lower right', fontsize=9)

    plt.tight_layout()
    for fmt in ['png', 'pdf']:
        fig.savefig(OUTPUT_DIR / f'figS2.{fmt}', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {OUTPUT_DIR / 'figS2.png'}")
    print(f"Path a: r={r:.3f}, P={p:.2e}")
    print(f"Path b: r={r2:.3f}, P={p2:.2e}")


if __name__ == '__main__':
    main()
