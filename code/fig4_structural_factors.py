"""
fig4_structural_factors.py  Structural metrics versus targeted-attack threshold (Fig. 4)
=====================================================================================
Bivariate scatter plots of the eight directed structural metrics against the
L2 Attack threshold, with match-level points, team means, OLS fit and 95%
confidence band. Primary L2 sample (N = 4,330).

Input  data/match_summary_mc100_guarded.csv
Output  figures/fig4.png, figures/fig4.pdf

Usage
  python code/fig4_structural_factors.py

Last modified 2026-10-27
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from style import COLORS, sig_stars, add_regression, load_published_matches, save


def main():
    df = load_published_matches()
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes = axes.flatten()

    # Match-level data with SCGC filter
    df_l2 = df[(df['Scale'] == 'L2') & (df['SCGC_Ratio'] >= 0.3) & (df['pc_Error_Strong'] > 0) & (df['pc_Attack_Strong'] > 0)]
    # Team means for overlay
    team_agg = df_l2.groupby(['TeamShort', 'League']).agg(
        {c: 'mean' for c in df_l2.select_dtypes(include='number').columns}
    ).reset_index()

    factors = [
        ('AvgDegree', 'Mean Degree $\\langle k \\rangle$'),
        ('Clustering', 'Clustering $C$'),
        ('Assortativity', 'Assortativity $r$'),
        ('Reciprocity', 'Reciprocity $\\rho_{\\leftrightarrow}$'),
        ('FlowHierarchy', 'Flow Hierarchy $h$'),
        ('OutDegreeCV', 'Out-Degree $\\mathrm{CV}_{\\mathrm{out}}$'),
        ('Mu11_Mu', 'Directed MR $\\mu_{11}/\\mu$'),
        ('CyclicTriadFraction', 'Cyclic Triad $f_{\\mathrm{cyc}}$'),
    ]
    panels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']

    for idx, (ax, (col, label), panel) in enumerate(zip(axes, factors, panels)):
        if col not in df_l2.columns:
            ax.text(0.5, 0.5, f'{col}\nnot available',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, color='gray')
            ax.set_title(f'{panel} {label}', fontsize=14, fontweight='bold')
            continue

        x_match = df_l2[col].values
        y_match = df_l2['pc_Attack_Strong'].values
        mask = ~(np.isnan(x_match) | np.isnan(y_match))
        x_m, y_m = x_match[mask], y_match[mask]

        # Match-level scatter (small, transparent)
        for league, color, mkr in [('Premier League', COLORS['epl'], 'o'),
                                    ('La Liga', COLORS['laliga'], 's')]:
            d = df_l2[df_l2['League'] == league]
            ax.scatter(d[col], d['pc_Attack_Strong'],
                      c=color, marker=mkr, s=8, alpha=0.15,
                      edgecolors='none', zorder=3)
        # Team means overlay (larger, opaque)
        for league, color, mkr in [('Premier League', COLORS['epl'], 'o'),
                                    ('La Liga', COLORS['laliga'], 's')]:
            d = team_agg[team_agg['League'] == league]
            ax.scatter(d[col], d['pc_Attack_Strong'],
                      c=color, marker=mkr, s=50, alpha=0.9,
                      edgecolors='black', linewidth=0.5,
                      label=league, zorder=5)

        r, p = add_regression(ax, pd.Series(x_m), pd.Series(y_m), show_label=False)
        ax.set_xlabel('')
        # y-axis label on the left column only
        if idx % 4 == 0:
            ax.set_ylabel(r'$f_{c,\mathrm{L2}}^{Attack}$', fontsize=20)
        else:
            ax.set_ylabel('')
        ax.set_title(f'{panel}  {label}', fontsize=16, fontweight='bold')
        # r and p in a lower corner, left for negative and right for positive r
        if p < 0.001:
            p_str = 'p < 0.001'
        else:
            p_str = f'p = {p:.3f}'
        if r < 0:
            ax.text(0.05, 0.05, f'r = {r:.3f}\n{p_str}',
                    transform=ax.transAxes, fontsize=12, va='bottom', ha='left',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))
        else:
            ax.text(0.95, 0.05, f'r = {r:.3f}\n{p_str}',
                    transform=ax.transAxes, fontsize=12, va='bottom', ha='right',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))
        # league legend in panel (e) only
        if idx == 4:
            ax.legend(loc='upper right', fontsize=12)
        else:
            ax.get_legend() and ax.get_legend().remove() if ax.get_legend() else None
        ax.tick_params(axis='both', labelsize=14)
        ax.grid(True, alpha=0.3, color=COLORS['grid'])

    plt.tight_layout()
    save(fig, 'fig4')


if __name__ == '__main__':
    main()
