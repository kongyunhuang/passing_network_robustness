"""
fig3_error_vs_attack.py  Random failure versus targeted attack (Fig. 3)
=====================================================================
(a) Mean pseudo-critical thresholds at the four scales L0 to L3 under random
failure (Error) and targeted attack (Attack). (b) Match-level Error versus
Attack thresholds at L2 with team means overlaid. (c) Distributions of the two
L2 thresholds. Panels (b) and (c) use the primary L2 sample (N = 4,330).

Input  data/match_summary_mc100_guarded.csv
Output  figures/fig3.png, figures/fig3.pdf

Usage
  python code/fig3_error_vs_attack.py

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
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax in axes:
        ax.tick_params(axis='both', labelsize=16)

    # (a) Scale-level bar chart
    ax1 = axes[0]
    scale_data = []
    for scale in ['L0', 'L1', 'L2', 'L3']:
        ds = df[df['Scale'] == scale]
        scale_data.append({
            'Scale': scale,
            'Error': ds['pc_Error_Strong'].mean(),
            'Attack': ds['pc_Attack_Strong'].mean(),
        })
    sdf = pd.DataFrame(scale_data)
    x = np.arange(4)
    w = 0.35
    ax1.bar(x - w/2, sdf['Error'], w, color=COLORS['robust'],
           label='Error (Random)', alpha=0.85)
    ax1.bar(x + w/2, sdf['Attack'], w, color=COLORS['fragile'],
           label='Attack (Targeted)', alpha=0.85)
    for i in range(4):
        diff = sdf.iloc[i]['Error'] - sdf.iloc[i]['Attack']
        y_top = max(sdf.iloc[i]['Error'], sdf.iloc[i]['Attack'])
        ax1.annotate(f'$\\Delta$={diff:.2f}', (i, y_top + 0.02),
                    ha='center', fontsize=16, color='gray')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['L0\n(384)', 'L1\n(96)', 'L2\n(24)', 'L3\n(6)'])
    ax1.set_ylabel(r'$f_{c,\ell}$', fontsize=20)
    ax1.set_xlabel('')
    ax1.set_title(r'(a)  $f_{c,\ell}$ across scales', fontsize=18, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=13)
    ax1.set_ylim(0, 1.08)
    ax1.grid(True, alpha=0.3, axis='y', color=COLORS['grid'])

    # (b) Match-level scatter with team-mean overlay
    ax2 = axes[1]
    df_l2_scatter = df[(df['Scale'] == 'L2') & (df['SCGC_Ratio'] >= 0.3) & (df['pc_Error_Strong'] > 0) & (df['pc_Attack_Strong'] > 0)]
    n_matches = len(df_l2_scatter)
    # Match-level scatter (small, transparent)
    for league, color, marker in [('Premier League', COLORS['epl'], 'o'),
                                   ('La Liga', COLORS['laliga'], 's')]:
        d = df_l2_scatter[df_l2_scatter['League'] == league]
        ax2.scatter(d['pc_Error_Strong'], d['pc_Attack_Strong'],
                   c=color, marker=marker, s=10, alpha=0.18,
                   edgecolors='none', zorder=3)
    # Team-mean overlay (larger, opaque)
    team_agg_3b = df_l2_scatter.groupby(['TeamShort', 'League']).agg(
        {'pc_Error_Strong': 'mean', 'pc_Attack_Strong': 'mean'}).reset_index()
    for league, color, marker in [('Premier League', COLORS['epl'], 'o'),
                                   ('La Liga', COLORS['laliga'], 's')]:
        d = team_agg_3b[team_agg_3b['League'] == league]
        ax2.scatter(d['pc_Error_Strong'], d['pc_Attack_Strong'],
                   c=color, marker=marker, s=50, alpha=0.9,
                   edgecolors='black', linewidth=0.5,
                   label=league, zorder=5)
    r, p = add_regression(ax2, df_l2_scatter['pc_Error_Strong'],
                          df_l2_scatter['pc_Attack_Strong'], show_label=False)
    ax2.set_xlabel(r'$f_{c,\mathrm{L2}}^{Error}$', fontsize=20)
    ax2.set_ylabel(r'$f_{c,\mathrm{L2}}^{Attack}$', fontsize=20)
    sig = sig_stars(p) if p is not np.nan else ''
    ax2.set_title('(b)  Error vs Attack', fontsize=18, fontweight='bold')
    # r and p appended to the legend
    from matplotlib.patches import Patch
    handles, labels = ax2.get_legend_handles_labels()
    if p < 0.001:
        p_str = 'p < 0.001'
    else:
        p_str = f'p = {p:.3f}'
    handles.append(plt.Line2D([0],[0], color='none', label=f'r = {r:.3f}'))
    handles.append(plt.Line2D([0],[0], color='none', label=p_str))
    ax2.legend(handles=handles, loc='upper left', fontsize=13)
    ax2.grid(True, alpha=0.3, color=COLORS['grid'])

    # (c) Distribution (SCGC≥0.3)
    ax3 = axes[2]
    df_l2 = df[(df['Scale'] == 'L2') & (df['SCGC_Ratio'] >= 0.3) & (df['pc_Error_Strong'] > 0) & (df['pc_Attack_Strong'] > 0)]
    ax3.hist(df_l2['pc_Error_Strong'], bins=20, alpha=0.6, color=COLORS['robust'],
            label=f'Error (mean = {df_l2["pc_Error_Strong"].mean():.2f})', density=True)
    ax3.hist(df_l2['pc_Attack_Strong'], bins=20, alpha=0.6, color=COLORS['fragile'],
            label=f'Attack (mean = {df_l2["pc_Attack_Strong"].mean():.2f})', density=True)
    ax3.axvline(df_l2['pc_Error_Strong'].mean(), color=COLORS['robust'],
               linestyle='--', lw=2)
    ax3.axvline(df_l2['pc_Attack_Strong'].mean(), color=COLORS['fragile'],
               linestyle='--', lw=2)
    ax3.set_xlabel(r'$f_{c,\mathrm{L2}}$', fontsize=20)
    ax3.set_ylabel('Density', fontsize=18)
    ax3.set_title(r'(c)  $f_{c,\mathrm{L2}}$ distribution', fontsize=18, fontweight='bold')
    # headroom so the legend sits above the bars
    ax3.set_ylim(0, ax3.get_ylim()[1] * 1.28)
    ax3.legend(loc='upper left', fontsize=13)
    ax3.grid(True, alpha=0.3, color=COLORS['grid'])

    # No suptitle — description goes in figure caption
    plt.tight_layout()
    save(fig, 'fig3')


if __name__ == '__main__':
    main()
