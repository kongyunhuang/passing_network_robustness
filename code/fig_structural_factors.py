"""
fig_structural_factors.py — Structural factor scatter plots for robustness prediction (Fig. 3)
==============================================================================================
Generates a 2x4 scatter panel showing the relationship between 8 directed
structural metrics and fc(Attack). Each panel includes match-level scatter
(small transparent points) and team-mean overlay (large points).

Input:  data/match_summary.csv
Output: figures/fig3.png, figures/fig3.pdf

Usage:
  python code/fig_structural_factors.py

Last modified: 2026-03-27
"""

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# ============================================================================
# Paths
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / 'data' / 'match_summary.csv'
OUTPUT_DIR = BASE_DIR / 'figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Color scheme
# ============================================================================
COLORS = {
    'epl': '#1B4F72',
    'laliga': '#C0392B',
    'robust': '#1B4F72',
    'fragile': '#C0392B',
    'grid': '#EAECEE',
}

TEAM_SHORT_MAP = {
    "Manchester United": "Man Utd", "Manchester City": "Man City",
    "Tottenham Hotspur": "Spurs", "Brighton & Hove Albion": "Brighton",
    "Wolverhampton Wanderers": "Wolves", "Newcastle United": "Newcastle",
    "West Ham United": "West Ham", "Nottingham Forest": "Nott'm Forest",
    "AFC Bournemouth": "Bournemouth", "Sheffield United": "Sheff Utd",
}


def shorten_name(name: str) -> str:
    return TEAM_SHORT_MAP.get(name, name[:12] if len(name) > 12 else name)


def sig_stars(p: float) -> str:
    if p < 0.001: return '***'
    elif p < 0.01: return '**'
    elif p < 0.05: return '*'
    return ''


def add_regression(ax, x, y, show_label=True):
    """Add regression line with 95% CI band. Returns (r, p)."""
    mask = ~(np.isnan(x) | np.isnan(y))
    x_c, y_c = np.array(x)[mask], np.array(y)[mask]
    if len(x_c) < 5:
        return np.nan, np.nan

    r, p = stats.pearsonr(x_c, y_c)
    slope, intercept, _, _, _ = stats.linregress(x_c, y_c)
    x_line = np.linspace(x_c.min(), x_c.max(), 100)
    y_line = slope * x_line + intercept

    line_color = COLORS['robust'] if r > 0 else COLORS['fragile']
    label = f'r = {r:.3f}{sig_stars(p)}' if show_label else None
    ax.plot(x_line, y_line, color=line_color, linewidth=2.2,
            linestyle='-', label=label, zorder=3)

    if len(x_c) > 5:
        n = len(x_c)
        x_mean = np.mean(x_c)
        se = np.sqrt(np.sum((y_c - slope * x_c - intercept)**2) / (n - 2))
        t_val = stats.t.ppf(0.975, n - 2)
        ci = t_val * se * np.sqrt(1/n + (x_line - x_mean)**2 / np.sum((x_c - x_mean)**2))
        ax.fill_between(x_line, y_line - ci, y_line + ci,
                        color=line_color, alpha=0.12, zorder=1)
    return r, p


# ============================================================================
# Publication style
# ============================================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 14,
    'axes.titlesize': 16, 'axes.titleweight': 'bold',
    'axes.labelsize': 16, 'axes.labelweight': 'medium',
    'xtick.labelsize': 14, 'ytick.labelsize': 14,
    'legend.fontsize': 12,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 1.2, 'axes.edgecolor': '#333333',
    'axes.unicode_minus': False, 'axes.grid': False,
    'figure.dpi': 150, 'figure.facecolor': 'white',
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'savefig.facecolor': 'white', 'savefig.pad_inches': 0.15,
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Times New Roman',
    'mathtext.it': 'Times New Roman:italic',
    'mathtext.bf': 'Times New Roman:bold',
})


def main():
    df = pd.read_csv(DATA_PATH)
    df['TeamShort'] = df['Team'].apply(shorten_name)

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes = axes.flatten()

    # Match-level data with SCGC filter
    df_l2 = df[(df['Scale'] == 'L2') & (df['SCGC_Ratio'] >= 0.3)
               & (df['fc_Error_Strong'] > 0) & (df['fc_Attack_Strong'] > 0)]
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
        y_match = df_l2['fc_Attack_Strong'].values
        mask = ~(np.isnan(x_match) | np.isnan(y_match))
        x_m, y_m = x_match[mask], y_match[mask]

        # Match-level scatter (small, transparent)
        for league, color, mkr in [('Premier League', COLORS['epl'], 'o'),
                                    ('La Liga', COLORS['laliga'], 's')]:
            d = df_l2[df_l2['League'] == league]
            ax.scatter(d[col], d['fc_Attack_Strong'],
                       c=color, marker=mkr, s=8, alpha=0.15,
                       edgecolors='none', zorder=3)
        # Team means overlay
        for league, color, mkr in [('Premier League', COLORS['epl'], 'o'),
                                    ('La Liga', COLORS['laliga'], 's')]:
            d = team_agg[team_agg['League'] == league]
            ax.scatter(d[col], d['fc_Attack_Strong'],
                       c=color, marker=mkr, s=50, alpha=0.9,
                       edgecolors='black', linewidth=0.5,
                       label=league, zorder=5)

        r, p = add_regression(ax, pd.Series(x_m), pd.Series(y_m), show_label=False)
        if idx % 4 == 0:
            ax.set_ylabel('$f_c^{Attack}$', fontsize=20)
        else:
            ax.set_ylabel('')
        ax.set_title(f'{panel}  {label}', fontsize=16, fontweight='bold')

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
        if idx == 7:
            ax.legend(loc='best', fontsize=12)
        else:
            if ax.get_legend():
                ax.get_legend().remove()
        ax.tick_params(axis='both', labelsize=14)
        ax.grid(True, alpha=0.3, color=COLORS['grid'])

    plt.tight_layout()
    for fmt in ['png', 'pdf']:
        fig.savefig(OUTPUT_DIR / f'fig3.{fmt}')
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR / 'fig3.png'}")


if __name__ == '__main__':
    main()
