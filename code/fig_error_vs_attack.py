"""
fig_error_vs_attack.py — Differentiated vulnerability: random failure vs targeted attack (Fig. 2)
=================================================================================================
Generates a three-panel composite figure:
(a) Bar chart of mean fc across four scales L0-L3 (Error vs Attack)
(b) Match-level fc_Error vs fc_Attack scatter plot (with team-mean overlay)
(c) L2 fc distribution histograms

Input:  data/match_summary.csv
Output: figures/fig2.png, figures/fig2.pdf

Usage:
  python code/fig_error_vs_attack.py

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

# ============================================================================
# Team short names
# ============================================================================
TEAM_SHORT_MAP = {
    "Manchester United": "Man Utd", "Manchester City": "Man City",
    "Tottenham Hotspur": "Spurs", "Brighton & Hove Albion": "Brighton",
    "Wolverhampton Wanderers": "Wolves", "Newcastle United": "Newcastle",
    "West Ham United": "West Ham", "Nottingham Forest": "Nott'm Forest",
    "AFC Bournemouth": "Bournemouth", "Sheffield United": "Sheff Utd",
    "Leicester City": "Leicester", "Leeds United": "Leeds",
    "Crystal Palace": "C. Palace", "Luton Town": "Luton",
    "Real Madrid": "R. Madrid", "FC Barcelona": "Barcelona",
    "Atletico Madrid": "Atletico", "Real Sociedad": "R. Sociedad",
    "Athletic Club": "Ath. Bilbao", "Villarreal CF": "Villarreal",
    "Sevilla FC": "Sevilla", "Real Betis": "Betis",
    "Valencia CF": "Valencia", "Celta Vigo": "Celta",
    "Getafe CF": "Getafe", "Girona FC": "Girona",
    "Rayo Vallecano": "Rayo", "RCD Mallorca": "Mallorca",
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

    # 95% confidence band
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
    df['Scale'] = pd.Categorical(df['Scale'], categories=['L0', 'L1', 'L2', 'L3'], ordered=True)

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
            'Error': ds['fc_Error_Strong'].mean(),
            'Attack': ds['fc_Attack_Strong'].mean(),
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
    ax1.set_ylabel('$f_c$', fontsize=20)
    ax1.set_title('(a)  $f_c$ across scales', fontsize=18, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=13)
    ax1.set_ylim(0, 1.08)
    ax1.grid(True, alpha=0.3, axis='y', color=COLORS['grid'])

    # (b) Match-level scatter with team-mean overlay
    ax2 = axes[1]
    df_l2_scatter = df[(df['Scale'] == 'L2') & (df['SCGC_Ratio'] >= 0.3)
                       & (df['fc_Error_Strong'] > 0) & (df['fc_Attack_Strong'] > 0)]
    for league, color, marker in [('Premier League', COLORS['epl'], 'o'),
                                   ('La Liga', COLORS['laliga'], 's')]:
        d = df_l2_scatter[df_l2_scatter['League'] == league]
        ax2.scatter(d['fc_Error_Strong'], d['fc_Attack_Strong'],
                    c=color, marker=marker, s=10, alpha=0.18,
                    edgecolors='none', zorder=3)
    team_agg_3b = df_l2_scatter.groupby(['TeamShort', 'League']).agg(
        {'fc_Error_Strong': 'mean', 'fc_Attack_Strong': 'mean'}).reset_index()
    for league, color, marker in [('Premier League', COLORS['epl'], 'o'),
                                   ('La Liga', COLORS['laliga'], 's')]:
        d = team_agg_3b[team_agg_3b['League'] == league]
        ax2.scatter(d['fc_Error_Strong'], d['fc_Attack_Strong'],
                    c=color, marker=marker, s=50, alpha=0.9,
                    edgecolors='black', linewidth=0.5,
                    label=league, zorder=5)
    r, p = add_regression(ax2, df_l2_scatter['fc_Error_Strong'],
                          df_l2_scatter['fc_Attack_Strong'], show_label=False)
    ax2.set_xlabel('$f_c^{Error}$', fontsize=20)
    ax2.set_ylabel('$f_c^{Attack}$', fontsize=20)
    ax2.set_title('(b)  Error vs Attack', fontsize=18, fontweight='bold')
    handles, labels = ax2.get_legend_handles_labels()
    if p < 0.001:
        p_str = 'p < 0.001'
    else:
        p_str = f'p = {p:.3f}'
    handles.append(plt.Line2D([0], [0], color='none', label=f'r = {r:.3f}'))
    handles.append(plt.Line2D([0], [0], color='none', label=p_str))
    ax2.legend(handles=handles, loc='upper left', fontsize=13)
    ax2.grid(True, alpha=0.3, color=COLORS['grid'])

    # (c) Distribution (SCGC >= 0.3)
    ax3 = axes[2]
    df_l2 = df[(df['Scale'] == 'L2') & (df['SCGC_Ratio'] >= 0.3)
               & (df['fc_Error_Strong'] > 0) & (df['fc_Attack_Strong'] > 0)]
    ax3.hist(df_l2['fc_Error_Strong'], bins=20, alpha=0.6, color=COLORS['robust'],
             label=f'Error ($\\mu$={df_l2["fc_Error_Strong"].mean():.2f})', density=True)
    ax3.hist(df_l2['fc_Attack_Strong'], bins=20, alpha=0.6, color=COLORS['fragile'],
             label=f'Attack ($\\mu$={df_l2["fc_Attack_Strong"].mean():.2f})', density=True)
    ax3.axvline(df_l2['fc_Error_Strong'].mean(), color=COLORS['robust'],
                linestyle='--', lw=2)
    ax3.axvline(df_l2['fc_Attack_Strong'].mean(), color=COLORS['fragile'],
                linestyle='--', lw=2)
    ax3.set_xlabel('$f_c$', fontsize=20)
    ax3.set_ylabel('Density', fontsize=18)
    ax3.set_title('(c)  $f_c$ distribution', fontsize=18, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=13, bbox_to_anchor=(1.18, 1.0))
    ax3.grid(True, alpha=0.3, color=COLORS['grid'])

    plt.tight_layout()
    for fmt in ['png', 'pdf']:
        fig.savefig(OUTPUT_DIR / f'fig2.{fmt}')
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR / 'fig2.png'}")


if __name__ == '__main__':
    main()
