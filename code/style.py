"""
style.py  Shared plotting style and helpers for Figures 3 to 5
=============================================================
Colour palette, matplotlib settings, team short names and the regression
helper used by fig3_error_vs_attack.py, fig4_structural_factors.py and
fig5_partial_correlation.py, plus the loader for the match-level results of
the published analysis.

Input  data/match_summary_mc100_guarded.csv
Output  none (imported by the figure scripts)

Last modified 2026-10-27
"""

import matplotlib
matplotlib.use('Agg')
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
PUBLISHED_RESULTS = DATA_DIR / 'match_summary_mc100_guarded.csv'
OUTPUT_DIR = BASE_DIR / 'figures'

# Color scheme — unified, colorblind-friendly, top-journal quality
# ============================================================================
COLORS = {
    # Arctic Academic palette — deep blue + rust red, cold tone
    # League
    'epl': '#1B4F72',       # deep blue
    'laliga': '#C0392B',    # rust red
    # Robustness semantic
    'robust': '#1B4F72',    # deep blue (robust)
    'fragile': '#C0392B',   # rust red (fragile)
    'neutral': '#7F8C8D',   # grey
    # Scale gradient (sequential blue)
    'L0': '#0B3D6B',
    'L1': '#1B6FA8',
    'L2': '#5DADE2',
    'L3': '#AED6F1',
    # Home/Away
    'home': '#1B4F72',      # deep blue
    'away': '#C0392B',      # rust red
    # Error/Attack bars
    'error_bar': '#2980B9', # blue
    'attack_bar': '#C0392B', # rust red
    # Partial correlation
    'zero_order': '#AED6F1', # light blue
    'partial': '#1B4F72',    # deep blue
    # Null model
    'null': '#BDC3C7',
    'null_ci': '#D5D8DC',
    # Synthetic baselines
    'football': '#2C3E50',
    'syn_er': '#2980B9',    # blue
    'syn_ba': '#C0392B',    # rust red
    'syn_ws': '#E69F00',    # warm amber/sand
    'syn_rgg': '#85C1E9',   # light blue
    # Misc
    'grid': '#EAECEE',
    'annotation': '#2C3E50',
    'ci_band': '#D6DBDF',
}

# ============================================================================
# Global style — publication-ready (KBS / KDD level)
# ============================================================================
plt.rcParams.update({
    # Font — Times New Roman for KBS/KDD journals
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.titleweight': 'bold',
    'axes.labelsize': 16,
    'axes.labelweight': 'medium',
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 12,
    'legend.title_fontsize': 13,
    # Spines & axes
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 1.2,
    'axes.edgecolor': '#333333',
    'axes.unicode_minus': False,
    # Ticks
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    # Grid (subtle)
    'axes.grid': False,
    # Figure
    'figure.dpi': 150,
    'figure.facecolor': 'white',
    'figure.edgecolor': 'white',
    # Save
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
    'savefig.pad_inches': 0.15,
    # Lines
    'lines.linewidth': 1.8,
    'lines.markersize': 7,
    # Legend
    'legend.framealpha': 0.9,
    'legend.edgecolor': '#CCCCCC',
    # Math text: use same serif font
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Times New Roman',
    'mathtext.it': 'Times New Roman:italic',
    'mathtext.bf': 'Times New Roman:bold',
})


# ============================================================================
# Helper functions
# ============================================================================

TEAM_SHORT_MAP = {
    "Manchester United": "Man Utd", "Manchester City": "Man City",
    "Tottenham Hotspur": "Spurs", "Brighton & Hove Albion": "Brighton",
    "Wolverhampton Wanderers": "Wolves", "Newcastle United": "Newcastle",
    "West Ham United": "West Ham", "Nottingham Forest": "Nott'm Forest",
    "AFC Bournemouth": "Bournemouth", "Sheffield United": "Sheff Utd",
    "Leicester City": "Leicester", "Leeds United": "Leeds",
    "Crystal Palace": "C. Palace", "Luton Town": "Luton",
    "Ipswich Town": "Ipswich", "Southampton": "Southampton",
    "Real Madrid": "R. Madrid", "FC Barcelona": "Barcelona",
    "Atletico Madrid": "Atlético", "Atlético Madrid": "Atlético",
    "Real Sociedad": "R. Sociedad", "Athletic Club": "Ath. Bilbao",
    "Villarreal CF": "Villarreal", "Sevilla FC": "Sevilla",
    "CA Osasuna": "Osasuna", "UD Las Palmas": "Las Palmas",
    "Granada CF": "Granada", "CD Leganes": "Leganés",
    "Deportivo Alaves": "Alavés", "Real Valladolid CF": "Valladolid",
    "Real Valladolid": "Valladolid",
    "Elche CF": "Elche", "RCD Espanyol": "Espanyol",
    "Real Betis": "Betis", "Valencia CF": "Valencia",
    "Celta Vigo": "Celta", "Getafe CF": "Getafe",
    "Girona FC": "Girona", "Rayo Vallecano": "Rayo",
    "RCD Mallorca": "Mallorca", "Cadiz CF": "Cádiz",
    "UD Almeria": "Almería", "Almería": "Almería",
}


def shorten_name(name: str) -> str:
    return TEAM_SHORT_MAP.get(name, name[:12] if len(name) > 12 else name)


def sig_stars(p: float) -> str:
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    return ''


def add_regression(ax, x, y, color='gray', linewidth=2.2, show_label=True,
                   show_ci=True, ci_alpha=0.12):
    """Add regression line with 95% CI band and r/p annotation. Returns (r, p)."""
    mask = ~(np.isnan(x) | np.isnan(y))
    x_c, y_c = np.array(x)[mask], np.array(y)[mask]
    if len(x_c) < 5:
        return np.nan, np.nan

    r, p = stats.pearsonr(x_c, y_c)
    slope, intercept, _, _, stderr = stats.linregress(x_c, y_c)
    x_line = np.linspace(x_c.min(), x_c.max(), 100)
    y_line = slope * x_line + intercept

    line_color = COLORS['robust'] if r > 0 else COLORS['fragile']
    if color != 'gray':
        line_color = color

    label = f'r = {r:.3f}{sig_stars(p)}' if show_label else None
    ax.plot(x_line, y_line, color=line_color, linewidth=linewidth,
            linestyle='-', label=label, zorder=3)

    # 95% confidence band
    if show_ci and len(x_c) > 5:
        n = len(x_c)
        x_mean = np.mean(x_c)
        se = np.sqrt(np.sum((y_c - slope * x_c - intercept)**2) / (n - 2))
        t_val = stats.t.ppf(0.975, n - 2)
        ci = t_val * se * np.sqrt(1/n + (x_line - x_mean)**2 / np.sum((x_c - x_mean)**2))
        ax.fill_between(x_line, y_line - ci, y_line + ci,
                        color=line_color, alpha=ci_alpha, zorder=1)

    return r, p


def load_published_matches(path=PUBLISHED_RESULTS):
    """Match-level results of the published analysis with team short names."""
    df = pd.read_csv(path)
    df['TeamShort'] = df['Team'].apply(shorten_name)
    df['Scale'] = pd.Categorical(df['Scale'], categories=['L0', 'L1', 'L2', 'L3'], ordered=True)
    df['Season'] = pd.Categorical(df['Season'], categories=sorted(df['Season'].unique()), ordered=True)
    return df


def primary_l2(df):
    """Primary L2 sample (N = 4,330), initial SCGC >= 0.3 and both thresholds > 0."""
    return df[(df['Scale'] == 'L2') & (df['SCGC_Ratio'] >= 0.3)
              & (df['pc_Error_Strong'] > 0) & (df['pc_Attack_Strong'] > 0)]


def save(fig, name):
    """Write figures/<name>.png and figures/<name>.pdf."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for fmt in ['png', 'pdf']:
        fig.savefig(OUTPUT_DIR / f"{name}.{fmt}")
    plt.close(fig)
    print(f"  Saved: figures/{name}.pdf")
