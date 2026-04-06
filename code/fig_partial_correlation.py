"""
fig_partial_correlation.py — Partial correlation forest plot (Fig. 4)
====================================================================
Generates a horizontal bar forest plot of zero-order vs partial correlations,
showing the independent association of 8 directed structural metrics with
fc(Attack). Uses match-level L2 data (N~4343).

Input:  data/match_summary.csv
Output: figures/fig4.png, figures/fig4.pdf

Usage:
  python code/fig_partial_correlation.py

Last modified: 2026-03-27
"""

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pingouin as pg
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
    'neutral': '#7F8C8D',
    'grid': '#EAECEE',
}

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

    METRICS = ['AvgDegree', 'Clustering', 'Assortativity', 'Reciprocity',
               'FlowHierarchy', 'OutDegreeCV', 'Mu11_Mu', 'CyclicTriadFraction']
    TARGET = 'fc_Attack_Strong'

    # Use match-level L2 data with dual filter
    df_l2 = df[(df['Scale'] == 'L2') & (df['SCGC_Ratio'] >= 0.3)
               & (df['fc_Error_Strong'] > 0) & (df['fc_Attack_Strong'] > 0)]
    df_l2 = df_l2.dropna(subset=METRICS + [TARGET])

    rows = []
    for metric in METRICS:
        r_zero, p_zero = stats.pearsonr(df_l2[metric], df_l2[TARGET])
        covars = [m for m in METRICS if m != metric]
        pc = pg.partial_corr(data=df_l2, x=metric, y=TARGET, covar=covars, method='pearson')
        r_partial = pc['r'].values[0]
        p_col = 'p-val' if 'p-val' in pc.columns else 'p_val'
        p_partial = pc[p_col].values[0]
        rows.append({'metric': metric, 'zero_order_r': r_zero, 'zero_order_p': p_zero,
                     'partial_r': r_partial, 'partial_p': p_partial,
                     'r_change': abs(r_partial) - abs(r_zero)})
    pcorr = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 6))

    metrics = pcorr['metric'].values
    y_pos = np.arange(len(metrics))
    metric_labels = {
        'AvgDegree': 'Mean Degree $\\langle k \\rangle$',
        'Clustering': 'Clustering $C$',
        'Assortativity': 'Assortativity $r$',
        'Reciprocity': 'Reciprocity $\\rho_{\\leftrightarrow}$',
        'FlowHierarchy': 'Flow Hierarchy $h$',
        'OutDegreeCV': 'Out-Degree $\\mathrm{CV}_{\\mathrm{out}}$',
        'Mu11_Mu': 'MR $\\mu_{11}/\\mu$',
        'CyclicTriadFraction': 'Cyclic Triad $f_{\\mathrm{cyc}}$',
    }

    # Zero-order r
    ax.barh(y_pos + 0.15, pcorr['zero_order_r'], height=0.3,
            color=COLORS['neutral'], alpha=0.6, label='Zero-order r')

    # Partial r
    ax.barh(y_pos - 0.15, pcorr['partial_r'], height=0.3,
            color=COLORS['epl'], alpha=0.8, label='Partial r')

    def p_label(p_val):
        if p_val < 0.001: return 'p<.001'
        elif p_val < 0.01: return f'p={p_val:.3f}'
        elif p_val < 0.05: return f'p={p_val:.3f}'
        else: return 'n.s.'

    for i, row in pcorr.iterrows():
        # Zero-order
        x_z = row['zero_order_r']
        offset = 0.02 if x_z >= 0 else -0.02
        ha = 'left' if x_z >= 0 else 'right'
        ax.text(x_z + offset, i + 0.15, f'{x_z:.3f}',
                va='center', ha=ha, fontsize=11)

        # Partial
        x_p = row['partial_r']
        p_str = p_label(row['partial_p'])
        offset = 0.02 if x_p >= 0 else -0.02
        ha = 'left' if x_p >= 0 else 'right'
        ax.text(x_p + offset, i - 0.15, f'{x_p:.3f} ({p_str})',
                va='center', ha=ha, fontsize=11, fontweight='bold')

    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([metric_labels.get(m, m) for m in metrics])
    ax.set_xlabel('Correlation with $f_c^{Attack}$', fontsize=16)
    ax.set_title('')
    ax.legend(loc='upper left', fontsize=12)
    ax.tick_params(axis='both', labelsize=14)
    ax.grid(True, axis='x', alpha=0.3, color=COLORS['grid'])
    xlim = ax.get_xlim()
    ax.set_xlim(xlim[0] - 0.15, xlim[1] + 0.12)

    plt.tight_layout()
    for fmt in ['png', 'pdf']:
        fig.savefig(OUTPUT_DIR / f'fig4.{fmt}')
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR / 'fig4.png'}")


if __name__ == '__main__':
    main()
