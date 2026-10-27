"""
fig5_partial_correlation.py  Zero-order versus partial correlations (Fig. 5)
==========================================================================
For each of the eight directed structural metrics, the zero-order Pearson
correlation with the L2 Attack threshold and the partial correlation
controlling for the other seven (pingouin). Primary L2 sample (N = 4,330).
Requires pingouin 0.5.x.

Input  data/match_summary_mc100_guarded.csv
Output  figures/fig5.png, figures/fig5.pdf

Usage
  python code/fig5_partial_correlation.py

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
    """Forest plot showing zero-order vs partial correlations (match-level)."""

    import pingouin as pg

    METRICS_14 = ['AvgDegree', 'Clustering', 'Assortativity', 'Reciprocity',
                   'FlowHierarchy', 'OutDegreeCV', 'Mu11_Mu', 'CyclicTriadFraction']
    TARGET_14 = 'pc_Attack_Strong'

    # Use match-level L2 data with dual filter
    df_l2 = df[(df['Scale'] == 'L2') & (df['SCGC_Ratio'] >= 0.3)
                    & (df['pc_Error_Strong'] > 0) & (df['pc_Attack_Strong'] > 0)]
    df_l2 = df_l2.dropna(subset=METRICS_14 + [TARGET_14])

    rows = []
    for metric in METRICS_14:
        r_zero, p_zero = stats.pearsonr(df_l2[metric], df_l2[TARGET_14])
        covars = [m for m in METRICS_14 if m != metric]
        pc = pg.partial_corr(data=df_l2, x=metric, y=TARGET_14, covar=covars, method='pearson')
        r_partial = pc['r'].values[0]
        # pingouin names the column 'p-val' or 'p_val' depending on version
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

    # Plot zero-order r
    ax.barh(y_pos + 0.15, pcorr['zero_order_r'], height=0.3,
           color=COLORS['neutral'], alpha=0.6, label='Zero-order r')

    # Plot partial r
    ax.barh(y_pos - 0.15, pcorr['partial_r'], height=0.3,
           color=COLORS['epl'], alpha=0.8, label='Partial r')

    # Add significance markers (use p values instead of ***)
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
    ax.set_xlabel(r'Correlation with $f_{c,\mathrm{L2}}^{Attack}$', fontsize=16)
    ax.set_title('')
    ax.legend(loc='upper left', fontsize=12)
    ax.tick_params(axis='both', labelsize=14)
    ax.grid(True, axis='x', alpha=0.3, color=COLORS['grid'])
    # Expand x-axis to avoid label clipping
    xlim = ax.get_xlim()
    ax.set_xlim(xlim[0] - 0.15, xlim[1] + 0.12)

    plt.tight_layout()
    save(fig, 'fig5')


if __name__ == '__main__':
    main()
