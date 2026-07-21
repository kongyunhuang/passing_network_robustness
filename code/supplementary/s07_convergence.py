"""
s07_convergence.py — Monte Carlo trial-count convergence analysis (Supplementary S7)
====================================================================================
Re-plots the trial-count convergence diagnostics from the archived convergence
experiment (200 sampled L2 networks x 10 seeds x trial counts B in
{5, 10, 20, 40, 80, 160}). Panel (a) shows the per-match dispersion (SD across
seeds) of the random-removal critical threshold f_c as a function of B; panel
(b) shows the median absolute deviation of the per-match mean f_c from the
B = 160 reference. Pure re-plot: no simulation is re-run.

Input:  ../../data/supplementary/convergence_results.csv (read-only)
Output: convergence_curve.pdf / .png (written next to this script)

Usage:
    cd code/supplementary
    python s07_convergence.py

Last modified: 2026-07-21
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times'],
                     'font.size': 10, 'pdf.fonttype': 42,
                     'mathtext.fontset': 'cm', 'axes.unicode_minus': False})

_SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = _SCRIPT_DIR.parent.parent / 'data' / 'supplementary'
df = pd.read_csv(DATA_DIR / 'convergence_results.csv')
GRID_FLOOR = 1 / 49

# Per match and per B: SD and mean across seeds
g = df.groupby(['MatchID', 'TeamID', 'n_trials'])['fc'].agg(['std', 'mean']).reset_index()
Bs = sorted(df.n_trials.unique())

sd_median = [g[g.n_trials == b]['std'].median() for b in Bs]
sd_q25 = [g[g.n_trials == b]['std'].quantile(0.25) for b in Bs]
sd_q75 = [g[g.n_trials == b]['std'].quantile(0.75) for b in Bs]
sd_p90 = [g[g.n_trials == b]['std'].quantile(0.90) for b in Bs]

# Panel (b): median absolute offset of the per-match mean from the B=160 reference
piv = g.pivot_table(index=['MatchID', 'TeamID'], columns='n_trials', values='mean')
offset = [np.median(np.abs(piv[b] - piv[160])) for b in Bs]

# Acceptance gate: reproduce the SD medians quoted in Supplementary Section S7
assert abs(sd_median[Bs.index(5)] - 0.083) < 0.001, sd_median
assert abs(sd_median[Bs.index(20)] - 0.055) < 0.001
assert abs(sd_median[Bs.index(160)] - 0.027) < 0.001
assert abs(sd_median[Bs.index(80)] - 0.035) < 0.001
print('GATE PASS: median SD at B = 5/20/80/160 =',
      [round(sd_median[Bs.index(b)], 4) for b in (5, 20, 80, 160)])

fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
ax = axes[0]
ax.fill_between(Bs, sd_q25, sd_q75, alpha=0.25, color='#3b7cb8', label='IQR (25–75%)')
ax.plot(Bs, sd_median, 'o-', color='#1f77b4', label='Median SD across seeds')
ax.plot(Bs, sd_p90, 's--', color='#e8862d', label='90th percentile')
ref = sd_median[0] * np.sqrt(Bs[0]) / np.sqrt(np.array(Bs, float))
ax.plot(Bs, ref, ':', color='grey', label=r'$\propto 1/\sqrt{B}$ reference')
ax.axhline(GRID_FLOOR, color='#8b1a1a', ls='-.', lw=1,
           label=r'$f$-grid resolution ($1/49\approx0.020$)')
ax.set_xscale('log'); ax.set_xticks(Bs); ax.set_xticklabels(Bs)
ax.set_xlabel(r'Monte Carlo trials $B$')
ax.set_ylabel(r'SD of $f_c^{\mathrm{Error}}$ across 10 seeds')
ax.set_title(r'(a) Per-match dispersion of $f_c$', fontsize=10.5)
ax.legend(fontsize=7.5, frameon=True)
ax.set_ylim(0, 0.145)

ax = axes[1]
ax.plot(Bs, offset, 'o-', color='#2e7d32')
ax.axhline(GRID_FLOOR, color='#e05555', ls='-.', lw=1)
ax.set_xscale('log'); ax.set_xticks(Bs); ax.set_xticklabels(Bs)
ax.set_xlabel(r'Monte Carlo trials $B$')
ax.set_ylabel(r'median $|\bar{f}_c(B)-\bar{f}_c(160)|$')
ax.set_title('(b) Deviation from high-trial reference', fontsize=10.5)

fig.tight_layout()
fig.savefig(_SCRIPT_DIR / 'convergence_curve.pdf')
fig.savefig(_SCRIPT_DIR / 'convergence_curve.png', dpi=200)
print('saved convergence_curve.pdf/.png')
