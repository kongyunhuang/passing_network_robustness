"""
s07_convergence.py  Monte Carlo trial-count convergence of the threshold (Fig. S3)
=================================================================================
Plots the convergence diagnostics of the random-failure threshold from the
shipped convergence experiment (200 sampled L2 networks, 10 seeds, trial counts
B in 5, 10, 20, 40, 80 and 160). Panel (a) shows the per-network dispersion
(SD across seeds) as a function of B, panel (b) the median absolute deviation
of the per-network mean threshold from the B = 160 reference. No simulation is run.

Input  data/supplementary/convergence_results.csv
Output  figures/figS3.pdf, figures/figS3.png

Usage
  python code/supplementary/s07_convergence.py

Last modified 2026-10-27
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times'],
                     'font.size': 10, 'pdf.fonttype': 42,
                     'mathtext.fontset': 'cm', 'axes.unicode_minus': False})

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / 'data' / 'supplementary'
FIG_DIR = BASE_DIR / 'figures'
df = pd.read_csv(DATA_DIR / 'convergence_results.csv')
GRID_FLOOR = 1 / 49

# Per network and per B, SD and mean across seeds
g = df.groupby(['MatchID', 'TeamID', 'n_trials'])['fc'].agg(['std', 'mean']).reset_index()
Bs = sorted(df.n_trials.unique())

sd_median = [g[g.n_trials == b]['std'].median() for b in Bs]
sd_q25 = [g[g.n_trials == b]['std'].quantile(0.25) for b in Bs]
sd_q75 = [g[g.n_trials == b]['std'].quantile(0.75) for b in Bs]
sd_p90 = [g[g.n_trials == b]['std'].quantile(0.90) for b in Bs]

# Panel (b), median absolute offset of the per-network mean from the B = 160 reference
piv = g.pivot_table(index=['MatchID', 'TeamID'], columns='n_trials', values='mean')
offset = [np.median(np.abs(piv[b] - piv[160])) for b in Bs]

# Acceptance check against Supplementary Section S7. Median SD across seeds
# at B = 5, 20, 80 and 160, and the paired difference of the per-network
# mean threshold from the B = 160 reference (mean and standard error over
# the 200 networks) at B = 20 and 80.
checks = []
for b, exp in [(5, 0.083), (20, 0.055), (80, 0.035), (160, 0.027)]:
    checks.append((f'median SD at B = {b}', sd_median[Bs.index(b)], exp))
for b, exp_mean, exp_se in [(20, -0.016, 0.003), (80, -0.002, 0.001)]:
    d = piv[b] - piv[160]
    checks.append((f'paired difference at B = {b}', d.mean(), exp_mean))
    checks.append((f'paired difference SE at B = {b}', d.std(ddof=1) / np.sqrt(len(d)), exp_se))
failed = False
for label, got, exp in checks:
    ok = abs(got - exp) <= 0.0005
    failed |= not ok
    print(f'  {label:<34} {got:+.4f} vs {exp:+.3f}  {"PASS" if ok else "FAIL"}')
if failed:
    print('GATE FAILED  computed values differ from Supplementary Section S7')
    sys.exit(1)
print(f'GATE PASS  all {len(checks)} values match Supplementary Section S7')

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
ax.set_ylabel(r'SD of $f_{c,\mathrm{L2}}^{\mathrm{Error}}$ across 10 seeds')
ax.set_title(r'(a) Per-match dispersion of $f_{c,\mathrm{L2}}$', fontsize=10.5)
ax.legend(fontsize=7.5, frameon=True)
ax.set_ylim(0, 0.145)

ax = axes[1]
ax.plot(Bs, offset, 'o-', color='#2e7d32')
ax.axhline(GRID_FLOOR, color='#e05555', ls='-.', lw=1)
ax.set_xscale('log'); ax.set_xticks(Bs); ax.set_xticklabels(Bs)
ax.set_xlabel(r'Monte Carlo trials $B$')
ax.set_ylabel(r'median $|\bar{f}_{c,\mathrm{L2}}(B)-\bar{f}_{c,\mathrm{L2}}(160)|$')
ax.set_title('(b) Deviation from high-trial reference', fontsize=10.5)

fig.tight_layout()
FIG_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG_DIR / 'figS3.pdf')
fig.savefig(FIG_DIR / 'figS3.png', dpi=200)
print('Saved figures/figS3.pdf, figures/figS3.png')
