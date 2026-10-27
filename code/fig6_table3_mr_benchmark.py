"""
fig6_table3_mr_benchmark.py  Directed MR benchmark versus Monte Carlo simulation (Fig. 6, Table 3)
==========================================================================================
Compares the directed Molloy-Reed (MR) analytical threshold of Eq. (1) with the
Monte Carlo random-failure threshold, match by match. Table 3 covers the four
scales L0 to L3 (4,560 team-match networks each) and keeps the networks with a
positive analytical threshold (mu11/mu > 2). Fig. 6 is the L2 hexbin of the
4,325 valid networks. Every value printed in Table 3 and in the Fig. 6
annotation is compared with the article, and the script exits with status 1 on
a mismatch.

Input  data/match_summary_mc100_guarded.csv (the published analysis, 100 Monte
       Carlo trials at all scales, restricted threshold estimator)
Output  figures/fig6.png, figures/fig6.pdf, results/table3.csv

Usage
  python code/fig6_table3_mr_benchmark.py

Last modified 2026-10-27
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

# ============================================================================
# Paths
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / 'data' / 'match_summary_mc100_guarded.csv'
OUTPUT_DIR = BASE_DIR / 'figures'
RESULTS_DIR = BASE_DIR / 'results'

# Column names in the match-level results
MR_COL = 'pc_Theory'          # directed MR threshold f_c^dMR, Eq. (1)
SIM_COL = 'pc_Error_Strong'   # Monte Carlo random-failure threshold f_c^Error

SCALES = ['L0', 'L1', 'L2', 'L3']

# ============================================================================
# Published values
# ============================================================================
# Table 3 of the article. L0 has no network with a positive analytical threshold.
PUBLISHED_TABLE3 = {
    'L1': dict(n=1305, pct=29, theory=0.259, sim=0.416, bias=-0.157, mae=0.176, r=0.599),
    'L2': dict(n=4325, pct=95, theory=0.601, sim=0.601, bias=-0.001, mae=0.064, r=0.809),
    'L3': dict(n=4554, pct=99.9, theory=0.794, sim=0.804, bias=-0.011, mae=0.041, r=0.754),
}
# Number of decimals printed for the valid percentage of each scale
PCT_DECIMALS = {'L1': 0, 'L2': 0, 'L3': 1}
# Fig. 6 annotation and caption (N = 4,325)
PUBLISHED_FIG6 = dict(n=4325, r=0.809, r2=0.654, bias=-0.001, mae=0.064)

# ============================================================================
# Publication style
# ============================================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 12,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.labelsize': 13,
    'axes.labelweight': 'medium',
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 1.2,
    'axes.edgecolor': '#333333',
    'axes.unicode_minus': False,
    'axes.grid': False,
    'figure.dpi': 150,
    'figure.facecolor': 'white',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
    'savefig.pad_inches': 0.15,
    'mathtext.fontset': 'cm',
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.major.size': 5,
    'ytick.major.size': 5,
})


# ============================================================================
# Table 3
# ============================================================================
def table3_rows(df):
    """Scale comparison of the analytical and simulated thresholds.

    Valid networks have a positive analytical threshold (equivalent to
    mu11/mu > 2). Bias = mean(theory - sim), MAE = mean|theory - sim|,
    r = Pearson(theory, sim), as defined in the Table 3 footnote.
    """
    valid = df[df[MR_COL].notna() & (df[MR_COL] > 0)]
    rows = []
    for s in SCALES:
        sub = valid[valid['Scale'] == s]
        n_total = int((df['Scale'] == s).sum())
        n = len(sub)
        if n == 0:
            rows.append(dict(Scale=s, N_valid=0, N_total=n_total, pct_valid=0.0,
                             theory_mean=np.nan, theory_sd=np.nan,
                             sim_mean=np.nan, sim_sd=np.nan,
                             bias=np.nan, mae=np.nan, r=np.nan))
            continue
        t, m = sub[MR_COL], sub[SIM_COL]
        gap = t - m
        r, _ = sp_stats.pearsonr(t, m)
        rows.append(dict(Scale=s, N_valid=n, N_total=n_total,
                         pct_valid=100.0 * n / n_total,
                         theory_mean=t.mean(), theory_sd=t.std(ddof=1),
                         sim_mean=m.mean(), sim_sd=m.std(ddof=1),
                         bias=gap.mean(), mae=gap.abs().mean(), r=r))
    return rows


def format_table3(rows):
    hdr = (f"{'Scale':<5} {'N_valid':>8} {'(%)':>7} {'theory':>8} {'sim':>8} "
           f"{'Bias':>8} {'MAE':>7} {'r':>7}")
    lines = [hdr, '-' * len(hdr)]
    for row in rows:
        if row['N_valid'] == 0:
            lines.append(f"{row['Scale']:<5} {0:>8} {'n/a':>7} {'n/a':>8} {'n/a':>8} "
                         f"{'n/a':>8} {'n/a':>7} {'n/a':>7}")
        else:
            lines.append(
                f"{row['Scale']:<5} {row['N_valid']:>8} {row['pct_valid']:>6.1f}% "
                f"{row['theory_mean']:>8.3f} {row['sim_mean']:>8.3f} {row['bias']:>+8.3f} "
                f"{row['mae']:>7.3f} {row['r']:>7.3f}")
    return '\n'.join(lines)


# ============================================================================
# Fig. 6
# ============================================================================
def fig6_mr_benchmark(df_valid):
    """L2 density scatter of the directed MR threshold against the simulated one."""
    df_l2 = df_valid[(df_valid['Scale'] == 'L2')]

    fig, ax = plt.subplots(figsize=(7, 6))

    # Density scatter (hexbin)
    hb = ax.hexbin(df_l2[MR_COL], df_l2[SIM_COL],
                   gridsize=35, cmap='Blues', mincnt=1, zorder=2)
    cb = plt.colorbar(hb, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label('Count', fontsize=12)

    # Identity line
    lims = [0, 1.05]
    ax.plot(lims, lims, '--', color='#333333', linewidth=1.2, alpha=0.7, zorder=10)

    # Stats (residual = dMR - simulation, the Table 3 definition)
    r_val, p_val = sp_stats.pearsonr(df_l2[MR_COL], df_l2[SIM_COL])
    residuals = df_l2[MR_COL] - df_l2[SIM_COL]
    bias = residuals.mean()
    mae = residuals.abs().mean()
    r2 = r_val ** 2

    # Annotation
    ax.text(0.05, 0.95,
            f'r = {r_val:.3f}\n$R^2$ = {r2:.3f}\nBias = ${bias:+.3f}$\nMAE = {mae:.3f}',
            transform=ax.transAxes, fontsize=13, va='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8, edgecolor='none'))

    ax.set_xlabel(r'$f_{c,\mathrm{L2}}^{\mathrm{dMR}}$', fontsize=14)
    ax.set_ylabel(r'$f_{c,\mathrm{L2}}^{\mathrm{Error}}$', fontsize=14)
    ax.set_title('')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal')
    ax.tick_params(axis='both', labelsize=12)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'fig6.pdf')
    fig.savefig(OUTPUT_DIR / 'fig6.png', dpi=300)
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'fig6.pdf'}")
    print(f"Saved: {OUTPUT_DIR / 'fig6.png'}")
    return dict(n=len(df_l2), r=r_val, r2=r2, bias=bias, mae=mae)


# ============================================================================
# Acceptance check
# ============================================================================
def check(label, got, expected, decimals):
    """Compare with the published value at half a unit of the printed precision."""
    tol = 0.5 * 10 ** (-decimals)
    ok = abs(got - expected) <= tol + 1e-12
    print(f"  {label:<22} {got:>10.{decimals + 1}f} vs {expected:>8.{decimals}f} "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


def check_count(label, got, expected):
    ok = got == expected
    print(f"  {label:<22} {got:>10d} vs {expected:>8d} -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    assert len(df) == 18240, f'unexpected number of networks: {len(df)}'
    df_valid = df[df[MR_COL].notna() & (df[MR_COL] > 0)].copy()

    # ---------- Table 3 ----------
    rows = table3_rows(df)
    print('Table 3  Directed MR analytical threshold versus Monte Carlo simulation threshold')
    print(format_table3(rows))
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / 'table3.csv', index=False, float_format='%.6f')
    print(f"Saved: {RESULTS_DIR / 'table3.csv'}")

    # ---------- Fig. 6 ----------
    print()
    fig6 = fig6_mr_benchmark(df_valid)

    # ---------- Acceptance check ----------
    print('\nCHECK against the published Table 3 and Fig. 6')
    ok_all = True
    by_scale = {r['Scale']: r for r in rows}
    ok_all &= check_count('L0 N_valid', by_scale['L0']['N_valid'], 0)
    for s, exp in PUBLISHED_TABLE3.items():
        got = by_scale[s]
        ok_all &= check_count(f'{s} N_valid', got['N_valid'], exp['n'])
        ok_all &= check(f'{s} N_valid (%)', got['pct_valid'], exp['pct'], PCT_DECIMALS[s])
        ok_all &= check(f'{s} mean dMR', got['theory_mean'], exp['theory'], 3)
        ok_all &= check(f'{s} mean Error', got['sim_mean'], exp['sim'], 3)
        ok_all &= check(f'{s} Bias', got['bias'], exp['bias'], 3)
        ok_all &= check(f'{s} MAE', got['mae'], exp['mae'], 3)
        ok_all &= check(f'{s} r', got['r'], exp['r'], 3)
    ok_all &= check_count('Fig. 6 N', fig6['n'], PUBLISHED_FIG6['n'])
    for key in ('r', 'r2', 'bias', 'mae'):
        ok_all &= check(f'Fig. 6 {key}', fig6[key], PUBLISHED_FIG6[key], 3)
    if not ok_all:
        print('CHECK FAILED, the values above marked FAIL differ from the article.')
        sys.exit(1)
    print('CHECK ALL PASS')


if __name__ == '__main__':
    main()
