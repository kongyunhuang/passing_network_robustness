"""
fig8_team_stability.py  Stability of team robustness (Fig. 8)
============================================================
Composite figure of the targeted-attack threshold, each panel placed with
fig.add_axes. (a) and (b) team means across the four scales for EPL and La Liga,
(c) cross-season heatmap of the 32 teams present in all three seasons with
ICC(3,1), (d) to (f) home versus away team means per season, (g) and (h) EPL
versus La Liga match-level distributions. Panels (c) to (h) use the primary L2
sample (N = 4,330). The values quoted in the article are checked, and the script
exits with status 1 on a mismatch. The ICC needs pingouin 0.5.x.

Input  data/match_summary_mc100_guarded.csv (the published analysis, 100 Monte
       Carlo trials at all scales, restricted threshold estimator)
Output  figures/fig8.png, figures/fig8.pdf

Usage
  python code/fig8_team_stability.py

Last modified 2026-10-27
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.lines import Line2D

plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 12,
    'xtick.labelsize': 11.5, 'ytick.labelsize': 11.5,
    'figure.dpi': 300, 'savefig.dpi': 300,
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
})

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / 'data' / 'match_summary_mc100_guarded.csv'
OUTPUT_DIR = BASE_DIR / 'figures'
FC = 'pc_Attack_Strong'

# Published values (Fig. 8 and the Section 3.5 text)
PUBLISHED = dict(
    n_l2=4330, n_common=32, icc=0.829,
    r_min=0.81, r_max=0.89,              # home versus away, all P < 0.001
    delta_min=0.01, delta_max=0.03,      # home minus away, 1 to 3 percentage points, P < 0.05
    dz_min=0.36, dz_max=0.78,            # Cohen's d_z of the paired differences
    fc_epl=0.422, fc_liga=0.417, fc_d=0.03,
    rec_epl=0.220, rec_liga=0.211,       # Reciprocity, P < 0.001
)

SHORT = {
    'AFC Bournemouth': 'Bournemouth', 'Brighton & Hove Albion': 'Brighton',
    'Manchester City': 'Man City', 'Manchester United': 'Man Utd',
    'Newcastle United': 'Newcastle', 'Nottingham Forest': "Nott'm Forest",
    'Tottenham Hotspur': 'Tottenham', 'West Ham United': 'West Ham',
    'Wolverhampton Wanderers': 'Wolves', 'Crystal Palace': 'C. Palace',
    'Leicester City': 'Leicester', 'Atlético Madrid': 'Atlético',
    'Real Sociedad': 'R. Sociedad', 'Real Madrid': 'R. Madrid',
    'Real Betis': 'R. Betis', 'Athletic Club': 'Ath. Bilbao',
    'Rayo Vallecano': 'Rayo', 'Celta Vigo': 'Celta',
    'Real Valladolid': 'Valladolid', 'Sheffield United': 'Sheff Utd',
    'Deportivo Alavés': 'Deportivo Al',
}


def short(t):
    return SHORT.get(t, t)


# ============================================================================
# Panel positions [left, bottom, width, height] in figure fractions
# ============================================================================
# Top block, scale sensitivity (left) and cross-season heatmap (right)
POS_A1 = [0.08, 0.58, 0.30, 0.18]   # EPL scale sensitivity
POS_A2 = [0.08, 0.33, 0.30, 0.18]   # La Liga scale sensitivity
POS_A3 = [0.535, 0.30, 0.35, 0.5]   # heatmap
POS_CB = [0.90, 0.37, 0.015, 0.32]  # colorbar

# Middle row, home versus away scatter
POS_B1 = [0.07, 0.09, 0.25, 0.15]
POS_B2 = [0.39, 0.09, 0.25, 0.15]
POS_B3 = [0.71, 0.09, 0.25, 0.15]

# Bottom row, cross-league violins
POS_C1 = [0.24, -0.11, 0.21, 0.12]
POS_C2 = [0.55, -0.11, 0.21, 0.12]


def cohen_d(x, y):
    """Cohen's d with the pooled standard deviation."""
    nx, ny = len(x), len(y)
    sp = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2))
    return (np.mean(x) - np.mean(y)) / sp


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ms = pd.read_csv(DATA_PATH)
    l2 = ms[(ms['Scale'] == 'L2') & (ms['SCGC_Ratio'] >= 0.3) & (ms['pc_Error_Strong'] > 0)].copy()

    epl_teams = set(ms[ms['League'] == 'Premier League']['Team'])

    def get_league(t):
        return 'EPL' if t in epl_teams else 'La Liga'

    ts = l2.groupby(['Team', 'Season'])[FC].mean().reset_index()
    ts['League'] = ts['Team'].apply(get_league)
    seasons = sorted(ts['Season'].unique())
    teams_per = [set(ts[ts['Season'] == s]['Team']) for s in seasons]
    common = teams_per[0] & teams_per[1] & teams_per[2]

    C_EPL = '#1B4F72'      # deep blue
    C_LIGA = '#C0392B'     # rust red
    C_ROBUST = '#1B6FA8'   # medium blue
    C_FRAGILE = '#C0392B'  # rust red

    got = dict(n_l2=len(l2), n_common=len(common))

    fig = plt.figure(figsize=(7.5, 10.5))

    # ══════════════════════════════════════════
    # (a) Scale sensitivity, EPL
    # ══════════════════════════════════════════
    ax = fig.add_axes(POS_A1)
    team_scale = ms.groupby(['Team', 'Scale', 'League'])[FC].mean().reset_index()
    df_l = team_scale[team_scale['League'] == 'Premier League']
    scales = ['L0', 'L1', 'L2', 'L3']

    for team in df_l['Team'].unique():
        dt = df_l[df_l['Team'] == team].set_index('Scale')[FC]
        if len(dt) == 4:
            ax.plot(range(4), [dt.get(s, np.nan) for s in scales],
                    color='gray', alpha=0.12, linewidth=0.6)

    mean_s = df_l.groupby('Scale')[FC].mean().reindex(scales)
    std_s = df_l.groupby('Scale')[FC].std().reindex(scales)
    ax.fill_between(range(4), mean_s - std_s, mean_s + std_s, color=C_EPL, alpha=0.15)
    ax.plot(range(4), mean_s, color=C_EPL, linewidth=2, marker='o', markersize=5, label='League Mean')

    rob = df_l.groupby('Team')[FC].mean()
    for team, color in [(rob.idxmax(), C_ROBUST), (rob.idxmin(), C_FRAGILE)]:
        dt = df_l[df_l['Team'] == team].set_index('Scale')[FC]
        if len(dt) == 4:
            ax.plot(range(4), [dt.get(s, np.nan) for s in scales],
                    color=color, linewidth=1.8, marker='s', markersize=5, label=short(team))

    ax.set_xticks(range(4))
    ax.set_xticklabels(['L0', 'L1', 'L2', 'L3'])
    ax.set_ylabel(r'$f_{c,\ell}^{Attack}$', fontsize=16)
    ax.set_title('(a)  Premier League', fontweight='bold')
    ax.legend(loc='upper left', fontsize=9, framealpha=0.8)
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.set_ylim(-0.02, 0.75)

    # ══════════════════════════════════════════
    # (b) Scale sensitivity, La Liga
    # ══════════════════════════════════════════
    ax = fig.add_axes(POS_A2)
    df_l = team_scale[team_scale['League'] != 'Premier League']

    for team in df_l['Team'].unique():
        dt = df_l[df_l['Team'] == team].set_index('Scale')[FC]
        if len(dt) == 4:
            ax.plot(range(4), [dt.get(s, np.nan) for s in scales],
                    color='gray', alpha=0.12, linewidth=0.6)

    mean_s = df_l.groupby('Scale')[FC].mean().reindex(scales)
    std_s = df_l.groupby('Scale')[FC].std().reindex(scales)
    ax.fill_between(range(4), mean_s - std_s, mean_s + std_s, color=C_LIGA, alpha=0.15)
    ax.plot(range(4), mean_s, color=C_LIGA, linewidth=2, marker='o', markersize=5, label='League Mean')

    rob = df_l.groupby('Team')[FC].mean()
    for team, color in [(rob.idxmax(), C_ROBUST), (rob.idxmin(), C_FRAGILE)]:
        dt = df_l[df_l['Team'] == team].set_index('Scale')[FC]
        if len(dt) == 4:
            ax.plot(range(4), [dt.get(s, np.nan) for s in scales],
                    color=color, linewidth=1.8, marker='s', markersize=5, label=short(team))

    ax.set_xticks(range(4))
    ax.set_xticklabels(['L0', 'L1', 'L2', 'L3'])
    ax.set_ylabel(r'$f_{c,\ell}^{Attack}$', fontsize=16)
    ax.set_title('(b)  La Liga', fontweight='bold')
    ax.legend(loc='upper left', fontsize=9, framealpha=0.8)
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.set_ylim(-0.02, 0.75)

    # ══════════════════════════════════════════
    # (c) Cross-season heatmap
    # ══════════════════════════════════════════
    ax_hm = fig.add_axes(POS_A3)
    pivot = ts[ts['Team'].isin(common)].pivot(index='Team', columns='Season', values=FC)
    pivot = pivot[seasons]
    pivot['mean'] = pivot.mean(axis=1)
    pivot = pivot.sort_values('mean', ascending=False).drop(columns='mean')
    pivot.index = [short(t) for t in pivot.index]

    im = ax_hm.imshow(pivot.values, cmap='Blues', aspect='auto', vmin=0.25, vmax=0.60)
    ax_hm.set_xticks(range(len(seasons)))
    ax_hm.set_xticklabels(seasons)
    ax_hm.set_yticks(range(len(pivot)))
    ax_hm.set_yticklabels(pivot.index, fontsize=11)

    for i in range(len(pivot)):
        for j in range(len(seasons)):
            val = pivot.iloc[i, j]
            color = 'white' if val > 0.45 else '#1B4F72'
            ax_hm.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=9, color=color)

    # ICC(3,1), two-way mixed effects, consistency, single rater (pingouin 0.5.x type 'ICC3')
    try:
        from pingouin import intraclass_corr
        icc_data = []
        for team in common:
            for s in seasons:
                val = ts[(ts['Team'] == team) & (ts['Season'] == s)][FC].values
                if len(val) > 0:
                    icc_data.append({'targets': team, 'raters': s, 'ratings': val[0]})
        icc_r = intraclass_corr(pd.DataFrame(icc_data), targets='targets',
                                raters='raters', ratings='ratings')
        icc_val = icc_r[icc_r['Type'] == 'ICC3']['ICC'].values[0]
        icc_str = f'ICC = {icc_val:.3f}'
    except Exception:
        icc_val = np.nan
        icc_str = ''
    got['icc'] = icc_val

    ax_hm.set_title(f'(c)  Cross-season ({len(common)} teams, {icc_str})', fontweight='bold')

    # colorbar in its own axes
    cax = fig.add_axes(POS_CB)
    cb = plt.colorbar(im, cax=cax)
    cb.set_label(r'$f_{c,\mathrm{L2}}^{Attack}$', fontsize=16)

    # ══════════════════════════════════════════
    # (d) to (f) Home versus away scatter
    # ══════════════════════════════════════════
    ha_r, ha_pr, ha_diff, ha_p, ha_dz = [], [], [], [], []
    mn, mx = 0.18, 0.65
    for si, (season, pos) in enumerate(zip(seasons, [POS_B1, POS_B2, POS_B3])):
        ax = fig.add_axes(pos)
        sl = l2[l2['Season'] == season]
        ts_hv = sl.groupby(['Team', 'Venue'])[FC].mean().reset_index()
        home = ts_hv[ts_hv['Venue'] == 'Home'].set_index('Team')[FC]
        away = ts_hv[ts_hv['Venue'] == 'Away'].set_index('Team')[FC]
        common_ha = sorted(set(home.index) & set(away.index))

        for t in common_ha:
            c = C_EPL if get_league(t) == 'EPL' else C_LIGA
            ax.scatter(home[t], away[t], c=c, s=22, alpha=0.75,
                       edgecolors='white', linewidth=0.4, zorder=3)
        ax.plot([mn, mx], [mn, mx], '--', color='#999', lw=0.8, alpha=0.5)

        r, p_r = stats.pearsonr([home[t] for t in common_ha],
                                [away[t] for t in common_ha])
        _, p_paired = stats.ttest_rel([home[t] for t in common_ha],
                                      [away[t] for t in common_ha])
        diff = np.mean([home[t] for t in common_ha]) - np.mean([away[t] for t in common_ha])
        d_paired = np.array([home[t] - away[t] for t in common_ha])
        ha_r.append(r); ha_pr.append(p_r); ha_diff.append(diff); ha_p.append(p_paired)
        ha_dz.append(d_paired.mean() / d_paired.std(ddof=1))

        ax.text(0.05, 0.95,
                f'r = {r:.3f}\nΔ = {diff:+.3f}\n' + ('p < 0.001' if p_paired < 0.001 else f'p = {p_paired:.3f}'),
                transform=ax.transAxes, fontsize=10, va='top')

        ax.set_xlim(mn, mx)
        ax.set_ylim(mn, mx)
        ax.set_aspect('equal')
        panel = chr(ord('d') + si)  # d, e, f
        ax.set_title(f'({panel})  {season}', fontweight='bold')
        if si == 0:
            ax.set_ylabel(r'Away $f_{c,\mathrm{L2}}^{Attack}$')
        else:
            ax.set_yticklabels([])
        ax.set_xlabel(r'Home $f_{c,\mathrm{L2}}^{Attack}$')
        if si == 2:
            ax.legend(
                [Line2D([0], [0], marker='o', color='w', markerfacecolor=C_EPL, ms=5),
                 Line2D([0], [0], marker='o', color='w', markerfacecolor=C_LIGA, ms=5)],
                ['EPL', 'La Liga'], fontsize=9, loc='lower right')
    got.update(r_min=min(ha_r), r_max=max(ha_r), dz_min=min(ha_dz), dz_max=max(ha_dz))

    # ══════════════════════════════════════════
    # (g) and (h) Cross-league violins
    # ══════════════════════════════════════════
    epl_data = l2[l2['League'] == 'Premier League']
    liga_data = l2[l2['League'] != 'Premier League']

    metrics_c = [
        (r'$f_{c,\mathrm{L2}}^{\mathrm{Attack}}$', FC, '(g)'),
        ('Reciprocity $\\rho_{\\leftrightarrow}$', 'Reciprocity', '(h)'),
    ]

    league_p = {}
    for mi, (label, col, pid, pos) in enumerate(
        [(m[0], m[1], m[2], p) for m, p in zip(metrics_c, [POS_C1, POS_C2])]
    ):
        ax = fig.add_axes(pos)
        epl_v = epl_data[col].dropna().values
        liga_v = liga_data[col].dropna().values

        for data, xpos, c in [(epl_v, 0, C_EPL), (liga_v, 1, C_LIGA)]:
            parts = ax.violinplot([data], positions=[xpos], showmeans=False,
                                  showmedians=False, showextrema=False, widths=0.65)
            for pc in parts['bodies']:
                pc.set_facecolor(c)
                pc.set_alpha(0.5)
                pc.set_edgecolor(c)
                pc.set_linewidth(0.8)

            avg = np.mean(data)
            ax.plot([xpos - 0.15, xpos + 0.15], [avg, avg], color='black', lw=2, zorder=5)
            q75 = np.percentile(data, 75)
            ax.text(xpos, avg + (q75 - avg) * 0.2, f'{avg:.3f}',
                    ha='center', fontsize=11, fontweight='bold')

        _, p = stats.ttest_ind(epl_v, liga_v, equal_var=False)
        league_p[col] = p
        if col == FC:
            got.update(fc_epl=np.mean(epl_v), fc_liga=np.mean(liga_v), fc_d=cohen_d(epl_v, liga_v))
        else:
            got.update(rec_epl=np.mean(epl_v), rec_liga=np.mean(liga_v))
        if p < 0.001:
            p_str = 'p < 0.001'
        elif p < 0.05:
            p_str = f'p = {p:.3f}'
        else:
            p_str = 'n.s.'
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['EPL', 'La Liga'])
        ax.set_title(f'{pid}  {label}  ({p_str})', fontweight='bold')

        all_data = np.concatenate([epl_v, liga_v])
        p5, p95 = np.percentile(all_data, [2, 98])
        margin = (p95 - p5) * 0.2
        ax.set_ylim(p5 - margin, p95 + margin)
        if mi == 0:
            ax.set_ylabel('Value')

    out = OUTPUT_DIR / 'fig8.png'
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(str(out).replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Saved: {out}")
    print(f"Saved: {str(out).replace('.png', '.pdf')}")
    plt.close()

    # ══════════════════════════════════════════
    # Acceptance check
    # ══════════════════════════════════════════
    print('\nCHECK against the published Fig. 8 and Section 3.5')
    ok_all = True

    def report(label, ok, text):
        print(f'  {label:<34} {text} -> {"PASS" if ok else "FAIL"}')
        return ok

    def near(key, decimals):
        tol = 0.5 * 10 ** (-decimals)
        ok = abs(got[key] - PUBLISHED[key]) <= tol + 1e-12
        return report(key, ok, f'{got[key]:.{decimals + 1}f} vs {PUBLISHED[key]:.{decimals}f}')

    ok_all &= report('n_l2', got['n_l2'] == PUBLISHED['n_l2'], f"{got['n_l2']} vs {PUBLISHED['n_l2']}")
    ok_all &= report('n_common', got['n_common'] == PUBLISHED['n_common'],
                     f"{got['n_common']} vs {PUBLISHED['n_common']}")
    if np.isnan(got['icc']):
        ok_all &= report('icc', False, 'not computed (pingouin 0.5.x is required)')
    else:
        ok_all &= near('icc', 3)
    for key in ('r_min', 'r_max', 'dz_min', 'dz_max', 'fc_d'):
        ok_all &= near(key, 2)
    for key in ('fc_epl', 'fc_liga', 'rec_epl', 'rec_liga'):
        ok_all &= near(key, 3)
    ok_all &= report('home versus away r, all P < 0.001', max(ha_pr) < 0.001, f'max P {max(ha_pr):.1e}')
    ok_all &= report('home minus away within 0.01 to 0.03',
                     all(PUBLISHED['delta_min'] <= d <= PUBLISHED['delta_max'] for d in ha_diff),
                     ', '.join(f'{d:+.4f}' for d in ha_diff))
    ok_all &= report('paired t test, all P < 0.05', max(ha_p) < 0.05, f'max P {max(ha_p):.3f}')
    ok_all &= report('Reciprocity Welch P < 0.001', league_p['Reciprocity'] < 0.001,
                     f"P {league_p['Reciprocity']:.1e}")
    ok_all &= report('attack threshold Welch n.s.', league_p[FC] >= 0.05, f'P {league_p[FC]:.3f}')
    if not ok_all:
        print('CHECK FAILED, the values above marked FAIL differ from the article.')
        sys.exit(1)
    print('CHECK ALL PASS')


if __name__ == '__main__':
    main()
