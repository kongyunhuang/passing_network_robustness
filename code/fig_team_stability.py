"""
fig_team_stability.py — Team robustness stability composite figure (Fig. 7)
===========================================================================
8-panel composite figure using fig.add_axes for independent positioning:
(a1-a2) Scale sensitivity: EPL / La Liga
(c)     Cross-season heatmap (ICC annotated)
(d-f)   Home/Away scatter plots (three seasons)
(g-h)   Cross-league violin plots

Input:  data/match_summary.csv
Output: figures/fig7.png, figures/fig7.pdf

Usage:
  python code/fig_team_stability.py

Last modified: 2026-03-27
"""

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.lines import Line2D
from pathlib import Path

# ============================================================================
# Paths
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / 'data' / 'match_summary.csv'
OUTPUT_DIR = BASE_DIR / 'figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FC = 'fc_Attack_Strong'

# ============================================================================
# Publication style
# ============================================================================
plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 12,
    'xtick.labelsize': 11.5, 'ytick.labelsize': 11.5,
    'figure.dpi': 300, 'savefig.dpi': 300,
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
})

SHORT = {
    'AFC Bournemouth': 'Bournemouth', 'Brighton & Hove Albion': 'Brighton',
    'Manchester City': 'Man City', 'Manchester United': 'Man Utd',
    'Newcastle United': 'Newcastle', 'Nottingham Forest': "Nott'm Forest",
    'Tottenham Hotspur': 'Tottenham', 'West Ham United': 'West Ham',
    'Wolverhampton Wanderers': 'Wolves', 'Crystal Palace': 'C. Palace',
    'Leicester City': 'Leicester', 'Atletico Madrid': 'Atletico',
    'Real Sociedad': 'R. Sociedad', 'Real Madrid': 'R. Madrid',
    'Real Betis': 'R. Betis', 'Athletic Club': 'Ath. Bilbao',
    'Rayo Vallecano': 'Rayo', 'Celta Vigo': 'Celta',
    'Real Valladolid': 'Valladolid', 'Sheffield United': 'Sheff Utd',
    'Deportivo Alaves': 'Deportivo Al',
}


def short(t):
    return SHORT.get(t, t)


# Panel positions [left, bottom, width, height]
POS_A1 = [0.08, 0.58, 0.30, 0.18]
POS_A2 = [0.08, 0.33, 0.30, 0.18]
POS_A3 = [0.535, 0.30, 0.35, 0.5]
POS_CB = [0.90, 0.37, 0.015, 0.32]
POS_B1 = [0.07, 0.09, 0.25, 0.15]
POS_B2 = [0.39, 0.09, 0.25, 0.15]
POS_B3 = [0.71, 0.09, 0.25, 0.15]
POS_C1 = [0.24, -0.11, 0.21, 0.12]
POS_C2 = [0.55, -0.11, 0.21, 0.12]


def main():
    ms = pd.read_csv(DATA_PATH)
    l2 = ms[(ms['Scale'] == 'L2') & (ms['SCGC_Ratio'] >= 0.3) & (ms['fc_Error_Strong'] > 0)].copy()

    epl_teams = set(ms[ms['League'] == 'Premier League']['Team'])

    def get_league(t):
        return 'EPL' if t in epl_teams else 'La Liga'

    ts = l2.groupby(['Team', 'Season'])[FC].mean().reset_index()
    ts['League'] = ts['Team'].apply(get_league)
    seasons = sorted(ts['Season'].unique())
    teams_per = [set(ts[ts['Season'] == s]['Team']) for s in seasons]
    common = teams_per[0] & teams_per[1] & teams_per[2]

    C_EPL = '#1B4F72'
    C_LIGA = '#C0392B'
    C_ROBUST = '#1B6FA8'
    C_FRAGILE = '#C0392B'

    fig = plt.figure(figsize=(7.5, 10.5))

    # (a1) Scale sensitivity - EPL
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
    ax.set_ylabel('$f_c^{Attack}$', fontsize=16)
    ax.set_title('(a)  Premier League', fontweight='bold')
    ax.legend(loc='upper left', fontsize=9, framealpha=0.8)
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.set_ylim(-0.02, 0.75)

    # (a2) Scale sensitivity - La Liga
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
    ax.set_ylabel('$f_c^{Attack}$', fontsize=16)
    ax.set_title('(b)  La Liga', fontweight='bold')
    ax.legend(loc='upper left', fontsize=9, framealpha=0.8)
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.set_ylim(-0.02, 0.75)

    # (a3) Cross-season heatmap
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
        icc_str = ''

    ax_hm.set_title(f'(c)  Cross-season ({len(common)} teams, {icc_str})', fontweight='bold')

    cax = fig.add_axes(POS_CB)
    cb = plt.colorbar(im, cax=cax)
    cb.set_label('$f_c^{Attack}$', fontsize=16)

    # (b1-b3) Home/Away scatter
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

        r, _ = stats.pearsonr([home[t] for t in common_ha],
                              [away[t] for t in common_ha])
        _, p_paired = stats.ttest_rel([home[t] for t in common_ha],
                                      [away[t] for t in common_ha])
        diff = np.mean([home[t] for t in common_ha]) - np.mean([away[t] for t in common_ha])

        ax.text(0.05, 0.95,
                f'r = {r:.3f}\n\u0394 = {diff:+.3f}\np = {p_paired:.3f}',
                transform=ax.transAxes, fontsize=10, va='top')

        ax.set_xlim(mn, mx)
        ax.set_ylim(mn, mx)
        ax.set_aspect('equal')
        panel = chr(ord('d') + si)
        ax.set_title(f'({panel})  {season}', fontweight='bold')
        if si == 0:
            ax.set_ylabel('Away $f_c^{Attack}$')
        else:
            ax.set_yticklabels([])
        ax.set_xlabel('Home $f_c^{Attack}$')
        if si == 2:
            ax.legend(
                [Line2D([0], [0], marker='o', color='w', markerfacecolor=C_EPL, ms=5),
                 Line2D([0], [0], marker='o', color='w', markerfacecolor=C_LIGA, ms=5)],
                ['EPL', 'La Liga'], fontsize=9, loc='lower right')

    # (c1-c2) Cross-league violin
    epl_data = l2[l2['League'] == 'Premier League']
    liga_data = l2[l2['League'] != 'Premier League']

    metrics_c = [
        ('$f_c^{\\mathrm{Attack}}$', FC, '(g)'),
        ('Reciprocity $\\rho_{\\leftrightarrow}$', 'Reciprocity', '(h)'),
    ]

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

    out = OUTPUT_DIR / 'fig7.png'
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(str(out).replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close()


if __name__ == '__main__':
    main()
