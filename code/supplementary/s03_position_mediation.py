"""
s03_position_mediation.py  Positional degree hierarchy and mediation (Fig. S2, Tables S2 and S3)
==============================================================================================
Directed degree by position over the 509,491 labelled nodes of the 4,560 L2
single-match networks (Table S2, Section S3), Baron and Kenny mediation over
48 teams (Table S3) and Fig. S2. Team-level threshold and degree come from the
published analysis (100 Monte Carlo trials at all scales, restricted threshold
estimator), averaged per team, season and venue and then per team.

Input  data/position_node_records.csv, data/position_network_records.csv,
  data/match_summary_mc100_guarded.csv
Output  figures/figS2.png, figures/figS2.pdf, results/tableS2.csv,
  results/tableS3.csv

Usage
  python code/supplementary/s03_position_mediation.py

Last modified 2026-10-27
"""

import sys

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
from pathlib import Path

# ============================================================================
# Paths
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / 'data'
# Published analysis (100 Monte Carlo trials at all scales, restricted
# threshold estimator)
PUBLISHED = DATA_DIR / 'match_summary_mc100_guarded.csv'
OUTPUT_DIR = BASE_DIR / 'figures'
RESULTS_DIR = BASE_DIR / 'results'

# ============================================================================
# Publication style
# ============================================================================
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 13,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.dpi': 300,
    'mathtext.fontset': 'cm',
})

POS4_ORDER = ['Goalkeeper', 'Defender', 'Midfielder', 'Forward']
POS4_SHORT = {'Goalkeeper': 'GK', 'Defender': 'DEF', 'Midfielder': 'MID', 'Forward': 'FWD'}
POSITION_SCORE = {'Goalkeeper': 0, 'Defender': 1, 'Midfielder': 2, 'Forward': 3}
LEAGUE_SHORT = {'Premier League': 'EPL', 'La Liga': 'La Liga'}

# Supplementary Table S2 as printed (N, directed degree, SD, in, out, hub enrichment).
# The degrees of the 23 fine-grained positions quoted in Section S3 are not
# checked, because the node file stores only the four position categories.
TABLE_S2 = {
    'Goalkeeper': (16915, 7.57, 5.84, 3.46, 4.11, 2.48),
    'Defender': (191571, 6.05, 5.20, 2.83, 3.21, 1.48),
    'Midfielder': (201757, 4.81, 3.79, 2.27, 2.54, 0.69),
    'Forward': (99248, 4.09, 3.34, 2.14, 1.95, 0.45),
}
# Section S3 text
N_NODES, N_NETWORKS = 509491, 4560
ANOVA_F, ANOVA_DF, ETA2 = 6702.05, (3, 509487), 0.038
CHI2_HUB = 17738.24
HUB_SHARE = {'Goalkeeper': (8.2, 3.3), 'Forward': (8.7, 19.5)}   # (hub %, overall %)
# Supplementary Table S3 and Section S3 text
TABLE_S3 = {
    'A': dict(indirect=0.102, z=4.099, pct=78.8, ci=(0.036, 0.149), conclusion='PARTIAL mediation'),
    'B': dict(indirect=0.115, z=4.598, pct=88.2, ci=(0.050, 0.162), conclusion='PARTIAL mediation'),
    'C': dict(p_total=0.435, conclusion='No total effect'),
    'D': dict(p_total=0.625, conclusion='No total effect'),
}
PATHS = [('A', 'degree_disparity', 'AvgDegree'),
         ('B', 'degree_disparity', 'Mu11_Mu'),
         ('C', 'mean_pos_score', 'AvgDegree'),
         ('D', 'frac_DEF', 'AvgDegree')]
# Fig. S2 annotations (position means, panel b and panel c correlations)
FIG_S2_MEANS = {'Goalkeeper': 7.57, 'Defender': 6.05, 'Midfielder': 4.81, 'Forward': 4.09}
FIG_S2_R = (0.520, 0.983)


# ============================================================================
# Network-level positional composition
# ============================================================================
def network_blocks(nodes, nets):
    """Split the node file into the networks of the network file.

    The node file lists the labelled nodes network by network, in the order of
    the network file, without a network identifier. Consecutive node rows of
    one (team, league) belong to one network, except where a team closes one
    match and opens the next; such a run is split at the unique row where both
    parts reproduce the stored back and front mean degrees. Every network is
    checked against these stored means. Returns the (start, end) row range of
    every network.
    """
    team = nodes['team'].to_numpy()
    league = nodes['league'].to_numpy()
    deg = nodes['degree_total'].to_numpy(float)
    back = nodes['position_4'].isin(['Goalkeeper', 'Defender']).to_numpy()
    cum_bd = np.r_[0, np.cumsum(np.where(back, deg, 0))]
    cum_bn = np.r_[0, np.cumsum(back)]
    cum_fd = np.r_[0, np.cumsum(np.where(back, 0, deg))]
    cum_fn = np.r_[0, np.cumsum(~back)]

    def back_front(a, b):
        nb, nf = cum_bn[b] - cum_bn[a], cum_fn[b] - cum_fn[a]
        return ((cum_bd[b] - cum_bd[a]) / nb if nb else 0.0,
                (cum_fd[b] - cum_fd[a]) / nf if nf else 0.0)

    target = nets[['mean_deg_back', 'mean_deg_front']].to_numpy()
    net_team = nets['team_name'].to_numpy()
    net_league = nets['league'].to_numpy()

    def matches(a, b, k):
        return np.allclose(back_front(a, b), target[k], rtol=0, atol=1e-9)

    starts = np.r_[0, np.flatnonzero((team[1:] != team[:-1]) | (league[1:] != league[:-1])) + 1]
    ends = np.r_[starts[1:], len(nodes)]
    blocks, k = [], 0
    for a, b in zip(starts, ends):
        m = 1
        while k + m < len(nets) and net_team[k + m] == team[a] and net_league[k + m] == league[a]:
            m += 1
        if net_team[k] != team[a] or net_league[k] != league[a] or m > 2:
            sys.exit(f"Node and network files are not aligned at network {k}.")
        if m == 1:
            if not matches(a, b, k):
                sys.exit(f"Network {k} does not reproduce its stored mean degrees.")
            blocks.append((a, b))
        else:
            cut = [e for e in range(a + 1, b) if matches(a, e, k) and matches(e, b, k + 1)]
            if len(cut) != 1:
                sys.exit(f"Networks {k} and {k + 1} cannot be separated uniquely.")
            blocks += [(a, cut[0]), (cut[0], b)]
        k += m
    if k != len(nets) or blocks[-1][1] != len(nodes):
        sys.exit("Node and network files do not cover the same networks.")
    return blocks


def add_composition(nodes, nets):
    """Mean position score (GK 0, DEF 1, MID 2, FWD 3) and defender fraction of
    the labelled nodes of each network, used in Paths C and D of Table S3, and
    the season of each node."""
    blocks = network_blocks(nodes, nets)
    a = np.array([s for s, _ in blocks])
    b = np.array([e for _, e in blocks])
    n = b - a
    score = np.r_[0, np.cumsum(nodes['position_4'].map(POSITION_SCORE).to_numpy(float))]
    is_def = np.r_[0, np.cumsum((nodes['position_4'] == 'Defender').to_numpy(float))]
    nets = nets.copy()
    nets['mean_pos_score'] = (score[b] - score[a]) / n
    nets['frac_DEF'] = (is_def[b] - is_def[a]) / n
    nodes = nodes.copy()
    nodes['season'] = np.repeat(nets['season'].to_numpy(), n)
    return nodes, nets


# ============================================================================
# Table S2 and Section S3 node statistics
# ============================================================================
def position_statistics(nodes):
    rows = []
    t_hi = nodes['degree_total'].quantile(0.90)
    top = nodes[nodes['degree_total'] >= t_hi]
    for pos in POS4_ORDER:
        sub = nodes[nodes['position_4'] == pos]
        f_all = (nodes['position_4'] == pos).mean() * 100
        f_top = (top['position_4'] == pos).mean() * 100
        rows.append({'position': POS4_SHORT[pos], 'position_4': pos, 'N': len(sub),
                     'dir_degree': sub['degree_total'].mean(),
                     'sd': sub['degree_total'].std(),
                     'in_degree': sub['in_degree'].mean(),
                     'out_degree': sub['out_degree'].mean(),
                     'overall_pct': f_all, 'hub_pct': f_top, 'hub_enrichment': f_top / f_all})
    table = pd.DataFrame(rows)

    groups = [nodes[nodes['position_4'] == pos]['degree_total'].values for pos in POS4_ORDER]
    F, p_anova = stats.f_oneway(*groups)
    gm = nodes['degree_total'].mean()
    ss_b = sum(len(g) * (g.mean() - gm) ** 2 for g in groups)
    ss_t = ((nodes['degree_total'] - gm) ** 2).sum()
    # Chi-squared test of independence between hub membership (degree at or
    # above the 90th percentile) and position, a 2 x 4 contingency table
    hub = nodes['degree_total'] >= t_hi
    table_hub = pd.crosstab(hub, nodes['position_4'])[POS4_ORDER]
    chi2, p_chi, _, _ = stats.chi2_contingency(table_hub)

    subgroups = []
    for lg in sorted(nodes['league'].unique()):
        for sn in sorted(nodes['season'].unique()):
            sub = nodes[(nodes['league'] == lg) & (nodes['season'] == sn)]
            means = sub.groupby('position_4')['degree_total'].mean()
            order = list(means.sort_values(ascending=False).index)
            subgroups.append((lg, sn, order == POS4_ORDER))
    omnibus = dict(F=F, p_anova=p_anova, df=(len(groups) - 1, len(nodes) - len(groups)),
                   eta2=ss_b / ss_t, chi2=chi2, p_chi=p_chi, subgroups=subgroups)
    return table, omnibus


# ============================================================================
# Baron and Kenny mediation with Sobel test and bootstrap CI
# ============================================================================
def sobel_test(a, b, se_a, se_b):
    ab = a * b
    se_ab = np.sqrt(a**2 * se_b**2 + b**2 * se_a**2)
    z = ab / se_ab if se_ab > 0 else 0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return ab, se_ab, z, p


def baron_kenny(X, M, Y):
    """Total, a, b and direct paths, Sobel test and 5,000-resample bootstrap CI."""
    mask = np.isfinite(X) & np.isfinite(M) & np.isfinite(Y)
    X, M, Y = X[mask], M[mask], Y[mask]
    n = len(X)
    results = {"n": n}

    # Path c (total), X on Y
    slope_c, intercept_c, r_c, p_c, se_c = stats.linregress(X, Y)
    results["path_c"] = slope_c
    results["p_c"] = p_c

    # Path a, X on M
    slope_a, intercept_a, r_a, p_a, se_a = stats.linregress(X, M)
    results["path_a"] = slope_a
    results["p_a"] = p_a

    # Paths b and c', Y on X and M
    A = np.column_stack([X, M, np.ones(n)])
    coeffs, _, _, _ = np.linalg.lstsq(A, Y, rcond=None)
    c_prime, b, intercept = coeffs
    resid = Y - A @ coeffs
    mse = np.sum(resid**2) / (n - 3)
    cov_matrix = mse * np.linalg.inv(A.T @ A)
    se_c_prime = np.sqrt(cov_matrix[0, 0])
    se_b = np.sqrt(cov_matrix[1, 1])
    p_c_prime = 2 * (1 - stats.t.cdf(abs(c_prime / se_c_prime), n - 3))
    p_b = 2 * (1 - stats.t.cdf(abs(b / se_b), n - 3))
    results["path_b"] = b
    results["p_b"] = p_b
    results["path_c_prime"] = c_prime
    results["p_c_prime"] = p_c_prime

    ab, se_ab, z_sobel, p_sobel = sobel_test(slope_a, b, se_a, se_b)
    results["indirect"] = ab
    results["sobel_z"] = z_sobel
    results["p_sobel"] = p_sobel
    results["pct_mediated"] = (ab / slope_c) * 100 if abs(slope_c) > 1e-10 else float('nan')

    rng = np.random.default_rng(42)
    boot_indirect = []
    for _ in range(5000):
        idx = rng.choice(n, n, replace=True)
        Xb, Mb, Yb = X[idx], M[idx], Y[idx]
        a_b, _, _, _, _ = stats.linregress(Xb, Mb)
        Ab = np.column_stack([Xb, Mb, np.ones(n)])
        try:
            cb = np.linalg.lstsq(Ab, Yb, rcond=None)[0]
            boot_indirect.append(a_b * cb[1])
        except Exception:
            pass
    boot_indirect = np.array(boot_indirect)
    results["boot_lo"] = np.percentile(boot_indirect, 2.5)
    results["boot_hi"] = np.percentile(boot_indirect, 97.5)

    if p_c > 0.05:
        interp = "No total effect"
    elif p_sobel > 0.05:
        interp = "No mediation"
    elif p_c_prime > 0.05:
        interp = "FULL mediation"
    else:
        interp = "PARTIAL mediation"
    results["interpretation"] = interp
    return results


def team_level_l2(matches):
    """L2 networks averaged per team, season and venue (240 rows)."""
    l2 = matches[matches.Scale == 'L2']
    agg = l2.groupby(['Team', 'League', 'Season', 'Venue'], as_index=False).agg(
        pc_Attack_Strong=('pc_Attack_Strong', 'mean'),
        AvgDegree=('AvgDegree', 'mean'),
        Mu11_Mu=('Mu11_Mu', 'mean'))
    agg['Scale'] = 'L2'
    return agg


def mediation(batch, nets):
    team_rob = batch.groupby(['Team', 'League'], as_index=False).mean(numeric_only=True)
    team_rob['league'] = team_rob['League'].map(LEAGUE_SHORT)
    team_net = nets.groupby(['league', 'team_name'], as_index=False).agg(
        degree_disparity=('degree_disparity', 'mean'),
        mean_pos_score=('mean_pos_score', 'mean'),
        frac_DEF=('frac_DEF', 'mean')).rename(columns={'team_name': 'Team'})
    merged = team_rob.merge(team_net, on=['league', 'Team'], how='inner')
    out = {}
    for lab, iv, med in PATHS:
        out[lab] = baron_kenny(merged[iv].values, merged[med].values,
                               merged['pc_Attack_Strong'].values)
    return out, len(merged)


def main():
    nodes = pd.read_csv(DATA_DIR / 'position_node_records.csv')
    nets = pd.read_csv(DATA_DIR / 'position_network_records.csv')
    batch = team_level_l2(pd.read_csv(PUBLISHED))
    nodes, nets = add_composition(nodes, nets)

    checks = []

    def check(label, got, expected, tol):
        checks.append((label, got, expected, abs(got - expected) <= tol + 1e-12))

    # ---- Table S2 and Section S3 node statistics ----
    s2, omni = position_statistics(nodes)
    check('Section S3 node observations', len(nodes), N_NODES, 0)
    check('Section S3 team-match networks', len(nets), N_NETWORKS, 0)
    for _, row in s2.iterrows():
        exp = TABLE_S2[row['position_4']]
        check(f"Table S2 {row['position']} N", row['N'], exp[0], 0)
        for col, e in zip(['dir_degree', 'sd', 'in_degree', 'out_degree', 'hub_enrichment'], exp[1:]):
            check(f"Table S2 {row['position']} {col}", row[col], e, 0.005)
    check('Section S3 ANOVA F', omni['F'], ANOVA_F, 0.005)
    check('Section S3 ANOVA df1', omni['df'][0], ANOVA_DF[0], 0)
    check('Section S3 ANOVA df2', omni['df'][1], ANOVA_DF[1], 0)
    check('Section S3 eta squared', omni['eta2'], ETA2, 0.0005)
    check('Section S3 hub chi squared', omni['chi2'], CHI2_HUB, 0.005)
    checks.append(('Section S3 hub chi squared P < 0.001', omni['p_chi'], 0.001, omni['p_chi'] < 0.001))
    by_pos = s2.set_index('position_4')
    for pos, (hub, overall) in HUB_SHARE.items():
        check(f"Section S3 {POS4_SHORT[pos]} share of hubs (%)", by_pos.loc[pos, 'hub_pct'], hub, 0.05)
        check(f"Section S3 {POS4_SHORT[pos]} overall share (%)", by_pos.loc[pos, 'overall_pct'], overall, 0.05)
    n_ok = sum(ok for _, _, ok in omni['subgroups'])
    checks.append(('Section S3 GK>DEF>MID>FWD in league-season subgroups',
                   n_ok, 6, n_ok == 6 and len(omni['subgroups']) == 6))

    # ---- Table S3 mediation ----
    med, n_teams = mediation(batch, nets)
    check('Table S3 teams', n_teams, 48, 0)
    for lab, iv, mv in PATHS:
        res, exp = med[lab], TABLE_S3[lab]
        checks.append((f"Table S3 path {lab} conclusion is {exp['conclusion']}",
                       float(res['interpretation'] == exp['conclusion']), 1.0,
                       res['interpretation'] == exp['conclusion']))
        if 'indirect' in exp:
            checks.append((f"Table S3 path {lab} total P < 0.001", res['p_c'], 0.001, res['p_c'] < 0.001))
            check(f"Table S3 path {lab} indirect effect", res['indirect'], exp['indirect'], 0.0005)
            check(f"Table S3 path {lab} Sobel z", res['sobel_z'], exp['z'], 0.0005)
            check(f"Table S3 path {lab} percent mediated", res['pct_mediated'], exp['pct'], 0.05)
            check(f"Section S3 path {lab} bootstrap CI low", res['boot_lo'], exp['ci'][0], 0.0005)
            check(f"Section S3 path {lab} bootstrap CI high", res['boot_hi'], exp['ci'][1], 0.0005)
        else:
            check(f"Table S3 path {lab} total P", res['p_c'], exp['p_total'], 0.0005)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    s2.drop(columns='position_4').to_csv(RESULTS_DIR / 'tableS2.csv', index=False, float_format='%.6f')
    s3 = pd.DataFrame([{'path': lab, 'iv': iv, 'mediator': mv, 'n': med[lab]['n'],
                        'p_total': med[lab]['p_c'], 'indirect': med[lab]['indirect'],
                        'sobel_z': med[lab]['sobel_z'], 'p_sobel': med[lab]['p_sobel'],
                        'pct_mediated': med[lab]['pct_mediated'],
                        'boot_ci_low': med[lab]['boot_lo'], 'boot_ci_high': med[lab]['boot_hi'],
                        'p_direct': med[lab]['p_c_prime'],
                        'conclusion': med[lab]['interpretation']} for lab, iv, mv in PATHS])
    s3.to_csv(RESULTS_DIR / 'tableS3.csv', index=False, float_format='%.6g')

    print(f"Table S2  Directed degree by position (L2 single-match networks, "
          f"N = {len(nodes):,} node observations in {len(nets):,} networks)")
    print(f"{'Position':<9}{'N':>9}{'Dir. degree':>13}{'SD':>7}{'In-degree':>11}"
          f"{'Out-degree':>12}{'Hub enrich.':>13}")
    for _, r in s2.iterrows():
        print(f"{r['position']:<9}{r['N']:>9,}{r['dir_degree']:>13.2f}{r['sd']:>7.2f}"
              f"{r['in_degree']:>11.2f}{r['out_degree']:>12.2f}{r['hub_enrichment']:>12.2f}x")
    print(f"One-way ANOVA  F({omni['df'][0]}, {omni['df'][1]:,}) = {omni['F']:.2f}, "
          f"eta^2 = {omni['eta2']:.3f}")
    print(f"Hub composition (top 10% by degree)  chi^2 = {omni['chi2']:.2f}, P = {omni['p_chi']:.1e}")
    for pos in HUB_SHARE:
        print(f"  {POS4_SHORT[pos]}  {by_pos.loc[pos, 'hub_pct']:.1f}% of hubs, "
              f"{by_pos.loc[pos, 'overall_pct']:.1f}% overall")
    print("Ordering GK > DEF > MID > FWD by league and season  "
          + ", ".join(f"{lg} {sn} {'yes' if ok else 'no'}" for lg, sn, ok in omni['subgroups']))

    print(f"\nTable S3  Mediation analysis (N = {n_teams} teams, dependent variable L2 attack threshold)")
    print(f"{'Path':<5}{'IV':<18}{'Mediator':<11}{'Total P':>9}{'Indir. eff.':>13}{'Sobel z':>9}"
          f"{'% med.':>8}{'95% CI':>18}  Conclusion")
    for _, r in s3.iterrows():
        p_tot = '< 0.001' if r['p_total'] < 0.001 else f"{r['p_total']:.3f}"
        print(f"{r['path']:<5}{r['iv']:<18}{r['mediator']:<11}{p_tot:>9}{r['indirect']:>13.3f}"
              f"{r['sobel_z']:>9.3f}{r['pct_mediated']:>7.1f}%"
              f"{'[' + format(r['boot_ci_low'], '.3f') + ', ' + format(r['boot_ci_high'], '.3f') + ']':>18}"
              f"  {r['conclusion']}")
    print("Paths C and D are reported in Table S3 by their total P only.")
    print("\nSaved results/tableS2.csv, results/tableS3.csv\n")

    # ---- Fig. S2 ----
    nets['league'] = nets['league'].replace({'EPL': 'Premier League'})

    # Aggregate position network data to team level
    team_pos = nets.groupby(['team_name', 'league']).agg(
        degree_disparity=('degree_disparity', 'mean'),
    ).reset_index()

    # Aggregate L2 robustness to team level (across seasons and venues)
    batch_l2 = batch[batch['Scale'] == 'L2'].groupby(['Team', 'League']).agg(
        pc_Attack=('pc_Attack_Strong', 'mean'),
        AvgDegree=('AvgDegree', 'mean'),
        Mu11_Mu=('Mu11_Mu', 'mean'),
    ).reset_index()

    # Merge
    merged = team_pos.merge(batch_l2, left_on=['team_name', 'league'], right_on=['Team', 'League'], how='inner')
    print(f"Merged teams: {len(merged)}")
    print(f"EPL: {(merged['league']=='Premier League').sum()}, La Liga: {(merged['league']=='La Liga').sum()}")

    # --- Figure ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel (a), violin plot
    pos_order = ['Goalkeeper', 'Defender', 'Midfielder', 'Forward']
    pos_labels = ['GK', 'DEF', 'MID', 'FWD']
    pos_colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']

    ax = axes[0]
    for i, (pos, label, color) in enumerate(zip(pos_order, pos_labels, pos_colors)):
        data = nodes[nodes['position_4'] == pos]['degree_total']
        vp = ax.violinplot([data.values], positions=[i], showmeans=True, showmedians=True, showextrema=False)
        for body in vp['bodies']:
            body.set_facecolor(color)
            body.set_alpha(0.7)
        vp['cmeans'].set_color('black')
        vp['cmedians'].set_color('gray')
        ax.text(i, data.mean() + 0.3, f'{data.mean():.2f}', ha='center', fontsize=12, fontweight='bold')
        check(f"Fig. S2 (a) {label} mean", data.mean(), FIG_S2_MEANS[pos], 0.005)

    ax.set_xticks(range(4))
    ax.set_xticklabels(pos_labels)
    ax.set_ylabel('Directed Degree (in + out)')
    ax.set_title('(a) Position Degree Hierarchy', fontsize=13)
    ax.set_ylim(0, 15)

    def add_bracket(ax, x1, x2, y, text):
        ax.plot([x1, x1, x2, x2], [y, y+0.15, y+0.15, y], 'k-', lw=1)
        ax.text((x1+x2)/2, y+0.2, text, ha='center', fontsize=8)
    add_bracket(ax, 0, 1, 6.5, '***')
    add_bracket(ax, 1, 2, 5.5, '***')
    add_bracket(ax, 2, 3, 4.8, '***')

    # Color map for leagues
    cmap = {'Premier League': '#7B68EE', 'La Liga': '#FFA500'}
    colors = merged['league'].map(cmap)

    # Panel (b), degree_disparity against AvgDegree (L2)
    ax = axes[1]
    x, y = merged['degree_disparity'], merged['AvgDegree']
    ax.scatter(x, y, c=colors, s=50, alpha=0.7, edgecolors='white', linewidth=0.5)
    slope, intercept, r, p, se = stats.linregress(x, y)
    xline = np.linspace(x.min(), x.max(), 100)
    ax.plot(xline, slope * xline + intercept, 'k--', lw=1.5)
    ax.set_xlabel('Back–Front Degree Disparity')
    ax.set_ylabel(r'Mean Degree $\langle k \rangle$ (L2)')
    ax.set_title('(b) Path a: Disparity → Density', fontsize=13)
    p_str = 'P < 0.001' if p < 0.001 else f'P = {p:.3f}'
    ax.text(0.05, 0.92, f'r = {r:.3f}\n{p_str}', transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Panel (c), AvgDegree against pc_Attack
    ax = axes[2]
    x2, y2 = merged['AvgDegree'], merged['pc_Attack']
    ax.scatter(x2, y2, c=colors, s=50, alpha=0.7, edgecolors='white', linewidth=0.5)
    slope2, intercept2, r2, p2, se2 = stats.linregress(x2, y2)
    xline2 = np.linspace(x2.min(), x2.max(), 100)
    ax.plot(xline2, slope2 * xline2 + intercept2, 'k--', lw=1.5)
    ax.set_xlabel(r'Mean Degree $\langle k \rangle$ (L2)')
    ax.set_ylabel(r'$f_{c,\mathrm{L2}}^{\mathrm{Attack}}$')
    ax.set_title('(c) Path b: Density → Robustness', fontsize=13)
    p_str2 = 'P < 0.001' if p2 < 0.001 else f'P = {p2:.3f}'
    ax.text(0.05, 0.92, f'r = {r2:.3f}\n{p_str2}', transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='#7B68EE', markersize=8, label='EPL'),
                       Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFA500', markersize=8, label='La Liga')]
    axes[2].legend(handles=legend_elements, loc='lower right', fontsize=9)

    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / 'figS2.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / 'figS2.pdf', bbox_inches='tight')
    plt.close(fig)
    print("Saved figures/figS2.png, figures/figS2.pdf")
    print(f"Path a: r={r:.3f}, P={p:.2e}")
    print(f"Path b: r={r2:.3f}, P={p2:.2e}\n")

    check('Fig. S2 teams', len(merged), 48, 0)
    check('Fig. S2 (b) r', r, FIG_S2_R[0], 0.0005)
    checks.append(('Fig. S2 (b) P < 0.001', p, 0.001, p < 0.001))
    check('Fig. S2 (c) r', r2, FIG_S2_R[1], 0.0005)
    checks.append(('Fig. S2 (c) P < 0.001', p2, 0.001, p2 < 0.001))

    print("Acceptance check against Supplementary Tables S2 and S3, Section S3 and Fig. S2")
    for label, got, exp, ok in checks:
        print(f"  {label:<56} {got:12.6g} vs {exp:<9g} {'PASS' if ok else 'FAIL'}")
    failed = [c for c in checks if not c[3]]
    if failed:
        print(f"GATE FAILED  {len(failed)} of {len(checks)} values differ from the published values")
        for label, got, exp, _ in failed:
            print(f"  {label}  computed {got:.5f}, published {exp}")
        sys.exit(1)
    print(f"GATE PASS  all {len(checks)} values match")


if __name__ == '__main__':
    main()
