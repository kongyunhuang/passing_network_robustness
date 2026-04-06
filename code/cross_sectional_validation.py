"""
cross_sectional_validation.py — Cross-sectional association between structural residuals and market valuation (Table 4)
======================================================================================================================
Computes player structural residual z-scores independently per season, then
correlates them with same-season Transfermarkt valuations using Spearman
correlation and partial correlation (controlling for ppg + age). Outputs
all numbers for Table 4.

Input:
  - data/player_impact_{season}.csv (three seasons)
  - data/market_value_validation_all_seasons.csv
  - data/transfermarkt_player_valuations.csv
  - data/transfermarkt_players.csv

Output: Table 4 values (printed to console)

Usage:
  python code/cross_sectional_validation.py

Last modified: 2026-03-27
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from pathlib import Path

# ============================================================================
# Paths
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'


def compute_team_residuals(df):
    """Within each team, regress impact_per_match on ppg, take z-score residuals."""
    residuals = {}
    for team, grp in df.groupby('team'):
        if len(grp) < 5:
            continue
        X = grp['ppg'].values.reshape(-1, 1)
        y = grp['impact_per_match'].values
        reg = LinearRegression().fit(X, y)
        res = y - reg.predict(X)
        std = res.std()
        for i, (_, row) in enumerate(grp.iterrows()):
            z = res[i] / std if std > 0 else 0
            residuals[row['player']] = {
                'z': z, 'ppg': row['ppg'], 'team': team,
                'impact': row['impact_per_match'], 'played': row['played']
            }
    return residuals


def get_nearest_value(pid, val_start, val_end, target_date, vals):
    """Get valuation record nearest to target_date."""
    pv = vals[(vals['player_id'] == pid)
              & (vals['date'] >= val_start) & (vals['date'] <= val_end)]
    if len(pv) == 0:
        return None
    pv = pv.copy()
    pv['dist'] = abs(pd.to_datetime(pv['date']) - pd.Timestamp(target_date)).dt.days
    return pv.sort_values('dist').iloc[0]['market_value_in_eur']


def partial_spearman(z_vals, value_vals, ppg_vals, age_vals):
    """Partial Spearman: residualise on controls, then compute Spearman."""
    X = np.column_stack([ppg_vals, age_vals])
    z_r = z_vals - LinearRegression().fit(X, z_vals).predict(X)
    v_r = value_vals - LinearRegression().fit(X, value_vals).predict(X)
    return stats.spearmanr(z_r, v_r)


def main():
    mv_match = pd.read_csv(DATA_DIR / 'market_value_validation_all_seasons.csv')
    vals = pd.read_csv(DATA_DIR / 'transfermarkt_player_valuations.csv')
    players_tm = pd.read_csv(DATA_DIR / 'transfermarkt_players.csv')

    name_to_pid = dict(zip(mv_match['sb_name'], mv_match['tm_player_id']))
    pid_to_dob = {}
    for _, r in players_tm.iterrows():
        if pd.notna(r.get('date_of_birth')):
            pid_to_dob[r['player_id']] = r['date_of_birth']

    SEASONS = [
        ('2022/23', 'player_impact_2022_23.csv', '2022-07-01', '2023-12-31', '2023-06-15'),
        ('2023/24', 'player_impact_202324.csv', '2023-07-01', '2024-12-31', '2024-06-15'),
        ('2024/25', 'player_impact_202425.csv', '2024-07-01', '2025-12-31', '2025-01-15'),
    ]

    print("=" * 70)
    print("Table 4: Cross-sectional association (per-season, Spearman rho)")
    print("=" * 70)

    all_season_data = {}

    for season, impact_file, val_start, val_end, target_date in SEASONS:
        df = pd.read_csv(DATA_DIR / impact_file)
        residuals = compute_team_residuals(df)

        rows = []
        for player, info in residuals.items():
            pid = name_to_pid.get(player)
            if pid is None:
                continue
            value = get_nearest_value(pid, val_start, val_end, target_date, vals)
            if value is None or value <= 0:
                continue
            dob = pid_to_dob.get(pid)
            age = None
            if dob:
                age = (pd.Timestamp(target_date) - pd.Timestamp(dob)).days / 365.25
            rows.append({
                'player': player, 'team': info['team'],
                'z': info['z'], 'ppg': info['ppg'],
                'impact': info['impact'], 'played': info['played'],
                'value': value, 'log_value': np.log10(value), 'age': age,
            })

        result = pd.DataFrame(rows)
        all_season_data[season] = result

        rho, p = stats.spearmanr(result['z'], result['log_value'])

        valid = result.dropna(subset=['age'])
        rho_p, p_p = partial_spearman(
            valid['z'].values, valid['log_value'].values,
            valid['ppg'].values, valid['age'].values
        )

        print(f"\n{season}:")
        print(f"  n = {len(result)}")
        print(f"  Spearman rho = {rho:.3f}, P = {p:.2e}")
        print(f"  Partial rho = {rho_p:.3f}, P = {p_p:.2e} (n_valid={len(valid)})")

    # Within-team rank analysis
    print("\n" + "=" * 70)
    print("Within-team rank analysis")
    print("=" * 70)

    for season, result in all_season_data.items():
        team_rhos_z = []
        team_rhos_ppg = []
        for team, grp in result.groupby('team'):
            if len(grp) < 5:
                continue
            r_z, _ = stats.spearmanr(grp['z'].values, grp['log_value'].values)
            r_ppg, _ = stats.spearmanr(grp['ppg'].values, grp['log_value'].values)
            if not np.isnan(r_z):
                team_rhos_z.append(r_z)
            if not np.isnan(r_ppg):
                team_rhos_ppg.append(r_ppg)

        avg_z = np.mean(team_rhos_z)
        _, p_z = stats.ttest_1samp(team_rhos_z, 0)
        avg_ppg = np.mean(team_rhos_ppg)

        print(f"\n{season} ({len(team_rhos_z)} teams):")
        print(f"  Within-team residual vs value: avg rho = {avg_z:.3f}, P = {p_z:.2e}")
        print(f"  Within-team pass volume vs value: avg rho = {avg_ppg:.3f}")

    # High-residual vs neutral group
    print("\n" + "=" * 70)
    print("High-residual vs neutral group value comparison")
    print("=" * 70)

    for season, result in all_season_data.items():
        high = result[result['z'] > 1.0]
        neutral = result[(result['z'] > -0.5) & (result['z'] < 0.5)]
        ratio = high['value'].median() / neutral['value'].median()
        _, p_mw = stats.mannwhitneyu(high['value'], neutral['value'], alternative='two-sided')

        print(f"\n{season}:")
        print(f"  High residual (z>1s): n={len(high)}, median=EUR{high['value'].median()/1e6:.1f}M")
        print(f"  Neutral (|z|<0.5s): n={len(neutral)}, median=EUR{neutral['value'].median()/1e6:.1f}M")
        print(f"  Ratio: {ratio:.1f}x, P = {p_mw:.2e}")


if __name__ == '__main__':
    main()
