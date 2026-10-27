"""
s14_ebc_ranking.py  Edge betweenness versus degree-based targeted removal (Fig. S10)
===================================================================================
Compares the degree-based attack of the main pipeline with an attack ordered by
edge betweenness centrality (EBC) on the 4,343 L2 networks that are valid under
the first setting of Supplementary Table S8 (20 random-failure trials,
unrestricted estimator). Both attacks use the same percolation machinery
(50 removal steps, unrestricted susceptibility-peak estimator, one
deterministic removal per ranking). The plot mode draws Fig. S10 from the
shipped results and needs no raw data.

Input  data/match_summary.csv and raw event data under DATA_ROOT (full run),
       data/supplementary/ebc_vs_degree_all_matches.csv (plot mode)
Output  results/figS10_ebc_vs_degree_all_matches.csv (full run),
        figures/figS10.pdf, figures/figS10.png (plot mode)

Usage
  python code/supplementary/s14_ebc_ranking.py --plot          # Fig. S10 from shipped file
  DATA_ROOT=/path/to/statsbomb/datos python code/supplementary/s14_ebc_ranking.py --limit 30
  DATA_ROOT=/path/to/statsbomb/datos python code/supplementary/s14_ebc_ranking.py

Last modified 2026-10-27
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "code" / "pipeline"))

from data_loader import (                    # noqa: E402
    load_match_events, extract_passes, build_substitution_map,
    apply_substitution_map,
)
from robustness import edge_removal_simulation, calculate_metrics_directed  # noqa: E402

# ============================================================================
# Configuration
# ============================================================================
# Root directory of the StatsBomb event data (set the DATA_ROOT environment variable)
DATA_DIR = Path(os.environ.get("DATA_ROOT", "/path/to/statsbomb/datos"))

# First setting of Supplementary Table S8 (20 trials at L2, unrestricted),
# used for the L2 validity filter and the reference attack threshold
MATCH_SUMMARY_PATH = BASE_DIR / "data" / "match_summary.csv"
SUPP_DATA = BASE_DIR / "data" / "supplementary"
FIG_DIR = BASE_DIR / "figures"
RESULTS_DIR = BASE_DIR / "results"

N_STEPS = 50
PITCH_X, PITCH_Y = 120, 80
L2_COLS, L2_ROWS = 6, 4   # L2 scale

# League/season -> directory mapping
DIR_MAP = {
    ("Premier League", "2022/23"): "2_235_2022_23",
    ("Premier League", "2023/24"): "2_281_2023_24",
    ("Premier League", "2024/25"): "2_317_2024_25",
    ("La Liga", "2022/23"): "11_235_2022_23",
    ("La Liga", "2023/24"): "11_281_2023_24",
    ("La Liga", "2024/25"): "11_317_2024_25",
}

# Supplementary Section S14
GATE_N = 4343
GATE_PCT_EBC_LOWER = 74.6         # percent of networks with f_c(EBC) < f_c(degree)
GATE_MEAN = {"fc_degree": 0.423, "fc_ebc": 0.314}
GATE_PCT_BELOW_01 = {"fc_ebc": 13.7, "fc_degree": 1.8}   # percent with f_c < 0.1


# ============================================================================
# Network construction, consistent with pipeline/network_builder.py
# ============================================================================
def build_l2_network(passes):
    """Build the L2 (6x4) directed simple graph from passes.

    Consistent with the network construction of the main text
    - nodes = (player_id, zone_id)
    - directed edge = passer node -> receiver node
    - parallel edges merged (simple directed graph)
    """
    G = nx.DiGraph()
    for p in passes:
        zx = min(int(p['x_start'] / (PITCH_X / L2_COLS)), L2_COLS - 1)
        zy = min(int(p['y_start'] / (PITCH_Y / L2_ROWS)), L2_ROWS - 1)
        passer_node = (p['player_id'], zy * L2_COLS + zx)

        rzx = min(int(p['x_end'] / (PITCH_X / L2_COLS)), L2_COLS - 1)
        rzy = min(int(p['y_end'] / (PITCH_Y / L2_ROWS)), L2_ROWS - 1)
        receiver_node = (p['recipient_id'], rzy * L2_COLS + rzx)

        if passer_node != receiver_node:
            G.add_edge(passer_node, receiver_node)
    return G


# ============================================================================
# EBC percolation, the same logic as edge_removal_simulation(method='attack');
# the only difference is the ordering criterion (EBC instead of degree sum)
# ============================================================================
def ebc_percolation(G, n_steps=N_STEPS):
    """Run edge betweenness centrality attack percolation.

    Aligned with edge_removal_simulation(method='attack')
    1. convert to a simple directed graph
    2. compute EBC for all edges (once, static strategy)
    3. sort edges by descending EBC
    4. remove progressively at n_steps sampling points, recording
       P_inf_strong and Chi_strong
    """
    G_work = nx.DiGraph(G)
    E_0 = G_work.number_of_edges()

    if E_0 == 0:
        metrics = calculate_metrics_directed(G_work)
        return pd.DataFrame([{'f': 0.0, **metrics}])

    # Compute EBC and sort (static, determined once)
    ebc = nx.edge_betweenness_centrality(G_work, normalized=True)
    edges = list(G_work.edges())
    edges.sort(key=lambda e: ebc.get(e, 0), reverse=True)
    removal_order = edges

    # Sampling points, identical to edge_removal_simulation
    f_values = np.linspace(0, 1, n_steps)
    target_removals = sorted(set(int(f * E_0) for f in f_values))

    # Initial state
    results = [{'f': 0.0, **calculate_metrics_directed(G_work)}]

    edges_removed = 0
    removal_idx = 0

    for target_n in target_removals:
        if target_n == 0:
            continue

        num_to_remove = target_n - edges_removed
        for _ in range(num_to_remove):
            if removal_idx >= len(removal_order):
                break
            edge = removal_order[removal_idx]
            if G_work.has_edge(*edge):
                G_work.remove_edge(*edge)
            removal_idx += 1
            edges_removed += 1

        metrics = calculate_metrics_directed(G_work)
        results.append({'f': edges_removed / E_0, **metrics})

    return pd.DataFrame(results)


def find_fc_from_chi(df):
    """Locate f_c at the Chi_strong peak (unrestricted estimator)."""
    if df.empty or 'Chi_strong' not in df.columns:
        return 0.0, 0.0
    idx = df['Chi_strong'].idxmax()
    return float(df.loc[idx, 'f']), float(df.loc[idx, 'Chi_strong'])


# ============================================================================
# Full run over all valid L2 networks
# ============================================================================
def run_all(limit=None):
    # L2 networks valid under the first setting of Supplementary Table S8
    ms = pd.read_csv(MATCH_SUMMARY_PATH)
    l2 = ms[(ms['Scale'] == 'L2') &
            (ms['SCGC_Ratio'] >= 0.3) &
            (ms['fc_Error_Strong'] > 0) &
            (ms['fc_Attack_Strong'] > 0)].copy()
    print(f"L2 filtered: {len(l2)} matches")
    if limit:
        l2 = l2.head(limit)
        print(f"--limit {limit}: processing first {len(l2)} rows only")

    results = []
    errors = 0
    skipped = 0
    t0 = time.time()

    for i, (_, row) in enumerate(l2.iterrows()):
        mid = int(row['MatchID'])
        tid = int(row['TeamID'])
        league = row['League']
        season = row['Season']

        dir_key = (league, season)
        if dir_key not in DIR_MAP:
            skipped += 1
            continue

        events_path = DATA_DIR / DIR_MAP[dir_key] / f"{mid}_events.json"
        if not events_path.exists():
            skipped += 1
            continue

        try:
            events = load_match_events(str(events_path))
            # Substitution inheritance, identical to the main pipeline
            sub_map = build_substitution_map(events, team_id=tid)
            passes = extract_passes(events, team_id=tid,
                                    successful_only=True, exclude_set_pieces=True)
            apply_substitution_map(passes, sub_map)
            G = build_l2_network(passes)

            if G.number_of_edges() < 5:
                skipped += 1
                continue

            # Degree-sum attack (identical to the main pipeline)
            df_deg = edge_removal_simulation(G, method='attack', n_steps=N_STEPS)
            fc_deg, chi_deg = find_fc_from_chi(df_deg)

            # EBC attack
            df_ebc = ebc_percolation(G, n_steps=N_STEPS)
            fc_ebc, chi_ebc = find_fc_from_chi(df_ebc)

            results.append({
                'MatchID': mid,
                'TeamID': tid,
                'Team': row['Team'],
                'Opponent': row.get('Opponent', ''),
                'League': league,
                'Season': season,
                'N': G.number_of_nodes(),
                'E': G.number_of_edges(),
                'fc_degree': fc_deg,
                'chi_degree': chi_deg,
                'fc_ebc': fc_ebc,
                'chi_ebc': chi_ebc,
                'fc_paper': float(row['fc_Attack_Strong']),  # attack threshold in match_summary.csv
            })

            count = len(results)
            if count % 200 == 0:
                elapsed = time.time() - t0
                rate = count / elapsed
                remaining = (len(l2) - count - skipped) / rate
                print(f"  {count}/{len(l2)} done ({elapsed:.0f}s, ~{remaining:.0f}s remaining)")

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  Error {mid}: {e}")

    elapsed = time.time() - t0
    count = len(results)
    print(f"\nDone: {count} matches in {elapsed:.1f}s ({count/max(elapsed, 1e-9):.1f} matches/s)")
    print(f"Skipped: {skipped}, Errors: {errors}")

    rdf = pd.DataFrame(results)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "figS10_ebc_vs_degree_all_matches.csv"
    rdf.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path.relative_to(BASE_DIR)}")

    return rdf


def print_summary(rdf):
    """Print summary statistics."""
    print(f"\n{'='*60}")
    print(f"SUMMARY (N={len(rdf)} matches, L2 scale)")
    print(f"{'='*60}")
    print(f"fc_degree:  mean={rdf['fc_degree'].mean():.3f}, std={rdf['fc_degree'].std():.3f}")
    print(f"fc_ebc:     mean={rdf['fc_ebc'].mean():.3f}, std={rdf['fc_ebc'].std():.3f}")
    print(f"fc_paper:   mean={rdf['fc_paper'].mean():.3f} (attack threshold in match_summary.csv)")
    print(f"fc_degree vs fc_paper correlation: {rdf['fc_degree'].corr(rdf['fc_paper']):.4f}")
    print(f"fc_ebc < fc_degree: {(rdf['fc_ebc'] < rdf['fc_degree']).sum()}/{len(rdf)} "
          f"= {(rdf['fc_ebc'] < rdf['fc_degree']).mean():.1%}")
    print(f"chi_ebc < chi_degree: {(rdf['chi_ebc'] < rdf['chi_degree']).sum()}/{len(rdf)} "
          f"= {(rdf['chi_ebc'] < rdf['chi_degree']).mean():.1%}")


def check_shipped(rdf):
    """Acceptance check of the shipped results against Supplementary Section S14."""
    checks = [("networks", len(rdf), GATE_N, 0),
              ("f_c(EBC) < f_c(degree) (%)",
               100 * (rdf['fc_ebc'] < rdf['fc_degree']).mean(), GATE_PCT_EBC_LOWER, 0.05)]
    for col, exp in GATE_MEAN.items():
        checks.append((f"mean {col}", rdf[col].mean(), exp, 0.0005))
    for col, exp in GATE_PCT_BELOW_01.items():
        checks.append((f"{col} < 0.1 (%)", 100 * (rdf[col] < 0.1).mean(), exp, 0.05))
    failed = False
    for label, got, exp, tol in checks:
        ok = abs(got - exp) <= tol + 1e-12
        failed |= not ok
        print(f"  {label:<28} {got:10.4f} vs {exp:8.3f}  {'PASS' if ok else 'FAIL'}")
    if failed:
        print("GATE FAILED  shipped results differ from Supplementary Section S14")
        sys.exit(1)
    print("GATE PASS  shipped results match Supplementary Section S14")


def run_plot():
    """Draw Fig. S10 from the shipped results file."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.family': 'serif',
                         'font.serif': ['Times New Roman', 'Times'],
                         'font.size': 9, 'pdf.fonttype': 42,
                         'mathtext.fontset': 'cm', 'axes.unicode_minus': False})

    rdf = pd.read_csv(SUPP_DATA / 'ebc_vs_degree_all_matches.csv')
    check_shipped(rdf)
    n = len(rdf)
    pct = (rdf['fc_ebc'] < rdf['fc_degree']).mean()

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))

    ax = axes[0]
    ax.scatter(rdf['fc_degree'], rdf['fc_ebc'], s=4, alpha=0.15, c='#1B4F72', linewidths=0)
    ax.plot([0, 0.9], [0, 0.9], 'k--', alpha=0.5, lw=0.8)
    ax.set_xlabel(r'$f_{c,\mathrm{L2}}$ (degree-based attack)')
    ax.set_ylabel(r'$f_{c,\mathrm{L2}}$ (EBC attack)')
    ax.set_title(rf'(a) $f_{{c,\mathrm{{L2}}}}$ comparison (N = {n:,})', fontsize=9.5)
    ax.set_xlim(0, 0.9); ax.set_ylim(0, 0.9)
    ax.text(0.04, 0.86, f'{pct:.1%} of matches:\n' + r'$f_{c,\mathrm{L2}}^{\mathrm{EBC}} < f_{c,\mathrm{L2}}^{\mathrm{degree}}$',
            transform=ax.transAxes, fontsize=8, bbox=dict(facecolor='#f5e9d0', alpha=0.9, lw=0.4))

    ax = axes[1]
    ax.scatter(rdf['chi_degree'], rdf['chi_ebc'], s=4, alpha=0.15, c='#C0392B', linewidths=0)
    upper = max(rdf['chi_degree'].quantile(0.99), rdf['chi_ebc'].quantile(0.99))
    ax.plot([0, upper], [0, upper], 'k--', alpha=0.5, lw=0.8)
    ax.set_xlabel(r'$\chi_{\max}$ (degree-based)')
    ax.set_ylabel(r'$\chi_{\max}$ (EBC)')
    ax.set_title(r'(b) $\chi$ peak comparison', fontsize=9.5)
    pct2 = (rdf['chi_ebc'] < rdf['chi_degree']).mean()
    ax.text(0.04, 0.86, f'{pct2:.1%} of matches:\n' + r'$\chi_{\mathrm{EBC}} < \chi_{\mathrm{degree}}$',
            transform=ax.transAxes, fontsize=8, bbox=dict(facecolor='#f5e9d0', alpha=0.9, lw=0.4))

    ax = axes[2]
    ax.hist(rdf['fc_degree'], bins=40, alpha=0.6, color='#C0392B',
            label=f'Degree-based (mean = {rdf.fc_degree.mean():.3f})')
    ax.hist(rdf['fc_ebc'], bins=40, alpha=0.6, color='#2e7d32',
            label=f'EBC (mean = {rdf.fc_ebc.mean():.3f})')
    ax.set_xlabel(r'$f_{c,\mathrm{L2}}$'); ax.set_ylabel('Count')
    ax.set_title(r'(c) $f_{c,\mathrm{L2}}$ distributions', fontsize=9.5)
    ax.legend(fontsize=7.5, frameon=False)

    for a in axes:
        a.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / 'figS10.pdf')
    fig.savefig(FIG_DIR / 'figS10.png', dpi=200)
    print('Saved figures/figS10.pdf, figures/figS10.png')


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true",
                    help="only draw Fig. S10 from the shipped results file")
    ap.add_argument("--limit", type=int, default=None,
                    help="process only the first N filtered rows (quick check)")
    args = ap.parse_args()
    if args.plot:
        run_plot()
    else:
        rdf = run_all(limit=args.limit)
        print_summary(rdf)
