"""
s14_ebc_ranking.py — Edge betweenness vs degree-based attack (Supplementary S14)
================================================================================
Compares the degree-sum attack of the main pipeline against an edge
betweenness centrality (EBC) attack on the filtered L2 match networks
(originally submitted analysis: n = 4,343, unrestricted estimator, single
deterministic removal per strategy). For each network both attacks are run
with identical percolation machinery (n_steps = 50, susceptibility peak
estimator) and the resulting thresholds and susceptibility peaks are
recorded.

Running the full analysis requires the StatsBomb event data (set the
DATA_ROOT environment variable or edit DATA_DIR below). The plot-only mode
reproduces the supplementary three-panel figure from the shipped results CSV
and needs no raw data.

Input:  ../../data/match_summary.csv (read-only; original submission
        protocol values, used for the L2 filter and the reference column)
        StatsBomb event JSON files under DATA_ROOT (full run only)
        ../../data/supplementary/ebc_vs_degree_all_matches.csv (plot mode)
Output: ebc_vs_degree_all_matches.csv + ebc_vs_degree_summary.png (full run)
        fig_ebc_summary.pdf/.png (plot-only mode)
        (all written next to this script)

Usage:
    cd code/supplementary
    python s14_ebc_ranking.py --limit 30   # adaptation check on 30 networks
    python s14_ebc_ranking.py              # full run (requires DATA_ROOT)
    python s14_ebc_ranking.py --plot       # figure from shipped CSV

Last modified: 2026-07-21
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

_SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_SCRIPT_DIR.parent / "pipeline"))

from data_loader import (                    # noqa: E402
    load_match_events, extract_passes, build_substitution_map,
    apply_substitution_map,
)
from robustness import edge_removal_simulation, calculate_metrics_directed  # noqa: E402

# ============================================================================
# Configuration - aligned with the manuscript's percolation protocol
# ============================================================================
# Users must point this at the StatsBomb event data root directory
# (or set the DATA_ROOT environment variable).
DATA_DIR = Path(os.environ.get("DATA_ROOT", "/path/to/statsbomb/datos"))

MATCH_SUMMARY_PATH = REPO_ROOT / "data" / "match_summary.csv"
SUPP_DATA = REPO_ROOT / "data" / "supplementary"

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


# ============================================================================
# Network construction - consistent with pipeline/network_builder.py
# ============================================================================
def build_l2_network(passes):
    """Build the L2 (6x4) directed simple graph from passes.

    Consistent with the manuscript's network construction:
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
# EBC percolation - identical logic to edge_removal_simulation(method='attack');
# the only difference is the ordering criterion (EBC instead of degree sum)
# ============================================================================
def ebc_percolation(G, n_steps=N_STEPS):
    """Run edge betweenness centrality attack percolation.

    Fully aligned with edge_removal_simulation(method='attack'):
    1. convert to a simple directed graph
    2. compute EBC for all edges (once; static strategy)
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

    # Sampling points - identical to edge_removal_simulation
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
    """Find f_c as the f value at the Chi_strong peak (unrestricted estimator,
    matching the originally submitted analysis)."""
    if df.empty or 'Chi_strong' not in df.columns:
        return 0.0, 0.0
    idx = df['Chi_strong'].idxmax()
    return float(df.loc[idx, 'f']), float(df.loc[idx, 'Chi_strong'])


# ============================================================================
# Main: run all matches
# ============================================================================
def run_all(limit=None):
    # Load and filter match_summary (original submission L2 filter)
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
            # Substitution inheritance - identical to the main pipeline
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
                'fc_paper': float(row['fc_Attack_Strong']),  # reference value
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
    csv_path = _SCRIPT_DIR / "ebc_vs_degree_all_matches.csv"
    rdf.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    return rdf


def print_summary(rdf):
    """Print summary statistics."""
    print(f"\n{'='*60}")
    print(f"SUMMARY (N={len(rdf)} matches, L2 scale)")
    print(f"{'='*60}")
    print(f"fc_degree:  mean={rdf['fc_degree'].mean():.3f}, std={rdf['fc_degree'].std():.3f}")
    print(f"fc_ebc:     mean={rdf['fc_ebc'].mean():.3f}, std={rdf['fc_ebc'].std():.3f}")
    print(f"fc_paper:   mean={rdf['fc_paper'].mean():.3f} (reference column)")
    print(f"fc_degree vs fc_paper correlation: {rdf['fc_degree'].corr(rdf['fc_paper']):.4f}")
    print(f"fc_ebc < fc_degree: {(rdf['fc_ebc'] < rdf['fc_degree']).sum()}/{len(rdf)} "
          f"= {(rdf['fc_ebc'] < rdf['fc_degree']).mean():.1%}")
    print(f"chi_ebc < chi_degree: {(rdf['chi_ebc'] < rdf['chi_degree']).sum()}/{len(rdf)} "
          f"= {(rdf['chi_ebc'] < rdf['chi_degree']).mean():.1%}")


def run_plot():
    """Reproduce the supplementary three-panel figure from the shipped CSV."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.family': 'serif',
                         'font.serif': ['Times New Roman', 'Times'],
                         'font.size': 9, 'pdf.fonttype': 42,
                         'mathtext.fontset': 'cm', 'axes.unicode_minus': False})

    rdf = pd.read_csv(SUPP_DATA / 'ebc_vs_degree_all_matches.csv')

    # Acceptance gate: reproduce the numbers quoted in Supplementary S14
    n = len(rdf)
    pct = (rdf['fc_ebc'] < rdf['fc_degree']).mean()
    assert n == 4343, n
    assert abs(pct - 0.746) < 0.001, pct
    assert abs(rdf['fc_degree'].mean() - 0.423) < 0.001
    assert abs(rdf['fc_ebc'].mean() - 0.314) < 0.001
    print(f'GATE PASS: n={n}, EBC collapses earlier in {pct:.1%} of matches, '
          f'means {rdf.fc_degree.mean():.3f}/{rdf.fc_ebc.mean():.3f}')

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))

    ax = axes[0]
    ax.scatter(rdf['fc_degree'], rdf['fc_ebc'], s=4, alpha=0.15, c='#1B4F72', linewidths=0)
    ax.plot([0, 0.9], [0, 0.9], 'k--', alpha=0.5, lw=0.8)
    ax.set_xlabel(r'$f_c$ (degree-based attack)')
    ax.set_ylabel(r'$f_c$ (EBC attack)')
    ax.set_title(rf'(a) $f_c$ comparison (N = {n:,})', fontsize=9.5)
    ax.set_xlim(0, 0.9); ax.set_ylim(0, 0.9)
    ax.text(0.04, 0.86, f'{pct:.1%} of matches:\n' + r'$f_c^{\mathrm{EBC}} < f_c^{\mathrm{degree}}$',
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
    ax.set_xlabel(r'$f_c$'); ax.set_ylabel('Count')
    ax.set_title(r'(c) $f_c$ distributions', fontsize=9.5)
    ax.legend(fontsize=7.5, frameon=False)

    for a in axes:
        a.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(_SCRIPT_DIR / 'fig_ebc_summary.pdf')
    fig.savefig(_SCRIPT_DIR / 'fig_ebc_summary.png', dpi=200)
    print('saved fig_ebc_summary.pdf/.png')


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true",
                    help="only reproduce the figure from the shipped results CSV")
    ap.add_argument("--limit", type=int, default=None,
                    help="process only the first N filtered rows (adaptation check)")
    args = ap.parse_args()
    if args.plot:
        run_plot()
    else:
        rdf = run_all(limit=args.limit)
        print_summary(rdf)
