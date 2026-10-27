"""
in_out_degree_correlation.py  Node-level correlation between in-degree and out-degree (Section 3.3)
==================================================================================================
For every team-match network at L0 to L3, the Pearson correlation between the
in-degree and the out-degree of its nodes (isolated nodes removed). Section 3.3
reports the median over the 4,560 networks of each resolution, -0.110 at L0,
0.447 at L1, 0.747 at L2 and 0.809 at L3. The full run rebuilds all networks
from the raw event data and writes the per-network values; without raw data the
medians are checked from the shipped per-network file.

Input  DATA_ROOT (StatsBomb event JSON, full run) or
       data/supplementary/in_out_degree_correlation.csv (default)
Output  results/in_out_degree_correlation.csv (full run), printed medians with PASS or FAIL

Usage
  python code/in_out_degree_correlation.py                                    # shipped values
  DATA_ROOT=/path/to/statsbomb/datos python code/in_out_degree_correlation.py --full

Last modified 2026-10-27
"""

import argparse
import os
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "code" / "pipeline"))

SHIPPED = BASE_DIR / "data" / "supplementary" / "in_out_degree_correlation.csv"
RESULTS_DIR = BASE_DIR / "results"
SCALES = ["L0", "L1", "L2", "L3"]
LEAGUES = {2: "Premier League", 11: "La Liga"}
SEASONS = {235: "2022/23", 281: "2023/24", 317: "2024/25"}
PUBLISHED_MEDIAN = {"L0": -0.110, "L1": 0.447, "L2": 0.747, "L3": 0.809}


def in_out_corr(graph):
    """Pearson correlation of node in- and out-degree on the simple directed graph without isolates."""
    G = nx.DiGraph(graph)
    G.remove_nodes_from(list(nx.isolates(G)))
    k_in = np.array([d for _, d in G.in_degree()], dtype=float)
    k_out = np.array([d for _, d in G.out_degree()], dtype=float)
    if len(k_in) > 2 and k_in.std() > 0 and k_out.std() > 0:
        return float(np.corrcoef(k_in, k_out)[0, 1])
    return 0.0


def full_run(data_root):
    from data_loader import (safe_load_match_events, extract_passes, get_match_files,
                             get_teams_from_events, build_substitution_map, apply_substitution_map)
    from network_builder import ZonalNetworkBuilder

    builder = ZonalNetworkBuilder()
    rows, t0 = [], time.time()
    for league_id, league in LEAGUES.items():
        for season_id, season in SEASONS.items():
            files = get_match_files(str(data_root), league_id, season_id)
            for path in files:
                events = safe_load_match_events(str(path))
                if events is None:
                    continue
                match_id = int(path.name.split("_")[0])
                home, away = get_teams_from_events(events)
                if not home or not away:
                    continue
                for team, venue in [(home, "Home"), (away, "Away")]:
                    team_id = int(team["id"])
                    passes = extract_passes(events, match_id=match_id, team_id=team_id)
                    if not passes:
                        continue
                    apply_substitution_map(passes, build_substitution_map(events, team_id=team_id))
                    for scale in SCALES:
                        graph = builder.build_network(passes, scale=scale)
                        rows.append(dict(League=league, Season=season, Team=team["name"], TeamID=team_id,
                                         Venue=venue, MatchID=match_id, Scale=scale,
                                         InOutDegreeCorr=in_out_corr(graph)))
            print(f"  {league} {season}: {len(rows)} networks so far ({time.time() - t0:.0f} s)", flush=True)
    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "in_out_degree_correlation.csv", index=False)
    print(f"Saved: results/in_out_degree_correlation.csv ({len(df)} rows)")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="rebuild all networks from DATA_ROOT")
    args = ap.parse_args()

    if args.full:
        data_root = Path(os.environ.get("DATA_ROOT", "/path/to/statsbomb/datos"))
        if not data_root.exists():
            sys.exit(f"DATA_ROOT not found: {data_root}")
        df = full_run(data_root)
        shipped = pd.read_csv(SHIPPED)
        key = ["MatchID", "TeamID", "Scale"]
        merged = df.merge(shipped[key + ["InOutDegreeCorr"]], on=key, suffixes=("", "_shipped"))
        same = len(merged) == len(df) == len(shipped) and \
            np.allclose(merged.InOutDegreeCorr, merged.InOutDegreeCorr_shipped, atol=1e-12)
        print(f"Per-network values equal the shipped file: {same}")
        if not same:
            sys.exit(1)
    else:
        df = pd.read_csv(SHIPPED)

    failures = []
    print(f"\n{'Scale':5s} {'networks':>9s} {'median':>8s}   article")
    for scale in SCALES:
        values = df[df.Scale == scale].InOutDegreeCorr
        med = values.median()
        ok = len(values) == 4560 and abs(med - PUBLISHED_MEDIAN[scale]) <= 5e-4
        print(f"{scale:5s} {len(values):9d} {med:8.3f}   {PUBLISHED_MEDIAN[scale]:.3f}  {'PASS' if ok else 'FAIL'}")
        if not ok:
            failures.append(scale)
    if failures:
        print("ACCEPTANCE CHECK FAILED for", ", ".join(failures))
        sys.exit(1)
    print("ACCEPTANCE CHECK PASSED  medians match Section 3.3")


if __name__ == "__main__":
    main()
