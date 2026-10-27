"""
s04_substitute_inheritance.py  Robustness check of the substitute-inheritance version (Section S4)
=================================================================================================
Rebuilds the player-level structural impact of every team and season from the
raw event data with substitute inheritance (a substitute takes over the
identity of the player replaced) and without it, checks the rebuild without
inheritance against the shipped player files, and compares the volume-adjusted
structural residuals of the two versions (Supplementary Section S4).

Input  data/player_impact_{2022_23,202324,202425}.csv, data/match_summary_mc100_guarded.csv,
  data/supplementary/player_impact_{season}_inherit.csv and s4_identities_per_match.csv
  (results of a build, shipped), raw event data under DATA_ROOT (build mode only)
Output  results/player_impact_{season}_inherit.csv, results/s4_identities_per_match.csv
  (build mode), results/s4_substitute_inheritance.csv

Usage
  python code/supplementary/s04_substitute_inheritance.py    # comparison from the shipped files
  DATA_ROOT=/path/to/statsbomb/datos python code/supplementary/s04_substitute_inheritance.py --build

Last modified 2026-10-27
"""

import argparse
import glob
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "code" / "pipeline"))

from data_loader import build_substitution_map          # noqa: E402
from network_builder import ZonalNetworkBuilder           # noqa: E402
from robustness import calculate_metrics_directed         # noqa: E402

DATA_DIR = BASE_DIR / "data"
SUPP_DATA = DATA_DIR / "supplementary"
RESULTS_DIR = BASE_DIR / "results"
# Published analysis (100 Monte Carlo trials at all scales, restricted
# threshold estimator), used here only for the list of L2 team-match networks
PUBLISHED_CSV = DATA_DIR / "match_summary_mc100_guarded.csv"

# Root directory of the StatsBomb event data (set the DATA_ROOT environment variable)
RAW_DIR = os.environ.get("DATA_ROOT", "/path/to/statsbomb/datos")

SEASONS = {"2022/23": "2022_23", "2023/24": "202324", "2024/25": "202425"}
MIN_PLAYED = 5        # matches a player must appear in to be listed
MIN_EDGES = 10        # networks with fewer edges are skipped
GATE_ATOL = 1e-9
IDENTITY_FILE = "s4_identities_per_match.csv"
PLACEHOLDER = "?"     # name used where the event has no player or recipient name

# Supplementary Section S4 as printed
PUBLISHED = {
    "players_per_match_without_low": 14,
    "players_per_match_without_high": 16,
    "players_per_match_with": 11,
    "three_season_players_without": 484,
    "three_season_players_with": 383,
    "cross_version_spearman": 0.89,
    "pct_absdiff_gt_half_sd": 15,
    "stability_with_min": 0.60,
    "stability_with_max": 0.64,
    "stability_without_min": 0.54,
    "stability_without_max": 0.62,
}

builder = ZonalNetworkBuilder()


def inherit_file(tag, folder):
    return folder / f"player_impact_{tag}_inherit.csv"


# ============================================================================
# Build from raw event data
# ============================================================================
def name_substitution_map(events, team_id):
    """Substitution map of one team in one match, by player name.

    Uses the pipeline map by player id (chained substitutions map to the
    original starter) and translates the ids to the names used as node labels.
    """
    id_map = build_substitution_map(events, team_id=team_id)
    id2name = {}
    for e in events:
        for p in (e.get("player", {}), e.get("pass", {}).get("recipient", {}),
                  e.get("substitution", {}).get("replacement", {})):
            if p.get("id") is not None:
                id2name[p["id"]] = p.get("name")
    return {id2name[a]: id2name[b] for a, b in id_map.items()}


def match_impact(events, team, sub_map, info):
    """Add one match to the per-player totals of one team.

    Builds the directed (player, L2 zone) network of completed passes, removes
    each edge in turn and credits the drop of the SCGC fraction to the passer
    and the recipient. Returns the named players of the match, or None if the
    network has fewer than MIN_EDGES edges.
    """
    G = nx.DiGraph()
    match_players = set()
    for e in events:
        if e.get("type", {}).get("name") != "Pass":
            continue
        if e.get("team", {}).get("name") != team:
            continue
        if e.get("pass", {}).get("outcome", {}).get("name", "Complete") in ["Incomplete", "Out", "Unknown"]:
            continue
        p = e.get("player", {}).get("name", PLACEHOLDER)
        r2 = e.get("pass", {}).get("recipient", {}).get("name", PLACEHOLDER)
        p, r2 = sub_map.get(p, p), sub_map.get(r2, r2)
        loc = e.get("location", [60, 40])
        end = e.get("pass", {}).get("end_location", [60, 40])
        s = builder.get_zone_id(loc[0], loc[1], 6, 4)
        t = builder.get_zone_id(end[0], end[1], 6, 4)
        src, tgt = f"{p}@Z{s}", f"{r2}@Z{t}"
        if src != tgt:
            if G.has_edge(src, tgt):
                G[src][tgt]["weight"] += 1
            else:
                G.add_edge(src, tgt, weight=1)
        match_players.add(p)
        match_players.add(r2)
        info.setdefault(p, {"played": 0, "passes": 0, "total_drop": 0})
        info[p]["passes"] += 1

    if G.number_of_edges() < MIN_EDGES:
        return None
    for p in match_players:
        info.setdefault(p, {"played": 0, "passes": 0, "total_drop": 0})
        info[p]["played"] += 1

    scgc0 = calculate_metrics_directed(G)["P_inf_strong"]
    # Removing an edge can lower the SCGC fraction only if the edge lies inside
    # the unique largest strongly connected component; for every other edge the
    # drop is exactly zero, so those removals are skipped.
    sccs = list(nx.strongly_connected_components(G))
    sizes = sorted((len(c) for c in sccs), reverse=True)
    giant = max(sccs, key=len) if len(sizes) == 1 or sizes[0] > sizes[1] else set()
    for u, v in list(G.edges()):
        if u not in giant or v not in giant:
            continue
        attrs = G[u][v]
        G.remove_edge(u, v)
        drop = scgc0 - calculate_metrics_directed(G)["P_inf_strong"]
        G.add_edge(u, v, **attrs)
        if drop > 0:
            p1, p2 = u.split("@")[0], v.split("@")[0]
            if p1 in info:
                info[p1]["total_drop"] += drop
            if p2 in info:
                info[p2]["total_drop"] += drop
    return match_players - {PLACEHOLDER}


def team_season(args):
    """Both versions for one team and season, matches in file order."""
    team, team_id, season, league, match_files = args
    info = {"without": {}, "with": {}}
    identities = []
    for match_id, path in match_files:
        with open(path) as f:
            events = json.load(f)
        named = {}
        for version, sub_map in [("without", {}), ("with", name_substitution_map(events, team_id))]:
            named[version] = match_impact(events, team, sub_map, info[version])
        if named["without"] is not None and named["with"] is not None:
            identities.append({"League": league, "Season": season, "Team": team,
                               "MatchID": match_id,
                               "players_without": len(named["without"]),
                               "players_with": len(named["with"])})
    rows = {}
    for version, d in info.items():
        rows[version] = [dict(team=team, player=p, played=v["played"], passes=v["passes"],
                              ppg=v["passes"] / v["played"], total_drop=v["total_drop"],
                              impact_per_match=v["total_drop"] / v["played"])
                         for p, v in d.items() if v["played"] >= MIN_PLAYED]
    # Inheritance only renames passers and recipients, so the team total of
    # passes over all players must be the same in both versions
    same_passes = (sum(v["passes"] for v in info["without"].values())
                   == sum(v["passes"] for v in info["with"].values()))
    return team, season, len(match_files), rows, identities, same_passes


def add_ranks(df):
    df["team_impact_rank"] = df.groupby("team")["impact_per_match"].rank(ascending=False).astype(int)
    df["team_pass_rank"] = df.groupby("team")["ppg"].rank(ascending=False).astype(int)
    df["rank_gap"] = df["team_pass_rank"] - df["team_impact_rank"]
    return df


def build(workers):
    files = {int(os.path.basename(f).split("_")[0]): f
             for f in glob.glob(os.path.join(RAW_DIR, "*", "*_events.json"))}
    if not files:
        print(f"No event files under DATA_ROOT={RAW_DIR}. Set DATA_ROOT to the StatsBomb data folder.")
        sys.exit(1)
    ms = pd.read_csv(PUBLISHED_CSV, usecols=["League", "Season", "Team", "TeamID", "MatchID", "Scale"])
    l2 = ms[ms.Scale == "L2"]
    jobs = []
    for season in SEASONS:
        for team in sorted(l2[l2.Season == season].Team.unique()):
            rows = l2[(l2.Season == season) & (l2.Team == team)]
            missing = [m for m in rows.MatchID if int(m) not in files]
            if missing:
                print(f"Event files missing for {team} {season}: {missing[:5]}")
                sys.exit(1)
            jobs.append((team, int(rows.TeamID.iloc[0]), season, rows.League.iloc[0],
                         [(int(m), files[int(m)]) for m in rows.MatchID]))

    print(f"Rebuilding {len(jobs)} team-seasons with {workers} workers", flush=True)
    t0 = time.time()
    results = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(team_season, j) for j in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            team, season, n, rows, ident, same = fut.result()
            results[(team, season)] = (rows, ident, same)
            el = time.time() - t0
            print(f"  [{i}/{len(jobs)}] {season} {team}  {n} matches  "
                  f"elapsed {el / 60:.1f} min, remaining about {el / i * (len(jobs) - i) / 60:.1f} min",
                  flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    gate = []
    identities = []
    for season, tag in SEASONS.items():
        keys = [k for k in results if k[1] == season]
        keys.sort(key=lambda k: k[0])
        without = add_ranks(pd.DataFrame([r for k in keys for r in results[k][0]["without"]]))
        with_ = add_ranks(pd.DataFrame([r for k in keys for r in results[k][0]["with"]]))
        identities += [x for k in keys for x in results[k][1]]
        with_.to_csv(inherit_file(tag, RESULTS_DIR), index=False)
        print(f"Saved {inherit_file(tag, RESULTS_DIR).relative_to(BASE_DIR)} ({len(with_)} players)")

        # Gate, the rebuild without inheritance must equal the shipped file
        shipped = pd.read_csv(DATA_DIR / f"player_impact_{tag}.csv")
        m = without.merge(shipped, on=["team", "player"], how="outer",
                          suffixes=("_new", "_shipped"), indicator=True)
        unmatched = int((m["_merge"] != "both").sum())
        both = m[m["_merge"] == "both"]
        diff = max(float((both[f"{c}_new"] - both[f"{c}_shipped"]).abs().max())
                   for c in ["played", "passes", "ppg", "total_drop", "impact_per_match"])
        gate.append((f"Rebuild without inheritance {season}, rows not in both", unmatched, 0, unmatched == 0))
        gate.append((f"Rebuild without inheritance {season}, largest difference", diff, 0,
                     diff <= GATE_ATOL))
        n_diff = sum(not results[k][2] for k in keys)
        gate.append((f"Team pass totals equal in both versions {season}, teams that differ",
                     n_diff, 0, n_diff == 0))
    pd.DataFrame(identities).to_csv(RESULTS_DIR / IDENTITY_FILE, index=False)
    print(f"Saved {(RESULTS_DIR / IDENTITY_FILE).relative_to(BASE_DIR)} ({len(identities)} team-match networks)")
    print(f"Build finished in {(time.time() - t0) / 60:.1f} min\n")
    return gate


# ============================================================================
# Comparison of the two versions
# ============================================================================
def compute_residual(df, metric_col="impact_per_match"):
    """Within each team, regress metric_col on passes per game and take the
    residual in units of its SD; a player listed for two teams in one season
    gets the mean. Returns {player name: residual}."""
    result = {}
    for team, grp in df.groupby("team"):
        if len(grp) < 5:
            continue
        X = grp["ppg"].values.reshape(-1, 1)
        y = grp[metric_col].values
        reg = LinearRegression().fit(X, y)
        res = y - reg.predict(X)
        std = res.std()
        for i, (_, row) in enumerate(grp.iterrows()):
            z = res[i] / std if std > 0 else 0
            result.setdefault(row["player"], []).append(z)
    return {p: np.mean(v) for p, v in result.items()}


def residual_set(dfs):
    """Players present in all three seasons, the Spearman correlation of their
    residuals between each pair of seasons, and their three-season mean."""
    r = {s: compute_residual(d) for s, d in dfs.items()}
    common = sorted(set.intersection(*(set(v) for v in r.values())))
    seasons = list(dfs)
    stab = []
    for a, b in [(0, 1), (1, 2), (0, 2)]:
        stab.append(stats.spearmanr([r[seasons[a]][p] for p in common],
                                    [r[seasons[b]][p] for p in common])[0])
    mean3 = {p: np.mean([r[s][p] for s in seasons]) for p in common}
    return common, stab, mean3


def find_inheritance_files():
    for folder in [RESULTS_DIR, SUPP_DATA]:
        if all(inherit_file(tag, folder).exists() for tag in SEASONS.values()):
            return folder
    return None


def compare(gate):
    folder = find_inheritance_files()
    if folder is None:
        print("Inheritance files not found in results/ or data/supplementary/. "
              "Run with --build and DATA_ROOT set first.")
        sys.exit(1)
    print(f"Inheritance files read from {folder.relative_to(BASE_DIR)}/")
    without = {s: pd.read_csv(DATA_DIR / f"player_impact_{t}.csv") for s, t in SEASONS.items()}
    with_ = {s: pd.read_csv(inherit_file(t, folder)) for s, t in SEASONS.items()}

    # Passes without a recipient name go to the placeholder name '?', which is
    # listed like a player in both versions and in all three seasons and is
    # therefore one of the 484 and 383 three-season players.
    c_wo, stab_wo, m_wo = residual_set(without)
    c_wi, stab_wi, m_wi = residual_set(with_)
    both = sorted(set(m_wo) & set(m_wi))
    v_wo = np.array([m_wo[p] for p in both])
    v_wi = np.array([m_wi[p] for p in both])
    rho_cross = stats.spearmanr(v_wo, v_wi)[0]
    pct = 100 * np.mean(np.abs(v_wo - v_wi) > 0.5)

    checks = list(gate)

    def check(label, got, expected, tol):
        checks.append((label, got, expected, abs(got - expected) <= tol + 1e-12))

    ident_path = None
    for f in [RESULTS_DIR / IDENTITY_FILE, SUPP_DATA / IDENTITY_FILE]:
        if f.exists():
            ident_path = f
            break
    if ident_path is not None:
        ident = pd.read_csv(ident_path)
        q10, q90 = np.percentile(ident.players_without, [10, 90])
        n_wi_max = int(ident.players_with.max())
        share_11 = 100 * (ident.players_with == 11).mean()
        share_14_16 = 100 * ident.players_without.between(14, 16).mean()
        print(f"Named players per team-match network ({len(ident)} networks)")
        print("  without inheritance  " + ", ".join(
            f"{k}: {v}" for k, v in ident.players_without.value_counts().sort_index().items()))
        print("  with inheritance     " + ", ".join(
            f"{k}: {v}" for k, v in ident.players_with.value_counts().sort_index().items()))
        print(f"  without inheritance, 10th to 90th percentile {q10:.0f} to {q90:.0f}, "
              f"14 to 16 in {share_14_16:.1f}%; with inheritance, at most {n_wi_max}, "
              f"exactly 11 in {share_11:.2f}%\n")
        check("Section S4 players per match without inheritance, 10th percentile", q10,
              PUBLISHED["players_per_match_without_low"], 0)
        check("Section S4 players per match without inheritance, 90th percentile", q90,
              PUBLISHED["players_per_match_without_high"], 0)
        check("Section S4 players per match with inheritance, maximum", n_wi_max,
              PUBLISHED["players_per_match_with"], 0)
    else:
        print(f"{IDENTITY_FILE} not found; the players-per-match check needs --build.\n")

    check("Section S4 players in all three seasons without inheritance", len(c_wo),
          PUBLISHED["three_season_players_without"], 0)
    check("Section S4 players in all three seasons with inheritance", len(c_wi),
          PUBLISHED["three_season_players_with"], 0)
    check("Section S4 Spearman rho of three-season mean residuals", rho_cross,
          PUBLISHED["cross_version_spearman"], 0.005)
    check("Section S4 percent with |difference| > 0.5 SD", pct,
          PUBLISHED["pct_absdiff_gt_half_sd"], 0.5)
    check("Section S4 cross-season stability with inheritance, lowest", min(stab_wi),
          PUBLISHED["stability_with_min"], 0.005)
    check("Section S4 cross-season stability with inheritance, highest", max(stab_wi),
          PUBLISHED["stability_with_max"], 0.005)
    check("Section S4 cross-season stability without inheritance, lowest", min(stab_wo),
          PUBLISHED["stability_without_min"], 0.005)
    check("Section S4 cross-season stability without inheritance, highest", max(stab_wo),
          PUBLISHED["stability_without_max"], 0.005)
    checks.append(("Three-season players with inheritance are a subset of those without",
                   float(set(c_wi) <= set(c_wo)), 1.0, set(c_wi) <= set(c_wo)))

    pairs = ["2022/23 vs 2023/24", "2023/24 vs 2024/25", "2022/23 vs 2024/25"]
    summary = pd.DataFrame(
        [{"quantity": "three_season_players", "without_inheritance": len(c_wo),
          "with_inheritance": len(c_wi)}]
        + [{"quantity": f"stability {p}", "without_inheritance": a, "with_inheritance": b}
           for p, a, b in zip(pairs, stab_wo, stab_wi)]
        + [{"quantity": "players in both versions", "without_inheritance": len(both),
            "with_inheritance": len(both)},
           {"quantity": "spearman rho of three-season mean residuals",
            "without_inheritance": rho_cross, "with_inheritance": rho_cross},
           {"quantity": "percent with |difference| > 0.5 SD",
            "without_inheritance": pct, "with_inheritance": pct}])
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULTS_DIR / "s4_substitute_inheritance.csv", index=False, float_format="%.6g")

    print("Section S4  Structural residuals without and with substitute inheritance")
    print(f"  Players in all three seasons  without {len(c_wo)}, with {len(c_wi)}")
    for p, a, b in zip(pairs, stab_wo, stab_wi):
        print(f"  Cross-season stability {p}  without {a:.3f}, with {b:.3f}")
    print(f"  Three-season mean residuals of the {len(both)} players in both versions  "
          f"Spearman rho = {rho_cross:.3f}, |difference| > 0.5 SD for {pct:.1f}%")
    print(f"Saved {(RESULTS_DIR / 's4_substitute_inheritance.csv').relative_to(BASE_DIR)}\n")

    print("Acceptance check against Supplementary Section S4")
    for label, got, exp, ok in checks:
        print(f"  {label:<72} {got:12.6g} vs {exp:<6g} {'PASS' if ok else 'FAIL'}")
    failed = [c for c in checks if not c[3]]
    if failed:
        print(f"GATE FAILED  {len(failed)} of {len(checks)} values differ")
        for label, got, exp, _ in failed:
            print(f"  {label}  computed {got:.5f}, expected {exp}")
        sys.exit(1)
    print(f"GATE PASS  all {len(checks)} checks match")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--build", action="store_true",
                    help="rebuild both versions from the raw event data under DATA_ROOT")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    gate = build(args.workers) if args.build else []
    compare(gate)


if __name__ == "__main__":
    main()
