"""
batch_runner.py — Single-match passing network batch analysis orchestrator
==========================================================================
Iterates over all league/season matches, using single-match team networks
as the basic unit:
1. Parse StatsBomb JSON events -> extract passes
2. Build multi-scale (L0-L3) position-dependent directed passing networks
3. Compute structural indicators + bond percolation simulation -> robustness metrics
4. Export match_summary.csv (match level) and batch_summary.csv (season summary level)

Input:  StatsBomb event data directory (DATA_ROOT)
Output: ../../data/match_summary.csv, ../../data/batch_summary.csv

Usage:
    # First set the DATA_ROOT variable to point to the StatsBomb data directory
    cd code/pipeline
    python batch_runner.py          # Full run
    python batch_runner.py --test   # Test mode (EPL 22/23 only, first 50 files)

Last modified: 2026-03-27
"""

import argparse
import pandas as pd
from pathlib import Path
from collections import defaultdict
import traceback

from data_loader import (
    safe_load_match_events,
    extract_passes,
    get_match_files,
    get_teams_from_events,
    load_match_info,
    build_substitution_map,
    apply_substitution_map,
)
from network_builder import ZonalNetworkBuilder
from robustness import (
    edge_removal_simulation,
    multi_trial_edge_removal,
    find_critical_threshold,
    calculate_robustness_integral,
    calculate_molloy_reed_threshold,
)
from compute_directed_indicators import compute_directed_indicators

# ============== Configuration ==============

# Users must set this path to the root directory of StatsBomb event data.
# The directory should contain season subdirectories such as "2_235_2022_23/",
# each containing *_events.json and 00_info_all_matches_*.json files.
DATA_ROOT = Path("/path/to/statsbomb/datos")

# Output directory: relative to this file at ../../data/
_SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = _SCRIPT_DIR.parent.parent / "data"
MATCH_OUTPUT = OUTPUT_DIR / "match_summary.csv"
SEASON_OUTPUT = OUTPUT_DIR / "batch_summary.csv"
SKIPPED_LOG = OUTPUT_DIR / "skipped_files.log"

LEAGUES = {
    2: "Premier League",
    11: "La Liga",
}

SEASONS = {
    235: "2022/23",
    281: "2023/24",
    317: "2024/25",
}

SCALES_TO_ANALYZE = ["L0", "L1", "L2", "L3"]

PERCOLATION_STEPS = 50
PERCOLATION_TRIALS_L2 = 20
PERCOLATION_TRIALS_OTHER = 5

MATCH_RESULT_COLUMNS = [
    "League",
    "Season",
    "Team",
    "TeamID",
    "Venue",
    "MatchID",
    "Opponent",
    "Scale",
    "PassCount",
    "UniquePlayers",
    "Nodes",
    "Edges",
    "AvgDegree",
    "AvgInDegree",
    "AvgOutDegree",
    "Clustering",
    "Assortativity",
    "Reciprocity",
    "FlowHierarchy",
    "InDegreeCV",
    "OutDegreeCV",
    "Mu11_Mu",
    "Triad030T",
    "Triad030C",
    "Triad120D",
    "Triad120U",
    "Triad120C",
    "Triad210",
    "Triad300",
    "CyclicTriadFraction",
    "TransitiveTriadFraction",
    "SCGC_Size",
    "SCGC_Ratio",
    "WGCC_Size",
    "WGCC_Ratio",
    "fc_Error_Strong",
    "R_Error_Strong",
    "ChiMax_Error_Strong",
    "fc_Attack_Strong",
    "R_Attack_Strong",
    "ChiMax_Attack_Strong",
    "fc_Error_Weak",
    "R_Error_Weak",
    "ChiMax_Error_Weak",
    "fc_Attack_Weak",
    "R_Attack_Weak",
    "ChiMax_Attack_Weak",
    "fc_dMR",
    "dMR_Deviation",
]

SEASON_MEAN_COLUMNS = [
    "PassCount",
    "UniquePlayers",
    "Nodes",
    "Edges",
    "AvgDegree",
    "AvgInDegree",
    "AvgOutDegree",
    "Clustering",
    "Assortativity",
    "Reciprocity",
    "FlowHierarchy",
    "InDegreeCV",
    "OutDegreeCV",
    "Mu11_Mu",
    "Triad030T",
    "Triad030C",
    "Triad120D",
    "Triad120U",
    "Triad120C",
    "Triad210",
    "Triad300",
    "CyclicTriadFraction",
    "TransitiveTriadFraction",
    "SCGC_Size",
    "SCGC_Ratio",
    "WGCC_Size",
    "WGCC_Ratio",
    "fc_Error_Strong",
    "R_Error_Strong",
    "ChiMax_Error_Strong",
    "fc_Attack_Strong",
    "R_Attack_Strong",
    "ChiMax_Attack_Strong",
    "fc_Error_Weak",
    "R_Error_Weak",
    "ChiMax_Error_Weak",
    "fc_Attack_Weak",
    "R_Attack_Weak",
    "ChiMax_Attack_Weak",
    "fc_dMR",
    "dMR_Deviation",
]

SEASON_RESULT_COLUMNS = [
    "League",
    "Season",
    "Team",
    "TeamID",
    "Venue",
    "Scale",
    "MatchCount",
    *SEASON_MEAN_COLUMNS,
]


class BatchAnalyzer:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.match_results = []
        self.results = []
        self.skipped_files = []
        self.builder = ZonalNetworkBuilder()

    def log_skipped(self, filepath: str, reason: str):
        self.skipped_files.append(f"{filepath} | {reason}")
        with open(SKIPPED_LOG, "a", encoding="utf-8") as f:
            f.write(f"{filepath} | {reason}\n")

    def process_season(self, competition_id: int, season_id: int, test_mode: bool = False):
        """Process all matches in a single season, building per-match team networks."""
        print(
            f"\nStarting processing: "
            f"{LEAGUES.get(competition_id, 'Unknown')} - {SEASONS.get(season_id, 'Unknown')}"
        )

        try:
            files = get_match_files(str(DATA_ROOT), competition_id, season_id)
        except FileNotFoundError as e:
            print(f"Season data not found: {e}")
            return

        # Load authoritative home/away info from 00_info_all_matches JSON
        try:
            match_info_lookup = load_match_info(str(DATA_ROOT), competition_id, season_id)
            print(f"  - Loaded metadata for {len(match_info_lookup)} matches (home/away/score)")
        except FileNotFoundError as e:
            print(f"  Match info file not found, falling back to Starting XI order: {e}")
            match_info_lookup = {}

        team_match_records = defaultdict(lambda: defaultdict(list))
        team_id_map = {}

        print(f"  - Scanning {len(files)} matches...")
        for i, filepath in enumerate(files):
            if test_mode and i >= 50:
                print("  Test Mode: Stopping loading after 50 files.")
                break

            if i % 50 == 0:
                print(f"    Progress: {i}/{len(files)}")

            events = safe_load_match_events(str(filepath))
            if events is None:
                self.log_skipped(str(filepath), "Load Failed")
                continue

            try:
                match_id = int(filepath.name.split("_")[0])

                # Prefer home/away from match info JSON (authoritative source)
                info = match_info_lookup.get(match_id)
                if info:
                    home_team = info["home_team"]
                    away_team = info["away_team"]
                else:
                    # Fallback: Starting XI order (verified consistent across 2280 matches)
                    home_team, away_team = get_teams_from_events(events)

                if not home_team.get("id") or not away_team.get("id"):
                    self.log_skipped(str(filepath), "No Team Info")
                    continue

                match_team_rows = [
                    (home_team, away_team.get("name"), "Home"),
                    (away_team, home_team.get("name"), "Away"),
                ]

                for team_info, opponent_name, venue in match_team_rows:
                    team_id = team_info["id"]
                    team_name = team_info["name"]
                    team_id_map[team_name] = team_id

                    passes = extract_passes(events, match_id=match_id, team_id=team_id)
                    if not passes:
                        continue

                    sub_map = build_substitution_map(events, team_id=team_id)
                    apply_substitution_map(passes, sub_map)

                    team_match_records[team_name][venue].append(
                        {
                            "match_id": match_id,
                            "opponent": opponent_name,
                            "passes": passes,
                        }
                    )

            except Exception as e:
                print(f"    Error processing file {filepath.name}: {e}")
                traceback.print_exc()
                self.log_skipped(str(filepath), f"Process Error: {e}")

        print(f"  - Analysing single-match networks for {len(team_match_records)} teams...")
        for i, (team_name, venues) in enumerate(team_match_records.items()):
            if test_mode and i >= 1:
                print(f"  Test Mode: Stopping after 1 team ({team_name}).")
                break

            for venue, match_records in venues.items():
                self.analyze_team_season_matches(
                    competition_id=competition_id,
                    season_id=season_id,
                    team_name=team_name,
                    team_id=team_id_map.get(team_name),
                    venue=venue,
                    match_records=match_records,
                )

    def analyze_team_season_matches(
        self,
        competition_id: int,
        season_id: int,
        team_name: str,
        team_id: int,
        venue: str,
        match_records,
    ):
        """Compute match-level indicators, then aggregate by team-season-venue."""
        for scale in SCALES_TO_ANALYZE:
            scale_rows = []
            for match_record in match_records:
                row = self.analyze_single_match(
                    competition_id=competition_id,
                    season_id=season_id,
                    team_name=team_name,
                    team_id=team_id,
                    venue=venue,
                    match_id=match_record["match_id"],
                    opponent=match_record["opponent"],
                    passes=match_record["passes"],
                    scale=scale,
                )
                self.match_results.append(row)
                scale_rows.append(row)

            if scale_rows:
                self.results.append(self.aggregate_match_rows(scale_rows))

    def analyze_single_match(
        self,
        competition_id: int,
        season_id: int,
        team_name: str,
        team_id: int,
        venue: str,
        match_id: int,
        opponent: str,
        passes,
        scale: str,
    ):
        """Analyse a single match team network."""
        graph = self.builder.build_network(passes, scale=scale)
        stats = self.builder.get_network_stats(graph)

        # 11-player invariant check (only meaningful at L0 where zones are fine enough)
        n_unique = stats.get("n_unique_players", 0)
        if scale == "L0" and n_unique != 11:
            msg = (f"UniquePlayers={n_unique} (expected 11) | "
                   f"Match={match_id} Team={team_name} Scale={scale}")
            print(msg)
            self.log_skipped(str(match_id), f"UniquePlayers={n_unique}")

        directed_indicators = compute_directed_indicators(graph)

        n_trials = PERCOLATION_TRIALS_L2 if scale == "L2" else PERCOLATION_TRIALS_OTHER

        sim_error = multi_trial_edge_removal(
            graph,
            method="error",
            n_steps=PERCOLATION_STEPS,
            n_trials=n_trials,
            seed=42,
        )
        sim_attack = edge_removal_simulation(
            graph,
            method="attack",
            n_steps=PERCOLATION_STEPS,
        )

        fc_error_s, chi_error_s = find_critical_threshold(sim_error, "strong")
        fc_error_w, chi_error_w = find_critical_threshold(sim_error, "weak")
        fc_attack_s, chi_attack_s = find_critical_threshold(sim_attack, "strong")
        fc_attack_w, chi_attack_w = find_critical_threshold(sim_attack, "weak")

        r_error_s = calculate_robustness_integral(sim_error, "strong")
        r_error_w = calculate_robustness_integral(sim_error, "weak")
        r_attack_s = calculate_robustness_integral(sim_attack, "strong")
        r_attack_w = calculate_robustness_integral(sim_attack, "weak")

        fc_theory = calculate_molloy_reed_threshold(graph)

        row = {
            "League": LEAGUES.get(competition_id, str(competition_id)),
            "Season": SEASONS.get(season_id, str(season_id)),
            "Team": team_name,
            "TeamID": team_id,
            "Venue": venue,
            "MatchID": match_id,
            "Opponent": opponent,
            "Scale": scale,
            "PassCount": len(passes),
            "UniquePlayers": stats.get("n_unique_players", 0),
            "Nodes": stats["n_nodes"],
            "Edges": stats["n_unique_edges"],
            "AvgDegree": stats["avg_degree"],
            "AvgInDegree": stats.get("avg_in_degree", 0),
            "AvgOutDegree": stats.get("avg_out_degree", 0),
            "Clustering": stats["clustering_coefficient"],
            "Assortativity": stats["assortativity"],
            "Reciprocity": directed_indicators["Reciprocity"],
            "FlowHierarchy": directed_indicators["FlowHierarchy"],
            "InDegreeCV": directed_indicators["InDegreeCV"],
            "OutDegreeCV": directed_indicators["OutDegreeCV"],
            "Mu11_Mu": directed_indicators["Mu11_Mu"],
            "Triad030T": directed_indicators["Triad030T"],
            "Triad030C": directed_indicators["Triad030C"],
            "Triad120D": directed_indicators["Triad120D"],
            "Triad120U": directed_indicators["Triad120U"],
            "Triad120C": directed_indicators["Triad120C"],
            "Triad210": directed_indicators["Triad210"],
            "Triad300": directed_indicators["Triad300"],
            "CyclicTriadFraction": directed_indicators["CyclicTriadFraction"],
            "TransitiveTriadFraction": directed_indicators["TransitiveTriadFraction"],
            "SCGC_Size": stats.get("largest_scc_size", 0),
            "SCGC_Ratio": stats.get("largest_scc_ratio", 0),
            "WGCC_Size": stats.get("largest_wcc_size", 0),
            "WGCC_Ratio": stats.get("largest_wcc_ratio", 0),
            "fc_Error_Strong": fc_error_s,
            "R_Error_Strong": r_error_s,
            "ChiMax_Error_Strong": chi_error_s,
            "fc_Attack_Strong": fc_attack_s,
            "R_Attack_Strong": r_attack_s,
            "ChiMax_Attack_Strong": chi_attack_s,
            "fc_Error_Weak": fc_error_w,
            "R_Error_Weak": r_error_w,
            "ChiMax_Error_Weak": chi_error_w,
            "fc_Attack_Weak": fc_attack_w,
            "R_Attack_Weak": r_attack_w,
            "ChiMax_Attack_Weak": chi_attack_w,
            "fc_dMR": fc_theory,
            "dMR_Deviation": fc_theory - fc_error_s,
        }
        return row

    def aggregate_match_rows(self, match_rows):
        """Aggregate match results into team-season-venue-scale season means."""
        df = pd.DataFrame(match_rows)
        first = df.iloc[0]
        row = {
            "League": first["League"],
            "Season": first["Season"],
            "Team": first["Team"],
            "TeamID": first["TeamID"],
            "Venue": first["Venue"],
            "Scale": first["Scale"],
            "MatchCount": len(df),
        }
        for col in SEASON_MEAN_COLUMNS:
            row[col] = df[col].mean()
        return row

    def save_results(self):
        match_df = pd.DataFrame(self.match_results, columns=MATCH_RESULT_COLUMNS)
        season_df = pd.DataFrame(self.results, columns=SEASON_RESULT_COLUMNS)

        match_df.to_csv(MATCH_OUTPUT, index=False)
        season_df.to_csv(SEASON_OUTPUT, index=False)

        print(f"\nMatch-level results saved: {MATCH_OUTPUT}")
        print(f"Match-level record count: {len(match_df)}")
        print(f"Season summary saved: {SEASON_OUTPUT}")
        print(f"Season summary record count: {len(season_df)}")


def main():
    parser = argparse.ArgumentParser(description="Batch runner for single-match passing networks")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode (Premier League 2022/23, first 50 files, first team only)",
    )
    args = parser.parse_args()

    analyzer = BatchAnalyzer(OUTPUT_DIR)

    if SKIPPED_LOG.exists():
        SKIPPED_LOG.unlink()

    if args.test:
        print("TEST MODE: Processing only Premier League 22/23")
        analyzer.process_season(2, 235, test_mode=True)
    else:
        for comp_id in LEAGUES:
            for season_id in SEASONS:
                analyzer.process_season(comp_id, season_id, test_mode=False)

    analyzer.save_results()


if __name__ == "__main__":
    main()
