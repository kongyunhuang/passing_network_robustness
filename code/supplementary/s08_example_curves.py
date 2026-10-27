"""
s08_example_curves.py  Trial-averaged SCGC and susceptibility curves for six networks (Fig. S4)
==============================================================================================
Rebuilds six representative networks from the raw StatsBomb event data (three
L3 networks and one L2 network whose unrestricted estimate falls on the
late-stage plateau, two L2 networks with a clear interior maximum), runs 100
random-failure trials with seed 42 on each, and plots P_inf(f) and chi(f) with
the unrestricted (red) and restricted (green) estimates of f_c. League, season
and team of each network are read from the published analysis (100 Monte Carlo
trials at all scales, restricted threshold estimator).

Input  data/match_summary_mc100_guarded.csv, DATA_ROOT (StatsBomb event JSON)
Output  figures/figS4.png, figures/figS4.pdf

Usage
  DATA_ROOT=/path/to/statsbomb/datos python code/supplementary/s08_example_curves.py

Last modified 2026-10-27
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "code" / "pipeline"))

from data_loader import (                    # noqa: E402
    safe_load_match_events, extract_passes,
    build_substitution_map, apply_substitution_map, get_match_files,
)
from network_builder import ZonalNetworkBuilder      # noqa: E402
from robustness import (                             # noqa: E402
    multi_trial_edge_removal, find_critical_threshold,
)

PUBLISHED_RESULTS = BASE_DIR / "data" / "match_summary_mc100_guarded.csv"
OUTPUT_DIR = BASE_DIR / "figures"

# Root directory of the StatsBomb event data (set the DATA_ROOT environment variable)
DATA_DIR = os.environ.get("DATA_ROOT", "/path/to/statsbomb/datos")

LEAGUE_TO_COMP = {"Premier League": 2, "La Liga": 11}
SEASON_TO_ID = {"2022/23": 235, "2023/24": 281, "2024/25": 317}

N_TRIALS, SEED, N_STEPS = 100, 42, 50

# Restricted estimator of the article, the susceptibility maximum is searched
# only where P_inf(f) >= 0.05 P_inf(0)
PINF_RESTRICTION_FRACTION = 0.05
# Green dotted reference line drawn at P_inf = 0.05 in every panel. For the six
# networks shown, the restriction floor 0.05 P_inf(0) lies between 0.027 and 0.045.
PINF_REFERENCE_LINE = 0.05

# The six networks, in panel order (MatchID, TeamID, scale)
CASES = [
    (3894867, 1049, "L3"),
    (3946688, 422, "L3"),
    (3850864, 216, "L3"),
    (3837547, 55, "L2"),
    (3837505, 33, "L2"),
    (3837320, 28, "L2"),
]

# Panel labels and f_c values printed in the published Fig. S4 (2 decimals),
# (verdict, unrestricted f_c, restricted f_c)
PUBLISHED_PANELS = [
    ("spurious peak", 1.00, 0.85),
    ("spurious peak", 0.98, 0.81),
    ("spurious peak", 1.00, 0.69),
    ("spurious peak", 0.98, 0.59),
    ("normal", 0.63, 0.63),
    ("normal", 0.61, 0.61),
]


def build_graph(league, season, match_id, team_id, scale):
    """Rebuild one single-match team network exactly as in the main pipeline."""
    files = get_match_files(DATA_DIR, LEAGUE_TO_COMP[league], SEASON_TO_ID[season])
    fp = [f for f in files if f.name.startswith(f"{match_id}_")][0]
    ev = safe_load_match_events(str(fp))
    ps = extract_passes(ev, match_id=match_id, team_id=team_id)
    apply_substitution_map(ps, build_substitution_map(ev, team_id=team_id))
    return ZonalNetworkBuilder().build_network(ps, scale=scale)


def load_cases():
    """League, season and team of each network, with its published f_c values."""
    df = pd.read_csv(PUBLISHED_RESULTS)
    cases = []
    for match_id, team_id, scale in CASES:
        row = df[(df.MatchID == match_id) & (df.TeamID == team_id) & (df.Scale == scale)]
        if len(row) != 1:
            sys.exit(f"Network {match_id} {team_id} {scale} not found once in {PUBLISHED_RESULTS.name}")
        r = row.iloc[0]
        cases.append({"League": r.League, "Season": r.Season, "MatchID": match_id,
                      "TeamID": team_id, "Team": r.Team, "Scale": scale,
                      "fc_restricted_published": r.pc_Error_Strong,
                      "fc_unrestricted_published": r.fcE_raw_strong})
    return cases


def check_published(cases, computed):
    """Acceptance check against the published Fig. S4 and the published analysis."""
    print("\nCheck  f_c values against the published Fig. S4 (tolerance 0.005) "
          "and the published analysis (tolerance 1e-9)")
    ok_all = True
    for c, (verdict, fc_raw, fc_res), (p_verdict, p_raw, p_res) in zip(
            cases, computed, PUBLISHED_PANELS):
        ok_fig = (verdict == p_verdict and abs(fc_raw - p_raw) <= 0.005
                  and abs(fc_res - p_res) <= 0.005)
        ok_csv = (abs(fc_raw - c["fc_unrestricted_published"]) <= 1e-9
                  and abs(fc_res - c["fc_restricted_published"]) <= 1e-9)
        ok_all &= ok_fig and ok_csv
        print(f"  {c['Team']:<16} {c['Scale']}  {verdict:<13} unrestricted {fc_raw:.4f}, "
              f"restricted {fc_res:.4f}  vs figure {p_raw:.2f}, {p_res:.2f} "
              f"{'PASS' if ok_fig else 'FAIL'}  vs analysis "
              f"{c['fc_unrestricted_published']:.4f}, {c['fc_restricted_published']:.4f} "
              f"{'PASS' if ok_csv else 'FAIL'}")
    if not ok_all:
        print("CHECK FAILED, the curves differ from the published figure.")
        sys.exit(1)
    print("CHECK ALL PASS")


def main():
    cases = load_cases()
    computed = []

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, c in zip(axes.flat, cases):
        G = build_graph(c["League"], c["Season"], int(c["MatchID"]), int(c["TeamID"]), c["Scale"])
        sim = multi_trial_edge_removal(G, method="error", n_steps=N_STEPS, n_trials=N_TRIALS, seed=SEED)
        fc_raw, chi_raw = find_critical_threshold(sim, "strong")
        fc_g, chi_g = find_critical_threshold(sim, "strong",
                                              min_pinf_fraction=PINF_RESTRICTION_FRACTION)
        ax.plot(sim.f, sim.P_inf_strong, color="#1f77b4", lw=2, label=r"$P_\infty$ (SCGC)")
        ax2 = ax.twinx()
        ax2.plot(sim.f, sim.Chi_strong, color="#7f7f7f", lw=1.2, label=r"$\chi$")
        ax2.axhline(0.98, color="#bbbbbb", ls=":", lw=0.8)
        ax.axhline(PINF_REFERENCE_LINE, color="green", ls=":", lw=0.8)
        ax.annotate("", xy=(fc_raw, 0.02), xytext=(fc_raw, 0.28),
                    arrowprops=dict(color="crimson", width=2, headwidth=8))
        if not np.isnan(fc_g):
            ax.annotate("", xy=(fc_g, 0.02), xytext=(fc_g, 0.28),
                        arrowprops=dict(color="green", width=2, headwidth=8, alpha=0.7))
        verdict = "spurious peak" if (fc_raw >= 0.9 and chi_raw < 1.05) else "normal"
        computed.append((verdict, fc_raw, fc_g))
        ax.set_title(f"{c['Team']} {c['Scale']} ({verdict})\n"
                     rf"unrestricted $f_{{c,\mathrm{{{c['Scale']}}}}}$={fc_raw:.2f} (red) | restricted $f_{{c,\mathrm{{{c['Scale']}}}}}$={fc_g:.2f} (green)", fontsize=9)
        ax.set_xlabel(r"$f$ (removed fraction)"); ax.set_ylabel(r"$P_\infty$")
        ax2.set_ylabel(r"$\chi$")
        ax.set_ylim(0, 1.02)

    check_published(cases, computed)

    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_DIR / "figS4.png", dpi=180)
    plt.savefig(OUTPUT_DIR / "figS4.pdf")
    print("Saved: figures/figS4.png, figures/figS4.pdf")


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times"],
                         "mathtext.fontset": "cm", "pdf.fonttype": 42,
                         "axes.unicode_minus": False})
    main()
