"""
s01_descriptive.py  Descriptive network statistics across the four scales (Table S1)
====================================================================================
Mean and SD of 14 metrics per spatial scale over the 240 team-season-venue
aggregates of the 4,560 team-match networks (no SCGC filtering) of the
published analysis (100 Monte Carlo trials at all scales, restricted threshold
estimator), and the Spearman rank correlations between adjacent scales given
in Supplementary Section S1.

Input  data/match_summary_mc100_guarded.csv
Output  results/tableS1.csv, results/s1_rank_consistency.csv

Usage
  python code/supplementary/s01_descriptive.py

Last modified 2026-10-27
"""

import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import pandas as pd
from scipy import stats as sps

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
# Published analysis (100 Monte Carlo trials at all scales, restricted
# threshold estimator)
PUBLISHED = DATA_DIR / "match_summary_mc100_guarded.csv"

SCALES = ["L0", "L1", "L2", "L3"]
AGG_KEY = ["Team", "Season", "Venue"]

# Supplementary Table S1 as printed
# (column, row label, means L0 to L3, SDs L0 to L3, printed decimals)
TABLE_S1 = [
    ("Nodes", "Nodes", [444.2, 276.8, 131.8, 50.5], [79.4, 33.1, 8.7, 1.7], 1),
    ("Edges", "Edges", [354.9, 344.3, 305.0, 220.5], [88.6, 83.1, 65.7, 35.3], 1),
    ("AvgDegree", "<k>", [0.78, 1.21, 2.28, 4.35], [0.06, 0.16, 0.37, 0.60], 2),
    ("Clustering", "C", [0.002, 0.013, 0.075, 0.262], [0.001, 0.004, 0.018, 0.039], 3),
    ("Assortativity", "r", [0.040, 0.055, 0.042, 0.007], [0.021, 0.025, 0.022, 0.022], 3),
    ("Reciprocity", "rho", [0.022, 0.076, 0.211, 0.427], [0.006, 0.018, 0.036, 0.045], 3),
    ("FlowHierarchy", "h", [0.959, 0.659, 0.240, 0.051], [0.021, 0.116, 0.080, 0.021], 3),
    ("OutDegreeCV", "CV_out", [0.894, 0.943, 0.939, 0.752], [0.022, 0.035, 0.031, 0.025], 3),
    ("Mu11_Mu", "mu11/mu", [0.73, 1.72, 3.78, 6.29], [0.14, 0.37, 0.76, 0.94], 2),
    ("CyclicTriadFraction", "f_cyc", [0.0004, 0.0012, 0.0067, 0.0360],
     [0.0003, 0.0003, 0.0013, 0.0059], 4),
    ("pc_Attack_Strong", "fc_Attack", [0.023, 0.113, 0.403, 0.611],
     [0.015, 0.061, 0.084, 0.052], 3),
    ("pc_Error_Strong", "fc_Error", [0.146, 0.231, 0.580, 0.804],
     [0.094, 0.091, 0.091, 0.045], 3),
    ("R_Attack_Strong", "R_Attack", [0.004, 0.038, 0.232, 0.509],
     [0.001, 0.023, 0.067, 0.056], 3),
    ("R_Error_Strong", "R_Error", [0.005, 0.054, 0.243, 0.494],
     [0.001, 0.028, 0.057, 0.050], 3),
]

# Four printed cells lie on a rounding boundary and are one unit above the
# directly rounded value, which matches rounding first to four decimals and
# then to the printed precision (for example 0.03948 to 0.0395 to 0.040).
# For these cells the check applies that two-step rounding; every other cell
# must lie within half a unit of the printed value.
TWO_STEP_ROUNDING = {
    ("Assortativity", "L0", "mean"),
    ("Reciprocity", "L0", "mean"),
    ("Reciprocity", "L1", "sd"),
    ("R_Attack_Strong", "L0", "sd"),
}

# Supplementary Section S1, Spearman rank correlations between adjacent scales
# (all P < 0.001)
RANK_RHO = [
    ("pc_Attack_Strong", "L0", "L1", -0.289),
    ("pc_Attack_Strong", "L1", "L2", 0.897),
    ("pc_Attack_Strong", "L2", "L3", 0.861),
    ("R_Attack_Strong", "L1", "L2", 0.950),
    ("R_Attack_Strong", "L2", "L3", 0.963),
]


def round_half_up(x, dp):
    return float(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-dp),
                                                  rounding=ROUND_HALF_UP))


def aggregate(df, scale):
    """Team-season-venue means of the single-match networks at one scale."""
    d = df[df.Scale == scale]
    return d.groupby(AGG_KEY, as_index=False).mean(numeric_only=True)


def rank_rho(aggs, col, s1, s2):
    a = aggs[s1].set_index(AGG_KEY)[col]
    b = aggs[s2].set_index(AGG_KEY)[col]
    common = a.index.intersection(b.index)
    return sps.spearmanr(a.loc[common], b.loc[common]), len(common)


def main():
    df = pd.read_csv(PUBLISHED)
    aggs = {s: aggregate(df, s) for s in SCALES}
    checks = []

    for s in SCALES:
        n_net = int((df.Scale == s).sum())
        checks.append((f"{s} single-match networks", n_net, 4560, n_net == 4560))
        checks.append((f"{s} team-season-venue aggregates", len(aggs[s]), 240,
                       len(aggs[s]) == 240))

    rows = []
    for col, label, means, sds, dp in TABLE_S1:
        row = {"metric": label, "column": col, "decimals": dp}
        for i, s in enumerate(SCALES):
            values = {"mean": aggs[s][col].mean(), "sd": aggs[s][col].std(ddof=1)}
            published = {"mean": means[i], "sd": sds[i]}
            for stat, got in values.items():
                row[f"{s}_{stat}"] = got
                exp = published[stat]
                if (col, s, stat) in TWO_STEP_ROUNDING:
                    ok = round_half_up(round_half_up(got, 4), dp) == exp
                    tag = f"Table S1 {label} {s} {stat} (two-step rounding)"
                else:
                    ok = abs(got - exp) <= 0.5 * 10 ** (-dp) + 1e-12
                    tag = f"Table S1 {label} {s} {stat}"
                checks.append((tag, got, exp, ok))
        rows.append(row)
    table = pd.DataFrame(rows)

    rho_rows = []
    for col, s1, s2, exp in RANK_RHO:
        (rho, p), n = rank_rho(aggs, col, s1, s2)
        rho_rows.append({"metric": col, "scales": f"{s1}-{s2}", "n": n,
                         "spearman_rho": rho, "p": p})
        checks.append((f"Section S1 rho {col} {s1}-{s2}", rho, exp,
                       abs(rho - exp) <= 0.0005 + 1e-12))
        checks.append((f"Section S1 P {col} {s1}-{s2} < 0.001", p, 0.001, p < 0.001))
    rho_table = pd.DataFrame(rho_rows)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS_DIR / "tableS1.csv", index=False, float_format="%.6f")
    rho_table.to_csv(RESULTS_DIR / "s1_rank_consistency.csv", index=False,
                     float_format="%.6g")

    print("Table S1  Descriptive network statistics, mean +/- SD over 240 "
          "team-season-venue aggregates per scale")
    print(f"{'Metric':<11}" + "".join(f"{s:>21}" for s in SCALES))
    for _, r in table.iterrows():
        dp = int(r["decimals"])
        sign = "+" if r["column"] == "Assortativity" else ""
        cells = ""
        for s in SCALES:
            m = format(r[f"{s}_mean"], sign + f".{dp}f")
            m += "*" if (r["column"], s, "mean") in TWO_STEP_ROUNDING else ""
            sd = format(r[f"{s}_sd"], f".{dp}f")
            sd += "*" if (r["column"], s, "sd") in TWO_STEP_ROUNDING else ""
            cells += f"{m + ' +/- ' + sd:>21}"
        print(f"{r['metric']:<11}{cells}")
    print("* one unit higher in the printed Table S1 (rounded first to four decimals)")
    print("\nSection S1  Spearman rank correlation of team-season-venue rankings "
          "between adjacent scales")
    for _, r in rho_table.iterrows():
        print(f"  {r['metric']:<17} {r['scales']}  rho_s = {r['spearman_rho']:+.3f}  "
              f"P = {r['p']:.1e}  n = {r['n']}")
    print(f"\nSaved {(RESULTS_DIR / 'tableS1.csv').relative_to(BASE_DIR)}, "
          f"{(RESULTS_DIR / 's1_rank_consistency.csv').relative_to(BASE_DIR)}\n")

    print("Acceptance check against Supplementary Table S1 and Section S1")
    for label, got, exp, ok in checks:
        print(f"  {label:<58} {got:12.6g} vs {exp:<9g} {'PASS' if ok else 'FAIL'}")
    failed = [c for c in checks if not c[3]]
    if failed:
        print(f"GATE FAILED  {len(failed)} of {len(checks)} values differ from the published values")
        for label, got, exp, _ in failed:
            print(f"  {label}  computed {got:.5f}, published {exp}")
        sys.exit(1)
    print(f"GATE PASS  all {len(checks)} values match")


if __name__ == "__main__":
    main()
