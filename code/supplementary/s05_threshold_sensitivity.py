"""
s05_threshold_sensitivity.py  SCGC filtering threshold sensitivity analysis (Tables S4 and S5)
==========================================================================================
Bivariate (Table S4) and partial (Table S5, pingouin 0.5.x, controlling for
the other seven metrics) correlations of the eight structural metrics with the
L2 targeted-attack threshold, for SCGC ratio thresholds of 0 to 50 percent and
a positive random-failure threshold. Uses the published analysis (100 Monte
Carlo trials at all scales, restricted threshold estimator).

Input  data/match_summary_mc100_guarded.csv
Output  results/tableS4.csv, results/tableS5.csv

Usage
  python code/supplementary/s05_threshold_sensitivity.py

Last modified 2026-10-27
"""

import sys
import warnings
from pathlib import Path

import pandas as pd
import pingouin as pg
from scipy import stats

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
# Published analysis (100 Monte Carlo trials at all scales, restricted
# threshold estimator)
PUBLISHED = DATA_DIR / "match_summary_mc100_guarded.csv"

METRICS = ["AvgDegree", "Clustering", "Assortativity", "Reciprocity",
           "FlowHierarchy", "OutDegreeCV", "Mu11_Mu", "CyclicTriadFraction"]
LABELS = {"AvgDegree": "<k>", "Clustering": "C", "Assortativity": "r",
          "Reciprocity": "rho", "FlowHierarchy": "h", "OutDegreeCV": "CV_out",
          "Mu11_Mu": "mu11/mu", "CyclicTriadFraction": "f_cyc"}
TARGET = "pc_Attack_Strong"
ERROR = "pc_Error_Strong"
THRESHOLDS = [0, 10, 20, 30, 40, 50]
MAIN_THRESHOLD = 30

# Supplementary Tables S4 and S5 as printed, threshold in percent
# (N, then <k>, C, r, rho, h, CV_out, mu11/mu, f_cyc)
TABLE_S4 = {
    0: [4432, 0.801, 0.687, 0.095, 0.653, -0.773, 0.249, 0.798, 0.597],
    10: [4425, 0.799, 0.685, 0.094, 0.651, -0.775, 0.247, 0.797, 0.595],
    20: [4393, 0.794, 0.678, 0.093, 0.641, -0.772, 0.245, 0.791, 0.590],
    30: [4330, 0.785, 0.667, 0.094, 0.627, -0.766, 0.237, 0.782, 0.579],
    40: [4127, 0.759, 0.637, 0.093, 0.592, -0.734, 0.228, 0.757, 0.559],
    50: [3624, 0.713, 0.582, 0.100, 0.526, -0.674, 0.228, 0.710, 0.534],
}
TABLE_S5 = {
    0: [4432, 0.046, -0.036, 0.037, -0.080, -0.322, 0.035, 0.102, -0.030],
    10: [4425, 0.042, -0.038, 0.038, -0.080, -0.327, 0.035, 0.103, -0.027],
    20: [4393, 0.039, -0.040, 0.039, -0.080, -0.321, 0.039, 0.101, -0.026],
    30: [4330, 0.031, -0.040, 0.041, -0.080, -0.311, 0.037, 0.104, -0.028],
    40: [4127, 0.021, -0.039, 0.039, -0.076, -0.270, 0.023, 0.110, -0.026],
    50: [3624, 0.025, -0.037, 0.022, -0.077, -0.215, 0.016, 0.104, -0.018],
}
# Section S5 text
TOTAL_L2 = 4560
RETAINED_PCT = 95.0
EXCLUDED, EXCLUDED_PCT = 230, 5.0
P_FCYC_PARTIAL_30 = 0.065
PARTIAL_K_RANGE = (0.02, 0.05)
PARTIAL_ALWAYS_P001 = ["FlowHierarchy", "Mu11_Mu", "Reciprocity"]


def sample(df, threshold_pct):
    """L2 networks with SCGC ratio at or above the threshold and a positive
    random-failure threshold."""
    d = df[(df.Scale == "L2") & (df.SCGC_Ratio >= threshold_pct / 100)
           & (df[ERROR] > 0)]
    return d.dropna(subset=METRICS)


def correlations(d):
    rows = []
    for m in METRICS:
        r, p = stats.pearsonr(d[m], d[TARGET])
        covars = [x for x in METRICS if x != m]
        pc = pg.partial_corr(data=d, x=m, y=TARGET, covar=covars, method="pearson")
        p_col = "p-val" if "p-val" in pc.columns else "p_val"
        rows.append({"metric": m, "r": r, "p": p,
                     "partial_r": pc["r"].values[0], "partial_p": pc[p_col].values[0]})
    return pd.DataFrame(rows).set_index("metric")


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def print_table(title, results, r_col, p_col):
    print(title)
    print(f"{'Threshold':<10}{'N':>6}" + "".join(f"{LABELS[m]:>11}" for m in METRICS))
    for thr, (n, res) in results.items():
        cells = "".join(f"{format(res.loc[m, r_col], '+.3f') + stars(res.loc[m, p_col]):>11}"
                        for m in METRICS)
        print(f"{str(thr) + '%':<10}{n:>6}{cells}")
    print("* P < 0.05, ** P < 0.01, *** P < 0.001\n")


def main():
    df = pd.read_csv(PUBLISHED)
    results = {}
    for thr in THRESHOLDS:
        d = sample(df, thr)
        results[thr] = (len(d), correlations(d))

    checks = []

    def check(label, got, expected, tol):
        checks.append((label, got, expected, abs(got - expected) <= tol + 1e-12))

    n_l2 = int((df.Scale == "L2").sum())
    n_main = results[MAIN_THRESHOLD][0]
    check("Section S5 L2 networks", n_l2, TOTAL_L2, 0)
    check("Section S5 retained at 30% (percent)", 100 * n_main / n_l2, RETAINED_PCT, 0.05)
    check("Section S5 excluded at 30%", n_l2 - n_main, EXCLUDED, 0)
    check("Section S5 excluded at 30% (percent)", 100 * (n_l2 - n_main) / n_l2,
          EXCLUDED_PCT, 0.05)

    for thr in THRESHOLDS:
        n, res = results[thr]
        check(f"Table S4 N {thr}%", n, TABLE_S4[thr][0], 0)
        check(f"Table S5 N {thr}%", n, TABLE_S5[thr][0], 0)
        for i, m in enumerate(METRICS):
            check(f"Table S4 {LABELS[m]} {thr}%", res.loc[m, "r"], TABLE_S4[thr][i + 1], 0.0005)
        for i, m in enumerate(METRICS):
            check(f"Table S5 {LABELS[m]} {thr}%", res.loc[m, "partial_r"],
                  TABLE_S5[thr][i + 1], 0.0005)

    # Table S4 note and Section S5 text
    all_p = max(results[t][1].loc[m, "p"] for t in THRESHOLDS for m in METRICS)
    checks.append(("Table S4 all bivariate P < 0.001 (largest P)", all_p, 0.001, all_p < 0.001))
    for col, name in [("r", "Table S4"), ("partial_r", "Table S5")]:
        for m in METRICS:
            signs = {results[t][1].loc[m, col] > 0 for t in THRESHOLDS}
            checks.append((f"{name} sign of {LABELS[m]} unchanged across thresholds",
                           len(signs), 1, len(signs) == 1))
    for m in PARTIAL_ALWAYS_P001:
        p_max = max(results[t][1].loc[m, "partial_p"] for t in THRESHOLDS)
        checks.append((f"Section S5 partial {LABELS[m]} P < 0.001 at all thresholds",
                       p_max, 0.001, p_max < 0.001))
    k_abs = [abs(results[t][1].loc["AvgDegree", "partial_r"]) for t in THRESHOLDS]
    check("Section S5 partial <k> smallest magnitude", min(k_abs), PARTIAL_K_RANGE[0], 0.005)
    check("Section S5 partial <k> largest magnitude", max(k_abs), PARTIAL_K_RANGE[1], 0.005)
    check("Section S5 partial f_cyc P at 30%",
          results[MAIN_THRESHOLD][1].loc["CyclicTriadFraction", "partial_p"],
          P_FCYC_PARTIAL_30, 0.0005)

    long = []
    for thr, (n, res) in results.items():
        for m in METRICS:
            long.append({"threshold_pct": thr, "N": n, "metric": m,
                         "r": res.loc[m, "r"], "p": res.loc[m, "p"],
                         "partial_r": res.loc[m, "partial_r"],
                         "partial_p": res.loc[m, "partial_p"]})
    long = pd.DataFrame(long)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    long[["threshold_pct", "N", "metric", "r", "p"]].to_csv(
        RESULTS_DIR / "tableS4.csv", index=False, float_format="%.6g")
    long[["threshold_pct", "N", "metric", "partial_r", "partial_p"]].to_csv(
        RESULTS_DIR / "tableS5.csv", index=False, float_format="%.6g")

    print(f"L2 networks {n_l2}; at the 30% threshold {n_main} are retained "
          f"({100 * n_main / n_l2:.1f}%)\n")
    print_table("Table S4  Bivariate Pearson r with the L2 attack threshold at "
                "different SCGC thresholds", results, "r", "p")
    print_table("Table S5  Partial correlations (controlling for the remaining seven "
                "metrics) at different SCGC thresholds", results, "partial_r", "partial_p")
    print(f"Partial f_cyc at 30%  P = "
          f"{results[MAIN_THRESHOLD][1].loc['CyclicTriadFraction', 'partial_p']:.3f}")
    print(f"Saved {(RESULTS_DIR / 'tableS4.csv').relative_to(BASE_DIR)}, "
          f"{(RESULTS_DIR / 'tableS5.csv').relative_to(BASE_DIR)}\n")

    print("Acceptance check against Supplementary Tables S4 and S5 and Section S5")
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
