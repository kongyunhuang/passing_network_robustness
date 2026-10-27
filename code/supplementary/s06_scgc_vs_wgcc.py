"""
s06_scgc_vs_wgcc.py  Comparison of SCGC and WGCC order parameters (Table S6)
========================================================================
L2 thresholds with the strongly (SCGC) and weakly (WGCC) connected giant
component as order parameter, over all 4,560 L2 networks of the published
analysis (100 Monte Carlo trials at all scales, restricted threshold
estimator). Mean and SD include zero thresholds; the correlation uses the
networks where both thresholds are positive.

Input  data/match_summary_mc100_guarded.csv
Output  results/tableS6.csv, results/s6_scgc_wgcc_correlation.csv

Usage
  python code/supplementary/s06_scgc_vs_wgcc.py

Last modified 2026-10-27
"""

import sys
from pathlib import Path

import pandas as pd
from scipy import stats

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
# Published analysis (100 Monte Carlo trials at all scales, restricted
# threshold estimator)
PUBLISHED = DATA_DIR / "match_summary_mc100_guarded.csv"

ORDER_PARAMETERS = {
    "SCGC": {"Error": "pc_Error_Strong", "Attack": "pc_Attack_Strong"},
    "WGCC": {"Error": "pc_Error_Weak", "Attack": "pc_Attack_Weak"},
}

# Supplementary Table S6 as printed
# (Error mean, Error SD, Attack mean, Attack SD, Error zero count, Attack zero count)
TABLE_S6 = {
    "SCGC": (0.580, 0.177, 0.403, 0.156, 128, 46),
    "WGCC": (0.784, 0.094, 0.648, 0.125, 5, 2),
}
# Section S6 text, networks where both thresholds are positive
CORRELATION = {"Attack": (0.726, 4514), "Error": (0.833, 4431)}
N_L2 = 4560


def main():
    df = pd.read_csv(PUBLISHED)
    d = df[df.Scale == "L2"]

    rows = []
    for name, cols in ORDER_PARAMETERS.items():
        e, a = d[cols["Error"]], d[cols["Attack"]]
        rows.append({"order_parameter": name, "N": len(d),
                     "fcE_mean": e.mean(), "fcE_sd": e.std(ddof=1),
                     "fcA_mean": a.mean(), "fcA_sd": a.std(ddof=1),
                     "fcE_zero": int((e == 0).sum()), "fcA_zero": int((a == 0).sum())})
    table = pd.DataFrame(rows)

    corr_rows = []
    for removal in ["Attack", "Error"]:
        s, w = ORDER_PARAMETERS["SCGC"][removal], ORDER_PARAMETERS["WGCC"][removal]
        both = d[(d[s] > 0) & (d[w] > 0)]
        r, p = stats.pearsonr(both[s], both[w])
        corr_rows.append({"removal": removal, "r": r, "p": p, "N": len(both)})
    corr = pd.DataFrame(corr_rows)

    checks = []

    def check(label, got, expected, tol):
        checks.append((label, got, expected, abs(got - expected) <= tol + 1e-12))

    check("Table S6 L2 networks", len(d), N_L2, 0)
    fields = ["fcE_mean", "fcE_sd", "fcA_mean", "fcA_sd", "fcE_zero", "fcA_zero"]
    for _, row in table.iterrows():
        for f, exp in zip(fields, TABLE_S6[row["order_parameter"]]):
            check(f"Table S6 {row['order_parameter']} {f}", row[f], exp,
                  0 if f.endswith("zero") else 0.0005)
    for _, row in corr.iterrows():
        r_exp, n_exp = CORRELATION[row["removal"]]
        check(f"Section S6 r {row['removal']}", row["r"], r_exp, 0.0005)
        check(f"Section S6 N {row['removal']}", row["N"], n_exp, 0)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS_DIR / "tableS6.csv", index=False, float_format="%.6f")
    corr.to_csv(RESULTS_DIR / "s6_scgc_wgcc_correlation.csv", index=False,
                float_format="%.6g")

    print(f"Table S6  SCGC and WGCC percolation thresholds (L2 scale, N = {len(d):,})")
    print(f"{'Order parameter':<17}{'fc_Error mean +/- SD':>22}{'fc_Attack mean +/- SD':>23}"
          f"{'fc_Error = 0':>14}{'fc_Attack = 0':>15}")
    for _, r in table.iterrows():
        print(f"{r['order_parameter']:<17}"
              f"{format(r['fcE_mean'], '.3f') + ' +/- ' + format(r['fcE_sd'], '.3f'):>22}"
              f"{format(r['fcA_mean'], '.3f') + ' +/- ' + format(r['fcA_sd'], '.3f'):>23}"
              f"{r['fcE_zero']:>14}{r['fcA_zero']:>15}")
    print("\nPearson correlation between SCGC and WGCC thresholds (both positive)")
    for _, r in corr.iterrows():
        print(f"  {r['removal']:<7} r = {r['r']:.3f}, P = {r['p']:.1e}, N = {r['N']}")
    print(f"\nSaved {(RESULTS_DIR / 'tableS6.csv').relative_to(BASE_DIR)}, "
          f"{(RESULTS_DIR / 's6_scgc_wgcc_correlation.csv').relative_to(BASE_DIR)}\n")

    print("Acceptance check against Supplementary Table S6 and Section S6")
    for label, got, exp, ok in checks:
        print(f"  {label:<40} {got:12.6g} vs {exp:<9g} {'PASS' if ok else 'FAIL'}")
    failed = [c for c in checks if not c[3]]
    if failed:
        print(f"GATE FAILED  {len(failed)} of {len(checks)} values differ from the published values")
        for label, got, exp, _ in failed:
            print(f"  {label}  computed {got:.5f}, published {exp}")
        sys.exit(1)
    print(f"GATE PASS  all {len(checks)} values match")


if __name__ == "__main__":
    main()
