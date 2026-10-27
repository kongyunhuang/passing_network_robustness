"""
s08_threshold_settings.py  Threshold sensitivity to ensemble size and restriction (Table S8)
===========================================================================================
Compares the mean random-failure and targeted-attack thresholds of all 4,560
team-match networks per scale under the three settings of Supplementary
Table S8. The first setting (20 trials at L2 and 5 trials at L0, L1 and L3, no
order-parameter restriction) is read from data/match_summary.csv. The 100-trial
thresholds without and with the restriction are read from the published
analysis. No simulation is run.

Input  data/match_summary.csv, data/match_summary_mc100_guarded.csv
Output  results/tableS8.csv

Usage
  python code/supplementary/s08_threshold_settings.py

Last modified 2026-10-27
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
FIRST_SETTING = DATA_DIR / "match_summary.csv"
# Published analysis (100 Monte Carlo trials at all scales, restricted
# threshold estimator). Its audit columns fcE_raw_strong and fcA_raw_strong
# hold the 100-trial thresholds before the restriction is applied.
PUBLISHED = DATA_DIR / "match_summary_mc100_guarded.csv"

SCALES = ["L0", "L1", "L2", "L3"]
KEY = ["MatchID", "TeamID", "Scale"]

# Supplementary Table S8, one row per resolution
# (first setting, 100 unrestricted, 100 restricted, delta trials,
#  delta restriction, restriction active in percent, attack 100 restricted)
TABLE_S8 = {
    "L0": (0.103, 0.146, 0.146, +0.043, 0.000, 0.0, 0.023),
    "L1": (0.216, 0.233, 0.231, +0.017, -0.003, 1.0, 0.113),
    "L2": (0.578, 0.589, 0.580, +0.012, -0.009, 4.6, 0.403),
    "L3": (0.790, 0.914, 0.804, +0.125, -0.110, 71.2, 0.611),
}
# Table S8 note, attack thresholds without the restriction
ATTACK_UNRESTRICTED = {"L0": 0.024, "L1": 0.115, "L2": 0.408, "L3": 0.613}
# Section S8 text, computed on the Table 3 sample (positive dMR benchmark)
NET_CHANGE_TABLE3 = {"L1": +0.020, "L2": +0.003, "L3": +0.014}
R_BENCHMARK_FIRST = {"L1": 0.528, "L2": 0.751, "L3": 0.590}
R_BENCHMARK_PUBLISHED = {"L1": 0.599, "L2": 0.809, "L3": 0.754}
MEDIAN_ABS_TRIALS_L2 = 0.023
ATTACK_CHANGE_MAX = 0.005

COLUMNS = ["fcE_first_setting", "fcE_100_unrestricted", "fcE_100_restricted",
           "delta_trials", "delta_restriction", "restriction_active_pct",
           "fcA_100_restricted"]


def load():
    first = pd.read_csv(FIRST_SETTING).sort_values(KEY).reset_index(drop=True)
    pub = pd.read_csv(PUBLISHED).sort_values(KEY).reset_index(drop=True)
    if len(first) != len(pub) or not (first[KEY].values == pub[KEY].values).all():
        print("Input files do not list the same networks in the same order.")
        sys.exit(1)
    return first, pub


def main():
    first, pub = load()
    checks = []

    def check(label, got, expected, tol):
        checks.append((label, got, expected, abs(got - expected) <= tol + 1e-12))

    # The three settings share the networks, so the dMR benchmark must be
    # identical, and targeted attack (deterministic) must not depend on the
    # number of trials.
    def same(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        return bool(np.all((np.isnan(a) & np.isnan(b)) | (np.abs(a - b) <= 1e-12)))
    checks.append(("dMR benchmark identical in both files", 1, 1,
                   same(first.fc_dMR, pub.pc_Theory)))
    checks.append(("attack threshold identical at 20 or 5 and 100 trials", 1, 1,
                   same(first.fc_Attack_Strong, pub.fcA_raw_strong)))

    rows = []
    for s in SCALES:
        m = (pub.Scale == s).to_numpy()
        a = first.loc[m, "fc_Error_Strong"]          # first setting
        b = pub.loc[m, "fcE_raw_strong"]             # 100 trials, unrestricted
        c = pub.loc[m, "pc_Error_Strong"]            # 100 trials, restricted
        att = pub.loc[m, "pc_Attack_Strong"]
        att_raw = pub.loc[m, "fcA_raw_strong"]
        active = (c != b)
        row = {
            "Resolution": s, "n": int(m.sum()),
            "fcE_first_setting": a.mean(),
            "fcE_100_unrestricted": b.mean(),
            "fcE_100_restricted": c.mean(),
            "delta_trials": (b - a).mean(),
            "delta_restriction": (c - b).mean(),
            "restriction_active_pct": 100 * active.mean(),
            "fcA_100_restricted": att.mean(),
            "fcA_100_unrestricted": att_raw.mean(),
            "delta_restriction_attack": (att - att_raw).mean(),
            "restriction_active_n": int(active.sum()),
            "median_abs_delta_trials": (b - a).abs().median(),
        }
        rows.append(row)

        check(f"{s} networks", row["n"], 4560, 0)
        for col, exp in zip(COLUMNS, TABLE_S8[s]):
            tol = 0.05 if col == "restriction_active_pct" else 0.0005
            check(f"{s} {col}", row[col], exp, tol)
        check(f"{s} fcA_100_unrestricted", row["fcA_100_unrestricted"],
              ATTACK_UNRESTRICTED[s], 0.0005)
        checks.append((f"{s} |attack change from restriction| <= {ATTACK_CHANGE_MAX}",
                       abs(row["delta_restriction_attack"]), ATTACK_CHANGE_MAX,
                       abs(row["delta_restriction_attack"]) <= ATTACK_CHANGE_MAX))

    table = pd.DataFrame(rows)
    by = table.set_index("Resolution")
    check("L0 networks changed by the restriction", by.loc["L0", "restriction_active_n"], 1, 0)
    check("L2 median |delta trials|", by.loc["L2", "median_abs_delta_trials"],
          MEDIAN_ABS_TRIALS_L2, 0.0005)

    # Table 3 sample (positive dMR benchmark), net change and agreement with
    # the benchmark under the first setting and the published analysis
    for s in ["L1", "L2", "L3"]:
        m = ((pub.Scale == s) & (pub.pc_Theory > 0)).to_numpy()
        theory = pub.loc[m, "pc_Theory"]
        a = first.loc[m, "fc_Error_Strong"]
        c = pub.loc[m, "pc_Error_Strong"]
        att, att_raw = pub.loc[m, "pc_Attack_Strong"], pub.loc[m, "fcA_raw_strong"]
        check(f"{s} Table 3 sample net change", (c - a).mean(), NET_CHANGE_TABLE3[s], 0.0005)
        check(f"{s} Table 3 sample r(dMR, first setting)", sps.pearsonr(theory, a)[0],
              R_BENCHMARK_FIRST[s], 0.0005)
        check(f"{s} Table 3 sample r(dMR, published)", sps.pearsonr(theory, c)[0],
              R_BENCHMARK_PUBLISHED[s], 0.0005)
        d_att = abs((att - att_raw).mean())
        checks.append((f"{s} Table 3 sample |attack change| <= {ATTACK_CHANGE_MAX}",
                       d_att, ATTACK_CHANGE_MAX, d_att <= ATTACK_CHANGE_MAX))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "tableS8.csv"
    table.to_csv(out, index=False, float_format="%.6f")

    def signed(v):
        return f"{0.0:.3f}" if round(v, 3) == 0 else f"{v:+.3f}"

    print("Table S8  Mean thresholds under the three settings (all 4,560 networks per scale)")
    print(f"{'':4}{'20 or 5':>10}{'100':>10}{'100':>10}{'Delta':>9}{'Delta':>13}"
          f"{'Restr.':>10}{'Attack':>10}")
    print(f"{'':4}{'unrestr.':>10}{'unrestr.':>10}{'restr.':>10}{'trials':>9}"
          f"{'restriction':>13}{'active %':>10}{'restr.':>10}")
    for r in rows:
        print(f"{r['Resolution']:4}{r['fcE_first_setting']:10.3f}{r['fcE_100_unrestricted']:10.3f}"
              f"{r['fcE_100_restricted']:10.3f}{signed(r['delta_trials']):>9}"
              f"{signed(r['delta_restriction']):>13}{r['restriction_active_pct']:10.1f}"
              f"{r['fcA_100_restricted']:10.3f}")
    print("Attack without the restriction  " + ", ".join(
        f"{r['Resolution']} {r['fcA_100_unrestricted']:.3f}" for r in rows))
    print(f"Saved {out.relative_to(BASE_DIR)}\n")

    print("Acceptance check against Supplementary Table S8 and Section S8")
    for label, got, exp, ok in checks:
        print(f"  {label:<56} {got:10.4f} vs {exp:8.3f}  {'PASS' if ok else 'FAIL'}")
    failed = [c for c in checks if not c[3]]
    if failed:
        print(f"GATE FAILED  {len(failed)} of {len(checks)} values differ from the published values")
        for label, got, exp, _ in failed:
            print(f"  {label}  computed {got:.4f}, published {exp}")
        sys.exit(1)
    print(f"GATE PASS  all {len(checks)} values match")


if __name__ == "__main__":
    main()
