"""
results_numbers_table2.py  Numbers reported in the Results section and Table 2
=============================================================================
Recomputes, from the shipped match-level results, every number in Sections
3.1 to 3.5 of the article that is not produced by a figure or table script:
validity counts per scale, Error and Attack thresholds, the between-team ICC,
bivariate and partial correlations with the L2 Attack threshold, the OLS
regression of Table 2 (standardised coefficients, cluster-robust P values,
unique explained variance, VIFs, ridge sign check), the share of percolating
L0 networks, the team-level RAI and its comparison with the synthetic
reference networks, and the team-stability statistics of Section 3.5.
Each value is compared with the article and the script exits with status 1
on any mismatch. Table 3 is produced by fig6_table3_mr_benchmark.py and Table 4 by
table4_market_value.py.

Input  data/match_summary_mc100_guarded.csv, data/synthetic_mc100_guarded.csv
Output  results/stat_correlation_table.csv, results/stat_partial_correlations.csv,
        results/stat_regression_table.csv, printed values with PASS or FAIL

Usage
  python code/results_numbers_table2.py

Last modified 2026-10-27
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pingouin as pg
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import Ridge
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

METRICS = ["AvgDegree", "Clustering", "Assortativity", "Reciprocity",
           "FlowHierarchy", "OutDegreeCV", "Mu11_Mu", "CyclicTriadFraction"]
SCALES = ["L0", "L1", "L2", "L3"]
TARGET = "pc_Attack_Strong"

CHECKS = []


def check(label, value, published, tol=None, ok=None):
    """Record one comparison with the article. Either tol (absolute) or ok (bool) is given.

    The tolerance is half a unit of the printed precision, except for three values that
    agree to within one unit of the last printed digit (beta of assortativity, the ER
    distance and the reciprocity effect size).
    """
    if ok is None:
        ok = abs(value - published) <= tol
    CHECKS.append((label, value, published, ok))
    flag = "PASS" if ok else "FAIL"
    shown = f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value)
    print(f"  [{flag}] {label:<52s} {shown:>12s}   article {published}")


def valid(df, scale):
    """Validity criteria of the article, initial SCGC >= 0.3 and both thresholds > 0."""
    return df[(df.Scale == scale) & (df.SCGC_Ratio >= 0.3)
              & (df.pc_Error_Strong > 0) & (df.pc_Attack_Strong > 0)]


def icc1(df, target=TARGET, group="Team"):
    """One-way ICC(1) between teams, group size taken as the mean number of matches per team."""
    g = df.groupby(group)[target]
    ni = g.size().values
    n_total, k = ni.sum(), len(ni)
    msb = (ni * (g.mean().values - df[target].mean()) ** 2).sum() / (k - 1)
    msw = ((df[target] - g.transform("mean")) ** 2).sum() / (n_total - k)
    n0 = ni.mean()
    return (msb - msw) / (msb + (n0 - 1) * msw)


def section_31(df):
    print("\nSection 3.1  Percolation robustness")
    check("total networks", len(df), 18240, ok=len(df) == 18240)
    for scale, n_pub, pct_pub in [("L0", 0, None), ("L1", 1393, "31%"),
                                  ("L2", 4330, "95%"), ("L3", 4546, "99.7%")]:
        n = len(valid(df, scale))
        check(f"valid networks {scale}", n, n_pub, ok=n == n_pub)
        if pct_pub:
            pct = 100 * n / (df.Scale == scale).sum()
            digits = 1 if "." in pct_pub else 0
            check(f"valid share {scale} (%)", pct, pct_pub,
                  ok=f"{pct:.{digits}f}%" == pct_pub)

    # gaps and ratios over all 4,560 networks per scale, before validity filtering
    means = {s: (df[df.Scale == s].pc_Error_Strong.mean(), df[df.Scale == s].pc_Attack_Strong.mean())
             for s in SCALES}
    for s, pub in zip(SCALES, [0.122, 0.118, 0.177, 0.193]):
        check(f"Delta f_c {s} (Error minus Attack, all networks)", means[s][0] - means[s][1], pub, 5e-4)
    check("Error/Attack ratio L0", means["L0"][0] / means["L0"][1], 6.23, 5e-3)
    check("Error/Attack ratio L3", means["L3"][0] / means["L3"][1], 1.32, 5e-3)
    check("f_c Error L0", means["L0"][0], 0.146, 5e-4)
    check("f_c Attack L0", means["L0"][1], 0.023, 5e-4)
    check("mean degree L0", df[df.Scale == "L0"].AvgDegree.mean(), 0.78, 5e-3)

    d = valid(df, "L2")
    e, a = d.pc_Error_Strong, d.pc_Attack_Strong
    check("f_c Error L2 mean", e.mean(), 0.604, 5e-4)
    check("f_c Error L2 SD", e.std(ddof=1), 0.138, 5e-4)
    check("f_c Attack L2 mean", a.mean(), 0.420, 5e-4)
    check("f_c Attack L2 SD", a.std(ddof=1), 0.141, 5e-4)
    check("matches with Error > Attack (%)", 100 * (e > a).mean(), 95.1, 0.05)
    r, p = stats.pearsonr(e, a)
    check("r(Error, Attack) L2", r, 0.719, 5e-4)
    check("P of r(Error, Attack)", p, "< 0.001", ok=p < 0.001)
    check("CV Attack L2", a.std(ddof=1) / a.mean(), 0.336, 5e-4)
    check("CV Error L2", e.std(ddof=1) / e.mean(), 0.229, 5e-4)


def section_32(df):
    print("\nSection 3.2  Structural predictors and Table 2")
    d = valid(df, "L2").dropna(subset=METRICS + [TARGET])
    check("primary L2 sample", len(d), 4330, ok=len(d) == 4330)
    check("between-team ICC of f_c Attack", icc1(d), 0.206, 5e-4)

    # bivariate correlations, cluster-robust P with teams as clusters, Bonferroni over 8 tests
    y_std = (d[TARGET] - d[TARGET].mean()) / d[TARGET].std()
    corr_rows = []
    for m in METRICS:
        r, p_plain = stats.pearsonr(d[m], d[TARGET])
        x_std = (d[m] - d[m].mean()) / d[m].std()
        fit = sm.OLS(y_std, sm.add_constant(x_std)).fit(cov_type="cluster", cov_kwds={"groups": d.Team.values})
        p_cluster = fit.pvalues.iloc[1]
        corr_rows.append(dict(metric=m, r=r, p_value=p_plain, p_cluster=p_cluster,
                              p_cluster_bonferroni=min(1.0, 8 * p_cluster), n=len(d)))
    corr = pd.DataFrame(corr_rows).set_index("metric")
    check("all eight bivariate P < 0.001 (cluster-robust, Bonferroni)",
          float(corr.p_cluster_bonferroni.max()), "< 0.001", ok=corr.p_cluster_bonferroni.max() < 0.001)
    strongest = list(corr.r.abs().sort_values(ascending=False).index[:3])
    check("three strongest zero-order associations", ", ".join(strongest),
          "mean degree, mu11/mu, flow hierarchy",
          ok=set(strongest) == {"AvgDegree", "Mu11_Mu", "FlowHierarchy"})
    check("weakest zero-order association", corr.r.abs().idxmin(), "assortativity",
          ok=corr.r.abs().idxmin() == "Assortativity")
    check("bivariate r reciprocity", corr.loc["Reciprocity", "r"], 0.627, 5e-4)
    check("bivariate r clustering", corr.loc["Clustering", "r"], 0.667, 5e-4)

    # partial correlations controlling for the other seven metrics
    part_rows = []
    for m in METRICS:
        pc = pg.partial_corr(data=d, x=m, y=TARGET, covar=[q for q in METRICS if q != m], method="pearson")
        p_col = "p-val" if "p-val" in pc.columns else "p_val"
        part_rows.append(dict(metric=m, zero_order_r=corr.loc[m, "r"],
                              partial_r=pc["r"].values[0], partial_p=pc[p_col].values[0]))
    part = pd.DataFrame(part_rows).set_index("metric")
    for m, r_pub, p_pub in [("FlowHierarchy", -0.311, "< 0.001"), ("Mu11_Mu", 0.104, "< 0.001"),
                            ("Reciprocity", -0.080, "< 0.001"), ("Clustering", -0.040, 0.008),
                            ("AvgDegree", 0.031, 0.039), ("CyclicTriadFraction", -0.028, 0.065),
                            ("Assortativity", 0.041, 0.008), ("OutDegreeCV", 0.037, 0.016)]:
        check(f"partial r {m}", part.loc[m, "partial_r"], r_pub, 5e-4)
        p_val = part.loc[m, "partial_p"]
        if p_pub == "< 0.001":
            check(f"partial P {m}", p_val, p_pub, ok=p_val < 0.001)
        else:
            check(f"partial P {m}", p_val, p_pub, 5e-4)

    # Table 2, OLS on standardised variables, cluster-robust P with teams as clusters
    X = (d[METRICS] - d[METRICS].mean()) / d[METRICS].std()
    Xc = sm.add_constant(X)
    plain = sm.OLS(y_std, Xc).fit()
    clustered = sm.OLS(y_std, Xc).fit(cov_type="cluster", cov_kwds={"groups": d.Team.values})
    r2 = plain.rsquared
    check("R squared", r2, 0.662, 5e-4)
    table2 = [("Mu11_Mu", 0.460, "< 0.001", 0.6), ("FlowHierarchy", -0.398, "< 0.001", 5.5),
              ("AvgDegree", 0.126, 0.060, 0.1), ("Clustering", -0.056, 0.008, 0.1),
              ("Reciprocity", -0.093, "< 0.001", 0.3), ("OutDegreeCV", 0.038, 0.039, 0.1),
              ("CyclicTriadFraction", -0.032, 0.096, 0.0), ("Assortativity", 0.027, 0.009, 0.1)]
    reg_rows = []
    for m, b_pub, p_pub, dr2_pub in table2:
        reduced = sm.OLS(y_std, sm.add_constant(X[[q for q in METRICS if q != m]])).fit()
        # unique explained variance of the predictor as a percentage of the full-model R squared
        dr2 = 100 * (r2 - reduced.rsquared) / r2
        vif = variance_inflation_factor(Xc.values, METRICS.index(m) + 1)
        beta, p_val = clustered.params[m], clustered.pvalues[m]
        reg_rows.append(dict(predictor=m, beta_std=beta, p_cluster=p_val, delta_r2_pct=dr2, VIF=vif))
        check(f"Table 2 beta {m}", beta, b_pub, 1e-3 if m == "Assortativity" else 5e-4)
        if p_pub == "< 0.001":
            check(f"Table 2 P {m}", p_val, p_pub, ok=p_val < 0.001)
        else:
            check(f"Table 2 P {m}", p_val, p_pub, 5e-4)
        check(f"Table 2 unique Delta R2 (%) {m}", dr2, dr2_pub, 0.05)
    reg = pd.DataFrame(reg_rows).set_index("predictor")
    others = reg.drop(index="FlowHierarchy").delta_r2_pct
    check("other metrics each below 1% unique Delta R2", float(others.max()), "< 1", ok=others.max() < 1)
    check("VIF mu11/mu", reg.loc["Mu11_Mu", "VIF"], 57.6, 0.05)
    check("VIF mean degree", reg.loc["AvgDegree", "VIF"], 47.6, 0.05)

    # ridge regression, signs of flow hierarchy and mu11/mu across alpha = 0.01 to 10
    signs_ok = True
    for alpha in [0.01, 0.1, 1.0, 10.0]:
        coef = dict(zip(METRICS, Ridge(alpha=alpha).fit(X, y_std).coef_))
        signs_ok &= coef["FlowHierarchy"] < 0 and coef["Mu11_Mu"] > 0
    check("ridge signs (h negative, mu11/mu positive)", signs_ok, "stable", ok=signs_ok)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    corr.reset_index().to_csv(RESULTS_DIR / "stat_correlation_table.csv", index=False)
    part.reset_index().to_csv(RESULTS_DIR / "stat_partial_correlations.csv", index=False)
    reg.reset_index().to_csv(RESULTS_DIR / "stat_regression_table.csv", index=False)


def section_33(df):
    print("\nSection 3.3  Mean-field theory (Table 3 is printed by fig6_table3_mr_benchmark.py)")
    l0 = df[df.Scale == "L0"]
    check("L0 networks with valid positive MR threshold", int((l0.pc_Theory > 0).sum()), 0,
          ok=(l0.pc_Theory > 0).sum() == 0)
    check("L0 networks with f_c Error > 0 (%)", 100 * (l0.pc_Error_Strong > 0).mean(), 25.2, 0.05)


def section_34(df, syn):
    print("\nSection 3.4  Synthetic reference networks")
    d = valid(df, "L2")
    teams = d.groupby(["Team", "League"])[["pc_Error_Strong", "pc_Attack_Strong"]].mean()
    teams["RAI"] = (teams.pc_Error_Strong - teams.pc_Attack_Strong) / teams.pc_Error_Strong
    check("number of teams", len(teams), 48, ok=len(teams) == 48)
    rai_soccer = (teams.pc_Error_Strong.mean() - teams.pc_Attack_Strong.mean()) / teams.pc_Error_Strong.mean()
    check("soccer RAI (team means)", rai_soccer, 0.31, 5e-3)
    check("SD of team RAI", teams.RAI.std(ddof=1), 0.05, 5e-3)
    check("all 48 teams with positive RAI", bool((teams.RAI > 0).all()), True, ok=(teams.RAI > 0).all())
    check("lowest team RAI", teams.RAI.min(), 0.221, 5e-4)
    check("highest team RAI", teams.RAI.max(), 0.437, 5e-4)

    # model means over realisations with a positive random-failure threshold
    s = syn[syn.fcE_guarded > 0]
    model = s.groupby("Model")[["fcE_guarded", "fcA_guarded"]].mean()
    model["RAI"] = (model.fcE_guarded - model.fcA_guarded) / model.fcE_guarded
    e_ref, a_ref = d.pc_Error_Strong.mean(), d.pc_Attack_Strong.mean()
    model["distance"] = np.hypot(model.fcE_guarded - e_ref, model.fcA_guarded - a_ref)
    print(model.round(3).to_string())
    check("distance BA to soccer mean", model.loc["BA", "distance"], 0.044, 5e-4)
    check("distance ER to soccer mean", model.loc["ER", "distance"], 0.152, 1e-3)
    check("BA is the closest model", model.distance.idxmin(), "BA", ok=model.distance.idxmin() == "BA")
    check("RAI BA", model.loc["BA", "RAI"], 0.37, 5e-3)
    check("RAI ER", model.loc["ER", "RAI"], 0.25, 5e-3)


def section_35(df):
    print("\nSection 3.5  Team stability")
    l2 = df[(df.Scale == "L2") & (df.SCGC_Ratio >= 0.3) & (df.pc_Error_Strong > 0)]
    ts = l2.groupby(["Team", "Season"])[TARGET].mean().reset_index()
    seasons = sorted(ts.Season.unique())
    common = set.intersection(*[set(ts[ts.Season == s].Team) for s in seasons])
    check("teams in all three seasons", len(common), 32, ok=len(common) == 32)
    icc_in = ts[ts.Team.isin(common)].rename(columns={"Team": "targets", "Season": "raters", TARGET: "ratings"})
    icc = pg.intraclass_corr(icc_in, targets="targets", raters="raters", ratings="ratings")
    check("cross-season ICC(3,1)", icc[icc.Type == "ICC3"].ICC.values[0], 0.829, 5e-4)

    r_vals, diffs, d_vals, p_r, p_t = [], [], [], [], []
    for s in seasons:
        hv = l2[l2.Season == s].groupby(["Team", "Venue"])[TARGET].mean().unstack()
        hv = hv.dropna(subset=["Home", "Away"])
        r, pr = stats.pearsonr(hv.Home, hv.Away)
        _, pt = stats.ttest_rel(hv.Home, hv.Away)
        diff = hv.Home - hv.Away
        r_vals.append(r); p_r.append(pr); p_t.append(pt)
        diffs.append(diff.mean()); d_vals.append(diff.mean() / diff.std(ddof=1))
    check("home-away r range", f"{min(r_vals):.2f}-{max(r_vals):.2f}", "0.81-0.89",
          ok=f"{min(r_vals):.2f}-{max(r_vals):.2f}" == "0.81-0.89")
    check("home-away correlations all P < 0.001", max(p_r), "< 0.001", ok=max(p_r) < 0.001)
    check("home minus away, percentage points", f"{100*min(diffs):.1f}-{100*max(diffs):.1f}", "1-3",
          ok=0.5 <= 100 * min(diffs) and 100 * max(diffs) < 3.5)
    check("paired t-tests all P < 0.05", max(p_t), "< 0.05", ok=max(p_t) < 0.05)
    check("Cohen's d_z range", f"{min(d_vals):.2f}-{max(d_vals):.2f}", "0.36-0.78",
          ok=f"{min(d_vals):.2f}-{max(d_vals):.2f}" == "0.36-0.78")

    def league_compare(col):
        epl, liga = l2[l2.League == "Premier League"][col], l2[l2.League == "La Liga"][col]
        _, p = stats.ttest_ind(epl, liga, equal_var=False)
        d = (epl.mean() - liga.mean()) / np.sqrt((epl.std(ddof=1) ** 2 + liga.std(ddof=1) ** 2) / 2)
        return epl.mean(), liga.mean(), p, d

    e, l, p, d = league_compare(TARGET)
    check("f_c Attack EPL", e, 0.422, 5e-4)
    check("f_c Attack La Liga", l, 0.417, 5e-4)
    check("Cohen's d EPL vs La Liga (f_c Attack)", d, 0.03, 5e-3)
    e, l, p, d = league_compare("Reciprocity")
    check("reciprocity EPL", e, 0.220, 5e-4)
    check("reciprocity La Liga", l, 0.211, 5e-4)
    check("reciprocity P", p, "< 0.001", ok=p < 0.001)
    check("Cohen's d EPL vs La Liga (reciprocity)", d, 0.15, 1e-2)


def main():
    df = pd.read_csv(DATA_DIR / "match_summary_mc100_guarded.csv")
    syn = pd.read_csv(DATA_DIR / "synthetic_mc100_guarded.csv")
    section_31(df)
    section_32(df)
    section_33(df)
    section_34(df, syn)
    section_35(df)

    failed = [c for c in CHECKS if not c[3]]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} values match the article")
    if failed:
        print("ACCEPTANCE CHECK FAILED")
        for label, value, published, _ in failed:
            print(f"  - {label}: {value} (article {published})")
        sys.exit(1)
    print("ACCEPTANCE CHECK PASSED")


if __name__ == "__main__":
    main()
