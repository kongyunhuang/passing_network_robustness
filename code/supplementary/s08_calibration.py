"""
s08_calibration.py — ER known-threshold exam for the estimator (Supplementary S8)
=================================================================================
Calibrates the susceptibility-peak f_c estimator against directed
Erdos-Renyi graphs, where the exact answer is known analytically. Independent
edge dilution of a directed ER graph yields another ER graph with mean degree
c' = (1 - f) c, so the strongly connected giant component vanishes exactly at
f_c = 1 - 1/c. Large-N points test instrument accuracy; small-N points
(matching the empirical L2 and L3 regimes) reproduce the spurious tail peak
of the unrestricted estimator and show that the peak-validity restriction
recovers the true threshold. The reading of the manuscript's theoretical
benchmark formula on each ER graph is also reported as-is.

Four calibration settings (n, c, realisations):
    (1000, 3.0, 5), (1000, 5.0, 5), (57, 6.0, 10)   [original three points]
    (130, 2.3, 10)                                   [L2-like small-sparse point]
The two blocks below use separate random-number streams so that each block
reproduces the corresponding production run bit-for-bit.

Input:  none (synthetic ER graphs only)
Output: er_exam_results.csv, er_exam_L2point_results.csv (next to this script)
        + terminal summaries (mean +/- SD per setting)

Usage:
    cd code/supplementary
    python s08_calibration.py

Last modified: 2026-07-21
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent / "pipeline"))

from robustness import (                     # noqa: E402
    multi_trial_edge_removal, find_critical_threshold,
    calculate_molloy_reed_threshold,
)

PINF_GUARD = 0.05


def guarded_threshold(sim: pd.DataFrame, component: str = "strong",
                      pinf_min: float = PINF_GUARD):
    """Peak-validity restriction with an absolute P_inf floor.

    Restricts the susceptibility argmax to rows where P_inf >= pinf_min.
    This is the exact restriction variant used in the production calibration
    runs (absolute floor; for the main empirical pipeline the floor is
    expressed relative to P_inf(0), see pipeline/robustness.py).
    """
    chi_col, p_col = f"Chi_{component}", f"P_inf_{component}"
    ok = sim[sim[p_col] >= pinf_min]
    if ok.empty or ok[chi_col].max() == 0:
        return np.nan, 0.0
    row = ok.loc[ok[chi_col].idxmax()]
    return row["f"], row[chi_col]


def run_block(settings, record_extras):
    """Run one exam block with its own random-number stream.

    Each block re-initialises numpy's default_rng(7), preserving the exact
    graph seeds of the corresponding production run.
    """
    rows = []
    rng = np.random.default_rng(7)
    for n, c, reps in settings:
        for rep in range(reps):
            G = nx.gnp_random_graph(n, c / (n - 1), seed=int(rng.integers(1e9)),
                                    directed=True)
            sim = multi_trial_edge_removal(G, method="error", n_steps=50,
                                           n_trials=100, seed=42 + rep)
            fc_raw, chi_raw = find_critical_threshold(sim, "strong")
            fc_g, _ = guarded_threshold(sim, "strong")
            row = {"n": n, "c": c, "rep": rep, "fc_true": 1 - 1 / c,
                   "fc_raw": fc_raw, "fc_guarded": fc_g}
            if record_extras:
                row["chi_raw"] = chi_raw
                row["fc_eq1_paper"] = calculate_molloy_reed_threshold(G)
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    # Block 1: the original three calibration points
    settings = [(1000, 3.0, 5), (1000, 5.0, 5), (57, 6.0, 10)]
    df = run_block(settings, record_extras=True)
    df = df[["n", "c", "rep", "fc_true", "fc_raw", "fc_guarded",
             "chi_raw", "fc_eq1_paper"]]
    df.to_csv(_SCRIPT_DIR / "er_exam_results.csv", index=False)
    print("\n=== ER exam results (mean +/- SD across realisations) ===")
    g = df.groupby(["n", "c"]).agg(
        fc_true=("fc_true", "first"),
        fc_raw_mean=("fc_raw", "mean"), fc_raw_sd=("fc_raw", "std"),
        fc_guard_mean=("fc_guarded", "mean"), fc_guard_sd=("fc_guarded", "std"),
        fc_eq1=("fc_eq1_paper", "mean"),
        tail_rate=("fc_raw", lambda s: (s >= 0.9).mean()),
    ).round(3)
    print(g.to_string())

    # Block 2: the L2-like small-sparse point (separate random stream)
    df2 = run_block([(130, 2.3, 10)], record_extras=False)
    df2.to_csv(_SCRIPT_DIR / "er_exam_L2point_results.csv", index=False)
    print("\n=== L2-like point (n=130, c=2.3) ===")
    g2 = df2.groupby(["n", "c"]).agg(
        fc_true=("fc_true", "first"),
        raw_mean=("fc_raw", "mean"), raw_sd=("fc_raw", "std"),
        g_mean=("fc_guarded", "mean"), g_sd=("fc_guarded", "std"),
        tail=("fc_raw", lambda s: (s >= 0.9).mean()))
    print(g2.round(4).to_string())


if __name__ == "__main__":
    main()
