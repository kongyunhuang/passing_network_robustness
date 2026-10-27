"""
synthetic_baselines_generator.py  Calibrated synthetic reference networks (Section 3.4, Fig. 7)
==============================================================================================
Generates the four synthetic model families (ER, BA, WS, RGG, 100 realisations
each) calibrated to the empirical L2 networks (mean number of nodes, mean
degree, and reciprocity through mixed edge orientation), and evaluates them
with the percolation protocol of the article (random failure with 100 Monte
Carlo trials, targeted attack as a single deterministic removal, both with the
restricted threshold estimator). The regenerated results are compared with the
shipped file data/synthetic_mc100_guarded.csv and the script exits with status
1 on any difference. A model-level summary is printed at the end.

Input  data/match_summary_mc100_guarded.csv (calibration targets)
       data/synthetic_mc100_guarded.csv (shipped results, for the comparison)
Output  results/synthetic_mc100_guarded.csv, results/synthetic_summary_by_model.csv

Usage
  python code/synthetic_baselines_generator.py --test        # 5 realisations per model
  python code/synthetic_baselines_generator.py --workers 8   # full run, 100 per model

Last modified 2026-10-27
"""

import argparse
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict

import networkx as nx
import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_SCRIPT_DIR / "pipeline"))

from robustness import (                                  # noqa: E402
    multi_trial_edge_removal, edge_removal_simulation,
    find_critical_threshold, calculate_robustness_integral,
    calculate_molloy_reed_threshold,
)

MATCH_RESULTS = REPO_ROOT / "data" / "match_summary_mc100_guarded.csv"
SHIPPED = REPO_ROOT / "data" / "synthetic_mc100_guarded.csv"
RESULTS_DIR = REPO_ROOT / "results"

MODEL_TYPES = ["ER", "BA", "WS", "RGG"]
N_REALISATIONS = 100
SEED_BASE = 42
N_TRIALS, SEED, N_STEPS = 100, 42, 50
PINF_RESTRICTION_FRACTION = 0.05


# ============================================================================
# Graph generation
# ============================================================================

def load_empirical_l2_targets(match_csv_path: Path = MATCH_RESULTS) -> Dict[str, float]:
    """Calibrate synthetic models from empirical match-level L2 means.

    Uses mixed orientation to match empirical reciprocity:
    - Each undirected edge becomes bidirectional with probability p_r,
      or a single random directed edge with probability 1-p_r.
    - This gives: E_directed = E_undirected * (1 + p_r)
    - Reciprocity ~ 2*p_r / (1+p_r)
    - Undirected generators must produce AvgDeg_undir = 2*k_target / (1+p_r)
    """
    if not match_csv_path.exists():
        raise FileNotFoundError(f"Missing empirical match summary: {match_csv_path}")

    df = pd.read_csv(match_csv_path)
    l2 = df[df["Scale"] == "L2"].dropna(subset=["Nodes", "AvgDegree"])
    if l2.empty:
        raise ValueError("No L2 match-level rows found")

    n_mean = int(round(float(l2["Nodes"].mean())))
    k_total = float(l2["AvgDegree"].mean())  # directed: E/N
    recip = float(l2["Reciprocity"].mean())   # empirical reciprocity

    # Mixed orientation: p_r = fraction of edges made bidirectional
    # Reciprocity = 2*p_r / (1+p_r)  =>  p_r = recip / (2 - recip)
    p_r = recip / (2.0 - recip)
    mult = 1.0 + p_r  # E_directed / E_undirected multiplier

    # Undirected target: AvgDeg_undir = 2 * k_total / mult
    k_undir = 2.0 * k_total / mult

    ws_k = max(2, 2 * int(round(k_undir / 2)))  # nearest even integer

    # RGG: expected degree = n * pi * r^2; need r >= connectivity threshold
    r_degree = float(np.sqrt(k_undir / max(n_mean * np.pi, 1e-6)))
    r_connectivity = math.sqrt(math.log(n_mean) / (math.pi * n_mean))
    rgg_r = max(r_degree, 1.30 * r_connectivity)  # ensure near-full connectivity

    return {
        "N_MEAN": n_mean,
        "K_TOTAL": k_total,
        "RECIPROCITY": recip,
        "P_RECIP": p_r,
        "MULT": mult,
        "ER_P": k_undir / max(n_mean - 1, 1),
        "BA_M": max(1, int(round(k_total / mult))),
        "WS_K": ws_k,
        "WS_P_REWIRE": 0.1,
        "RGG_R": rgg_r,
        "RGG_R_DEGREE": r_degree,
        "RGG_R_CONNECTIVITY": r_connectivity,
    }


def _mixed_orient(G_undirected: nx.Graph, p_recip: float,
                  rng: np.random.Generator) -> nx.DiGraph:
    """Orient undirected graph to directed, matching empirical reciprocity.

    For each undirected edge (u,v):
    - With probability p_recip: create both u->v and v->u (bidirectional)
    - With probability 1-p_recip: create one random direction
    """
    G_dir = nx.DiGraph()
    G_dir.add_nodes_from(G_undirected.nodes(data=True))
    for u, v in G_undirected.edges():
        if rng.random() < p_recip:
            G_dir.add_edge(u, v)
            G_dir.add_edge(v, u)
        else:
            if rng.random() < 0.5:
                G_dir.add_edge(u, v)
            else:
                G_dir.add_edge(v, u)
    return G_dir


def generate_synthetic_graph(model_type: str, params: Dict[str, float], seed: int) -> nx.DiGraph:
    """Generate one directed synthetic graph with mixed orientation."""
    n = params["N_MEAN"]
    p_r = params["P_RECIP"]

    if model_type == "ER":
        G = nx.erdos_renyi_graph(n, params["ER_P"], seed=seed)
    elif model_type == "BA":
        G = nx.barabasi_albert_graph(n, params["BA_M"], seed=seed)
    elif model_type == "WS":
        G = nx.watts_strogatz_graph(n, params["WS_K"], params["WS_P_REWIRE"], seed=seed)
    elif model_type == "RGG":
        G = nx.random_geometric_graph(n, params["RGG_R"], seed=seed)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if G.number_of_nodes() == 0:
        return nx.DiGraph()

    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()

    rng = np.random.default_rng(seed + 10000)
    return _mixed_orient(G, p_r, rng)


# ============================================================================
# Percolation evaluation
# ============================================================================

def restricted_fc(sim, component):
    """Restricted threshold estimator (peak search where P_inf >= 5% of P_inf(0))."""
    return find_critical_threshold(sim, component,
                                   min_pinf_fraction=PINF_RESTRICTION_FRACTION)


def process(task):
    model, i, params = task
    out = {"Model": model, "Realisation": i}
    try:
        G = nx.DiGraph(generate_synthetic_graph(model, params, SEED_BASE + i))
        out["Nodes"], out["Edges"] = G.number_of_nodes(), G.number_of_edges()
        sim_e = multi_trial_edge_removal(G, method="error", n_steps=N_STEPS,
                                         n_trials=N_TRIALS, seed=SEED)
        sim_a = edge_removal_simulation(G, method="attack", n_steps=N_STEPS)
        fcE, chiE = restricted_fc(sim_e, "strong")
        fcA, chiA = restricted_fc(sim_a, "strong")
        fcE_raw, _ = find_critical_threshold(sim_e, "strong")
        fcA_raw, _ = find_critical_threshold(sim_a, "strong")
        out.update({
            "AvgDegree": round(out["Edges"] / max(out["Nodes"], 1), 4),
            "Reciprocity": round(float(nx.reciprocity(G)), 4),
            "SCGC_Ratio": round(len(max(nx.strongly_connected_components(G), key=len))
                                / out["Nodes"], 4),
            "fcE_guarded": fcE, "ChiE": chiE,
            "R_Error": round(calculate_robustness_integral(sim_e, "strong"), 4),
            "fcA_guarded": fcA, "ChiA": chiA,
            "R_Attack": round(calculate_robustness_integral(sim_a, "strong"), 4),
            "fcE_raw": fcE_raw, "fcA_raw": fcA_raw,
            "pc_Theory": float(calculate_molloy_reed_threshold(G)),
            "RAI": (fcE - fcA) / fcE if (fcE and fcE > 0 and not np.isnan(fcA)) else np.nan,
        })
        out["error"] = ""
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="5 realisations per model")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    params = load_empirical_l2_targets()
    print(f"Calibration: N={params['N_MEAN']}, k={params['K_TOTAL']:.3f}, "
          f"reciprocity={params['RECIPROCITY']:.3f}")
    n_real = 5 if args.test else N_REALISATIONS
    tasks = [(m, i, params) for m in MODEL_TYPES for i in range(n_real)]
    print(f"Processing {len(tasks)} realisations (random failure with {N_TRIALS} trials)")

    rows, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(process, t) for t in tasks]
        for k, f in enumerate(as_completed(futs), 1):
            r = f.result()
            rows.append(r)
            if r.get("error"):
                print(f"  [error] {r['Model']}#{r['Realisation']}: {r['error']}")
            if k % 50 == 0 or k == len(tasks):
                print(f"  {k}/{len(tasks)} ({time.time()-t0:.0f}s)")

    df = pd.DataFrame(rows).sort_values(["Model", "Realisation"]).reset_index(drop=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "synthetic_mc100_guarded.csv", index=False)

    # Comparison with the shipped results
    shipped = pd.read_csv(SHIPPED)
    shipped = shipped[shipped.Realisation < n_real].sort_values(["Model", "Realisation"]).reset_index(drop=True)
    cols = [c for c in shipped.columns if c not in ("Model", "Realisation", "error")]
    mismatch = []
    for c in cols:
        a, b = df[c].astype(float), shipped[c].astype(float)
        same = (a.isna() & b.isna()) | ((a - b).abs() <= 1e-9)
        if not same.all():
            mismatch.append(f"{c} ({int((~same).sum())} rows)")

    # Model-level summary, realisations with a positive random-failure threshold
    summary = []
    for m in MODEL_TYPES:
        d = df[(df.Model == m) & (df.fcE_guarded > 0)]
        fcE, fcA = d.fcE_guarded.mean(), d.fcA_guarded.mean()
        summary.append({"Model": m, "n": len(d), "fcE": round(fcE, 3), "fcA": round(fcA, 3),
                        "RAI": round((fcE - fcA) / fcE, 3)})
    summary = pd.DataFrame(summary)
    summary.to_csv(RESULTS_DIR / "synthetic_summary_by_model.csv", index=False)
    print("\nModel-level summary (realisations with f_c Error > 0):")
    print(summary.to_string(index=False))

    if mismatch:
        print("\nCOMPARISON FAILED  differs from data/synthetic_mc100_guarded.csv in", ", ".join(mismatch))
        sys.exit(1)
    print(f"\nCOMPARISON PASSED  all {len(df)} regenerated realisations equal the shipped results")


if __name__ == "__main__":
    main()
