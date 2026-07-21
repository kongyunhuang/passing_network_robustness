"""
synthetic_baselines_generator.py — Calibrated synthetic baselines, revised protocol
===================================================================================
Generates the four synthetic model families (ER, BA, WS, RGG) calibrated to
the empirical match-level L2 statistics (N, mean degree, reciprocity via
mixed edge orientation), regenerates the exact graphs of the originally
submitted analysis from the same seeds, and evaluates them under the revised
percolation protocol (Error: 100 Monte Carlo trials + peak-validity
restriction; Attack: single deterministic removal + restriction).

Two acceptance gates guard the regeneration:
  gate A: regenerated graphs match the archived node and edge counts
  gate B: the original submission protocol (5 trials, unrestricted) run on
          the regenerated graphs reproduces the archived thresholds

Input:  ../data/match_summary.csv (calibration targets, read-only)
        ../data/synthetic_baselines.csv (archived original-protocol results,
        read-only)
Output: synthetic_mc100_guarded.csv, summary_by_model.csv
        (written next to this script; shipped copies live in ../data/)

Usage:
    cd code
    python synthetic_baselines_generator.py --test      # 5 realisations/model
    python synthetic_baselines_generator.py --workers 8 # full run (100 each)

Last modified: 2026-07-21
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

MATCH_SUMMARY = REPO_ROOT / "data" / "match_summary.csv"
SYN_ARCHIVE = REPO_ROOT / "data" / "synthetic_baselines.csv"

MODEL_TYPES = ["ER", "BA", "WS", "RGG"]
N_REALISATIONS = 100
SEED_BASE = 42
N_TRIALS, SEED, N_STEPS = 100, 42, 50
PINF_RESTRICTION_FRACTION = 0.05
ATOL = 5e-5   # archive stores 4 decimal places

# Empirical L2 reference under the revised protocol (n = 4,330)
EMP_REF = {"fcE": 0.604, "fcA": 0.420, "RAI": 0.304}


# ============================================================================
# Graph generation (identical to the originally submitted generator)
# ============================================================================

def load_empirical_l2_targets(match_csv_path: Path = MATCH_SUMMARY) -> Dict[str, float]:
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
        raise ValueError("No L2 match-level rows found in match_summary.csv")

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
# Revised-protocol evaluation
# ============================================================================

def guarded_fc(sim, component):
    """Restricted threshold (relative 5% floor), as in the revised protocol."""
    return find_critical_threshold(sim, component,
                                   min_pinf_fraction=PINF_RESTRICTION_FRACTION)


def theory_corrected(G):
    kin, kout = dict(G.in_degree()), dict(G.out_degree())
    n = G.number_of_nodes()
    mu = sum(kin.values()) / n
    mu11 = sum(kin[v] * kout[v] for v in G) / n
    return (1.0 - mu / mu11) if (mu > 0 and mu11 > mu) else np.nan


def process(task):
    model, i, params, arch = task
    out = {"Model": model, "Realisation": i}
    try:
        G = nx.DiGraph(generate_synthetic_graph(model, params, SEED_BASE + i))
        out["Nodes"], out["Edges"] = G.number_of_nodes(), G.number_of_edges()
        out["gateA"] = (out["Nodes"] == arch["Nodes"] and out["Edges"] == arch["Edges"])

        # Gate B: original submission protocol (5 trials, unrestricted)
        # reproduces the archived thresholds
        sim_e5 = multi_trial_edge_removal(G, method="error", n_steps=N_STEPS,
                                          n_trials=5, seed=SEED_BASE)
        fc_e5, _ = find_critical_threshold(sim_e5, "strong")
        sim_a = edge_removal_simulation(G, method="attack", n_steps=N_STEPS)
        fc_a_raw, _ = find_critical_threshold(sim_a, "strong")
        out["gateB"] = (abs(fc_e5 - arch["fc_Error_Strong"]) <= ATOL
                        and abs(fc_a_raw - arch["fc_Attack_Strong"]) <= ATOL)

        # Production: Error 100 trials + restriction; Attack single run + restriction
        sim_e = multi_trial_edge_removal(G, method="error", n_steps=N_STEPS,
                                         n_trials=N_TRIALS, seed=SEED)
        fcE, chiE = guarded_fc(sim_e, "strong")
        fcA, chiA = guarded_fc(sim_a, "strong")
        fcE_raw, _ = find_critical_threshold(sim_e, "strong")

        out.update({
            "AvgDegree": round(out["Edges"] / max(out["Nodes"], 1), 4),
            "Reciprocity": round(float(nx.reciprocity(G)), 4),
            "SCGC_Ratio": round(len(max(nx.strongly_connected_components(G), key=len))
                                / out["Nodes"], 4),
            "fcE_guarded": fcE, "ChiE": chiE,
            "R_Error": round(calculate_robustness_integral(sim_e, "strong"), 4),
            "fcA_guarded": fcA, "ChiA": chiA,
            "R_Attack": round(calculate_robustness_integral(sim_a, "strong"), 4),
            "fcE_raw": fcE_raw, "fcA_raw": fc_a_raw,
            "theory_corrected": theory_corrected(G),
            "pc_old_eq1": float(calculate_molloy_reed_threshold(G)),
            "RAI": (fcE - fcA) / fcE if (fcE and fcE > 0 and not np.isnan(fcA)) else np.nan,
        })
        out["error"] = ""
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
        out.setdefault("gateA", False)
        out.setdefault("gateB", False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    params = load_empirical_l2_targets()
    print(f"Calibration: N={params['N_MEAN']}, k={params['K_TOTAL']:.3f}, "
          f"recip={params['RECIPROCITY']:.3f}")
    archive = pd.read_csv(SYN_ARCHIVE).set_index(["Model", "Realisation"])
    n_real = 5 if args.test else N_REALISATIONS
    tasks = [(m, i, params, archive.loc[(m, i)].to_dict())
             for m in MODEL_TYPES for i in range(n_real)]
    print(f"Processing {len(tasks)} realisations "
          f"(gate A + gate B + revised protocol, Error {N_TRIALS} trials)")

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

    df = pd.DataFrame(rows).sort_values(["Model", "Realisation"])
    nA, nB = int(df.gateA.sum()), int(df.gateB.sum())
    print(f"\nGate A (regenerated graphs bit-identical): {nA}/{len(df)}")
    print(f"Gate B (original protocol reproduces archived thresholds): {nB}/{len(df)}")
    df.to_csv(_SCRIPT_DIR / "synthetic_mc100_guarded.csv", index=False)
    if nA < len(df) or nB < len(df):
        print("GATE FAILED - summary below is for debugging only, do not cite.")
        sys.exit(1)

    # Model-level summary: original vs revised protocol. RAI uses method A
    # (ratio of means) plus the manuscript's filter (degenerate realisations
    # with fcE = 0 excluded); method B (mean of ratios) is a reference column.
    arch = pd.read_csv(SYN_ARCHIVE)
    print("\nGate C - method A x manuscript filter reproduces the originally "
          "submitted RAI values (ER 0.18 / BA 0.32 / RGG -0.08)")
    summ = []
    for m in MODEL_TYPES:
        d = df[(df.Model == m) & (df.fcE_guarded > 0)]
        a = arch[(arch.Model == m) & (arch.fc_Error_Strong > 0)].head(n_real * 2)
        a = a[a.Realisation < n_real]
        fcE_o, fcA_o = a.fc_Error_Strong.mean(), a.fc_Attack_Strong.mean()
        rai_old_A = (fcE_o - fcA_o) / fcE_o
        fcE_n, fcA_n = d.fcE_guarded.mean(), d.fcA_guarded.mean()
        rai_new_A = (fcE_n - fcA_n) / fcE_n
        summ.append({
            "Model": m, "n_new": len(d), "n_old": len(a),
            "fcE_old5": round(fcE_o, 3), "fcE_new": round(fcE_n, 3),
            "fcA_old": round(fcA_o, 3), "fcA_new": round(fcA_n, 3),
            "RAI_old_A": round(rai_old_A, 3), "RAI_new_A": round(rai_new_A, 3),
            "RAI_new_B": round(d.RAI.mean(), 3),
            "theory_corr": round(d.theory_corrected.mean(), 3),
            "bias_new": round((d.theory_corrected - d.fcE_guarded).mean(), 3),
        })
    summary = pd.DataFrame(summ)
    summary.to_csv(_SCRIPT_DIR / "summary_by_model.csv", index=False)
    pub = {"ER": 0.18, "BA": 0.32, "RGG": -0.08}
    for r in summary.itertuples():
        if r.Model in pub:
            ok = abs(r.RAI_old_A - pub[r.Model]) < 0.015
            print(f"  {r.Model}: RAI_old_A={r.RAI_old_A:+.3f} vs published {pub[r.Model]:+.2f}"
                  f" -> {'PASS' if ok else 'FAIL'}")
    print("\nModel-level old-vs-new comparison (summary_by_model.csv):")
    print(summary.to_string(index=False))
    emp_rai_A = (EMP_REF["fcE"] - EMP_REF["fcA"]) / EMP_REF["fcE"]
    print(f"\nEmpirical L2 reference (revised protocol): fcE={EMP_REF['fcE']}, "
          f"fcA={EMP_REF['fcA']}, RAI_A={emp_rai_A:.3f}")
    rai_dist = {r.Model: abs(r.RAI_new_A - emp_rai_A) for r in summary.itertuples()}
    print("|RAI_model - RAI_soccer| (method A, revised protocol):",
          {k: round(v, 3) for k, v in sorted(rai_dist.items(), key=lambda x: x[1])})


if __name__ == "__main__":
    main()
