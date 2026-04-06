"""
robustness.py — Bond percolation simulation and robustness computation
======================================================================
Performs bond percolation simulations on directed passing networks,
tracking the decay of the strongly connected giant component (SCGC)
and weakly connected giant component (WGCC), and computing percolation
phase transition indicators.

Functionality:
1. Bond percolation simulation: random removal (Error) and targeted removal (Attack)
2. Dual order parameter tracking: SCGC (P_inf_strong) + WGCC (P_inf_weak)
3. Susceptibility computation and critical threshold (f_c) identification
4. Robustness integral R = integral P_inf(f) df
5. Directed Molloy-Reed theoretical percolation threshold

Input:  NetworkX directed graph (DiGraph or MultiDiGraph)
Output: Percolation curve DataFrame, critical threshold, robustness integral

Usage:
    from pipeline.robustness import (
        multi_trial_edge_removal, find_critical_threshold,
        calculate_robustness_integral, calculate_molloy_reed_threshold,
    )
    sim = multi_trial_edge_removal(G, method='error', n_steps=50, n_trials=20)
    fc, chi_max = find_critical_threshold(sim, 'strong')
    R = calculate_robustness_integral(sim, 'strong')

Last modified: 2026-03-27
"""

import random
import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, Tuple


# ============== Core metric computation ==============

def calculate_metrics_directed(G) -> Dict[str, float]:
    """
    Compute connectivity metrics for the current state of a directed network.

    Tracks both strongly connected (SCGC) and weakly connected (WGCC)
    order parameters simultaneously.

    Args:
        G: Directed network (DiGraph or MultiDiGraph)

    Returns:
        dict containing P_inf and Susceptibility for both SCGC and WGCC
    """
    N = G.number_of_nodes()
    if N == 0:
        return {
            'P_inf_strong': 0.0, 'Chi_strong': 0.0,
            'P_inf_weak': 0.0, 'Chi_weak': 0.0,
            'n_nodes': 0, 'n_edges': 0,
        }

    # Strongly connected components
    scc_sizes = sorted(
        [len(c) for c in nx.strongly_connected_components(G)], reverse=True
    )
    max_scc = scc_sizes[0]
    P_inf_strong = max_scc / N
    scc_excl = scc_sizes[1:]  # Exclude only the largest, not all of the same size
    Chi_strong = sum(s**2 for s in scc_excl) / N if scc_excl else 0.0

    # Weakly connected components
    wcc_sizes = sorted(
        [len(c) for c in nx.weakly_connected_components(G)], reverse=True
    )
    max_wcc = wcc_sizes[0]
    P_inf_weak = max_wcc / N
    wcc_excl = wcc_sizes[1:]  # Exclude only the largest
    Chi_weak = sum(s**2 for s in wcc_excl) / N if wcc_excl else 0.0

    return {
        'P_inf_strong': P_inf_strong, 'Chi_strong': Chi_strong,
        'P_inf_weak': P_inf_weak, 'Chi_weak': Chi_weak,
        'n_nodes': N, 'n_edges': G.number_of_edges(),
    }


# ============== Bond percolation simulation ==============

def edge_removal_simulation(
    G,
    method: str = 'error',
    n_steps: int = 50
) -> pd.DataFrame:
    """
    Bond percolation simulation: progressively remove edges, tracking SCGC and WGCC decay.

    Args:
        G: Directed network (DiGraph or MultiDiGraph)
        method: Removal strategy
            'error' = random removal (simulates random passing route disruption)
            'attack' = removal in descending order of endpoint degree sum
                       (simulates tactical suppression of key passing routes)
        n_steps: Number of sampling points for f

    Returns:
        DataFrame: f, P_inf_strong, Chi_strong, P_inf_weak, Chi_weak per step
    """
    # Operate on a simple directed graph (percolation is defined on simple graphs)
    G_work = nx.DiGraph(G)
    E_0 = G_work.number_of_edges()

    if E_0 == 0:
        metrics = calculate_metrics_directed(G_work)
        return pd.DataFrame([{'f': 0.0, **metrics}])

    # Determine edge removal order
    if method == 'error':
        edges = list(G_work.edges())
        random.shuffle(edges)
        removal_order = edges
    elif method == 'attack':
        # Sort by endpoint degree sum in descending order (static ordering, determined once)
        # For directed graphs, use total degree = in_degree + out_degree
        deg = dict(G_work.degree())
        edges = list(G_work.edges())
        edges.sort(key=lambda e: deg[e[0]] + deg[e[1]], reverse=True)
        removal_order = edges
    else:
        raise ValueError(f"Unknown removal method: {method}")

    # Sampling points
    f_values = np.linspace(0, 1, n_steps)
    target_removals = sorted(set(int(f * E_0) for f in f_values))

    # Initial state
    results = [{'f': 0.0, **calculate_metrics_directed(G_work)}]

    edges_removed = 0
    removal_idx = 0

    for target_n in target_removals:
        if target_n == 0:
            continue

        num_to_remove = target_n - edges_removed
        for _ in range(num_to_remove):
            if removal_idx >= len(removal_order):
                break
            edge = removal_order[removal_idx]
            if G_work.has_edge(*edge):
                G_work.remove_edge(*edge)
            removal_idx += 1
            edges_removed += 1

        metrics = calculate_metrics_directed(G_work)
        results.append({'f': edges_removed / E_0, **metrics})

    return pd.DataFrame(results)


def multi_trial_edge_removal(
    G,
    method: str = 'error',
    n_steps: int = 50,
    n_trials: int = 20,
    seed: int = 42
) -> pd.DataFrame:
    """
    Multi-trial bond percolation simulation, returning averaged results.

    Attack mode executes only 1 trial (deterministic ordering).
    Error mode executes n_trials random orderings and averages.

    Args:
        G: Directed network
        method: 'error' or 'attack'
        n_steps: Number of sampling points
        n_trials: Monte Carlo trial count (error mode only)
        seed: Random seed

    Returns:
        DataFrame: Averaged percolation curve
    """
    if method == 'attack':
        return edge_removal_simulation(G, method='attack', n_steps=n_steps)

    rng = random.Random(seed)
    all_results = []

    for trial in range(n_trials):
        random.seed(rng.randint(0, 2**31))
        result = edge_removal_simulation(G, method='error', n_steps=n_steps)
        all_results.append(result)

    # Average
    avg_result = all_results[0].copy()
    n_rows = min(len(r) for r in all_results)

    for col in ['P_inf_strong', 'Chi_strong', 'P_inf_weak', 'Chi_weak']:
        stacked = np.array([r[col].values[:n_rows] for r in all_results])
        avg_result.loc[:n_rows - 1, col] = np.mean(stacked, axis=0)

    return avg_result.iloc[:n_rows].reset_index(drop=True)


# ============== Threshold and integral ==============

def find_critical_threshold(
    sim_results: pd.DataFrame,
    component_type: str = 'strong'
) -> Tuple[float, float]:
    """
    Find the critical threshold f_c (susceptibility peak position) from simulation results.

    Args:
        sim_results: Percolation simulation result DataFrame
        component_type: 'strong' (SCGC) or 'weak' (WGCC)

    Returns:
        (f_c, max_chi): Critical threshold and susceptibility peak value
    """
    if sim_results.empty:
        return 0.0, 0.0

    chi_col = f'Chi_{component_type}'
    if chi_col not in sim_results.columns:
        # Backwards compatibility with old format
        chi_col = 'Susceptibility'
        if chi_col not in sim_results.columns:
            return 0.0, 0.0

    max_chi = sim_results[chi_col].max()
    if max_chi == 0:
        return np.nan, 0.0
    peak_row = sim_results.loc[sim_results[chi_col].idxmax()]
    return peak_row['f'], peak_row[chi_col]


def calculate_robustness_integral(
    sim_results: pd.DataFrame,
    component_type: str = 'strong'
) -> float:
    """
    Compute the robustness integral R = integral P_inf(f) df.

    Args:
        sim_results: Percolation simulation result DataFrame
        component_type: 'strong' (SCGC) or 'weak' (WGCC)

    Returns:
        R: Integral value [0, 1]; larger = more robust
    """
    if sim_results.empty:
        return 0.0

    p_col = f'P_inf_{component_type}'
    if p_col not in sim_results.columns:
        # Backwards compatibility with old format
        p_col = 'P_inf'
        if p_col not in sim_results.columns:
            return 0.0

    df = sim_results.sort_values('f')
    return float(np.trapz(df[p_col], df['f']))


def calculate_molloy_reed_threshold(G) -> float:
    """
    Compute the directed Molloy-Reed theoretical percolation threshold.
    pc = 1 - mu / (mu_11 - mu)
    where mu = <k_in>, mu_11 = <k_in * k_out>

    Based on directed configuration model GSCC percolation theory
    (Boguna & Serrano 2005).

    Args:
        G: NetworkX directed graph (DiGraph or MultiDiGraph)

    Returns:
        fc_theory: Theoretical critical threshold
    """
    # Ensure computation on simple directed graph (consistent with percolation simulation)
    if isinstance(G, nx.MultiDiGraph):
        G = nx.DiGraph(G)
    elif isinstance(G, nx.MultiGraph):
        G = nx.Graph(G)

    if not G.is_directed():
        # Undirected graph fallback: use classical kappa formula
        degrees = [d for n, d in G.degree()]
        if not degrees:
            return np.nan
        k_mean = np.mean(degrees)
        k2_mean = np.mean([d**2 for d in degrees])
        if k_mean == 0:
            return np.nan
        kappa = k2_mean / k_mean
        if kappa <= 2:
            return np.nan
        return 1.0 - 1.0 / (kappa - 1.0)

    # Directed graph: use directed MR formula
    in_deg = np.array([d for _, d in G.in_degree()], dtype=float)
    out_deg = np.array([d for _, d in G.out_degree()], dtype=float)

    if len(in_deg) == 0:
        return np.nan

    mu = np.mean(in_deg)          # <k_in>
    mu_11 = np.mean(in_deg * out_deg)  # <k_in * k_out>

    if mu <= 0 or mu_11 <= 2 * mu:
        # mu_11 <= mu: denominator <= 0, formula inapplicable
        # mu < mu_11 <= 2*mu: fc = 1 - mu/(mu_11 - mu) < 0, no physical meaning
        # i.e. network is already below percolation threshold (GSCC absent or negligible)
        return np.nan

    pc = 1.0 - mu / (mu_11 - mu)
    # Theoretically pc should be in (0, 1); defensive check
    if pc <= 0 or pc >= 1:
        return np.nan
    return pc
