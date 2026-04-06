"""
compute_directed_indicators.py — Directed network structural indicator computation
===================================================================================
Computes directed-specific structural indicators on a simple directed projection,
including reciprocity, flow hierarchy, degree coefficient of variation,
Molloy-Reed ratio, and triadic census.

Called by batch_runner.py for each single-match network.

Input:  NetworkX directed graph (DiGraph or MultiDiGraph)
Output: Dictionary containing 14 directed indicators

Usage:
    from pipeline.compute_directed_indicators import compute_directed_indicators
    indicators = compute_directed_indicators(G)

Last modified: 2026-03-27
"""

import networkx as nx
import numpy as np


DIRECTED_INDICATOR_COLUMNS = [
    "Reciprocity",
    "FlowHierarchy",
    "InDegreeCV",
    "OutDegreeCV",
    "Mu11_Mu",
    "Triad030T",
    "Triad030C",
    "Triad120D",
    "Triad120U",
    "Triad120C",
    "Triad210",
    "Triad300",
    "CyclicTriadFraction",
    "TransitiveTriadFraction",
]


def compute_directed_indicators(G):
    """Compute directed-specific structural indicators on a simple directed projection."""
    G_dir = nx.DiGraph(G)
    n = G_dir.number_of_nodes()
    m = G_dir.number_of_edges()

    result = {}

    if m > 0:
        try:
            result["Reciprocity"] = nx.reciprocity(G_dir)
        except Exception:
            result["Reciprocity"] = 0.0
    else:
        result["Reciprocity"] = 0.0

    if n > 1 and m > 0:
        try:
            result["FlowHierarchy"] = nx.flow_hierarchy(G_dir)
        except nx.NetworkXError:
            result["FlowHierarchy"] = 0.0
    else:
        result["FlowHierarchy"] = 0.0

    if n > 1:
        in_deg = np.array([d for _, d in G_dir.in_degree()], dtype=float)
        out_deg = np.array([d for _, d in G_dir.out_degree()], dtype=float)

        result["InDegreeCV"] = np.std(in_deg) / np.mean(in_deg) if np.mean(in_deg) > 0 else 0.0
        result["OutDegreeCV"] = np.std(out_deg) / np.mean(out_deg) if np.mean(out_deg) > 0 else 0.0

        mu = np.mean(in_deg)
        mu_11 = np.mean(in_deg * out_deg)
        result["Mu11_Mu"] = (mu_11 / mu) if mu > 0 else 0.0
    else:
        result["InDegreeCV"] = 0.0
        result["OutDegreeCV"] = 0.0
        result["Mu11_Mu"] = 0.0

    if n >= 3:
        try:
            census = nx.triadic_census(G_dir)
            result["Triad030T"] = census.get("030T", 0)
            result["Triad030C"] = census.get("030C", 0)
            result["Triad120D"] = census.get("120D", 0)
            result["Triad120U"] = census.get("120U", 0)
            result["Triad120C"] = census.get("120C", 0)
            result["Triad210"] = census.get("210", 0)
            result["Triad300"] = census.get("300", 0)

            total_nontrivial = sum(v for k, v in census.items() if k not in {"003", "012"})
            if total_nontrivial > 0:
                cyclic_triads = (
                    census.get("030C", 0)
                    + census.get("120C", 0)
                    + census.get("210", 0)
                    + census.get("300", 0)
                )
                transitive_triads = census.get("030T", 0) + census.get("120D", 0)
                result["CyclicTriadFraction"] = cyclic_triads / total_nontrivial
                result["TransitiveTriadFraction"] = transitive_triads / total_nontrivial
            else:
                result["CyclicTriadFraction"] = 0.0
                result["TransitiveTriadFraction"] = 0.0
        except Exception:
            for key in DIRECTED_INDICATOR_COLUMNS[5:]:
                result[key] = 0.0
    else:
        for key in DIRECTED_INDICATOR_COLUMNS[5:]:
            result[key] = 0.0

    return result
