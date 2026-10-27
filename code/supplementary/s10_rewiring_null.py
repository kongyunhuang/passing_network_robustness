"""
s10_rewiring_null.py  Degree-preserving rewiring null model (Fig. S6)
====================================================================
Samples 50 networks per scale (stratified by league and season), rebuilds each
network from the raw event data and reproduces its published random-failure
threshold and structural metrics exactly (gates A and C). It then draws 5
degree-preserving rewired copies (directed configuration model, collapsed to a
simple graph without self-loops, gate B checks the degree sequences node by
node), applies the same restricted percolation protocol (100 trials, seed 42)
and computes clustering, reciprocity, flow hierarchy and mu_11/mu on each copy.
The plot mode draws Fig. S6 from the shipped summary files and needs no raw data.

Input  data/match_summary_mc100_guarded.csv and raw event data under DATA_ROOT (full run),
       data/supplementary/summary_by_scale.csv and structure_summary.csv (plot mode)
Output  results/figS6_sampled_networks.csv, results/figS6_rewiring_results.csv,
        results/figS6_summary_by_scale.csv, results/figS6_structure_summary.csv (full run),
        figures/figS6.pdf, figures/figS6.png (plot mode)

Usage
  python code/supplementary/s10_rewiring_null.py --plot          # Fig. S6 from shipped files
  DATA_ROOT=/path/to/statsbomb/datos python code/supplementary/s10_rewiring_null.py --test
  DATA_ROOT=/path/to/statsbomb/datos python code/supplementary/s10_rewiring_null.py --workers 8

Last modified 2026-10-27
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "code" / "pipeline"))

from data_loader import (                    # noqa: E402
    safe_load_match_events, extract_passes,
    build_substitution_map, apply_substitution_map, get_match_files,
)
from network_builder import ZonalNetworkBuilder      # noqa: E402
from robustness import (                             # noqa: E402
    multi_trial_edge_removal, find_critical_threshold,
)
from compute_directed_indicators import compute_directed_indicators  # noqa: E402

# Published analysis (100 Monte Carlo trials at all scales, restricted threshold estimator)
PUBLISHED_CSV = BASE_DIR / "data" / "match_summary_mc100_guarded.csv"
SUPP_DATA = BASE_DIR / "data" / "supplementary"
FIG_DIR = BASE_DIR / "figures"
RESULTS_DIR = BASE_DIR / "results"

# Root directory of the StatsBomb event data (set the DATA_ROOT environment variable)
DATA_DIR = os.environ.get("DATA_ROOT", "/path/to/statsbomb/datos")

LEAGUE_TO_COMP = {"Premier League": 2, "La Liga": 11}
SEASON_TO_ID = {"2022/23": 235, "2023/24": 281, "2024/25": 317}

N_PER_SCALE = 50
N_REWIRES = 5
SAMPLING_SEED = 26070901
N_TRIALS, SEED, N_STEPS = 100, 42, 50
PINF_RESTRICTION_FRACTION = 0.05
GATE_ATOL = 1e-9
STRUCTURE_METRICS = ["Clustering", "Reciprocity", "FlowHierarchy", "Mu11_Mu"]

RESULTS_OUT = RESULTS_DIR / "figS6_rewiring_results.csv"
SAMPLE_OUT = RESULTS_DIR / "figS6_sampled_networks.csv"
SUMMARY_OUT = RESULTS_DIR / "figS6_summary_by_scale.csv"
STRUCTURE_OUT = RESULTS_DIR / "figS6_structure_summary.csv"

# Supplementary Section S10, mean Delta and its standard error per scale, and
# the L2 structural comparison (empirical, rewired)
DELTA_S10 = {"L0": (0.130, 0.030), "L1": (0.095, 0.020),
             "L2": (0.065, 0.013), "L3": (0.026, 0.012)}
STRUCTURE_L2 = {"Clustering": (0.078, 0.032), "Reciprocity": (0.223, 0.049),
                "FlowHierarchy": (0.209, 0.211)}


def restricted_fc(sim, component):
    """Restricted threshold, argmax of chi where P_inf >= 0.05 P_inf(0)."""
    return find_critical_threshold(sim, component,
                                   min_pinf_fraction=PINF_RESTRICTION_FRACTION)


def structure_metrics(G):
    """Clustering, reciprocity, flow hierarchy and mu_11/mu of a directed graph,
    computed with the same functions as the match-level values in Table 1."""
    ind = compute_directed_indicators(G)
    return {"Clustering": float(nx.average_clustering(G)),
            "Reciprocity": ind["Reciprocity"],
            "FlowHierarchy": ind["FlowHierarchy"],
            "Mu11_Mu": ind["Mu11_Mu"]}


def sample_networks():
    g = pd.read_csv(PUBLISHED_CSV)
    valid = g[(g.Mu11_Mu > 1) & (g.pc_Error_Strong > 0)]
    rng = np.random.default_rng(SAMPLING_SEED)
    parts = []
    for scale in ["L0", "L1", "L2", "L3"]:
        sub = valid[valid.Scale == scale]
        strata = sorted(sub.groupby(["League", "Season"]).groups.keys())
        per = N_PER_SCALE // len(strata) if strata else 0
        extra = N_PER_SCALE - per * len(strata)
        picked = []
        for i, key in enumerate(strata):
            grp = sub[(sub.League == key[0]) & (sub.Season == key[1])]
            k = min(per + (1 if i < extra else 0), len(grp))
            picked.append(grp.loc[rng.choice(grp.index.to_numpy(), size=k, replace=False)])
        got = pd.concat(picked) if picked else sub.iloc[:0]
        # If any stratum is short, top up from the remaining valid rows of the scale
        if len(got) < min(N_PER_SCALE, len(sub)):
            rest = sub.drop(index=got.index)
            need = min(N_PER_SCALE, len(sub)) - len(got)
            got = pd.concat([got, rest.loc[rng.choice(rest.index.to_numpy(),
                                                      size=need, replace=False)]])
        parts.append(got)
    cols = ["League", "Season", "Team", "TeamID", "Venue", "MatchID", "Scale",
            "Nodes", "Edges", "Mu11_Mu", "pc_Error_Strong", "P0_strong",
            "Clustering", "Reciprocity", "FlowHierarchy"]
    return pd.concat(parts)[cols].reset_index(drop=True)


def process_network(row):
    """Worker, rebuild the network, run gates A and C, then 5 rewired copies."""
    t0 = time.time()
    out = dict(MatchID=row["MatchID"], TeamID=row["TeamID"], Scale=row["Scale"],
               League=row["League"], Season=row["Season"])
    try:
        comp = LEAGUE_TO_COMP[row["League"]]
        season = SEASON_TO_ID[row["Season"]]
        files = get_match_files(DATA_DIR, comp, season)
        target = [f for f in files if f.name.startswith(f"{int(row['MatchID'])}_")]
        events = safe_load_match_events(str(target[0]))
        passes = extract_passes(events, match_id=int(row["MatchID"]),
                                team_id=int(row["TeamID"]))
        sub_map = build_substitution_map(events, team_id=int(row["TeamID"]))
        apply_substitution_map(passes, sub_map)
        g_multi = ZonalNetworkBuilder().build_network(passes, scale=row["Scale"])
        g_real = nx.DiGraph(g_multi)

        # Gate A, same graph and exact reproduction of the published threshold
        out["nodes_ok"] = g_multi.number_of_nodes() == row["Nodes"]
        out["edges_ok"] = g_real.number_of_edges() == row["Edges"]
        sim_real = multi_trial_edge_removal(g_multi, method="error", n_steps=N_STEPS,
                                            n_trials=N_TRIALS, seed=SEED)
        fc_real, _ = restricted_fc(sim_real, "strong")
        out["fc_real"] = fc_real
        out["fc_real_published"] = row["pc_Error_Strong"]
        out["fc_ok"] = abs(fc_real - row["pc_Error_Strong"]) <= GATE_ATOL
        out["gateA"] = out["nodes_ok"] and out["edges_ok"] and out["fc_ok"]
        if not out["gateA"]:
            out["error"] = "gate A failed"
            return out

        # Gate C, structural metrics of the empirical graph equal the shipped
        # match-level values in data/match_summary_mc100_guarded.csv
        m_real = structure_metrics(g_real)
        for k, v in m_real.items():
            out[f"real_{k}"] = v
        out["gateC"] = all(abs(m_real[k] - row[k]) <= GATE_ATOL for k in STRUCTURE_METRICS)

        # Rewired copies, gate B checks that the degree sequences are unchanged
        nodes_order = list(g_real.nodes())
        in_seq = [g_real.in_degree(v) for v in nodes_order]
        out_seq = [g_real.out_degree(v) for v in nodes_order]
        rng = np.random.default_rng(SAMPLING_SEED + int(row["MatchID"]) * 10
                                    + int(row["TeamID"]) % 10)
        fc_rw, eloss, gateB_ok = [], [], True
        m_rw = {k: [] for k in STRUCTURE_METRICS}
        for j in range(N_REWIRES):
            G_null_multi = nx.directed_configuration_model(
                in_seq, out_seq, seed=int(rng.integers(0, 2**31)))
            # Gate B, degree sequences before the collapse match node by node
            if ([G_null_multi.in_degree(i) for i in range(len(nodes_order))] != in_seq
                    or [G_null_multi.out_degree(i) for i in range(len(nodes_order))] != out_seq):
                gateB_ok = False
                break
            G_null = nx.DiGraph(G_null_multi)
            G_null.remove_edges_from(nx.selfloop_edges(G_null))
            eloss.append(1.0 - G_null.number_of_edges() / max(g_real.number_of_edges(), 1))
            for k, v in structure_metrics(G_null).items():
                m_rw[k].append(v)
            sim_n = multi_trial_edge_removal(G_null, method="error", n_steps=N_STEPS,
                                             n_trials=N_TRIALS, seed=SEED)
            fc_j, _ = restricted_fc(sim_n, "strong")
            fc_rw.append(fc_j)
        out["gateB"] = gateB_ok
        out["fc_rewired_mean"] = float(np.nanmean(fc_rw)) if fc_rw else np.nan
        out["fc_rewired_sd"] = float(np.nanstd(fc_rw, ddof=1)) if len(fc_rw) > 1 else np.nan
        out["edge_loss_mean"] = float(np.mean(eloss)) if eloss else np.nan
        out["n_rewires_valid"] = int(np.sum(~np.isnan(fc_rw)))
        for k, vs in m_rw.items():
            out[f"rewired_{k}"] = float(np.mean(vs)) if vs else np.nan
        out["error"] = ""
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
        out.setdefault("gateA", False)
    out["elapsed_s"] = round(time.time() - t0, 2)
    return out


def summarise_percolation(ok):
    """Scale-level summary of Fig. S6a, same columns as data/supplementary/summary_by_scale.csv."""
    ok = ok.assign(delta_structure=ok.fc_rewired_mean - ok.fc_real)
    summ = []
    for sc in ["L0", "L1", "L2", "L3"]:
        d = ok[ok.Scale == sc]
        n = len(d)
        if n == 0:
            continue
        summ.append({
            "Scale": sc, "n": n,
            "fc_real": round(d.fc_real.mean(), 3),
            "fc_rewired": round(d.fc_rewired_mean.mean(), 3),
            "delta_structure": round(d.delta_structure.mean(), 3),
            "delta_structure_se": round(d.delta_structure.std(ddof=1) / np.sqrt(n), 3),
            "edge_loss": round(d.edge_loss_mean.mean(), 4),
        })
    return pd.DataFrame(summ)


def summarise_structure(ok):
    """Scale-level summary of Fig. S6b, same columns as data/supplementary/structure_summary.csv."""
    summ = []
    for sc in ["L0", "L1", "L2", "L3"]:
        d = ok[ok.Scale == sc]
        if len(d) == 0:
            continue
        row = {"Scale": sc, "n": len(d)}
        for k in STRUCTURE_METRICS:
            row[f"real_{k}"] = round(d[f"real_{k}"].mean(), 3)
            row[f"rewired_{k}"] = round(d[f"rewired_{k}"].mean(), 3)
        summ.append(row)
    return pd.DataFrame(summ)


def compare_with_shipped(computed, shipped_path):
    """True if a computed scale-level summary equals the shipped file value by value."""
    shipped = pd.read_csv(shipped_path)
    if list(computed.columns) != list(shipped.columns) or len(computed) != len(shipped):
        print(f"  {shipped_path.name}  columns or rows differ")
        return False
    same = True
    for col in shipped.columns:
        if col == "Scale":
            same &= bool((computed[col].values == shipped[col].values).all())
            continue
        diff = np.abs(computed[col].to_numpy(float) - shipped[col].to_numpy(float))
        if (diff > GATE_ATOL).any():
            same = False
            print(f"  {shipped_path.name}  column {col} differs, computed "
                  f"{computed[col].tolist()}, shipped {shipped[col].tolist()}")
    print(f"  {shipped_path.name:<24} {'PASS' if same else 'FAIL'}")
    return same


def run_engine(args):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sample = sample_networks()
    sample.to_csv(SAMPLE_OUT, index=False)
    if args.test:
        sample = sample.groupby("Scale").head(3)
    todo = sample.to_dict("records")
    print(f"Processing {len(todo)} networks (1 gate check on the empirical network + "
          f"{N_REWIRES} rewired copies each, {N_TRIALS} trials, restricted)")

    rows, n_gA, n_gB, n_gC = [], 0, 0, 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(process_network, r) for r in todo]
        for i, f in enumerate(as_completed(futs), 1):
            res = f.result()
            rows.append(res)
            n_gA += bool(res.get("gateA"))
            n_gB += bool(res.get("gateB"))
            n_gC += bool(res.get("gateC"))
            if res.get("error"):
                print(f"  [error] {res['Scale']} Match={res['MatchID']}: {res['error']}")
            if i % 20 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} ({time.time()-t0:.0f}s) gate A passed {n_gA}, "
                      f"gate B passed {n_gB}, gate C passed {n_gC}")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_OUT, index=False)
    print(f"\nGate A (graph and restricted threshold reproduced exactly): {n_gA}/{len(df)}")
    print(f"Gate B (rewired degree sequences unchanged): {n_gB}/{len(df)}")
    print(f"Gate C (empirical structural metrics equal the shipped values): {n_gC}/{len(df)}")
    failed = n_gA < len(df) or n_gB < len(df) or n_gC < len(df)
    if failed:
        print("GATE FAILED  the summaries below are for debugging only")

    passed = df.reindex(columns=["gateA", "gateB", "gateC"]).fillna(False).astype(bool).all(axis=1)
    ok = df[passed].copy()
    summary = summarise_percolation(ok)
    structure = summarise_structure(ok)
    summary.to_csv(SUMMARY_OUT, index=False)
    structure.to_csv(STRUCTURE_OUT, index=False)
    print(f"\nPer-scale summary of Fig. S6a ({SUMMARY_OUT.relative_to(BASE_DIR)}):")
    print(summary.to_string(index=False))
    print(f"\nPer-scale summary of Fig. S6b ({STRUCTURE_OUT.relative_to(BASE_DIR)}):")
    print(structure.to_string(index=False))
    print(f"Saved {SAMPLE_OUT.relative_to(BASE_DIR)}, {RESULTS_OUT.relative_to(BASE_DIR)}, "
          f"{SUMMARY_OUT.relative_to(BASE_DIR)}, {STRUCTURE_OUT.relative_to(BASE_DIR)}")

    if args.test:
        print("\nTest subset, the scale-level summaries are not compared with the shipped files")
    else:
        print("\nScale-level summaries against the shipped files")
        failed |= not compare_with_shipped(summary, SUPP_DATA / "summary_by_scale.csv")
        failed |= not compare_with_shipped(structure, SUPP_DATA / "structure_summary.csv")
    if failed:
        print("GATE FAILED")
        sys.exit(1)
    print("GATE PASS")


def check_shipped(s, st):
    """Acceptance check of the shipped summaries against Supplementary Section S10."""
    checks = [(f"{sc} networks", s.loc[sc, "n"], 50, 0) for sc in DELTA_S10]
    for sc, (delta, se) in DELTA_S10.items():
        checks.append((f"{sc} Delta", s.loc[sc, "delta_structure"], delta, 0.0005))
        checks.append((f"{sc} Delta SE", s.loc[sc, "delta_structure_se"], se, 0.0005))
    for col, (real, rewired) in STRUCTURE_L2.items():
        checks.append((f"L2 {col} empirical", st.loc["L2", f"real_{col}"], real, 0.0005))
        checks.append((f"L2 {col} rewired", st.loc["L2", f"rewired_{col}"], rewired, 0.0005))
    failed = False
    for label, got, exp, tol in checks:
        ok = abs(got - exp) <= tol + 1e-12
        failed |= not ok
        print(f"  {label:<28} {got:8.3f} vs {exp:.3f}  {'PASS' if ok else 'FAIL'}")
    deltas = [s.loc[sc, "delta_structure"] for sc in ["L1", "L2", "L3"]]
    ok = deltas[0] > deltas[1] > deltas[2]
    failed |= not ok
    print(f"  {'Delta decreases L1 > L2 > L3':<28} {'PASS' if ok else 'FAIL'}")
    if failed:
        print("GATE FAILED  shipped summaries differ from Supplementary Section S10")
        sys.exit(1)
    print("GATE PASS  shipped summaries match Supplementary Section S10")


def run_plot():
    """Draw Fig. S6 from the shipped summary files."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times'],
        'font.size': 9, 'axes.linewidth': 0.7, 'pdf.fonttype': 42,
        'mathtext.fontset': 'cm', 'axes.unicode_minus': False,
    })

    NAVY, RED, GREY = '#1B4F72', '#C0392B', '#8B9DAF'

    s = pd.read_csv(SUPP_DATA / 'summary_by_scale.csv').set_index('Scale')
    st = pd.read_csv(SUPP_DATA / 'structure_summary.csv').set_index('Scale')
    check_shipped(s, st)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))

    # (a) Delta structure bars
    ax = axes[0]
    scales = ['L0', 'L1', 'L2', 'L3']
    vals = [s.loc[sc, 'delta_structure'] for sc in scales]
    errs = [s.loc[sc, 'delta_structure_se'] for sc in scales]
    colors = [GREY, NAVY, NAVY, NAVY]
    ax.bar(scales, vals, yerr=errs, capsize=3, width=0.55,
           color=colors, edgecolor='black', linewidth=0.5, alpha=0.9,
           error_kw={'elinewidth': 0.8})
    for x, v, e in zip(range(4), vals, errs):
        ax.text(x, v + e + 0.006, f'+{v:.3f}', ha='center', fontsize=8)
    ax.axhline(0, color='black', lw=0.7)
    ax.set_ylabel(r'$\Delta = f_{c,\ell}^{\mathrm{Error}}$(rewired) $- f_{c,\ell}^{\mathrm{Error}}$(empirical)')
    ax.set_xlabel('Spatial scale')
    ax.set_ylim(0, 0.19)
    ax.set_title('(a) Structural fragility gap', fontsize=9.5)
    ax.text(0, 0.005, 'transparency\nonly', ha='center', va='bottom', fontsize=6.5,
            color='#555555', rotation=90)
    ax.spines[['top', 'right']].set_visible(False)

    # (b) what rewiring destroys (L1 to L3)
    ax = axes[1]
    sc3 = ['L1', 'L2', 'L3']
    x = np.arange(3)
    w = 0.13
    pairs = [('Clustering', r'$C$'), ('Reciprocity', r'$\rho_{\leftrightarrow}$'),
             ('FlowHierarchy', r'$h$')]
    for i, (col, lab) in enumerate(pairs):
        off = (i - 1) * 2.3 * w
        rv = [st.loc[sc, f'real_{col}'] for sc in sc3]
        wv = [st.loc[sc, f'rewired_{col}'] for sc in sc3]
        ax.bar(x + off - w/2, rv, w, color=NAVY, edgecolor='black', lw=0.4,
               label='empirical' if i == 0 else None)
        ax.bar(x + off + w/2, wv, w, color=RED, edgecolor='black', lw=0.4, alpha=0.85,
               label='rewired' if i == 0 else None)
        for xi in range(3):
            ax.text(x[xi] + off, max(rv[xi], wv[xi]) + 0.015, lab, ha='center', fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels(sc3)
    ax.set_ylabel('Metric value')
    ax.set_xlabel('Spatial scale')
    ax.set_ylim(0, 0.72)
    ax.set_title('(b) What rewiring destroys', fontsize=9.5)
    ax.legend(frameon=False, fontsize=8, loc='upper center', ncol=2, bbox_to_anchor=(0.55, 1.02))
    ax.spines[['top', 'right']].set_visible(False)

    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / 'figS6.pdf')
    fig.savefig(FIG_DIR / 'figS6.png', dpi=300)
    print('Saved figures/figS6.pdf, figures/figS6.png')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="3 networks per scale")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--plot", action="store_true",
                    help="only draw Fig. S6 from the shipped summary files")
    args = ap.parse_args()
    if args.plot:
        run_plot()
    else:
        run_engine(args)


if __name__ == "__main__":
    main()
