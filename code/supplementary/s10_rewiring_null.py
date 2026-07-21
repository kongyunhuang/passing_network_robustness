"""
s10_rewiring_null.py — Degree-preserving rewiring null model (Supplementary S10)
================================================================================
Structural attribution analysis. Samples 50 networks per scale (stratified by
league and season from the revised-protocol valid set), rebuilds each real
network from raw event data and reproduces its published revised-protocol f_c
bit-for-bit (gate A), then generates 5 degree-preserving rewired replicas
(directed configuration model -> DiGraph collapse -> self-loop removal;
gate B checks the pre-collapse degree sequences node-by-node) and runs the
same restricted percolation (100 trials, seed 42) on each replica. Outputs
the per-network results and a per-scale summary of the structural gap
Delta = f_c(rewired) - f_c(real).

Running the full analysis requires the StatsBomb event data (set DATA_ROOT
below or the DATA_ROOT environment variable). The plot-only mode reproduces
the supplementary figure from the shipped summary CSVs and needs no raw data.

Input:  ../../data/match_summary_mc100_guarded.csv (read-only)
        StatsBomb event JSON files under DATA_ROOT (full run only)
        ../../data/supplementary/summary_by_scale.csv + structure_summary.csv
        (plot-only mode)
Output: sampled_networks.csv, rewiring_results.csv, summary_by_scale.csv,
        fig_rewiring_null.png (full run; written next to this script)
        fig_rewiring_delta.pdf/.png (plot-only mode)

Usage:
    cd code/supplementary
    python s10_rewiring_null.py --test        # 3 networks per scale
    python s10_rewiring_null.py --workers 8   # full run
    python s10_rewiring_null.py --plot        # figure from shipped CSVs

Last modified: 2026-07-21
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

_SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_SCRIPT_DIR.parent / "pipeline"))

from data_loader import (                    # noqa: E402
    safe_load_match_events, extract_passes,
    build_substitution_map, apply_substitution_map, get_match_files,
)
from network_builder import ZonalNetworkBuilder      # noqa: E402
from robustness import (                             # noqa: E402
    multi_trial_edge_removal, find_critical_threshold,
)

GUARDED_CSV = REPO_ROOT / "data" / "match_summary_mc100_guarded.csv"
SUPP_DATA = REPO_ROOT / "data" / "supplementary"

# Users must point this at the StatsBomb event data root directory
# (or set the DATA_ROOT environment variable).
DATA_DIR = os.environ.get("DATA_ROOT", "/path/to/statsbomb/datos")

LEAGUE_TO_COMP = {"Premier League": 2, "La Liga": 11}
SEASON_TO_ID = {"2022/23": 235, "2023/24": 281, "2024/25": 317}

N_PER_SCALE = 50
N_REWIRES = 5
SAMPLING_SEED = 26070901
N_TRIALS, SEED, N_STEPS = 100, 42, 50
PINF_RESTRICTION_FRACTION = 0.05
GATE_ATOL = 1e-9

RESULTS_OUT = _SCRIPT_DIR / "rewiring_results.csv"
SAMPLE_OUT = _SCRIPT_DIR / "sampled_networks.csv"


def guarded_fc(sim, component):
    """Restricted threshold (relative 5% floor), as in the revised protocol."""
    return find_critical_threshold(sim, component,
                                   min_pinf_fraction=PINF_RESTRICTION_FRACTION)


def theory_corrected(G):
    """Benchmark 1 - mu/mu_11 from the graph's own degree moments."""
    kin = dict(G.in_degree())
    kout = dict(G.out_degree())
    n = G.number_of_nodes()
    if n == 0:
        return np.nan
    mu = sum(kin.values()) / n
    mu11 = sum(kin[v] * kout[v] for v in G) / n
    if mu <= 0 or mu11 <= mu:          # validity: mu_11/mu > 1
        return np.nan
    return 1.0 - mu / mu11


def sample_networks():
    g = pd.read_csv(GUARDED_CSV)
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
            "Nodes", "Edges", "Mu11_Mu", "pc_Error_Strong", "P0_strong"]
    return pd.concat(parts)[cols].reset_index(drop=True)


def process_network(row):
    """Worker: rebuild -> gate A -> 5 rewires -> restricted percolation."""
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

        # ---- Gate A: graph identity + bit-exact reproduction of the shipped f_c ----
        out["nodes_ok"] = g_multi.number_of_nodes() == row["Nodes"]
        out["edges_ok"] = g_real.number_of_edges() == row["Edges"]
        sim_real = multi_trial_edge_removal(g_multi, method="error", n_steps=N_STEPS,
                                            n_trials=N_TRIALS, seed=SEED)
        fc_real, _ = guarded_fc(sim_real, "strong")
        out["fc_real"] = fc_real
        out["fc_real_published"] = row["pc_Error_Strong"]
        out["fc_ok"] = abs(fc_real - row["pc_Error_Strong"]) <= GATE_ATOL
        out["gateA"] = out["nodes_ok"] and out["edges_ok"] and out["fc_ok"]
        if not out["gateA"]:
            out["error"] = "gate A failed"
            return out

        out["theory_real"] = theory_corrected(g_real)

        # ---- Rewiring x N_REWIRES (gate B: degree-sequence invariant) ----
        nodes_order = list(g_real.nodes())
        in_seq = [g_real.in_degree(v) for v in nodes_order]
        out_seq = [g_real.out_degree(v) for v in nodes_order]
        rng = np.random.default_rng(SAMPLING_SEED + int(row["MatchID"]) * 10
                                    + int(row["TeamID"]) % 10)
        fc_rw, th_rw, eloss, gateB_ok = [], [], [], True
        for j in range(N_REWIRES):
            G_null_multi = nx.directed_configuration_model(
                in_seq, out_seq, seed=int(rng.integers(0, 2**31)))
            # Gate B: pre-collapse degree sequences unchanged node-by-node
            if ([G_null_multi.in_degree(i) for i in range(len(nodes_order))] != in_seq
                    or [G_null_multi.out_degree(i) for i in range(len(nodes_order))] != out_seq):
                gateB_ok = False
                break
            G_null = nx.DiGraph(G_null_multi)
            G_null.remove_edges_from(nx.selfloop_edges(G_null))
            eloss.append(1.0 - G_null.number_of_edges() / max(g_real.number_of_edges(), 1))
            th_rw.append(theory_corrected(G_null))
            sim_n = multi_trial_edge_removal(G_null, method="error", n_steps=N_STEPS,
                                             n_trials=N_TRIALS, seed=SEED)
            fc_j, _ = guarded_fc(sim_n, "strong")
            fc_rw.append(fc_j)
        out["gateB"] = gateB_ok
        out["fc_rewired_mean"] = float(np.nanmean(fc_rw)) if fc_rw else np.nan
        out["fc_rewired_sd"] = float(np.nanstd(fc_rw, ddof=1)) if len(fc_rw) > 1 else np.nan
        out["theory_rewired_mean"] = float(np.nanmean(th_rw)) if th_rw else np.nan
        out["edge_loss_mean"] = float(np.mean(eloss)) if eloss else np.nan
        out["n_rewires_valid"] = int(np.sum(~np.isnan(fc_rw)))
        out["error"] = ""
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
        out.setdefault("gateA", False)
    out["elapsed_s"] = round(time.time() - t0, 2)
    return out


def run_engine(args):
    sample = sample_networks()
    sample.to_csv(SAMPLE_OUT, index=False)
    if args.test:
        sample = sample.groupby("Scale").head(3)
    todo = sample.to_dict("records")
    print(f"Processing {len(todo)} networks (1 real-network gate check + "
          f"{N_REWIRES} rewired replicas each, {N_TRIALS} trials, restricted)")

    rows, n_gA, n_gB = [], 0, 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(process_network, r) for r in todo]
        for i, f in enumerate(as_completed(futs), 1):
            res = f.result()
            rows.append(res)
            if res.get("gateA"):
                n_gA += 1
            if res.get("gateB"):
                n_gB += 1
            if res.get("error"):
                print(f"  [error] {res['Scale']} Match={res['MatchID']}: {res['error']}")
            if i % 20 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} ({time.time()-t0:.0f}s) "
                      f"gate A passed {n_gA}, gate B passed {n_gB}")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_OUT, index=False)
    print(f"\nGate A (graph + restricted percolation reproduced bit-exact): {n_gA}/{len(df)}")
    print(f"Gate B (rewired degree-sequence invariant): {n_gB}/{len(df)}")
    if n_gA < len(df) or n_gB < len(df):
        print("GATE FAILED - summary below is for debugging only, do not cite.")

    ok = df[df.gateA & df.gateB].copy()
    ok["bias_real"] = ok.theory_real - ok.fc_real
    ok["bias_rewired"] = ok.theory_rewired_mean - ok.fc_rewired_mean
    ok["delta_structure"] = ok.fc_rewired_mean - ok.fc_real

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
            "theory_real": round(d.theory_real.mean(), 3),
            "theory_rewired": round(d.theory_rewired_mean.mean(), 3),
            "bias_real": round(d.bias_real.mean(), 3),
            "bias_rewired": round(d.bias_rewired.mean(), 3),
            "bias_rewired_se": round(d.bias_rewired.std(ddof=1) / np.sqrt(n), 3),
            "delta_structure": round(d.delta_structure.mean(), 3),
            "delta_structure_se": round(d.delta_structure.std(ddof=1) / np.sqrt(n), 3),
            "attribution_ratio": round(
                (d.delta_structure.mean() / d.bias_real.mean())
                if abs(d.bias_real.mean()) > 1e-9 else np.nan, 3),
            "edge_loss": round(d.edge_loss_mean.mean(), 4),
        })
    summary = pd.DataFrame(summ)
    summary.to_csv(_SCRIPT_DIR / "summary_by_scale.csv", index=False)
    print("\nPer-scale summary (summary_by_scale.csv):")
    print(summary.to_string(index=False))

    # ---- Figure: real vs rewired vs theory per scale ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    x = np.arange(len(summary))
    w = 0.25
    ax.bar(x - w, summary.fc_real, w, label="Simulated $f_c$ (real)", color="#2166ac")
    ax.bar(x, summary.fc_rewired, w, label="Simulated $f_c$ (degree-preserving rewired)",
           color="#92c5de")
    ax.bar(x + w, summary.theory_real, w, label=r"Theory $1-\mu/\mu_{11}$ (real)",
           color="#b2182b", alpha=0.85)
    ymax = float(max(summary.fc_rewired.max(), summary.theory_real.max()))
    ax.set_ylim(0, ymax * 1.30)
    for i, r in summary.iterrows():
        ax.text(i, max(r.fc_rewired, r.theory_real) + ymax * 0.04,
                f"struct: {r.delta_structure:+.3f}\nrw-bias: {r.bias_rewired:+.3f}",
                ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(summary.Scale)
    ax.set_ylabel("$f_c$")
    ax.set_title("Degree-preserving rewiring null model\n"
                 f"(n = {int(summary.n.iloc[0])} per scale, {N_REWIRES} rewires each, "
                 f"restricted, {N_TRIALS} trials, Error side)", fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(_SCRIPT_DIR / "fig_rewiring_null.png", dpi=150, bbox_inches="tight")
    print("\nFigure saved: fig_rewiring_null.png")


def run_plot():
    """Reproduce the supplementary two-panel figure from the shipped CSVs."""
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
    ax.set_ylabel(r'$\Delta = f_c^{\mathrm{Error}}$(rewired) $- f_c^{\mathrm{Error}}$(empirical)')
    ax.set_xlabel('Spatial scale')
    ax.set_ylim(0, 0.19)
    ax.set_title('(a) Structural fragility gap', fontsize=9.5)
    ax.text(0, 0.005, 'transparency\nonly', ha='center', va='bottom', fontsize=6.5,
            color='#555555', rotation=90)
    ax.spines[['top', 'right']].set_visible(False)

    # (b) what rewiring destroys (L1-L3)
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
    fig.savefig(_SCRIPT_DIR / 'fig_rewiring_delta.pdf')
    fig.savefig(_SCRIPT_DIR / 'fig_rewiring_delta.png', dpi=300)
    print('saved', _SCRIPT_DIR / 'fig_rewiring_delta.pdf')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="3 networks per scale")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--plot", action="store_true",
                    help="only reproduce the figure from the shipped summary CSVs")
    args = ap.parse_args()
    if args.plot:
        run_plot()
    else:
        run_engine(args)


if __name__ == "__main__":
    main()
