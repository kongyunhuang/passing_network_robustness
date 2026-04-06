"""
fig_baena_case.py — Baena case study: edge removal network visualisation (Fig. 8)
=================================================================================
Illustrates Baena's structural contribution in Villarreal vs Girona (24/25
season, match 3946543):
(a) Complete passing network, highlighting Baena's critical edges (red) and
    non-critical edges (dashed orange)
(b) Network after removing all of Baena's edges, SCGC drops by 39%
(c) Horizontal bar chart ranking each critical edge by SCGC drop + cumulative line

Note: This script requires raw StatsBomb event JSON data and the shared data
      pipeline modules (network_builder, robustness). Users must set the
      DATA_ROOT path and ensure the pipeline directory is on sys.path.

Input:  StatsBomb raw event JSON (DATA_ROOT), data/match_summary.csv
Output: figures/fig8.png, figures/fig8.pdf

Usage:
  python code/fig_baena_case.py

Last modified: 2026-03-27
"""

import matplotlib
matplotlib.use('Agg')
import sys
import json
import glob
import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ============================================================================
# Paths
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATA_ROOT: Users must set this path to the root directory of StatsBomb
# raw event JSON data. Directory structure should be:
# DATA_ROOT/{competition_id}_{season_id}/{match_id}_events.json
# ============================================================================
# Set this to your local StatsBomb event data directory
DATA_ROOT = Path('data/raw')

# ============================================================================
# Shared data pipeline dependencies: network_builder, robustness
# Users must ensure these modules are available on the Python path
# ============================================================================
sys.path.insert(0, str(Path(__file__).resolve().parent / 'pipeline'))
from network_builder import ZonalNetworkBuilder
from robustness import calculate_metrics_directed

# ============================================================================
# Publication style
# ============================================================================
plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 11,
    'figure.dpi': 300, 'savefig.dpi': 300,
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
})

builder = ZonalNetworkBuilder()

MATCH_ID = 3946543
TEAM = 'Villarreal'
TARGET_PLAYER = 'Alejandro Baena Rodriguez'

SHORT_NAMES = {'Alejandro Baena Rodriguez': 'Baena'}
LABEL_OFFSETS = {'Baena': (8, -5)}
DEFAULT_OFFSET = (0, 4)

# Colors
C_CRIT = '#c0392b'
C_NONCRIT = '#e67e22'
C_OTHER = '#bdc3c7'
C_NODE = '#2c3e50'
C_NODE_OUT = '#95a5a6'
C_LOST = '#e74c3c'


def get_short(full_name):
    if full_name in SHORT_NAMES:
        return SHORT_NAMES[full_name]
    parts = full_name.split()
    return parts[-1] if len(parts) >= 2 else full_name


def build_network_with_sub(events, team):
    """Build network with substitution inheritance (11 players)."""
    name_sub_map = {}
    for e in events:
        if e.get('type', {}).get('name') == 'Substitution' and e.get('team', {}).get('name') == team:
            out_name = e.get('player', {}).get('name', '?')
            in_name = e.get('substitution', {}).get('replacement', {}).get('name', '?')
            target = out_name
            for k, v in list(name_sub_map.items()):
                if k == out_name:
                    target = v
                    break
            name_sub_map[in_name] = target

    G = nx.DiGraph()
    player_locs = {}
    player_full = {}

    for e in events:
        if e.get('type', {}).get('name') != 'Pass':
            continue
        if e.get('team', {}).get('name') != team:
            continue
        if e.get('pass', {}).get('outcome', {}).get('name', 'Complete') in ['Incomplete', 'Out', 'Unknown']:
            continue

        p_full = name_sub_map.get(e.get('player', {}).get('name', '?'), e.get('player', {}).get('name', '?'))
        r_full = name_sub_map.get(e.get('pass', {}).get('recipient', {}).get('name', '?'),
                                   e.get('pass', {}).get('recipient', {}).get('name', '?'))
        if p_full == '?' or r_full == '?':
            continue

        loc = e.get('location', [60, 40])
        end = e.get('pass', {}).get('end_location', [60, 40])
        s = builder.get_zone_id(loc[0], loc[1], 6, 4)
        t = builder.get_zone_id(end[0], end[1], 6, 4)
        src, tgt = f'{p_full}@Z{s}', f'{r_full}@Z{t}'

        if src != tgt:
            if G.has_edge(src, tgt):
                G[src][tgt]['weight'] += 1
            else:
                G.add_edge(src, tgt, weight=1)

        p_s, r_s = get_short(p_full), get_short(r_full)
        player_locs.setdefault(p_s, []).append(loc)
        player_locs.setdefault(r_s, []).append(end)
        player_full[p_s] = p_full
        player_full[r_s] = r_full

    player_pos = {p: (np.mean([l[0] for l in locs]), np.mean([l[1] for l in locs]))
                  for p, locs in player_locs.items()}

    return G, player_pos, player_full


def analyze_target_edges(G, target_full):
    """Analyse target player edges: critical vs non-critical."""
    scgc0 = calculate_metrics_directed(G)['P_inf_strong']
    critical = []
    noncritical = []

    for u, v in G.edges():
        if target_full not in u and target_full not in v:
            continue
        Gt = G.copy()
        Gt.remove_edge(u, v)
        drop = scgc0 - calculate_metrics_directed(Gt)['P_inf_strong']
        label = f'{get_short(u.split("@")[0])}\u2192{get_short(v.split("@")[0])}'
        if drop > 0:
            critical.append({'label': label, 'drop': drop, 'u': u, 'v': v})
        else:
            noncritical.append((u, v))

    critical.sort(key=lambda x: x['drop'], reverse=True)

    all_edges = [(u, v) for u, v in G.edges() if target_full in u or target_full in v]
    G_no = G.copy()
    G_no.remove_edges_from(all_edges)
    scgc_no = calculate_metrics_directed(G_no)['P_inf_strong']

    scc_before = max(nx.strongly_connected_components(G), key=len)
    scc_after = max(nx.strongly_connected_components(G_no), key=len) if G_no.number_of_edges() > 0 else set()

    return scgc0, scgc_no, critical, noncritical, all_edges, G_no, scc_before, scc_after


def draw_pitch(ax):
    """Minimal pitch lines."""
    ax.set_xlim(-5, 125)
    ax.set_ylim(-5, 85)
    ax.set_aspect('equal')
    ax.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_visible(False)
    lc, lw = '#d0d0d0', 0.6
    ax.plot([0, 120, 120, 0, 0], [0, 0, 80, 80, 0], color=lc, lw=lw)
    ax.plot([60, 60], [0, 80], color=lc, lw=lw * 0.7)
    ax.plot([0, 18, 18, 0], [18, 18, 62, 62], color=lc, lw=lw * 0.7)
    ax.plot([120, 102, 102, 120], [18, 18, 62, 62], color=lc, lw=lw * 0.7)
    ax.add_patch(plt.Circle((60, 40), 9.15, fill=False, color=lc, lw=lw * 0.7))
    ax.set_xticks([])
    ax.set_yticks([])


def draw_network_panel(ax, G_draw, player_pos, player_full, target_short,
                       critical_edges, noncritical_edges, show_target, scc_players=None,
                       players_before=None):
    """Draw one network panel."""
    target_full_name = player_full.get(target_short, target_short)

    for u, v in G_draw.edges():
        if target_full_name in u or target_full_name in v:
            continue
        p1 = get_short(u.split('@')[0])
        p2 = get_short(v.split('@')[0])
        if p1 in player_pos and p2 in player_pos:
            x1, y1 = player_pos[p1]
            x2, y2 = player_pos[p2]
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle='->', color=C_OTHER, lw=0.15, alpha=0.25,
                                        connectionstyle='arc3,rad=0.08'))

    if show_target:
        for u, v in noncritical_edges:
            p1 = get_short(u.split('@')[0])
            p2 = get_short(v.split('@')[0])
            if p1 in player_pos and p2 in player_pos:
                x1, y1 = player_pos[p1]
                x2, y2 = player_pos[p2]
                ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                            arrowprops=dict(arrowstyle='->', color=C_NONCRIT, lw=0.4, alpha=0.2,
                                            connectionstyle='arc3,rad=0.08', linestyle='--'))

        if critical_edges:
            mx = max([d['drop'] for d in critical_edges])
            for d in sorted(critical_edges, key=lambda x: x['drop']):
                p1 = get_short(d['u'].split('@')[0])
                p2 = get_short(d['v'].split('@')[0])
                if p1 in player_pos and p2 in player_pos:
                    x1, y1 = player_pos[p1]
                    x2, y2 = player_pos[p2]
                    lw = 0.8 + 2.5 * (d['drop'] / mx)
                    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                                arrowprops=dict(arrowstyle='->', color=C_CRIT, lw=lw, alpha=0.75,
                                                connectionstyle='arc3,rad=0.1'))

    for p, (x, y) in player_pos.items():
        full = player_full.get(p, p)
        dx, dy = LABEL_OFFSETS.get(p, DEFAULT_OFFSET)

        if p == target_short:
            if show_target:
                ax.plot(x, y, 'o', color=C_CRIT, ms=9, zorder=10,
                        markeredgecolor='white', markeredgewidth=1.2)
                ax.text(x + dx, y + dy, p, fontsize=7.5, ha='center', fontweight='bold',
                        color=C_CRIT, zorder=11)
            else:
                ax.plot(x, y, 'o', color='white', ms=9, zorder=10,
                        markeredgecolor=C_CRIT, markeredgewidth=1.5)
                ax.plot(x, y, 'x', color=C_CRIT, ms=6, zorder=11, markeredgewidth=1.5)
                ax.text(x + dx, y + dy, p, fontsize=7.5, ha='center',
                        color=C_CRIT, alpha=1.0, zorder=11)
        else:
            if scc_players is not None:
                in_scc = full in scc_players
                lost = players_before and (full in players_before) and (full not in scc_players)
                if lost:
                    ax.plot(x, y, 'o', color=C_LOST, ms=5.5, zorder=8,
                            markeredgecolor='white', markeredgewidth=0.6)
                    ax.text(x + dx, y + dy, p[:10], fontsize=6.5, ha='center',
                            color=C_LOST, fontweight='bold', zorder=9)
                else:
                    c = C_NODE if in_scc else C_NODE_OUT
                    ax.plot(x, y, 'o', color=c, ms=5.5, zorder=8,
                            markeredgecolor='white', markeredgewidth=0.6)
                    ax.text(x + dx, y + dy, p[:10], fontsize=6.5, ha='center',
                            color='#555', zorder=9)
            else:
                ax.plot(x, y, 'o', color=C_NODE, ms=5.5, zorder=8,
                        markeredgecolor='white', markeredgewidth=0.6)
                ax.text(x + dx, y + dy, p[:10], fontsize=6.5, ha='center',
                        color='#555', zorder=9)


def main():
    # Build file index
    file_idx = {}
    for ef in glob.glob(str(DATA_ROOT / '*/*_events.json')):
        file_idx[int(os.path.basename(ef).split('_')[0])] = ef

    with open(file_idx[MATCH_ID]) as f:
        events = json.load(f)

    opponent = ''
    for e in events:
        if e.get('type', {}).get('name') == 'Starting XI' and e.get('team', {}).get('name') != TEAM:
            opponent = e.get('team', {}).get('name', '')
            break

    G, player_pos, player_full = build_network_with_sub(events, TEAM)
    target_short = get_short(TARGET_PLAYER)

    scgc0, scgc_no, critical, noncritical, all_edges, G_no, scc_b, scc_a = \
        analyze_target_edges(G, TARGET_PLAYER)

    players_before = set(n.split('@')[0] for n in scc_b)
    players_after = set(n.split('@')[0] for n in scc_a)

    print(f"SCGC: {scgc0:.3f} -> {scgc_no:.3f} (-{(scgc0 - scgc_no) / scgc0 * 100:.1f}%)")
    print(f"{target_short}: {len(critical)} critical / {len(all_edges)} total edges")

    fig = plt.figure(figsize=(7.2, 6.5))
    gs_top = gridspec.GridSpec(1, 2, wspace=0.08, left=0.01, right=0.99, top=0.95, bottom=0.52)
    gs_bot = gridspec.GridSpec(1, 1, left=0.12, right=0.95, top=0.44, bottom=0.02)

    # (a) Full network
    ax_a = fig.add_subplot(gs_top[0, 0])
    draw_pitch(ax_a)
    draw_network_panel(ax_a, G, player_pos, player_full, target_short,
                       critical, noncritical, show_target=True)
    ax_a.text(0.5, 1.03, f'(a)  {TEAM} vs {opponent}',
              transform=ax_a.transAxes, fontsize=12, ha='center', fontweight='bold')
    ax_a.text(0.5, -0.03,
              f'SCGC = {scgc0:.3f}   |   {target_short}: {len(critical)} critical / {len(all_edges)} edges',
              transform=ax_a.transAxes, fontsize=9, ha='center', color='#666')

    # (b) After removal
    ax_b = fig.add_subplot(gs_top[0, 1])
    draw_pitch(ax_b)
    draw_network_panel(ax_b, G_no, player_pos, player_full, target_short,
                       [], [], show_target=False, scc_players=players_after,
                       players_before=players_before)
    ax_b.text(0.5, 1.03, f'(b)  After removing {target_short}\'s edges',
              transform=ax_b.transAxes, fontsize=12, ha='center', fontweight='bold')
    ax_b.text(0.5, -0.03,
              f'SCGC = {scgc_no:.3f}   |   \u0394SCGC = \u2212{(scgc0 - scgc_no) / scgc0 * 100:.0f}%',
              transform=ax_b.transAxes, fontsize=9, ha='center', color='#666')

    # (c) Bar chart
    ax_c = fig.add_subplot(gs_bot[0, 0])
    n_bars = len(critical)
    y_pos = np.arange(n_bars)
    drops = [d['drop'] for d in critical]
    labels = [d['label'] for d in critical]

    norm_drops = [d / max(drops) for d in drops]
    colors = [plt.cm.Reds(0.3 + 0.6 * nd) for nd in norm_drops]

    ax_c.barh(y_pos, drops, color=colors, edgecolor='white', linewidth=0.5, height=0.7)
    ax_c.set_yticks(y_pos)
    ax_c.set_yticklabels(labels, fontsize=8.5)
    ax_c.invert_yaxis()
    ax_c.set_xlabel('SCGC drop (single edge removal)', fontsize=12)
    ax_c.set_title(f'(c)  {target_short}\'s edges ranked by structural impact',
                   fontsize=12, fontweight='bold')

    for i, d in enumerate(drops):
        ax_c.text(d + 0.0005, i, f'{d:.4f}', va='center', fontsize=8, color='#555')

    cum = np.cumsum(drops)
    ax_c2 = ax_c.twiny()
    ax_c2.plot(cum, y_pos, 'o-', color='#2980b9', ms=3, lw=1, alpha=0.7)
    ax_c2.set_xlabel('Cumulative drop', fontsize=8.5, color='#2980b9')
    ax_c2.tick_params(axis='x', labelsize=8, colors='#2980b9')

    if n_bars >= 3:
        cum3 = cum[2]
        total_drop = sum(drops)
        ax_c.axhline(2.5, color='grey', lw=0.5, ls='--', alpha=0.5)
        ax_c.text(max(drops) * 0.85, 3.2,
                  f'Top 3 edges: {cum3 / total_drop * 100:.0f}% of total impact',
                  fontsize=8, color='#666', style='italic')
    ax_c.set_xlim(0, 0.077)
    ax_c.spines['top'].set_visible(False)
    ax_c.spines['right'].set_visible(False)

    out = OUTPUT_DIR / 'fig8.png'
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(str(out).replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close()


if __name__ == '__main__':
    main()
