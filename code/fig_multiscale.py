"""
fig_multiscale.py — Multi-scale passing network visualisation (Fig. 1)
=====================================================================
Generates a 4-panel figure showing a single match (Manchester City vs
Burnley, EPL 2023/24 Matchday 1) at four spatial coarse-graining scales:
  (a) L0  24x16 = 384 zones
  (b) L1  12x8  = 96  zones
  (c) L2  6x4   = 24  zones
  (d) L3  3x2   = 6   zones

Nodes are coloured by position (Goalkeeper=gold, Defender=navy,
Midfielder=steel blue, Forward=red), sized by degree. Directed edges
use arc arrows at L2/L3, width proportional to pass count.

Note: This script requires raw StatsBomb event data (JSON format).

Input:  StatsBomb event JSON (DATA_ROOT / '2_281_2023_24/3892755_events.json')
Output: figures/fig1_multiscale_directed.png, figures/fig1_multiscale_directed.pdf

Usage:
  python code/fig_multiscale.py

Last modified: 2026-03-27
"""

import matplotlib
matplotlib.use('Agg')

import sys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import FancyArrowPatch, Rectangle
from collections import defaultdict
from pathlib import Path

# ── Path setup ──
sys.path.insert(0, str(Path(__file__).resolve().parent / 'pipeline'))

from data_loader import (
    load_match_events, extract_passes, get_teams_from_events,
    build_substitution_map, apply_substitution_map
)
from network_builder import ZonalNetworkBuilder, SCALE_CONFIGS

# ── Config ──
# User must set this to the root directory of StatsBomb event data
DATA_ROOT = Path('/path/to/statsbomb/datos')
MATCH_FILE = DATA_ROOT / '2_281_2023_24/3892755_events.json'
OUTPUT_DIR = Path(__file__).resolve().parent.parent / 'figures'

FIELD_L, FIELD_W = 120.0, 80.0

# ── Global style — Arctic Academic (Times New Roman, deep-blue + rust-red) ──
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.titleweight': 'bold',
    'axes.unicode_minus': False,
    'figure.dpi': 150,
    'figure.facecolor': 'white',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
    'savefig.pad_inches': 0.1,
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Times New Roman',
    'mathtext.it': 'Times New Roman:italic',
    'mathtext.bf': 'Times New Roman:bold',
})

# Arctic Academic color scheme — no green, use blue/red tonal family
POS_COLORS = {
    'Goalkeeper': '#D4A03C',   # amber gold (warm accent)
    'Defender':   '#1B4F72',   # deep navy blue (Arctic primary)
    'Midfield':   '#5B8DB8',   # steel blue (Arctic mid-tone)
    'Forward':    '#C0392B',   # rust red (Arctic secondary)
    'Unknown':    '#8B9DAF',   # blue-grey
}

POS_MAP = {
    'Goalkeeper': 'Goalkeeper',
    'Right Back': 'Defender', 'Left Back': 'Defender',
    'Right Wing Back': 'Defender', 'Left Wing Back': 'Defender',
    'Center Back': 'Defender', 'Left Center Back': 'Defender', 'Right Center Back': 'Defender',
    'Right Defensive Midfield': 'Midfield', 'Left Defensive Midfield': 'Midfield',
    'Center Defensive Midfield': 'Midfield',
    'Right Midfield': 'Midfield', 'Left Midfield': 'Midfield', 'Center Midfield': 'Midfield',
    'Right Center Midfield': 'Midfield', 'Left Center Midfield': 'Midfield',
    'Right Attacking Midfield': 'Midfield', 'Left Attacking Midfield': 'Midfield',
    'Center Attacking Midfield': 'Midfield',
    'Right Wing': 'Forward', 'Left Wing': 'Forward',
    'Center Forward': 'Forward', 'Striker': 'Forward',
    'Right Center Forward': 'Forward', 'Left Center Forward': 'Forward',
    'Secondary Striker': 'Forward',
}

# Per-scale visual parameters
SCALE_VIZ = {
    'L0': {
        'node_base': 5,  'node_scale': 50,
        'edge_alpha_min': 0.04, 'edge_alpha_max': 0.35,
        'edge_lw_min': 0.1, 'edge_lw_max': 0.5,
        'node_lw': 0.15, 'jitter_frac': 0.38,
        'show_arrows': False, 'arc_rad': 0.12,
        'min_weight': 1,
        'zone_fill_alpha': 0.018,
    },
    'L1': {
        'node_base': 8,  'node_scale': 100,
        'edge_alpha_min': 0.06, 'edge_alpha_max': 0.45,
        'edge_lw_min': 0.15, 'edge_lw_max': 1.0,
        'node_lw': 0.3, 'jitter_frac': 0.35,
        'show_arrows': False, 'arc_rad': 0.15,
        'min_weight': 1,
        'zone_fill_alpha': 0.022,
    },
    'L2': {
        'node_base': 15, 'node_scale': 220,
        'edge_alpha_min': 0.10, 'edge_alpha_max': 0.65,
        'edge_lw_min': 0.3, 'edge_lw_max': 2.5,
        'node_lw': 0.5, 'jitter_frac': 0.32,
        'show_arrows': True, 'arc_rad': 0.15,
        'arrow_mutation': 15,
        'min_weight': 1,
        'zone_fill_alpha': 0.025,
    },
    'L3': {
        'node_base': 35, 'node_scale': 400,
        'edge_alpha_min': 0.10, 'edge_alpha_max': 0.70,
        'edge_lw_min': 0.4, 'edge_lw_max': 3.5,
        'node_lw': 0.7, 'jitter_frac': 0.35,
        'show_arrows': True, 'arc_rad': 0.15,
        'arrow_mutation': 18,
        'min_weight': 1,
        'zone_fill_alpha': 0.025,
    },
}

PANEL_LABELS = ['(a)', '(b)', '(c)', '(d)']


def get_player_positions(events, team_id):
    """Extract player position categories from Starting XI and event data."""
    positions = {}
    for event in events:
        if event.get('type', {}).get('name') == 'Starting XI' and event.get('team', {}).get('id') == team_id:
            lineup = event.get('tactics', {}).get('lineup', [])
            for p in lineup:
                pid = p.get('player', {}).get('id')
                pos_name = p.get('position', {}).get('name', 'Unknown')
                positions[pid] = POS_MAP.get(pos_name, 'Unknown')
    for event in events:
        if event.get('team', {}).get('id') == team_id:
            pid = event.get('player', {}).get('id')
            pos = event.get('position', {}).get('name', '')
            if pid and pid not in positions and pos:
                positions[pid] = POS_MAP.get(pos, 'Unknown')
    return positions


def draw_pitch(ax):
    """Draw a simplified football pitch outline."""
    color = '#888888'  # visible grey pitch lines
    lw = 0.8
    ax.plot([0, FIELD_L], [0, 0], color=color, lw=lw)
    ax.plot([0, FIELD_L], [FIELD_W, FIELD_W], color=color, lw=lw)
    ax.plot([0, 0], [0, FIELD_W], color=color, lw=lw)
    ax.plot([FIELD_L, FIELD_L], [0, FIELD_W], color=color, lw=lw)
    ax.plot([FIELD_L/2, FIELD_L/2], [0, FIELD_W], color=color, lw=lw*0.7, ls='--', alpha=0.5)
    circle = plt.Circle((FIELD_L/2, FIELD_W/2), 9.15, fill=False, color=color, lw=lw*0.7, alpha=0.5)
    ax.add_patch(circle)
    for x0 in [0, FIELD_L - 18]:
        ax.plot([x0, x0+18], [18, 18], color=color, lw=lw*0.6, alpha=0.45)
        ax.plot([x0, x0+18], [FIELD_W-18, FIELD_W-18], color=color, lw=lw*0.6, alpha=0.45)
        xv = 18 if x0 == 0 else FIELD_L - 18
        ax.plot([xv, xv], [18, FIELD_W-18], color=color, lw=lw*0.6, alpha=0.45)


def draw_zone_grid(ax, n_cols, n_rows, alpha=0.12, lw=0.2):
    """Thin grid lines to indicate zones."""
    color = '#AAAAAA'
    zw = FIELD_L / n_cols
    zh = FIELD_W / n_rows
    for i in range(1, n_cols):
        ax.plot([i*zw, i*zw], [0, FIELD_W], color=color, lw=lw, ls=':', alpha=alpha)
    for j in range(1, n_rows):
        ax.plot([0, FIELD_L], [j*zh, j*zh], color=color, lw=lw, ls=':', alpha=alpha)


def draw_zone_fills(ax, n_cols, n_rows, alpha=0.04, occupied_zones=None):
    """Checkerboard fills. If occupied_zones given, only fill those."""
    zw = FIELD_L / n_cols
    zh = FIELD_W / n_rows
    for i in range(n_cols):
        for j in range(n_rows):
            zone_id = j * n_cols + i
            if occupied_zones is not None and zone_id not in occupied_zones:
                continue
            if (i + j) % 2 == 0:
                rect = Rectangle((i * zw, j * zh), zw, zh,
                                  facecolor='#000000', alpha=alpha,
                                  edgecolor='none', zorder=0)
                ax.add_patch(rect)


def compute_node_positions(G, n_cols, n_rows, jitter_frac=0.35):
    """Compute node positions within their assigned zones with jitter."""
    builder = ZonalNetworkBuilder()
    zw = FIELD_L / n_cols
    zh = FIELD_W / n_rows

    zone_nodes = defaultdict(list)
    for node in G.nodes():
        z = node[1]
        zone_nodes[z].append(node)

    pos = {}
    rng = np.random.RandomState(42)

    for z, nodes in zone_nodes.items():
        cx, cy = builder.get_zone_center(z, n_cols, n_rows)
        n = len(nodes)

        if n == 1:
            pos[nodes[0]] = (cx, cy)
        elif n <= 6:
            radius = min(zw, zh) * jitter_frac * 0.5
            angles = np.linspace(0, 2*np.pi, n, endpoint=False) + rng.uniform(0, np.pi/4)
            for i, node in enumerate(nodes):
                pos[node] = (cx + radius * np.cos(angles[i]),
                             cy + radius * np.sin(angles[i]))
        else:
            sub_u = nx.Graph(G.subgraph(nodes))
            local_pos = nx.spring_layout(sub_u, k=0.5, iterations=50, seed=42)
            if local_pos:
                lx = np.array([p[0] for p in local_pos.values()])
                ly = np.array([p[1] for p in local_pos.values()])
                rx = (max(lx) - min(lx)) if max(lx) != min(lx) else 1
                ry = (max(ly) - min(ly)) if max(ly) != min(ly) else 1
                half_w = zw * jitter_frac
                half_h = zh * jitter_frac
                for node in nodes:
                    nx_val = (local_pos[node][0] - np.mean(lx)) / rx * half_w
                    ny_val = (local_pos[node][1] - np.mean(ly)) / ry * half_h
                    pos[node] = (cx + nx_val, cy + ny_val)

    return pos


def aggregate_directed_edges(G_multi):
    """Aggregate multi-edges into weighted directed edges."""
    edge_weights = defaultdict(int)
    for u, v in G_multi.edges():
        edge_weights[(u, v)] += 1
    return dict(edge_weights)


def draw_directed_edge(ax, pos_u, pos_v, weight, max_weight, params, is_reciprocal):
    """Draw a single directed edge with weight-dependent styling."""
    w_norm = weight / max_weight
    alpha = params['edge_alpha_min'] + (params['edge_alpha_max'] - params['edge_alpha_min']) * w_norm
    lw = params['edge_lw_min'] + (params['edge_lw_max'] - params['edge_lw_min']) * w_norm

    # Moderate curve for reciprocal, near-straight otherwise
    rad = params['arc_rad'] if is_reciprocal else 0.03
    style = f"arc3,rad={rad}"

    # Blue-grey edge tones (Arctic Academic style)
    r_val = int(0x3A + (0xA8 - 0x3A) * (1 - w_norm))
    g_val = int(0x50 + (0xB8 - 0x50) * (1 - w_norm))
    b_val = int(0x6A + (0xCC - 0x6A) * (1 - w_norm))
    edge_color = f'#{r_val:02x}{g_val:02x}{b_val:02x}'

    if params['show_arrows']:
        ms = params['arrow_mutation']
        arrow = FancyArrowPatch(
            posA=pos_u, posB=pos_v,
            connectionstyle=style,
            arrowstyle='-|>',
            mutation_scale=ms,
            shrinkA=3, shrinkB=3,
            color=edge_color, lw=lw, alpha=alpha, zorder=1,
        )
        arrow.set_capstyle('round')
        arrow.set_joinstyle('round')
    else:
        arrow = FancyArrowPatch(
            posA=pos_u, posB=pos_v,
            connectionstyle=style,
            arrowstyle='-',
            color=edge_color, lw=lw, alpha=alpha, zorder=1,
        )
        arrow.set_capstyle('round')
    ax.add_patch(arrow)


def plot_directed_network(ax, G_multi, scale, n_cols, n_rows, positions, panel_label):
    """Draw a complete directed network panel for one scale."""
    params = SCALE_VIZ[scale]

    draw_pitch(ax)

    # Subtle zone grid lines (no checkerboard — cleaner Arctic Academic look)
    draw_zone_grid(ax, n_cols, n_rows, alpha=0.15, lw=0.3)

    pos = compute_node_positions(G_multi, n_cols, n_rows, params['jitter_frac'])

    # Aggregate and filter
    edge_weights = aggregate_directed_edges(G_multi)
    min_w = params['min_weight']
    filtered = {k: v for k, v in edge_weights.items() if v >= min_w}
    max_ew = max(filtered.values()) if filtered else 1

    filt_set = set(filtered.keys())
    reciprocal = {(u, v) for (u, v) in filt_set if (v, u) in filt_set}

    # Draw edges
    for (u, v), w in filtered.items():
        if u not in pos or v not in pos:
            continue
        draw_directed_edge(ax, pos[u], pos[v], w, max_ew, params, (u, v) in reciprocal)

    # Draw nodes
    degrees = {}
    for node in G_multi.nodes():
        degrees[node] = G_multi.in_degree(node) + G_multi.out_degree(node)
    max_deg = max(degrees.values()) if degrees else 1

    for node in G_multi.nodes():
        if node not in pos:
            continue
        x, y = pos[node]
        pid = node[0]
        pos_cat = positions.get(pid, 'Unknown')
        color = POS_COLORS.get(pos_cat, '#999999')
        deg = degrees.get(node, 0)
        size = params['node_base'] + params['node_scale'] * (deg / max_deg)
        ax.scatter(x, y, s=size, c=color, edgecolors='white',
                   linewidths=params['node_lw'] * 0.8, zorder=3, alpha=0.92)

    # Labels
    n_nodes = G_multi.number_of_nodes()
    n_edges_total = len(edge_weights)
    n_zones = n_cols * n_rows
    ax.set_title(f'{panel_label}  {scale}  ({n_cols}$\\times${n_rows} = {n_zones} zones)',
                 fontsize=17, fontweight='bold', color='#222222', pad=8, loc='left')
    ax.text(2, -5, f'|V| = {n_nodes}    |E| = {n_edges_total}',
            fontsize=14, color='#666666', va='top')


def main():
    print("Loading match data...")
    events = load_match_events(str(MATCH_FILE))
    _, away_team = get_teams_from_events(events)

    team_id = away_team['id']
    team_name = away_team['name']
    print(f"Team: {team_name} (ID: {team_id})")

    sub_map = build_substitution_map(events, team_id=team_id)
    passes = extract_passes(events, team_id=team_id)
    passes = apply_substitution_map(passes, sub_map)
    print(f"Passes: {len(passes)} (after substitution mapping)")
    print(f"Substitution map: {len(sub_map)} substitutions merged")

    positions = get_player_positions(events, team_id)
    for sub_id, starter_id in sub_map.items():
        if starter_id in positions and sub_id not in positions:
            positions[sub_id] = positions[starter_id]

    builder = ZonalNetworkBuilder()

    networks = {}
    for scale in ['L0', 'L1', 'L2', 'L3']:
        G = builder.build_network(passes, scale=scale)
        networks[scale] = G
        n_unique_players = len(set(node[0] for node in G.nodes()))
        print(f"  {scale}: {G.number_of_nodes()} nodes, {G.number_of_edges()} directed edges, "
              f"{n_unique_players} unique players")

    print("\nGenerating directed network figure...")
    fig, axes = plt.subplots(2, 2, figsize=(11, 9.5), facecolor='white')

    for i, scale in enumerate(['L0', 'L1', 'L2', 'L3']):
        ax = axes[i // 2][i % 2]
        ax.set_facecolor('white')
        n_cols, n_rows = SCALE_CONFIGS[scale]
        plot_directed_network(ax, networks[scale], scale, n_cols, n_rows,
                              positions, PANEL_LABELS[i])
        ax.set_xlim(-3, FIELD_L + 3)
        ax.set_ylim(-8, FIELD_W + 3)
        ax.set_aspect('equal')
        ax.axis('off')

    legend_handles = [
        mlines.Line2D([], [], marker='o', color='w', markerfacecolor=POS_COLORS[p],
                       markersize=10, label=p, linewidth=0,
                       markeredgecolor='white', markeredgewidth=0.5)
        for p in ['Goalkeeper', 'Defender', 'Midfield', 'Forward']
    ]
    fig.legend(handles=legend_handles, loc='lower center',
               fontsize=14, facecolor='white', edgecolor='#CCCCCC',
               labelcolor='#333333', ncol=4, bbox_to_anchor=(0.5, 0.005),
               framealpha=0.95, columnspacing=2.0, handletextpad=0.5)

    fig.tight_layout(rect=[0, 0.04, 1, 1.0], h_pad=3.5)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for fmt in ['png', 'pdf']:
        path = OUTPUT_DIR / f'fig1_multiscale_directed.{fmt}'
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {path}")
    plt.close(fig)

    print("\nDone!")


if __name__ == '__main__':
    main()
