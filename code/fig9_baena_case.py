"""
fig9_baena_case.py  Case study of Baena's structural contribution (Fig. 9)
========================================================================
Builds Villarreal's L2 passing network against Girona (2024/25 La Liga,
match 3946543) with substitutes mapped to the players they replaced, and
removes Baena's edges one at a time and all together. (a) Full network with
his critical and redundant edges, (b) network after removing all his edges,
(c) critical edges ranked by single-edge SCGC decrease with the cumulative curve.

Input  DATA_ROOT/*/3946543_events.json (StatsBomb event JSON)
Output  figures/fig9.png, figures/fig9.pdf

Usage
  DATA_ROOT=/path/to/statsbomb/datos python code/fig9_baena_case.py

Last modified 2026-10-27
"""

import matplotlib
matplotlib.use('Agg')

import sys
import json
import glob
import os
from pathlib import Path

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / 'code' / 'pipeline'))

from network_builder import ZonalNetworkBuilder
from robustness import calculate_metrics_directed

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
})

# Root directory of the StatsBomb event data (set the DATA_ROOT environment variable)
DATA_ROOT = Path(os.environ.get('DATA_ROOT', '/path/to/statsbomb/datos'))
OUTPUT_DIR = BASE_DIR / 'figures'

builder = ZonalNetworkBuilder()

MATCH_ID = 3946543  # Villarreal vs Girona, 2024/25
TEAM = 'Villarreal'
TARGET_PLAYER = 'Alejandro Baena Rodríguez'

# Values printed in the article (Section 3.5 and the Fig. 9 caption) and in Fig. 9c
PUBLISHED = {
    'scgc_before': 0.533,      # 3 decimals
    'scgc_after': 0.326,       # 3 decimals
    'decrease_pct': 39,        # integer percent
    'top3_share_pct': 47,      # integer percent
    'n_critical': 14,
    'n_edges': 56,
    'n_redundant': 42,
    'drops': [0.0667, 0.0667, 0.0444, 0.0370, 0.0296, 0.0296, 0.0222,
              0.0222, 0.0148, 0.0148, 0.0074, 0.0074, 0.0074, 0.0074],  # 4 decimals
}

# Custom short names
SHORT_NAMES = {
    'Alejandro Baena Rodríguez': 'Baena',
}

# Label offsets chosen from the plotted coordinates so that the labels of
# Baena, Muñoz, Barry and Gueye do not overlap. dy > 0 places a label above
# the node in the displayed figure (the sign is flipped when drawing because
# the pitch y axis points down), dx > 0 places it to the right.
LABEL_OFFSETS = {
    'Baena': (6, 6),      # upper right, clear of Muñoz (left) and Barry and Gueye (lower right)
    'Muñoz': (-8, -4),    # lower left, open area
    'Gueye': (0, -4),     # below
}
DEFAULT_OFFSET = (0, 4)


def get_short(full_name):
    if full_name in SHORT_NAMES:
        return SHORT_NAMES[full_name]
    parts = full_name.split()
    return parts[-1] if len(parts) >= 2 else full_name


def build_network_with_sub(events, team):
    """Build the L2 player-zone network with substitution inheritance (11 players)."""
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
    """Split the target player's edges into critical and redundant ones."""
    scgc0 = calculate_metrics_directed(G)['P_inf_strong']
    critical = []
    noncritical = []

    for u, v in G.edges():
        if target_full not in u and target_full not in v:
            continue
        Gt = G.copy()
        Gt.remove_edge(u, v)
        drop = scgc0 - calculate_metrics_directed(Gt)['P_inf_strong']
        label = f'{get_short(u.split("@")[0])}→{get_short(v.split("@")[0])}'
        if drop > 0:
            critical.append({'label': label, 'drop': drop, 'u': u, 'v': v})
        else:
            noncritical.append((u, v))

    critical.sort(key=lambda x: x['drop'], reverse=True)

    # After removing all of the target player's edges
    all_edges = [(u, v) for u, v in G.edges() if target_full in u or target_full in v]
    G_no = G.copy()
    G_no.remove_edges_from(all_edges)
    scgc_no = calculate_metrics_directed(G_no)['P_inf_strong']

    scc_before = max(nx.strongly_connected_components(G), key=len)
    scc_after = max(nx.strongly_connected_components(G_no), key=len) if G_no.number_of_edges() > 0 else set()

    return scgc0, scgc_no, critical, noncritical, all_edges, G_no, scc_before, scc_after


def draw_pitch(ax):
    """StatsBomb pitch from mplsoccer (120 x 80, y axis pointing down, so the
    attacking team's left flank, low y, appears at the top)."""
    from mplsoccer import Pitch
    pitch = Pitch(pitch_type='statsbomb', pitch_color='white',
                  line_color='#111111', linewidth=0.7, line_alpha=0.4, line_zorder=1)
    pitch.draw(ax=ax)


# Colours
C_CRIT = '#c0392b'
C_NONCRIT = '#e67e22'
C_OTHER = '#bdc3c7'
C_NODE = '#2c3e50'
C_NODE_OUT = '#95a5a6'
C_LOST = '#e74c3c'


def draw_network_panel(ax, G_draw, player_pos, player_full, target_short,
                       critical_edges, noncritical_edges, show_target, scc_players=None,
                       players_before=None):
    """Draw one network panel."""
    target_full_name = player_full.get(target_short, target_short)

    # Edges between other players
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
        # Redundant edges (dashed orange)
        for u, v in noncritical_edges:
            p1 = get_short(u.split('@')[0])
            p2 = get_short(v.split('@')[0])
            if p1 in player_pos and p2 in player_pos:
                x1, y1 = player_pos[p1]
                x2, y2 = player_pos[p2]
                ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                            arrowprops=dict(arrowstyle='->', color=C_NONCRIT, lw=0.4, alpha=0.2,
                                            connectionstyle='arc3,rad=0.08', linestyle='--'))

        # Critical edges (red, width proportional to the SCGC decrease)
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

    # Player nodes
    for p, (x, y) in player_pos.items():
        full = player_full.get(p, p)
        dx, dy = LABEL_OFFSETS.get(p, DEFAULT_OFFSET)
        dy = -dy  # the pitch y axis points down, so flip to keep the displayed direction

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


def check_published_values(scgc0, scgc_no, critical, noncritical, all_edges):
    """Acceptance check against the published values, within half a unit of the printed precision."""
    drops = [d['drop'] for d in critical]
    decrease = (scgc0 - scgc_no) / scgc0 * 100
    top3 = sum(drops[:3]) / sum(drops) * 100
    checks = [
        ('SCGC before removal', scgc0, PUBLISHED['scgc_before'], 0.0005, '.3f'),
        ('SCGC after removal', scgc_no, PUBLISHED['scgc_after'], 0.0005, '.3f'),
        ('SCGC decrease (%)', decrease, PUBLISHED['decrease_pct'], 0.5, '.1f'),
        ('top three edges share (%)', top3, PUBLISHED['top3_share_pct'], 0.5, '.1f'),
        ('critical edges', len(critical), PUBLISHED['n_critical'], 0, 'd'),
        ('directed edges of Baena', len(all_edges), PUBLISHED['n_edges'], 0, 'd'),
        ('redundant edges', len(noncritical), PUBLISHED['n_redundant'], 0, 'd'),
    ]
    print("\nCheck  values against the article and Fig. 9")
    ok_all = True
    for name, got, exp, tol, fmt in checks:
        ok = abs(got - exp) <= tol
        ok_all &= ok
        print(f"  {name:<27} {got:{fmt}} vs published {exp}  {'PASS' if ok else 'FAIL'}")
    ok_drops = len(drops) == len(PUBLISHED['drops'])
    for i, (got, exp) in enumerate(zip(drops, PUBLISHED['drops'])):
        if abs(got - exp) > 0.00005:
            ok_drops = False
            print(f"  edge {i + 1} drop {got:.4f} vs published {exp:.4f}  FAIL")
    ok_all &= ok_drops
    print(f"  {'single-edge drops (c)':<27} {len(drops)} values vs published "
          f"{len(PUBLISHED['drops'])}  {'PASS' if ok_drops else 'FAIL'}")
    if not ok_all:
        print("CHECK FAILED, the values differ from the published article.")
        sys.exit(1)
    print("CHECK ALL PASS")


def main():
    # Locate the match file
    file_idx = {}
    for ef in glob.glob(str(DATA_ROOT / '*/*_events.json')):
        file_idx[int(os.path.basename(ef).split('_')[0])] = ef
    if MATCH_ID not in file_idx:
        sys.exit(f"{MATCH_ID}_events.json not found under DATA_ROOT ({DATA_ROOT}), "
                 "set the DATA_ROOT environment variable")
    with open(file_idx[MATCH_ID]) as f:
        events = json.load(f)

    # Opponent name
    opponent = ''
    for e in events:
        if e.get('type', {}).get('name') == 'Starting XI' and e.get('team', {}).get('name') != TEAM:
            opponent = e.get('team', {}).get('name', '')
            break

    # Build the network
    G, player_pos, player_full = build_network_with_sub(events, TEAM)
    target_short = get_short(TARGET_PLAYER)

    # Edge analysis
    scgc0, scgc_no, critical, noncritical, all_edges, G_no, scc_b, scc_a = \
        analyze_target_edges(G, TARGET_PLAYER)

    players_before = set(n.split('@')[0] for n in scc_b)
    players_after = set(n.split('@')[0] for n in scc_a)

    print(f"SCGC: {scgc0:.3f} -> {scgc_no:.3f} (-{(scgc0 - scgc_no) / scgc0 * 100:.1f}%)")
    print(f"{target_short}: {len(critical)} critical / {len(all_edges)} total edges")
    print(f"Players: {len(player_pos)}")

    check_published_values(scgc0, scgc_no, critical, noncritical, all_edges)

    # Figure
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
              f'SCGC = {scgc_no:.3f}   |   ΔSCGC = −{(scgc0 - scgc_no) / scgc0 * 100:.0f}%',
              transform=ax_b.transAxes, fontsize=9, ha='center', color='#666')

    # (c) Horizontal bar chart
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

    # Cumulative curve
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

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / 'fig9.png'
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(out.with_suffix('.pdf'), bbox_inches='tight')
    print(f"Saved: {out.relative_to(BASE_DIR)}, {out.with_suffix('.pdf').relative_to(BASE_DIR)}")
    plt.close()


if __name__ == '__main__':
    main()
