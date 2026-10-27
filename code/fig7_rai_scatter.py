"""
fig7_rai_scatter.py  Robustness landscape of soccer and synthetic networks at L2 (Fig. 7)
========================================================================================
Places the 48 teams (mean thresholds over their L2 matches with SCGC ratio >= 0.3
and both thresholds positive) in the random-failure versus targeted-attack plane,
with the soccer mean, the 1 sigma ellipse of the teams and the four synthetic
reference models (ER, BA, WS, RGG, 100 realisations each, degenerate realisations
excluded). Thirteen teams are labelled with fixed offsets. The legend values and
the team RAI statistics quoted in the article are checked, and the script exits
with status 1 on a mismatch.

Input  data/match_summary_mc100_guarded.csv, data/synthetic_mc100_guarded.csv
       (the published analysis, 100 Monte Carlo trials at all scales, restricted
       threshold estimator)
Output  figures/fig7.png, figures/fig7.pdf

Usage
  python code/fig7_rai_scatter.py

Last modified 2026-10-27
"""
import sys
from pathlib import Path

import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Patch
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
import matplotlib.transforms as transforms

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
OUT = BASE_DIR / 'figures'

# Published values (Fig. 7 legend and the Section 3.4 text)
PUBLISHED = {
    'n_epl': 24, 'n_liga': 24,
    'rai_soccer': 0.31, 'rai_team_sd': 0.05,
    'rai_team_min': 0.221, 'rai_team_max': 0.437,
    'rai_ER': 0.25, 'rai_BA': 0.37, 'rai_WS': 0.22, 'rai_RGG': 0.02,
}
DECIMALS = {'rai_soccer': 2, 'rai_team_sd': 2, 'rai_team_min': 3, 'rai_team_max': 3,
            'rai_ER': 2, 'rai_BA': 2, 'rai_WS': 2, 'rai_RGG': 2}

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
    'mathtext.fontset': 'cm', 'xtick.labelsize': 9, 'ytick.labelsize': 9,
})

# ---------------- data ----------------
# Team means over the L2 matches with SCGC ratio >= 0.3 and both thresholds positive
ms = pd.read_csv(DATA_DIR / 'match_summary_mc100_guarded.csv')
l2 = ms[(ms['Scale']=='L2') & (ms['SCGC_Ratio']>=0.3)
        & (ms['pc_Error_Strong']>0) & (ms['pc_Attack_Strong']>0)]
ta = l2.groupby(['Team','League']).agg(
    {'pc_Error_Strong':'mean','pc_Attack_Strong':'mean'}).reset_index()
ta['RAI'] = (ta['pc_Error_Strong'] - ta['pc_Attack_Strong']) / ta['pc_Error_Strong']
ta['ShortName'] = ta['Team'].str.replace('Manchester ','Man ') \
    .str.replace('Sheffield ','Sheff ') \
    .str.replace('Brighton & Hove Albion','Brighton') \
    .str.replace('Wolverhampton Wanderers','Wolves') \
    .str.replace('Nottingham Forest','Nott. Forest') \
    .str.replace('AFC Bournemouth','Bournemouth') \
    .str.replace('Deportivo Alavés','Alavés') \
    .str.replace('Real Valladolid','Valladolid') \
    .str.replace('Atlético Madrid','Atlético') \
    .str.replace('Rayo Vallecano','Rayo')

# Synthetic reference networks, degenerate realisations (a threshold of zero) excluded
syn = pd.read_csv(DATA_DIR / 'synthetic_mc100_guarded.csv')
syn_rename = {'fcA_guarded':'pc_Attack_Strong','fcE_guarded':'pc_Error_Strong'}
syn.rename(columns={k:v for k,v in syn_rename.items() if k in syn.columns}, inplace=True)
syn = syn[(syn['pc_Error_Strong']>0) & (syn['pc_Attack_Strong']>0)]

def confidence_ellipse(x, y, ax, n_std=1.0, **kwargs):
    cov = np.cov(x, y)
    p = cov[0,1]/np.sqrt(cov[0,0]*cov[1,1]) if cov[0,0]*cov[1,1]>0 else 0
    ellipse = Ellipse((0,0), width=np.sqrt(1+p)*2, height=np.sqrt(1-p)*2, **kwargs)
    transf = transforms.Affine2D().rotate_deg(45) \
        .scale(np.sqrt(cov[0,0])*n_std, np.sqrt(cov[1,1])*n_std) \
        .translate(np.mean(x), np.mean(y))
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

ANNOTATE = ['Manchester City', 'Barcelona', 'Real Madrid', 'Liverpool', 'Chelsea',
            'Brighton & Hove Albion', 'Arsenal', 'Girona', 'Sheffield United',
            'Luton Town', 'Nottingham Forest', 'Granada', 'Espanyol']
# Manual label offsets in points relative to the data point, text anchored at its
# lower left corner. Teams listed here skip the automatic placement below; a team
# in ANNOTATE without an entry here is placed automatically.
MANUAL = {
    'Manchester City':        (7.9, 5.3),
    'Barcelona':              (7.9, -2.6),
    'Real Madrid':            (-54.2, -6.3),
    'Liverpool':              (22.3, -1.8),
    'Chelsea':                (9.9, -13.9),
    'Brighton & Hove Albion': (-40.5, 3.1),
    'Arsenal':                (-42.9, -5.9),
    'Girona':                 (7.9, -17.9),
    'Sheffield United':       (-27, -19.5),
    'Luton Town':             (-12.9, -15.8),
    'Nottingham Forest':      (-70, -8.9),
    'Granada':                (9.9, -8.9),
    'Espanyol':               (-12.9, -11.3),
}


# ---------------- encodings ----------------
C_EPL, C_LIGA = '#1B4F72', '#C0392B'
C_MODEL_LINE, C_MODEL_EDGE = '#7F7F7F', '#4D4D4D'
PLUM = '#6C3483'
LEG_LS = {'ER': (0,(2.2,1.2)), 'BA': (0,(1.6,0.9,0.7,0.9)), 'WS': '-', 'RGG': (0,(0.7,0.9))}
SYN = {
    'ER':  dict(marker='s', ls=(0,(6,3)),           lw=1.2),   # long dash
    'BA':  dict(marker='^', ls=(0,(6,2,1.5,2)),     lw=1.3),   # dash-dot
    'WS':  dict(marker='D', ls='-',                 lw=1.1),   # solid
    'RGG': dict(marker='v', ls=(0,(1.6,2.0)),       lw=1.1),   # dotted (coarser)
}
S_TEAM, S_MODEL, S_STAR = 12, 65, 130
LABEL_FS = 7

# ---------------- plot ----------------
fig, ax = plt.subplots(figsize=(4.6, 3.95))
diag, = ax.plot([0, 0.9], [0, 0.9], color='#A6A6A6', ls=':', lw=0.7, zorder=1)

handles, labels = [], []
team_xy = {}
for league, color, marker in [('Premier League', C_EPL, 'o'), ('La Liga', C_LIGA, 's')]:
    d = ta[ta['League']==league]
    ax.scatter(d['pc_Error_Strong'], d['pc_Attack_Strong'], c=color, marker=marker,
               s=S_TEAM, alpha=0.95, edgecolors='#1B2631', linewidths=0.5, zorder=5)
    handles.append(Line2D([], [], marker=marker, color='none', markerfacecolor=color,
                          markeredgecolor='#1B2631', markeredgewidth=0.5, markersize=4))
    labels.append(f'{league} (n={len(d)})')
for _, r in ta.iterrows():
    team_xy[r.Team] = (r.pc_Error_Strong, r.pc_Attack_Strong, r.League, r.ShortName)

fb_me = ta['pc_Error_Strong'].mean(); fb_ma = ta['pc_Attack_Strong'].mean()
fb_rai = (fb_me-fb_ma)/fb_me
ax.scatter(fb_me, fb_ma, c=PLUM, marker='*', s=S_STAR, edgecolors='white', linewidths=1.0, zorder=11)
handles.append(Line2D([], [], marker='*', color='none', markerfacecolor=PLUM,
                      markeredgecolor='white', markeredgewidth=0.6, markersize=10))
labels.append(f'Soccer mean (RAI={fb_rai:.2f})')
# 1 sigma ellipse of the teams, fill below the points and outline above them
confidence_ellipse(ta['pc_Error_Strong'], ta['pc_Attack_Strong'], ax, n_std=1.0,
                   facecolor=PLUM, alpha=0.35, edgecolor='none', zorder=4)
soc_edge = confidence_ellipse(ta['pc_Error_Strong'], ta['pc_Attack_Strong'], ax, n_std=1.0,
                   facecolor='none', edgecolor=PLUM, linewidth=1.0, zorder=6)
handles.append(Patch(facecolor=PLUM, alpha=0.35, edgecolor=PLUM, linewidth=1.0))
labels.append(r'Soccer $1\sigma$ (48 teams)')

model_patches, model_xy, model_rai = [], {}, {}
for model, cfg in SYN.items():
    d = syn[syn['Model']==model]
    pt = confidence_ellipse(d['pc_Error_Strong'], d['pc_Attack_Strong'], ax, n_std=1.0,
                            facecolor='none', edgecolor=C_MODEL_LINE, linewidth=cfg['lw'],
                            linestyle=cfg['ls'], zorder=3)
    model_patches.append(pt)
    me, ma = d['pc_Error_Strong'].mean(), d['pc_Attack_Strong'].mean()
    model_xy[model] = (me, ma)
    rai = (me-ma)/me if me>0 else 0
    model_rai[model] = rai
    ax.scatter(me, ma, c='white', marker=cfg['marker'], s=S_MODEL,
               edgecolors=C_MODEL_EDGE, linewidths=1.0, zorder=9)
    h_line = Line2D([], [], color=C_MODEL_LINE, linestyle=LEG_LS[model], linewidth=cfg['lw'])
    h_mark = Line2D([], [], marker=cfg['marker'], color='none', markerfacecolor='white',
                    markeredgecolor=C_MODEL_EDGE, markeredgewidth=1.0, markersize=6.5)
    handles.append((h_line, h_mark)); labels.append(f'{model} (RAI={rai:.2f})')

ax.set_xlabel(r'$f_{c,\mathrm{L2}}^{Error}$', fontsize=13)
ax.set_ylabel(r'$f_{c,\mathrm{L2}}^{Attack}$', fontsize=13)
ax.set_xlim(0.05, 0.86); ax.set_ylim(0.05, 0.69)
ax.grid(False)
leg1 = ax.legend(handles[:4], labels[:4], loc='upper left', fontsize=7, framealpha=0.95,
                 fancybox=False, edgecolor='#BBBBBB', handlelength=2.4, labelspacing=0.45, borderpad=0.5)
leg1.get_frame().set_linewidth(0.6)
ax.add_artist(leg1)
leg2 = ax.legend(handles[4:], labels[4:], loc='lower right', fontsize=7, framealpha=0.95,
                 fancybox=False, edgecolor='#BBBBBB', handler_map={tuple: HandlerTuple(ndivide=None, pad=0.5)},
                 handlelength=2.4, labelspacing=0.45, borderpad=0.5)
leg2.get_frame().set_linewidth(0.6)
plt.tight_layout()

# ---------------- label placement ----------------
# Teams with a manual offset are drawn at that offset. Any other team in ANNOTATE
# is placed automatically at the first of 16 directions times 8 distances whose
# text box overlaps no marker, ellipse outline, diagonal, legend or other label and
# whose leader line crosses no other marker or leader line.
fig.canvas.draw()
ren = fig.canvas.get_renderer()
PX = fig.dpi / 72.0                      # pixels per point
def to_px(x, y): return ax.transData.transform((x, y))
# obstacles as circles (cx, cy, r) in pixels
obst = []
r_team = (np.sqrt(S_TEAM/np.pi) + 1.2) * PX
for t,(x,y,_,_) in team_xy.items(): obst.append((*to_px(x,y), r_team))
r_model = (np.sqrt(S_MODEL/np.pi) + 1.5) * PX
for m,(x,y) in model_xy.items(): obst.append((*to_px(x,y), r_model))
obst.append((*to_px(fb_me, fb_ma), (np.sqrt(S_STAR/np.pi) + 2.5) * PX))
# sampled lines, ellipse outlines and the diagonal
for pt in model_patches + [soc_edge]:
    path = pt.get_path().transformed(pt.get_transform())
    for v in path.interpolated(200).vertices: obst.append((v[0], v[1], 1.6*PX))
for x in np.linspace(0.03, 0.69, 300): obst.append((*to_px(x, x), 1.2*PX))
# rectangles, the two legends and the axes limits
rects = [tuple(leg1.get_window_extent(ren).padded(2*PX).extents), tuple(leg2.get_window_extent(ren).padded(2*PX).extents)]
axbb = ax.get_window_extent(ren)

def text_size(s):
    w, h, d = ren.get_text_width_height_descent(s, matplotlib.font_manager.FontProperties(
        family='serif', weight='bold', size=LABEL_FS), ismath=False)
    return w, h + d
N_PTS = len(team_xy) + len(model_xy) + 1      # obst[:N_PTS] are markers; the rest are line samples
def box_hits_circles(bb, points_only=False):
    x0,y0,x1,y1 = bb
    for cx,cy,r in (obst[:N_PTS] if points_only else obst):
        nx, ny = min(max(cx,x0),x1), min(max(cy,y0),y1)
        if (cx-nx)**2 + (cy-ny)**2 < r*r: return True
    return False
def box_hits_rects(bb, others):
    x0,y0,x1,y1 = bb
    for rc in others:
        if not (x1 < rc[0] or x0 > rc[2] or y1 < rc[1] or y0 > rc[3]): return True
    return False
def seg_hits_points(p, q, skip_xy):
    for cx,cy,r in obst[:len(team_xy)+len(model_xy)+1]:
        if abs(cx-skip_xy[0])<1e-6 and abs(cy-skip_xy[1])<1e-6: continue
        for t in np.linspace(0.08, 0.92, 12):
            sx, sy = p[0]+(q[0]-p[0])*t, p[1]+(q[1]-p[1])*t
            if (cx-sx)**2+(cy-sy)**2 < (r*0.9)**2: return True
    return False

def segs_cross(p1, p2, q1, q2):
    def orient(a, b, c): return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
    d1, d2 = orient(q1, q2, p1), orient(q1, q2, p2)
    d3, d4 = orient(p1, p2, q1), orient(p1, p2, q2)
    return (d1*d2 < 0) and (d3*d4 < 0)

placed, leaders, unresolved = [], [], []
cx0, cy0 = to_px(fb_me, fb_ma)
# hardest first, ordered by local crowding (number of team points within 18 pt)
def crowd(team):
    px, py = to_px(team_xy[team][0], team_xy[team][1])
    return sum(1 for t,(x,y,_,_) in team_xy.items() if (to_px(x,y)[0]-px)**2+(to_px(x,y)[1]-py)**2 < (18*PX)**2)
order = sorted(ANNOTATE, key=crowd, reverse=True)
base_angles = np.deg2rad(np.arange(0, 360, 22.5))
for team in order:
    x, y, league, short = team_xy[team]
    px, py = to_px(x, y)
    w, h = text_size(short)
    away = np.arctan2(py-cy0, px-cx0)          # prefer directions pointing away from the cluster centre
    angles = sorted(base_angles, key=lambda a: abs((a-away+np.pi)%(2*np.pi)-np.pi))
    best = None
    manual = team in MANUAL
    if manual:                                # manual offset (dx, dy), no search
        dx, dy = MANUAL[team]
        a, dist = np.arctan2(dy, dx), float(np.hypot(dx, dy))
        bb = (px+dx*PX, py+dy*PX, px+dx*PX+w, py+dy*PX+h)   # text box anchored at its lower left corner
        anchor = (px+dx*PX, py+dy*PX)
        placed.append(bb); leaders.append(((px,py), anchor))
    for relaxed in ((False, True) if not manual else ()):   # strict pass first; relaxed pass may sit on ellipse/diagonal lines
      for dist in (10, 15, 21, 28, 36, 44, 54, 64):
        for a in angles:
            ox, oy = px + dist*PX*np.cos(a), py + dist*PX*np.sin(a)
            x0 = ox - w/2 + (w/2)*np.cos(a); y0 = oy - h/2 + (h/2)*np.sin(a)
            bb = (x0-1.5*PX, y0-1.5*PX, x0+w+1.5*PX, y0+h+1.5*PX)
            if bb[0] < axbb.x0+2 or bb[2] > axbb.x1-2 or bb[1] < axbb.y0+2 or bb[3] > axbb.y1-2: continue
            if box_hits_circles(bb, points_only=relaxed) or box_hits_rects(bb, rects + placed): continue
            anchor = (min(max(px, bb[0]), bb[2]), min(max(py, bb[1]), bb[3]))
            if seg_hits_points((px,py), anchor, (px,py)): continue
            if any(segs_cross((px,py), anchor, l[0], l[1]) for l in leaders): continue
            best = (a, dist, bb, anchor, relaxed); break
        if best: break
      if best: break
    if manual:
        pass
    elif best is None:
        unresolved.append(short); a, dist, bb, anchor = away, 30, None, (px,py)
    else:
        a, dist, bb, anchor, relaxed = best; placed.append(bb); leaders.append(((px,py), anchor))
        if relaxed: unresolved.append(short + ' (relaxed: may touch a line)')
    lc = C_EPL if league=='Premier League' else C_LIGA
    ax.annotate(short, xy=(x, y), xytext=(dist*np.cos(a), dist*np.sin(a)), textcoords='offset points',
                fontsize=LABEL_FS, color=lc, fontweight='bold',
                ha=('left' if manual else 'left' if np.cos(a)>0.3 else 'right' if np.cos(a)<-0.3 else 'center'),
                va=('baseline' if manual else 'bottom' if np.sin(a)>0.3 else 'top' if np.sin(a)<-0.3 else 'center'),
                arrowprops=dict(arrowstyle='-', color=lc, alpha=0.85, lw=0.7, shrinkA=0, shrinkB=2.5), zorder=12)

OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / 'fig7.pdf', bbox_inches='tight')
fig.savefig(OUT / 'fig7.png', dpi=220, bbox_inches='tight')
plt.close()
n_manual = sum(t in MANUAL for t in ANNOTATE)
print(f'Saved: {OUT / "fig7.pdf"}')
print(f'Saved: {OUT / "fig7.png"}')
print(f'soccer mean = ({fb_me:.3f}, {fb_ma:.3f}), RAI = {fb_rai:.3f} | labels {len(placed)}/{len(ANNOTATE)} '
      f'({n_manual} manual offsets, {len(ANNOTATE) - n_manual} automatic); unresolved: {unresolved}')

# ---------------- acceptance check ----------------
got = {
    'n_epl': int((ta['League'] == 'Premier League').sum()),
    'n_liga': int((ta['League'] == 'La Liga').sum()),
    'rai_soccer': fb_rai,                          # RAI of the soccer mean (legend)
    'rai_team_sd': ta['RAI'].std(ddof=1),          # spread of the 48 team RAI values
    'rai_team_min': ta['RAI'].min(), 'rai_team_max': ta['RAI'].max(),
    **{f'rai_{m}': v for m, v in model_rai.items()},
}
print('\nCHECK against the published Fig. 7 and Section 3.4')
ok_all = True
for key, exp in PUBLISHED.items():
    if key in DECIMALS:
        dec = DECIMALS[key]
        ok = abs(got[key] - exp) <= 0.5 * 10 ** (-dec) + 1e-12
        print(f'  {key:<14} {got[key]:>8.{dec + 1}f} vs {exp:>6.{dec}f} -> {"PASS" if ok else "FAIL"}')
    else:
        ok = got[key] == exp
        print(f'  {key:<14} {got[key]:>8d} vs {exp:>6d} -> {"PASS" if ok else "FAIL"}')
    ok_all &= ok
ok = bool((ta['RAI'] > 0).all())
print(f'  all 48 team RAI values positive -> {"PASS" if ok else "FAIL"}')
ok_all &= ok
if not ok_all:
    print('CHECK FAILED, the values above marked FAIL differ from the article.')
    sys.exit(1)
print('CHECK ALL PASS')
