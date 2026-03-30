#!/usr/bin/env python3
"""
step5_ochando_slides_v3_1.py — iMEMORY V4 Ochando Slide 1

OUTPUT: Figure_Slide1_DiseaseAxis_TrainingOverlay.png
  A single wide figure composed of two 2×2 UMAP grids side by side.

  LEFT 2×2  (New Slide 1 — LEFT panel):
    Same 4 panels as Figure 1 (CKD/HD/Control/KTx), same UMAP coordinates,
    same axis limits. But SIMPLIFIED:
      - All CKD samples = single solid red (#C62828), no subclusters
      - All HD samples  = single solid blue (#1565C0), no subclusters
      - All KTx samples = single solid green (#2E7D32), no subclusters
      - Control         = single solid purple (#6A1B9A)
      - Background (other groups) = grey cloud
      - NO legend, NO subcluster color coding, NO pan-disease overlay
      - Disease name label only (no n=, no subcategory text)

  RIGHT 2×2 (New Slide 1 — RIGHT panel):
    Same 4 panels, same coordinates, same layout. But overlays TI phase:
      - Background = grey cloud (all other groups)
      - Control panel: Control samples in purple (same as LEFT, no TI overlay)
      - CKD/HD/KTx panels:
          · "No TI signal" samples = very dark (#1A1A2E), plotted first as base
          · E4 Priming    = blue  (#0077CC), on top
          · E1/E5 Induction = red (#DD2200), on top
          · E20 Maintenance = purple (#8800CC), on top
      - TI phase assigned per-sample using IDENTICAL method to step3 Figure 2:
        k=50 nearest TI neighbors, majority vote over E4/E1E5/E20 cluster sets
      - Legend: one shared legend strip at bottom of right grid showing phase colors

  Data: Uses umap_x/umap_y from endotype CSV (NO recomputation).
  TI phase method: identical to step3 Figure 2 (V2-matched k-NN vote).
"""
import os, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from scipy.spatial import cKDTree
from collections import Counter
import pathlib

ENDO_PATH = '/mnt/nvme15tb/trained_im/outputs/V4/stage3/endotype_assignments_v4_final.csv'
EMB_PATH  = '/mnt/nvme15tb/trained_im/outputs/V4/stage1/embeddings_v4.npy'
OUTDIR    = '/mnt/nvme15tb/trained_im/outputs/V4/figures/'
COPY_DIR  = '/mnt/user-data/outputs/'
pathlib.Path(OUTDIR).mkdir(parents=True, exist_ok=True)
# COPY_DIR disabled

K_PHASE = 50

# ── TI cluster assignments (identical to step3) ───────────────────────────────
CLEAN_TI = {
    515,516,10,525,526,532,533,534,535,537,538,539,542,543,544,548,549,550,551,
    552,558,559,560,579,587,588,590,591,594,596,597,598,599,600,601,602,603,604,
    605,606,607,609,610,611,612,613,614,615,616,617,618,619,245,247,293,305,315,
    317,318,411,421,426,432,438,443,447,451,456,465,466,470,471,472,476,479,485,
    486,488,489,490,491,494,495,496,497,498,499,500,503,504,505,508
}
MIXED_TI = {
    0,7,9,14,15,16,29,31,33,34,35,38,42,43,49,50,52,56,58,59,63,67,68,69,70,72,
    73,74,75,77,78,79,81,85,88,89,90,92,95,97,99,100,101,105,110,114,115,118,121,
    122,124,125,129,130,135,136,139,140,141,142,143,144,145,146,148,149,151,155,
    159,160,161,164,165,166,168,169,173,175,176,178,179,180,182,183,185,186,191,
    194,195,196,198,201,202,203,204,207,208,209,210,211,216,217,218,219,221,223,
    224,225,229,232,235,238,239,256,284,291,307,314,319,320,402,404,406,408,415,
    416,419,420,424,430,431,434,441,453,454,457,458,462,463,468,492,493,501,502,
    507,519,523,547,589,592,593,595,608
}
E4_CLUSTERS   = {128, 196, 173, 142}
E1E5_CLUSTERS = {488,587,526,432,465,466,500,438,247,479}
E20_CLUSTERS  = CLEAN_TI - E1E5_CLUSTERS

# ── Colors ────────────────────────────────────────────────────────────────────
GROUP_COLORS = {'CKD':'#C62828','HD':'#1565C0','KTx':'#2E7D32','Control':'#6A1B9A'}
PHASE_COLORS = {
    'E4 Priming':     '#0077CC',
    'E1/E5 Induction':'#DD2200',
    'E20 Maintenance':'#8800CC',
    'No TI signal':   '#1A1A2E',
}
GREY = '#CCCCCC'

GROUP_ORDER   = ['CKD', 'HD', 'Control', 'KTx']
GROUP_TITLES  = {'CKD':'CKD', 'HD':'HD', 'Control':'Control', 'KTx':'KTx'}

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
                     'savefig.dpi':300,'savefig.bbox':'tight','savefig.pad_inches':0.15})

# ── DATA AUDIT ────────────────────────────────────────────────────────────────
print("=== DATA AUDIT ===")
emb_raw = np.load(EMB_PATH)
norms   = np.linalg.norm(emb_raw, axis=1)
assert norms.min() > 0.01 and not np.isnan(emb_raw).any()
endo = pd.read_csv(ENDO_PATH)
assert 'umap_x' in endo.columns and 'umap_y' in endo.columns
print(f"Embeddings: {emb_raw.shape}  norms=[{norms.min():.3f},{norms.max():.3f}]")
print(f"Endotype CSV: {len(endo):,} rows  umap_x=[{endo['umap_x'].min():.2f},{endo['umap_x'].max():.2f}]")
print("=== AUDIT PASSED ✅ ===\n")

# ── Map display groups (identical to step3) ───────────────────────────────────
DISPLAY_MAP = {0:'KTx',1:'KTx',2:'CKD',3:'HD',4:'KTx'}
endo['display_group'] = endo['disease_class'].astype(int).map(DISPLAY_MAP)
ec = endo['refined_endotype'].astype(int)
ti = endo['disease_class'].astype(int) == 5

endo.loc[ti & ec.isin(CLEAN_TI),  'display_group'] = 'Control'
endo.loc[ti & ec.isin(MIXED_TI),  'display_group'] = 'TI_mixed'
endo.loc[ti & ~ec.isin(CLEAN_TI) & ~ec.isin(MIXED_TI), 'display_group'] = 'TI_impure'

df_all = endo[endo['display_group'].notna()].copy().reset_index(drop=True)
df_all['is_ti_class'] = (df_all['disease_class'].astype(int) == 5)
df_all['ux'] = df_all['umap_x']
df_all['uy'] = df_all['umap_y']
print("Using pre-computed UMAP coordinates from step2 CSV (no recomputation).")

pad = 0.3
XMN=np.percentile(df_all['ux'],1)-pad; XMX=np.percentile(df_all['ux'],99)+pad
YMN=np.percentile(df_all['uy'],1)-pad; YMX=np.percentile(df_all['uy'],99)+pad

df      = df_all[(df_all['ux']>=XMN)&(df_all['ux']<=XMX)&
                  (df_all['uy']>=YMN)&(df_all['uy']<=YMX)].copy()
df_plot = df[df['display_group'].isin(['CKD','HD','KTx','Control'])].copy()
X1, X2  = 'ux', 'uy'

print("Plot groups:")
for g in GROUP_ORDER:
    print(f"  {g}: {(df_plot['display_group']==g).sum():,}")

# ── Build TI k-NN tree (identical to step3) ───────────────────────────────────
ti_df     = df[df['is_ti_class']].copy().reset_index(drop=True)
ti_coords = ti_df[[X1,X2]].values
ti_cids   = ti_df['refined_endotype'].values.astype(int)
ti_tree   = cKDTree(ti_coords)
print(f"\nTI k-NN tree: {len(ti_df):,} TI samples")
for name, cset in [('E4',E4_CLUSTERS),('E1E5',E1E5_CLUSTERS),('E20',E20_CLUSTERS)]:
    n = sum(1 for c in ti_cids if c in cset)
    print(f"  TI samples in {name}: {n:,}")


def assign_ti_phase(grp_df, k=K_PHASE):
    """Exact same method as step3 Figure 2."""
    pts    = grp_df[[X1,X2]].values
    phases = []
    for px, py in zip(pts[:,0], pts[:,1]):
        _, idx = ti_tree.query([px,py], k=min(k, len(ti_coords)))
        nbr    = ti_cids[idx]
        counts = {
            'E4 Priming':     sum(c in E4_CLUSTERS   for c in nbr),
            'E1/E5 Induction':sum(c in E1E5_CLUSTERS for c in nbr),
            'E20 Maintenance':sum(c in E20_CLUSTERS  for c in nbr),
        }
        best = max(counts, key=counts.get)
        phases.append(best if counts[best] >= 1 else 'No TI signal')
    return np.array(phases)


# ── Pre-compute TI phases for all disease groups ──────────────────────────────
# Do this once before plotting so both grids share the same assignments
disease_phases = {}
for group in ['CKD', 'HD', 'KTx']:
    fg = df_plot[df_plot['display_group'] == group].copy().reset_index(drop=True)
    print(f"\nPhase assignment: {group} (n={len(fg):,}, k={K_PHASE})...")
    phases = assign_ti_phase(fg, k=K_PHASE)
    disease_phases[group] = phases
    counts = Counter(phases)
    for ph, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {ph}: {n:,} ({100*n/len(fg):.1f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE: Side-by-side (LEFT simplified  |  RIGHT training overlay)
# Layout: 2 columns of 2×2 subplots, separated by a narrow gap
# ══════════════════════════════════════════════════════════════════════════════
print("\nBuilding Ochando Slide 1 figure...")

# Total figure: 2 groups of 4 axes each = 8 axes
# Use GridSpec: 2 rows × 4 cols, with a wider space between col 1 and col 2
fig = plt.figure(figsize=(22, 10))
# LEFT grid: cols 0-1, RIGHT grid: cols 2-3, gap via hspace/wspace
gs = gridspec.GridSpec(2, 4, figure=fig,
                       left=0.03, right=0.97,
                       top=0.90, bottom=0.10,
                       hspace=0.20, wspace=0.08)
# Insert extra gap between col 1 and col 2 by adjusting left/right of each sub-gridspec
gs_left  = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[:, 0:2],
                                             hspace=0.20, wspace=0.06)
gs_right = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[:, 2:4],
                                             hspace=0.20, wspace=0.06)

# Map GROUP_ORDER to grid positions: CKD=top-left, HD=top-right, Control=bot-left, KTx=bot-right
GRID_POSITIONS = {
    'CKD':     (0, 0),
    'HD':      (0, 1),
    'Control': (1, 0),
    'KTx':     (1, 1),
}

axes_left  = {}
axes_right = {}
for group in GROUP_ORDER:
    r, c = GRID_POSITIONS[group]
    axes_left[group]  = fig.add_subplot(gs_left[r, c])
    axes_right[group] = fig.add_subplot(gs_right[r, c])


def _style_ax(ax):
    ax.set_xlim(XMN, XMX); ax.set_ylim(YMN, YMX)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.5); sp.set_color('#DDDDDD')


# ── LEFT 2×2: Simplified disease clusters ────────────────────────────────────
for group in GROUP_ORDER:
    ax  = axes_left[group]
    fg  = df_plot[df_plot['display_group'] == group]
    bg  = df_plot[df_plot['display_group'] != group]

    # Grey background
    ax.scatter(bg[X1], bg[X2], c=GREY, s=1.5, alpha=0.07, linewidths=0,
               rasterized=True, zorder=1)
    # Single solid color per disease — no subclusters, no pan-disease overlay
    ax.scatter(fg[X1], fg[X2], c=GROUP_COLORS[group], s=3, alpha=1.0,
               linewidths=0, rasterized=True, zorder=2)

    # Title: disease name only, bold, matching group color
    ax.set_title(GROUP_TITLES[group], fontsize=13, fontweight='bold',
                 color=GROUP_COLORS[group], pad=5)
    _style_ax(ax)


# ── RIGHT 2×2: Training overlay ───────────────────────────────────────────────
for group in GROUP_ORDER:
    ax  = axes_right[group]
    fg  = df_plot[df_plot['display_group'] == group].copy().reset_index(drop=True)
    bg  = df_plot[df_plot['display_group'] != group]

    # Grey background
    ax.scatter(bg[X1], bg[X2], c=GREY, s=1.5, alpha=0.07, linewidths=0,
               rasterized=True, zorder=1)

    if group == 'Control':
        # Control: same as left — solid purple, no TI overlay on control
        ax.scatter(fg[X1], fg[X2], c=GROUP_COLORS['Control'], s=3, alpha=1.0,
                   linewidths=0, rasterized=True, zorder=2)
        ax.set_title(GROUP_TITLES['Control'], fontsize=13, fontweight='bold',
                     color=GROUP_COLORS['Control'], pad=5)
    else:
        phases = disease_phases[group]

        # 1. No TI signal — dark, plotted first as base layer
        no_ti = phases == 'No TI signal'
        if no_ti.sum() > 0:
            ax.scatter(fg[X1].values[no_ti], fg[X2].values[no_ti],
                       c=PHASE_COLORS['No TI signal'], s=3, alpha=0.85,
                       linewidths=0, rasterized=True, zorder=2)

        # 2. TI phases on top — E4, E1/E5, E20
        for phase in ['E4 Priming', 'E1/E5 Induction', 'E20 Maintenance']:
            mask = phases == phase
            if mask.sum() < 3: continue
            ax.scatter(fg[X1].values[mask], fg[X2].values[mask],
                       c=PHASE_COLORS[phase], s=4, alpha=1.0,
                       linewidths=0, rasterized=True, zorder=4)

        ax.set_title(GROUP_TITLES[group], fontsize=13, fontweight='bold',
                     color=GROUP_COLORS[group], pad=5)

    _style_ax(ax)


# ── Section headers ────────────────────────────────────────────────────────────
# "Disease Progression Axis" over left grid
fig.text(0.255, 0.935, 'Disease Progression Axis',
         ha='center', va='bottom', fontsize=14, fontweight='bold', color='#222222')
# "Trained Immunity Endotype Overlay" over right grid
fig.text(0.745, 0.935, 'Trained Immunity Endotype Overlay',
         ha='center', va='bottom', fontsize=14, fontweight='bold', color='#222222')

# Thin vertical separator line between the two grids
fig.add_artist(plt.Line2D([0.505, 0.505], [0.05, 0.93],
                           transform=fig.transFigure,
                           color='#CCCCCC', linewidth=1.2, linestyle='--'))

# ── Shared legend for RIGHT grid — disease colors ─────────────────────────────
disease_handles = [
    mpatches.Patch(facecolor=GROUP_COLORS[g], label=g)
    for g in ['CKD', 'HD', 'KTx', 'Control']
]
phase_handles = [
    mpatches.Patch(facecolor=PHASE_COLORS[p], label=p)
    for p in ['E4 Priming', 'E1/E5 Induction', 'E20 Maintenance', 'No TI signal']
]

# Disease legend: bottom-left (under left grid)
fig.legend(handles=disease_handles,
           loc='lower left', bbox_to_anchor=(0.03, 0.01),
           ncol=4, fontsize=9, frameon=True, framealpha=0.92,
           title='Disease group', title_fontsize=8,
           handlelength=1.0, columnspacing=1.0)

# Phase legend: bottom-right (under right grid)
fig.legend(handles=phase_handles,
           loc='lower right', bbox_to_anchor=(0.97, 0.01),
           ncol=4, fontsize=9, frameon=True, framealpha=0.92,
           title='TI endotype phase', title_fontsize=8,
           handlelength=1.0, columnspacing=1.0)

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = os.path.join(OUTDIR, 'Figure_Slide1_DiseaseAxis_TrainingOverlay.png')
fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"\nSaved: {out_path}")

import shutil
# shutil.copy(out_path, os.path.join(COPY_DIR, 'Figure_Slide1_DiseaseAxis_TrainingOverlay.png'))
print(f"Copied to {COPY_DIR}")
print("DONE.")
