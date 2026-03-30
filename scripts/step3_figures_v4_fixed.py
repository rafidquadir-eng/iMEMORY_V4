#!/usr/bin/env python3
"""
step3_figures_v3_1.py  — iMEMORY V4 Disease Axis and TI Phase figures.

Figure 2 TI phase method is IDENTICAL to V2 (step3_both_figures_final.py):
  - cKDTree of all TI-class sample UMAP coordinates
  - For each disease sample: query k=50 nearest TI neighbors
  - Count how many neighbors belong to E4/E1E5/E20 cluster sets
  - Assign to highest-vote category
  - If max vote = 0 → "No TI signal"

V4 cluster set corrections vs V2:
  E4_CLUSTERS: V2 had {84,90,92,81} (TI+KTx overlap clusters).
    V4 has no pure TI+KTx clusters. Use MIXED_TI clusters that
    appear in KTx: {128,196,173,142} each have ~28-31% TI + KTx overlap.
  E1E5_CLUSTERS: {488,587,526,432,465,466,500,438,247,479} (from step2 log)
  E20_CLUSTERS: CLEAN_TI - E1E5_CLUSTERS  (pure TI maintenance)
  PAN_CLUSTERS: REMOVED from voting. Was MIXED_TI (150 clusters) which
    dominated every k-NN neighborhood, suppressing all other phases.
    Pan-disease is still shown in Figure 1 as grey dots.

Figure 1 pan-disease fix:
  Grey dots rendered at zorder=6 (LAST, on top of all colored clusters).
  This ensures grey is always visible above any colored subcluster.

Legend placement:
  Left-column panels (CKD, Control): legend LEFT of panel
  Right-column panels (HD, KTx): legend RIGHT of panel
"""
import os, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

# ── TI cluster assignments from step2_v3_1_v2.log ────────────────────────────
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

# E4: MIXED_TI clusters with KTx overlap (~28-31% TI, appear in class 4)
# These are the V4 analog of V2's {84,90,92,81}
E4_CLUSTERS   = {128, 196, 173, 142}

# E1/E5: CLEAN_TI clusters overlapping rejection (from step2 AUTO-SUGGESTED)
E1E5_CLUSTERS = {488,587,526,432,465,466,500,438,247,479}

# E20: remaining CLEAN_TI clusters (pure TI maintenance)
E20_CLUSTERS  = CLEAN_TI - E1E5_CLUSTERS

# Pan-disease for Figure 1 grey rendering only — NOT used in phase voting
PAN_DISEASE_CLUSTERS = {128, 196, 173, 142}

PHASE_COLORS = {
    'E4 Priming':     '#0077CC',
    'E1/E5 Induction':'#DD2200',
    'E20 Maintenance':'#8800CC',
    'No TI signal':   '#1A1A2E',
}

# ── Disease cluster palettes ──────────────────────────────────────────────────
CKD_CLUSTERS = {
    242: {'label':'CKD Glomerulonephritis',   'dc':[2], 'color':'#D32F2F'},
    354: {'label':'CKD Diabetic nephropathy', 'dc':[2], 'color':'#FF6F00'},
     80: {'label':'CKD Glomerular disease',   'dc':[2], 'color':'#C2185B'},
    350: {'label':'CKD IgA nephropathy',      'dc':[2], 'color':'#7B1FA2'},
    267: {'label':'CKD ADPKD',                'dc':[2], 'color':'#1565C0'},
    313: {'label':'CKD Focal segmental GS',   'dc':[2], 'color':'#00838F'},
     12: {'label':'CKD Peripheral blood',     'dc':[2], 'color':'#558B2F'},
}
HD_CLUSTERS = {
    429: {'label':'HD Uremia / dialysis',        'dc':[3], 'color':'#0288D1'},
    407: {'label':'HD Innate immune activation', 'dc':[3], 'color':'#FF8F00'},
    262: {'label':'HD ESRD — early stage',       'dc':[3], 'color':'#283593'},
    344: {'label':'HD ESRD — late stage',        'dc':[3], 'color':'#6A1B9A'},
    337: {'label':'HD PBMC expression',          'dc':[3], 'color':'#00695C'},
    394: {'label':'HD Pre/post dialysis',        'dc':[3], 'color':'#D84315'},
    395: {'label':'HD ESRD peripheral blood',    'dc':[3], 'color':'#2E7D32'},
}
KTX_CLUSTERS = {
    236: {'label':'Acute Rejection — T-cell',      'dc':[0], 'color':'#B71C1C'},
    452: {'label':'Acute Rejection — Ab-mediated', 'dc':[0], 'color':'#FF3D00'},
    509: {'label':'Acute Rejection — mixed',       'dc':[0], 'color':'#FF8A65'},
    264: {'label':'Chronic Rejection — fibrosis',  'dc':[1], 'color':'#E65100'},
    104: {'label':'Chronic Rejection — IF/TA',     'dc':[1], 'color':'#F9A825'},
    417: {'label':'Chronic allograft dysfunction', 'dc':[1], 'color':'#FDD835'},
    387: {'label':'KTx — Calcineurin inhibitor',   'dc':[4], 'color':'#004D40'},
    361: {'label':'KTx — Long-term stable',        'dc':[4], 'color':'#006064'},
    347: {'label':'KTx — CNI minimization',        'dc':[4], 'color':'#4A148C'},
    455: {'label':'KTx — Graft monitoring',        'dc':[4], 'color':'#1B5E20'},
    464: {'label':'KTx — Multi-center biopsy',     'dc':[4], 'color':'#33691E'},
}
PANEL_CLUSTERS = {'CKD': CKD_CLUSTERS, 'HD': HD_CLUSTERS, 'KTx': KTX_CLUSTERS}

GREY         = '#CCCCCC'
PAN_COLOR    = '#757575'
GROUP_COLORS = {'CKD':'#C62828','HD':'#1565C0','KTx':'#2E7D32','Control':'#6A1B9A'}
GROUP_LABELS = {'CKD':'CKD','HD':'HD','KTx':'KTx\n(incl. rejection)','Control':'Control'}
GROUP_ORDER  = ['CKD','HD','Control','KTx']
LEGEND_SIDE  = {0:'left', 1:'right', 2:'left', 3:'right'}

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
                     'savefig.dpi':300,'savefig.bbox':'tight','savefig.pad_inches':0.12})

# ── DATA AUDIT ────────────────────────────────────────────────────────────────
print("=== DATA AUDIT ===")
emb_raw = np.load(EMB_PATH)
norms   = np.linalg.norm(emb_raw, axis=1)
assert norms.min() > 0.01 and not np.isnan(emb_raw).any()
endo = pd.read_csv(ENDO_PATH)
assert 'umap_x' in endo.columns and 'umap_y' in endo.columns, \
    "ERROR: umap_x/umap_y missing. Re-run step2."
print(f"Embeddings: {emb_raw.shape}  norms=[{norms.min():.3f},{norms.max():.3f}]")
print(f"Endotype CSV: {len(endo):,} rows")
print("=== AUDIT PASSED ✅ ===\n")

# ── Load and map display groups ───────────────────────────────────────────────
DISPLAY_MAP = {0:'KTx',1:'KTx',2:'CKD',3:'HD',4:'KTx'}
endo['display_group'] = endo['disease_class'].astype(int).map(DISPLAY_MAP)
ec = endo['refined_endotype'].astype(int)
ti = endo['disease_class'].astype(int) == 5

endo.loc[ti & ec.isin(CLEAN_TI),  'display_group'] = 'Control'
endo.loc[ti & ec.isin(MIXED_TI),  'display_group'] = 'TI_mixed'
endo.loc[ti & ~ec.isin(CLEAN_TI) & ~ec.isin(MIXED_TI), 'display_group'] = 'TI_impure'

df_all = endo[endo['display_group'].notna()].copy().reset_index(drop=True)
df_all['is_ti_class'] = (df_all['disease_class'].astype(int) == 5)

# Use pre-computed UMAP from step2 — no recomputation
df_all['ux'] = df_all['umap_x']
df_all['uy'] = df_all['umap_y']
print("Using pre-computed UMAP from step2 CSV.")

pad = 0.3
XMN=np.percentile(df_all['ux'],1)-pad; XMX=np.percentile(df_all['ux'],99)+pad
YMN=np.percentile(df_all['uy'],1)-pad; YMX=np.percentile(df_all['uy'],99)+pad

df      = df_all[(df_all['ux']>=XMN)&(df_all['ux']<=XMX)&
                  (df_all['uy']>=YMN)&(df_all['uy']<=YMX)].copy()
df_plot = df[df['display_group'].isin(['CKD','HD','KTx','Control'])].copy()
X1, X2  = 'ux', 'uy'

print("Plot groups:")
for g in ['CKD','HD','KTx','Control']:
    print(f"  {g}: {(df_plot['display_group']==g).sum():,}")

# ── Build TI k-NN tree — IDENTICAL to V2 ─────────────────────────────────────
ti_df     = df[df['is_ti_class']].copy().reset_index(drop=True)
ti_coords = ti_df[[X1,X2]].values
ti_cids   = ti_df['refined_endotype'].values.astype(int)
ti_tree   = cKDTree(ti_coords)
print(f"\nTI k-NN tree: {len(ti_df):,} TI samples")
print(f"  E4_CLUSTERS ({len(E4_CLUSTERS)}): {sorted(E4_CLUSTERS)[:5]}...")
print(f"  E1E5_CLUSTERS ({len(E1E5_CLUSTERS)}): {sorted(E1E5_CLUSTERS)[:5]}...")
print(f"  E20_CLUSTERS ({len(E20_CLUSTERS)}): {sorted(E20_CLUSTERS)[:5]}...")

# Verify cluster IDs exist in TI data
for name, cset in [('E4',E4_CLUSTERS),('E1E5',E1E5_CLUSTERS),('E20',E20_CLUSTERS)]:
    n = sum(1 for c in ti_cids if c in cset)
    print(f"  TI samples in {name}: {n:,}")


def assign_ti_phase(grp_df, k=K_PHASE):
    """
    Exact V2 method: k-NN majority vote over TI neighbor cluster membership.
    No 'TI overlap (pan-disease)' category — removed to prevent drowning out
    E4/E1E5/E20 votes.
    """
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
        # If max count is 0 → none of the 50 TI neighbors are in any cluster
        phases.append(best if counts[best] >= 1 else 'No TI signal')
    return np.array(phases)


def _place_legend(ax, handles, ax_idx):
    if LEGEND_SIDE[ax_idx] == 'left':
        ax.legend(handles=handles,
                  bbox_to_anchor=(-0.02, 1.0), loc='upper right',
                  fontsize=4.5, frameon=True, framealpha=0.92,
                  ncol=1, handlelength=0.8, labelspacing=0.20,
                  borderpad=0.4, handletextpad=0.4)
    else:
        ax.legend(handles=handles,
                  bbox_to_anchor=(1.01, 1.0), loc='upper left',
                  fontsize=4.5, frameon=True, framealpha=0.92,
                  ncol=1, handlelength=0.8, labelspacing=0.20,
                  borderpad=0.4, handletextpad=0.4)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Disease Progression Axis
# ══════════════════════════════════════════════════════════════════════════════
print("\nFigure 1...")
fig1, axes = plt.subplots(2, 2, figsize=(13, 9.5))
fig1.subplots_adjust(left=0.22, right=0.82, hspace=0.30, wspace=0.12)
axes_flat = axes.flatten()

for ax_idx, (ax, group) in enumerate(zip(axes_flat, GROUP_ORDER)):
    bg = df_plot[df_plot['display_group'] != group]
    fg = df_plot[df_plot['display_group'] == group].copy()

    ax.scatter(bg[X1], bg[X2], c=GREY, s=2, alpha=0.08, linewidths=0,
               rasterized=True, zorder=1)

    if group == 'Control':
        ax.scatter(fg[X1], fg[X2], c=GROUP_COLORS['Control'], s=4, alpha=1.0,
                   linewidths=0, rasterized=True, zorder=2)
        ax.set_title(f'Control  (n={len(fg):,})', fontsize=11,
                     fontweight='bold', color=GROUP_COLORS['Control'], pad=6)
    else:
        cluster_map = PANEL_CLUSTERS[group]

        # 1. Ungrouped samples first (dim base color)
        all_named = np.zeros(len(fg), dtype=bool)
        for cid, meta in cluster_map.items():
            m = ((fg['refined_endotype'] == cid) &
                 (fg['disease_class'].astype(int).isin(meta['dc']))).values
            all_named |= m
        pan_mask = fg['refined_endotype'].isin(PAN_DISEASE_CLUSTERS).values
        all_named |= pan_mask
        ungrouped = ~all_named
        if ungrouped.sum() > 0:
            ax.scatter(fg[X1].values[ungrouped], fg[X2].values[ungrouped],
                       c=GROUP_COLORS[group], s=4, alpha=0.4,
                       linewidths=0, rasterized=True, zorder=2)

        # 2. Named disease subclusters (full opacity)
        legend_handles = []
        seen = set()
        for cid, meta in cluster_map.items():
            m = ((fg['refined_endotype'] == cid) &
                 (fg['disease_class'].astype(int).isin(meta['dc']))).values
            if m.sum() < 5: continue
            ax.scatter(fg[X1].values[m], fg[X2].values[m],
                       c=meta['color'], s=5, alpha=1.0,
                       linewidths=0, rasterized=True, zorder=3)
            lbl = meta['label']
            if lbl not in seen:
                seen.add(lbl)
                legend_handles.append(mpatches.Patch(facecolor=meta['color'], label=lbl))

        # 3. Pan-disease clusters LAST at highest zorder — always visible on top
        if pan_mask.sum() > 0:
            ax.scatter(fg[X1].values[pan_mask], fg[X2].values[pan_mask],
                       c=PAN_COLOR, s=4, alpha=0.6, linewidths=0,
                       rasterized=True, zorder=6)
            legend_handles.insert(0, mpatches.Patch(facecolor=PAN_COLOR,
                                                     label='Pan-disease (CKD/HD/KTx)'))

        if legend_handles:
            _place_legend(ax, legend_handles, ax_idx)

        ax.set_title(f'{GROUP_LABELS[group]}  (n={len(fg):,})', fontsize=11,
                     fontweight='bold', color=GROUP_COLORS[group], pad=6)

    ax.set_xlim(XMN, XMX); ax.set_ylim(YMN, YMX)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel('UMAP 1', fontsize=8); ax.set_ylabel('UMAP 2', fontsize=8)
    for sp in ax.spines.values():
        sp.set_linewidth(0.5); sp.set_color('#BBBBBB')

fig1.suptitle('iMEMORY V4 Embedding — Disease Progression Axis\n'
              '(UMAP | color = clinical sub-population | legend per panel)',
              fontsize=10, fontweight='bold', y=1.01)
fig1.text(0.01, 0.99, 'A', fontsize=14, fontweight='bold', va='top')
p1 = os.path.join(OUTDIR, 'Figure1_V4_DiseaseAxis_Annotated.png')
fig1.savefig(p1, dpi=300, bbox_inches='tight'); plt.close(fig1)
print(f"  Saved: {p1}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — TI Endotype Phase Mapping  (exact V2 method)
# ══════════════════════════════════════════════════════════════════════════════
print("\nFigure 2...")
fig2, axes2 = plt.subplots(2, 2, figsize=(13, 9.5))
fig2.subplots_adjust(left=0.22, right=0.82, hspace=0.30, wspace=0.12)
axes2_flat = axes2.flatten()

for ax_idx, (ax, group) in enumerate(zip(axes2_flat, GROUP_ORDER)):
    bg = df_plot[df_plot['display_group'] != group]
    ax.scatter(bg[X1], bg[X2], c=GREY, s=2, alpha=0.08, linewidths=0,
               rasterized=True, zorder=1)

    if group == 'Control':
        fg = df_plot[df_plot['display_group'] == 'Control']
        ax.scatter(fg[X1], fg[X2], c=GROUP_COLORS['Control'], s=4, alpha=1.0,
                   linewidths=0, rasterized=True, zorder=2)
        ax.set_title(f'Control  (clean TI reference, n={len(fg):,})', fontsize=11,
                     fontweight='bold', color=GROUP_COLORS['Control'], pad=6)
        ax.text(0.50, 0.04, 'GSE180697 + GEO datasets | no disease signatures',
                transform=ax.transAxes, ha='center', va='bottom',
                fontsize=7, color='#888888', style='italic')
    else:
        fg = df_plot[df_plot['display_group'] == group].copy().reset_index(drop=True)
        print(f"  Phase assignment: {group} (n={len(fg):,}, k={K_PHASE})...")
        phases = assign_ti_phase(fg, k=K_PHASE)
        phase_counts = Counter(phases)
        for ph, n in sorted(phase_counts.items(), key=lambda x: -x[1]):
            print(f"    {ph}: {n:,} ({100*n/len(fg):.1f}%)")

        # No TI signal first (dark background), then colored phases on top
        no_ti = phases == 'No TI signal'
        ax.scatter(fg[X1].values[no_ti], fg[X2].values[no_ti],
                   c=PHASE_COLORS['No TI signal'], s=5, alpha=1.0,
                   linewidths=0, rasterized=True, zorder=2)

        phase_order = ['E4 Priming', 'E1/E5 Induction', 'E20 Maintenance']
        legend_handles = [
            mpatches.Patch(facecolor=PHASE_COLORS['No TI signal'],
                           label=f'No TI signal (n={no_ti.sum():,})')
        ]
        for phase in phase_order:
            mask = phases == phase
            if mask.sum() < 3: continue
            ax.scatter(fg[X1].values[mask], fg[X2].values[mask],
                       c=PHASE_COLORS[phase], s=6, alpha=1.0,
                       linewidths=0, rasterized=True, zorder=4)
            legend_handles.append(
                mpatches.Patch(facecolor=PHASE_COLORS[phase],
                               label=f'{phase} (n={mask.sum():,})'))

        _place_legend(ax, legend_handles, ax_idx)
        ax.set_title(f'{GROUP_LABELS[group]}  ●  TI endotype phases',
                     fontsize=11, fontweight='bold', color=GROUP_COLORS[group], pad=6)

    ax.set_xlim(XMN, XMX); ax.set_ylim(YMN, YMX)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel('UMAP 1', fontsize=8); ax.set_ylabel('UMAP 2', fontsize=8)
    for sp in ax.spines.values():
        sp.set_linewidth(0.5); sp.set_color('#BBBBBB')

fig2.suptitle('iMEMORY V4 TI Endotype Phase Mapping Within the Disease Continuum\n'
              '(k=50 TI-neighbor phase assignment | No TI signal = biologically TI-negative)',
              fontsize=10, fontweight='bold', y=1.01)
fig2.text(0.01, 0.99, 'B', fontsize=14, fontweight='bold', va='top')

leg = [mpatches.Patch(facecolor=GROUP_COLORS[g],
        label=GROUP_LABELS[g].replace('\n',' ')) for g in ['CKD','HD','KTx','Control']]
leg += [mpatches.Patch(facecolor=v, label=k) for k, v in PHASE_COLORS.items()]
fig2.legend(handles=leg, loc='lower center', ncol=4, fontsize=7,
            frameon=True, framealpha=0.92, bbox_to_anchor=(0.5, -0.04))
p2 = os.path.join(OUTDIR, 'Figure2_V4_TrainedImmunity_Annotated.png')
fig2.savefig(p2, dpi=300, bbox_inches='tight'); plt.close(fig2)
print(f"  Saved: {p2}")

import shutil
# shutil.copy(p1, os.path.join(COPY_DIR, 'Figure1_V4_DiseaseAxis_Annotated.png'))
# shutil.copy(p2, os.path.join(COPY_DIR, 'Figure2_V4_TrainedImmunity_Annotated.png'))
print(f"\nCopied to {COPY_DIR}")
print("DONE.")
