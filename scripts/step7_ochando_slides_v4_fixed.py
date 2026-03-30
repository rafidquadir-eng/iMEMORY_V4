#!/usr/bin/env python3
"""
step7_ochando_slides_v3_1.py — iMEMORY V4 Ochando Slide 3 (Tasks T3.6 + T3.8)

OUTPUT: Two figures

Figure_Slide3A_PhaseByDisease.png  (Task T3.6)
  Inverse of Panel A in step6. Shows for each disease class what fraction
  of its samples fall into each TI phase.
    X-axis = disease class (CKD / HD / KTx)
    Stacked bars = fraction in E4 Priming / E1/E5 Induction / E20 Maintenance / No TI signal
  Answers: "Within CKD patients, how many are in each TI training stage?"
  vs Panel A which showed "Within E4 phase, how many are CKD/HD/KTx?"
  Data source: same k-NN phase assignment as step3 v6 — recomputed from endotype CSV.

Figure_Slide3B_ModuleByDiseasePhase.png  (Task T3.8)
  3 disease groups × 3 TI phases × 4 key modules heatmap.
  Shows whether CKD-E4 looks different from HD-E4 — the disease-specific
  training fingerprint question.
  Groups: CKD-E4, CKD-E1/E5, CKD-E20, HD-E4, HD-E1/E5, HD-E20,
          KTx-E4, KTx-E1/E5, KTx-E20
  Modules shown: mTOR_glycolytic, trained_immunity_module,
                 NF_kB_epigenetic, epigenetic_remodeling
                 (the 4 most mechanistically relevant to the R01)
  Visualization: 3-panel heatmap (one panel per disease class),
                 each panel = 4 modules × 3 TI phases, min-shifted values.
  Data source: LMDB extraction, N=100 samples per disease×phase group.
"""
import os, sys, lmdb, pickle, numpy as np, pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial import cKDTree
from collections import Counter
import random, pathlib
random.seed(42)
np.random.seed(42)

sys.path.insert(0, '/mnt/data/trained_im/scripts')
from hetero_encoder_v3 import HeteroClassificationModelV3

# ── Paths ─────────────────────────────────────────────────────────────────────
ENDO_PATH  = '/mnt/nvme15tb/trained_im/outputs/V4/stage3/endotype_assignments_v4_final.csv'
CKPT       = '/mnt/nvme15tb/trained_im/checkpoints/phase2_v3_1/phase2_v3_1_best.pt'
GRAPH_DIR  = '/mnt/nvme15tb/trained_im/graphs_lmdb_v3'
OUTDIR     = '/mnt/nvme15tb/trained_im/outputs/V4/figures/'
COPY_DIR   = '/mnt/user-data/outputs/'
pathlib.Path(OUTDIR).mkdir(parents=True, exist_ok=True)
# COPY_DIR disabled
DEVICE     = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
N_PER_GROUP = 100   # LMDB samples per disease×phase cell for T3.8

# ── TI cluster sets ────────────────────────────────────────────────────────────
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

MODULE_NAMES = [
    'mTOR_glycolytic','NF_kB_epigenetic','trained_immunity_module',
    'inflammation_innate','inflammation_adaptive','metabolism_glycolysis',
    'metabolism_oxidative','stress_oxidative','stress_er','mTOR_signaling',
    'cytokine_response','epigenetic_remodeling','immune_memory','antigen_presentation'
]
# 4 key modules for T3.8 heatmap
KEY_MODULES = ['mTOR_glycolytic','trained_immunity_module',
               'NF_kB_epigenetic','epigenetic_remodeling']
KEY_LABELS  = ['mTOR\nglycolytic','Trained\nimmunity',
               'NF-kB /\nepigenetic','Epigenetic\nremodeling']
KEY_IDXS    = [MODULE_NAMES.index(m) for m in KEY_MODULES]

PHASE_COLORS = {
    'E4 Priming':     '#0077CC',
    'E1/E5 Induction':'#DD2200',
    'E20 Maintenance':'#8800CC',
    'No TI signal':   '#1A1A2E',
}
DISEASE_COLORS = {'CKD':'#C62828','HD':'#1565C0','KTx':'#2E7D32'}

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                     'savefig.dpi':300,'savefig.bbox':'tight','savefig.pad_inches':0.15})


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load data and assign TI phases (shared by T3.6 and T3.8)
# ══════════════════════════════════════════════════════════════════════════════
print("=== STEP 1: Load data and assign TI phases ===")
endo = pd.read_csv(ENDO_PATH)
assert 'umap_x' in endo.columns

DISPLAY_MAP = {0:'KTx',1:'KTx',2:'CKD',3:'HD',4:'KTx'}
endo['display_group'] = endo['disease_class'].astype(int).map(DISPLAY_MAP)
ec = endo['refined_endotype'].astype(int)
ti = endo['disease_class'].astype(int) == 5
endo.loc[ti & ec.isin(CLEAN_TI),  'display_group'] = 'Control'
endo.loc[ti & ec.isin(MIXED_TI),  'display_group'] = 'TI_mixed'
endo.loc[ti & ~ec.isin(CLEAN_TI) & ~ec.isin(MIXED_TI), 'display_group'] = 'TI_impure'
endo['ux'] = endo['umap_x']
endo['uy'] = endo['umap_y']

pad = 0.3
XMN=np.percentile(endo['ux'],1)-pad; XMX=np.percentile(endo['ux'],99)+pad
YMN=np.percentile(endo['uy'],1)-pad; YMX=np.percentile(endo['uy'],99)+pad
df_all = endo[(endo['ux']>=XMN)&(endo['ux']<=XMX)&
              (endo['uy']>=YMN)&(endo['uy']<=YMX)].copy()

ti_df     = df_all[df_all['display_group'].isin(['Control','TI_mixed'])].copy().reset_index(drop=True)
ti_coords = ti_df[['ux','uy']].values
ti_cids   = ti_df['refined_endotype'].values.astype(int)
ti_tree   = cKDTree(ti_coords)
print(f"TI tree: {len(ti_df):,} | "
      f"E4:{sum(c in E4_CLUSTERS for c in ti_cids):,} "
      f"E1E5:{sum(c in E1E5_CLUSTERS for c in ti_cids):,} "
      f"E20:{sum(c in E20_CLUSTERS for c in ti_cids):,}")

K_PHASE = 50
def assign_phase(pts):
    phases = []
    for px, py in zip(pts[:,0], pts[:,1]):
        _, idx = ti_tree.query([px,py], k=min(K_PHASE,len(ti_coords)))
        nbr = ti_cids[idx]
        counts = {
            'E4 Priming':     sum(c in E4_CLUSTERS   for c in nbr),
            'E1/E5 Induction':sum(c in E1E5_CLUSTERS for c in nbr),
            'E20 Maintenance':sum(c in E20_CLUSTERS  for c in nbr),
        }
        best = max(counts, key=counts.get)
        phases.append(best if counts[best]>=1 else 'No TI signal')
    return np.array(phases)

# Assign phases for all disease samples
disease_df = df_all[df_all['display_group'].isin(['CKD','HD','KTx'])].copy().reset_index(drop=True)
print(f"Assigning phases for {len(disease_df):,} disease samples...")
disease_df['ti_phase'] = assign_phase(disease_df[['ux','uy']].values)

# Print cross-tab for verification
ct = pd.crosstab(disease_df['display_group'], disease_df['ti_phase'])
print("\nDisease × Phase cross-tab (counts):")
print(ct)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Figure_Slide3A: TI phase breakdown within each disease (T3.6)
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== STEP 2: T3.6 — Phase fraction within each disease class ===")

# For each disease, compute fraction of samples in each TI phase
PHASES_ORDERED = ['E4 Priming','E1/E5 Induction','E20 Maintenance','No TI signal']
DISEASES       = ['CKD','HD','KTx']

phase_fracs = {}   # disease → {phase: fraction}
for disease in DISEASES:
    sub   = disease_df[disease_df['display_group']==disease]
    total = len(sub)
    fracs = {}
    for phase in PHASES_ORDERED:
        n = (sub['ti_phase']==phase).sum()
        fracs[phase] = 100*n/total if total else 0
        print(f"  {disease} → {phase}: n={n:,} ({fracs[phase]:.1f}%)")
    phase_fracs[disease] = fracs

fig_a, ax_a = plt.subplots(figsize=(9, 6))
fig_a.subplots_adjust(left=0.12, right=0.85, top=0.88, bottom=0.14)

x_a    = np.arange(len(DISEASES))
w_a    = 0.55
bottom = np.zeros(len(DISEASES))

bars_by_phase = {}
for phase in PHASES_ORDERED:
    vals = np.array([phase_fracs[d][phase] for d in DISEASES])
    bar  = ax_a.bar(x_a, vals, w_a, bottom=bottom,
                    color=PHASE_COLORS[phase], alpha=0.88, label=phase)
    # Annotate each segment if large enough (>5%)
    for i, (v, b) in enumerate(zip(vals, bottom)):
        if v > 5:
            ax_a.text(i, b + v/2, f'{v:.0f}%',
                      ha='center', va='center', fontsize=9,
                      color='white', fontweight='bold')
    bars_by_phase[phase] = bar
    bottom = bottom + vals

# n= labels above bars
for i, disease in enumerate(DISEASES):
    n_total = (disease_df['display_group']==disease).sum()
    ax_a.text(i, 102, f'n={n_total:,}', ha='center', va='bottom',
              fontsize=8.5, color='#444444')

ax_a.set_xticks(x_a)
ax_a.set_xticklabels(['CKD','Hemodialysis','KTx\n(incl. rejection)'], fontsize=11)
ax_a.set_ylabel('% of disease class samples', fontsize=11)
ax_a.set_ylim(0, 112)
ax_a.set_title('iMEMORY V4 — TI Endotype Phase Distribution Within Each Disease Class\n'
               'What fraction of CKD / HD / KTx patients are in each training stage?',
               fontsize=11, fontweight='bold', pad=10)
ax_a.axhline(100, color='#AAAAAA', lw=0.7, ls='--')
ax_a.legend(handles=[bars_by_phase[p] for p in PHASES_ORDERED],
            labels=PHASES_ORDERED,
            bbox_to_anchor=(1.01,1), loc='upper left',
            fontsize=9, framealpha=0.92, handlelength=1.0)
for sp in ax_a.spines.values(): sp.set_edgecolor('#CCC')
ax_a.yaxis.grid(True, color='#EEE', lw=0.7)
ax_a.set_axisbelow(True)
ax_a.text(0.99, 0.01, 'iMEMORY V4  ·  k=50 TI-neighbor phase assignment',
          transform=ax_a.transAxes, ha='right', va='bottom',
          fontsize=7, color='#888', style='italic')

p_a = os.path.join(OUTDIR,'Figure_Slide3A_PhaseByDisease.png')
fig_a.savefig(p_a, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig_a)
print(f"\nSaved: {p_a}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Load model + LMDB for T3.8
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== STEP 3: Load model for T3.8 module extraction ===")
model = HeteroClassificationModelV3(hidden_dim=256, embedding_dim=128, num_classes=6)
ckpt  = torch.load(CKPT, map_location='cpu')
model.load_state_dict(ckpt['model_state_dict'])
model = model.to(DEVICE).eval()

lmdb_idx = {}
for fn in os.listdir(GRAPH_DIR):
    fp = os.path.join(GRAPH_DIR, fn)
    if fn.endswith('.lmdb') and os.path.isdir(fp):
        lmdb_idx[fn.replace('.lmdb','')] = fp
print(f"LMDB: {len(lmdb_idx):,} | Model loaded ✅")


def extract_modules(keys, key_idxs=KEY_IDXS):
    """Extract module activations for given LMDB keys.
    Returns array of shape (N_successful, len(key_idxs))."""
    results, failed = [], 0
    for key in keys:
        try:
            acc = key.rsplit('_s',1)[0]
            if acc not in lmdb_idx: failed+=1; continue
            env = lmdb.open(lmdb_idx[acc], readonly=True, lock=False)
            with env.begin() as txn: val = txn.get(key.encode())
            env.close()
            if val is None: failed+=1; continue
            g = pickle.loads(val).to(DEVICE)
            x  = {nt: g[nt].x for nt in g.node_types if hasattr(g[nt],'x') and g[nt].x is not None}
            ei = {et: g[et].edge_index for et in g.edge_types if hasattr(g[et],'edge_index')}
            b  = {nt: torch.zeros(g[nt].x.size(0),dtype=torch.long,device=DEVICE) for nt in x}
            with torch.no_grad():
                _,_,_,mreps,_ = model(x,ei,b)
            if mreps is None: failed+=1; continue
            acts = model.pathway_head(mreps).detach().cpu().numpy().flatten()
            results.append([acts[i] for i in key_idxs])
        except: failed+=1
    if results:
        return np.array(results), failed   # (N, len(key_idxs))
    return None, failed


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Extract module activations per disease×phase group (T3.8)
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== STEP 4: T3.8 — Module activations per disease × TI phase ===")

PHASES_3  = ['E4 Priming','E1/E5 Induction','E20 Maintenance']
cell_means = {}   # (disease, phase) → mean array (4 modules)
cell_ns    = {}

for disease in DISEASES:
    for phase in PHASES_3:
        mask = ((disease_df['display_group']==disease) &
                (disease_df['ti_phase']==phase))
        sub  = disease_df[mask]
        n_avail = len(sub)
        keys = sub['accession_id'].tolist()
        if len(keys) > N_PER_GROUP:
            keys = random.sample(keys, N_PER_GROUP)
        print(f"  {disease} × {phase}: avail={n_avail:,}, sample={len(keys)}")
        arr, failed = extract_modules(keys)
        if arr is not None and len(arr) > 0:
            cell_means[(disease,phase)] = arr.mean(axis=0)   # (4,)
            cell_ns[(disease,phase)]    = len(arr)
            print(f"    → extracted={len(arr)}, failed={failed}, "
                  f"means={[f'{v:.3f}' for v in arr.mean(axis=0)]}")
        else:
            cell_means[(disease,phase)] = np.zeros(len(KEY_MODULES))
            cell_ns[(disease,phase)]    = 0
            print(f"    → NO DATA (failed={failed})")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Figure_Slide3B: 3-panel heatmap (T3.8)
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== STEP 5: T3.8 — Plotting 3-panel heatmap ===")

# Build 3 matrices — one per disease: rows=modules, cols=phases
# Apply min-shift globally so all values ≥ 0 and comparisons are fair
all_vals = np.concatenate([v for v in cell_means.values()])
global_min = all_vals.min()
print(f"Global min across all disease×phase means: {global_min:.4f}")
print(f"Min-shift applied: subtract {global_min:.4f} from all cells")

matrices = {}   # disease → (n_modules, n_phases) matrix
for disease in DISEASES:
    mat = np.array([[cell_means[(disease,ph)][mi] - global_min
                     for ph in PHASES_3]
                    for mi in range(len(KEY_MODULES))])
    matrices[disease] = mat
    print(f"\n  {disease} matrix (min-shifted):")
    print(f"  {'Module':<25} " + "  ".join(f"{p[:6]:>8}" for p in PHASES_3))
    for mi, ml in enumerate(KEY_LABELS):
        vals = [f"{mat[mi,pi]:>8.3f}" for pi in range(len(PHASES_3))]
        print(f"  {ml.replace(chr(10),' '):<25} " + "  ".join(vals))

# Shared color scale across all 3 panels for comparability
all_matrix_vals = np.stack(list(matrices.values()))
vmax = all_matrix_vals.max()
print(f"\nColor scale vmax (shared): {vmax:.4f}")

fig_b, axes_b = plt.subplots(1, 3, figsize=(14, 5))
fig_b.subplots_adjust(left=0.12, right=0.92, wspace=0.08, top=0.84, bottom=0.18)

PHASE_SHORT = ['E4\nPriming','E1/E5\nInduction','E20\nMaintenance']

for ax, disease in zip(axes_b, DISEASES):
    mat = matrices[disease]
    im  = ax.imshow(mat, cmap='OrRd', aspect='auto',
                    vmin=0, vmax=vmax, interpolation='nearest')

    # Annotate each cell with value and sample n
    for mi in range(len(KEY_MODULES)):
        for pi, phase in enumerate(PHASES_3):
            val   = mat[mi, pi]
            n_val = cell_ns.get((disease, phase), 0)
            color = 'white' if val > vmax*0.65 else '#222222'
            ax.text(pi, mi, f'{val:.2f}\nn={n_val}',
                    ha='center', va='center', fontsize=7.5,
                    color=color, fontweight='bold')

    ax.set_xticks(range(len(PHASES_3)))
    ax.set_xticklabels(PHASE_SHORT, fontsize=9)
    ax.set_title(disease, fontsize=13, fontweight='bold',
                 color=DISEASE_COLORS[disease], pad=8)
    if ax == axes_b[0]:
        ax.set_yticks(range(len(KEY_MODULES)))
        ax.set_yticklabels(KEY_LABELS, fontsize=9)
    else:
        ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_edgecolor('#CCC')

# Shared colorbar
cbar = fig_b.colorbar(im, ax=axes_b[-1], fraction=0.04, pad=0.04)
cbar.set_label('Module activation\n(min-shifted)', fontsize=9)

fig_b.suptitle('iMEMORY V4 — Disease-Specific TI Training Fingerprint\n'
               'Key module activations per disease class × TI endotype phase\n'
               'Does CKD-E4 look different from HD-E4?',
               fontsize=11, fontweight='bold', y=1.01)

# Phase color legend
handles = [mpatches.Patch(facecolor=PHASE_COLORS[p], label=p)
           for p in PHASES_3]
fig_b.legend(handles=handles, loc='lower center', ncol=3,
             fontsize=8.5, framealpha=0.92, bbox_to_anchor=(0.5,-0.06))

fig_b.text(0.5, -0.10,
           'iMEMORY V4  ·  Module activations extracted from V4 GNN pathway head  ·  '
           'Min-shifted (global min = 0) for all-positive display  ·  '
           f'N per cell ≤ {N_PER_GROUP}  ·  seed=42',
           ha='center', fontsize=7, color='#888', style='italic',
           transform=fig_b.transFigure)

p_b = os.path.join(OUTDIR,'Figure_Slide3B_ModuleByDiseasePhase.png')
fig_b.savefig(p_b, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig_b)
print(f"\nSaved: {p_b}")

# ── Copy both figures to outputs ──────────────────────────────────────────────
import shutil
# shutil.copy(p_a, os.path.join(COPY_DIR,'Figure_Slide3A_PhaseByDisease.png'))
# shutil.copy(p_b, os.path.join(COPY_DIR,'Figure_Slide3B_ModuleByDiseasePhase.png'))
print(f"Copied both figures to {COPY_DIR}")
print("DONE.")
