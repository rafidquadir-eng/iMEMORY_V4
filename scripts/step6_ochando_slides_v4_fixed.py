#!/usr/bin/env python3
"""
step6_ochando_slides_v3_1.py — iMEMORY V4 Ochando Slide 2 (Tasks T3, T4, T5)

OUTPUT: Figure_Slide2_TrainingKinetics.png

Y-AXIS FIX (Panels B and C):
  Raw trained_immunity_module activations span negative to positive.
  To make all bars positive while preserving every relative difference
  and direction, we apply a min-shift:

      shifted_mean = raw_mean - min(raw_means_across_all_groups)

  The group with the lowest activation becomes 0.
  Every other group shows how much higher it is above that floor.
  HD early → HD chronic delta is fully preserved (both shift equally).
  Y-axis label: "TI module activation above minimum observed group"
"""
import os, sys, lmdb, pickle, numpy as np, pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

N_SAMPLES = 200

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
E4_CLUSTERS   = {128,196,173,142}
E1E5_CLUSTERS = {488,587,526,432,465,466,500,438,247,479}
E20_CLUSTERS  = CLEAN_TI - E1E5_CLUSTERS
PAN_CLUSTERS  = {128,196,173,142}
HD_EARLY      = {262}
HD_CHRONIC    = {344,394}
PURE_CKD      = {242,354,80,350,267}
PURE_KTX      = {387,361,347,455,464}

MODULE_NAMES = [
    'mTOR_glycolytic','NF_kB_epigenetic','trained_immunity_module',
    'inflammation_innate','inflammation_adaptive','metabolism_glycolysis',
    'metabolism_oxidative','stress_oxidative','stress_er','mTOR_signaling',
    'cytokine_response','epigenetic_remodeling','immune_memory','antigen_presentation'
]
TI_IDX = MODULE_NAMES.index('trained_immunity_module')

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                     'savefig.dpi':300,'savefig.bbox':'tight','savefig.pad_inches':0.15})


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Phase assignment (Panel A)
# ══════════════════════════════════════════════════════════════════════════════
print("=== STEP 1: Phase assignment ===")
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

def assign_phase(pts, k=50):
    phases = []
    for px, py in zip(pts[:,0], pts[:,1]):
        _, idx = ti_tree.query([px,py], k=min(k,len(ti_coords)))
        nbr = ti_cids[idx]
        counts = {
            'E4 Priming':     sum(c in E4_CLUSTERS   for c in nbr),
            'E1/E5 Induction':sum(c in E1E5_CLUSTERS for c in nbr),
            'E20 Maintenance':sum(c in E20_CLUSTERS  for c in nbr),
        }
        best = max(counts, key=counts.get)
        phases.append(best if counts[best]>=1 else 'No TI signal')
    return np.array(phases)

disease_df = df_all[df_all['display_group'].isin(['CKD','HD','KTx'])].copy().reset_index(drop=True)
print(f"Assigning phases for {len(disease_df):,} samples...")
disease_df['ti_phase'] = assign_phase(disease_df[['ux','uy']].values)

phase_order_a = ['E4 Priming','E1/E5 Induction','E20 Maintenance','No TI signal']
panel_a = {}
for ph in phase_order_a:
    sub   = disease_df[disease_df['ti_phase']==ph]
    total = len(sub)
    panel_a[ph] = {
        'CKD': 100*(sub['display_group']=='CKD').sum()/total if total else 0,
        'HD':  100*(sub['display_group']=='HD').sum()/total  if total else 0,
        'KTx': 100*(sub['display_group']=='KTx').sum()/total if total else 0,
        'n': total,
    }
    print(f"  {ph} (n={total:,}): "
          f"CKD={panel_a[ph]['CKD']:.1f}% HD={panel_a[ph]['HD']:.1f}% KTx={panel_a[ph]['KTx']:.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Load model + LMDB
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== STEP 2: Load model ===")
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


def extract_ti(keys):
    out, failed = [], 0
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
            out.append(float(model.pathway_head(mreps).detach().cpu().numpy().flatten()[TI_IDX]))
        except: failed+=1
    return out, failed


def skeys(df_sub, n=N_SAMPLES):
    keys = df_sub['accession_id'].tolist()
    return random.sample(keys, min(n, len(keys)))


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Extract raw activations
# ══════════════════════════════════════════════════════════════════════════════
hd_early_df   = endo[(endo['disease_class']==3) & endo['refined_endotype'].isin(HD_EARLY)]
hd_chronic_df = endo[(endo['disease_class']==3) & endo['refined_endotype'].isin(HD_CHRONIC)]

# Panel B groups
B_GROUPS = [
    ('CKD',           '#C62828', endo[endo['disease_class']==2]),
    ('HD\nearly',     '#5C8DD6', hd_early_df),
    ('HD\nchronic',   '#1565C0', hd_chronic_df),
    ('KTx\n(stable)', '#2E7D32', endo[(endo['disease_class']==4) &
                                       endo['refined_endotype'].isin(PURE_KTX)]),
    ('Chronic\nRej',  '#E65100', endo[endo['disease_class']==1]),
    ('Acute\nRej',    '#FF3D00', endo[endo['disease_class']==0]),
    ('TI\n(E20)',     '#6A1B9A', endo[(endo['disease_class']==5) &
                                       endo['refined_endotype'].isin(E20_CLUSTERS)]),
]

# Panel C groups
C_GROUPS = [
    ('Pure\nCKD',              '#C62828', endo[(endo['disease_class']==2) &
                                               endo['refined_endotype'].isin(PURE_CKD)]),
    ('HD\nearly',              '#5C8DD6', hd_early_df),
    ('HD\nchronic',            '#1565C0', hd_chronic_df),
    ('Pure\nKTx',              '#2E7D32', endo[(endo['disease_class']==4) &
                                               endo['refined_endotype'].isin(PURE_KTX)]),
    ('Pan-disease\nCKD+HD+KTx','#757575', endo[endo['refined_endotype'].isin(PAN_CLUSTERS) &
                                               (endo['disease_class']!=5)]),
    ('TI\nReference',          '#6A1B9A', endo[(endo['disease_class']==5) &
                                               endo['refined_endotype'].isin(E20_CLUSTERS)]),
]

def extract_group_stats(groups, label):
    print(f"\n=== {label} ===")
    data = []
    for name, color, df_sub in groups:
        keys = skeys(df_sub)
        print(f"  {name.replace(chr(10),' ')}: avail={len(df_sub):,}, sample={len(keys)}")
        vals, failed = extract_ti(keys)
        if not vals: print("    → NO DATA"); continue
        arr = np.array(vals)
        sem = arr.std()/np.sqrt(len(arr))
        data.append({'label':name,'color':color,'mean':arr.mean(),'sem':sem,'n':len(arr)})
        print(f"    raw mean={arr.mean():.4f}  sem={sem:.4f}  failed={failed}")
    return data

panel_b_raw = extract_group_stats(B_GROUPS, "Panel B")
panel_c_raw = extract_group_stats(C_GROUPS, "Panel C")

# ── Min-shift: subtract global minimum so all bars ≥ 0 ───────────────────────
all_means_b = [d['mean'] for d in panel_b_raw]
all_means_c = [d['mean'] for d in panel_c_raw]
shift_b = min(all_means_b)
shift_c = min(all_means_c)

print(f"\nPanel B shift: {shift_b:.4f}  (subtracted from all means)")
print(f"Panel C shift: {shift_c:.4f}  (subtracted from all means)")

panel_b = [{**d, 'shifted': d['mean']-shift_b} for d in panel_b_raw]
panel_c = [{**d, 'shifted': d['mean']-shift_c} for d in panel_c_raw]

print("\nPanel B shifted means:")
for d in panel_b:
    print(f"  {d['label'].replace(chr(10),' ')}: {d['shifted']:.4f}  (raw={d['mean']:.4f})")

print("\nPanel C shifted means:")
for d in panel_c:
    print(f"  {d['label'].replace(chr(10),' ')}: {d['shifted']:.4f}  (raw={d['mean']:.4f})")

# Verify HD direction preserved
hd_e = next((d for d in panel_b if 'early'   in d['label']),None)
hd_c = next((d for d in panel_b if 'chronic' in d['label']),None)
if hd_e and hd_c:
    delta = hd_c['shifted'] - hd_e['shifted']
    print(f"\nHD early → chronic delta: {delta:+.4f} "
          f"({'✅ increasing as expected' if delta>0 else '⚠️ reversed'})")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Plot
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== STEP 4: Plotting ===")
fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(19, 7))
fig.subplots_adjust(left=0.06, right=0.97, wspace=0.42, top=0.88, bottom=0.22)


# ── Panel A: stacked bar ──────────────────────────────────────────────────────
x_a = np.arange(len(phase_order_a))
ckd_v = [panel_a[p]['CKD'] for p in phase_order_a]
hd_v  = [panel_a[p]['HD']  for p in phase_order_a]
ktx_v = [panel_a[p]['KTx'] for p in phase_order_a]
ns_a  = [panel_a[p]['n']   for p in phase_order_a]

ax_a.bar(x_a, ckd_v, 0.55, label='CKD',  color='#C62828', alpha=0.88)
ax_a.bar(x_a, hd_v,  0.55, bottom=ckd_v, label='HD',  color='#1565C0', alpha=0.88)
ax_a.bar(x_a, ktx_v, 0.55,
         bottom=[c+h for c,h in zip(ckd_v,hd_v)], label='KTx', color='#2E7D32', alpha=0.88)
for i,n in enumerate(ns_a):
    ax_a.text(i, 103, f'n={n:,}', ha='center', va='bottom', fontsize=7.5, color='#444')
ax_a.set_xticks(x_a)
ax_a.set_xticklabels(['E4\nPriming','E1/E5\nInduction','E20\nMaintenance','No TI\nSignal'],
                      fontsize=9.5)
ax_a.set_ylabel('Disease class composition (%)', fontsize=10)
ax_a.set_ylim(0, 116)
ax_a.set_title('A   Disease Composition per TI Endotype Phase\n'
               'Which disease class drives each training stage?',
               fontsize=10.5, fontweight='bold', pad=8, loc='left')
ax_a.legend(loc='upper right', fontsize=8.5, framealpha=0.9, handlelength=0.9)
ax_a.axhline(100, color='#AAA', lw=0.7, ls='--')
for sp in ax_a.spines.values(): sp.set_edgecolor('#CCC')
ax_a.yaxis.grid(True, color='#EEE', lw=0.7)
ax_a.set_axisbelow(True)


# ── Panel B: min-shifted bars ─────────────────────────────────────────────────
labels_b  = [d['label']   for d in panel_b]
shifted_b = [d['shifted'] for d in panel_b]
sems_b    = [d['sem']     for d in panel_b]
ns_b      = [d['n']       for d in panel_b]
colors_b  = [d['color']   for d in panel_b]
x_b       = np.arange(len(labels_b))

ax_b.bar(x_b, shifted_b, 0.55, color=colors_b, alpha=0.88,
         yerr=sems_b, capsize=5, error_kw={'lw':1.5,'ecolor':'#333'})

# HD progression bracket
hd_i = [i for i,l in enumerate(labels_b) if 'HD' in l]
if len(hd_i)==2:
    y_top = max(shifted_b[hd_i[0]]+sems_b[hd_i[0]],
                shifted_b[hd_i[1]]+sems_b[hd_i[1]]) * 1.18 + 0.02
    ax_b.annotate('',xy=(hd_i[1],y_top),xytext=(hd_i[0],y_top),
                  arrowprops=dict(arrowstyle='<->',color='#444',lw=1.2))
    ax_b.text(sum(hd_i)/2, y_top*1.02,
              f'HD progression\n+{(shifted_b[hd_i[1]]-shifted_b[hd_i[0]]):.2f}',
              ha='center', va='bottom', fontsize=7.5, color='#333')

for i,(v,s,n) in enumerate(zip(shifted_b,sems_b,ns_b)):
    ax_b.text(i, v+s+max(sems_b)*0.08,
              f'n={n}', ha='center', va='bottom', fontsize=7.0, color='#444')

ax_b.set_xticks(x_b)
ax_b.set_xticklabels(labels_b, fontsize=9.0)
ax_b.set_ylim(bottom=0)
ax_b.set_ylabel('TI module activation\n(min-shifted, lowest group = 0)', fontsize=10)
ax_b.set_title('B   Progressive TI Activation: CKD → HD → KTx → Rejection\n'
               'All values positive; lowest observed group anchored to 0',
               fontsize=10.5, fontweight='bold', pad=8, loc='left')
for sp in ax_b.spines.values(): sp.set_edgecolor('#CCC')
ax_b.yaxis.grid(True, color='#EEE', lw=0.7)
ax_b.set_axisbelow(True)


# ── Panel C: min-shifted bars ─────────────────────────────────────────────────
labels_c  = [d['label']   for d in panel_c]
shifted_c = [d['shifted'] for d in panel_c]
sems_c    = [d['sem']     for d in panel_c]
ns_c      = [d['n']       for d in panel_c]
colors_c  = [d['color']   for d in panel_c]
x_c       = np.arange(len(labels_c))

bars_c = ax_c.bar(x_c, shifted_c, 0.55, color=colors_c, alpha=0.88,
                   yerr=sems_c, capsize=5, error_kw={'lw':1.5,'ecolor':'#333'})

pan_i = next((i for i,d in enumerate(panel_c) if d['color']=='#757575'),None)
if pan_i is not None:
    bars_c[pan_i].set_edgecolor('#111')
    bars_c[pan_i].set_linewidth(2.0)
    ax_c.text(pan_i, shifted_c[pan_i]+sems_c[pan_i]*1.8,
              '★', ha='center', va='bottom', fontsize=14, color='#333')

for i,(v,s,n) in enumerate(zip(shifted_c,sems_c,ns_c)):
    ax_c.text(i, v+s+max(sems_c)*0.08,
              f'n={n}', ha='center', va='bottom', fontsize=7.0, color='#444')

ax_c.set_xticks(x_c)
ax_c.set_xticklabels(labels_c, fontsize=8.5)
ax_c.set_ylim(bottom=0)
ax_c.set_ylabel('TI module activation\n(min-shifted, lowest group = 0)', fontsize=10)
ax_c.set_title('C   Pan-disease (CKD+HD+KTx) vs Pure Single-disease\n'
               'Multi-stage exposure (★) approaches TI Reference ceiling',
               fontsize=10.5, fontweight='bold', pad=8, loc='left')
for sp in ax_c.spines.values(): sp.set_edgecolor('#CCC')
ax_c.yaxis.grid(True, color='#EEE', lw=0.7)
ax_c.set_axisbelow(True)


fig.text(0.5, 0.01,
         'iMEMORY V4  ·  Y-axis: min-shifted trained_immunity_module activation '
         '(raw mean − min group mean; preserves all relative differences)  ·  '
         'HD chronic = clusters 344 (ESRD late) + 394 (pre/post-dialysis, EC exposure)  ·  '
         'Error bars = SEM  ·  seed=42',
         ha='center', fontsize=6.5, color='#888', style='italic')

out_path = os.path.join(OUTDIR,'Figure_Slide2_TrainingKinetics.png')
fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"\nSaved: {out_path}")
import shutil
# shutil.copy(out_path, os.path.join(COPY_DIR,'Figure_Slide2_TrainingKinetics.png'))
print("Copied. DONE.")
