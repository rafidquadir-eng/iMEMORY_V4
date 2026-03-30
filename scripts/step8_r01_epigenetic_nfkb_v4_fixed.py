#!/usr/bin/env python3
"""
step8_r01_epigenetic_nfkb.py — iMEMORY V4 R01 Support: Epigenetic + NF-kB Progression
Tasks 4.6, 4.7, 4.12

OUTPUT: Figure_R01_Epigenetic_NF-kB_Progression.png
  Two-panel bar chart showing progressive module activation across disease stages.

  Panel A: epigenetic_remodeling module (proxy for H3K4me3 at TNF/IL-6 promoters)
    R01 Aim 1A / Aim 2: "epigenetic reprogramming pathways associated with
    functional training." CKD→HD→KTx progression should show increasing
    epigenetic_remodeling consistent with cumulative DAMP-driven chromatin
    remodeling.

  Panel B: NF_kB_epigenetic module (NF-kB as trained immunity mechanism)
    R01 Aim 1A: "inflammatory and fibrotic pathways." NF-kB is the canonical
    transcription factor linking mTOR activation to trained macrophage
    inflammatory gene expression. Should increase CKD→HD→KTx.

  Both panels use:
    - Same disease groups and HD split as step6 (HD early=cluster 262,
      HD chronic=clusters 344+394) for consistency
    - Min-shift so all bars ≥ 0
    - TI(E20) as reference ceiling
    - Same LMDB extraction as step6 (N=200 per group, seed=42)
"""
import os, sys, lmdb, pickle, numpy as np, pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random, pathlib
random.seed(42)
np.random.seed(42)

sys.path.insert(0, '/mnt/data/trained_im/scripts')
from hetero_encoder_v4 import HeteroClassificationModelV3

ENDO_PATH = '/mnt/nvme15tb/trained_im/outputs/V4/stage3/endotype_assignments_v4_final.csv'
CKPT      = '/mnt/nvme15tb/trained_im/checkpoints/phase2_v4/phase2_v4_best.pt'
GRAPH_DIR = '/mnt/nvme15tb/trained_im/graphs_lmdb_v4'
OUTDIR    = '/mnt/nvme15tb/trained_im/outputs/V4/figures/'
COPY_DIR  = '/mnt/user-data/outputs/'
pathlib.Path(OUTDIR).mkdir(parents=True, exist_ok=True)
# COPY_DIR disabled
DEVICE    = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
N_SAMPLES = 200

MODULE_NAMES = [
    'mTOR_glycolytic','NF_kB_epigenetic','trained_immunity_module',
    'inflammation_innate','inflammation_adaptive','metabolism_glycolysis',
    'metabolism_oxidative','stress_oxidative','stress_er','mTOR_signaling',
    'cytokine_response','epigenetic_remodeling','immune_memory','antigen_presentation'
]
TARGET_MODULES = {
    'epigenetic_remodeling': {
        'idx': MODULE_NAMES.index('epigenetic_remodeling'),
        'color_base': '#4CC9F0',
        'title': 'A   Epigenetic Remodeling Module Progression\n'
                 'Proxy for H3K4me3 at TNF/IL-6 promoters (R01 Aim 1A/2)',
        'ylabel': 'Epigenetic remodeling activation\n(min-shifted, lowest = 0)',
    },
    'NF_kB_epigenetic': {
        'idx': MODULE_NAMES.index('NF_kB_epigenetic'),
        'color_base': '#F15BB5',
        'title': 'B   NF-κB / Epigenetic Module Progression\n'
                 'Canonical trained immunity transcription factor (R01 Aim 1A)',
        'ylabel': 'NF-κB/epigenetic activation\n(min-shifted, lowest = 0)',
    },
}

# Cluster IDs
CLEAN_TI = {
    515,516,10,525,526,532,533,534,535,537,538,539,542,543,544,548,549,550,551,
    552,558,559,560,579,587,588,590,591,594,596,597,598,599,600,601,602,603,604,
    605,606,607,609,610,611,612,613,614,615,616,617,618,619,245,247,293,305,315,
    317,318,411,421,426,432,438,443,447,451,456,465,466,470,471,472,476,479,485,
    486,488,489,490,491,494,495,496,497,498,499,500,503,504,505,508
}
E1E5_CLUSTERS = {488,587,526,432,465,466,500,438,247,479}
E20_CLUSTERS  = CLEAN_TI - E1E5_CLUSTERS
HD_EARLY      = {262}
HD_CHRONIC    = {344, 394}
PURE_KTX      = {387,361,347,455,464}

print("Loading model and data...")
endo  = pd.read_csv(ENDO_PATH)
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


def extract_module(keys, module_idx):
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
            results.append(float(acts[module_idx]))
        except: failed+=1
    return results, failed


def skeys(df_sub, n=N_SAMPLES):
    keys = df_sub['accession_id'].tolist()
    return random.sample(keys, min(n, len(keys)))


# Disease groups — identical to step6
GROUPS = [
    ('CKD',           '#C62828', endo[endo['disease_class']==2]),
    ('HD\nearly',     '#5C8DD6', endo[(endo['disease_class']==3) &
                                       endo['refined_endotype'].isin(HD_EARLY)]),
    ('HD\nchronic',   '#1565C0', endo[(endo['disease_class']==3) &
                                       endo['refined_endotype'].isin(HD_CHRONIC)]),
    ('KTx\n(stable)', '#2E7D32', endo[(endo['disease_class']==4) &
                                       endo['refined_endotype'].isin(PURE_KTX)]),
    ('Chronic\nRej',  '#E65100', endo[endo['disease_class']==1]),
    ('Acute\nRej',    '#FF3D00', endo[endo['disease_class']==0]),
    ('TI\n(E20)',     '#6A1B9A', endo[(endo['disease_class']==5) &
                                       endo['refined_endotype'].isin(E20_CLUSTERS)]),
]

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                     'savefig.dpi':300,'savefig.bbox':'tight','savefig.pad_inches':0.15})

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.subplots_adjust(left=0.08, right=0.97, wspace=0.32, top=0.85, bottom=0.22)

for ax, (mod_name, meta) in zip(axes, TARGET_MODULES.items()):
    print(f"\n=== Extracting {mod_name} ===")
    mod_idx = meta['idx']

    raw_data = []
    for label, color, df_sub in GROUPS:
        keys = skeys(df_sub)
        print(f"  {label.replace(chr(10),' ')}: avail={len(df_sub):,}, sample={len(keys)}")
        vals, failed = extract_module(keys, mod_idx)
        if not vals: print("    → NO DATA"); continue
        arr = np.array(vals)
        sem = arr.std() / np.sqrt(len(arr))
        raw_data.append({'label':label,'color':color,
                          'mean':arr.mean(),'sem':sem,'n':len(arr)})
        print(f"    → n={len(arr)}, mean={arr.mean():.4f}, sem={sem:.4f}, failed={failed}")

    # Min-shift
    shift = min(d['mean'] for d in raw_data)
    data  = [{**d, 'shifted': d['mean']-shift} for d in raw_data]
    print(f"  Min-shift: {shift:.4f}")

    labels  = [d['label']   for d in data]
    shifted = [d['shifted'] for d in data]
    sems    = [d['sem']     for d in data]
    ns      = [d['n']       for d in data]
    colors  = [d['color']   for d in data]
    x       = np.arange(len(labels))

    ax.bar(x, shifted, 0.55, color=colors, alpha=0.88,
           yerr=sems, capsize=5, error_kw={'lw':1.5,'ecolor':'#333'})
    ax.axhline(0, color='#333', lw=0.8)

    # HD bracket
    hd_i = [i for i,l in enumerate(labels) if 'HD' in l]
    if len(hd_i)==2:
        y_top = max(shifted[hd_i[0]]+sems[hd_i[0]],
                    shifted[hd_i[1]]+sems[hd_i[1]]) * 1.15 + 0.01
        ax.annotate('', xy=(hd_i[1],y_top), xytext=(hd_i[0],y_top),
                    arrowprops=dict(arrowstyle='<->',color='#444',lw=1.2))
        delta = shifted[hd_i[1]] - shifted[hd_i[0]]
        ax.text(sum(hd_i)/2, y_top*1.02,
                f'HD progression\n{delta:+.2f}',
                ha='center', va='bottom', fontsize=7.5, color='#333')

    for i,(v,s,n) in enumerate(zip(shifted,sems,ns)):
        ax.text(i, v+s+max(sems)*0.08,
                f'n={n}', ha='center', va='bottom', fontsize=7.0, color='#444')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.0)
    ax.set_ylim(bottom=0)
    ax.set_ylabel(meta['ylabel'], fontsize=10)
    ax.set_title(meta['title'], fontsize=10.5, fontweight='bold', pad=8, loc='left')
    for sp in ax.spines.values(): sp.set_edgecolor('#CCC')
    ax.yaxis.grid(True, color='#EEE', lw=0.7)
    ax.set_axisbelow(True)

fig.text(0.5, 0.01,
         'iMEMORY V4  ·  Module activations from V4 GNN pathway head  ·  '
         'Min-shifted (lowest group = 0)  ·  '
         'HD chronic = clusters 344 (ESRD late) + 394 (pre/post-dialysis)  ·  '
         'Error bars = SEM  ·  seed=42',
         ha='center', fontsize=6.5, color='#888', style='italic')

out_path = os.path.join(OUTDIR,'Figure_R01_Epigenetic_NF-kB_Progression.png')
fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"\nSaved: {out_path}")
import shutil
# shutil.copy(out_path, os.path.join(COPY_DIR,'Figure_R01_Epigenetic_NF-kB_Progression.png'))
print("DONE.")
