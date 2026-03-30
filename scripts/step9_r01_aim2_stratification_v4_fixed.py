#!/usr/bin/env python3
"""
step9_r01_metadata_stratification.py — iMEMORY V4 R01 GEO Metadata Stratification
Tasks 4.8, 4.9, 4.10

Three separate analyses that all require parsing GEO accession metadata:

ANALYSIS A (4.8): CKD stage 3-5 stratification
  R01 Aim 2A: 100 CKD patients stratified by eGFR (stages 3/4/5).
  Approach: scan GEO series accession IDs for CKD samples, fetch GSE metadata
  from NCBI to find stage/eGFR labels. Map to trained_immunity_module scores.
  If stage metadata unavailable: use CKD cluster IDs as proxy
  (cluster 242=mixed stages, cluster 354=diabetic nephropathy/advanced,
   cluster 267=ADPKD/milder).

ANALYSIS B (4.9): HD pre-dialysis vs post-dialysis chromatin accessibility
  R01 Aim 2B: serial HD samples (pre-HD, 3 sessions, 6 months).
  Approach: HD_ESRD_early (cluster 262) vs HD_ESRD_late (cluster 344) +
  HD_Pre_post_dialysis (cluster 394). Extract epigenetic_remodeling and
  NF_kB_epigenetic module scores per cluster as proxy for chromatin state.
  n is small (31 chromatin graphs) but all 108 HD cluster samples have
  gene_expression nodes and valid module scores.

ANALYSIS C (4.10): Preemptive KTx vs post-HD KTx
  R01 Aim 2C: preemptive transplant (no prior HD) vs HD-exposed transplant.
  Approach: scan KTx accession IDs for GEO series that specifically compare
  preemptive vs dialysis-exposed recipients. Map trained_immunity_module.
  Proxy if metadata unavailable: KTx stable clusters (387,361) =
  well-functioning grafts (likely include preemptive), rejection clusters
  (236,264) = likely HD-exposed (worse outcomes expected with prior HD).
"""
import os, sys, lmdb, pickle, numpy as np, pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random, pathlib, json
from collections import defaultdict
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
N_SAMPLES = 150

MODULE_NAMES = [
    'mTOR_glycolytic','NF_kB_epigenetic','trained_immunity_module',
    'inflammation_innate','inflammation_adaptive','metabolism_glycolysis',
    'metabolism_oxidative','stress_oxidative','stress_er','mTOR_signaling',
    'cytokine_response','epigenetic_remodeling','immune_memory','antigen_presentation'
]
TI_IDX  = MODULE_NAMES.index('trained_immunity_module')
EPI_IDX = MODULE_NAMES.index('epigenetic_remodeling')
NFB_IDX = MODULE_NAMES.index('NF_kB_epigenetic')

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

# CKD cluster → disease subtype proxy for staging
CKD_CLUSTER_STAGE_MAP = {
    267: 'CKD Stage 3\n(ADPKD — slower progression)',
    350: 'CKD Stage 3-4\n(IgA nephropathy)',
     80: 'CKD Stage 3-4\n(Glomerular disease)',
    242: 'CKD Mixed\n(Glomerulonephritis)',
    354: 'CKD Stage 4-5\n(Diabetic nephropathy)',
     12: 'CKD Stage 5\n(Peripheral blood ESRD)',
}
# HD cluster → dialysis stage proxy
HD_CLUSTER_STAGE_MAP = {
    262: 'HD early\n(ESRD initiation)',
    344: 'HD chronic\n(late ESRD)',
    394: 'HD post-dialysis\n(pre/post session)',
    429: 'HD uremia\n(established HD)',
}
# KTx cluster → graft outcome proxy for preemptive vs HD-exposed
KTX_CLUSTER_TYPE_MAP = {
    387: 'KTx stable\n(CNI, likely any)',
    361: 'KTx long-term\n(>1yr stable)',
    347: 'KTx CNI-min\n(protocol biopsy)',
    264: 'KTx chronic rej\n(HD-exposed likely)',
    104: 'KTx IF/TA\n(HD-exposed likely)',
    236: 'KTx acute rej\n(worst outcome)',
}

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


def extract_modules_multi(keys, idxs):
    """Extract multiple module activations per sample. Returns (N, len(idxs)) array."""
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
            results.append([acts[i] for i in idxs])
        except: failed+=1
    return (np.array(results) if results else None), failed


def skeys(df_sub, n=N_SAMPLES):
    keys = df_sub['accession_id'].tolist()
    return random.sample(keys, min(n, len(keys)))


plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                     'savefig.dpi':300,'savefig.bbox':'tight','savefig.pad_inches':0.15})

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.subplots_adjust(left=0.07, right=0.97, wspace=0.45, top=0.84, bottom=0.28)


# ── ANALYSIS A: CKD staging proxy (4.8) ──────────────────────────────────────
print("\n=== ANALYSIS A: CKD stage stratification (cluster proxy) ===")
ax_a = axes[0]
ckd_data = []
for cid, label in CKD_CLUSTER_STAGE_MAP.items():
    sub  = endo[(endo['disease_class']==2) & (endo['refined_endotype']==cid)]
    keys = skeys(sub)
    print(f"  Cluster {cid} '{label.split(chr(10))[0]}': avail={len(sub):,}, sample={len(keys)}")
    arr, failed = extract_modules_multi(keys, [TI_IDX])
    if arr is None: print("    → NO DATA"); continue
    mean = arr[:,0].mean()
    sem  = arr[:,0].std() / np.sqrt(len(arr))
    ckd_data.append({'label':label,'mean':mean,'sem':sem,'n':len(arr),'cid':cid})
    print(f"    → n={len(arr)}, TI_mean={mean:.4f}")

if ckd_data:
    shift_a = min(d['mean'] for d in ckd_data)
    x_a = np.arange(len(ckd_data))
    colors_a = ['#FFCCCC','#FF9999','#FF6666','#CC3333','#990000','#660000'][:len(ckd_data)]
    bars_a = ax_a.bar(x_a, [d['mean']-shift_a for d in ckd_data], 0.55,
                      color=colors_a, alpha=0.88,
                      yerr=[d['sem'] for d in ckd_data], capsize=5,
                      error_kw={'lw':1.5,'ecolor':'#333'})
    for i,(d) in enumerate(ckd_data):
        ax_a.text(i, (d['mean']-shift_a)+d['sem']+0.01,
                  f"n={d['n']}", ha='center', va='bottom', fontsize=7.5, color='#444')
    ax_a.set_xticks(x_a)
    ax_a.set_xticklabels([d['label'] for d in ckd_data], fontsize=7.5, rotation=0)
    ax_a.set_ylim(bottom=0)
    ax_a.set_title('A   CKD Subtype Stratification (Aim 2A proxy)\n'
                   'TI activation by CKD disease subtype cluster\n'
                   '(darker = more advanced/uremic CKD)',
                   fontsize=9.5, fontweight='bold', pad=6, loc='left')
    ax_a.set_ylabel('TI module activation\n(min-shifted)', fontsize=9.5)
    for sp in ax_a.spines.values(): sp.set_edgecolor('#CCC')
    ax_a.yaxis.grid(True, color='#EEE', lw=0.7)
    ax_a.set_axisbelow(True)
    ax_a.text(0.99, 0.01,
              'Cluster IDs proxy CKD severity\n(stage metadata pending GEO parse)',
              transform=ax_a.transAxes, ha='right', va='bottom',
              fontsize=6.5, color='#888', style='italic')


# ── ANALYSIS B: HD pre vs post-dialysis epigenetic proxy (4.9) ───────────────
print("\n=== ANALYSIS B: HD dialysis stage epigenetic proxy ===")
ax_b = axes[1]
hd_data = []
hd_colors = ['#ADD8E6','#5C8DD6','#1565C0','#0D47A1']
for cid, label in HD_CLUSTER_STAGE_MAP.items():
    sub  = endo[(endo['disease_class']==3) & (endo['refined_endotype']==cid)]
    keys = skeys(sub)
    print(f"  Cluster {cid} '{label.split(chr(10))[0]}': avail={len(sub):,}, sample={len(keys)}")
    arr, failed = extract_modules_multi(keys, [TI_IDX, EPI_IDX, NFB_IDX])
    if arr is None: print("    → NO DATA"); continue
    means = arr.mean(axis=0)
    sems  = arr.std(axis=0) / np.sqrt(len(arr))
    hd_data.append({'label':label,'ti':means[0],'epi':means[1],'nfb':means[2],
                    'ti_sem':sems[0],'epi_sem':sems[1],'nfb_sem':sems[2],'n':len(arr)})
    print(f"    → n={len(arr)}, TI={means[0]:.4f}, EPI={means[1]:.4f}, NFkB={means[2]:.4f}")

if hd_data:
    # Show TI and epigenetic_remodeling side by side per HD stage
    x_b  = np.arange(len(hd_data))
    w    = 0.25
    shift_ti  = min(d['ti']  for d in hd_data)
    shift_epi = min(d['epi'] for d in hd_data)
    shift_nfb = min(d['nfb'] for d in hd_data)

    b1 = ax_b.bar(x_b - w, [d['ti']-shift_ti   for d in hd_data], w,
                  color='#6A1B9A', alpha=0.85, label='TI module',
                  yerr=[d['ti_sem'] for d in hd_data], capsize=3,
                  error_kw={'lw':1.2,'ecolor':'#333'})
    b2 = ax_b.bar(x_b,     [d['epi']-shift_epi  for d in hd_data], w,
                  color='#4CC9F0', alpha=0.85, label='Epigenetic remodeling',
                  yerr=[d['epi_sem'] for d in hd_data], capsize=3,
                  error_kw={'lw':1.2,'ecolor':'#333'})
    b3 = ax_b.bar(x_b + w, [d['nfb']-shift_nfb  for d in hd_data], w,
                  color='#F15BB5', alpha=0.85, label='NF-κB/epigenetic',
                  yerr=[d['nfb_sem'] for d in hd_data], capsize=3,
                  error_kw={'lw':1.2,'ecolor':'#333'})

    for i, d in enumerate(hd_data):
        ax_b.text(i, max(d['ti']-shift_ti, d['epi']-shift_epi, d['nfb']-shift_nfb)
                  + 0.02, f"n={d['n']}", ha='center', va='bottom', fontsize=7.5, color='#444')

    ax_b.set_xticks(x_b)
    ax_b.set_xticklabels([d['label'] for d in hd_data], fontsize=8)
    ax_b.set_ylim(bottom=0)
    ax_b.set_title('B   HD Dialysis Stage: TI + Epigenetic + NF-κB (Aim 2B)\n'
                   'Pre-dialysis → early → late → post-session',
                   fontsize=9.5, fontweight='bold', pad=6, loc='left')
    ax_b.set_ylabel('Module activation\n(min-shifted per module)', fontsize=9.5)
    ax_b.legend(fontsize=8, framealpha=0.9, loc='upper left')
    for sp in ax_b.spines.values(): sp.set_edgecolor('#CCC')
    ax_b.yaxis.grid(True, color='#EEE', lw=0.7)
    ax_b.set_axisbelow(True)


# ── ANALYSIS C: KTx preemptive vs HD-exposed proxy (4.10) ────────────────────
print("\n=== ANALYSIS C: KTx outcome cluster as preemptive vs HD-exposed proxy ===")
ax_c = axes[2]
ktx_data = []
ktx_colors = ['#C8E6C9','#81C784','#388E3C','#E65100','#F9A825','#B71C1C']
for cid, label in KTX_CLUSTER_TYPE_MAP.items():
    sub  = endo[endo['refined_endotype']==cid]
    sub  = sub[sub['disease_class'].isin([0,1,4])]  # rejection + KTx classes
    keys = skeys(sub)
    print(f"  Cluster {cid}: avail={len(sub):,}, sample={len(keys)}")
    arr, failed = extract_modules_multi(keys, [TI_IDX])
    if arr is None: print("    → NO DATA"); continue
    mean = arr[:,0].mean()
    sem  = arr[:,0].std() / np.sqrt(len(arr))
    ktx_data.append({'label':label,'mean':mean,'sem':sem,'n':len(arr)})
    print(f"    → n={len(arr)}, TI_mean={mean:.4f}")

if ktx_data:
    shift_c = min(d['mean'] for d in ktx_data)
    x_c     = np.arange(len(ktx_data))
    bars_c  = ax_c.bar(x_c, [d['mean']-shift_c for d in ktx_data], 0.55,
                       color=ktx_colors[:len(ktx_data)], alpha=0.88,
                       yerr=[d['sem'] for d in ktx_data], capsize=5,
                       error_kw={'lw':1.5,'ecolor':'#333'})
    for i, d in enumerate(ktx_data):
        ax_c.text(i, (d['mean']-shift_c)+d['sem']+0.01,
                  f"n={d['n']}", ha='center', va='bottom', fontsize=7.5, color='#444')

    # Bracket: stable (preemptive proxy) vs rejection (HD-exposed proxy)
    stable_i   = [0,1,2]
    rejected_i = [3,4,5]
    y_top = max([(ktx_data[i]['mean']-shift_c)+ktx_data[i]['sem']
                  for i in range(len(ktx_data))]) * 1.12
    ax_c.annotate('', xy=(stable_i[-1]+0.3, y_top),
                  xytext=(stable_i[0]-0.3, y_top),
                  arrowprops=dict(arrowstyle='-', color='#2E7D32', lw=1.8))
    ax_c.text(np.mean(stable_i), y_top*1.03, 'Likely preemptive\n(good outcome)',
              ha='center', va='bottom', fontsize=7.5, color='#2E7D32', fontweight='bold')
    ax_c.annotate('', xy=(rejected_i[-1]+0.3, y_top),
                  xytext=(rejected_i[0]-0.3, y_top),
                  arrowprops=dict(arrowstyle='-', color='#B71C1C', lw=1.8))
    ax_c.text(np.mean(rejected_i), y_top*1.03, 'Likely HD-exposed\n(rejection/poor outcome)',
              ha='center', va='bottom', fontsize=7.5, color='#B71C1C', fontweight='bold')

    ax_c.set_xticks(x_c)
    ax_c.set_xticklabels([d['label'] for d in ktx_data], fontsize=7.5)
    ax_c.set_ylim(bottom=0)
    ax_c.set_title('C   KTx Graft Outcome as Preemptive vs HD-exposed Proxy (Aim 2C)\n'
                   'Stable graft=preemptive likely | Rejection=HD-exposed likely',
                   fontsize=9.5, fontweight='bold', pad=6, loc='left')
    ax_c.set_ylabel('TI module activation\n(min-shifted)', fontsize=9.5)
    for sp in ax_c.spines.values(): sp.set_edgecolor('#CCC')
    ax_c.yaxis.grid(True, color='#EEE', lw=0.7)
    ax_c.set_axisbelow(True)
    ax_c.text(0.99, 0.01,
              'Proxy analysis — explicit preemptive/HD metadata\npending GEO series parse (see step9_metadata_notes)',
              transform=ax_c.transAxes, ha='right', va='bottom',
              fontsize=6, color='#888', style='italic')

fig.text(0.5, 0.01,
         'iMEMORY V4  ·  Cluster-based proxy stratification for R01 Aims 2A/2B/2C  ·  '
         'All bars min-shifted  ·  Error bars = SEM  ·  seed=42  ·  '
         'Note: explicit CKD stage / preemptive vs HD metadata requires GEO series parsing (V4)',
         ha='center', fontsize=6.0, color='#888', style='italic')

out_path = os.path.join(OUTDIR,'Figure_R01_AimTwo_Stratification.png')
fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"\nSaved: {out_path}")
import shutil
# shutil.copy(out_path, os.path.join(COPY_DIR,'Figure_R01_AimTwo_Stratification.png'))
print("DONE.")
