#!/usr/bin/env python3
"""
step12_longitudinal_detection_profiling.py — iMEMORY V4
Longitudinal paired sample detection and trajectory profiling.

PURPOSE (R01 Aims 1A, 2B, 2C):
  R01 Aim 1A:  "longitudinally profile trained immunity at 0, 2, 4, and 6 weeks"
  R01 Aim 2B:  "PBMC at HD initiation, early after HD (3 sessions), chronic HD (6 months)"
  R01 Aim 2C:  "pre-transplant and at 1, 3, 6, and 12 months post-transplantation"

This script:
  1. Detects longitudinal/serial GEO studies within the LMDB dataset by parsing
     accession_id patterns. A GEO series (GSE prefix) that contributes samples
     to both 'early' and 'late' disease clusters is flagged as potentially
     longitudinal.
  2. Identifies specifically:
     - HD longitudinal: GSE studies with samples in BOTH HD_ESRD_early (cluster 262)
       AND HD_ESRD_late (cluster 344) — these are the best proxy for Aim 2B
       serial samples (pre-HD → chronic HD)
     - KTx longitudinal: GSE studies with samples in BOTH stable KTx clusters
       (387,361) AND rejection clusters (236,264) — proxy for pre/post-transplant
       trajectory
  3. For detected longitudinal series: extract trained_immunity_module, mTOR_glycolytic,
     epigenetic_remodeling, and NF_kB_epigenetic at each cluster (time proxy).
  4. Generates trajectory line plots per study if ≥2 timepoints detected.
  5. Saves: longitudinal_studies.csv, longitudinal_trajectories.csv,
     Figure_Longitudinal_HD_Trajectory.png,
     Figure_Longitudinal_KTx_Trajectory.png

NOTES:
  - True longitudinal pairing (same patient at multiple timepoints) requires
    sample-level metadata from GEO (GSM IDs with time labels). This script
    uses cluster-level GEO study membership as a proxy — if the same GSE
    contributes to both early and late clusters, it likely contains serial data.
  - For definitive pairing, metadata download from NCBI using esearch/efetch
    would be needed (flagged as V4 task T17/T18 in checklist).
"""
import os, sys, lmdb, pickle, numpy as np, pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import random, pathlib
random.seed(42)
np.random.seed(42)

sys.path.insert(0, '/mnt/data/trained_im/scripts')
from hetero_encoder_v4 import HeteroClassificationModelV3

ENDO_PATH = '/mnt/nvme15tb/trained_im/outputs/V4/stage3/endotype_assignments_v4_final.csv'
CKPT      = '/mnt/nvme15tb/trained_im/checkpoints/phase2_v4/phase2_v4_best.pt'
GRAPH_DIR = '/mnt/nvme15tb/trained_im/graphs_lmdb_v4'
OUT_ROOT  = '/mnt/nvme15tb/trained_im/outputs/V4/profiling/longitudinal/'
COPY_DIR  = '/mnt/user-data/outputs/'
pathlib.Path(OUT_ROOT).mkdir(parents=True, exist_ok=True)
# COPY_DIR disabled
DEVICE    = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
N_PER_CLUSTER = 100

MODULE_NAMES = [
    'mTOR_glycolytic','NF_kB_epigenetic','trained_immunity_module',
    'inflammation_innate','inflammation_adaptive','metabolism_glycolysis',
    'metabolism_oxidative','stress_oxidative','stress_er','mTOR_signaling',
    'cytokine_response','epigenetic_remodeling','immune_memory','antigen_presentation'
]
KEY_MODULE_IDXS = {mn: i for i,mn in enumerate(MODULE_NAMES)}
KEY_MODS = ['mTOR_glycolytic','trained_immunity_module','epigenetic_remodeling','NF_kB_epigenetic']

# Cluster → time proxy label for HD (Aim 2B)
HD_TIME_CLUSTERS = {
    262: {'label':'Pre-HD / ESRD early',   'order':0, 'color':'#ADD8E6'},
    429: {'label':'Established HD',        'order':1, 'color':'#5C8DD6'},
    344: {'label':'ESRD late (chronic HD)', 'order':2, 'color':'#1565C0'},
    394: {'label':'Post-dialysis session', 'order':3, 'color':'#0D47A1'},
}

# Cluster → time proxy label for KTx (Aim 2C)
KTX_TIME_CLUSTERS = {
    387: {'label':'KTx stable\n(long-term)',   'order':0, 'color':'#C8E6C9'},
    361: {'label':'KTx long-term\n(>1 yr)',    'order':1, 'color':'#81C784'},
    347: {'label':'KTx CNI-min\n(protocol)',   'order':2, 'color':'#388E3C'},
    417: {'label':'KTx chronic allograft',     'order':3, 'color':'#F9A825'},
    264: {'label':'Chronic rejection\nfibrosis','order':4, 'color':'#E65100'},
    104: {'label':'Chronic rejection\nIF/TA',  'order':5, 'color':'#BF360C'},
    236: {'label':'Acute rejection\nT-cell',   'order':6, 'color':'#B71C1C'},
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


def extract_key_modules(keys, n=N_PER_CLUSTER):
    if len(keys) > n: keys = random.sample(keys, n)
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
            results.append([acts[KEY_MODULE_IDXS[m]] for m in KEY_MODS])
        except: failed+=1
    return np.array(results) if results else None, failed


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Detect longitudinal GEO series
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== STEP 1: Detect longitudinal GEO series ===")

# Extract GSE prefix from accession_id: "GSE249006_s001234" → "GSE249006"
endo['gse_id'] = endo['accession_id'].str.extract(r'^(GSE\d+|GSM\d+|E-\w+-\d+|ST\d+)', expand=False)

# For each cluster of interest, collect GSE IDs
hd_gse_per_cluster = {}
for cid in HD_TIME_CLUSTERS:
    sub = endo[(endo['refined_endotype']==cid) & (endo['disease_class']==3)]
    hd_gse_per_cluster[cid] = set(sub['gse_id'].dropna().unique())
    print(f"  HD cluster {cid}: {len(sub):,} samples from {len(hd_gse_per_cluster[cid])} GSE series")

ktx_gse_per_cluster = {}
for cid in KTX_TIME_CLUSTERS:
    sub = endo[endo['refined_endotype']==cid]
    ktx_gse_per_cluster[cid] = set(sub['gse_id'].dropna().unique())
    print(f"  KTx cluster {cid}: {len(sub):,} samples from {len(ktx_gse_per_cluster[cid])} GSE series")

# Find GSE series appearing in ≥2 HD time clusters (= likely longitudinal)
all_hd_gse = defaultdict(list)
for cid, gses in hd_gse_per_cluster.items():
    for gse in gses:
        all_hd_gse[gse].append(cid)

longitudinal_hd  = {gse: clusters for gse, clusters in all_hd_gse.items() if len(clusters) >= 2}
longitudinal_ktx = {}
all_ktx_gse = defaultdict(list)
for cid, gses in ktx_gse_per_cluster.items():
    for gse in gses:
        all_ktx_gse[gse].append(cid)
longitudinal_ktx = {gse: clusters for gse, clusters in all_ktx_gse.items() if len(clusters) >= 2}

print(f"\nLongitudinal HD series (appearing in ≥2 HD clusters): {len(longitudinal_hd)}")
for gse, clusters in sorted(longitudinal_hd.items()):
    print(f"  {gse}: clusters {clusters}")

print(f"\nLongitudinal KTx series (appearing in ≥2 KTx clusters): {len(longitudinal_ktx)}")
for gse, clusters in sorted(longitudinal_ktx.items()):
    print(f"  {gse}: clusters {clusters}")

# Save detection results
det_rows = []
for gse, clusters in longitudinal_hd.items():
    det_rows.append({'gse':gse,'type':'HD','n_clusters':len(clusters),'clusters':str(clusters)})
for gse, clusters in longitudinal_ktx.items():
    det_rows.append({'gse':gse,'type':'KTx','n_clusters':len(clusters),'clusters':str(clusters)})
pd.DataFrame(det_rows).to_csv(os.path.join(OUT_ROOT,'longitudinal_studies.csv'), index=False)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Extract module activations along HD time axis (cluster proxy)
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== STEP 2: HD time-proxy trajectory ===")
hd_traj = []
for cid, meta in sorted(HD_TIME_CLUSTERS.items(), key=lambda x: x[1]['order']):
    sub  = endo[(endo['refined_endotype']==cid) & (endo['disease_class']==3)]
    keys = sub['accession_id'].tolist()
    print(f"  {meta['label']}: n={len(sub):,}")
    arr, failed = extract_key_modules(keys)
    if arr is not None:
        means = arr.mean(axis=0)
        sems  = arr.std(axis=0)/np.sqrt(len(arr))
        hd_traj.append({
            'cluster_id': cid, 'label': meta['label'],
            'order': meta['order'], 'color': meta['color'],
            'n': len(arr), 'failed': failed,
            **{f'{m}_mean': means[i] for i,m in enumerate(KEY_MODS)},
            **{f'{m}_sem':  sems[i]  for i,m in enumerate(KEY_MODS)},
        })
        print(f"    n={len(arr)}, TI={means[KEY_MODS.index('trained_immunity_module')]:.4f}")

df_hd_traj = pd.DataFrame(hd_traj).sort_values('order')


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Extract module activations along KTx time axis (cluster proxy)
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== STEP 3: KTx time-proxy trajectory ===")
ktx_traj = []
for cid, meta in sorted(KTX_TIME_CLUSTERS.items(), key=lambda x: x[1]['order']):
    sub  = endo[endo['refined_endotype']==cid]
    keys = sub['accession_id'].tolist()
    print(f"  {meta['label'].replace(chr(10),' ')}: n={len(sub):,}")
    arr, failed = extract_key_modules(keys)
    if arr is not None:
        means = arr.mean(axis=0)
        sems  = arr.std(axis=0)/np.sqrt(len(arr))
        ktx_traj.append({
            'cluster_id': cid, 'label': meta['label'],
            'order': meta['order'], 'color': meta['color'],
            'n': len(arr), 'failed': failed,
            **{f'{m}_mean': means[i] for i,m in enumerate(KEY_MODS)},
            **{f'{m}_sem':  sems[i]  for i,m in enumerate(KEY_MODS)},
        })
        print(f"    n={len(arr)}, TI={means[KEY_MODS.index('trained_immunity_module')]:.4f}")

df_ktx_traj = pd.DataFrame(ktx_traj).sort_values('order')

pd.concat([df_hd_traj.assign(cohort='HD'),
           df_ktx_traj.assign(cohort='KTx')]).to_csv(
    os.path.join(OUT_ROOT,'longitudinal_trajectories.csv'), index=False)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: Plot HD trajectory figure
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Plotting HD trajectory ===")
MOD_STYLES = {
    'trained_immunity_module': {'color':'#9B5DE5','lw':2.8,'ls':'-', 'marker':'s','ms':9},
    'mTOR_glycolytic':         {'color':'#FF6B35','lw':2.8,'ls':'-', 'marker':'o','ms':9},
    'epigenetic_remodeling':   {'color':'#4CC9F0','lw':1.6,'ls':'--','marker':'D','ms':6},
    'NF_kB_epigenetic':        {'color':'#F15BB5','lw':1.6,'ls':'--','marker':'^','ms':6},
}

def make_trajectory_fig(df_traj, title, xlabel, filename):
    if len(df_traj) == 0:
        print(f"  No data for {filename}")
        return
    fig, (ax_main, ax_bar) = plt.subplots(1, 2, figsize=(14, 5.5),
                                           gridspec_kw={'width_ratios':[2,1]})
    fig.subplots_adjust(wspace=0.38)

    # LEFT: trajectory line plot
    x = np.arange(len(df_traj))
    all_means = []
    for mod in KEY_MODS:
        col_m = f'{mod}_mean'
        col_s = f'{mod}_sem'
        if col_m not in df_traj.columns: continue
        means = df_traj[col_m].values
        sems  = df_traj[col_s].values if col_s in df_traj.columns else np.zeros(len(means))
        st    = MOD_STYLES[mod]
        label = mod.replace('_module','').replace('_',' ').title()
        ax_main.plot(x, means, color=st['color'], lw=st['lw'], ls=st['ls'],
                     marker=st['marker'], ms=st['ms'],
                     markerfacecolor=st['color'], markeredgecolor='white',
                     markeredgewidth=0.5, label=label, zorder=3)
        ax_main.fill_between(x, means-sems, means+sems, color=st['color'], alpha=0.12)
        all_means.extend(means)

    ax_main.set_xticks(x)
    ax_main.set_xticklabels(df_traj['label'].values, fontsize=8.5)
    ax_main.set_xlabel(xlabel, fontsize=10)
    ax_main.set_ylabel('Module activation (raw)', fontsize=10)
    ax_main.set_title(title, fontsize=11, fontweight='bold', pad=8)
    ax_main.axhline(0, color='#AAA', lw=0.8)
    ax_main.legend(fontsize=8.5, framealpha=0.9, loc='upper left')
    ax_main.grid(axis='y', color='#EEE', lw=0.7)
    ax_main.set_axisbelow(True)
    for sp in ax_main.spines.values(): sp.set_edgecolor('#CCC')

    # Add n= below x-axis
    for i, row in df_traj.iterrows():
        ax_main.text(row['order'], ax_main.get_ylim()[0] - abs(ax_main.get_ylim()[0])*0.12,
                     f"n={row['n']}", ha='center', va='top', fontsize=7, color='#666',
                     transform=ax_main.transData)

    # RIGHT: trained_immunity_module bar per timepoint
    col_ti = 'trained_immunity_module_mean'
    col_ts = 'trained_immunity_module_sem'
    if col_ti in df_traj.columns:
        shift = min(df_traj[col_ti])
        vals  = df_traj[col_ti].values - shift
        sems  = df_traj[col_ts].values if col_ts in df_traj.columns else np.zeros(len(vals))
        colors_bar = df_traj['color'].values if 'color' in df_traj.columns else ['#9B5DE5']*len(vals)
        ax_bar.bar(x, vals, 0.6, color=colors_bar, alpha=0.88,
                   yerr=sems, capsize=5, error_kw={'lw':1.5,'ecolor':'#333'})
        for i,(v,s,row) in enumerate(zip(vals,sems,df_traj.itertuples())):
            ax_bar.text(i, v+s+max(vals)*0.04, f'n={row.n}',
                        ha='center', va='bottom', fontsize=7, color='#444')
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(df_traj['label'].values, fontsize=7.5)
        ax_bar.set_ylim(bottom=0)
        ax_bar.set_ylabel('TI module (min-shifted)', fontsize=10)
        ax_bar.set_title('TI Activation\n(min-shifted)', fontsize=10, fontweight='bold')
        ax_bar.grid(axis='y', color='#EEE', lw=0.7)
        ax_bar.set_axisbelow(True)
        for sp in ax_bar.spines.values(): sp.set_edgecolor('#CCC')

    plt.tight_layout()
    p = os.path.join(OUT_ROOT, filename)
    fig.savefig(p, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    import shutil; # shutil.copy(p, os.path.join(COPY_DIR, filename))
    print(f"  Saved: {p}")


make_trajectory_fig(
    df_hd_traj,
    'iMEMORY V4 — HD Disease Stage Trajectory (Aim 2B proxy)\n'
    'Pre-dialysis ESRD → Established HD → Chronic HD → Post-session',
    'HD disease stage (cluster-based time proxy\nPre-HD → 3 sessions → 6 months)',
    'Figure_Longitudinal_HD_Trajectory.png'
)

make_trajectory_fig(
    df_ktx_traj,
    'iMEMORY V4 — KTx Graft Outcome Trajectory (Aim 2C proxy)\n'
    'Long-term stable → Allograft dysfunction → Rejection',
    'KTx graft state (cluster-based time proxy\nStable → Dysfunction → Rejection)',
    'Figure_Longitudinal_KTx_Trajectory.png'
)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: Per longitudinal GSE study — within-study trajectory (if any found)
# ══════════════════════════════════════════════════════════════════════════════
if longitudinal_hd:
    print(f"\n=== STEP 5: Within-study longitudinal trajectories (HD) ===")
    study_rows = []
    for gse, cids_in_study in longitudinal_hd.items():
        for cid in cids_in_study:
            if cid not in HD_TIME_CLUSTERS: continue
            meta = HD_TIME_CLUSTERS[cid]
            sub  = endo[(endo['gse_id']==gse) & (endo['refined_endotype']==cid) &
                        (endo['disease_class']==3)]
            if len(sub) == 0: continue
            keys = sub['accession_id'].tolist()
            arr, failed = extract_key_modules(keys, n=50)
            if arr is not None:
                means = arr.mean(axis=0)
                for i,m in enumerate(KEY_MODS):
                    study_rows.append({
                        'gse': gse, 'cluster_id': cid,
                        'time_label': meta['label'], 'order': meta['order'],
                        'module': m, 'mean': means[i], 'n': len(arr)
                    })
                print(f"  {gse} × cluster {cid}: n={len(arr)}")

    if study_rows:
        df_study = pd.DataFrame(study_rows)
        df_study.to_csv(os.path.join(OUT_ROOT,'within_study_trajectories.csv'), index=False)

        # Plot one figure per GSE with ≥2 timepoints
        for gse in df_study['gse'].unique():
            gse_df = df_study[df_study['gse']==gse]
            if gse_df['cluster_id'].nunique() < 2: continue

            fig, ax = plt.subplots(figsize=(8, 5))
            for mod in KEY_MODS:
                sub_m = gse_df[gse_df['module']==mod].sort_values('order')
                if len(sub_m) < 2: continue
                st = MOD_STYLES[mod]
                ax.plot(sub_m['order'], sub_m['mean'],
                        color=st['color'], lw=st['lw'], ls=st['ls'],
                        marker=st['marker'], ms=st['ms'],
                        markerfacecolor=st['color'], markeredgecolor='white',
                        label=mod.replace('_module','').replace('_',' ').title())
            ax.set_xticks(sorted(gse_df['order'].unique()))
            ax.set_xticklabels([gse_df[gse_df['order']==o]['time_label'].iloc[0]
                                 for o in sorted(gse_df['order'].unique())], fontsize=8.5)
            ax.set_xlabel('Disease stage / timepoint', fontsize=10)
            ax.set_ylabel('Module activation', fontsize=10)
            ax.set_title(f'Within-study trajectory: {gse}\n'
                         f'iMEMORY V4 (n samples per timepoint in figure)',
                         fontsize=11, fontweight='bold')
            ax.axhline(0, color='#AAA', lw=0.8)
            ax.legend(fontsize=8.5, framealpha=0.9)
            ax.grid(axis='y', color='#EEE', lw=0.7)
            ax.set_axisbelow(True)
            for sp in ax.spines.values(): sp.set_edgecolor('#CCC')
            plt.tight_layout()
            p = os.path.join(OUT_ROOT, f'Figure_Study_{gse}_Trajectory.png')
            fig.savefig(p, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            import shutil; # shutil.copy(p, os.path.join(COPY_DIR, f'Figure_Study_{gse}_Trajectory.png'))
            print(f"  Saved: Figure_Study_{gse}_Trajectory.png")

print(f"\n=== step12 DONE ===")
print(f"Outputs in {OUT_ROOT}")
print(f"Figures copied to {COPY_DIR}")
