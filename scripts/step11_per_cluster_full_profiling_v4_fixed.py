#!/usr/bin/env python3
"""
step11_per_cluster_full_profiling.py — iMEMORY V4
Comprehensive multimodal profiling per disease subcluster + TI phase overlay.

PURPOSE (R01 Aim 1A): "Characterize the profile of trained macrophage induced
by CKD, EC and KTx independently." Each of the 25 named HDBSCAN subclusters
receives a full multimodal profile across all available data modalities, then
each cluster's samples are stratified by TI endotype phase (E4/E1E5/E20/No TI)
to show the within-cluster training fingerprint.

CLUSTERS PROFILED:
  CKD (7): 242,354,80,350,267,313,12
  HD  (7): 429,407,262,344,337,394,395
  KTx (11): 236,452,509,264,104,417,387,361,347,455,464

MODALITIES EXTRACTED per cluster (up to N_MAX samples each):
  1. Module activations (14 modules) — gene expression transcriptomics
  2. Cell type node activations (12 canonical cell types)
  3. Methylation: mean_beta, hyper_frac (>0.7), hypo_frac (<0.3), n_cpg
  4. Chromatin accessibility: open_frac (signal>0.5), mean_signal, n_regions
  5. SNP variant burden: het_frac, hom_alt_frac, total_burden
  6. miRNA: mean expression, n_mirna_nodes
  7. Metabolomics: uremic_frac, mean_intensity (sparse, noted)

TI PHASE OVERLAY per cluster:
  Each sample assigned E4/E1E5/E20/No_TI by k-NN majority vote (step3 method).
  Phase distribution per cluster shown as stacked bar.
  Within each cluster × phase cell: mean trained_immunity_module activation.

OUTPUTS:
  /profiling_v4/per_cluster/
    cluster_full_profile.csv         — all modalities × all clusters
    cluster_ti_phase_distribution.csv — phase fractions per cluster
    cluster_phase_ti_activation.csv  — trained_immunity_module per cluster×phase
    Figure_PerCluster_ModuleHeatmap.png  — 14 modules × 25 clusters heatmap
    Figure_PerCluster_TIPhase.png        — TI phase distribution per cluster
    Figure_PerCluster_Epigenetic.png     — methylation + chromatin per cluster
    Figure_PerCluster_CellType.png       — cell type per cluster
    Figure_PerCluster_OtherModalities.png — SNP/miRNA/metabolomics
  Per-cluster mini-reports uploaded to Box profiling folder.
"""
import os, sys, lmdb, pickle, numpy as np, pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial import cKDTree
from collections import Counter, defaultdict
import random, pathlib
random.seed(42)
np.random.seed(42)

sys.path.insert(0, '/mnt/data/trained_im/scripts')
from hetero_encoder_v4 import HeteroClassificationModelV3

# ── Paths ─────────────────────────────────────────────────────────────────────
ENDO_PATH = '/mnt/nvme15tb/trained_im/outputs/V4/stage3/endotype_assignments_v4_final.csv'
CKPT      = '/mnt/nvme15tb/trained_im/checkpoints/phase2_v4/phase2_v4_best.pt'
GRAPH_DIR = '/mnt/nvme15tb/trained_im/graphs_lmdb_v4'
OUT_ROOT  = '/mnt/nvme15tb/trained_im/outputs/V4/profiling/per_cluster/'
COPY_DIR  = '/mnt/user-data/outputs/'
pathlib.Path(OUT_ROOT).mkdir(parents=True, exist_ok=True)
# COPY_DIR disabled
DEVICE    = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
N_MAX     = 150   # max samples per cluster for module extraction
N_MOD     = 100   # max for methylation/chromatin/SNP (slower)

# ── Module definitions ─────────────────────────────────────────────────────────
MODULE_NAMES = [
    'mTOR_glycolytic','NF_kB_epigenetic','trained_immunity_module',
    'inflammation_innate','inflammation_adaptive','metabolism_glycolysis',
    'metabolism_oxidative','stress_oxidative','stress_er','mTOR_signaling',
    'cytokine_response','epigenetic_remodeling','immune_memory','antigen_presentation'
]
TI_IDX = MODULE_NAMES.index('trained_immunity_module')

CANONICAL_CELL_TYPES = [
    'monocyte_classical','monocyte_nonclassical','macrophage_M1','macrophage_M2',
    'CD4_Tcell','CD8_Tcell','Treg','NK_cell',
    'B_cell','dendritic_cDC1','dendritic_cDC2','MAIT_cell',
]

# ── Named cluster definitions ─────────────────────────────────────────────────
NAMED_CLUSTERS = {
    # CKD
    242: {'name':'CKD_Glomerulonephritis',   'disease':'CKD', 'dc':[2], 'color':'#D32F2F'},
    354: {'name':'CKD_Diabetic_nephropathy', 'disease':'CKD', 'dc':[2], 'color':'#FF6F00'},
     80: {'name':'CKD_Glomerular_disease',   'disease':'CKD', 'dc':[2], 'color':'#C2185B'},
    350: {'name':'CKD_IgA_nephropathy',      'disease':'CKD', 'dc':[2], 'color':'#7B1FA2'},
    267: {'name':'CKD_ADPKD',                'disease':'CKD', 'dc':[2], 'color':'#1565C0'},
    313: {'name':'CKD_Focal_segmental_GS',   'disease':'CKD', 'dc':[2], 'color':'#00838F'},
     12: {'name':'CKD_Peripheral_blood',     'disease':'CKD', 'dc':[2], 'color':'#558B2F'},
    # HD
    429: {'name':'HD_Uremia_dialysis',          'disease':'HD', 'dc':[3], 'color':'#0288D1'},
    407: {'name':'HD_Innate_immune_activation', 'disease':'HD', 'dc':[3], 'color':'#FF8F00'},
    262: {'name':'HD_ESRD_early',               'disease':'HD', 'dc':[3], 'color':'#283593'},
    344: {'name':'HD_ESRD_late',                'disease':'HD', 'dc':[3], 'color':'#6A1B9A'},
    337: {'name':'HD_PBMC',                     'disease':'HD', 'dc':[3], 'color':'#00695C'},
    394: {'name':'HD_Pre_post_dialysis',        'disease':'HD', 'dc':[3], 'color':'#D84315'},
    395: {'name':'HD_ESRD_peripheral_blood',    'disease':'HD', 'dc':[3], 'color':'#2E7D32'},
    # KTx
    236: {'name':'KTx_Acute_rej_Tcell',       'disease':'KTx', 'dc':[0], 'color':'#B71C1C'},
    452: {'name':'KTx_Acute_rej_AbMed',       'disease':'KTx', 'dc':[0], 'color':'#FF3D00'},
    509: {'name':'KTx_Acute_rej_mixed',       'disease':'KTx', 'dc':[0], 'color':'#FF8A65'},
    264: {'name':'KTx_Chronic_fibrosis',      'disease':'KTx', 'dc':[1], 'color':'#E65100'},
    104: {'name':'KTx_Chronic_IFTA',          'disease':'KTx', 'dc':[1], 'color':'#F9A825'},
    417: {'name':'KTx_Chronic_allograft',     'disease':'KTx', 'dc':[1], 'color':'#FDD835'},
    387: {'name':'KTx_CNI_inhibitor',         'disease':'KTx', 'dc':[4], 'color':'#004D40'},
    361: {'name':'KTx_Long_term_stable',      'disease':'KTx', 'dc':[4], 'color':'#006064'},
    347: {'name':'KTx_CNI_minimization',      'disease':'KTx', 'dc':[4], 'color':'#4A148C'},
    455: {'name':'KTx_Graft_monitoring',      'disease':'KTx', 'dc':[4], 'color':'#1B5E20'},
    464: {'name':'KTx_Multi_center_biopsy',   'disease':'KTx', 'dc':[4], 'color':'#33691E'},
}

# TI cluster sets
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

# ══════════════════════════════════════════════════════════════════════════════
# SETUP: Load model, LMDB index, endotype CSV
# ══════════════════════════════════════════════════════════════════════════════
print("=== SETUP ===")
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

# ── Build TI k-NN tree for phase assignment ────────────────────────────────────
DISPLAY_MAP = {0:'KTx',1:'KTx',2:'CKD',3:'HD',4:'KTx'}
endo['display_group'] = endo['disease_class'].astype(int).map(DISPLAY_MAP)
ec = endo['refined_endotype'].astype(int)
ti = endo['disease_class'].astype(int) == 5
endo.loc[ti & ec.isin(CLEAN_TI),  'display_group'] = 'Control'
endo.loc[ti & ec.isin(MIXED_TI),  'display_group'] = 'TI_mixed'

ti_df     = endo[endo['display_group'].isin(['Control','TI_mixed'])].copy().reset_index(drop=True)
ti_coords = ti_df[['umap_x','umap_y']].values
ti_cids   = ti_df['refined_endotype'].values.astype(int)
ti_tree   = cKDTree(ti_coords)
print(f"TI tree: {len(ti_df):,} samples")

def assign_phase(umap_pts, k=50):
    phases = []
    for px,py in zip(umap_pts[:,0], umap_pts[:,1]):
        _,idx = ti_tree.query([px,py], k=min(k,len(ti_coords)))
        nbr   = ti_cids[idx]
        counts = {
            'E4':  sum(c in E4_CLUSTERS   for c in nbr),
            'E1E5':sum(c in E1E5_CLUSTERS for c in nbr),
            'E20': sum(c in E20_CLUSTERS  for c in nbr),
        }
        best = max(counts, key=counts.get)
        phases.append(best if counts[best]>=1 else 'No_TI')
    return np.array(phases)

# ══════════════════════════════════════════════════════════════════════════════
# CORE EXTRACTION FUNCTION
# ══════════════════════════════════════════════════════════════════════════════
def extract_cluster_profile(keys, n_max):
    """
    For a list of LMDB keys (up to n_max), extract:
    - 14 module activations
    - 12 cell type node L2 norms
    - CpG beta statistics (if present)
    - Chromatin signal statistics (if present)
    - SNP genotype statistics (if present)
    - miRNA statistics (if present)
    - Metabolomics statistics (if present)
    Returns dict of lists.
    """
    if len(keys) > n_max:
        keys = random.sample(keys, n_max)

    records = defaultdict(list)
    failed  = 0

    for key in keys:
        try:
            acc = key.rsplit('_s',1)[0]
            if acc not in lmdb_idx: failed+=1; continue
            env = lmdb.open(lmdb_idx[acc], readonly=True, lock=False)
            with env.begin() as txn: val = txn.get(key.encode())
            env.close()
            if val is None: failed+=1; continue
            g = pickle.loads(val).to(DEVICE)
            nts = g.node_types

            # Module activations
            x  = {nt: g[nt].x for nt in nts if hasattr(g[nt],'x') and g[nt].x is not None}
            ei = {et: g[et].edge_index for et in g.edge_types if hasattr(g[et],'edge_index')}
            b  = {nt: torch.zeros(g[nt].x.size(0),dtype=torch.long,device=DEVICE) for nt in x}
            with torch.no_grad():
                _,_,_,mreps,_ = model(x,ei,b)
            if mreps is not None:
                acts = model.pathway_head(mreps).detach().cpu().numpy().flatten()
                records['module_acts'].append(acts)  # (14,)

            # Cell type activation norms
            if 'cell_type' in nts and g['cell_type'].x is not None:
                ct = g['cell_type'].x.norm(dim=1).cpu().numpy()  # (12,)
                if len(ct) == 12:
                    records['cell_type'].append(ct)

            # Methylation (CpG nodes)
            if 'cpg' in nts and g['cpg'].x is not None:
                b_vals = g['cpg'].x.cpu().numpy().flatten()
                records['meth_mean'].append(b_vals.mean())
                records['meth_hyper'].append((b_vals>0.7).mean())
                records['meth_hypo'].append((b_vals<0.3).mean())
                records['meth_n'].append(len(b_vals))

            # Chromatin (chromatin_region nodes)
            if 'chromatin_region' in nts and g['chromatin_region'].x is not None:
                sig = g['chromatin_region'].x.cpu().numpy().flatten()
                records['chrom_open'].append((sig>0.5).mean())
                records['chrom_mean'].append(sig.mean())
                records['chrom_n'].append(len(sig))

            # SNP
            if 'snp' in nts and g['snp'].x is not None:
                geno = g['snp'].x[:,0].cpu().numpy()
                records['snp_burden'].append((geno>0).mean())
                records['snp_het'].append((geno==0.5).mean())
                records['snp_hom_alt'].append((geno==1.0).mean())
                records['snp_n'].append(len(geno))

            # miRNA
            if 'mirna' in nts and g['mirna'].x is not None:
                mir = g['mirna'].x[:,0].cpu().numpy()
                records['mirna_mean'].append(mir.mean())
                records['mirna_n'].append(len(mir))

            # Metabolomics
            if 'metabolite' in nts and g['metabolite'].x is not None:
                met = g['metabolite'].x.cpu().numpy()
                records['met_intensity'].append(met[:,0].mean())
                records['met_uremic'].append(met[:,1].mean() if met.shape[1]>1 else 0.0)
                records['met_n'].append(met.shape[0])

        except: failed+=1

    records['failed'] = failed
    return records


# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP: Profile each named cluster
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n=== Profiling {len(NAMED_CLUSTERS)} named subclusters ===")

cluster_rows   = []   # one row per cluster — all modalities
phase_rows     = []   # one row per cluster×phase — TI activation
cluster_order  = sorted(NAMED_CLUSTERS.keys(),
                         key=lambda c: (NAMED_CLUSTERS[c]['disease'], c))

for cid in cluster_order:
    meta = NAMED_CLUSTERS[cid]
    cname = meta['name']
    print(f"\n--- {cname} (cluster {cid}) ---")

    # Get all samples in this cluster (matching disease class)
    sub = endo[(endo['refined_endotype']==cid) &
                (endo['disease_class'].isin(meta['dc']))]
    n_avail = len(sub)
    print(f"  Available: {n_avail:,}")
    if n_avail == 0:
        print("  SKIP — no samples")
        continue

    # Extract multimodal profile
    keys = sub['accession_id'].tolist()
    prof = extract_cluster_profile(keys, N_MAX)

    # Build summary row
    row = {
        'cluster_id': cid, 'cluster_name': cname, 'disease': meta['disease'],
        'n_available': n_avail, 'failed': prof['failed'],
    }

    # Module activations
    if prof['module_acts']:
        mat = np.array(prof['module_acts'])
        row['n_module'] = len(mat)
        for i,mn in enumerate(MODULE_NAMES):
            row[f'{mn}_mean'] = mat[:,i].mean()
            row[f'{mn}_sem']  = mat[:,i].std()/np.sqrt(len(mat))
        print(f"  Modules: n={len(mat)}")
    else:
        row['n_module'] = 0

    # Cell type
    if prof['cell_type']:
        ct_mat = np.array(prof['cell_type'])
        row['n_celltype'] = len(ct_mat)
        for i,ct in enumerate(CANONICAL_CELL_TYPES):
            row[f'ct_{ct}_mean'] = ct_mat[:,i].mean()
    else:
        row['n_celltype'] = 0

    # Methylation
    if prof['meth_mean']:
        row['n_meth']      = len(prof['meth_mean'])
        row['meth_mean']   = np.mean(prof['meth_mean'])
        row['meth_hyper']  = np.mean(prof['meth_hyper'])
        row['meth_hypo']   = np.mean(prof['meth_hypo'])
        print(f"  Methylation: n={row['n_meth']}, hyper={row['meth_hyper']:.3f}")
    else:
        row['n_meth'] = 0

    # Chromatin
    if prof['chrom_open']:
        row['n_chrom']     = len(prof['chrom_open'])
        row['chrom_open']  = np.mean(prof['chrom_open'])
        row['chrom_mean']  = np.mean(prof['chrom_mean'])
        print(f"  Chromatin: n={row['n_chrom']}, open={row['chrom_open']:.3f}")
    else:
        row['n_chrom'] = 0

    # SNP
    if prof['snp_burden']:
        row['n_snp']        = len(prof['snp_burden'])
        row['snp_burden']   = np.mean(prof['snp_burden'])
        row['snp_het']      = np.mean(prof['snp_het'])
        row['snp_hom_alt']  = np.mean(prof['snp_hom_alt'])
    else:
        row['n_snp'] = 0

    # miRNA
    if prof['mirna_mean']:
        row['n_mirna']     = len(prof['mirna_mean'])
        row['mirna_mean']  = np.mean(prof['mirna_mean'])
    else:
        row['n_mirna'] = 0

    # Metabolomics
    if prof['met_intensity']:
        row['n_met']          = len(prof['met_intensity'])
        row['met_intensity']  = np.mean(prof['met_intensity'])
        row['met_uremic']     = np.mean(prof['met_uremic'])
    else:
        row['n_met'] = 0

    cluster_rows.append(row)

    # TI phase assignment for samples with UMAP coords
    sub_umap = sub.dropna(subset=['umap_x','umap_y'])
    if len(sub_umap) > 0:
        pts    = sub_umap[['umap_x','umap_y']].values
        phases = assign_phase(pts)
        counts = Counter(phases)
        total  = len(phases)
        print(f"  TI phases: {dict(counts)}")

        # Per cluster×phase: mean TI module
        if prof['module_acts'] and len(prof['module_acts']) > 0:
            # Sample enough keys per phase for module extraction
            for phase in ['E4','E1E5','E20','No_TI']:
                phase_mask  = phases == phase
                phase_keys  = sub_umap[phase_mask]['accession_id'].tolist()
                n_phase     = len(phase_keys)
                if n_phase == 0: continue
                phase_keys_s = random.sample(phase_keys, min(50, n_phase))
                p_prof = extract_cluster_profile(phase_keys_s, 50)
                if p_prof['module_acts']:
                    p_mat = np.array(p_prof['module_acts'])
                    phase_rows.append({
                        'cluster_id':   cid,
                        'cluster_name': cname,
                        'disease':      meta['disease'],
                        'ti_phase':     phase,
                        'n_phase':      n_phase,
                        'n_extracted':  len(p_mat),
                        'ti_module_mean': p_mat[:,TI_IDX].mean(),
                        'ti_module_sem':  p_mat[:,TI_IDX].std()/np.sqrt(len(p_mat)),
                        **{f'{mn}_mean': p_mat[:,i].mean() for i,mn in enumerate(MODULE_NAMES)}
                    })

        for phase in ['E4','E1E5','E20','No_TI']:
            n = counts.get(phase,0)
            row[f'phase_{phase}_n']   = n
            row[f'phase_{phase}_pct'] = 100*n/total if total>0 else 0

# ══════════════════════════════════════════════════════════════════════════════
# SAVE CSVs
# ══════════════════════════════════════════════════════════════════════════════
df_profile = pd.DataFrame(cluster_rows)
df_phases  = pd.DataFrame(phase_rows)
df_profile.to_csv(os.path.join(OUT_ROOT,'cluster_full_profile.csv'), index=False)
df_phases.to_csv(os.path.join(OUT_ROOT,'cluster_phase_ti_activation.csv'), index=False)
print(f"\nSaved cluster_full_profile.csv ({len(df_profile)} clusters)")
print(f"Saved cluster_phase_ti_activation.csv ({len(df_phases)} cluster×phase rows)")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Module activation heatmap (14 modules × N clusters)
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Plotting Figure 1: Module Heatmap ===")
mod_cols = [f'{mn}_mean' for mn in MODULE_NAMES]
mod_df   = df_profile[['cluster_name','disease'] + mod_cols].dropna()
if len(mod_df) > 0:
    mat   = mod_df[mod_cols].values   # (N_clusters, 14)
    # Z-score across clusters per module for comparability
    mat_z = (mat - mat.mean(axis=0)) / (mat.std(axis=0) + 1e-8)

    fig, ax = plt.subplots(figsize=(16, max(6, len(mod_df)*0.42)))
    vmax = max(2.0, np.abs(mat_z).max())
    im = ax.imshow(mat_z.T, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(mod_df)))
    xlabels = [f"{r['cluster_name'].replace('CKD_','').replace('HD_','').replace('KTx_','')}\n({r['disease']})"
               for _,r in mod_df.iterrows()]
    ax.set_xticklabels(xlabels, rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(14))
    ax.set_yticklabels(MODULE_NAMES, fontsize=8)
    ax.set_title('iMEMORY V4 — Full Module Profile per Disease Subcluster\n'
                 '(Z-score across clusters | 14 modules × 25 subclusters)',
                 fontsize=12, fontweight='bold', pad=10)
    plt.colorbar(im, ax=ax, label='Z-score', fraction=0.015, pad=0.02)

    # Vertical disease group dividers
    disease_breaks = []
    prev = None
    for i, (_, r) in enumerate(mod_df.iterrows()):
        if r['disease'] != prev:
            disease_breaks.append(i)
            prev = r['disease']
    for b in disease_breaks[1:]:
        ax.axvline(b-0.5, color='white', lw=2.5)

    # Disease group labels
    disease_positions = {}
    for i, (_, r) in enumerate(mod_df.iterrows()):
        d = r['disease']
        if d not in disease_positions:
            disease_positions[d] = []
        disease_positions[d].append(i)
    for d, positions in disease_positions.items():
        mid = np.mean(positions)
        color = {'CKD':'#C62828','HD':'#1565C0','KTx':'#2E7D32'}[d]
        ax.text(mid, -1.2, d, ha='center', va='top', fontsize=10,
                fontweight='bold', color=color, transform=ax.transData)

    plt.tight_layout()
    p = os.path.join(OUT_ROOT,'Figure_PerCluster_ModuleHeatmap.png')
    fig.savefig(p, dpi=300, bbox_inches='tight')
    plt.close(fig)
    import shutil; # shutil.copy(p, os.path.join(COPY_DIR,'Figure_PerCluster_ModuleHeatmap.png'))
    print(f"Saved: {p}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: TI phase distribution per cluster (stacked bars)
# ══════════════════════════════════════════════════════════════════════════════
print("=== Plotting Figure 2: TI Phase Distribution ===")
phase_df   = df_profile[['cluster_name','disease'] +
                          [f'phase_{p}_pct' for p in ['E4','E1E5','E20','No_TI']]].dropna()
if len(phase_df) > 0:
    PHASE_COLORS_MAP = {'E4':'#0077CC','E1E5':'#DD2200','E20':'#8800CC','No_TI':'#1A1A2E'}
    fig, ax = plt.subplots(figsize=(max(12, len(phase_df)*0.55), 6))
    x = np.arange(len(phase_df))
    bottom = np.zeros(len(phase_df))
    for phase, color in PHASE_COLORS_MAP.items():
        col  = f'phase_{phase}_pct'
        if col not in phase_df.columns: continue
        vals = phase_df[col].values
        ax.bar(x, vals, 0.7, bottom=bottom, color=color, alpha=0.88,
               label=phase.replace('_TI',' TI').replace('E1E5','E1/E5'))
        for i, (v, b) in enumerate(zip(vals, bottom)):
            if v > 8:
                ax.text(i, b+v/2, f'{v:.0f}%', ha='center', va='center',
                        fontsize=6.5, color='white', fontweight='bold')
        bottom += vals

    ax.set_xticks(x)
    xlabels2 = [r['cluster_name'].replace('CKD_','').replace('HD_','').replace('KTx_','')
                for _,r in phase_df.iterrows()]
    ax.set_xticklabels(xlabels2, rotation=45, ha='right', fontsize=7.5)
    ax.set_ylim(0, 108)
    ax.set_ylabel('% of cluster samples', fontsize=10)
    ax.set_title('iMEMORY V4 — TI Endotype Phase Distribution Within Each Subcluster\n'
                 'What fraction of each subcluster is in each TI training stage?',
                 fontsize=11, fontweight='bold', pad=10)
    ax.legend(loc='upper right', fontsize=8.5, framealpha=0.9)
    ax.axhline(100, color='#AAA', lw=0.7, ls='--')
    for b in disease_breaks[1:]:
        ax.axvline(b-0.5, color='#AAA', lw=1.2, ls=':')
    for sp in ax.spines.values(): sp.set_edgecolor('#CCC')
    ax.yaxis.grid(True, color='#EEE', lw=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()
    p = os.path.join(OUT_ROOT,'Figure_PerCluster_TIPhase.png')
    fig.savefig(p, dpi=300, bbox_inches='tight')
    plt.close(fig)
    import shutil; # shutil.copy(p, os.path.join(COPY_DIR,'Figure_PerCluster_TIPhase.png'))
    print(f"Saved: {p}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Epigenetic profile per cluster (methylation + chromatin)
# ══════════════════════════════════════════════════════════════════════════════
print("=== Plotting Figure 3: Epigenetic Profile ===")
epi_df = df_profile[(df_profile['n_meth']>0) | (df_profile['n_chrom']>0)].copy() \
    if 'n_meth' in df_profile.columns else pd.DataFrame()
if len(epi_df) > 0:
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(max(12, len(epi_df)*0.7), 5))
    fig.subplots_adjust(wspace=0.35)
    x_e = np.arange(len(epi_df))

    for ax, col, title, color in [
        (ax1, 'meth_hyper', 'CpG Hypermethylation fraction\n(β > 0.7)', '#C2185B'),
        (ax2, 'chrom_open', 'Open chromatin fraction\n(ATAC signal > 0.5)', '#1565C0'),
    ]:
        if col not in epi_df.columns: continue
        vals = epi_df[col].fillna(0).values
        ns   = epi_df[f'n_{col.split("_")[0]}'].values if f'n_{col.split("_")[0]}' in epi_df else np.ones(len(vals))
        ax.bar(x_e, vals, 0.7, color=color, alpha=0.85)
        for i,(v,n) in enumerate(zip(vals,ns)):
            if n>0:
                ax.text(i, v+vals.max()*0.02, f'n={int(n)}',
                        ha='center', va='bottom', fontsize=6.5, color='#444')
        ax.set_xticks(x_e)
        ax.set_xticklabels([r['cluster_name'].replace('CKD_','').replace('HD_','').replace('KTx_','')
                             for _,r in epi_df.iterrows()],
                            rotation=45, ha='right', fontsize=7.5)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_ylabel('Fraction', fontsize=9)
        for sp in ax.spines.values(): sp.set_edgecolor('#CCC')
        ax.yaxis.grid(True, color='#EEE', lw=0.7)
        ax.set_axisbelow(True)

    fig.suptitle('iMEMORY V4 — Epigenetic Profile per Disease Subcluster\n'
                 '(Proxy for H3K4me3 / ATAC-seq chromatin remodeling — R01 Aim 1A/2B)',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    p = os.path.join(OUT_ROOT,'Figure_PerCluster_Epigenetic.png')
    fig.savefig(p, dpi=300, bbox_inches='tight')
    plt.close(fig)
    import shutil; # shutil.copy(p, os.path.join(COPY_DIR,'Figure_PerCluster_Epigenetic.png'))
    print(f"Saved: {p}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: Cell type activation heatmap per cluster
# ══════════════════════════════════════════════════════════════════════════════
print("=== Plotting Figure 4: Cell Type Activation ===")
ct_cols = [f'ct_{ct}_mean' for ct in CANONICAL_CELL_TYPES]
ct_df   = df_profile[df_profile['n_celltype']>0].copy() if 'n_celltype' in df_profile.columns else pd.DataFrame()
if len(ct_df) > 0 and all(c in ct_df.columns for c in ct_cols):
    ct_mat   = ct_df[ct_cols].values
    ct_mat_z = (ct_mat - ct_mat.mean(axis=0)) / (ct_mat.std(axis=0) + 1e-8)

    fig, ax = plt.subplots(figsize=(max(12, len(ct_df)*0.55), 6))
    im = ax.imshow(ct_mat_z.T, cmap='RdBu_r', aspect='auto', vmin=-2, vmax=2)
    ax.set_xticks(range(len(ct_df)))
    ax.set_xticklabels([r['cluster_name'].replace('CKD_','').replace('HD_','').replace('KTx_','')
                         for _,r in ct_df.iterrows()],
                        rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(12))
    ax.set_yticklabels([c.replace('_',' ') for c in CANONICAL_CELL_TYPES], fontsize=8)
    ax.set_title('iMEMORY V4 — Cell Type Module Co-activation per Subcluster\n'
                 '(Z-score | static GNN embeddings reflect module-cell type wiring)',
                 fontsize=11, fontweight='bold', pad=10)
    plt.colorbar(im, ax=ax, label='Z-score', fraction=0.015)
    plt.tight_layout()
    p = os.path.join(OUT_ROOT,'Figure_PerCluster_CellType.png')
    fig.savefig(p, dpi=300, bbox_inches='tight')
    plt.close(fig)
    import shutil; # shutil.copy(p, os.path.join(COPY_DIR,'Figure_PerCluster_CellType.png'))
    print(f"Saved: {p}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5: Other modalities (SNP / miRNA / metabolomics)
# ══════════════════════════════════════════════════════════════════════════════
print("=== Plotting Figure 5: Other Modalities ===")
other_metrics = [
    ('snp_burden', 'SNP variant burden', '#1565C0', 'n_snp'),
    ('mirna_mean', 'miRNA mean expression', '#2E7D32', 'n_mirna'),
    ('met_uremic', 'Uremic toxin fraction', '#7B1FA2', 'n_met'),
]
available = [(m,t,c,n) for m,t,c,n in other_metrics
             if m in df_profile.columns and df_profile[m].notna().any()]
if available:
    fig, axes = plt.subplots(1, len(available), figsize=(5.5*len(available), 5))
    if len(available)==1: axes=[axes]
    for ax, (metric, title, color, ncol) in zip(axes, available):
        sub = df_profile[df_profile[metric].notna() & (df_profile[ncol]>0)] \
              if ncol in df_profile.columns else df_profile[df_profile[metric].notna()]
        if len(sub)==0: continue
        vals = sub[metric].values
        ns   = sub[ncol].values if ncol in sub.columns else np.ones(len(sub))
        x_o  = np.arange(len(sub))
        ax.bar(x_o, vals, 0.7, color=color, alpha=0.85)
        for i,(v,n) in enumerate(zip(vals,ns)):
            ax.text(i, v+vals.max()*0.02, f'n={int(n)}',
                    ha='center', va='bottom', fontsize=6.5, color='#444')
        ax.set_xticks(x_o)
        ax.set_xticklabels([r['cluster_name'].replace('CKD_','').replace('HD_','').replace('KTx_','')
                             for _,r in sub.iterrows()],
                            rotation=45, ha='right', fontsize=7.5)
        ax.set_title(title, fontsize=10, fontweight='bold')
        for sp in ax.spines.values(): sp.set_edgecolor('#CCC')
        ax.yaxis.grid(True, color='#EEE', lw=0.7)
        ax.set_axisbelow(True)
    fig.suptitle('iMEMORY V4 — SNP / miRNA / Metabolomics per Subcluster\n'
                 '(sparse data — proof of concept)',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    p = os.path.join(OUT_ROOT,'Figure_PerCluster_OtherModalities.png')
    fig.savefig(p, dpi=300, bbox_inches='tight')
    plt.close(fig)
    import shutil; # shutil.copy(p, os.path.join(COPY_DIR,'Figure_PerCluster_OtherModalities.png'))
    print(f"Saved: {p}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6: Per-cluster TI phase module heatmap (cluster×phase × key modules)
# ══════════════════════════════════════════════════════════════════════════════
print("=== Plotting Figure 6: Per-cluster TI Phase Module Heatmap ===")
if len(df_phases) > 0:
    KEY_MODS = ['mTOR_glycolytic','trained_immunity_module','NF_kB_epigenetic','epigenetic_remodeling']
    key_cols = [f'{m}_mean' for m in KEY_MODS]

    for disease_group in ['CKD','HD','KTx']:
        sub_p = df_phases[df_phases['disease']==disease_group].copy()
        if len(sub_p) == 0: continue

        # Pivot: rows=clusters, cols=phases, value=trained_immunity_module
        clusters_in_group = sub_p['cluster_name'].unique()
        phases_in_group   = ['E4','E1E5','E20']
        fig, axes_p = plt.subplots(1, len(KEY_MODS), figsize=(4.5*len(KEY_MODS), max(4, len(clusters_in_group)*0.5+2)))
        fig.subplots_adjust(wspace=0.08)

        for ax_p, mod in zip(axes_p, KEY_MODS):
            col = f'{mod}_mean'
            pivot_data = np.full((len(clusters_in_group), len(phases_in_group)), np.nan)
            for i, cn in enumerate(clusters_in_group):
                for j, ph in enumerate(phases_in_group):
                    row_match = sub_p[(sub_p['cluster_name']==cn) & (sub_p['ti_phase']==ph)]
                    if len(row_match) > 0 and col in row_match.columns:
                        pivot_data[i,j] = row_match[col].iloc[0]

            # Min-shift for all-positive
            valid = pivot_data[~np.isnan(pivot_data)]
            if len(valid) == 0: continue
            shift = valid.min()
            pivot_shifted = pivot_data - shift
            pivot_shifted[np.isnan(pivot_data)] = 0

            vmax_p = max(0.1, np.nanmax(pivot_shifted))
            im_p = ax_p.imshow(pivot_shifted, cmap='OrRd', aspect='auto', vmin=0, vmax=vmax_p)
            ax_p.set_xticks(range(len(phases_in_group)))
            ax_p.set_xticklabels(['E4','E1/E5','E20'], fontsize=9)
            if ax_p == axes_p[0]:
                ax_p.set_yticks(range(len(clusters_in_group)))
                ax_p.set_yticklabels([cn.replace(f'{disease_group}_','') for cn in clusters_in_group], fontsize=8)
            else:
                ax_p.set_yticks([])
            ax_p.set_title(mod.replace('_',' ').title(), fontsize=9, fontweight='bold')

            for ri in range(len(clusters_in_group)):
                for ci in range(len(phases_in_group)):
                    if not np.isnan(pivot_data[ri,ci]):
                        v = pivot_shifted[ri,ci]
                        clr = 'white' if v > vmax_p*0.65 else '#222'
                        n_r = sub_p[(sub_p['cluster_name']==clusters_in_group[ri]) &
                                    (sub_p['ti_phase']==phases_in_group[ci])]['n_extracted'].values
                        n_txt = int(n_r[0]) if len(n_r)>0 else 0
                        ax_p.text(ci, ri, f'{v:.2f}\nn={n_txt}',
                                  ha='center', va='center', fontsize=6.5, color=clr, fontweight='bold')

            plt.colorbar(im_p, ax=ax_p, fraction=0.04, pad=0.04)

        fig.suptitle(f'iMEMORY V4 — {disease_group} Subclusters: Key Module Activation by TI Phase\n'
                     'Does CKD-E4 look different from HD-E4? Disease-specific training fingerprint.',
                     fontsize=11, fontweight='bold', y=1.02)
        plt.tight_layout()
        p = os.path.join(OUT_ROOT, f'Figure_PerCluster_{disease_group}_PhaseHeatmap.png')
        fig.savefig(p, dpi=300, bbox_inches='tight')
        plt.close(fig)
        import shutil; # shutil.copy(p, os.path.join(COPY_DIR, f'Figure_PerCluster_{disease_group}_PhaseHeatmap.png'))
        print(f"Saved: {p}")

print("\n=== step11 DONE ===")
print(f"All outputs in {OUT_ROOT}")
print(f"Figures copied to {COPY_DIR}")
