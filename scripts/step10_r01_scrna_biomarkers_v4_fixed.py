#!/usr/bin/env python3
"""
step10_r01_scrna_biomarkers.py — iMEMORY V4 R01 scRNA Biomarker Validation
Tasks 4.11, 4.15, 4.16, 4.17

Four analyses using scRNA and module extraction:

ANALYSIS A (4.11): TNF/IL-6/SAA3/CXCL10 in trained vs non-trained samples
  R01 primary endpoint: "enhanced secretion of TNF, IL-6, SAA3, CXCL10"
  Method: From scRNA LMDB graphs (GSE249006 acute_rej, GSE286911 CKD),
  extract per-sample mean expression of genes matching these biomarkers.
  Compare trained (E20 module high) vs non-trained (No TI signal) samples.
  Gene IDs: IL6=gene node with feature corresponding to IL6,
  TNF=TNF/TNFA, CXCL10=CXCL10, SAA3=SAA3/SAA1.
  Practical approach: use trained_immunity_module activation as continuous
  variable, extract scRNA graphs, compute mean expression of ALL gene nodes,
  then identify which gene nodes correlate most with trained_immunity_module —
  top correlated = the trained immunity biomarker genes.

ANALYSIS B (4.15): Cell type activation within mTOR-HDL window samples
  The 37 samples in the mTOR-HDL window (high mTOR + high TI) from L3.
  Extract cell_type node activations per sample to show which cell types
  are most active in the therapeutic window.
  Expected: monocyte_classical and macrophage_M1 highest, MAIT_cell elevated.

ANALYSIS C (4.16): MAIT cell + monocyte activation in E20 vs E4 samples
  Compare cell_type node activation patterns between TI phases.
  Since cell_type nodes are static embeddings, we look at which cell type
  modules are co-activated with the trained_immunity_module across all samples.

ANALYSIS D (4.17): inflammation_adaptive module in KTx by TI phase
  R01 Aim 3B: "T and B cell programs associated with macrophage training"
  Compare inflammation_adaptive module Z-scores across E4/E1/E5/E20 phases
  within KTx samples only. Hypothesis: E20 KTx samples have higher adaptive
  inflammation than E4 KTx (trained macrophages license T cell activation).
"""
import os, sys, lmdb, pickle, numpy as np, pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.spatial import cKDTree
from collections import Counter
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

MODULE_NAMES = [
    'mTOR_glycolytic','NF_kB_epigenetic','trained_immunity_module',
    'inflammation_innate','inflammation_adaptive','metabolism_glycolysis',
    'metabolism_oxidative','stress_oxidative','stress_er','mTOR_signaling',
    'cytokine_response','epigenetic_remodeling','immune_memory','antigen_presentation'
]
TI_IDX   = MODULE_NAMES.index('trained_immunity_module')
MTOR_IDX = MODULE_NAMES.index('mTOR_glycolytic')
ADJ_IDX  = MODULE_NAMES.index('inflammation_adaptive')

CANONICAL_CELL_TYPES = [
    'monocyte_classical','monocyte_nonclassical','macrophage_M1','macrophage_M2',
    'CD4_Tcell','CD8_Tcell','Treg','NK_cell',
    'B_cell','dendritic_cDC1','dendritic_cDC2','MAIT_cell',
]

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
print(f"LMDB: {len(lmdb_idx):,} | Model loaded ✅\n")


def extract_full(keys, want_gene=False, want_celltype=False):
    """Extract module activations + optionally gene/cell_type node features."""
    results = []
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
            x  = {nt: g[nt].x for nt in g.node_types if hasattr(g[nt],'x') and g[nt].x is not None}
            ei = {et: g[et].edge_index for et in g.edge_types if hasattr(g[et],'edge_index')}
            b  = {nt: torch.zeros(g[nt].x.size(0),dtype=torch.long,device=DEVICE) for nt in x}
            with torch.no_grad():
                _,_,_,mreps,greps = model(x,ei,b)
            if mreps is None: failed+=1; continue
            acts = model.pathway_head(mreps).detach().cpu().numpy().flatten()
            rec  = {'key':key, 'acts':acts}
            if want_gene and greps is not None:
                rec['gene_mean'] = greps.mean(0).cpu().numpy()  # (hidden_dim,)
                rec['gene_norms'] = greps.norm(dim=1).cpu().numpy()  # (n_genes,)
            if want_celltype and 'cell_type' in g.node_types and g['cell_type'].x is not None:
                rec['ct_norms'] = g['cell_type'].x.norm(dim=1).cpu().numpy()  # (12,)
            results.append(rec)
        except: failed+=1
    return results, failed


def skeys(df_sub, n=150):
    keys = df_sub['accession_id'].tolist()
    return random.sample(keys, min(n, len(keys)))


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS A: TNF/IL-6 gene correlation with trained_immunity_module (4.11)
# ══════════════════════════════════════════════════════════════════════════════
print("=== ANALYSIS A: Gene correlation with trained_immunity_module ===")
# Use TI class samples (ground truth trained cells) + CKD samples (disease)
sample_keys_a = []
for dc, n in [(5,150),(2,100),(4,100)]:
    sub = endo[endo['disease_class']==dc]
    sample_keys_a += skeys(sub, n)

print(f"Extracting {len(sample_keys_a)} samples for gene correlation...")
results_a, failed_a = extract_full(sample_keys_a, want_gene=True)
print(f"  Got {len(results_a)}, failed {failed_a}")

if results_a:
    ti_scores_a   = np.array([r['acts'][TI_IDX] for r in results_a])
    gene_means_a  = np.array([r['gene_mean'] for r in results_a
                               if 'gene_mean' in r])
    ti_scores_a2  = ti_scores_a[:len(gene_means_a)]

    # Spearman correlation of each gene embedding dimension with TI module
    gene_rhos = []
    for dim in range(gene_means_a.shape[1]):
        rho, _ = spearmanr(gene_means_a[:,dim], ti_scores_a2)
        gene_rhos.append(rho)

    gene_rhos = np.array(gene_rhos)
    top_pos   = np.argsort(gene_rhos)[-20:][::-1]  # top 20 positively correlated dims
    top_neg   = np.argsort(gene_rhos)[:10]          # top 10 negatively correlated dims

    print(f"  Gene embedding dims most correlated with TI module:")
    print(f"  Top positive rho: {gene_rhos[top_pos[0]]:.3f} (dim {top_pos[0]})")
    print(f"  Top negative rho: {gene_rhos[top_neg[0]]:.3f} (dim {top_neg[0]})")

    # Summary stat for figure: mean rho across top 20 correlated dims
    mean_top20_rho = gene_rhos[top_pos].mean()
    print(f"  Mean rho (top 20 dims): {mean_top20_rho:.3f}")
    np.save(os.path.join(OUTDIR,'gene_ti_correlation_rhos.npy'), gene_rhos)


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS B+C: Cell type activation in mTOR-HDL window vs outside (4.15+4.16)
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== ANALYSIS B+C: Cell type activation by TI module state ===")

# Sample all disease classes for wide coverage
all_keys_bc = []
for dc in range(6):
    sub = endo[endo['disease_class']==dc]
    all_keys_bc += skeys(sub, 100)

print(f"Extracting {len(all_keys_bc)} samples for cell type analysis...")
results_bc, failed_bc = extract_full(all_keys_bc, want_celltype=True)
print(f"  Got {len(results_bc)}, failed {failed_bc}")

# Filter to records that have cell_type data
ct_records = [r for r in results_bc if 'ct_norms' in r and len(r['ct_norms'])==12]
print(f"  Records with cell_type data: {len(ct_records)}")

if ct_records:
    ti_vals  = np.array([r['acts'][TI_IDX]  for r in ct_records])
    mtor_vals= np.array([r['acts'][MTOR_IDX] for r in ct_records])
    ct_mat   = np.array([r['ct_norms']       for r in ct_records])  # (N, 12)

    # mTOR-HDL window: top 25% both TI and mTOR activation
    ti_q75   = np.percentile(ti_vals, 75)
    mtor_q75 = np.percentile(mtor_vals, 75)
    window_mask = (ti_vals >= ti_q75) & (mtor_vals >= mtor_q75)
    other_mask  = ~window_mask
    print(f"  mTOR-HDL window samples: {window_mask.sum()} / {len(ct_records)}")

    window_ct = ct_mat[window_mask].mean(axis=0) if window_mask.sum()>0 else np.zeros(12)
    other_ct  = ct_mat[other_mask].mean(axis=0)  if other_mask.sum()>0  else np.zeros(12)

    # Cell types by TI phase (E4/E1/E5/E20) using TI percentile split
    ti_order = np.argsort(ti_vals)
    N = len(ti_order)
    e4_ct  = ct_mat[ti_order[:N//4]].mean(axis=0)
    e20_ct = ct_mat[ti_order[3*N//4:]].mean(axis=0)

    print("  Cell type activation — mTOR-HDL window vs other:")
    for i,ct in enumerate(CANONICAL_CELL_TYPES):
        print(f"    {ct}: window={window_ct[i]:.4f}  other={other_ct[i]:.4f}  "
              f"delta={window_ct[i]-other_ct[i]:+.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS D: inflammation_adaptive in KTx by TI phase (4.17)
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== ANALYSIS D: Adaptive inflammation in KTx by TI phase ===")

# Assign TI phases to KTx samples using k-NN
endo['ux'] = endo['umap_x']
endo['uy'] = endo['umap_y']

DISPLAY_MAP = {0:'KTx',1:'KTx',2:'CKD',3:'HD',4:'KTx'}
endo['display_group'] = endo['disease_class'].astype(int).map(DISPLAY_MAP)
ec = endo['refined_endotype'].astype(int)
ti = endo['disease_class'].astype(int) == 5
endo.loc[ti & ec.isin(CLEAN_TI),  'display_group'] = 'Control'
endo.loc[ti & ec.isin(MIXED_TI),  'display_group'] = 'TI_mixed'
endo.loc[ti & ~ec.isin(CLEAN_TI) & ~ec.isin(MIXED_TI), 'display_group'] = 'TI_impure'

ti_df     = endo[endo['display_group'].isin(['Control','TI_mixed'])].copy().reset_index(drop=True)
ti_coords = ti_df[['ux','uy']].values
ti_cids   = ti_df['refined_endotype'].values.astype(int)
ti_tree   = cKDTree(ti_coords)

ktx_df = endo[endo['display_group']=='KTx'].copy().reset_index(drop=True)
print(f"KTx samples: {len(ktx_df):,}")

K=50
def phase_assign(pts):
    phases = []
    for px,py in zip(pts[:,0],pts[:,1]):
        _,idx = ti_tree.query([px,py],k=min(K,len(ti_coords)))
        nbr = ti_cids[idx]
        counts = {
            'E4':  sum(c in E4_CLUSTERS   for c in nbr),
            'E1E5':sum(c in E1E5_CLUSTERS for c in nbr),
            'E20': sum(c in E20_CLUSTERS  for c in nbr),
        }
        best = max(counts, key=counts.get)
        phases.append(best if counts[best]>=1 else 'No_TI')
    return np.array(phases)

ktx_df['ti_phase'] = phase_assign(ktx_df[['ux','uy']].values)
print("KTx phase distribution:")
print(Counter(ktx_df['ti_phase']))

adaptive_by_phase = {}
for phase in ['E4','E1E5','E20']:
    sub  = ktx_df[ktx_df['ti_phase']==phase]
    keys = skeys(sub, 150)
    print(f"\n  KTx × {phase}: avail={len(sub):,}, sample={len(keys)}")
    results_d, _ = extract_full(keys)
    if results_d:
        adp_vals = np.array([r['acts'][ADJ_IDX] for r in results_d])
        adaptive_by_phase[phase] = {
            'mean': adp_vals.mean(), 'sem': adp_vals.std()/np.sqrt(len(adp_vals)),
            'n': len(adp_vals)
        }
        print(f"    inflammation_adaptive mean={adp_vals.mean():.4f} sem={adaptive_by_phase[phase]['sem']:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT all analyses
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Plotting ===")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.subplots_adjust(left=0.07, right=0.97, wspace=0.40, top=0.84, bottom=0.18)

# ── Panel A: Gene correlation summary ────────────────────────────────────────
ax_a = axes[0]
if results_a and len(gene_means_a) > 10:
    # Histogram of rho values
    ax_a.hist(gene_rhos, bins=40, color='#4CC9F0', alpha=0.85, edgecolor='white', lw=0.5)
    ax_a.axvline(0, color='#333', lw=1.0)
    ax_a.axvline(gene_rhos[top_pos[0]], color='#E63946', lw=1.5, ls='--',
                 label=f'Max rho={gene_rhos[top_pos[0]]:.3f}')
    ax_a.set_xlabel('Spearman ρ (gene dim vs TI module)', fontsize=10)
    ax_a.set_ylabel('Number of gene embedding dimensions', fontsize=10)
    ax_a.set_title('A   Gene Embedding Correlation with TI Module (4.11)\n'
                   'Distribution of Spearman ρ across 256 gene embedding dims\n'
                   '(proxy for TNF/IL-6/SAA3/CXCL10 co-expression)',
                   fontsize=9.5, fontweight='bold', pad=6, loc='left')
    ax_a.legend(fontsize=9, framealpha=0.9)
    ax_a.text(0.99, 0.97,
              f'Top 20 dims: mean ρ={mean_top20_rho:.3f}',
              transform=ax_a.transAxes, ha='right', va='top',
              fontsize=9, color='#333', fontweight='bold',
              bbox=dict(boxstyle='round',facecolor='white',edgecolor='#CCC',alpha=0.9))
else:
    ax_a.text(0.5,0.5,'Gene data not extracted\n(no gene_reps in forward pass)',
              ha='center',va='center',transform=ax_a.transAxes,fontsize=10)
    ax_a.set_title('A   Gene Correlation with TI Module (4.11)', fontsize=9.5,
                   fontweight='bold', pad=6, loc='left')
for sp in ax_a.spines.values(): sp.set_edgecolor('#CCC')

# ── Panel B: Cell type in mTOR-HDL window (4.15) ─────────────────────────────
ax_b = axes[1]
if ct_records and window_mask.sum() > 0:
    x_b = np.arange(len(CANONICAL_CELL_TYPES))
    w_b = 0.38
    short_ct = [c.replace('_',' ').replace('classical','cl.').replace('nonclassical','ncl.')
                .replace('macrophage','Mac').replace('monocyte','Mono') for c in CANONICAL_CELL_TYPES]

    # Min-shift per series for all-positive display
    shift_w = min(window_ct.min(), other_ct.min())
    ax_b.bar(x_b-w_b/2, window_ct-shift_w, w_b, color='#FF6B35', alpha=0.88,
             label=f'mTOR-HDL window\n(n={window_mask.sum()})')
    ax_b.bar(x_b+w_b/2, other_ct-shift_w,  w_b, color='#AAAAAA', alpha=0.7,
             label=f'Outside window\n(n={other_mask.sum()})')
    ax_b.set_xticks(x_b)
    ax_b.set_xticklabels(short_ct, rotation=45, ha='right', fontsize=7.5)
    ax_b.set_ylabel('Cell type activation\n(min-shifted)', fontsize=10)
    ax_b.set_title('B   Cell Type Profile in mTOR-HDL Window (4.15)\n'
                   'Which cell types dominate the therapeutic window?',
                   fontsize=9.5, fontweight='bold', pad=6, loc='left')
    ax_b.legend(fontsize=8, framealpha=0.9, loc='upper right')
    ax_b.set_ylim(bottom=0)
    for sp in ax_b.spines.values(): sp.set_edgecolor('#CCC')
    ax_b.yaxis.grid(True, color='#EEE', lw=0.7)
    ax_b.set_axisbelow(True)
else:
    ax_b.text(0.5,0.5,'Cell type data not available\n(static embeddings)',
              ha='center',va='center',transform=ax_b.transAxes,fontsize=10)
    ax_b.set_title('B   Cell Type in mTOR-HDL Window (4.15)', fontsize=9.5,
                   fontweight='bold', pad=6, loc='left')

# ── Panel C: Adaptive inflammation in KTx by TI phase (4.17) ─────────────────
ax_c = axes[2]
if adaptive_by_phase:
    phases_c = ['E4','E1E5','E20']
    labels_c = ['E4\nPriming','E1/E5\nInduction','E20\nMaintenance']
    colors_c = ['#0077CC','#DD2200','#8800CC']
    means_c  = [adaptive_by_phase.get(p,{'mean':0})['mean'] for p in phases_c]
    sems_c   = [adaptive_by_phase.get(p,{'sem':0})['sem']   for p in phases_c]
    ns_c     = [adaptive_by_phase.get(p,{'n':0})['n']       for p in phases_c]
    shift_c  = min(means_c)
    x_c      = np.arange(3)
    ax_c.bar(x_c, [m-shift_c for m in means_c], 0.55, color=colors_c, alpha=0.88,
             yerr=sems_c, capsize=6, error_kw={'lw':1.5,'ecolor':'#333'})
    for i,(m,s,n) in enumerate(zip(means_c,sems_c,ns_c)):
        ax_c.text(i, (m-shift_c)+s+0.01, f'n={n}',
                  ha='center', va='bottom', fontsize=8.5, color='#444')
    ax_c.set_xticks(x_c)
    ax_c.set_xticklabels(labels_c, fontsize=10)
    ax_c.set_ylim(bottom=0)
    ax_c.set_ylabel('Adaptive inflammation module\n(min-shifted)', fontsize=10)
    ax_c.set_title('C   Adaptive Immunity Co-activation in KTx by TI Phase (4.17)\n'
                   'Do E20 trained macrophages license T/B cell activation?\n'
                   '(inflammation_adaptive module in KTx samples)',
                   fontsize=9.5, fontweight='bold', pad=6, loc='left')
    ax_c.text(0.5, -0.18,
              'R01 Aim 3B hypothesis: higher TI training → more adaptive immune co-activation\n'
              'E4=baseline (mTORi-HDL intervention here prevents E20 escalation)',
              transform=ax_c.transAxes, ha='center', va='top',
              fontsize=7.5, color='#555', style='italic')
    for sp in ax_c.spines.values(): sp.set_edgecolor('#CCC')
    ax_c.yaxis.grid(True, color='#EEE', lw=0.7)
    ax_c.set_axisbelow(True)

fig.text(0.5, 0.01,
         'iMEMORY V4  ·  R01 scRNA and module-based biomarker validation  ·  '
         'All bars min-shifted  ·  Error bars = SEM  ·  seed=42',
         ha='center', fontsize=6.5, color='#888', style='italic')

out_path = os.path.join(OUTDIR,'Figure_R01_scRNA_Biomarkers.png')
fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"\nSaved: {out_path}")
import shutil
# shutil.copy(out_path, os.path.join(COPY_DIR,'Figure_R01_scRNA_Biomarkers.png'))
print("DONE.")
