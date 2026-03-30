#!/usr/bin/env python3
"""
graph_builder_lmdb_three_tier_v2.py

V2 of the production graph builder. Adds:
  1. All missing modalities: scRNA, CyTOF, HiC, SNP, Metabolomics
  2. 8-GPU parallel processing for large gene_expression/scRNA files
     (>GPU_SAMPLE_THRESHOLD samples → shared memory + 8 GPU worker processes)
  3. All V1 patches preserved:
     - CANONICAL_MODULES = 14 pathway-biology modules (mTOR_glycolytic, etc.)
     - _PATHWAY_GENE_SETS_UPPER precomputed at startup
     - pathway_gene_masks + module_pathway_ids on ALL graph types

NEW MODALITY BUILDERS:
  build_scrna_graph     — scRNA-seq (per-cell or pseudobulk)
                          Uses gene embedding path where HGNC coverage ≥ 10%.
                          Falls back to PCA-compressed raw expression if coverage < 10%.
  build_cytof_graph     — CyTOF / mass cytometry protein marker intensities
                          Marker names mapped to cell_type nodes where possible.
                          Node type: cytof_marker (n_markers, 64)
  build_hic_graph       — Hi-C chromatin contact matrices
                          Bin-level A/B compartment + insulation score summary.
                          Node type: hic_region (n_bins, 64)
  build_snp_graph       — SNP / genotype arrays (rsIDs or numeric)
                          Node type: snp (n_snps, 64)
  build_metabolite_graph — Metabolomics / lipidomics
                          Uremic toxin flags applied where metabolite name matches.
                          Node type: metabolite (n_mets, 64)

GPU PARALLELISM:
  Files with > GPU_SAMPLE_THRESHOLD (50,000) samples use the GPU path.
  Main process loads the NPZ matrix into POSIX shared memory once.
  8 GPU worker processes (one per A100) each take chunks from a task queue,
  build graphs, and write to LMDB. No data copying between workers.
  All other modalities use the sequential CPU path.

Save as: /mnt/data/trained_im/scripts/graph_builder_lmdb_three_tier_v2.py

Run as:
  python3 /mnt/data/trained_im/scripts/graph_builder_lmdb_three_tier_v2.py \
      --npz_dir  /mnt/nvme15tb/trained_im/NPZ_fresh \
      --output_dir /mnt/nvme15tb/trained_im/graphs_lmdb_v3 \
      --manifest /mnt/nvme15tb/trained_im/logs/rebuild_manifest.txt \
      2>&1 | tee /mnt/nvme15tb/trained_im/logs/graph_builder_v3_v2.log
"""
import os, sys, gc, json, pickle, argparse, logging
import numpy as np
import torch
from torch_geometric.data import HeteroData
import lmdb
from multiprocessing import Process, Queue
from multiprocessing import shared_memory as shm_mod
import torch.multiprocessing as mp

try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

from pipeline_config import PATHS, MTOR_CENTRALITY_GENES, MTOR_CENTRALITY_PRIOR, HIGH_VALUE_DATASETS, MTORI_HDL_BIAS
from gene_id_normalizer import GeneIDNormalizer
from extended_pathway_modules import EXTENDED_PATHWAY_MODULES, N_EXTENDED_PATHWAYS

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
LABEL_MAP = {
    'acute_rejection': 0, 'chronic_rejection': 1, 'ckd': 2,
    'hemodialysis': 3, 'kidney_transplant': 4, 'trained_immunity': 5,
}
MODALITY_INDEX = {
    'gene_expression': 0, 'bulk_rna': 0,
    'methylation': 1, 'mirna': 2,
    'cytof': 3, 'chipseq': 4, 'chromatin': 4, 'atac': 4,
    'scrna': 5, 'hic': 7, 'snp': 8,
    'metabolite': 9, 'metabolomics': 9, 'lipidomics': 9,
    'chromatin_region': 10, 'unknown': 11,
}
MIN_COVERAGE_THRESHOLD  = 0.30   # gene_expression
SCRNA_COVERAGE_THRESHOLD = 0.10  # scrna: accept lower coverage
LMDB_MAP_SMALL  = 5   * 1024**3  # 5 GB
LMDB_MAP_LARGE  = 500 * 1024**3  # 500 GB
LARGE_THRESHOLD = 10_000         # samples → big LMDB map
GPU_THRESHOLD   = 50_000         # samples → use 8-GPU path
NUM_GPUS        = 8
GPU_CHUNK       = 10_000

# ── CANONICAL_MODULES — V3 patch (14 pathway-biology modules) ─────────────────
CANONICAL_MODULES = {
    'mTOR_glycolytic': [
        'MTOR','RPTOR','RICTOR','AKT1','AKT2','PIK3CA','PIK3CB','TSC1','TSC2',
        'PTEN','HIF1A','LDHA','ENO1','ALDOA','PFKL','HK2','PKM','GAPDH',
        'SLC2A1','SLC2A3','VEGFA','MYC','PDK1','PDK2','PFKFB3',
    ],
    'NF_kB_epigenetic': [
        'NFKB1','NFKB2','RELA','RELB','REL','NFKBIA','NFKBIB','IKBKA','IKBKB',
        'TNFAIP3','BIRC2','BIRC3','TRAF2','TRAF6','MAP3K7',
        'KMT2A','KMT2B','KMT2D','EP300','CREBBP','HDAC1','HDAC2','EZH2',
    ],
    'trained_immunity_module': [
        'IL6','IL1B','TNF','IL18','CSF1','CSF2','CXCL8','CXCL10',
        'CCL2','CCL5','NLRP3','CASP1','PYCARD','TLR4','TLR2','NOD2',
        'MYD88','DECTIN1','CLEC7A','SETDB1','DOT1L','KMT2A',
    ],
    'inflammation_innate': [
        'TNF','IL1B','IL6','IL18','TLR2','TLR4','TLR7','TLR9','MYD88',
        'TRIF','IRF3','IRF7','STAT1','STAT3','NLRP3','CASP1','PYCARD',
        'HMGB1','S100A8','S100A9','IFIT1','IFIT2','ISG15',
    ],
    'inflammation_adaptive': [
        'CD3D','CD4','CD8A','FOXP3','GATA3','TBX21','RORC','BCL6',
        'IL2','IL4','IL5','IL10','IL13','IL17A','IL21','IFNG',
        'CTLA4','PDCD1','LAG3','TIM3','CD28','TIGIT',
    ],
    'metabolism_glycolysis': [
        'HK1','HK2','GPI','PFKL','PFKM','ALDOA','GAPDH','PGK1',
        'ENO1','PKM','LDHA','LDHB','SLC2A1','SLC2A3','PDK1',
        'PGAM1','TPI1','PFKFB3','PFKFB4',
    ],
    'metabolism_oxidative': [
        'NDUFS1','NDUFS2','SDHA','SDHB','UQCRC1','COX5A','ATP5A1',
        'CS','IDH1','IDH2','MDH2','OGDH','ACO2',
        'PPARGC1A','TFAM','NRF1','ESRRA','CPT1A','CPT2',
    ],
    'stress_oxidative': [
        'SOD1','SOD2','CAT','GPX1','GPX4','PRDX1','PRDX2','PRDX3',
        'HMOX1','NQO1','NFE2L2','KEAP1','TXN','TXNRD1',
        'NOX2','CYBA','NCF1','NCF2',
    ],
    'stress_er': [
        'HSPA5','DDIT3','ATF4','ATF6','XBP1','EIF2AK3','ERN1',
        'CALR','CANX','PDIA3','PDIA4','HSP90B1','SEC61A1',
    ],
    'mTOR_signaling': [
        'MTOR','RPTOR','RICTOR','MLST8','DEPTOR','AKT1S1',
        'RPS6KB1','RPS6KB2','EIF4EBP1','EIF4E','EIF4G1',
        'RHEB','RRAGA','RRAGB','TSC1','TSC2','STK11','PRKAA1','PRKAA2',
    ],
    'cytokine_response': [
        'JAK1','JAK2','TYK2','STAT1','STAT2','STAT3','STAT4',
        'STAT5A','STAT5B','STAT6','SOCS1','SOCS2','SOCS3','CISH',
        'IL6R','IL2RA','IFNGR1','IFNAR1','IL4R','TNFRSF1A',
    ],
    'epigenetic_remodeling': [
        'EZH2','EZH1','SUZ12','EED','KMT2A','KMT2B','KMT2C','KMT2D',
        'SETD2','SETD7','SETDB1','DOT1L','KDM1A','KDM4A','KDM5C',
        'KDM6A','DNMT1','DNMT3A','DNMT3B','TET1','TET2',
        'HDAC1','HDAC2','HDAC3','SIRT1','EP300','CREBBP',
    ],
    'immune_memory': [
        'IL7R','IL15RA','SELL','CCR7','TCF7','LEF1','KLF4','MYB',
        'BCL2','BCL2L1','MCL1','CD27','CD62L','S1PR1',
        'FOXO1','FOXO3','EOMES','TBX21','TOX','TOX2',
    ],
    'antigen_presentation': [
        'HLA-A','HLA-B','HLA-C','HLA-DRA','HLA-DRB1','HLA-DPA1',
        'B2M','TAP1','TAP2','TAPBP','CALR','CANX',
        'CD74','CIITA','NLRC5','RFXANK','RFX5','RFXAP',
    ],
}
MODULE_NAMES = list(CANONICAL_MODULES.keys())

# Precompute pathway gene sets (uppercase) for pathway_gene_masks construction
_PATHWAY_GENE_SETS_UPPER = {
    m_idx: set(g.upper() for g in genes)
    for m_idx, (_, genes) in enumerate(CANONICAL_MODULES.items())
}
MODULE_PATHWAY_IDS = torch.arange(len(MODULE_NAMES), dtype=torch.long)

CANONICAL_CELL_TYPES = [
    'monocyte_classical', 'monocyte_nonclassical', 'macrophage_M1', 'macrophage_M2',
    'CD4_Tcell', 'CD8_Tcell', 'Treg', 'NK_cell',
    'B_cell', 'dendritic_cDC1', 'dendritic_cDC2', 'MAIT_cell',
]
CELL_TYPE_MODULE_EDGES = {
    'monocyte_classical':    ['inflammation_innate','trained_immunity_module','mTOR_glycolytic','cytokine_response'],
    'monocyte_nonclassical': ['inflammation_innate','cytokine_response'],
    'macrophage_M1':         ['inflammation_innate','trained_immunity_module','cytokine_response','stress_oxidative'],
    'macrophage_M2':         ['inflammation_adaptive','epigenetic_remodeling','immune_memory'],
    'CD4_Tcell':             ['inflammation_adaptive','mTOR_glycolytic','immune_memory'],
    'CD8_Tcell':             ['inflammation_adaptive','mTOR_glycolytic','cytokine_response'],
    'Treg':                  ['inflammation_adaptive','epigenetic_remodeling','immune_memory'],
    'NK_cell':               ['inflammation_innate','cytokine_response','stress_oxidative'],
    'B_cell':                ['antigen_presentation','NF_kB_epigenetic'],
    'dendritic_cDC1':        ['antigen_presentation','cytokine_response','inflammation_innate'],
    'dendritic_cDC2':        ['inflammation_innate','cytokine_response'],
    'MAIT_cell':             ['inflammation_innate','trained_immunity_module','cytokine_response'],
}

# CyTOF: protein marker → cell type & module mapping
CYTOF_MARKER_CELL_TYPE = {
    'CD14': 'monocyte_classical', 'CD16': 'monocyte_nonclassical',
    'CD3': 'CD4_Tcell', 'CD4': 'CD4_Tcell', 'CD8': 'CD8_Tcell',
    'CD25': 'Treg', 'FOXP3': 'Treg',
    'CD56': 'NK_cell', 'CD19': 'B_cell', 'CD20': 'B_cell',
    'HLA-DR': 'dendritic_cDC1', 'CD11c': 'dendritic_cDC2',
    'Va7.2': 'MAIT_cell', 'CD161': 'MAIT_cell',
}
CYTOF_MARKER_MODULE = {
    'pS6': 'mTOR_glycolytic', 'p4EBP1': 'mTOR_signaling',
    'pAKT': 'mTOR_glycolytic', 'pSTAT1': 'cytokine_response',
    'pSTAT3': 'cytokine_response', 'pSTAT5': 'cytokine_response',
    'pNFkB': 'NF_kB_epigenetic', 'Ki67': 'immune_memory',
    'BCL2': 'immune_memory', 'TNF': 'trained_immunity_module',
    'IL6': 'trained_immunity_module', 'IFNg': 'inflammation_adaptive',
    'IL17A': 'inflammation_adaptive', 'Granzyme B': 'inflammation_adaptive',
    'HIF1a': 'mTOR_glycolytic',
}

# Uremic toxins for metabolomics
UREMIC_TOXINS = {
    'indoxyl sulfate', 'p-cresol', 'hippuric acid', 'uric acid',
    'creatinine', 'urea', 'trimethylamine', 'phenol', 'guanidine',
    'asymmetric dimethylarginine', 'adma', 'kynurenic acid', 'kynurenine',
    'indole', '3-indoxyl', 'phenylacetic acid', 'quinolinic acid',
}

# ── Global caches (populated by load_global_caches in main process) ───────────
_GENE_EMBEDDINGS      = None
_CELL_EMBEDDINGS      = None
_DISEASE_EMBEDDINGS   = None
_MODALITY_CLASSIFICATION = None
_EXCLUSION_SET        = None
_GENE_NORMALIZER      = None
_PCA_BASIS            = None
_PCA_MEAN             = None


def load_global_caches():
    global _GENE_EMBEDDINGS, _CELL_EMBEDDINGS, _DISEASE_EMBEDDINGS
    global _MODALITY_CLASSIFICATION, _EXCLUSION_SET, _GENE_NORMALIZER
    global _PCA_BASIS, _PCA_MEAN

    gene_path = PATHS.get('gene_embeddings', '')
    if os.path.exists(gene_path):
        d = np.load(gene_path, allow_pickle=True)
        _GENE_EMBEDDINGS = {k: d[k].astype(np.float32) for k in d.files}
        logger.info(f"Loaded {len(_GENE_EMBEDDINGS)} gene embeddings")
    else:
        _GENE_EMBEDDINGS = {}

    cell_path = PATHS.get('celltype_embeddings', '')
    if os.path.exists(cell_path):
        d = np.load(cell_path, allow_pickle=True)
        _CELL_EMBEDDINGS = {k: d[k].astype(np.float32) for k in d.files}
    else:
        _CELL_EMBEDDINGS = {}

    disease_path = PATHS.get('disease_embeddings', '')
    if os.path.exists(disease_path):
        d = np.load(disease_path, allow_pickle=True)
        _DISEASE_EMBEDDINGS = {k: d[k].astype(np.float32) for k in d.files}
    else:
        _DISEASE_EMBEDDINGS = {}

    class_path = PATHS.get('multimodal_classification', '')
    if os.path.exists(class_path):
        with open(class_path) as f:
            raw = json.load(f)
        _MODALITY_CLASSIFICATION = {}
        if 'files_by_modality' in raw:
            for mod, files in raw['files_by_modality'].items():
                for acc in files:
                    _MODALITY_CLASSIFICATION[acc] = mod
        else:
            _MODALITY_CLASSIFICATION = raw
        logger.info(f"Modality classification: {len(_MODALITY_CLASSIFICATION)} accessions")
    else:
        _MODALITY_CLASSIFICATION = {}

    excl_path = PATHS.get('exclusion_list', '')
    if os.path.exists(excl_path):
        with open(excl_path) as f:
            d = json.load(f)
        _EXCLUSION_SET = set()
        for items in d.values():
            if isinstance(items, list):
                _EXCLUSION_SET.update(items)
    else:
        _EXCLUSION_SET = set()

    _GENE_NORMALIZER = GeneIDNormalizer()

    pca_path = PATHS.get('genome_pca_basis',
                         '/mnt/nvme15tb/trained_im/node_embeddings/genome_pca_basis.npz')
    if os.path.exists(pca_path):
        d = np.load(pca_path, allow_pickle=True)
        _PCA_BASIS = d['components'].astype(np.float32)
        _PCA_MEAN  = d['mean'].astype(np.float32)
        logger.info(f"PCA basis: {_PCA_BASIS.shape}")
    else:
        logger.warning("PCA basis not found — Tier 3 genome context will be zeros")

    # V3 addition: load contamination audit to skip at build time
    contam_path = '/mnt/nvme15tb/trained_im/outputs/profiling_v3_1/demographics/contamination_audit.csv'
    if os.path.exists(contam_path):
        import csv as _csv
        with open(contam_path) as f:
            reader = _csv.DictReader(f)
            contaminated = {row['study_acc'] for row in reader}
        # Merge with existing exclusion set
        _EXCLUSION_SET.update(contaminated)
        logger.info(f"Contamination exclusions loaded: {len(contaminated)} studies added to exclusion set")
    else:
        logger.warning(f"contamination_audit.csv not found — no contamination exclusions applied")


# ── Shared helpers ────────────────────────────────────────────────────────────
def normalize_disease(s):
    s = str(s).lower().strip()
    if any(k in s for k in ['reject', 'tcmr', 'abmr', 'acute']):    return 'acute_rejection'
    if any(k in s for k in ['chronic', 'ifta', 'fibrosis']):         return 'chronic_rejection'
    if any(k in s for k in ['healthy', 'normal', 'control']):        return 'kidney_transplant'
    if any(k in s for k in ['trained', 'bcg', 'beta-glucan']):       return 'trained_immunity'
    if any(k in s for k in ['dialysis', 'hemodialysis', 'esrd']):    return 'hemodialysis'
    if any(k in s for k in ['ckd', 'kidney disease', 'renal']):      return 'ckd'
    return 'unknown'

def get_disease_label(npz, sample_idx=None):
    if 'disease_class' in npz.files:
        dc = npz['disease_class']
        if hasattr(dc, 'item'): dc = dc.item()
        dc = str(dc).strip()
        return dc if dc in LABEL_MAP else normalize_disease(dc)
    if 'labels' in npz.files and sample_idx is not None:
        arr = npz['labels']
        if sample_idx < len(arr):
            v = int(arr[sample_idx])
            v = v if 0 <= v <= 5 else 2
            return {0:'acute_rejection',1:'chronic_rejection',2:'ckd',
                    3:'hemodialysis',4:'kidney_transplant',5:'trained_immunity'}.get(v, 'ckd')
    return 'unknown'

def build_modality_flag(modality):
    idx = MODALITY_INDEX.get(modality, 11)
    f = np.zeros((1, 12), dtype=np.float32)
    f[0, idx] = 1.0
    return torch.tensor(f)

def _stub_pathway_attrs(graph):
    """Attach zero pathway_gene_masks and module_pathway_ids (non-gene modalities)."""
    graph.pathway_gene_masks  = torch.zeros(len(MODULE_NAMES), 1, dtype=torch.bool)
    graph.module_pathway_ids  = MODULE_PATHWAY_IDS.clone()
    return graph

# CANONICAL MARKER GENES per cell type — used to derive activation from expression
CELL_TYPE_MARKER_GENES = {
    'monocyte_classical':    ['CD14','S100A8','S100A9','CD68','LYZ','FCN1','VCAN','CSF1R'],
    'monocyte_nonclassical': ['CD16','FCGR3A','CX3CR1','CDKN1C','HES4','LST1'],
    'macrophage_M1':         ['CD68','IL1B','TNF','CCL2','CXCL10','NOS2','MRC1'],
    'macrophage_M2':         ['MRC1','CD163','TGFB1','CCL18','IL10','HMOX1','ARG1'],
    'CD4_Tcell':             ['CD4','CD3D','CD3E','IL7R','CCR7','TCF7','SELL'],
    'CD8_Tcell':             ['CD8A','CD8B','GZMB','PRF1','NKG7','GNLY','IFNG'],
    'Treg':                  ['FOXP3','IL2RA','CTLA4','IKZF2','TNFRSF18','TIGIT'],
    'NK_cell':               ['NKG7','GNLY','NCAM1','KLRD1','KLRB1','FGFBP2','PRF1'],
    'B_cell':                ['CD19','MS4A1','CD79A','CD79B','IGHM','PAX5','BANK1'],
    'dendritic_cDC1':        ['CLEC9A','CADM1','XCR1','BATF3','IRF8','WDFY4'],
    'dendritic_cDC2':        ['CD1C','CLEC10A','FCER1A','HLA-DRA','ITGAM','SIRPA'],
    'MAIT_cell':             ['SLC4A10','NCR3','KLRB1','RORC','IL18RAP','ZBTB16'],
}

def _build_common_nodes(graph, label, expr_norm=None, gene_upper=None):
    """
    Attach cell_type, disease_state, modality_flag nodes and standard edges.

    FIX V3: cell_type node features are now expression-data-aware.
    When expr_norm and gene_upper are provided (gene_expression modality):
      - For each cell type, compute mean z-scored expression of its canonical
        marker genes present in the data.
      - Multiply the static embedding by this activity scalar (clamped to [0.1, 5.0]).
      - This makes cell_type.x data-dependent rather than a static lookup table.
    When expr_norm is None (non-gene modalities like methylation, chromatin):
      - Falls back to static embeddings (correct — no expression to derive from).
    """
    n_mod = len(MODULE_NAMES)
    cell_feats = np.zeros((len(CANONICAL_CELL_TYPES), 512), dtype=np.float32)

    for i, ct in enumerate(CANONICAL_CELL_TYPES):
        # Get static base embedding
        base_emb = np.zeros(512, dtype=np.float32)
        if _CELL_EMBEDDINGS and ct in _CELL_EMBEDDINGS:
            base_emb = _CELL_EMBEDDINGS[ct][:512]

        if expr_norm is not None and gene_upper is not None:
            # Derive activation scalar from marker gene expression
            markers = CELL_TYPE_MARKER_GENES.get(ct, [])
            marker_vals = []
            for mg in markers:
                mgu = mg.upper()
                if mgu in gene_upper:
                    marker_vals.append(float(expr_norm[gene_upper[mgu]]))
            if marker_vals:
                # Mean marker expression, clamped to avoid extreme scaling
                activity = float(np.mean(marker_vals))
                # Shift to positive range: relu-like with floor at 0.1
                scalar = float(np.clip(activity + 1.0, 0.1, 5.0))
                cell_feats[i] = base_emb * scalar
            else:
                # No marker genes found in this dataset — use static embedding
                cell_feats[i] = base_emb
        else:
            # Non-gene-expression modality — static embedding only
            cell_feats[i] = base_emb

    graph['cell_type'].x = torch.tensor(cell_feats, dtype=torch.float32)

    ct_src, ct_dst = [], []
    for ci, ct in enumerate(CANONICAL_CELL_TYPES):
        for mod in CELL_TYPE_MODULE_EDGES.get(ct, []):
            if mod in MODULE_NAMES:
                ct_src.append(ci); ct_dst.append(MODULE_NAMES.index(mod))
    if ct_src:
        graph['cell_type','activates','module'].edge_index = torch.tensor(
            [ct_src, ct_dst], dtype=torch.long)

    dis_feats = np.zeros((1, 768), dtype=np.float32)
    if _DISEASE_EMBEDDINGS and label in _DISEASE_EMBEDDINGS:
        dis_feats[0] = _DISEASE_EMBEDDINGS[label][:768]
    graph['disease_state'].x = torch.tensor(dis_feats, dtype=torch.float32)

    # Standard tier→disease edges
    graph['module','summary_of','disease_state'].edge_index = torch.tensor(
        [list(range(n_mod)), [0]*n_mod], dtype=torch.long)
    graph['pathway','summary_of','disease_state'].edge_index = torch.tensor(
        [list(range(N_EXTENDED_PATHWAYS)), [0]*N_EXTENDED_PATHWAYS], dtype=torch.long)
    graph['genome_context','context_of','disease_state'].edge_index = torch.tensor(
        [[0],[0]], dtype=torch.long)
    return graph

def _pca_context(expr_vec):
    """Compute Tier 3 genome context via PCA projection."""
    if _PCA_BASIS is None:
        return np.zeros((1, 128), dtype=np.float32)
    pca_dim = _PCA_BASIS.shape[1]
    n = len(expr_vec)
    if n < pca_dim:
        padded = np.zeros(pca_dim, dtype=np.float32)
        padded[:n] = expr_vec
        centered = padded - _PCA_MEAN
    elif n > pca_dim:
        centered = expr_vec[:pca_dim] - _PCA_MEAN
    else:
        centered = expr_vec - _PCA_MEAN
    return (_PCA_BASIS @ centered).reshape(1, -1)


# ── build_gene_expression_graph (V1, with pathway_gene_masks) ────────────────
def build_gene_expression_graph(expr_vec, gene_names, label, modality, accession, sample_idx):
    gene_dim = 512
    n_mod    = len(MODULE_NAMES)

    expr_vec = np.nan_to_num(expr_vec, nan=0.0, posinf=0.0, neginf=0.0)
    mean, std = np.nanmean(expr_vec), np.nanstd(expr_vec) + 1e-8
    expr_norm = np.clip((expr_vec - mean) / std, -5, 5)
    expr_norm = np.nan_to_num(expr_norm, nan=0.0)

    gene_upper = {str(g).upper(): i for i, g in enumerate(gene_names) if g}
    graph = HeteroData()

    # Tier 1
    canonical_upper = set()
    for gs in CANONICAL_MODULES.values():
        canonical_upper.update(g.upper() for g in gs)

    gene_list = []
    for gu in canonical_upper:
        if gu in gene_upper:
            idx = gene_upper[gu]
            for orig in gene_names:
                if orig and str(orig).upper() == gu:
                    if _GENE_EMBEDDINGS and orig in _GENE_EMBEDDINGS:
                        gene_list.append((orig, expr_norm[idx], _GENE_EMBEDDINGS[orig]))
                    break

    if gene_list:
        gf = np.zeros((len(gene_list), gene_dim), dtype=np.float32)
        gene_name_to_idx = {}
        emb_lookup = {}
        for i, (gn, ev, emb) in enumerate(gene_list):
            gf[i] = emb * np.clip(ev, -3, 3)
            gene_name_to_idx[gn.upper()] = i
            emb_lookup[gn.upper()] = (i, emb * np.clip(ev, -3, 3))
        graph['gene'].x = torch.tensor(gf, dtype=torch.float32)
    else:
        graph['gene'].x = torch.zeros((1, gene_dim), dtype=torch.float32)
        gene_name_to_idx = {}
        emb_lookup = {}

    module_feats = np.zeros((n_mod, gene_dim), dtype=np.float32)
    gm_src, gm_dst = [], []
    for m_idx, (mod_name, mod_genes) in enumerate(CANONICAL_MODULES.items()):
        member_embs = []
        for g in mod_genes:
            gu = g.upper()
            if gu in emb_lookup:
                gi, scaled = emb_lookup[gu]
                member_embs.append(scaled)
                gm_src.append(gi); gm_dst.append(m_idx)
        if member_embs:
            module_feats[m_idx] = np.mean(member_embs, axis=0)
    graph['module'].x = torch.tensor(module_feats, dtype=torch.float32)

    if gm_src:
        graph['gene','belongs_to','module'].edge_index = torch.tensor(
            [gm_src, gm_dst], dtype=torch.long)
        if MTORI_HDL_BIAS:
            ew = np.array([
                MTOR_CENTRALITY_PRIOR if gene_list[gi][0].upper() in MTOR_CENTRALITY_GENES else 1.0
                for gi in gm_src], dtype=np.float32)
            if len(ew): ew /= ew.mean()
            graph['gene','belongs_to','module'].edge_attr = torch.tensor(
                ew, dtype=torch.float32).unsqueeze(1)

    # Tier 2
    pf = np.zeros((N_EXTENDED_PATHWAYS, 2), dtype=np.float32)
    pm_src, pm_dst = [], []
    for p_idx, (pname, pgenes) in enumerate(EXTENDED_PATHWAY_MODULES.items()):
        pu = set(g.upper() for g in pgenes)
        vals = [expr_norm[gene_upper[g]] for g in pu if g in gene_upper]
        if vals:
            pf[p_idx, 0] = np.mean(vals); pf[p_idx, 1] = np.var(vals)
        for m_idx, (mname, mgenes) in enumerate(CANONICAL_MODULES.items()):
            if pu & set(g.upper() for g in mgenes):
                pm_src.append(p_idx); pm_dst.append(m_idx)
    graph['pathway'].x = torch.tensor(pf, dtype=torch.float32)
    if pm_src:
        graph['pathway','related_to','module'].edge_index = torch.tensor(
            [pm_src, pm_dst], dtype=torch.long)

    # Tier 3
    graph['genome_context'].x = torch.tensor(_pca_context(expr_vec), dtype=torch.float32)

    # Common nodes + edges
    graph['modality_flag'].x = build_modality_flag(modality)
    graph = _build_common_nodes(graph, label, expr_norm=expr_norm, gene_upper=gene_upper)

    # pathway_gene_masks for pathway supervision
    n_g = len(gene_list)
    if n_g > 0:
        pw = torch.zeros(len(MODULE_NAMES), n_g, dtype=torch.bool)
        for m_idx, gene_set_upper in _PATHWAY_GENE_SETS_UPPER.items():
            for gi, (gn, _, _) in enumerate(gene_list):
                if gn.upper() in gene_set_upper:
                    pw[m_idx, gi] = True
        graph.pathway_gene_masks = pw
    else:
        graph.pathway_gene_masks = torch.zeros(len(MODULE_NAMES), 1, dtype=torch.bool)
    graph.module_pathway_ids = MODULE_PATHWAY_IDS.clone()

    graph.y             = torch.tensor([LABEL_MAP.get(label, -1)], dtype=torch.long)
    graph.accession     = accession
    graph.sample_idx    = sample_idx
    graph.modality      = modality
    graph.dataset_weight = 2.0 if accession in HIGH_VALUE_DATASETS else 1.0

    for nt in graph.node_types:
        if hasattr(graph[nt], 'x') and graph[nt].x is not None:
            if torch.isnan(graph[nt].x).any():
                raise ValueError(f"NaN in {nt} for {accession}/{sample_idx}")
    return graph


# ── build_scrna_graph ─────────────────────────────────────────────────────────
def build_scrna_graph(expr_vec, gene_names, label, accession, sample_idx, pre_normalized=False):
    """
    scRNA-seq (per-cell or pseudobulk).
    If gene_names are mappable HGNC symbols (coverage ≥ SCRNA_COVERAGE_THRESHOLD),
    delegates to the gene_expression builder to get full embedding-based features.
    Falls back to a raw PCA-compressed scrna_cell node if coverage is too low.
    """
    # Attempt gene expression path (lower threshold for scrna)
    known = set(_GENE_EMBEDDINGS.keys()) if _GENE_EMBEDDINGS else set()
    if pre_normalized:
        normalized = gene_names
    else:
        normalized, _ = _GENE_NORMALIZER.normalize(gene_names, accession) \
            if _GENE_NORMALIZER else (gene_names, 0.0)
    found = sum(1 for s in normalized if s and s in known)
    coverage = found / len(gene_names) if gene_names else 0.0

    if coverage >= SCRNA_COVERAGE_THRESHOLD:
        # Use full gene_expression path — returns complete three-tier graph
        return build_gene_expression_graph(
            expr_vec, normalized, label, 'scrna', accession, sample_idx)

    # ── Fallback: raw PCA-compressed scrna_cell node ──────────────────────────
    graph = HeteroData()
    n_mod = len(MODULE_NAMES)

    # scrna_cell node: top 512 features by absolute value or PCA-projected
    MAX_CELL_FEATS = 512
    vec = np.nan_to_num(expr_vec, nan=0.0)
    if len(vec) > MAX_CELL_FEATS:
        top_idx = np.argsort(np.abs(vec))[-MAX_CELL_FEATS:]
        vec_trunc = np.zeros(MAX_CELL_FEATS, dtype=np.float32)
        vec_trunc[np.arange(MAX_CELL_FEATS)] = vec[top_idx]
        cell_feat = vec_trunc.reshape(1, -1)
    else:
        cell_feat = np.zeros((1, MAX_CELL_FEATS), dtype=np.float32)
        cell_feat[0, :len(vec)] = vec
    graph['scrna_cell'].x = torch.tensor(cell_feat, dtype=torch.float32)

    # Placeholder gene and module nodes for schema consistency
    graph['gene'].x           = torch.zeros((0, 512),              dtype=torch.float32)
    graph['module'].x         = torch.zeros((n_mod, 512),          dtype=torch.float32)
    graph['pathway'].x        = torch.zeros((N_EXTENDED_PATHWAYS, 2), dtype=torch.float32)
    graph['genome_context'].x = torch.zeros((1, 128),              dtype=torch.float32)
    graph['modality_flag'].x  = build_modality_flag('scrna')

    graph = _build_common_nodes(graph, label)
    graph = _stub_pathway_attrs(graph)

    graph.y          = torch.tensor([LABEL_MAP.get(label, -1)], dtype=torch.long)
    graph.accession  = accession
    graph.sample_idx = sample_idx
    graph.modality   = 'scrna'
    return graph


# ── build_cytof_graph ─────────────────────────────────────────────────────────
def build_cytof_graph(intensity_vec, marker_names, label, accession, sample_idx):
    """
    CyTOF / mass cytometry.
    Node type: cytof_marker (n_markers, 64)
      - Feature 0: arcsinh-transformed intensity (cofactor=5)
      - Feature 1: log1p intensity
      - Feature 2-9: cell-type association flags (one-hot, 8 major cell types)
      - Feature 10-23: module association flags (one-hot, 14 modules)

    cytof_marker → cell_type edges where marker is a known cell-type surface marker.
    """
    graph = HeteroData()
    n_mod = len(MODULE_NAMES)
    n_markers = len(marker_names)

    # arcsinh transform with cofactor=5 (standard CyTOF preprocessing)
    cofactor = 5.0
    arcsinh_int = np.arcsinh(intensity_vec / cofactor).astype(np.float32)
    log1p_int   = np.log1p(np.abs(intensity_vec)).astype(np.float32)

    N_FEAT = 64
    feat = np.zeros((n_markers, N_FEAT), dtype=np.float32)
    feat[:, 0] = arcsinh_int
    feat[:, 1] = log1p_int

    # Cell-type association flags (features 2-9)
    ct_idx_map = {ct: i for i, ct in enumerate(CANONICAL_CELL_TYPES)}
    marker_to_ct_idx  = {}
    marker_to_mod_idx = {}
    for i, mname in enumerate(marker_names):
        # Normalize: strip clone names, lowercase, try substring matching
        mn_clean = str(mname).split('_')[0].split('-')[0].strip()
        # Cell type lookup
        for key, ct in CYTOF_MARKER_CELL_TYPE.items():
            if key.lower() in mn_clean.lower() or mn_clean.lower() in key.lower():
                ci = ct_idx_map.get(ct, -1)
                if ci >= 0:
                    feat[i, 2 + min(ci, 7)] = 1.0
                    marker_to_ct_idx[i] = ci
                break
        # Module lookup
        for key, mod in CYTOF_MARKER_MODULE.items():
            if key.lower() in mn_clean.lower() or mn_clean.lower() in key.lower():
                mi = MODULE_NAMES.index(mod) if mod in MODULE_NAMES else -1
                if mi >= 0:
                    feat[i, 10 + min(mi, 13)] = 1.0
                    marker_to_mod_idx[i] = mi
                break

    graph['cytof_marker'].x = torch.tensor(feat, dtype=torch.float32)

    # cytof_marker → cell_type edges
    cyt_ct_src, cyt_ct_dst = [], []
    for mi, ci in marker_to_ct_idx.items():
        cyt_ct_src.append(mi); cyt_ct_dst.append(ci)
    if cyt_ct_src:
        graph['cytof_marker','stains','cell_type'].edge_index = torch.tensor(
            [cyt_ct_src, cyt_ct_dst], dtype=torch.long)

    # cytof_marker → module edges (functional markers)
    cyt_mod_src, cyt_mod_dst = [], []
    for mi, mdi in marker_to_mod_idx.items():
        cyt_mod_src.append(mi); cyt_mod_dst.append(mdi)
    if cyt_mod_src:
        graph['cytof_marker','signals','module'].edge_index = torch.tensor(
            [cyt_mod_src, cyt_mod_dst], dtype=torch.long)

    # Placeholder gene / pathway / genome nodes
    graph['gene'].x           = torch.zeros((0, 512),              dtype=torch.float32)
    graph['module'].x         = torch.zeros((n_mod, 512),          dtype=torch.float32)
    graph['pathway'].x        = torch.zeros((N_EXTENDED_PATHWAYS, 2), dtype=torch.float32)
    graph['genome_context'].x = torch.zeros((1, 128),              dtype=torch.float32)
    graph['modality_flag'].x  = build_modality_flag('cytof')

    graph = _build_common_nodes(graph, label)
    graph = _stub_pathway_attrs(graph)

    graph.y          = torch.tensor([LABEL_MAP.get(label, -1)], dtype=torch.long)
    graph.accession  = accession
    graph.sample_idx = sample_idx
    graph.modality   = 'cytof'
    return graph


# ── build_hic_graph ───────────────────────────────────────────────────────────
def build_hic_graph(contact_vec, region_ids, label, accession, sample_idx):
    """
    Hi-C chromatin contact data.
    Expects contact_vec to be either:
      (a) flattened upper-triangle of a contact matrix (n_bins*(n_bins-1)/2,)
      (b) already a 1-D per-region summary vector
    Node type: hic_region (n_regions, 64)
      - Feature 0: mean intra-TAD contact frequency (log-scaled)
      - Feature 1: A/B compartment score proxy (sign of PC1, estimated from contact decay)
      - Feature 2: insulation score proxy
    """
    graph = HeteroData()
    n_mod = len(MODULE_NAMES)

    MAX_REGIONS = 1000
    n_raw = len(contact_vec)

    # Reconstruct per-region summary from upper-triangle vector or direct vector
    # Try to infer n_bins if contact_vec looks like a flattened upper triangle
    # k = n*(n-1)/2  →  n ≈ (1 + sqrt(1 + 8k)) / 2
    n_bins_est = int((1 + (1 + 8 * n_raw) ** 0.5) / 2)
    if n_bins_est * (n_bins_est - 1) // 2 == n_raw and n_bins_est > 10:
        # Unflatten upper triangle to get per-row (per-region) sum
        mat_sum = np.zeros(n_bins_est, dtype=np.float32)
        idx = 0
        for i in range(n_bins_est):
            for j in range(i + 1, n_bins_est):
                mat_sum[i] += contact_vec[idx]
                mat_sum[j] += contact_vec[idx]
                idx += 1
        region_feat_vec = mat_sum
        n_regions = n_bins_est
    else:
        # Treat as direct per-region vector
        region_feat_vec = contact_vec.astype(np.float32)
        n_regions = n_raw

    # Cap regions
    if n_regions > MAX_REGIONS:
        top = np.argsort(region_feat_vec)[-MAX_REGIONS:]
        region_feat_vec = region_feat_vec[top]
        n_regions = MAX_REGIONS

    # Log-scale and normalise
    log_vec = np.log1p(np.abs(region_feat_vec)).astype(np.float32)
    if log_vec.std() > 0:
        norm_vec = (log_vec - log_vec.mean()) / log_vec.std()
    else:
        norm_vec = log_vec

    # Estimate A/B compartment: positive norm = A (active), negative = B (inactive)
    compartment = np.sign(norm_vec).astype(np.float32)  # +1 A, -1 B, 0 boundary

    # Simple insulation score proxy: local variance across neighbours
    insulation = np.zeros(n_regions, dtype=np.float32)
    for i in range(1, n_regions - 1):
        insulation[i] = np.abs(norm_vec[i] - 0.5 * (norm_vec[i-1] + norm_vec[i+1]))

    feat = np.zeros((n_regions, 64), dtype=np.float32)
    feat[:, 0] = norm_vec
    feat[:, 1] = compartment
    feat[:, 2] = insulation

    graph['hic_region'].x = torch.tensor(feat, dtype=torch.float32)

    # Placeholder nodes
    graph['gene'].x           = torch.zeros((0, 512),              dtype=torch.float32)
    graph['module'].x         = torch.zeros((n_mod, 512),          dtype=torch.float32)
    graph['pathway'].x        = torch.zeros((N_EXTENDED_PATHWAYS, 2), dtype=torch.float32)
    graph['genome_context'].x = torch.zeros((1, 128),              dtype=torch.float32)
    graph['modality_flag'].x  = build_modality_flag('hic')

    graph = _build_common_nodes(graph, label)
    graph = _stub_pathway_attrs(graph)

    graph.y          = torch.tensor([LABEL_MAP.get(label, -1)], dtype=torch.long)
    graph.accession  = accession
    graph.sample_idx = sample_idx
    graph.modality   = 'hic'
    return graph


# ── build_chromatin_graph ─────────────────────────────────────────────────────
def build_chromatin_graph(signal_vec, region_names, label, accession, sample_idx):
    MAX_REGIONS = 5000
    graph = HeteroData()
    n_mod = len(MODULE_NAMES)

    if len(signal_vec) > MAX_REGIONS:
        top = np.argsort(np.abs(signal_vec))[-MAX_REGIONS:]
        signal_vec = signal_vec[top]

    mean, std = np.mean(signal_vec), np.std(signal_vec) + 1e-8
    snorm = np.clip((signal_vec - mean) / std, -5, 5)
    graph['chromatin_region'].x = torch.tensor(snorm.reshape(-1, 1), dtype=torch.float32)

    graph['gene'].x           = torch.zeros((0, 512),              dtype=torch.float32)
    graph['module'].x         = torch.zeros((n_mod, 512),          dtype=torch.float32)
    graph['pathway'].x        = torch.zeros((N_EXTENDED_PATHWAYS, 2), dtype=torch.float32)
    graph['genome_context'].x = torch.zeros((1, 128),              dtype=torch.float32)
    graph['modality_flag'].x  = build_modality_flag('chromatin')

    graph = _build_common_nodes(graph, label)
    graph = _stub_pathway_attrs(graph)

    graph.y = torch.tensor([LABEL_MAP.get(label, -1)], dtype=torch.long)
    graph.accession = accession; graph.sample_idx = sample_idx; graph.modality = 'chromatin'
    return graph


# ── build_methylation_graph ───────────────────────────────────────────────────
def build_methylation_graph(beta_vec, cpg_names, label, accession, sample_idx):
    MAX_CPGS = 5000
    graph = HeteroData()
    n_mod = len(MODULE_NAMES)

    if len(beta_vec) > MAX_CPGS:
        top = np.argsort(np.abs(beta_vec - 0.5))[-MAX_CPGS:]
        beta_vec = beta_vec[top]

    graph['cpg'].x            = torch.tensor(beta_vec.reshape(-1,1), dtype=torch.float32)
    graph['gene'].x           = torch.zeros((0, 512),                dtype=torch.float32)
    graph['module'].x         = torch.zeros((n_mod, 512),            dtype=torch.float32)
    graph['pathway'].x        = torch.zeros((N_EXTENDED_PATHWAYS, 2),dtype=torch.float32)
    graph['genome_context'].x = torch.zeros((1, 128),                dtype=torch.float32)
    graph['modality_flag'].x  = build_modality_flag('methylation')

    graph = _build_common_nodes(graph, label)
    graph = _stub_pathway_attrs(graph)

    graph.y = torch.tensor([LABEL_MAP.get(label,-1)], dtype=torch.long)
    graph.accession = accession; graph.sample_idx = sample_idx; graph.modality = 'methylation'
    return graph


# ── build_mirna_graph ─────────────────────────────────────────────────────────
def build_mirna_graph(expr_vec, mirna_names, label, accession, sample_idx):
    n = len(mirna_names)
    graph = HeteroData()
    n_mod = len(MODULE_NAMES)

    mean, std = np.mean(expr_vec), np.std(expr_vec) + 1e-8
    enorm = np.clip((expr_vec - mean) / std, -5, 5)
    feat = np.zeros((n, 64), dtype=np.float32)
    feat[:, 0] = enorm
    graph['mirna'].x = torch.tensor(feat, dtype=torch.float32)

    graph['gene'].x           = torch.zeros((0, 512),              dtype=torch.float32)
    graph['module'].x         = torch.zeros((n_mod, 512),          dtype=torch.float32)
    graph['pathway'].x        = torch.zeros((N_EXTENDED_PATHWAYS, 2), dtype=torch.float32)
    graph['genome_context'].x = torch.zeros((1, 128),              dtype=torch.float32)
    graph['modality_flag'].x  = build_modality_flag('mirna')

    graph = _build_common_nodes(graph, label)
    graph = _stub_pathway_attrs(graph)

    graph.y = torch.tensor([LABEL_MAP.get(label,-1)], dtype=torch.long)
    graph.accession = accession; graph.sample_idx = sample_idx; graph.modality = 'mirna'
    return graph


# ── build_snp_graph ───────────────────────────────────────────────────────────
def build_snp_graph(geno_vec, snp_ids, label, accession, sample_idx):
    MAX_SNPS = 1000
    n = len(snp_ids)
    graph = HeteroData()
    n_mod = len(MODULE_NAMES)

    if n > MAX_SNPS:
        nz = np.where(geno_vec != 0)[0]
        sel = nz[:MAX_SNPS] if len(nz) >= MAX_SNPS else \
              np.concatenate([nz, np.random.choice(
                  np.where(geno_vec == 0)[0],
                  MAX_SNPS - len(nz), replace=False)]) if len(nz) > 0 else \
              np.random.choice(n, MAX_SNPS, replace=False)
        geno_vec = geno_vec[sel]; n = MAX_SNPS

    feat = np.zeros((n, 64), dtype=np.float32)
    feat[:, 0] = np.clip(geno_vec, 0, 2) / 2.0   # normalise 0/1/2 → 0/0.5/1
    graph['snp'].x = torch.tensor(feat, dtype=torch.float32)

    graph['gene'].x           = torch.zeros((0, 512),              dtype=torch.float32)
    graph['module'].x         = torch.zeros((n_mod, 512),          dtype=torch.float32)
    graph['pathway'].x        = torch.zeros((N_EXTENDED_PATHWAYS, 2), dtype=torch.float32)
    graph['genome_context'].x = torch.zeros((1, 128),              dtype=torch.float32)
    graph['modality_flag'].x  = build_modality_flag('snp')

    graph = _build_common_nodes(graph, label)
    graph = _stub_pathway_attrs(graph)

    graph.y = torch.tensor([LABEL_MAP.get(label,-1)], dtype=torch.long)
    graph.accession = accession; graph.sample_idx = sample_idx; graph.modality = 'snp'
    return graph


# ── build_metabolite_graph ────────────────────────────────────────────────────
def build_metabolite_graph(intensity_vec, met_ids, label, accession, sample_idx):
    n = len(met_ids)
    graph = HeteroData()
    n_mod = len(MODULE_NAMES)

    log_int = np.log1p(np.abs(intensity_vec)).astype(np.float32)
    norm_int = (log_int - log_int.mean()) / (log_int.std() + 1e-8)

    feat = np.zeros((n, 64), dtype=np.float32)
    feat[:, 0] = norm_int
    for i, mid in enumerate(met_ids):
        ml = str(mid).lower().replace('_',' ').replace('-',' ')
        for toxin in UREMIC_TOXINS:
            if toxin in ml or ml in toxin:
                feat[i, 1] = 1.0; break
    graph['metabolite'].x = torch.tensor(feat, dtype=torch.float32)

    graph['gene'].x           = torch.zeros((0, 512),              dtype=torch.float32)
    graph['module'].x         = torch.zeros((n_mod, 512),          dtype=torch.float32)
    graph['pathway'].x        = torch.zeros((N_EXTENDED_PATHWAYS, 2), dtype=torch.float32)
    graph['genome_context'].x = torch.zeros((1, 128),              dtype=torch.float32)
    graph['modality_flag'].x  = build_modality_flag('metabolite')

    graph = _build_common_nodes(graph, label)
    graph = _stub_pathway_attrs(graph)

    graph.y = torch.tensor([LABEL_MAP.get(label,-1)], dtype=torch.long)
    graph.accession = accession; graph.sample_idx = sample_idx; graph.modality = 'metabolite'
    return graph


# ── GPU worker (gene_expression large files only) ────────────────────────────
def _gpu_worker(gpu_id, shm_name, shape, is_transposed,
                gene_names, modality, label, accession,
                task_queue, result_queue, lmdb_path):
    """One worker per GPU. Reads from shared memory, builds graphs, writes LMDB."""
    import lmdb as _lmdb

    device = torch.device(f'cuda:{gpu_id}')
    torch.cuda.set_device(device)

    # Load embeddings locally (can't use global state across spawn)
    gene_emb, cell_emb, dis_emb = {}, {}, {}
    pca_b, pca_m = None, None

    gp = PATHS.get('gene_embeddings', '')
    if os.path.exists(gp):
        d = np.load(gp, allow_pickle=True)
        gene_emb = {k: d[k].astype(np.float32) for k in d.files}

    cp = PATHS.get('celltype_embeddings', '')
    if os.path.exists(cp):
        d = np.load(cp, allow_pickle=True)
        cell_emb = {k: d[k].astype(np.float32) for k in d.files}

    dp = PATHS.get('disease_embeddings', '')
    if os.path.exists(dp):
        d = np.load(dp, allow_pickle=True)
        dis_emb = {k: d[k].astype(np.float32) for k in d.files}

    pp = PATHS.get('genome_pca_basis', '')
    if os.path.exists(pp):
        d = np.load(pp, allow_pickle=True)
        pca_b = d['components'].astype(np.float32)
        pca_m = d['mean'].astype(np.float32)

    logger.info(f"[GPU {gpu_id}] ready: genes={len(gene_emb)}")

    # Attach shared memory
    shm = shm_mod.SharedMemory(name=shm_name)
    mat = np.ndarray(shape, dtype=np.float32, buffer=shm.buf)

    env = _lmdb.open(lmdb_path, map_size=LMDB_MAP_LARGE, writemap=True)
    built = 0

    while True:
        try:
            task = task_queue.get(timeout=30)
        except Exception:
            break
        if task is None:
            break

        s_start, s_end = task
        with env.begin(write=True) as txn:
            for si in range(s_start, s_end):
                key = f"{accession}_s{si:06d}".encode()
                if txn.get(key) is not None:
                    built += 1; continue
                vec = mat[:, si].astype(np.float32) if is_transposed \
                      else mat[si, :].astype(np.float32)
                try:
                    g = _build_gene_expr_worker(
                        vec, gene_names, label, modality, accession, si,
                        gene_emb, cell_emb, dis_emb, pca_b, pca_m)
                    txn.put(key, pickle.dumps(g))
                    built += 1
                except Exception as e:
                    logger.warning(f"[GPU {gpu_id}] s{si}: {e}")

        logger.info(f"[GPU {gpu_id}] chunk {s_start}-{s_end} done")

    env.close()
    shm.close()
    result_queue.put((gpu_id, built))
    logger.info(f"[GPU {gpu_id}] finished: {built} graphs")


def _build_gene_expr_worker(expr_vec, gene_names, label, modality, accession, si,
                             gene_emb, cell_emb, dis_emb, pca_b, pca_m):
    """Stateless graph builder for GPU workers (no global state access)."""
    gene_dim = 512
    n_mod    = len(MODULE_NAMES)

    expr_vec = np.nan_to_num(expr_vec, nan=0.0, posinf=0.0, neginf=0.0)
    mean, std = np.nanmean(expr_vec), np.nanstd(expr_vec) + 1e-8
    enorm = np.clip((expr_vec - mean) / std, -5, 5)
    enorm = np.nan_to_num(enorm, nan=0.0)

    gene_upper = {str(g).upper(): i for i, g in enumerate(gene_names) if g}
    graph = HeteroData()

    canonical_upper = set()
    for gs in CANONICAL_MODULES.values():
        canonical_upper.update(g.upper() for g in gs)

    gene_list, emb_lookup = [], {}
    for gu in canonical_upper:
        if gu in gene_upper:
            idx = gene_upper[gu]
            for orig in gene_names:
                if orig and str(orig).upper() == gu and orig in gene_emb:
                    ev = enorm[idx]
                    scaled = gene_emb[orig] * np.clip(ev, -3, 3)
                    gene_list.append((orig, ev, gene_emb[orig]))
                    emb_lookup[gu] = (len(gene_list)-1, scaled)
                    break

    if gene_list:
        gf = np.zeros((len(gene_list), gene_dim), dtype=np.float32)
        for i, (gn, ev, emb) in enumerate(gene_list):
            gf[i] = emb * np.clip(ev, -3, 3)
        graph['gene'].x = torch.tensor(gf, dtype=torch.float32)
    else:
        graph['gene'].x = torch.zeros((1, gene_dim), dtype=torch.float32)

    mf = np.zeros((n_mod, gene_dim), dtype=np.float32)
    gm_src, gm_dst = [], []
    for m_idx, (_, mg) in enumerate(CANONICAL_MODULES.items()):
        embs = []
        for g in mg:
            gu = g.upper()
            if gu in emb_lookup:
                gi, sc = emb_lookup[gu]
                embs.append(sc); gm_src.append(gi); gm_dst.append(m_idx)
        if embs: mf[m_idx] = np.mean(embs, axis=0)
    graph['module'].x = torch.tensor(mf, dtype=torch.float32)
    if gm_src:
        graph['gene','belongs_to','module'].edge_index = torch.tensor(
            [gm_src, gm_dst], dtype=torch.long)

    pf = np.zeros((N_EXTENDED_PATHWAYS, 2), dtype=np.float32)
    pm_src, pm_dst = [], []
    for pi, (pn, pg) in enumerate(EXTENDED_PATHWAY_MODULES.items()):
        pu = set(g.upper() for g in pg)
        vals = [enorm[gene_upper[g]] for g in pu if g in gene_upper]
        if vals: pf[pi, 0] = np.mean(vals); pf[pi, 1] = np.var(vals)
        for mi, (_, mg) in enumerate(CANONICAL_MODULES.items()):
            if pu & set(g.upper() for g in mg):
                pm_src.append(pi); pm_dst.append(mi)
    graph['pathway'].x = torch.tensor(pf, dtype=torch.float32)
    if pm_src:
        graph['pathway','related_to','module'].edge_index = torch.tensor(
            [pm_src, pm_dst], dtype=torch.long)

    # PCA context
    if pca_b is not None and pca_m is not None:
        pca_dim = pca_b.shape[1]
        n = len(expr_vec)
        if n < pca_dim:
            pad = np.zeros(pca_dim, dtype=np.float32); pad[:n] = expr_vec
            ctx = pca_b @ (pad - pca_m)
        elif n > pca_dim:
            ctx = pca_b @ (expr_vec[:pca_dim] - pca_m)
        else:
            ctx = pca_b @ (expr_vec - pca_m)
    else:
        ctx = np.zeros(128, dtype=np.float32)
    graph['genome_context'].x = torch.tensor(ctx.reshape(1,-1), dtype=torch.float32)

    # Cell type
    cf = np.zeros((len(CANONICAL_CELL_TYPES), 512), dtype=np.float32)
    for i, ct in enumerate(CANONICAL_CELL_TYPES):
        if ct in cell_emb: cf[i] = cell_emb[ct][:512]
    graph['cell_type'].x = torch.tensor(cf, dtype=torch.float32)

    ct_src, ct_dst = [], []
    for ci, ct in enumerate(CANONICAL_CELL_TYPES):
        for mod in CELL_TYPE_MODULE_EDGES.get(ct, []):
            if mod in MODULE_NAMES:
                ct_src.append(ci); ct_dst.append(MODULE_NAMES.index(mod))
    if ct_src:
        graph['cell_type','activates','module'].edge_index = torch.tensor(
            [ct_src, ct_dst], dtype=torch.long)

    df = np.zeros((1, 768), dtype=np.float32)
    if label in dis_emb: df[0] = dis_emb[label][:768]
    graph['disease_state'].x = torch.tensor(df, dtype=torch.float32)
    graph['modality_flag'].x  = build_modality_flag(modality)

    graph['module','summary_of','disease_state'].edge_index = torch.tensor(
        [list(range(n_mod)), [0]*n_mod], dtype=torch.long)
    graph['pathway','summary_of','disease_state'].edge_index = torch.tensor(
        [list(range(N_EXTENDED_PATHWAYS)), [0]*N_EXTENDED_PATHWAYS], dtype=torch.long)
    graph['genome_context','context_of','disease_state'].edge_index = torch.tensor(
        [[0],[0]], dtype=torch.long)

    # pathway_gene_masks
    n_g = len(gene_list)
    if n_g > 0:
        pw = torch.zeros(len(MODULE_NAMES), n_g, dtype=torch.bool)
        for m_idx, gset in _PATHWAY_GENE_SETS_UPPER.items():
            for gi, (gn, _, _) in enumerate(gene_list):
                if gn.upper() in gset: pw[m_idx, gi] = True
        graph.pathway_gene_masks = pw
    else:
        graph.pathway_gene_masks = torch.zeros(len(MODULE_NAMES), 1, dtype=torch.bool)
    graph.module_pathway_ids = MODULE_PATHWAY_IDS.clone()

    graph.y = torch.tensor([LABEL_MAP.get(label, -1)], dtype=torch.long)
    graph.accession = accession; graph.sample_idx = si; graph.modality = modality
    return graph


# ── Large file processor (8-GPU) ─────────────────────────────────────────────
def process_large_file_gpu(npz_path, lmdb_dir):
    accession = os.path.basename(npz_path).replace('.npz', '')
    lmdb_path = os.path.join(lmdb_dir, f'{accession}.lmdb')

    npz      = np.load(npz_path, allow_pickle=True)
    label    = get_disease_label(npz)
    modality = _MODALITY_CLASSIFICATION.get(accession, 'gene_expression')

    feat_key = next(
        (k for k in ['genes','gene_names','var_names','features','row_names']
         if k in npz.files), None)
    raw_names = [str(g) for g in list(npz[feat_key])] if feat_key else []

    expr_key = next(
        (k for k in ['matrix','expression','X','data'] if k in npz.files), None)
    if expr_key is None:
        del npz; return 0

    mat = npz[expr_key].astype(np.float32)
    rows, cols = mat.shape
    is_transposed = rows > cols
    n_samples = cols if is_transposed else rows

    # Normalize gene names
    gene_names, coverage = _GENE_NORMALIZER.normalize(raw_names, accession)
    known = set(_GENE_EMBEDDINGS.keys())
    final_names = [s if (s and s in known) else None for s in gene_names]
    final_cov   = sum(1 for s in final_names if s) / len(raw_names) if raw_names else 0.0
    if final_cov < MIN_COVERAGE_THRESHOLD:
        del npz, mat; return 0

    logger.info(f"[GPU] {accession}: {n_samples} samples, coverage={final_cov:.1%}")

    # Check existing
    env = lmdb.open(lmdb_path, map_size=LMDB_MAP_LARGE, writemap=True)
    with env.begin() as txn:
        existing = txn.stat()['entries']
    env.close()
    if existing >= n_samples:
        del npz, mat
        return existing

    # For scrna: correct orientation and apply cell cap before shared memory load
    if modality == 'scrna':
        # Always treat larger dimension as cells regardless of is_transposed flag
        n_cells_total = max(rows, cols)
        scrna_genes_are_rows = cols > rows   # True -> genes=rows, cells=cols
        MAX_SCRNA_GPU_CELLS = 50_000
        if n_cells_total > MAX_SCRNA_GPU_CELLS:
            import random as _rnd; _rnd.seed(42)
            sel = sorted(_rnd.sample(range(n_cells_total), MAX_SCRNA_GPU_CELLS))
            mat = mat[:, sel] if scrna_genes_are_rows else mat[sel, :]
            rows, cols = mat.shape
            n_samples = MAX_SCRNA_GPU_CELLS
            is_transposed = scrna_genes_are_rows  # re-align for worker
            logger.info(f"[GPU] scrna: sampled {MAX_SCRNA_GPU_CELLS} cells, "
                        f"new shape={mat.shape}")
        else:
            n_samples = n_cells_total
            is_transposed = scrna_genes_are_rows

    # Copy matrix into shared memory
    nbytes = rows * cols * 4
    logger.info(f"[GPU] Shared memory: {nbytes / 1024**3:.1f} GB")
    shm = shm_mod.SharedMemory(create=True, size=nbytes)
    shared = np.ndarray(mat.shape, dtype=np.float32, buffer=shm.buf)
    np.copyto(shared, mat)
    mat_shape = mat.shape  # Save shape before deleting
    del mat, npz
    gc.collect()

    # Task distribution
    task_q  = Queue()
    res_q   = Queue()
    for start in range(0, n_samples, GPU_CHUNK):
        task_q.put((start, min(start + GPU_CHUNK, n_samples)))
    for _ in range(NUM_GPUS):
        task_q.put(None)

    workers = []
    for gpu_id in range(NUM_GPUS):
        p = Process(target=_gpu_worker,
                    args=(gpu_id, shm.name, mat_shape, is_transposed,
                          final_names, modality, label, accession,
                          task_q, res_q, lmdb_path))
        p.start(); workers.append(p)
        logger.info(f"[GPU] Launched worker {gpu_id}")

    for p in workers: p.join()

    total = sum(res_q.get()[1] for _ in range(NUM_GPUS)
                if not res_q.empty())

    shm.close(); shm.unlink(); gc.collect()
    logger.info(f"[GPU] {accession}: {total} graphs total")
    return total


# ── CPU sequential processor ──────────────────────────────────────────────────
def process_file_to_lmdb(npz_path, lmdb_dir, commit_every=500):
    accession = os.path.basename(npz_path).replace('.npz', '')

    if accession in _EXCLUSION_SET:
        return accession, 0, 'excluded'

    modality = _MODALITY_CLASSIFICATION.get(accession, 'gene_expression')
    if modality == 'exclude':
        return accession, 0, 'excluded_by_classification'

    # Resume: skip fully built LMDBs
    lmdb_path = os.path.join(lmdb_dir, f'{accession}.lmdb')
    if os.path.exists(lmdb_path):
        try:
            env_c = lmdb.open(lmdb_path, readonly=True, lock=False)
            with env_c.begin() as txn:
                cnt = txn.stat()['entries']
            env_c.close()
            if cnt >= 50000:  # Only skip if fully complete (50K for scrna)
                logger.info(f"{accession}: skipping ({cnt} existing)")
                return accession, cnt, 'skipped_existing'
        except Exception:
            pass

    try:
        npz = np.load(npz_path, allow_pickle=True)
        label = get_disease_label(npz)

        # ── Feature names ───────────────────────────────────────────────────
        feat_key = None
        for k in ['genes','gene_names','var_names','features','feature_names',
                  'row_names','probes','cpg_ids','mirna_ids','peak_ids',
                  'region_names','snp_ids','rsids','variant_ids',
                  'metabolite_ids','compound_names','met_names',
                  'protein_ids','marker_names','antibody_names',
                  'bin_ids','region_ids']:
            if k in npz.files:
                feat_key = k; break
        if feat_key is None:
            return accession, 0, 'no_feature_names'

        raw_names = [str(g) for g in list(npz[feat_key])]

        # ── Data matrix ─────────────────────────────────────────────────────
        expr_key = None
        for k in ['matrix','expression','X','data','beta_values','signal',
                  'intensities','abundance','genotype','contacts']:
            if k in npz.files:
                expr_key = k; break
        if expr_key is None:
            return accession, 0, 'no_data_matrix'

        mat   = npz[expr_key]
        rows, cols = mat.shape
        is_T  = rows > cols
        n_s   = cols if is_T else rows

        map_size = LMDB_MAP_LARGE if n_s > LARGE_THRESHOLD else LMDB_MAP_SMALL
        env = lmdb.open(lmdb_path, map_size=map_size, writemap=True)
        with env.begin() as txn:
            existing = {k.decode() for k, _ in txn.cursor()}

        built = 0
        txn   = env.begin(write=True)

        def get_vec(idx):
            return (mat[:, idx].astype(np.float32) if is_T
                    else mat[idx, :].astype(np.float32))

        def maybe_commit():
            nonlocal txn, built
            if built % commit_every == 0:
                txn.commit(); txn = env.begin(write=True)

        def key_str(i): return f"{accession}_s{i:06d}"

        # ── Route by modality ────────────────────────────────────────────────
        if modality in ('chromatin', 'chipseq', 'atac'):
            for i in range(n_s):
                k = key_str(i)
                if k in existing: built += 1; continue
                g = build_chromatin_graph(get_vec(i), raw_names, label, accession, i)
                txn.put(k.encode(), pickle.dumps(g)); built += 1; maybe_commit()

        elif modality == 'methylation':
            for i in range(n_s):
                k = key_str(i)
                if k in existing: built += 1; continue
                g = build_methylation_graph(get_vec(i), raw_names, label, accession, i)
                txn.put(k.encode(), pickle.dumps(g)); built += 1; maybe_commit()

        elif modality == 'mirna':
            for i in range(n_s):
                k = key_str(i)
                if k in existing: built += 1; continue
                g = build_mirna_graph(get_vec(i), raw_names, label, accession, i)
                txn.put(k.encode(), pickle.dumps(g)); built += 1; maybe_commit()

        elif modality == 'scrna':
            # scRNA matrices are stored as (genes x cells).
            n_genes_dim  = min(rows, cols)
            n_cells_dim  = max(rows, cols)
            scrna_is_T   = (cols > rows)
            logger.info(f"{accession} [scrna]: shape=({rows},{cols}), "
                        f"genes={n_genes_dim}, cells={n_cells_dim}, "
                        f"transposed={scrna_is_T}")

            norm_names, _ = _GENE_NORMALIZER.normalize(raw_names, accession)

            MAX_SCRNA_CELLS = 50_000
            import random as _rnd
            _rnd.seed(42)
            cell_indices = list(range(n_cells_dim))
            if n_cells_dim > MAX_SCRNA_CELLS:
                cell_indices = sorted(_rnd.sample(cell_indices, MAX_SCRNA_CELLS))
                logger.info(f"{accession} [scrna]: sampling {MAX_SCRNA_CELLS}/{n_cells_dim} cells")

            # Filter already-built
            to_build = [i for i in cell_indices if key_str(i) not in existing]
            already_built = len(cell_indices) - len(to_build)
            if already_built > 0:
                logger.info(f"{accession} [scrna]: skipping {already_built} existing, building {len(to_build)}")
            built += already_built

            if to_build:
                # PARALLEL with ThreadPoolExecutor (no pickle issues)
                from concurrent.futures import ThreadPoolExecutor, as_completed
                NUM_WORKERS = 48
                BATCH_SIZE = 500

                logger.info(f"{accession} [scrna]: extracting {len(to_build)} cell vectors...")
                cell_vecs = []
                for ci in to_build:
                    vec = mat[:, ci].astype(np.float32) if scrna_is_T else mat[ci, :].astype(np.float32)
                    cell_vecs.append((ci, vec))
                logger.info(f"{accession} [scrna]: starting {NUM_WORKERS}-thread parallel build")

                def build_one(args):
                    ci, vec = args
                    g = build_scrna_graph(vec, norm_names, label, accession, ci, pre_normalized=True)
                    return (ci, pickle.dumps(g))

                batch = []
                with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                    futures = {executor.submit(build_one, item): item[0] for item in cell_vecs}
                    for future in as_completed(futures):
                        try:
                            ci, gdata = future.result()
                            batch.append((key_str(ci).encode(), gdata))
                            if len(batch) >= BATCH_SIZE:
                                for k, v in batch:
                                    txn.put(k, v)
                                built += len(batch)
                                maybe_commit()
                                logger.info(f"{accession} [scrna]: {built}/{len(cell_indices)}")
                                batch = []
                        except Exception as e:
                            logger.warning(f"{accession} [scrna]: error {e}")
                
                if batch:
                    for k, v in batch:
                        txn.put(k, v)
                    built += len(batch)
                    maybe_commit()
                    logger.info(f"{accession} [scrna]: {built}/{len(cell_indices)} DONE")

        elif modality == 'cytof':
            for i in range(n_s):
                k = key_str(i)
                if k in existing: built += 1; continue
                g = build_cytof_graph(get_vec(i), raw_names, label, accession, i)
                txn.put(k.encode(), pickle.dumps(g)); built += 1; maybe_commit()

        elif modality == 'hic':
            for i in range(n_s):
                k = key_str(i)
                if k in existing: built += 1; continue
                g = build_hic_graph(get_vec(i), raw_names, label, accession, i)
                txn.put(k.encode(), pickle.dumps(g)); built += 1; maybe_commit()

        elif modality == 'snp':
            for i in range(n_s):
                k = key_str(i)
                if k in existing: built += 1; continue
                g = build_snp_graph(get_vec(i), raw_names, label, accession, i)
                txn.put(k.encode(), pickle.dumps(g)); built += 1; maybe_commit()

        elif modality in ('metabolite', 'metabolomics', 'lipidomics'):
            for i in range(n_s):
                k = key_str(i)
                if k in existing: built += 1; continue
                sl = get_disease_label(npz, i)
                g  = build_metabolite_graph(get_vec(i), raw_names, sl, accession, i)
                txn.put(k.encode(), pickle.dumps(g)); built += 1; maybe_commit()

        else:
            # gene_expression / bulk_rna / unknown → coverage-gated gene embedding path
            norm_names, _ = _GENE_NORMALIZER.normalize(raw_names, accession)
            known = set(_GENE_EMBEDDINGS.keys())
            final = [s if (s and s in known) else None for s in norm_names]
            cov   = sum(1 for s in final if s) / len(raw_names) if raw_names else 0.0
            if cov < MIN_COVERAGE_THRESHOLD:
                env.close()
                return accession, 0, f'low_coverage_{cov:.1%}'
            for i in range(n_s):
                k = key_str(i)
                if k in existing: built += 1; continue
                g = build_gene_expression_graph(
                    get_vec(i), final, label, modality, accession, i)
                txn.put(k.encode(), pickle.dumps(g)); built += 1; maybe_commit()
                if built % 1000 == 0:
                    logger.info(f"{accession} [{modality}]: {built}/{n_s}")

        txn.commit(); env.close()
        return accession, built, f'success_{modality}'

    except Exception as e:
        return accession, 0, f'error:{str(e)[:120]}'


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz_dir',    default='/mnt/nvme15tb/trained_im/NPZ_fresh')
    parser.add_argument('--output_dir', default='/mnt/nvme15tb/trained_im/graphs_lmdb_v3')
    parser.add_argument('--manifest',   default='/mnt/nvme15tb/trained_im/logs/rebuild_manifest.txt')
    parser.add_argument('--accessions', nargs='*')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    load_global_caches()

    if not os.path.exists(args.manifest):
        logger.error(f"Manifest not found: {args.manifest}"); sys.exit(1)
    with open(args.manifest) as f:
        manifest = set(line.strip() for line in f if line.strip())
    logger.info(f"Manifest: {len(manifest)} accessions")

    npz_files = [
        os.path.join(args.npz_dir, fn)
        for fn in sorted(os.listdir(args.npz_dir))
        if fn.endswith('.npz')
        and fn[:-4] in manifest
        and (not args.accessions or fn[:-4] in args.accessions)
    ]
    logger.info(f"Processing {len(npz_files)} files")

    # Route large gene_expression files to GPU path
    results = {'success': 0, 'skipped': 0, 'gpu': 0, 'error': 0}
    total   = 0

    gpu_eligible_modalities = {'gene_expression', 'bulk_rna', ''}  # scrna uses CPU path

    for npz_path in npz_files:
        acc      = os.path.basename(npz_path).replace('.npz', '')
        modality = _MODALITY_CLASSIFICATION.get(acc, 'gene_expression')

        if modality in gpu_eligible_modalities:
            # Quick sample-count check
            try:
                npz = np.load(npz_path, allow_pickle=True)
                ek  = next((k for k in ['matrix','expression','X','data']
                            if k in npz.files), None)
                if ek:
                    r, c = npz[ek].shape
                    # scrna: larger dim = cells; all others: smaller dim = samples
                    n_s = max(r, c) if modality == 'scrna' else min(r, c)
                else:
                    n_s  = 0
                del npz
            except Exception:
                n_s = 0

            if n_s > GPU_THRESHOLD:
                logger.info(f"*** GPU path: {acc} ({n_s} samples) ***")
                try:
                    cnt = process_large_file_gpu(npz_path, args.output_dir)
                    total += cnt; results['gpu'] += 1
                    logger.info(f"{acc}: {cnt} graphs (GPU)")
                except Exception as e:
                    logger.error(f"{acc} GPU error: {e}"); results['error'] += 1
                continue

        # CPU path for everything else
        acc, cnt, status = process_file_to_lmdb(npz_path, args.output_dir)
        total += cnt
        if 'success'  in status: results['success'] += 1
        elif 'skipped' in status: results['skipped'] += 1
        elif 'error'   in status: results['error']   += 1
        else:                     results['skipped']  += 1
        logger.info(f"{acc}: {cnt} graphs ({status})")

    logger.info(f"=== COMPLETE: {total} graphs | {results} ===")


if __name__ == '__main__':
    main()
