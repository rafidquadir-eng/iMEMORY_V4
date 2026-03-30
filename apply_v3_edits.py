#!/usr/bin/env python3
"""Apply three surgical edits to graph_builder_lmdb_three_tier_v3.py"""
import re

INPUT_FILE = "/mnt/data/trained_im/scripts/graph_builder_lmdb_three_tier_v3.py"
OUTPUT_FILE = "/mnt/data/trained_im/scripts/graph_builder_lmdb_three_tier_v3.py"

# Read the file
with open(INPUT_FILE, "r") as f:
    content = f.read()

print(f"Original file size: {len(content)} chars")

# ═══════════════════════════════════════════════════════════════════════════════
# EDIT 1: Replace _build_common_nodes function with marker-gene activation version
# ═══════════════════════════════════════════════════════════════════════════════

# The new function definition with marker genes
MARKER_GENES_DICT = '''# CANONICAL MARKER GENES per cell type — used to derive activation from expression
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

    graph['cell_type'].x = torch.tensor(cell_feats, dtype=torch.float32)'''

# Find and replace the old _build_common_nodes function
# Pattern: from "def _build_common_nodes(graph, label):" to the line with graph['cell_type'].x = 
old_func_pattern = r"def _build_common_nodes\(graph, label\):.*?graph\['cell_type'\]\.x = torch\.tensor\(cell_feats, dtype=torch\.float32\)"
match = re.search(old_func_pattern, content, re.DOTALL)
if match:
    content = content[:match.start()] + MARKER_GENES_DICT + content[match.end():]
    print("Edit 1 applied: _build_common_nodes with marker-gene activation")
else:
    print("WARNING: Could not find _build_common_nodes function for Edit 1")

# ═══════════════════════════════════════════════════════════════════════════════
# EDIT 2: Update the call in build_gene_expression_graph to pass expr_norm/gene_upper
# ═══════════════════════════════════════════════════════════════════════════════

# Find the specific call in build_gene_expression_graph (near pathway_gene_masks comment)
old_call = "    graph = _build_common_nodes(graph, label)\n\n    # pathway_gene_masks for pathway supervision"
new_call = "    graph = _build_common_nodes(graph, label, expr_norm=expr_norm, gene_upper=gene_upper)\n\n    # pathway_gene_masks for pathway supervision"

if old_call in content:
    content = content.replace(old_call, new_call)
    print("Edit 2 applied: Pass expr_norm and gene_upper in build_gene_expression_graph")
else:
    print("WARNING: Could not find the specific call for Edit 2")

# ═══════════════════════════════════════════════════════════════════════════════
# EDIT 3: Add contamination exclusion in load_global_caches
# ═══════════════════════════════════════════════════════════════════════════════

# Find the end of load_global_caches (after PCA warning) and add contamination loading
old_pca_end = '''        logger.warning("PCA basis not found — Tier 3 genome context will be zeros")


# ── Shared helpers'''

new_pca_end = '''        logger.warning("PCA basis not found — Tier 3 genome context will be zeros")

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


# ── Shared helpers'''

if old_pca_end in content:
    content = content.replace(old_pca_end, new_pca_end)
    print("Edit 3 applied: Contamination exclusion in load_global_caches")
else:
    print("WARNING: Could not find PCA section for Edit 3")

# Write the modified file
with open(OUTPUT_FILE, "w") as f:
    f.write(content)

print(f"\nModified file size: {len(content)} chars")
print("All edits complete. File saved.")
