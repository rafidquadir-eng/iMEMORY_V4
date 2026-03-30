#!/usr/bin/env python3
"""
step11_per_cluster_v4.py — iMEMORY V4
Per-cluster profiling with cell type activation validation.
KEY VALIDATION: Cell type heatmap should show non-zero values since V4 graphs
have expression-derived cell_type node features.
"""
import os, sys, lmdb, pickle, numpy as np, pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
import random, pathlib
random.seed(42)
np.random.seed(42)

sys.path.insert(0, "/mnt/data/trained_im/scripts")
from hetero_encoder_v4 import HeteroClassificationModelV3

# V4 paths
ENDO_PATH = "/mnt/nvme15tb/trained_im/outputs/V4/stage3/endotype_assignments_v4_final.csv"
CKPT      = "/mnt/nvme15tb/trained_im/checkpoints/phase2_v4/phase2_v4_best.pt"
GRAPH_DIR = "/mnt/nvme15tb/trained_im/graphs_lmdb_v4"
OUT_ROOT  = "/mnt/nvme15tb/trained_im/outputs/V4/profiling/"
pathlib.Path(OUT_ROOT).mkdir(parents=True, exist_ok=True)
DEVICE    = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
N_MAX     = 150

MODULE_NAMES = [
    "mTOR_glycolytic", "NF_kB_epigenetic", "trained_immunity_module",
    "inflammation_innate", "inflammation_adaptive", "metabolism_glycolysis",
    "metabolism_oxidative", "stress_oxidative", "stress_er", "mTOR_signaling",
    "cytokine_response", "epigenetic_remodeling", "immune_memory", "antigen_presentation"
]

CANONICAL_CELL_TYPES = [
    "monocyte_classical", "monocyte_nonclassical", "macrophage_M1", "macrophage_M2",
    "CD4_Tcell", "CD8_Tcell", "Treg", "NK_cell",
    "B_cell", "dendritic_cDC1", "dendritic_cDC2", "MAIT_cell",
]

print("=== SETUP ===")
endo = pd.read_csv(ENDO_PATH)
print(f"Loaded {len(endo)} samples")

# Get top clusters per disease class
def get_top_clusters(df, disease_class, n=7):
    sub = df[df["disease_class"] == disease_class]
    return sub["refined_endotype"].value_counts().head(n).index.tolist()

CKD_CLUSTERS = get_top_clusters(endo, 2)
HD_CLUSTERS = get_top_clusters(endo, 3)
KTX_CLUSTERS = get_top_clusters(endo, 4) + get_top_clusters(endo, 0, 3) + get_top_clusters(endo, 1, 3)

ALL_CLUSTERS = list(set(CKD_CLUSTERS + HD_CLUSTERS + KTX_CLUSTERS))
print(f"Profiling {len(ALL_CLUSTERS)} clusters")

# Load model
model = HeteroClassificationModelV3(hidden_dim=256, embedding_dim=128, num_classes=6)
ckpt = torch.load(CKPT, map_location="cpu")
model.load_state_dict(ckpt["model_state_dict"])
model = model.to(DEVICE).eval()
print("Model loaded")

# Build LMDB index
lmdb_idx = {}
for fn in os.listdir(GRAPH_DIR):
    if not fn.endswith(".lmdb"): continue
    path = os.path.join(GRAPH_DIR, fn)
    if not os.path.isdir(path): continue
    try:
        env = lmdb.open(path, readonly=True, lock=False)
        with env.begin() as txn:
            for key, _ in txn.cursor():
                key_str = key.decode("utf-8")
                lmdb_idx[key_str] = path
        env.close()
    except: pass
print(f"LMDB index: {len(lmdb_idx)} samples")

# Extract profiles per cluster
cluster_profiles = []
cluster_cell_types = []

for cid in ALL_CLUSTERS:
    sub = endo[endo["refined_endotype"] == cid]
    if len(sub) < 5:
        continue
    
    sample_ids = sub["accession_id"].tolist()[:N_MAX]
    
    module_acts = []
    cell_type_acts = []
    
    for sid in sample_ids:
        if sid not in lmdb_idx:
            continue
        try:
            env = lmdb.open(lmdb_idx[sid], readonly=True, lock=False)
            with env.begin() as txn:
                val = txn.get(sid.encode("utf-8"))
                if val is None: continue
                g = pickle.loads(val)
            env.close()
            
            # Extract cell type features directly from graph
            if "cell_type" in g.node_types:
                ct_x = g["cell_type"].x.numpy()
                # Compute norm per cell type as activation measure
                ct_norms = np.linalg.norm(ct_x, axis=1)
                cell_type_acts.append(ct_norms)
            
            # Extract module activations via model
            if not hasattr(g, "dataset_weight") or not isinstance(g.dataset_weight, torch.Tensor):
                g.dataset_weight = torch.tensor([1.0])
            g = g.to(DEVICE)
            x_dict = {nt: g[nt].x for nt in g.node_types if hasattr(g[nt], "x")}
            ei_dict = {et: g[et].edge_index for et in g.edge_types if hasattr(g[et], "edge_index")}
            b_dict = {nt: torch.zeros(g[nt].x.size(0), dtype=torch.long, device=DEVICE)
                      for nt in g.node_types if hasattr(g[nt], "x")}
            
            with torch.no_grad():
                _, _, _, module_out, _ = model(x_dict, ei_dict, b_dict)
                if module_out is not None:
                    module_acts.append(module_out.cpu().numpy().flatten()[:14])
        except Exception as e:
            continue
    
    if module_acts:
        mean_modules = np.mean(module_acts, axis=0)
        cluster_profiles.append({
            "cluster_id": cid,
            "n_samples": len(module_acts),
            **{MODULE_NAMES[i]: mean_modules[i] for i in range(min(14, len(mean_modules)))}
        })
    
    if cell_type_acts:
        mean_ct = np.mean(cell_type_acts, axis=0)
        cluster_cell_types.append({
            "cluster_id": cid,
            "n_samples": len(cell_type_acts),
            **{CANONICAL_CELL_TYPES[i]: mean_ct[i] for i in range(min(12, len(mean_ct)))}
        })
    
    print(f"  Cluster {cid}: {len(module_acts)} module samples, {len(cell_type_acts)} cell type samples")

# Save profiles
df_modules = pd.DataFrame(cluster_profiles)
df_modules.to_csv(os.path.join(OUT_ROOT, "cluster_module_profile_v4.csv"), index=False)
print(f"\nSaved module profiles: {len(df_modules)} clusters")

df_cell_types = pd.DataFrame(cluster_cell_types)
df_cell_types.to_csv(os.path.join(OUT_ROOT, "cluster_celltype_profile_v4.csv"), index=False)
print(f"Saved cell type profiles: {len(df_cell_types)} clusters")

# Generate heatmaps
if len(df_modules) > 0:
    fig, ax = plt.subplots(figsize=(14, 8))
    data = df_modules[MODULE_NAMES].values
    im = ax.imshow(data, aspect="auto", cmap="RdBu_r")
    ax.set_xticks(range(len(MODULE_NAMES)))
    ax.set_xticklabels(MODULE_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(df_modules)))
    ax.set_yticklabels(df_modules["cluster_id"].values, fontsize=8)
    ax.set_xlabel("Module")
    ax.set_ylabel("Cluster ID")
    ax.set_title("V4 Per-Cluster Module Activation Heatmap")
    plt.colorbar(im, ax=ax, label="Mean Activation")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_ROOT, "Figure_PerCluster_ModuleHeatmap_V4.png"), dpi=300)
    plt.close(fig)
    print("Saved module heatmap")

if len(df_cell_types) > 0:
    fig, ax = plt.subplots(figsize=(12, 8))
    data = df_cell_types[CANONICAL_CELL_TYPES].values
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(CANONICAL_CELL_TYPES)))
    ax.set_xticklabels(CANONICAL_CELL_TYPES, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(df_cell_types)))
    ax.set_yticklabels(df_cell_types["cluster_id"].values, fontsize=8)
    ax.set_xlabel("Cell Type")
    ax.set_ylabel("Cluster ID")
    ax.set_title("V4 Per-Cluster Cell Type Activation Heatmap\n(KEY VALIDATION: Should show non-zero values)")
    plt.colorbar(im, ax=ax, label="Mean Activation (L2 norm)")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_ROOT, "Figure_PerCluster_CellType_V4.png"), dpi=300)
    plt.close(fig)
    print("Saved cell type heatmap")

# Validation check
print("\n=== CELL TYPE ACTIVATION VALIDATION ===")
if len(df_cell_types) > 0:
    ct_data = df_cell_types[CANONICAL_CELL_TYPES].values
    non_zero_pct = (ct_data > 0.1).mean() * 100
    mean_activation = ct_data.mean()
    max_activation = ct_data.max()
    print(f"Non-zero cells (>0.1): {non_zero_pct:.1f}%")
    print(f"Mean activation: {mean_activation:.3f}")
    print(f"Max activation: {max_activation:.3f}")
    if non_zero_pct > 50:
        print("VALIDATION PASSED: Cell type activation fix confirmed")
    else:
        print("WARNING: Low cell type activation - check graph builder")

print("\nDONE.")
