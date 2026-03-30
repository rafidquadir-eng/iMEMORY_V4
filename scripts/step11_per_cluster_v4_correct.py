#!/usr/bin/env python3
"""
step11_per_cluster_v4_correct.py — V4 Per-Cluster Profiling with named clusters
"""
import os, json, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial import cKDTree
from collections import Counter
import pathlib

ENDO_PATH = "/mnt/nvme15tb/trained_im/outputs/V4/stage3/endotype_assignments_v4_final.csv"
NAMED_PATH = "/mnt/nvme15tb/trained_im/outputs/V4/stage3/named_clusters_v4.json"
OUTDIR    = "/mnt/nvme15tb/trained_im/outputs/V4/profiling/per_cluster/"
pathlib.Path(OUTDIR).mkdir(parents=True, exist_ok=True)

K_PHASE = 50

# V4 TI CLUSTER SETS
CLEAN_TI = {5, 8, 14, 34, 60, 61, 66, 69, 86, 98, 101, 107, 154, 169, 192, 194, 205, 211, 213, 216, 221, 223, 224, 226, 227, 228, 231, 232, 234, 235, 236, 237, 238, 240, 244, 245, 247, 250, 253, 254, 255, 258, 263, 264, 265, 266, 269, 270, 271, 274, 275, 279, 280, 281, 282, 284, 285, 294, 301, 311, 312, 313, 315, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395}
E4_CLUSTERS = {107}
E1E5_CLUSTERS = {353, 194, 34, 294, 351, 265, 244, 347, 60, 223}
E20_CLUSTERS = CLEAN_TI - E4_CLUSTERS - E1E5_CLUSTERS

DC_COLORS = {
    0: "#E53935",  # KTx_Acute
    1: "#FB8C00",  # KTx_Chronic
    2: "#43A047",  # CKD
    3: "#1E88E5",  # HD
    4: "#8E24AA",  # KTx_Stable
    5: "#6A1B9A",  # Control/TI
}
DC_NAMES = {0: "KTx Acute", 1: "KTx Chronic", 2: "CKD", 3: "HD", 4: "KTx Stable", 5: "Control/TI"}

PHASE_COLORS = {
    "E4 Priming": "#0077CC",
    "E1/E5 Induction": "#DD2200",
    "E20 Maintenance": "#8800CC",
    "No TI signal": "#1A1A2E",
}

print("Loading V4 endotype CSV...")
endo = pd.read_csv(ENDO_PATH)
print(f"  {len(endo)} samples")

print("Loading V4 named clusters...")
with open(NAMED_PATH) as f:
    named_clusters = json.load(f)
print(f"  {len(named_clusters)} named clusters")

# Build TI k-NN tree
ti_df = endo[endo["disease_class"] == 5].copy().reset_index(drop=True)
ti_coords = ti_df[["umap_x", "umap_y"]].values
ti_cids = ti_df["refined_endotype"].values.astype(int)
ti_tree = cKDTree(ti_coords)

def assign_ti_phase(pts, k=K_PHASE):
    phases = []
    for px, py in pts:
        _, idx = ti_tree.query([px, py], k=min(k, len(ti_coords)))
        nbr = ti_cids[idx]
        counts = {
            "E4 Priming": sum(c in E4_CLUSTERS for c in nbr),
            "E1/E5 Induction": sum(c in E1E5_CLUSTERS for c in nbr),
            "E20 Maintenance": sum(c in E20_CLUSTERS for c in nbr),
        }
        best = max(counts, key=counts.get)
        phases.append(best if counts[best] >= 1 else "No TI signal")
    return np.array(phases)

# Assign TI phases
pts = endo[["umap_x", "umap_y"]].values
endo["ti_phase_assigned"] = assign_ti_phase(pts, k=K_PHASE)

# Get top 25 clusters by size
cluster_sizes = endo.groupby("refined_endotype").size().sort_values(ascending=False)
top_clusters = cluster_sizes.head(25).index.tolist()

print(f"\nTop 25 clusters: {top_clusters}")

# ─── Figure: Per-Cluster Cell Type Composition ────────────────────────────────
print("\nBuilding Per-Cluster Cell Type figure...")

# Get cluster names and disease class composition
cluster_data = []
for cid in top_clusters:
    sub = endo[endo["refined_endotype"] == cid]
    name = named_clusters.get(str(cid), {}).get("name", f"Cluster_{cid}")
    dc_counts = sub["disease_class"].value_counts().to_dict()
    n = len(sub)
    cluster_data.append({
        "cluster_id": cid,
        "name": name,
        "n": n,
        **{f"dc_{dc}": dc_counts.get(dc, 0) for dc in range(6)}
    })

df_clusters = pd.DataFrame(cluster_data)
df_clusters = df_clusters.sort_values("n", ascending=True)  # For horizontal bar

fig, ax = plt.subplots(figsize=(12, 10))

# Stacked horizontal bar
y_pos = np.arange(len(df_clusters))
left = np.zeros(len(df_clusters))

for dc in range(6):
    col = f"dc_{dc}"
    widths = df_clusters[col].values
    ax.barh(y_pos, widths, left=left, color=DC_COLORS[dc], label=DC_NAMES[dc], height=0.7)
    left += widths

ax.set_yticks(y_pos)
ax.set_yticklabels([f"{row['name']} (n={row['n']:,})" for _, row in df_clusters.iterrows()], fontsize=8)
ax.set_xlabel("Number of Samples", fontsize=11)
ax.set_title("iMEMORY V4 — Per-Cluster Disease Class Composition\n(Top 25 clusters by size)", fontsize=13, fontweight="bold")
ax.legend(loc="lower right", fontsize=9)

plt.tight_layout()
out_path = os.path.join(OUTDIR, "Figure_PerCluster_CellType.png")
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path}")

# ─── Figure: Per-Cluster TI Phase Composition ─────────────────────────────────
print("\nBuilding Per-Cluster TI Phase figure...")

cluster_phase_data = []
for cid in top_clusters:
    sub = endo[endo["refined_endotype"] == cid]
    name = named_clusters.get(str(cid), {}).get("name", f"Cluster_{cid}")
    phase_counts = sub["ti_phase_assigned"].value_counts().to_dict()
    n = len(sub)
    cluster_phase_data.append({
        "cluster_id": cid,
        "name": name,
        "n": n,
        **{phase: phase_counts.get(phase, 0) for phase in PHASE_COLORS.keys()}
    })

df_phase = pd.DataFrame(cluster_phase_data)
df_phase = df_phase.sort_values("n", ascending=True)

fig, ax = plt.subplots(figsize=(12, 10))

y_pos = np.arange(len(df_phase))
left = np.zeros(len(df_phase))

for phase in ["No TI signal", "E4 Priming", "E1/E5 Induction", "E20 Maintenance"]:
    widths = df_phase[phase].values
    ax.barh(y_pos, widths, left=left, color=PHASE_COLORS[phase], label=phase, height=0.7)
    left += widths

ax.set_yticks(y_pos)
ax.set_yticklabels([f"{row['name']} (n={row['n']:,})" for _, row in df_phase.iterrows()], fontsize=8)
ax.set_xlabel("Number of Samples", fontsize=11)
ax.set_title("iMEMORY V4 — Per-Cluster TI Phase Composition\n(Top 25 clusters by size)", fontsize=13, fontweight="bold")
ax.legend(loc="lower right", fontsize=9)

plt.tight_layout()
out_path = os.path.join(OUTDIR, "Figure_PerCluster_TIPhase.png")
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path}")

print("\nDONE.")
