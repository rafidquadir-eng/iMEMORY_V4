#!/usr/bin/env python3
"""
step3_figures_v4_correct.py — V4 Figure 1 (Disease Axis) and Figure 2 (TI Annotated)
Uses V4 named clusters and TI phase assignments.
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
OUTDIR    = "/mnt/nvme15tb/trained_im/outputs/V4/figures/"
pathlib.Path(OUTDIR).mkdir(parents=True, exist_ok=True)

K_PHASE = 50

# V4 TI CLUSTER SETS
CLEAN_TI = {5, 8, 14, 34, 60, 61, 66, 69, 86, 98, 101, 107, 154, 169, 192, 194, 205, 211, 213, 216, 221, 223, 224, 226, 227, 228, 231, 232, 234, 235, 236, 237, 238, 240, 244, 245, 247, 250, 253, 254, 255, 258, 263, 264, 265, 266, 269, 270, 271, 274, 275, 279, 280, 281, 282, 284, 285, 294, 301, 311, 312, 313, 315, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395}
E4_CLUSTERS = {107}
E1E5_CLUSTERS = {353, 194, 34, 294, 351, 265, 244, 347, 60, 223}
E20_CLUSTERS = CLEAN_TI - E4_CLUSTERS - E1E5_CLUSTERS

DC_COLORS = {
    0: "#E53935",  # KTx_Acute - red
    1: "#FB8C00",  # KTx_Chronic - orange
    2: "#43A047",  # CKD - green
    3: "#1E88E5",  # HD - blue
    4: "#8E24AA",  # KTx_Stable - purple
    5: "#6A1B9A",  # Control/TI - dark purple
}
DC_NAMES = {0: "KTx Acute", 1: "KTx Chronic", 2: "CKD", 3: "HD", 4: "KTx Stable", 5: "Control/TI"}

PHASE_COLORS = {
    "E4 Priming": "#0077CC",
    "E1/E5 Induction": "#DD2200",
    "E20 Maintenance": "#8800CC",
    "No TI signal": "#1A1A2E",
}
GREY = "#CCCCCC"

print("Loading V4 endotype CSV...")
endo = pd.read_csv(ENDO_PATH)
print(f"  {len(endo)} samples")

print("Loading V4 named clusters...")
with open(NAMED_PATH) as f:
    named_clusters = json.load(f)
print(f"  {len(named_clusters)} named clusters")

# Get top clusters per disease class for labeling
top_clusters = {}
for cid, info in named_clusters.items():
    dc = info["dc"][0]
    if dc not in top_clusters:
        top_clusters[dc] = []
    top_clusters[dc].append((int(cid), info["name"], info["n"]))

for dc in top_clusters:
    top_clusters[dc] = sorted(top_clusters[dc], key=lambda x: -x[2])[:5]

print("\nTop clusters per disease class:")
for dc, clusters in sorted(top_clusters.items()):
    print(f"  {DC_NAMES[dc]}:")
    for cid, name, n in clusters:
        print(f"    {cid}: {name} (n={n})")

# Prepare data
df = endo.copy()
df["ux"] = df["umap_x"]
df["uy"] = df["umap_y"]

pad = 0.3
XMN = np.percentile(df["ux"], 1) - pad
XMX = np.percentile(df["ux"], 99) + pad
YMN = np.percentile(df["uy"], 1) - pad
YMX = np.percentile(df["uy"], 99) + pad

df = df[(df["ux"] >= XMN) & (df["ux"] <= XMX) &
        (df["uy"] >= YMN) & (df["uy"] <= YMX)].copy()

# Build TI k-NN tree
ti_df = df[df["disease_class"] == 5].copy().reset_index(drop=True)
ti_coords = ti_df[["ux", "uy"]].values
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

# ─── Figure 1: Disease Axis UMAP ──────────────────────────────────────────────
print("\nBuilding Figure 1 (Disease Axis UMAP)...")
fig, ax = plt.subplots(figsize=(14, 12))

# Plot all points colored by disease class
for dc in sorted(DC_COLORS.keys()):
    sub = df[df["disease_class"] == dc]
    ax.scatter(sub["ux"], sub["uy"], c=DC_COLORS[dc], s=3, alpha=0.6, 
               linewidths=0, rasterized=True, label=f"{DC_NAMES[dc]} (n={len(sub):,})")

# Add cluster labels for top clusters
for dc, clusters in top_clusters.items():
    for cid, name, n in clusters[:3]:  # Top 3 per disease
        sub = df[df["refined_endotype"] == cid]
        if len(sub) > 50:
            cx, cy = sub["ux"].mean(), sub["uy"].mean()
            short_name = name.replace("_", " ").replace("Control TI", "TI")
            ax.annotate(short_name, (cx, cy), fontsize=7, fontweight="bold",
                       ha="center", va="center", color="white",
                       bbox=dict(boxstyle="round,pad=0.2", facecolor=DC_COLORS[dc], alpha=0.8))

ax.set_xlim(XMN, XMX)
ax.set_ylim(YMN, YMX)
ax.set_xlabel("UMAP 1", fontsize=12)
ax.set_ylabel("UMAP 2", fontsize=12)
ax.set_title("iMEMORY V4 — Figure 1: Disease Axis UMAP\n(45,670 samples, 6 disease classes)", fontsize=14, fontweight="bold")
ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

for sp in ax.spines.values():
    sp.set_linewidth(0.5)
    sp.set_color("#BBBBBB")

out_path = os.path.join(OUTDIR, "Figure1_DiseaseAxis_UMAP.png")
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path}")

# ─── Figure 2: TI Phase Annotated UMAP ────────────────────────────────────────
print("\nBuilding Figure 2 (TI Phase Annotated UMAP)...")
fig, ax = plt.subplots(figsize=(14, 12))

# Assign TI phases to all samples
pts = df[["ux", "uy"]].values
phases = assign_ti_phase(pts, k=K_PHASE)
df["ti_phase_plot"] = phases

# Plot by phase
for phase in ["No TI signal", "E4 Priming", "E1/E5 Induction", "E20 Maintenance"]:
    sub = df[df["ti_phase_plot"] == phase]
    zorder = 2 if phase == "No TI signal" else 3
    ax.scatter(sub["ux"], sub["uy"], c=PHASE_COLORS[phase], s=3, alpha=0.7,
               linewidths=0, rasterized=True, label=f"{phase} (n={len(sub):,})", zorder=zorder)

ax.set_xlim(XMN, XMX)
ax.set_ylim(YMN, YMX)
ax.set_xlabel("UMAP 1", fontsize=12)
ax.set_ylabel("UMAP 2", fontsize=12)
ax.set_title("iMEMORY V4 — Figure 2: TI Phase Annotated UMAP\n(k-NN phase assignment, k=50)", fontsize=14, fontweight="bold")
ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

for sp in ax.spines.values():
    sp.set_linewidth(0.5)
    sp.set_color("#BBBBBB")

out_path = os.path.join(OUTDIR, "Figure2_TI_Annotated_UMAP.png")
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path}")

print("\nDONE.")
