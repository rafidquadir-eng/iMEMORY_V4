#!/usr/bin/env python3
"""
step9_r01_aim2_v4_correct.py — V4 R01 Aim 2 Stratification with V4 named clusters
"""
import os, json, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial import cKDTree
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

# Get top clusters per disease class for R01 Aim 2
# Focus on KTx (acute/chronic) and CKD
ktx_acute_clusters = []
ktx_chronic_clusters = []
ckd_clusters = []

for cid, info in named_clusters.items():
    dc = info["dc"][0]
    n = info["n"]
    if dc == 0 and n >= 100:  # KTx Acute
        ktx_acute_clusters.append((int(cid), info["name"], n))
    elif dc == 1 and n >= 100:  # KTx Chronic
        ktx_chronic_clusters.append((int(cid), info["name"], n))
    elif dc == 2 and n >= 100:  # CKD
        ckd_clusters.append((int(cid), info["name"], n))

ktx_acute_clusters = sorted(ktx_acute_clusters, key=lambda x: -x[2])[:5]
ktx_chronic_clusters = sorted(ktx_chronic_clusters, key=lambda x: -x[2])[:5]
ckd_clusters = sorted(ckd_clusters, key=lambda x: -x[2])[:5]

print("\nTop KTx Acute clusters:", ktx_acute_clusters)
print("Top KTx Chronic clusters:", ktx_chronic_clusters)
print("Top CKD clusters:", ckd_clusters)

# ─── R01 Aim 2 Figure: Stratification by Disease Subtype ──────────────────────
print("\nBuilding R01 Aim 2 Stratification figure...")

fig = plt.figure(figsize=(18, 12))

# Left panel: UMAP colored by disease class
ax1 = fig.add_subplot(1, 3, 1)
df = endo.copy()
df["ux"] = df["umap_x"]
df["uy"] = df["umap_y"]

pad = 0.3
XMN = np.percentile(df["ux"], 1) - pad
XMX = np.percentile(df["ux"], 99) + pad
YMN = np.percentile(df["uy"], 1) - pad
YMX = np.percentile(df["uy"], 99) + pad

df_plot = df[(df["ux"] >= XMN) & (df["ux"] <= XMX) &
             (df["uy"] >= YMN) & (df["uy"] <= YMX)].copy()

for dc in [0, 1, 2]:  # KTx Acute, Chronic, CKD
    sub = df_plot[df_plot["disease_class"] == dc]
    ax1.scatter(sub["ux"], sub["uy"], c=DC_COLORS[dc], s=3, alpha=0.6, 
                linewidths=0, rasterized=True, label=f"{DC_NAMES[dc]} (n={len(sub):,})")

ax1.set_xlim(XMN, XMX)
ax1.set_ylim(YMN, YMX)
ax1.set_xlabel("UMAP 1", fontsize=11)
ax1.set_ylabel("UMAP 2", fontsize=11)
ax1.set_title("Disease Class Distribution\n(KTx Acute, Chronic, CKD)", fontsize=12, fontweight="bold")
ax1.legend(loc="upper right", fontsize=8)

# Middle panel: Top clusters highlighted
ax2 = fig.add_subplot(1, 3, 2)

# Background
ax2.scatter(df_plot["ux"], df_plot["uy"], c="#CCCCCC", s=2, alpha=0.1, linewidths=0, rasterized=True)

# Highlight top clusters
all_top = ktx_acute_clusters[:3] + ktx_chronic_clusters[:3] + ckd_clusters[:3]
colors = ["#E53935"]*3 + ["#FB8C00"]*3 + ["#43A047"]*3

for (cid, name, n), color in zip(all_top, colors):
    sub = df_plot[df_plot["refined_endotype"] == cid]
    ax2.scatter(sub["ux"], sub["uy"], c=color, s=5, alpha=0.8, linewidths=0, rasterized=True)
    if len(sub) > 50:
        cx, cy = sub["ux"].mean(), sub["uy"].mean()
        short_name = name.split("_")[-1] if "_" in name else name
        ax2.annotate(short_name, (cx, cy), fontsize=7, fontweight="bold",
                    ha="center", va="center", color="white",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.8))

ax2.set_xlim(XMN, XMX)
ax2.set_ylim(YMN, YMX)
ax2.set_xlabel("UMAP 1", fontsize=11)
ax2.set_ylabel("UMAP 2", fontsize=11)
ax2.set_title("Top Clusters per Disease\n(Top 3 each)", fontsize=12, fontweight="bold")

# Right panel: TI phase distribution per disease
ax3 = fig.add_subplot(1, 3, 3)

# Bar chart of TI phase by disease class
phase_by_dc = endo.groupby(["disease_class", "ti_phase_assigned"]).size().unstack(fill_value=0)
phase_by_dc = phase_by_dc.loc[[0, 1, 2]]  # KTx Acute, Chronic, CKD

x = np.arange(3)
width = 0.2
phases = ["E4 Priming", "E1/E5 Induction", "E20 Maintenance", "No TI signal"]

for i, phase in enumerate(phases):
    if phase in phase_by_dc.columns:
        vals = phase_by_dc[phase].values
    else:
        vals = [0, 0, 0]
    ax3.bar(x + i*width, vals, width, label=phase, color=PHASE_COLORS[phase])

ax3.set_xticks(x + 1.5*width)
ax3.set_xticklabels(["KTx Acute", "KTx Chronic", "CKD"], fontsize=10)
ax3.set_ylabel("Number of Samples", fontsize=11)
ax3.set_title("TI Phase Distribution\nby Disease Class", fontsize=12, fontweight="bold")
ax3.legend(loc="upper right", fontsize=8)

fig.suptitle("iMEMORY V4 — R01 Aim 2: Disease Stratification Analysis", fontsize=14, fontweight="bold", y=0.98)
plt.tight_layout()

out_path = os.path.join(OUTDIR, "Figure_R01_Aim2_Stratification.png")
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path}")

print("\nDONE.")
