#!/usr/bin/env python3
"""
step13_study_flow_v4_correct.py — V4 Study Flow / STROBE diagram
Fixed: Shows sample counts (not cluster counts)
"""
import os, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pathlib

ENDO_PATH = "/mnt/nvme15tb/trained_im/outputs/V4/stage3/endotype_assignments_v4_final.csv"
OUTDIR    = "/mnt/nvme15tb/trained_im/outputs/V4/figures/"
pathlib.Path(OUTDIR).mkdir(parents=True, exist_ok=True)

DC_COLORS = {
    0: "#E53935",  # KTx_Acute
    1: "#FB8C00",  # KTx_Chronic
    2: "#43A047",  # CKD
    3: "#1E88E5",  # HD
    4: "#8E24AA",  # KTx_Stable
    5: "#6A1B9A",  # Control/TI
}
DC_NAMES = {0: "KTx Acute", 1: "KTx Chronic", 2: "CKD", 3: "HD", 4: "KTx Stable", 5: "Control/TI"}

print("Loading V4 endotype CSV...")
endo = pd.read_csv(ENDO_PATH)
print(f"  {len(endo)} samples")

# Count samples per disease class
dc_counts = endo["disease_class"].value_counts().sort_index()
print("\nSamples per disease class:")
for dc, n in dc_counts.items():
    print(f"  {DC_NAMES[dc]}: {n:,}")

# Count unique studies
n_studies = endo["accession_id"].str.extract(r"(GSE\d+|E-MTAB-\d+)")[0].nunique()
print(f"\nUnique studies: {n_studies}")

# ─── Study Flow Diagram ───────────────────────────────────────────────────────
print("\nBuilding Study Flow diagram...")

fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis("off")

# Title
ax.text(8, 9.5, "iMEMORY V4 — Study Flow Diagram", fontsize=16, fontweight="bold", ha="center")

# Total box
total_n = len(endo)
rect = mpatches.FancyBboxPatch((5.5, 7.5), 5, 1.2, boxstyle="round,pad=0.05", 
                                facecolor="#E8E8E8", edgecolor="black", linewidth=2)
ax.add_patch(rect)
ax.text(8, 8.1, f"Total V4 Samples\nn = {total_n:,}", fontsize=12, fontweight="bold", ha="center", va="center")

# Arrow down
ax.annotate("", xy=(8, 7.0), xytext=(8, 7.5), arrowprops=dict(arrowstyle="->", lw=2))

# Disease class boxes
box_width = 2.2
box_height = 1.5
y_row = 4.5
x_positions = [1.5, 4.0, 6.5, 9.0, 11.5, 14.0]

for i, dc in enumerate([2, 3, 0, 1, 4, 5]):  # CKD, HD, KTx_Acute, KTx_Chronic, KTx_Stable, Control/TI
    x = x_positions[i]
    n = dc_counts.get(dc, 0)
    
    rect = mpatches.FancyBboxPatch((x - box_width/2, y_row), box_width, box_height, 
                                    boxstyle="round,pad=0.05", facecolor=DC_COLORS[dc], 
                                    edgecolor="black", linewidth=1.5, alpha=0.9)
    ax.add_patch(rect)
    
    # Text with word wrap for long names
    name = DC_NAMES[dc]
    if len(name) > 10:
        name = name.replace(" ", "\n")
    ax.text(x, y_row + box_height/2 + 0.2, name, fontsize=10, fontweight="bold", 
            ha="center", va="center", color="white")
    ax.text(x, y_row + box_height/2 - 0.3, f"n = {n:,}", fontsize=10, 
            ha="center", va="center", color="white")
    
    # Arrow from total
    ax.annotate("", xy=(x, y_row + box_height), xytext=(8, 7.0), 
                arrowprops=dict(arrowstyle="->", lw=1, color="#666666"))

# Summary stats at bottom
ax.text(8, 2.5, f"Total: {total_n:,} samples from {n_studies} studies", 
        fontsize=12, ha="center", va="center")
ax.text(8, 1.8, "V4 model trained on clean, human, nephrology-relevant samples only", 
        fontsize=10, ha="center", va="center", style="italic", color="#666666")
ax.text(8, 1.2, "Contaminated studies (brain, non-human) excluded at graph build time", 
        fontsize=10, ha="center", va="center", style="italic", color="#666666")

out_path = os.path.join(OUTDIR, "Figure_StudyFlow_STROBE.png")
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path}")

# ─── UMAP with sample counts ──────────────────────────────────────────────────
print("\nBuilding UMAP with sample counts...")

fig, ax = plt.subplots(figsize=(14, 12))

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

for dc in sorted(DC_COLORS.keys()):
    sub = df[df["disease_class"] == dc]
    n_samples = len(sub)  # SAMPLE count, not cluster count
    ax.scatter(sub["ux"], sub["uy"], c=DC_COLORS[dc], s=3, alpha=0.6, 
               linewidths=0, rasterized=True, label=f"{DC_NAMES[dc]} (n={n_samples:,})")

ax.set_xlim(XMN, XMX)
ax.set_ylim(YMN, YMX)
ax.set_xlabel("UMAP 1", fontsize=12)
ax.set_ylabel("UMAP 2", fontsize=12)
ax.set_title(f"iMEMORY V4 — Disease Class Distribution\n(n={len(df):,} samples)", fontsize=14, fontweight="bold")
ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

for sp in ax.spines.values():
    sp.set_linewidth(0.5)
    sp.set_color("#BBBBBB")

out_path = os.path.join(OUTDIR, "Figure_UMAP_SampleCounts.png")
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path}")

print("\nDONE.")
