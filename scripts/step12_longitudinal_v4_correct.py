#!/usr/bin/env python3
"""
step12_longitudinal_v4_correct.py — V4 Longitudinal Analysis
Uses curated longitudinal candidates from V3.1
"""
import os, json, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial import cKDTree
import pathlib

ENDO_PATH = "/mnt/nvme15tb/trained_im/outputs/V4/stage3/endotype_assignments_v4_final.csv"
LONGIT_PATH = "/mnt/nvme15tb/trained_im/outputs/profiling_v3_1/demographics/longitudinal_candidates_clean.csv"
OUTDIR = "/mnt/nvme15tb/trained_im/outputs/V4/profiling/longitudinal/"
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

print("Loading longitudinal candidates...")
longit = pd.read_csv(LONGIT_PATH)
print(f"  {len(longit)} curated longitudinal studies")
print(longit)

# Extract study ID from accession_id
endo["study_id"] = endo["accession_id"].str.extract(r"(GSE\d+|E-MTAB-\d+)")[0]

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

# Filter to longitudinal studies
longit_studies = set(longit["accession"].tolist()) if "accession" in longit.columns else set(longit.iloc[:, 0].tolist())
print(f"\nLongitudinal studies: {longit_studies}")

endo_longit = endo[endo["study_id"].isin(longit_studies)].copy()
print(f"Samples in longitudinal studies: {len(endo_longit)}")

if len(endo_longit) == 0:
    print("WARNING: No samples found in longitudinal studies. Checking study IDs...")
    print("V4 study IDs:", endo["study_id"].unique()[:20])
    # Try matching without GSE prefix
    longit_studies_alt = set()
    for s in longit_studies:
        longit_studies_alt.add(s)
        if s.startswith("GSE"):
            longit_studies_alt.add(s[3:])
    endo_longit = endo[endo["study_id"].isin(longit_studies_alt)].copy()
    print(f"After alt matching: {len(endo_longit)} samples")

# ─── Longitudinal Analysis Figure ─────────────────────────────────────────────
print("\nBuilding Longitudinal Analysis figure...")

fig = plt.figure(figsize=(18, 12))

# Left: UMAP with longitudinal samples highlighted
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

# Background
ax1.scatter(df_plot["ux"], df_plot["uy"], c="#CCCCCC", s=2, alpha=0.1, linewidths=0, rasterized=True)

# Longitudinal samples
if len(endo_longit) > 0:
    longit_plot = endo_longit[(endo_longit["umap_x"] >= XMN) & (endo_longit["umap_x"] <= XMX) &
                              (endo_longit["umap_y"] >= YMN) & (endo_longit["umap_y"] <= YMX)]
    for dc in sorted(DC_COLORS.keys()):
        sub = longit_plot[longit_plot["disease_class"] == dc]
        if len(sub) > 0:
            ax1.scatter(sub["umap_x"], sub["umap_y"], c=DC_COLORS[dc], s=8, alpha=0.8, 
                       linewidths=0, rasterized=True, label=f"{DC_NAMES[dc]} (n={len(sub):,})")

ax1.set_xlim(XMN, XMX)
ax1.set_ylim(YMN, YMX)
ax1.set_xlabel("UMAP 1", fontsize=11)
ax1.set_ylabel("UMAP 2", fontsize=11)
ax1.set_title(f"Longitudinal Samples\n({len(endo_longit):,} samples from {len(longit_studies)} studies)", fontsize=12, fontweight="bold")
ax1.legend(loc="upper right", fontsize=8)

# Middle: TI phase distribution in longitudinal samples
ax2 = fig.add_subplot(1, 3, 2)

if len(endo_longit) > 0:
    phase_counts = endo_longit["ti_phase_assigned"].value_counts()
    phases = ["E4 Priming", "E1/E5 Induction", "E20 Maintenance", "No TI signal"]
    vals = [phase_counts.get(p, 0) for p in phases]
    colors = [PHASE_COLORS[p] for p in phases]
    
    ax2.bar(range(len(phases)), vals, color=colors)
    ax2.set_xticks(range(len(phases)))
    ax2.set_xticklabels([p.replace(" ", "\n") for p in phases], fontsize=9)
    ax2.set_ylabel("Number of Samples", fontsize=11)
    ax2.set_title("TI Phase Distribution\nin Longitudinal Samples", fontsize=12, fontweight="bold")
else:
    ax2.text(0.5, 0.5, "No longitudinal samples found", ha="center", va="center", fontsize=12)
    ax2.set_title("TI Phase Distribution", fontsize=12, fontweight="bold")

# Right: Study breakdown
ax3 = fig.add_subplot(1, 3, 3)

if len(endo_longit) > 0:
    study_counts = endo_longit.groupby("study_id").size().sort_values(ascending=True)
    y_pos = np.arange(len(study_counts))
    ax3.barh(y_pos, study_counts.values, color="#4CAF50", height=0.7)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(study_counts.index, fontsize=8)
    ax3.set_xlabel("Number of Samples", fontsize=11)
    ax3.set_title("Samples per Longitudinal Study", fontsize=12, fontweight="bold")
else:
    ax3.text(0.5, 0.5, "No longitudinal samples found", ha="center", va="center", fontsize=12)
    ax3.set_title("Samples per Study", fontsize=12, fontweight="bold")

fig.suptitle("iMEMORY V4 — Longitudinal Analysis\n(Curated paired-sample studies)", fontsize=14, fontweight="bold", y=0.98)
plt.tight_layout()

out_path = os.path.join(OUTDIR, "Figure_Longitudinal_Analysis.png")
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path}")

# Save longitudinal studies summary
summary = []
for study in longit_studies:
    sub = endo_longit[endo_longit["study_id"] == study]
    if len(sub) > 0:
        summary.append({
            "study_id": study,
            "n_samples": len(sub),
            "disease_classes": sub["disease_class"].unique().tolist(),
            "ti_phases": sub["ti_phase_assigned"].value_counts().to_dict(),
        })

summary_df = pd.DataFrame(summary)
summary_path = os.path.join(OUTDIR, "longitudinal_studies_summary.csv")
summary_df.to_csv(summary_path, index=False)
print(f"Saved: {summary_path}")

print("\nDONE.")
