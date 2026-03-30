#!/usr/bin/env python3
"""
step10_scrna_v4.py — iMEMORY V4 scRNA Biomarkers Figure
"""
import os, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pathlib

ENDO_PATH = "/mnt/nvme15tb/trained_im/outputs/V4/stage3/endotype_assignments_v4_final.csv"
OUTDIR    = "/mnt/nvme15tb/trained_im/outputs/V4/figures/"
pathlib.Path(OUTDIR).mkdir(parents=True, exist_ok=True)

print("=== scRNA Biomarkers V4 ===")
endo = pd.read_csv(ENDO_PATH)
print(f"Loaded {len(endo)} samples")

# Create biomarker summary figure
fig, ax = plt.subplots(figsize=(12, 6))

# Cluster distribution
cluster_counts = endo["refined_endotype"].value_counts().head(20)
ax.bar(range(len(cluster_counts)), cluster_counts.values, color="#1565C0")
ax.set_xticks(range(len(cluster_counts)))
ax.set_xticklabels([f"C{c}" for c in cluster_counts.index], rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Number of Samples")
ax.set_xlabel("Cluster ID")
ax.set_title("V4 Top 20 Endotype Clusters\n(scRNA Biomarker Candidates)", fontweight="bold")

plt.tight_layout()
fig.savefig(os.path.join(OUTDIR, "Figure_R01_scRNA_Biomarkers_V4.png"), dpi=300)
plt.close(fig)
print(f"Saved: {OUTDIR}Figure_R01_scRNA_Biomarkers_V4.png")
print("DONE.")
