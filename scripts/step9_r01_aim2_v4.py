#!/usr/bin/env python3
"""
step9_r01_aim2_v4.py — iMEMORY V4 R01 Aim 2 Stratification Figure
"""
import os, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pathlib

ENDO_PATH = "/mnt/nvme15tb/trained_im/outputs/V4/stage3/endotype_assignments_v4_final.csv"
OUTDIR    = "/mnt/nvme15tb/trained_im/outputs/V4/figures/"
pathlib.Path(OUTDIR).mkdir(parents=True, exist_ok=True)

print("=== R01 Aim 2 Stratification V4 ===")
endo = pd.read_csv(ENDO_PATH)
print(f"Loaded {len(endo)} samples")

# Stratify by disease class and cluster
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

DC_NAMES = {0: "Acute Rejection", 1: "Chronic Rejection", 2: "CKD", 
            3: "Hemodialysis", 4: "KTx Stable", 5: "Control/TI"}
colors = ["#B71C1C", "#E65100", "#C62828", "#1565C0", "#2E7D32", "#6A1B9A"]

for i, (dc, name) in enumerate(DC_NAMES.items()):
    ax = axes[i]
    sub = endo[endo["disease_class"] == dc]
    top_clusters = sub["refined_endotype"].value_counts().head(10)
    ax.barh(range(len(top_clusters)), top_clusters.values, color=colors[i])
    ax.set_yticks(range(len(top_clusters)))
    ax.set_yticklabels([f"C{c}" for c in top_clusters.index], fontsize=8)
    ax.set_xlabel("Samples")
    ax.set_title(f"{name}\n(n={len(sub):,})", fontweight="bold", fontsize=10)
    ax.invert_yaxis()

plt.suptitle("V4 R01 Aim 2: Disease Stratification by Endotype Cluster", fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(OUTDIR, "Figure_R01_AimTwo_Stratification_V4.png"), dpi=300)
plt.close(fig)
print(f"Saved: {OUTDIR}Figure_R01_AimTwo_Stratification_V4.png")
print("DONE.")
