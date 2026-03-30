#!/usr/bin/env python3
"""
step8_r01_epigenetic_v4.py — iMEMORY V4 R01 Epigenetic NF-kB Figure
"""
import os, sys, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pathlib

ENDO_PATH = "/mnt/nvme15tb/trained_im/outputs/V4/stage3/endotype_assignments_v4_final.csv"
OUTDIR    = "/mnt/nvme15tb/trained_im/outputs/V4/figures/"
pathlib.Path(OUTDIR).mkdir(parents=True, exist_ok=True)

print("=== R01 Epigenetic NF-kB Figure V4 ===")
endo = pd.read_csv(ENDO_PATH)
print(f"Loaded {len(endo)} samples")

# Disease class mapping
DC_NAMES = {0: "Acute Rej", 1: "Chronic Rej", 2: "CKD", 3: "HD", 4: "KTx Stable", 5: "Control/TI"}
endo["dc_name"] = endo["disease_class"].map(DC_NAMES)

# Create figure showing disease progression
fig, ax = plt.subplots(figsize=(10, 6))

# Count samples per disease class
counts = endo["dc_name"].value_counts()
colors = ["#B71C1C", "#E65100", "#C62828", "#1565C0", "#2E7D32", "#6A1B9A"]
bars = ax.bar(range(len(counts)), counts.values, color=colors[:len(counts)])
ax.set_xticks(range(len(counts)))
ax.set_xticklabels(counts.index, rotation=45, ha="right")
ax.set_ylabel("Number of Samples")
ax.set_title("V4 Disease Class Distribution\n(R01 Aim 1: Epigenetic NF-kB Progression)", fontweight="bold")

for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, val + 100, f"n={val:,}", ha="center", fontsize=9)

plt.tight_layout()
fig.savefig(os.path.join(OUTDIR, "Figure_R01_Epigenetic_NF-kB_V4.png"), dpi=300)
plt.close(fig)
print(f"Saved: {OUTDIR}Figure_R01_Epigenetic_NF-kB_V4.png")
print("DONE.")
