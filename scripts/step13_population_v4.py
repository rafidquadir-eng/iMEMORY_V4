#!/usr/bin/env python3
"""
step13_population_v4.py — iMEMORY V4 Study Population Characterization
"""
import os, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pathlib

ENDO_PATH = "/mnt/nvme15tb/trained_im/outputs/V4/stage3/endotype_assignments_v4_final.csv"
OUTDIR    = "/mnt/nvme15tb/trained_im/outputs/V4/figures/"
pathlib.Path(OUTDIR).mkdir(parents=True, exist_ok=True)

print("=== Study Population Characterization V4 ===")
endo = pd.read_csv(ENDO_PATH)
endo["study_acc"] = endo["accession_id"].str.split("_").str[0]
print(f"Loaded {len(endo)} samples from {endo.study_acc.nunique()} studies")

DC_NAMES = {0: "Acute Rej", 1: "Chronic Rej", 2: "CKD", 3: "HD", 4: "KTx Stable", 5: "Control/TI"}

# Study-level disease class distribution
study_dc = endo.groupby(["study_acc", "disease_class"]).size().unstack(fill_value=0)
study_dc.columns = [DC_NAMES.get(c, c) for c in study_dc.columns]

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Studies per disease class
ax1 = axes[0]
dc_counts = endo.groupby("disease_class").apply(lambda x: x.study_acc.nunique())
dc_counts.index = [DC_NAMES.get(i, i) for i in dc_counts.index]
colors = ["#B71C1C", "#E65100", "#C62828", "#1565C0", "#2E7D32", "#6A1B9A"]
ax1.bar(range(len(dc_counts)), dc_counts.values, color=colors[:len(dc_counts)])
ax1.set_xticks(range(len(dc_counts)))
ax1.set_xticklabels(dc_counts.index, rotation=45, ha="right")
ax1.set_ylabel("Number of Studies")
ax1.set_title("Studies per Disease Class", fontweight="bold")

# Right: Samples per disease class
ax2 = axes[1]
sample_counts = endo["disease_class"].value_counts().sort_index()
sample_counts.index = [DC_NAMES.get(i, i) for i in sample_counts.index]
ax2.bar(range(len(sample_counts)), sample_counts.values, color=colors[:len(sample_counts)])
ax2.set_xticks(range(len(sample_counts)))
ax2.set_xticklabels(sample_counts.index, rotation=45, ha="right")
ax2.set_ylabel("Number of Samples")
ax2.set_title("Samples per Disease Class", fontweight="bold")

plt.suptitle("V4 Study Population Characterization", fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(OUTDIR, "Figure_StudyPopulation_V4.png"), dpi=300)
plt.close(fig)
print(f"Saved: {OUTDIR}Figure_StudyPopulation_V4.png")
print("DONE.")
