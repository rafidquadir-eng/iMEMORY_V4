#!/usr/bin/env python3
"""
step12_longitudinal_v4.py — iMEMORY V4 Longitudinal Detection Profiling
Uses clean longitudinal candidates (14 studies).
"""
import os, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pathlib

ENDO_PATH = "/mnt/nvme15tb/trained_im/outputs/V4/stage3/endotype_assignments_v4_final.csv"
OUTDIR    = "/mnt/nvme15tb/trained_im/outputs/V4/figures/"
pathlib.Path(OUTDIR).mkdir(parents=True, exist_ok=True)

print("=== Longitudinal Detection V4 ===")
endo = pd.read_csv(ENDO_PATH)
endo["study_acc"] = endo["accession_id"].str.split("_").str[0]
print(f"Loaded {len(endo)} samples from {endo.study_acc.nunique()} studies")

# Identify studies with multiple samples (potential longitudinal)
study_counts = endo.groupby("study_acc").size()
longitudinal_candidates = study_counts[study_counts >= 10].index.tolist()
print(f"Longitudinal candidates (>=10 samples): {len(longitudinal_candidates)} studies")

# Create figure
fig, ax = plt.subplots(figsize=(14, 6))

# Top 20 studies by sample count
top_studies = study_counts.nlargest(20)
colors = plt.cm.viridis(np.linspace(0, 1, len(top_studies)))
bars = ax.barh(range(len(top_studies)), top_studies.values, color=colors)
ax.set_yticks(range(len(top_studies)))
ax.set_yticklabels(top_studies.index, fontsize=8)
ax.set_xlabel("Number of Samples")
ax.set_title("V4 Top 20 Studies by Sample Count\n(Longitudinal Detection Candidates)", fontweight="bold")
ax.invert_yaxis()

for bar, val in zip(bars, top_studies.values):
    ax.text(val + 50, bar.get_y() + bar.get_height()/2, f"{val:,}", va="center", fontsize=8)

plt.tight_layout()
fig.savefig(os.path.join(OUTDIR, "Figure_Longitudinal_Detection_V4.png"), dpi=300)
plt.close(fig)
print(f"Saved: {OUTDIR}Figure_Longitudinal_Detection_V4.png")

# Save longitudinal candidates CSV
long_df = pd.DataFrame({
    "study_acc": longitudinal_candidates,
    "n_samples": [study_counts[s] for s in longitudinal_candidates]
}).sort_values("n_samples", ascending=False)
long_df.to_csv(os.path.join(OUTDIR, "longitudinal_candidates_v4.csv"), index=False)
print(f"Saved: {OUTDIR}longitudinal_candidates_v4.csv ({len(long_df)} studies)")
print("DONE.")
