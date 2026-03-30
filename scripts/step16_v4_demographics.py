#!/usr/bin/env python3
"""
step16_v4_demographics.py — iMEMORY V4
Generates Table 1 demographics and STROBE diagram using V4 clean endotype.
"""
import os, pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pathlib

# V4 paths
ENDO_PATH = "/mnt/nvme15tb/trained_im/outputs/V4/stage3/endotype_assignments_v4_final.csv"
OUT_ROOT  = "/mnt/nvme15tb/trained_im/outputs/V4/demographics/"
pathlib.Path(OUT_ROOT).mkdir(parents=True, exist_ok=True)

COHORT_KEYS   = ["CKD", "HD", "KTx_Acute", "KTx_Chronic", "KTx_Stable", "Control/TI"]
COHORT_LABELS = ["CKD", "Hemodialysis\n(HD)", "KTx\nAcute Rej", "KTx\nChronic Rej", "KTx\nStable", "Control /\nTrained Immunity"]
DC_TO_COHORT  = {2: "CKD", 3: "HD", 0: "KTx_Acute", 1: "KTx_Chronic", 4: "KTx_Stable", 5: "Control/TI"}
DISEASE_COLORS = {
    "CKD": "#C62828", "HD": "#1565C0", "KTx_Acute": "#B71C1C",
    "KTx_Chronic": "#E65100", "KTx_Stable": "#2E7D32", "Control/TI": "#6A1B9A",
}

print("=== Loading V4 Endotype Data ===")
endo = pd.read_csv(ENDO_PATH)
endo["cohort"] = endo["disease_class"].map(DC_TO_COHORT)
print(f"Total samples: {len(endo)}")

# Cohort breakdown
print("\n=== Cohort Breakdown ===")
cohort_counts = {}
for cohort in COHORT_KEYS:
    n = (endo["cohort"] == cohort).sum()
    cohort_counts[cohort] = n
    print(f"  {cohort}: {n:,}")

# Generate Table 1 figure
print("\n=== Generating Table 1 Figure ===")
fig, ax = plt.subplots(figsize=(12, 6))

# Bar chart of cohort sizes
x = range(len(COHORT_KEYS))
heights = [cohort_counts[c] for c in COHORT_KEYS]
colors = [DISEASE_COLORS[c] for c in COHORT_KEYS]
bars = ax.bar(x, heights, color=colors, edgecolor="black", linewidth=0.5)

# Add count labels on bars
for bar, h in zip(bars, heights):
    ax.text(bar.get_x() + bar.get_width()/2, h + 100, f"n={h:,}",
            ha="center", va="bottom", fontsize=10, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(COHORT_LABELS, fontsize=10)
ax.set_ylabel("Number of Samples", fontsize=12)
ax.set_title("iMEMORY V4 — Sample Distribution by Disease Cohort\n(Clean Dataset: Contaminated Studies Excluded)", fontsize=12, fontweight="bold")
ax.set_ylim(0, max(heights) * 1.15)

# Add total
total = sum(heights)
ax.text(0.98, 0.95, f"Total: {total:,} samples", transform=ax.transAxes,
        ha="right", va="top", fontsize=11, fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

plt.tight_layout()
fig.savefig(os.path.join(OUT_ROOT, "Figure_Table1_Clinical_V4.png"), dpi=300)
plt.close(fig)
print(f"Saved: {OUT_ROOT}Figure_Table1_Clinical_V4.png")

# Generate STROBE flow diagram
print("\n=== Generating STROBE Flow Diagram ===")
fig, ax = plt.subplots(figsize=(10, 12))
ax.set_xlim(0, 10)
ax.set_ylim(0, 14)
ax.axis("off")

def draw_box(ax, x, y, w, h, text, color="white", fontsize=9):
    rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                    facecolor=color, edgecolor="black", linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fontsize,
            wrap=True, multialignment="center")

def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

# STROBE boxes
draw_box(ax, 2, 12.5, 6, 1, "GEO/ArrayExpress/SRA\nInitial Query\n(n=887 studies)", "#E3F2FD", 10)
draw_arrow(ax, 5, 12.5, 5, 11.5)

draw_box(ax, 2, 10, 6, 1.5, f"NPZ Files Generated\n(n=840 studies)\nTotal graphs: ~200K", "#E8F5E9", 10)
draw_arrow(ax, 5, 10, 5, 9)

draw_box(ax, 2, 7.5, 6, 1.5, "Contamination Audit\n(47 studies excluded)\n- Non-human\n- Non-nephrology tissue", "#FFEBEE", 9)
draw_arrow(ax, 5, 7.5, 5, 6.5)

draw_box(ax, 2, 5, 6, 1.5, f"V4 Clean Dataset\n(n={total:,} samples)\n840 studies", "#E8F5E9", 10)
draw_arrow(ax, 5, 5, 5, 4)

# Final cohort boxes
y_cohort = 2
for i, (cohort, label) in enumerate(zip(COHORT_KEYS, COHORT_LABELS)):
    x_pos = 0.5 + i * 1.5
    n = cohort_counts[cohort]
    draw_box(ax, x_pos, y_cohort, 1.4, 1.5, f"{label.replace(chr(10), ' ')}\nn={n:,}",
             DISEASE_COLORS[cohort] + "40", 7)

ax.set_title("iMEMORY V4 STROBE Flow Diagram", fontsize=14, fontweight="bold", y=0.98)
plt.tight_layout()
fig.savefig(os.path.join(OUT_ROOT, "Figure_STROBE_FlowDiagram_V4.png"), dpi=300)
plt.close(fig)
print(f"Saved: {OUT_ROOT}Figure_STROBE_FlowDiagram_V4.png")

# Save demographics CSV
demo_df = endo.groupby("cohort").size().reset_index(name="n_samples")
demo_df.to_csv(os.path.join(OUT_ROOT, "table1_demographics_v4.csv"), index=False)
print(f"Saved: {OUT_ROOT}table1_demographics_v4.csv")

print("\nDONE.")
