#!/usr/bin/env python3
"""
step7_ochando_v4_correct.py — V4 Ochando Slide 3A/3B/3C with correct V4 cluster sets
"""
import os, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from scipy.spatial import cKDTree
from collections import Counter
import pathlib

ENDO_PATH = "/mnt/nvme15tb/trained_im/outputs/V4/stage3/endotype_assignments_v4_final.csv"
OUTDIR    = "/mnt/nvme15tb/trained_im/outputs/V4/figures/"
pathlib.Path(OUTDIR).mkdir(parents=True, exist_ok=True)

K_PHASE = 50

# V4 TI CLUSTER SETS
CLEAN_TI = {5, 8, 14, 34, 60, 61, 66, 69, 86, 98, 101, 107, 154, 169, 192, 194, 205, 211, 213, 216, 221, 223, 224, 226, 227, 228, 231, 232, 234, 235, 236, 237, 238, 240, 244, 245, 247, 250, 253, 254, 255, 258, 263, 264, 265, 266, 269, 270, 271, 274, 275, 279, 280, 281, 282, 284, 285, 294, 301, 311, 312, 313, 315, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395}
MIXED_TI = {257, 17, 162, 40, 41, 43, 51, 181, 187, 189, 191, 323, 206, 207, 208, 209, 345, 220, 93, 222, 99, 230, 233, 115, 246}
E4_CLUSTERS = {107}
E1E5_CLUSTERS = {353, 194, 34, 294, 351, 265, 244, 347, 60, 223}
E20_CLUSTERS = CLEAN_TI - E4_CLUSTERS - E1E5_CLUSTERS

GROUP_COLORS = {"CKD": "#C62828", "HD": "#1565C0", "KTx": "#2E7D32", "Control": "#6A1B9A"}
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

# Map display groups
DISPLAY_MAP = {0: "KTx", 1: "KTx", 2: "CKD", 3: "HD", 4: "KTx"}
endo["display_group"] = endo["disease_class"].astype(int).map(DISPLAY_MAP)
ec = endo["refined_endotype"].astype(int)
ti = endo["disease_class"].astype(int) == 5

endo.loc[ti & ec.isin(CLEAN_TI), "display_group"] = "Control"
endo.loc[ti & ec.isin(MIXED_TI), "display_group"] = "TI_mixed"
endo.loc[ti & ~ec.isin(CLEAN_TI) & ~ec.isin(MIXED_TI), "display_group"] = "TI_impure"

df_all = endo[endo["display_group"].notna()].copy().reset_index(drop=True)
df_all["is_ti_class"] = (df_all["disease_class"].astype(int) == 5)
df_all["ux"] = df_all["umap_x"]
df_all["uy"] = df_all["umap_y"]

pad = 0.3
XMN = np.percentile(df_all["ux"], 1) - pad
XMX = np.percentile(df_all["ux"], 99) + pad
YMN = np.percentile(df_all["uy"], 1) - pad
YMX = np.percentile(df_all["uy"], 99) + pad

df = df_all[(df_all["ux"] >= XMN) & (df_all["ux"] <= XMX) &
            (df_all["uy"] >= YMN) & (df_all["uy"] <= YMX)].copy()
df_plot = df[df["display_group"].isin(["CKD", "HD", "KTx", "Control"])].copy()
X1, X2 = "ux", "uy"

# Build TI k-NN tree
ti_df = df[df["is_ti_class"]].copy().reset_index(drop=True)
ti_coords = ti_df[[X1, X2]].values
ti_cids = ti_df["refined_endotype"].values.astype(int)
ti_tree = cKDTree(ti_coords)

def assign_ti_phase(grp_df, k=K_PHASE):
    pts = grp_df[[X1, X2]].values
    phases = []
    for px, py in zip(pts[:, 0], pts[:, 1]):
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

# ─── Slide 3A: E4 Priming ─────────────────────────────────────────────────────
print("\nBuilding Ochando Slide 3A (E4 Priming)...")
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

for ax_idx, group in enumerate(["CKD", "HD", "KTx", "Control"]):
    ax = axes[ax_idx]
    
    bg = df_plot[df_plot["display_group"] != group]
    ax.scatter(bg[X1], bg[X2], c=GREY, s=2, alpha=0.08, linewidths=0, rasterized=True, zorder=1)
    
    if group == "Control":
        # Show E4 clusters in Control
        fg = df_plot[(df_plot["display_group"] == "Control") & (df_plot["refined_endotype"].astype(int).isin(E4_CLUSTERS))]
        ax.scatter(fg[X1], fg[X2], c=PHASE_COLORS["E4 Priming"], s=6, alpha=0.9, linewidths=0, rasterized=True, zorder=3)
        ax.set_title(f"Control E4 Priming\n(n={len(fg):,})", fontsize=11, fontweight="bold", color=PHASE_COLORS["E4 Priming"])
    else:
        fg = df_plot[df_plot["display_group"] == group].copy().reset_index(drop=True)
        phases = assign_ti_phase(fg, k=K_PHASE)
        
        # Highlight E4 Priming
        e4_mask = phases == "E4 Priming"
        other_mask = ~e4_mask
        
        ax.scatter(fg[X1].values[other_mask], fg[X2].values[other_mask], c=GREY, s=3, alpha=0.3, linewidths=0, rasterized=True, zorder=2)
        ax.scatter(fg[X1].values[e4_mask], fg[X2].values[e4_mask], c=PHASE_COLORS["E4 Priming"], s=6, alpha=0.9, linewidths=0, rasterized=True, zorder=3)
        
        n_e4 = e4_mask.sum()
        pct = 100 * n_e4 / len(fg) if len(fg) > 0 else 0
        ax.set_title(f"{group} E4 Priming\n(n={n_e4:,}, {pct:.1f}%)", fontsize=11, fontweight="bold", color=PHASE_COLORS["E4 Priming"])
    
    ax.set_xlim(XMN, XMX)
    ax.set_ylim(YMN, YMX)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
        sp.set_color("#BBBBBB")

fig.suptitle("iMEMORY V4 — Ochando Slide 3A: E4 Priming Phase", fontsize=14, fontweight="bold", y=0.98)
out_path = os.path.join(OUTDIR, "Figure_Slide3A_E4_Priming.png")
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path}")

# ─── Slide 3B: E1/E5 Induction ────────────────────────────────────────────────
print("\nBuilding Ochando Slide 3B (E1/E5 Induction)...")
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

for ax_idx, group in enumerate(["CKD", "HD", "KTx", "Control"]):
    ax = axes[ax_idx]
    
    bg = df_plot[df_plot["display_group"] != group]
    ax.scatter(bg[X1], bg[X2], c=GREY, s=2, alpha=0.08, linewidths=0, rasterized=True, zorder=1)
    
    if group == "Control":
        fg = df_plot[(df_plot["display_group"] == "Control") & (df_plot["refined_endotype"].astype(int).isin(E1E5_CLUSTERS))]
        ax.scatter(fg[X1], fg[X2], c=PHASE_COLORS["E1/E5 Induction"], s=6, alpha=0.9, linewidths=0, rasterized=True, zorder=3)
        ax.set_title(f"Control E1/E5 Induction\n(n={len(fg):,})", fontsize=11, fontweight="bold", color=PHASE_COLORS["E1/E5 Induction"])
    else:
        fg = df_plot[df_plot["display_group"] == group].copy().reset_index(drop=True)
        phases = assign_ti_phase(fg, k=K_PHASE)
        
        e1e5_mask = phases == "E1/E5 Induction"
        other_mask = ~e1e5_mask
        
        ax.scatter(fg[X1].values[other_mask], fg[X2].values[other_mask], c=GREY, s=3, alpha=0.3, linewidths=0, rasterized=True, zorder=2)
        ax.scatter(fg[X1].values[e1e5_mask], fg[X2].values[e1e5_mask], c=PHASE_COLORS["E1/E5 Induction"], s=6, alpha=0.9, linewidths=0, rasterized=True, zorder=3)
        
        n_e1e5 = e1e5_mask.sum()
        pct = 100 * n_e1e5 / len(fg) if len(fg) > 0 else 0
        ax.set_title(f"{group} E1/E5 Induction\n(n={n_e1e5:,}, {pct:.1f}%)", fontsize=11, fontweight="bold", color=PHASE_COLORS["E1/E5 Induction"])
    
    ax.set_xlim(XMN, XMX)
    ax.set_ylim(YMN, YMX)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
        sp.set_color("#BBBBBB")

fig.suptitle("iMEMORY V4 — Ochando Slide 3B: E1/E5 Induction Phase", fontsize=14, fontweight="bold", y=0.98)
out_path = os.path.join(OUTDIR, "Figure_Slide3B_E1E5_Induction.png")
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path}")

# ─── Slide 3C: E20 Maintenance ────────────────────────────────────────────────
print("\nBuilding Ochando Slide 3C (E20 Maintenance)...")
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

for ax_idx, group in enumerate(["CKD", "HD", "KTx", "Control"]):
    ax = axes[ax_idx]
    
    bg = df_plot[df_plot["display_group"] != group]
    ax.scatter(bg[X1], bg[X2], c=GREY, s=2, alpha=0.08, linewidths=0, rasterized=True, zorder=1)
    
    if group == "Control":
        fg = df_plot[(df_plot["display_group"] == "Control") & (df_plot["refined_endotype"].astype(int).isin(E20_CLUSTERS))]
        ax.scatter(fg[X1], fg[X2], c=PHASE_COLORS["E20 Maintenance"], s=6, alpha=0.9, linewidths=0, rasterized=True, zorder=3)
        ax.set_title(f"Control E20 Maintenance\n(n={len(fg):,})", fontsize=11, fontweight="bold", color=PHASE_COLORS["E20 Maintenance"])
    else:
        fg = df_plot[df_plot["display_group"] == group].copy().reset_index(drop=True)
        phases = assign_ti_phase(fg, k=K_PHASE)
        
        e20_mask = phases == "E20 Maintenance"
        other_mask = ~e20_mask
        
        ax.scatter(fg[X1].values[other_mask], fg[X2].values[other_mask], c=GREY, s=3, alpha=0.3, linewidths=0, rasterized=True, zorder=2)
        ax.scatter(fg[X1].values[e20_mask], fg[X2].values[e20_mask], c=PHASE_COLORS["E20 Maintenance"], s=6, alpha=0.9, linewidths=0, rasterized=True, zorder=3)
        
        n_e20 = e20_mask.sum()
        pct = 100 * n_e20 / len(fg) if len(fg) > 0 else 0
        ax.set_title(f"{group} E20 Maintenance\n(n={n_e20:,}, {pct:.1f}%)", fontsize=11, fontweight="bold", color=PHASE_COLORS["E20 Maintenance"])
    
    ax.set_xlim(XMN, XMX)
    ax.set_ylim(YMN, YMX)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
        sp.set_color("#BBBBBB")

fig.suptitle("iMEMORY V4 — Ochando Slide 3C: E20 Maintenance Phase", fontsize=14, fontweight="bold", y=0.98)
out_path = os.path.join(OUTDIR, "Figure_Slide3C_E20_Maintenance.png")
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path}")

print("\nDONE.")
