#!/usr/bin/env python3
"""
step3_figures_v4.py — iMEMORY V4 Disease Axis and TI Phase figures.
Generates Figure 1 (Disease Axis) and Figure 2 (TI Phase Mapping).
Uses V4 embeddings and endotype assignments.
"""
import os, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial import cKDTree
from collections import Counter
import pathlib

# V4 paths
ENDO_PATH = "/mnt/nvme15tb/trained_im/outputs/V4/stage2/endotype_assignments_v4.csv"
UMAP_PATH = "/mnt/nvme15tb/trained_im/outputs/V4/stage2/umap_coords_v4.npy"
OUTDIR    = "/mnt/nvme15tb/trained_im/outputs/V4/figures/"
pathlib.Path(OUTDIR).mkdir(parents=True, exist_ok=True)

K_PHASE = 50

# Load data
print("Loading V4 data...")
endo = pd.read_csv(ENDO_PATH)
umap_coords = np.load(UMAP_PATH)
endo["umap_x"] = umap_coords[:, 0]
endo["umap_y"] = umap_coords[:, 1]
print(f"Loaded {len(endo)} samples, {endo.refined_endotype.nunique()} clusters")

# Get TI cluster assignments from the data
ti_samples = endo[endo["disease_class"] == 5]
ti_clusters = set(ti_samples["refined_endotype"].unique())
print(f"TI clusters: {len(ti_clusters)}")

# Identify CLEAN_TI (>80% TI) and MIXED_TI (30-80% TI)
CLEAN_TI = set()
MIXED_TI = set()
for cid in sorted(endo["refined_endotype"].unique()):
    sub = endo[endo["refined_endotype"] == cid]
    n_total = len(sub)
    if n_total < 5: continue
    n_ti = (sub["disease_class"] == 5).sum()
    frac = n_ti / n_total
    if frac > 0.80:
        CLEAN_TI.add(cid)
    elif frac > 0.30:
        MIXED_TI.add(cid)

print(f"CLEAN_TI: {len(CLEAN_TI)} clusters")
print(f"MIXED_TI: {len(MIXED_TI)} clusters")

# Identify E4/E1E5/E20 clusters
E4_CLUSTERS = set()
E1E5_CLUSTERS = set()
E20_CLUSTERS = set()

for cid in CLEAN_TI:
    sub = endo[endo["refined_endotype"] == cid]
    n_total = len(sub)
    n_ktx = sub["disease_class"].isin([4]).sum()
    n_rej = sub["disease_class"].isin([0, 1]).sum()
    if n_ktx / n_total > 0.10:
        E4_CLUSTERS.add(cid)
    elif n_rej / n_total > 0.10:
        E1E5_CLUSTERS.add(cid)
    else:
        E20_CLUSTERS.add(cid)

print(f"E4_CLUSTERS: {len(E4_CLUSTERS)}")
print(f"E1E5_CLUSTERS: {len(E1E5_CLUSTERS)}")
print(f"E20_CLUSTERS: {len(E20_CLUSTERS)}")

PHASE_COLORS = {
    "E4 Priming":      "#0077CC",
    "E1/E5 Induction": "#DD2200",
    "E20 Maintenance": "#8800CC",
    "No TI signal":    "#1A1A2E",
}

# Disease cluster palettes - auto-detect top clusters per disease class
def get_top_clusters(df, disease_class, n=7):
    sub = df[df["disease_class"] == disease_class]
    return sub["refined_endotype"].value_counts().head(n).index.tolist()

CKD_TOP = get_top_clusters(endo, 2)
HD_TOP = get_top_clusters(endo, 3)
ACUTE_TOP = get_top_clusters(endo, 0)
CHRONIC_TOP = get_top_clusters(endo, 1)
KTX_TOP = get_top_clusters(endo, 4)

# Color palettes
CKD_COLORS = ["#D32F2F", "#FF6F00", "#C2185B", "#7B1FA2", "#1565C0", "#00838F", "#558B2F"]
HD_COLORS = ["#0288D1", "#FF8F00", "#283593", "#6A1B9A", "#00695C", "#D84315", "#2E7D32"]
KTX_COLORS = ["#B71C1C", "#FF3D00", "#FF8A65", "#E65100", "#F9A825", "#FDD835", "#004D40", "#006064", "#4A148C", "#1B5E20", "#33691E"]

GREY = "#CCCCCC"
PAN_COLOR = "#757575"
GROUP_COLORS = {"CKD": "#C62828", "HD": "#1565C0", "KTx": "#2E7D32", "Control": "#6A1B9A"}
GROUP_LABELS = {"CKD": "CKD", "HD": "HD", "KTx": "KTx\n(incl. rejection)", "Control": "Control"}
GROUP_ORDER = ["CKD", "HD", "Control", "KTx"]
LEGEND_SIDE = {0: "left", 1: "right", 2: "left", 3: "right"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.12})

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

print("\nPlot groups:")
for g in ["CKD", "HD", "KTx", "Control"]:
    print(f"  {g}: {(df_plot['display_group'] == g).sum():,}")

# Build TI k-NN tree
ti_df = df[df["is_ti_class"]].copy().reset_index(drop=True)
ti_coords = ti_df[[X1, X2]].values
ti_cids = ti_df["refined_endotype"].values.astype(int)
ti_tree = cKDTree(ti_coords)
print(f"\nTI k-NN tree: {len(ti_df):,} TI samples")

def assign_ti_phase(grp_df, k=K_PHASE):
    pts = grp_df[[X1, X2]].values
    phases = []
    for px, py in zip(pts[:, 0], pts[:, 1]):
        _, idx = ti_tree.query([px, py], k=min(k, len(ti_coords)))
        nbr = ti_cids[idx]
        counts = {
            "E4 Priming":      sum(c in E4_CLUSTERS for c in nbr),
            "E1/E5 Induction": sum(c in E1E5_CLUSTERS for c in nbr),
            "E20 Maintenance": sum(c in E20_CLUSTERS for c in nbr),
        }
        best = max(counts, key=counts.get)
        phases.append(best if counts[best] >= 1 else "No TI signal")
    return np.array(phases)

def _place_legend(ax, handles, ax_idx):
    if LEGEND_SIDE[ax_idx] == "left":
        ax.legend(handles=handles, bbox_to_anchor=(-0.02, 1.0), loc="upper right",
                  fontsize=4.5, frameon=True, framealpha=0.92, ncol=1,
                  handlelength=0.8, labelspacing=0.20, borderpad=0.4, handletextpad=0.4)
    else:
        ax.legend(handles=handles, bbox_to_anchor=(1.01, 1.0), loc="upper left",
                  fontsize=4.5, frameon=True, framealpha=0.92, ncol=1,
                  handlelength=0.8, labelspacing=0.20, borderpad=0.4, handletextpad=0.4)

# FIGURE 1 — Disease Progression Axis
print("\nGenerating Figure 1...")
fig1, axes = plt.subplots(2, 2, figsize=(13, 9.5))
fig1.subplots_adjust(left=0.22, right=0.82, hspace=0.30, wspace=0.12)
axes_flat = axes.flatten()

for ax_idx, (ax, group) in enumerate(zip(axes_flat, GROUP_ORDER)):
    bg = df_plot[df_plot["display_group"] != group]
    fg = df_plot[df_plot["display_group"] == group].copy()

    ax.scatter(bg[X1], bg[X2], c=GREY, s=2, alpha=0.08, linewidths=0, rasterized=True, zorder=1)

    if group == "Control":
        ax.scatter(fg[X1], fg[X2], c=GROUP_COLORS["Control"], s=4, alpha=1.0,
                   linewidths=0, rasterized=True, zorder=2)
        ax.set_title(f"Control  (n={len(fg):,})", fontsize=11,
                     fontweight="bold", color=GROUP_COLORS["Control"], pad=6)
    else:
        ax.scatter(fg[X1], fg[X2], c=GROUP_COLORS[group], s=4, alpha=0.6,
                   linewidths=0, rasterized=True, zorder=2)
        ax.set_title(f"{GROUP_LABELS[group]}  (n={len(fg):,})", fontsize=11,
                     fontweight="bold", color=GROUP_COLORS[group], pad=6)

    ax.set_xlim(XMN, XMX)
    ax.set_ylim(YMN, YMX)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("UMAP 1", fontsize=8)
    ax.set_ylabel("UMAP 2", fontsize=8)
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
        sp.set_color("#BBBBBB")

fig1.suptitle("iMEMORY V4 Embedding — Disease Progression Axis\n"
              "(UMAP | color = clinical sub-population)",
              fontsize=10, fontweight="bold", y=1.01)
fig1.text(0.01, 0.99, "A", fontsize=14, fontweight="bold", va="top")
p1 = os.path.join(OUTDIR, "Figure1_V4_DiseaseAxis_Annotated.png")
fig1.savefig(p1, dpi=300, bbox_inches="tight")
plt.close(fig1)
print(f"  Saved: {p1}")

# FIGURE 2 — TI Endotype Phase Mapping
print("\nGenerating Figure 2...")
fig2, axes2 = plt.subplots(2, 2, figsize=(13, 9.5))
fig2.subplots_adjust(left=0.22, right=0.82, hspace=0.30, wspace=0.12)
axes2_flat = axes2.flatten()

for ax_idx, (ax, group) in enumerate(zip(axes2_flat, GROUP_ORDER)):
    bg = df_plot[df_plot["display_group"] != group]
    ax.scatter(bg[X1], bg[X2], c=GREY, s=2, alpha=0.08, linewidths=0, rasterized=True, zorder=1)

    if group == "Control":
        fg = df_plot[df_plot["display_group"] == "Control"]
        ax.scatter(fg[X1], fg[X2], c=GROUP_COLORS["Control"], s=4, alpha=1.0,
                   linewidths=0, rasterized=True, zorder=2)
        ax.set_title(f"Control  (clean TI reference, n={len(fg):,})", fontsize=11,
                     fontweight="bold", color=GROUP_COLORS["Control"], pad=6)
    else:
        fg = df_plot[df_plot["display_group"] == group].copy().reset_index(drop=True)
        print(f"  Phase assignment: {group} (n={len(fg):,}, k={K_PHASE})...")
        phases = assign_ti_phase(fg, k=K_PHASE)
        phase_counts = Counter(phases)
        for ph, n in sorted(phase_counts.items(), key=lambda x: -x[1]):
            print(f"    {ph}: {n:,} ({100*n/len(fg):.1f}%)")

        no_ti = phases == "No TI signal"
        ax.scatter(fg[X1].values[no_ti], fg[X2].values[no_ti],
                   c=PHASE_COLORS["No TI signal"], s=5, alpha=1.0,
                   linewidths=0, rasterized=True, zorder=2)

        phase_order = ["E4 Priming", "E1/E5 Induction", "E20 Maintenance"]
        legend_handles = [
            mpatches.Patch(facecolor=PHASE_COLORS["No TI signal"],
                           label=f"No TI signal (n={no_ti.sum():,})")
        ]
        for phase in phase_order:
            mask = phases == phase
            if mask.sum() < 3: continue
            ax.scatter(fg[X1].values[mask], fg[X2].values[mask],
                       c=PHASE_COLORS[phase], s=6, alpha=1.0,
                       linewidths=0, rasterized=True, zorder=4)
            legend_handles.append(
                mpatches.Patch(facecolor=PHASE_COLORS[phase],
                               label=f"{phase} (n={mask.sum():,})"))

        _place_legend(ax, legend_handles, ax_idx)
        ax.set_title(f"{GROUP_LABELS[group]}  ●  TI endotype phases",
                     fontsize=11, fontweight="bold", color=GROUP_COLORS[group], pad=6)

    ax.set_xlim(XMN, XMX)
    ax.set_ylim(YMN, YMX)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("UMAP 1", fontsize=8)
    ax.set_ylabel("UMAP 2", fontsize=8)
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
        sp.set_color("#BBBBBB")

fig2.suptitle("iMEMORY V4 TI Endotype Phase Mapping Within the Disease Continuum\n"
              "(k=50 TI-neighbor phase assignment | No TI signal = biologically TI-negative)",
              fontsize=10, fontweight="bold", y=1.01)
fig2.text(0.01, 0.99, "B", fontsize=14, fontweight="bold", va="top")

leg = [mpatches.Patch(facecolor=GROUP_COLORS[g],
        label=GROUP_LABELS[g].replace("\n", " ")) for g in ["CKD", "HD", "KTx", "Control"]]
leg += [mpatches.Patch(facecolor=v, label=k) for k, v in PHASE_COLORS.items()]
fig2.legend(handles=leg, loc="lower center", ncol=4, fontsize=7,
            frameon=True, framealpha=0.92, bbox_to_anchor=(0.5, -0.04))
p2 = os.path.join(OUTDIR, "Figure2_V4_TrainedImmunity_Annotated.png")
fig2.savefig(p2, dpi=300, bbox_inches="tight")
plt.close(fig2)
print(f"  Saved: {p2}")

# Save final V4 endotype assignments
final_endo = endo.copy()
final_endo_path = "/mnt/nvme15tb/trained_im/outputs/V4/stage3/endotype_assignments_v4_final.csv"
os.makedirs(os.path.dirname(final_endo_path), exist_ok=True)
final_endo.to_csv(final_endo_path, index=False)
print(f"\nSaved final endotype: {final_endo_path}")

print("\nDONE.")
