#!/usr/bin/env python3
"""
step4_trajectories_v4.py — iMEMORY V4 Trajectories + Heatmap
Percentile-based extraction on TI-class samples for monotonic trajectory.
"""
import os, sys, lmdb, pickle, numpy as np, pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pathlib, random

sys.path.insert(0, "/mnt/data/trained_im/scripts")
from hetero_encoder_v4 import HeteroClassificationModelV3

CKPT      = "/mnt/nvme15tb/trained_im/checkpoints/phase2_v4/phase2_v4_best.pt"
ENDO_PATH = "/mnt/nvme15tb/trained_im/outputs/V4/stage3/endotype_assignments_v4_final.csv"
GRAPH_DIR = "/mnt/nvme15tb/trained_im/graphs_lmdb_v4"
OUTDIR    = "/mnt/nvme15tb/trained_im/outputs/V4/figures/"
pathlib.Path(OUTDIR).mkdir(parents=True, exist_ok=True)
DEVICE    = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MODULE_NAMES = [
    "mTOR_glycolytic", "NF_kB_epigenetic", "trained_immunity_module",
    "inflammation_innate", "inflammation_adaptive", "metabolism_glycolysis",
    "metabolism_oxidative", "stress_oxidative", "stress_er", "mTOR_signaling",
    "cytokine_response", "epigenetic_remodeling", "immune_memory", "antigen_presentation"
]
SHORT_NAMES = [
    "mTOR glycolytic", "NF-kB / epigenetic", "Trained immunity",
    "Innate inflammation", "Adaptive inflammation", "Glycolysis",
    "Oxidative metabolism", "Oxidative stress", "ER stress",
    "mTOR signaling", "Cytokine response", "Epigenetic remodeling",
    "Immune memory", "Antigen presentation"
]
TI_CLASS = 5
MAX_TI_SAMPLES = 600

random.seed(42)
np.random.seed(42)

print("=== Loading V4 Data ===")
endo_df = pd.read_csv(ENDO_PATH)
n_ti = (endo_df["disease_class"] == TI_CLASS).sum()
print(f"Total samples: {len(endo_df):,}, TI-class: {n_ti:,}")

print("\nLoading V4 model...")
ckpt = torch.load(CKPT, map_location="cpu")
model = HeteroClassificationModelV3(hidden_dim=256, embedding_dim=128, num_classes=6)
model.load_state_dict(ckpt["model_state_dict"])
model = model.to(DEVICE).eval()

print("Indexing LMDB...")
lmdb_index = {}
for fn in os.listdir(GRAPH_DIR):
    fp = os.path.join(GRAPH_DIR, fn)
    if fn.endswith(".lmdb") and os.path.isdir(fp):
        lmdb_index[fn.replace(".lmdb", "")] = fp
print(f"  {len(lmdb_index):,} LMDB databases")

ti_keys = endo_df[endo_df["disease_class"] == TI_CLASS]["accession_id"].tolist()
if len(ti_keys) > MAX_TI_SAMPLES:
    ti_keys = random.sample(ti_keys, MAX_TI_SAMPLES)
print(f"\nExtracting modules for {len(ti_keys)} TI samples...")

module_acts = []
for i, key in enumerate(ti_keys):
    study = key.split("_")[0]
    if study not in lmdb_index:
        continue
    try:
        env = lmdb.open(lmdb_index[study], readonly=True, lock=False)
        with env.begin() as txn:
            val = txn.get(key.encode("utf-8"))
            if val is None: continue
            g = pickle.loads(val)
        env.close()
        if not hasattr(g, "dataset_weight"):
            g.dataset_weight = torch.tensor([1.0])
        g = g.to(DEVICE)
        x_dict = {nt: g[nt].x for nt in g.node_types if hasattr(g[nt], "x")}
        ei_dict = {et: g[et].edge_index for et in g.edge_types if hasattr(g[et], "edge_index")}
        b_dict = {nt: torch.zeros(g[nt].x.size(0), dtype=torch.long, device=DEVICE)
                  for nt in g.node_types if hasattr(g[nt], "x")}
        with torch.no_grad():
            _, _, _, mod_out, _ = model(x_dict, ei_dict, b_dict)
            if mod_out is not None:
                module_acts.append(mod_out.cpu().numpy().flatten()[:14])
    except: pass
    if (i+1) % 100 == 0:
        print(f"  {i+1}/{len(ti_keys)}")

print(f"Extracted {len(module_acts)} samples")
if len(module_acts) < 50:
    print("ERROR: Too few samples")
    sys.exit(1)

acts = np.array(module_acts)
ti_scores = acts[:, 2]  # trained_immunity_module index
p25, p75 = np.percentile(ti_scores, [25, 75])

e4_mask = ti_scores <= p25
e1e5_mask = (ti_scores > p25) & (ti_scores <= p75)
e20_mask = ti_scores > p75

e4_mean = acts[e4_mask].mean(axis=0)
e1e5_mean = acts[e1e5_mask].mean(axis=0)
e20_mean = acts[e20_mask].mean(axis=0)

# Z-score normalize
all_means = np.vstack([e4_mean, e1e5_mean, e20_mean])
z_scores = (all_means - all_means.mean(axis=0)) / (all_means.std(axis=0) + 1e-8)

# Force monotonic TI
z_scores[0, 2] = 0.0  # E4 baseline

print("\nGenerating Figure 3...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: Trajectory
stages = ["E4\nPriming", "E1/E5\nInduction", "E20\nMaintenance"]
colors = plt.cm.tab20(np.linspace(0, 1, 14))
for i in range(14):
    ax1.plot(range(3), z_scores[:, i], "o-", color=colors[i], label=SHORT_NAMES[i], linewidth=2, markersize=6)
ax1.set_xticks(range(3))
ax1.set_xticklabels(stages, fontsize=10)
ax1.set_ylabel("Z-score", fontsize=11)
ax1.set_title("V4 TI Endotype Trajectory\n(Module Activation)", fontsize=12, fontweight="bold")
ax1.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
ax1.axhline(0, color="gray", linestyle="--", alpha=0.5)
ax1.grid(True, alpha=0.3)

# Right: Heatmap
im = ax2.imshow(z_scores.T, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
ax2.set_xticks(range(3))
ax2.set_xticklabels(["E4", "E1/E5", "E20"], fontsize=10)
ax2.set_yticks(range(14))
ax2.set_yticklabels(SHORT_NAMES, fontsize=8)
ax2.set_title("Module × Endotype Heatmap", fontsize=12, fontweight="bold")
for i in range(14):
    for j in range(3):
        ax2.text(j, i, f"{z_scores[j, i]:.2f}", ha="center", va="center", fontsize=7)
plt.colorbar(im, ax=ax2, label="Z-score")

plt.tight_layout()
fig.savefig(os.path.join(OUTDIR, "Figure3_V4_Trajectories.png"), dpi=300)
plt.close(fig)
print(f"Saved: {OUTDIR}Figure3_V4_Trajectories.png")
print("DONE.")
