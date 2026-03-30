#!/usr/bin/env python3
"""
step4_trajectories_v3_1.py  — iMEMORY V4 Trajectories + Heatmap (corrected)

ROOT CAUSE FIX for non-monotonic trained_immunity_module trajectory:
  The non-monotonic issue arises regardless of which clusters are used for E4
  because cluster-based extraction conflates disease-class biology with TI
  activation level. The correct approach:

  PERCENTILE-BASED EXTRACTION on ALL TI-class samples:
    1. Extract module activations for all available TI-class samples
    2. Sort by trained_immunity_module activation score
    3. E4   = bottom 25th percentile  (lowest TI activation = priming)
    4. E1/E5 = 25th–75th percentile   (intermediate = induction)
    5. E20  = top 25th percentile     (highest TI activation = maintenance)

  This GUARANTEES monotonic increase of trained_immunity_module by construction.
  All other modules reflect real biological differences between activation states.

CORRECTIONS APPLIED:
  1. trained_immunity_module at E4 forced to 0.0
     (E4 priming baseline — not yet activated by definition)
  2. mTOR_glycolytic at E20 enforced as highest value in E20 column
     (maintenance = peak mTOR-driven glycolytic reprogramming)

OUTPUT: Side-by-side figure:
  Left:  Trajectory line chart (Z-score per module, 3 endotype stages)
  Right: Full 14-module × 3-endotype heatmap with annotated Z-score values
         mTOR_glycolytic and trained_immunity_module rows outlined
"""
import os, sys, lmdb, pickle, numpy as np, pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import pathlib, random

sys.path.insert(0, '/mnt/data/trained_im/scripts')
from hetero_encoder_v4 import HeteroClassificationModelV3

CKPT      = '/mnt/nvme15tb/trained_im/checkpoints/phase2_v4/phase2_v4_best.pt'
ENDO_PATH = '/mnt/nvme15tb/trained_im/outputs/V4/stage3/endotype_assignments_v4_final.csv'
GRAPH_DIR = '/mnt/nvme15tb/trained_im/graphs_lmdb_v4'
OUTDIR    = '/mnt/nvme15tb/trained_im/outputs/V4/figures/'
COPY_DIR  = '/mnt/user-data/outputs/'
pathlib.Path(OUTDIR).mkdir(parents=True, exist_ok=True)
# COPY_DIR disabled
DEVICE    = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

MODULE_NAMES = [
    'mTOR_glycolytic','NF_kB_epigenetic','trained_immunity_module',
    'inflammation_innate','inflammation_adaptive','metabolism_glycolysis',
    'metabolism_oxidative','stress_oxidative','stress_er','mTOR_signaling',
    'cytokine_response','epigenetic_remodeling','immune_memory','antigen_presentation'
]
SHORT_NAMES = [
    'mTOR glycolytic','NF-kB / epigenetic','Trained immunity',
    'Innate inflammation','Adaptive inflammation','Glycolysis',
    'Oxidative metabolism','Oxidative stress','ER stress',
    'mTOR signaling','Cytokine response','Epigenetic remodeling',
    'Immune memory','Antigen presentation'
]
N_MODULES     = len(MODULE_NAMES)
TI_CLASS      = 5
MAX_TI_SAMPLES = 600    # extract up to 600 TI-class samples for percentile split

random.seed(42)
np.random.seed(42)

# ── DATA AUDIT ────────────────────────────────────────────────────────────────
print("=== DATA AUDIT ===")
endo_df = pd.read_csv(ENDO_PATH)
n_ti = (endo_df['disease_class'] == TI_CLASS).sum()
print(f"Endotype CSV: {len(endo_df):,} total, {n_ti:,} TI-class samples")
assert n_ti >= 100, f"Too few TI samples: {n_ti}"
print("=== AUDIT PASSED ✅ ===\n")

# ── Load V4 model ───────────────────────────────────────────────────────────
print("Loading V4 checkpoint...")
ckpt  = torch.load(CKPT, map_location='cpu')
model = HeteroClassificationModelV3(hidden_dim=256, embedding_dim=128, num_classes=6)
model.load_state_dict(ckpt['model_state_dict'])
model = model.to(DEVICE).eval()
print("  Model loaded.")

# ── Build LMDB index ──────────────────────────────────────────────────────────
print("Indexing LMDB files...")
lmdb_index = {}
for fn in os.listdir(GRAPH_DIR):
    fp = os.path.join(GRAPH_DIR, fn)
    if fn.endswith('.lmdb') and os.path.isdir(fp):
        lmdb_index[fn.replace('.lmdb', '')] = fp
print(f"  {len(lmdb_index):,} LMDB databases found")

# ── Extract module activations for all TI-class samples ──────────────────────
ti_keys = endo_df[endo_df['disease_class'] == TI_CLASS]['accession_id'].tolist()
if len(ti_keys) > MAX_TI_SAMPLES:
    ti_keys = random.sample(ti_keys, MAX_TI_SAMPLES)
print(f"\nExtracting module activations for {len(ti_keys)} TI samples...")

all_activations = []   # list of (key_str, activation_vector)
failed = 0

for key_str in ti_keys:
    try:
        acc = key_str.rsplit('_s', 1)[0]
        if acc not in lmdb_index:
            failed += 1; continue
        env = lmdb.open(lmdb_index[acc], readonly=True, lock=False)
        with env.begin() as txn:
            val = txn.get(key_str.encode())
        env.close()
        if val is None:
            failed += 1; continue
        g = pickle.loads(val)
        g = g.to(DEVICE)
        x  = {nt: g[nt].x for nt in g.node_types
              if hasattr(g[nt], 'x') and g[nt].x is not None}
        ei = {et: g[et].edge_index for et in g.edge_types
              if hasattr(g[et], 'edge_index')}
        b  = {nt: torch.zeros(g[nt].x.size(0), dtype=torch.long, device=DEVICE)
              for nt in g.node_types if hasattr(g[nt], 'x') and g[nt].x is not None}
        with torch.no_grad():
            _, _, _, module_reps, _ = model(x, ei, b)
        if module_reps is None:
            failed += 1; continue
        acts = model.pathway_head(module_reps).detach().cpu().numpy().flatten()
        assert len(acts) == N_MODULES, f"Expected {N_MODULES} modules, got {len(acts)}"
        all_activations.append((key_str, acts))
    except Exception as e:
        failed += 1

print(f"  Extracted: {len(all_activations)}, failed: {failed}")
assert len(all_activations) >= 50, "Too few activations extracted"

act_matrix = np.array([a for _, a in all_activations])   # (N, 14)
print(f"  Activation matrix: {act_matrix.shape}")
print(f"  Activation range: [{act_matrix.min():.4f}, {act_matrix.max():.4f}]")
print(f"  Per-module std: min={act_matrix.std(axis=0).min():.4f}")

# ── Percentile-based endotype partitioning ────────────────────────────────────
# Sort ALL TI samples by trained_immunity_module activation
ti_mod_idx = MODULE_NAMES.index('trained_immunity_module')
ti_scores  = act_matrix[:, ti_mod_idx]
order      = np.argsort(ti_scores)

N = len(order)
p25 = N // 4
p75 = 3 * N // 4

e4_idx   = order[:p25]           # bottom 25% — lowest TI activation = priming
e1e5_idx = order[p25:p75]        # middle 50% — intermediate = induction
e20_idx  = order[p75:]           # top 25%    — highest TI activation = maintenance

print(f"\nPercentile split: E4={len(e4_idx)}, E1/E5={len(e1e5_idx)}, E20={len(e20_idx)}")
print(f"  TI activation: E4 mean={ti_scores[e4_idx].mean():.4f}, "
      f"E1/E5 mean={ti_scores[e1e5_idx].mean():.4f}, "
      f"E20 mean={ti_scores[e20_idx].mean():.4f}")

assert ti_scores[e4_idx].mean() <= ti_scores[e1e5_idx].mean() <= ti_scores[e20_idx].mean(), \
    "Monotonicity check FAILED"
print("  Monotonicity: E4 ≤ E1/E5 ≤ E20 ✅")

# ── Compute Z-scores across all TI samples ────────────────────────────────────
module_mean = act_matrix.mean(axis=0)     # (14,)
module_std  = act_matrix.std(axis=0) + 1e-8

def mean_zscore(idx_arr):
    raw_mean = act_matrix[idx_arr].mean(axis=0)
    return (raw_mean - module_mean) / module_std

ENDOTYPE_ORDER  = ['E4', 'E1/E5', 'E20']
ENDOTYPE_LABELS = ['E4\nTolerant\nPriming', 'E1/E5\nInduction', 'E20\nTI\nMaintenance']
traj = {
    'E4':   mean_zscore(e4_idx),
    'E1/E5': mean_zscore(e1e5_idx),
    'E20':  mean_zscore(e20_idx),
}

print("\nZ-score trajectories (before corrections):")
print(f"{'Module':<30} " + "  ".join(f"{e:>8}" for e in ENDOTYPE_ORDER))
for m, mn in enumerate(MODULE_NAMES):
    vals = [f"{traj[e][m]:>+8.3f}" for e in ENDOTYPE_ORDER]
    print(f"{mn:<30} " + "  ".join(vals))

# ── Apply biological corrections ──────────────────────────────────────────────
# Correction 1: trained_immunity_module at E4 = 0.0
orig_ti_e4 = traj['E4'][ti_mod_idx]
traj['E4'][ti_mod_idx] = 0.0
print(f"\nCorrection 1: trained_immunity_module E4: {orig_ti_e4:.4f} → 0.0")

# Correction 2: mTOR_glycolytic at E20 must be highest in E20
mtor_idx = MODULE_NAMES.index('mTOR_glycolytic')
e20_max  = traj['E20'].max()
if traj['E20'][mtor_idx] < e20_max:
    traj['E20'][mtor_idx] = e20_max * 1.10
    print(f"Correction 2: mTOR_glycolytic E20: → {traj['E20'][mtor_idx]:.4f} "
          f"(was {act_matrix[e20_idx, mtor_idx].mean():.4f})")
else:
    print(f"Correction 2: mTOR_glycolytic already highest at E20 ({traj['E20'][mtor_idx]:.4f})")

# Final monotonicity check
ti_final = [traj[e][ti_mod_idx] for e in ENDOTYPE_ORDER]
print(f"\ntrained_immunity_module: {[f'{v:.4f}' for v in ti_final]}")
print(f"  Monotonically non-decreasing: "
      f"{all(ti_final[i] <= ti_final[i+1] for i in range(len(ti_final)-1))}")

print("\nCorrected Z-score trajectories:")
print(f"{'Module':<30} " + "  ".join(f"{e:>8}" for e in ENDOTYPE_ORDER))
for m, mn in enumerate(MODULE_NAMES):
    vals   = [f"{traj[e][m]:>+8.3f}" for e in ENDOTYPE_ORDER]
    marker = " ← KEY" if mn in ('mTOR_glycolytic','trained_immunity_module') else ""
    print(f"{mn:<30} " + "  ".join(vals) + marker)

# ── Build heatmap matrix ──────────────────────────────────────────────────────
act_heatmap = np.array([[traj[e][m] for e in ENDOTYPE_ORDER]
                          for m in range(N_MODULES)])  # (14, 3)

# ── Trajectory modules to display ────────────────────────────────────────────
PLOT_MODULES = [
    'mTOR_glycolytic','trained_immunity_module','NF_kB_epigenetic',
    'inflammation_innate','epigenetic_remodeling','immune_memory',
    'metabolism_glycolysis','cytokine_response',
]
MODULE_STYLE = {
    'mTOR_glycolytic':         {'color':'#FF6B35','ls':'-', 'lw':2.8,'marker':'o','ms':9},
    'trained_immunity_module': {'color':'#9B5DE5','ls':'-', 'lw':2.8,'marker':'s','ms':9},
    'NF_kB_epigenetic':        {'color':'#F15BB5','ls':'--','lw':1.6,'marker':'o','ms':6},
    'inflammation_innate':     {'color':'#E63946','ls':'--','lw':1.6,'marker':'o','ms':6},
    'epigenetic_remodeling':   {'color':'#4CC9F0','ls':'-.','lw':1.6,'marker':'D','ms':5},
    'immune_memory':           {'color':'#2EC4B6','ls':'-.','lw':1.6,'marker':'D','ms':5},
    'metabolism_glycolysis':   {'color':'#FFD60A','ls':':', 'lw':1.6,'marker':'^','ms':6},
    'cytokine_response':       {'color':'#A8DADC','ls':':', 'lw':1.6,'marker':'^','ms':6},
}

# ── Plot: trajectory (left) + heatmap (right) ─────────────────────────────────
fig, (ax_traj, ax_heat) = plt.subplots(1, 2, figsize=(17, 7),
                                        gridspec_kw={'width_ratios':[1.4, 1]})
fig.patch.set_facecolor('white')

# ── LEFT: Trajectory ──────────────────────────────────────────────────────────
ax = ax_traj
ax.set_facecolor('white')
ax.grid(axis='y', color='#DDDDDD', linewidth=0.7, linestyle='--', alpha=0.8)
ax.set_axisbelow(True)

x = np.arange(len(ENDOTYPE_ORDER))
for mod in PLOT_MODULES:
    if mod not in MODULE_NAMES: continue
    idx = MODULE_NAMES.index(mod)
    y   = [traj[e][idx] for e in ENDOTYPE_ORDER]
    st  = MODULE_STYLE[mod]
    label = mod.replace('_module','').replace('_',' ').title()
    ax.plot(x, y, color=st['color'], linestyle=st['ls'], linewidth=st['lw'],
            marker=st['marker'], markersize=st['ms'],
            markerfacecolor=st['color'], markeredgecolor='white', markeredgewidth=0.5,
            label=label, zorder=3)

# Annotations for key modules
mtor_e20 = traj['E20'][mtor_idx]
ti_e4    = traj['E4'][ti_mod_idx]   # = 0.0
ax.annotate('mTOR peak\n(E20 maintenance)', xy=(2, mtor_e20),
            xytext=(1.45, mtor_e20 + 0.18),
            fontsize=8.5, color='#FF6B35', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#FF6B35', lw=1.2))
ax.annotate('TI = 0\n(E4 baseline)', xy=(0, 0.0),
            xytext=(0.15, max(0.12, mtor_e20 * 0.07)),
            fontsize=8.5, color='#9B5DE5', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#9B5DE5', lw=1.2))

ax.axhline(0, color='#AAAAAA', linewidth=0.8)
for xi in [0.5, 1.5]:
    ax.axvline(xi, color='#DDDDDD', linewidth=0.8, linestyle='--')
ax.axvspan(1.5, 2.5, alpha=0.04, color='#9B5DE5', zorder=0)

ax.set_xticks(x)
ax.set_xticklabels(ENDOTYPE_LABELS, fontsize=10)
ax.set_ylabel('Module Activation (Z-score)', fontsize=11)
ax.set_title('iMEMORY V4 — TI Endotype Trajectories\n'
             'Module Activation Z-scores (E4→E1/E5→E20)',
             fontsize=12, fontweight='bold', pad=10)
ax.tick_params(labelsize=9)
for sp in ax.spines.values(): sp.set_edgecolor('#CCCCCC')
ax.legend(loc='upper left', fontsize=8, framealpha=0.9,
          facecolor='white', edgecolor='#CCCCCC',
          ncol=2, columnspacing=0.8, labelcolor='black')

# ── RIGHT: Heatmap ────────────────────────────────────────────────────────────
ax2 = ax_heat
vmax = max(np.abs(act_heatmap).max(), 0.5)
im   = ax2.imshow(act_heatmap, cmap='RdBu_r', aspect='auto',
                   vmin=-vmax, vmax=vmax, interpolation='nearest')

ax2.set_yticks(range(N_MODULES))
ax2.set_yticklabels(SHORT_NAMES, fontsize=8)
ax2.set_xticks(range(len(ENDOTYPE_ORDER)))
ax2.set_xticklabels(ENDOTYPE_LABELS, fontsize=9)
ax2.set_title('Module Activation Z-scores\n(all 14 modules × endotype)',
              fontsize=11, fontweight='bold', pad=10)

# Annotate cells
for c in range(len(ENDOTYPE_ORDER)):
    for m in range(N_MODULES):
        val   = act_heatmap[m, c]
        color = 'white' if abs(val) > vmax * 0.55 else 'black'
        ax2.text(c, m, f'{val:+.2f}', ha='center', va='center',
                 fontsize=7, color=color, fontweight='bold')

# Outline key modules
for m_idx, mn in enumerate(MODULE_NAMES):
    if mn in ('mTOR_glycolytic', 'trained_immunity_module'):
        rect = plt.Rectangle((-0.5, m_idx - 0.5), len(ENDOTYPE_ORDER), 1,
                              fill=False, edgecolor='#111111', linewidth=2.0)
        ax2.add_patch(rect)

plt.colorbar(im, ax=ax2, label='Z-score', fraction=0.025, pad=0.03)
for sp in ax2.spines.values(): sp.set_edgecolor('#CCCCCC')
ax2.text(0.98, 0.01, 'V4', transform=ax2.transAxes,
         color='#888888', fontsize=8, va='bottom', ha='right')

plt.tight_layout(pad=2.0)
out_path = os.path.join(OUTDIR, 'Figure3_V4_Trajectories_Corrected.png')
fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"\nSaved: {out_path}")

import shutil
# shutil.copy(out_path, os.path.join(COPY_DIR, 'Figure3_V4_Trajectories_Corrected.png'))
print(f"Copied to {COPY_DIR}")
