#!/usr/bin/env python3
"""
fix_v4_ti_phase.py - Add ti_phase column to V4 endotype CSV
Uses V4-specific cluster sets derived from HDBSCAN analysis.
"""
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

ENDO_V4 = "/mnt/nvme15tb/trained_im/outputs/V4/stage3/endotype_assignments_v4_final.csv"

# V4 TI CLUSTER SETS (derived from V4 HDBSCAN analysis)
CLEAN_TI_V4 = {5, 8, 14, 34, 60, 61, 66, 69, 86, 98, 101, 107, 154, 169, 192, 194, 205, 211, 213, 216, 221, 223, 224, 226, 227, 228, 231, 232, 234, 235, 236, 237, 238, 240, 244, 245, 247, 250, 253, 254, 255, 258, 263, 264, 265, 266, 269, 270, 271, 274, 275, 279, 280, 281, 282, 284, 285, 294, 301, 311, 312, 313, 315, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395}
MIXED_TI_V4 = {257, 17, 162, 40, 41, 43, 51, 181, 187, 189, 191, 323, 206, 207, 208, 209, 345, 220, 93, 222, 99, 230, 233, 115, 246}
E4_CLUSTERS_V4 = {107}
E1E5_CLUSTERS_V4 = {353, 194, 34, 294, 351, 265, 244, 347, 60, 223}
E20_CLUSTERS_V4 = CLEAN_TI_V4 - E4_CLUSTERS_V4 - E1E5_CLUSTERS_V4

print("Loading V4 endotype CSV...")
endo = pd.read_csv(ENDO_V4)
print(f"  {len(endo)} samples")

# Build TI k-NN tree for phase assignment
ti_mask = endo["disease_class"] == 5
ti_df = endo[ti_mask].copy()
ti_coords = ti_df[["umap_x", "umap_y"]].values
ti_cids = ti_df["refined_endotype"].values.astype(int)
ti_tree = cKDTree(ti_coords)
print(f"  TI k-NN tree: {len(ti_df)} TI samples")

# Count TI samples in each phase cluster set
n_e4 = sum(1 for c in ti_cids if c in E4_CLUSTERS_V4)
n_e1e5 = sum(1 for c in ti_cids if c in E1E5_CLUSTERS_V4)
n_e20 = sum(1 for c in ti_cids if c in E20_CLUSTERS_V4)
print(f"  TI samples in E4: {n_e4}")
print(f"  TI samples in E1E5: {n_e1e5}")
print(f"  TI samples in E20: {n_e20}")

def assign_ti_phase(row, k=50):
    """Assign TI phase via k-NN majority vote over TI neighbor cluster membership."""
    px, py = row["umap_x"], row["umap_y"]
    _, idx = ti_tree.query([px, py], k=min(k, len(ti_coords)))
    nbr = ti_cids[idx]
    counts = {
        "E4 Priming": sum(c in E4_CLUSTERS_V4 for c in nbr),
        "E1/E5 Induction": sum(c in E1E5_CLUSTERS_V4 for c in nbr),
        "E20 Maintenance": sum(c in E20_CLUSTERS_V4 for c in nbr),
    }
    best = max(counts, key=counts.get)
    return best if counts[best] >= 1 else "No TI signal"

print("\nAssigning TI phases...")
endo["ti_phase"] = endo.apply(assign_ti_phase, axis=1)

# Print distribution
print("\nTI phase distribution:")
print(endo["ti_phase"].value_counts())

# Save
endo.to_csv(ENDO_V4, index=False)
print(f"\nSaved V4 endotype CSV with ti_phase column: {ENDO_V4}")
