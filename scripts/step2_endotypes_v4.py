#!/usr/bin/env python3
"""
step2_endotypes_v4.py
UMAP + HDBSCAN endotype clustering on V4 embeddings.
"""
import os, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler

EMB_DIR = "/mnt/nvme15tb/trained_im/outputs/V4/stage1/"
OUT_DIR = "/mnt/nvme15tb/trained_im/outputs/V4/stage2/"
os.makedirs(OUT_DIR, exist_ok=True)

emb  = np.load(os.path.join(EMB_DIR, "embeddings_v4.npy"))
meta = pd.read_csv(os.path.join(EMB_DIR, "metadata_v4.csv"))
print(f"Embeddings: {emb.shape} | Meta: {meta.shape}")

scaler = StandardScaler()
emb_sc = scaler.fit_transform(emb)

# UMAP
print("\nRunning UMAP (n_neighbors=50, min_dist=0.5)...")
import umap as umap_lib
umap_coords = umap_lib.UMAP(
    n_components=2, n_neighbors=50, min_dist=0.5,
    metric="euclidean", random_state=42, n_jobs=-1
).fit_transform(emb_sc)
print(f"  UMAP: {umap_coords.shape}")

# t-SNE
print("Running t-SNE (perplexity=40)...")
from sklearn.manifold import TSNE
tsne_coords = TSNE(
    n_components=2, perplexity=40, max_iter=2000,
    learning_rate="auto", random_state=42, init="pca", n_jobs=-1
).fit_transform(emb_sc)
print(f"  t-SNE: {tsne_coords.shape}")

# HDBSCAN
print("Running HDBSCAN (min_cluster_size=10, min_samples=5)...")
import hdbscan
clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5,
                             metric="euclidean", prediction_data=True)
labels = clusterer.fit_predict(emb_sc)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise    = (labels == -1).sum()
print(f"  Clusters: {n_clusters}, Noise: {n_noise}")

# Assign noise to nearest centroid
if n_noise > 0:
    cluster_ids = [c for c in set(labels) if c != -1]
    centroids   = np.array([emb_sc[labels==c].mean(axis=0) for c in cluster_ids])
    for idx in np.where(labels == -1)[0]:
        labels[idx] = cluster_ids[
            np.linalg.norm(centroids - emb_sc[idx], axis=1).argmin()]
    print(f"  Noise reassigned. Remaining: {(labels==-1).sum()}")

out = pd.DataFrame({
    "sample_idx":        range(len(meta)),
    "accession_id":      meta["lmdb_key"].values,
    "original_endotype": labels,
    "refined_endotype":  labels,
    "disease_class":     meta["label"].values,
    "umap_x":            umap_coords[:, 0],
    "umap_y":            umap_coords[:, 1],
    "tsne_x":            tsne_coords[:, 0],
    "tsne_y":            tsne_coords[:, 1],
    "label_changed":     False,
})
path = os.path.join(OUT_DIR, "endotype_assignments_v4.csv")
out.to_csv(path, index=False)
print(f"\nSaved: {path}")

# Also save UMAP coords separately for figure generation
np.save(os.path.join(OUT_DIR, "umap_coords_v4.npy"), umap_coords)
np.save(os.path.join(OUT_DIR, "tsne_coords_v4.npy"), tsne_coords)

CLASS_NAMES = {0:"acute_rej", 1:"chronic_rej", 2:"CKD", 3:"HD", 4:"KTx", 5:"TI"}

print("\n" + "="*60)
print("PANEL SAMPLE COUNTS:")
print("="*60)
DISPLAY_MAP = {0:"KTx", 1:"KTx", 2:"CKD", 3:"HD", 4:"KTx"}
out["display_group"] = out["disease_class"].astype(int).map(DISPLAY_MAP)
for g in ["CKD","HD","KTx"]:
    n = (out["display_group"] == g).sum()
    print(f"  {g}: {n}")
print(f"  TI (class 5): {(out[disease_class]==5).sum()}")

print("\n" + "="*60)
print("TOP HDBSCAN CLUSTERS PER DISEASE CLASS")
print("="*60)
for dc in range(6):
    top = out[out["disease_class"]==dc]["refined_endotype"].value_counts().head(8)
    print(f"\nClass {dc} ({CLASS_NAMES[dc]}):")
    for cid, cnt in top.items():
        total_in_cluster = (out["refined_endotype"]==cid).sum()
        frac_dc = cnt / total_in_cluster
        print(f"  Cluster {cid:4d}: {cnt:5d} samples  "
              f"({frac_dc:.0%} of cluster {cid} total {total_in_cluster})")

print("\n" + "="*60)
print("AUTO-SUGGESTED TI CLUSTER ASSIGNMENTS")
print("="*60)
clean_ti, mixed_ti = [], []
e4_candidates, e1e5_candidates, e20_candidates, pan_candidates = [], [], [], []

for cid in sorted(set(labels)):
    sub     = out[out["refined_endotype"] == cid]
    n_total = len(sub)
    if n_total < 5: continue
    n_ti   = (sub["disease_class"] == 5).sum()
    frac   = n_ti / n_total
    if frac > 0.80:
        clean_ti.append(cid)
        n_ktx  = sub["disease_class"].isin([4]).sum()
        n_rej  = sub["disease_class"].isin([0, 1]).sum()
        n_pan  = sub["disease_class"].isin([0,1,2,3,4]).sum()
        if n_ktx / n_total > 0.10:
            e4_candidates.append(cid)
        elif n_rej / n_total > 0.10:
            e1e5_candidates.append(cid)
        elif n_pan / n_total < 0.05:
            e20_candidates.append(cid)
        else:
            e20_candidates.append(cid)
    elif frac > 0.30:
        mixed_ti.append(cid)
        n_all_disease = sub["disease_class"].isin([0,1,2,3,4]).sum()
        if n_all_disease / n_total > 0.20:
            pan_candidates.append(cid)

print(f"\nCLEAN_TI   = {set(clean_ti)}")
print(f"MIXED_TI   = {set(mixed_ti)}")
print(f"E4_CLUSTERS   = {set(e4_candidates)}")
print(f"E1E5_CLUSTERS = {set(e1e5_candidates)}")
print(f"E20_CLUSTERS  = {set(e20_candidates)}")
print(f"PAN_CLUSTERS  = {set(pan_candidates)}")
