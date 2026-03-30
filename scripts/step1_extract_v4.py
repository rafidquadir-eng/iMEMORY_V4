#!/usr/bin/env python3
"""
step1_extract_v4.py
Extract embeddings from V4 checkpoint for figure generation.
Uses clean samples only (filtered by endotype_assignments_v3_1_clean.csv).
"""
import os, sys, lmdb, pickle, numpy as np, torch, pandas as pd
from collections import defaultdict

sys.path.insert(0, "/mnt/data/trained_im/scripts")
from hetero_encoder_v4 import HeteroClassificationModelV3

CKPT      = "/mnt/nvme15tb/trained_im/checkpoints/phase2_v4/phase2_v4_best.pt"
GRAPH_DIR = "/mnt/nvme15tb/trained_im/graphs_lmdb_v4"
OUT_DIR   = "/mnt/nvme15tb/trained_im/outputs/V4/stage1/"
DEVICE    = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load clean sample IDs
ENDO_CLEAN_PATH = "/mnt/nvme15tb/trained_im/outputs/stage3_v3_1/endotype_assignments_v3_1_clean.csv"
_endo_clean = pd.read_csv(ENDO_CLEAN_PATH)
CLEAN_SAMPLE_IDS = set(_endo_clean["accession_id"].tolist())
print(f"Loaded {len(CLEAN_SAMPLE_IDS)} clean sample IDs")

# Sample counts per class
N_PER_CLASS = {
    0: 10_000,   # acute_rejection
    1: 10_000,   # chronic_rejection
    2: 15_000,   # CKD
    3: 3_740,    # hemodialysis
    4: 10_000,   # kidney_transplant
    5: 10_834,   # trained_immunity
}

CLASS_NAMES = {0:"acute_rejection", 1:"chronic_rejection", 2:"CKD",
               3:"hemodialysis", 4:"kidney_transplant", 5:"trained_immunity"}

os.makedirs(OUT_DIR, exist_ok=True)

print(f"Loading V4 checkpoint: {CKPT}")
ckpt  = torch.load(CKPT, map_location="cpu")
model = HeteroClassificationModelV3(hidden_dim=256, embedding_dim=128, num_classes=6)
model.load_state_dict(ckpt["model_state_dict"])
model = model.to(DEVICE).eval()
print(f"  Target total: {sum(N_PER_CLASS.values()):,} embeddings")

db_paths = sorted([os.path.join(GRAPH_DIR, d) for d in os.listdir(GRAPH_DIR)
                   if d.endswith(".lmdb") and os.path.isdir(os.path.join(GRAPH_DIR, d))])
print(f"Scanning {len(db_paths)} LMDB shards...")

graphs_by_class = defaultdict(list)
skipped_not_clean = 0
for db_path in db_paths:
    if all(len(graphs_by_class[c]) >= N_PER_CLASS[c] for c in range(6)):
        break
    try:
        env = lmdb.open(db_path, readonly=True, lock=False)
        with env.begin() as txn:
            for key, value in txn.cursor():
                key_str = key.decode("utf-8") if isinstance(key, bytes) else str(key)
                # Filter to clean samples only
                if key_str not in CLEAN_SAMPLE_IDS:
                    skipped_not_clean += 1
                    continue
                g   = pickle.loads(value)
                lbl = getattr(g, "y", None)
                if lbl is None: continue
                if hasattr(lbl, "item"): lbl = lbl.item()
                lbl = int(lbl)
                if not (0 <= lbl <= 5): continue
                if len(graphs_by_class[lbl]) >= N_PER_CLASS[lbl]: continue
                graphs_by_class[lbl].append((key_str, g))
        env.close()
    except Exception as e:
        print(f"Error reading {db_path}: {e}")

print(f"Skipped {skipped_not_clean} samples not in clean endotype")
print("Collected per class:")
for lbl in sorted(graphs_by_class):
    print(f"  {lbl} ({CLASS_NAMES[lbl]}): {len(graphs_by_class[lbl]):,} / {N_PER_CLASS[lbl]:,}")

offset  = 0
records = []
all_embeddings = []
with torch.no_grad():
    for label in sorted(graphs_by_class.keys()):
        samples = graphs_by_class[label]
        for pos, (key_str, g) in enumerate(samples):
            try:
                if not hasattr(g, "dataset_weight") or \
                   not isinstance(g.dataset_weight, torch.Tensor):
                    g.dataset_weight = torch.tensor([1.0])
                g = g.to(DEVICE)
                x_dict  = {nt: g[nt].x for nt in g.node_types if hasattr(g[nt], "x")}
                ei_dict = {et: g[et].edge_index for et in g.edge_types
                           if hasattr(g[et], "edge_index")}
                b_dict  = {nt: torch.zeros(g[nt].x.size(0), dtype=torch.long, device=DEVICE)
                           for nt in g.node_types if hasattr(g[nt], "x")}
                _, _, h, _, _ = model(x_dict, ei_dict, b_dict)
                emb = h.cpu().numpy().flatten()
                all_embeddings.append(emb)
                records.append({
                    "lmdb_key":           key_str,
                    "label":              label,
                    "class_name":         CLASS_NAMES[label],
                    "embedding_idx":      offset,
                })
                offset += 1
            except Exception as e:
                print(f"Error processing {key_str}: {e}")
            if (pos + 1) % 1000 == 0:
                print(f"  {CLASS_NAMES[label]}: {pos+1}/{len(samples)}")

# Save embeddings and metadata
embeddings = np.vstack(all_embeddings)
np.save(os.path.join(OUT_DIR, "embeddings_v4.npy"), embeddings)
df = pd.DataFrame(records)
df.to_csv(os.path.join(OUT_DIR, "metadata_v4.csv"), index=False)

print(f"\nSaved {len(records)} embeddings to {OUT_DIR}")
print(f"  embeddings_v4.npy: {embeddings.shape}")
print(f"  metadata_v4.csv: {len(df)} rows")
