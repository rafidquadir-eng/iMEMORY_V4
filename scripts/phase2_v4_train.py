#!/usr/bin/env python3
"""
CytoSF V3 Training — phase2_v3_train.py

Adapts phase2_balanced_retrain.py (V2) for V3:
  1. Uses HeteroClassificationModelV3 (V2 + pathway supervision heads)
  2. Adds PathwayConsistencyLoss to every training step
  3. Warm-starts from V2 best checkpoint (encoder + classifier transfer,
     only pathway_head is randomly initialised)
  4. Points to graphs_lmdb_v4 directory
  5. Identical balanced sampler, val strategy, early stopping, gate check

Training loss each step:
  total_loss = cross_entropy(logits, y, class_weights)
             + PathwayConsistencyLoss(module_reps, gene_reps, pathway_gene_masks)

pathway_gene_masks is read directly from the graph object (g.pathway_gene_masks).
If absent (e.g., a resumed graph from a partially built shard), pathway loss
term is silently skipped for that sample.

Key hyperparameters vs V2:
  - LAMBDA_ALIGN = 0.10   (pathway alignment loss weight)
  - LAMBDA_SEP   = 0.05   (module separation loss weight)
  - LR = 3e-5             (slightly lower — warm-starting from V2)
  - FREEZE_EPOCHS = 3     (shorter freeze since encoder already trained)

Save as: /mnt/data/trained_im/scripts/phase2_v3_train.py
Run as:
  python3 /mnt/data/trained_im/scripts/phase2_v3_train.py \
      2>&1 | tee /mnt/nvme15tb/trained_im/logs/phase2_v3_train.log
"""
import os, sys, random, json, csv
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, classification_report, silhouette_score
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import lmdb, pickle

sys.path.insert(0, '/mnt/data/trained_im/scripts')
from hetero_encoder_v3 import (
    HeteroClassificationModelV3,
    PathwayConsistencyLoss,
    MODULE_PATHWAY_NAMES,
    N_MODULES,
)

# ── Config ─────────────────────────────────────────────────────────────────────
V2_CKPT   = '/mnt/nvme15tb/trained_im/checkpoints/phase2_v3/phase2_v3_best.pt'
OUT_DIR   = '/mnt/nvme15tb/trained_im/checkpoints/phase2_v4/'
GRAPH_DIR = '/mnt/nvme15tb/trained_im/graphs_lmdb_v4'
LOG_DIR   = '/mnt/nvme15tb/trained_im/logs/'

N_CLASSES          = 6
N_PER_CLASS_TRAIN  = 500   # balanced samples per class per epoch
N_PER_CLASS_VAL    = 200   # balanced validation samples per class (fixed)
TI_CLASS_WEIGHT    = 3.0
LR                 = 1e-5  # lower than V2 (5e-5) — warm-starting from trained ckpt
N_EPOCHS           = 30    # extra headroom for pathway loss to converge
PATIENCE           = 10
FREEZE_EPOCHS      = 0     # shorter freeze — V2 encoder already trained
LAMBDA_ALIGN       = 0.10  # PathwayConsistencyLoss alignment weight
LAMBDA_SEP         = 0.05  # PathwayConsistencyLoss separation weight

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.makedirs(OUT_DIR,  exist_ok=True)
os.makedirs(LOG_DIR,  exist_ok=True)

CLASS_NAMES   = ['acute_rej', 'chronic_rej', 'CKD', 'HD', 'KTx', 'TI']
CLASS_WEIGHTS = torch.ones(N_CLASSES)
CLASS_WEIGHTS[5] = TI_CLASS_WEIGHT  # trained_immunity
CLASS_WEIGHTS[0] = 3.0              # acute_rejection
CLASS_WEIGHTS[3] = 2.5              # hemodialysis
CLASS_WEIGHTS = CLASS_WEIGHTS.to(DEVICE)

print(f"Device:        {DEVICE}")
print(f"Graph dir:     {GRAPH_DIR}")
print(f"V2 ckpt:       {V2_CKPT}")
print(f"Class weights: {CLASS_WEIGHTS.cpu().tolist()}")
print(f"Lambda align:  {LAMBDA_ALIGN}")
print(f"Lambda sep:    {LAMBDA_SEP}")

# ── Dataset ────────────────────────────────────────────────────────────────────
# Load clean endotype CSV to filter samples
ENDO_CLEAN_PATH = '/mnt/nvme15tb/trained_im/outputs/stage3_v3_1/endotype_assignments_v3_1_clean.csv'
import pandas as pd
_endo_clean = pd.read_csv(ENDO_CLEAN_PATH)
CLEAN_SAMPLE_IDS = set(_endo_clean['accession_id'].tolist())
print(f"Loaded {len(CLEAN_SAMPLE_IDS)} clean sample IDs from endotype CSV")

class LMDBDataset(torch.utils.data.Dataset):
    def __init__(self, graph_dir):
        self.samples      = []
        self.class_indices = defaultdict(list)

        db_paths = sorted([
            os.path.join(graph_dir, d)
            for d in os.listdir(graph_dir)
            if d.endswith('.lmdb') and os.path.isdir(os.path.join(graph_dir, d))
        ])
        print(f"Loading index from {len(db_paths)} LMDB shards...")
        print(f"Filtering to {len(CLEAN_SAMPLE_IDS)} clean samples from endotype CSV...")

        skipped_not_clean = 0
        for db_path in db_paths:
            try:
                env = lmdb.open(db_path, readonly=True, lock=False)
                with env.begin() as txn:
                    for key, value in txn.cursor():
                        key_str = key.decode('utf-8')
                        # Filter: only include samples in clean endotype CSV
                        if key_str not in CLEAN_SAMPLE_IDS:
                            skipped_not_clean += 1
                            continue
                        g = pickle.loads(value)
                        label = getattr(g, 'y', None)
                        if label is None:
                            continue
                        if hasattr(label, 'item'):
                            label = label.item()
                        if not (0 <= int(label) <= 5):
                            continue
                        idx = len(self.samples)
                        self.samples.append((db_path, bytes(key), int(label)))
                        self.class_indices[int(label)].append(idx)
                env.close()
            except Exception:
                pass

        print(f"Total samples indexed: {len(self.samples)}")
        print(f"Skipped (not in clean endotype): {skipped_not_clean}")
        for c in range(N_CLASSES):
            print(f"  Class {c} ({CLASS_NAMES[c]}): {len(self.class_indices[c])}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        db_path, key_bytes, _ = self.samples[idx]
        env = lmdb.open(db_path, readonly=True, lock=False)
        with env.begin() as txn:
            value = txn.get(key_bytes)
        env.close()
        g = pickle.loads(value)
        if not hasattr(g, 'dataset_weight') or not isinstance(g.dataset_weight, torch.Tensor):
            g.dataset_weight = torch.tensor([1.0])
        return g


dataset = LMDBDataset(GRAPH_DIR)

# ── Fixed balanced validation set ─────────────────────────────────────────────
random.seed(42)
val_indices = []
for cls in range(N_CLASSES):
    avail = dataset.class_indices[cls]
    if not avail:
        print(f"WARNING: Class {cls} ({CLASS_NAMES[cls]}) has 0 samples")
        continue
    drawn = random.sample(avail, min(N_PER_CLASS_VAL, len(avail)))
    if len(drawn) < N_PER_CLASS_VAL:
        drawn += random.choices(avail, k=N_PER_CLASS_VAL - len(drawn))
    val_indices.extend(drawn)

val_set_indices = set(val_indices)
for cls in range(N_CLASSES):
    dataset.class_indices[cls] = [
        i for i in dataset.class_indices[cls]
        if i not in val_set_indices
    ]
print(f"\nFixed balanced val set: {len(val_indices)} samples")
print("Val indices removed from train pool.")

# ── Balanced train sampler ─────────────────────────────────────────────────────
class BalancedSampler(torch.utils.data.Sampler):
    def __init__(self, class_indices, n_per_class):
        self.class_indices = class_indices
        self.n_per_class   = n_per_class

    def __iter__(self):
        indices = []
        for cls in range(N_CLASSES):
            avail = self.class_indices[cls]
            if not avail:
                continue
            drawn = random.sample(avail, self.n_per_class) \
                if len(avail) >= self.n_per_class \
                else random.choices(avail, k=self.n_per_class)
            indices.extend(drawn)
        random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return self.n_per_class * N_CLASSES


train_sampler = BalancedSampler(dataset.class_indices, N_PER_CLASS_TRAIN)

# ── Model ──────────────────────────────────────────────────────────────────────
model = HeteroClassificationModelV3(hidden_dim=256, embedding_dim=128, num_classes=N_CLASSES)

# Warm-start from V2 checkpoint
if os.path.exists(V2_CKPT):
    v2_state = torch.load(V2_CKPT, map_location='cpu')
    # V2 checkpoint may be wrapped in 'model_state_dict' key
    if 'model_state_dict' in v2_state:
        v2_state = v2_state['model_state_dict']
    missing, unexpected = model.load_state_dict(v2_state, strict=False)
    print(f"\nV2 warm-start loaded.")
    print(f"  Missing (new V3 keys): {missing}")
    print(f"  Unexpected (V2 only):  {unexpected}")
    # Expected: missing = ['pathway_head.net.0.weight', 'pathway_head.net.0.bias', ...]
    #           unexpected = [] or small list of V2-only keys
else:
    print(f"\nWARNING: V2 checkpoint not found at {V2_CKPT}")
    print("  Training from random initialisation.")

model = model.to(DEVICE)

# Pathway loss
pw_loss_fn = PathwayConsistencyLoss(lambda_align=LAMBDA_ALIGN, lambda_sep=LAMBDA_SEP)

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=LR * 0.01)

# ── Helper: single-graph forward pass ─────────────────────────────────────────
def run_batch(g):
    """Single-graph forward. Returns (logits, mtor_pred, embeddings, module_reps, gene_reps)."""
    g = g.to(DEVICE)
    x_dict  = {nt: g[nt].x for nt in g.node_types if hasattr(g[nt], 'x') and g[nt].x is not None}
    ei_dict = {et: g[et].edge_index for et in g.edge_types
               if hasattr(g[et], 'edge_index') and g[et].edge_index is not None}
    b_dict  = {nt: torch.zeros(g[nt].x.size(0), dtype=torch.long, device=DEVICE)
               for nt in g.node_types if hasattr(g[nt], 'x') and g[nt].x is not None}
    return model(x_dict, ei_dict, b_dict)


# ── Evaluation on balanced val set ────────────────────────────────────────────
def evaluate_balanced(indices):
    model.eval()
    preds, labels, embeddings = [], [], []
    with torch.no_grad():
        for idx in indices:
            label = dataset.samples[idx][2]
            try:
                item    = dataset[idx]
                logits, _, emb, _, _ = run_batch(item)
                preds.append(logits.argmax(1).item())
                labels.append(label)
                embeddings.append(emb.cpu().numpy().flatten())
            except Exception:
                pass

    y_true = np.array(labels)
    y_pred = np.array(preds)
    embs   = np.array(embeddings) if embeddings else np.zeros((1, 128))

    macro_f1 = f1_score(y_true, y_pred, average='macro',  zero_division=0)
    ti_f1    = f1_score(y_true, y_pred, labels=[5], average='macro', zero_division=0)

    silhouette = 0.0
    if len(embs) > 1 and len(np.unique(y_true)) > 1:
        try:
            silhouette = silhouette_score(StandardScaler().fit_transform(embs), y_true)
        except Exception:
            pass

    return macro_f1, ti_f1, silhouette, y_true, y_pred


# ── Training loop ──────────────────────────────────────────────────────────────
best_macro_f1 = 0.0
best_ti_f1    = 0.0
patience_count = 0
history        = []

print(f"\nStarting Phase 2 V3 training")
print(f"  Train: {N_PER_CLASS_TRAIN} × {N_CLASSES} = {N_PER_CLASS_TRAIN * N_CLASSES} per epoch")
print(f"  Val:   {N_PER_CLASS_VAL}   × {N_CLASSES} = {N_PER_CLASS_VAL * N_CLASSES} fixed")
print(f"  TI class weight: {TI_CLASS_WEIGHT}")
print(f"  Encoder frozen for first {FREEZE_EPOCHS} epochs\n")

# Freeze encoder initially (pathway_head stays trainable)
for param in model.encoder.parameters():
    param.requires_grad_(False)

for epoch in range(1, N_EPOCHS + 1):

    # Unfreeze encoder after FREEZE_EPOCHS
    if epoch == FREEZE_EPOCHS + 1:
        for param in model.encoder.parameters():
            param.requires_grad_(True)
        print(f"Epoch {epoch}: Encoder unfrozen.")

    # ── Train ──────────────────────────────────────────────────────────────────
    model.train()
    train_indices = list(train_sampler)

    epoch_loss    = 0.0
    epoch_cls_loss = 0.0
    epoch_pw_loss  = 0.0
    epoch_correct = 0
    epoch_total   = 0

    for step, idx in enumerate(train_indices):
        try:
            item      = dataset[idx]
            label_val = dataset.samples[idx][2]
            y = torch.tensor([label_val], dtype=torch.long, device=DEVICE)

            optimizer.zero_grad()
            logits, _, emb, module_reps, gene_reps = run_batch(item)

            # Classification loss
            cls_loss = F.cross_entropy(logits, y, weight=CLASS_WEIGHTS)

            # Pathway consistency loss (only when gene nodes present)
            pw_val = torch.tensor(0.0, device=DEVICE)
            if (module_reps is not None and gene_reps is not None
                    and gene_reps.numel() > 0
                    and hasattr(item, 'pathway_gene_masks')):
                pw_val, _ = pw_loss_fn(
                    module_reps,
                    gene_reps,
                    item.pathway_gene_masks,
                )

            total_loss = cls_loss + pw_val
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss     += total_loss.item()
            epoch_cls_loss += cls_loss.item()
            epoch_pw_loss  += pw_val.item()
            epoch_correct  += (logits.argmax(1) == y).sum().item()
            epoch_total    += 1

        except Exception as e:
            continue

    scheduler.step()
    avg_loss     = epoch_loss     / max(epoch_total, 1)
    avg_cls_loss = epoch_cls_loss / max(epoch_total, 1)
    avg_pw_loss  = epoch_pw_loss  / max(epoch_total, 1)
    acc          = epoch_correct  / max(epoch_total, 1)

    # ── Validate ───────────────────────────────────────────────────────────────
    macro_f1, ti_f1, silhouette, y_true, y_pred = evaluate_balanced(val_indices)

    improved = macro_f1 > best_macro_f1
    status   = ''
    if improved:
        best_macro_f1  = macro_f1
        best_ti_f1     = ti_f1
        patience_count = 0
        ckpt_path      = os.path.join(OUT_DIR, 'phase2_v4_best.pt')
        torch.save({
            'epoch':              epoch,
            'model_state_dict':   model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_macro_f1':       macro_f1,
            'val_ti_f1':          ti_f1,
            'val_silhouette':     silhouette,
            'module_pathway_names': MODULE_PATHWAY_NAMES,
        }, ckpt_path)
        status = '* SAVED'
    else:
        patience_count += 1

    log_line = (
        f"Epoch {epoch:3d}/{N_EPOCHS} | "
        f"loss={avg_loss:.4f} (cls={avg_cls_loss:.4f} pw={avg_pw_loss:.4f}) | "
        f"acc={acc:.3f} | macro_f1={macro_f1:.4f} | ti_f1={ti_f1:.4f} | "
        f"sil={silhouette:.4f} | patience={patience_count}/{PATIENCE} {status}"
    )
    print(log_line)

    history.append({
        'epoch': epoch, 'total_loss': avg_loss, 'cls_loss': avg_cls_loss,
        'pw_loss': avg_pw_loss, 'acc': acc,
        'macro_f1': macro_f1, 'ti_f1': ti_f1, 'silhouette': silhouette,
    })

    # Per-class report every 5 epochs
    if epoch % 5 == 0:
        print(classification_report(y_true, y_pred,
                                     target_names=CLASS_NAMES,
                                     digits=4, zero_division=0))

    # Early stopping
    if patience_count >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch} (patience={PATIENCE})")
        break

    # Gate check at epoch 10: pathway loss should be converging
    if epoch == 10:
        print(f"\n── GATE CHECK EPOCH 10 ──")
        if ti_f1 < 0.30:
            print(f"!! TI F1 = {ti_f1:.4f} < 0.30. Consider TI_CLASS_WEIGHT=8.0.")
        else:
            print(f">> TI F1 = {ti_f1:.4f} ≥ 0.30  ✅")
        if avg_pw_loss < 0.001:
            print(f"!! pw_loss = {avg_pw_loss:.5f} very small — "
                  f"pathway_gene_masks may be absent on most samples.")
        else:
            print(f">> pw_loss = {avg_pw_loss:.5f}  ✅ (pathway supervision active)")

    # Gate check at epoch 20: V3 should be matching or beating V2 baseline
    if epoch == 20:
        print(f"\n── GATE CHECK EPOCH 20 ──")
        V2_BASELINE_MACRO = 0.642
        V2_BASELINE_TI    = 0.710
        if macro_f1 >= V2_BASELINE_MACRO * 0.95:
            print(f">> macro_f1={macro_f1:.4f} ≥ 95% of V2 baseline ({V2_BASELINE_MACRO})  ✅")
        else:
            print(f"!! macro_f1={macro_f1:.4f} below 95% of V2 baseline. "
                  f"Check graph schema or reduce lambda_align/sep.")
        if ti_f1 >= V2_BASELINE_TI * 0.90:
            print(f">> ti_f1={ti_f1:.4f} ≥ 90% of V2 baseline ({V2_BASELINE_TI})  ✅")
        else:
            print(f"!! ti_f1={ti_f1:.4f} below 90% of V2 baseline.")

# ── Save training history ──────────────────────────────────────────────────────
csv_path = os.path.join(OUT_DIR, 'training_history_v4.csv')
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(
        f, fieldnames=['epoch','total_loss','cls_loss','pw_loss',
                       'acc','macro_f1','ti_f1','silhouette'])
    writer.writeheader()
    writer.writerows(history)

print(f"\n{'='*60}")
print(f"PHASE 2 V3 TRAINING COMPLETE")
print(f"{'='*60}")
print(f"Best macro F1:  {best_macro_f1:.4f}  (V2 baseline: 0.642)")
print(f"Best TI F1:     {best_ti_f1:.4f}  (V2 baseline: 0.710)")
print(f"Best ckpt:      {os.path.join(OUT_DIR, 'phase2_v4_best.pt')}")
print(f"History CSV:    {csv_path}")
print(f"\nModule pathway names (for activation extraction):")
for i, name in enumerate(MODULE_PATHWAY_NAMES):
    print(f"  Module {i:2d}: {name}")

# Decision guidance
print(f"\nDECISION:")
V2_MACRO = 0.642
V2_TI    = 0.710
if best_macro_f1 >= V2_MACRO and best_ti_f1 >= V2_TI:
    print(">> V3 PASSES: Both macro F1 and TI F1 match or beat V2.")
    print("   Proceed to module activation extraction.")
elif best_macro_f1 >= V2_MACRO * 0.95 and best_ti_f1 >= V2_TI * 0.90:
    print(">> V3 ACCEPTABLE: Within 5%/10% of V2 baselines.")
    print("   Proceed — the pathway supervision signal is now present.")
elif best_ti_f1 < 0.50:
    print("!! TI F1 critically low. Check:")
    print("   1. pathway_gene_masks present in graphs (gate_check.txt)")
    print("   2. LMDBDataset loaded correct GRAPH_DIR")
    print("   3. NODE_INPUT_DIMS matches actual graph node feature dims")
else:
    print("!! V3 below V2 baselines. Review pw_loss curve for divergence.")
