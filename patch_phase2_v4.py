#!/usr/bin/env python3
"""Patch phase2_v4_train.py to filter samples to clean endotype CSV only."""

INPUT_FILE = "/mnt/data/trained_im/scripts/phase2_v4_train.py"
OUTPUT_FILE = "/mnt/data/trained_im/scripts/phase2_v4_train.py"

with open(INPUT_FILE, "r") as f:
    content = f.read()

# Find and replace the LMDBDataset class to add filtering
old_class = '''class LMDBDataset(torch.utils.data.Dataset):
    def __init__(self, graph_dir):
        self.samples      = []
        self.class_indices = defaultdict(list)

        db_paths = sorted([
            os.path.join(graph_dir, d)
            for d in os.listdir(graph_dir)
            if d.endswith('.lmdb') and os.path.isdir(os.path.join(graph_dir, d))
        ])
        print(f"Loading index from {len(db_paths)} LMDB shards...")

        for db_path in db_paths:
            try:
                env = lmdb.open(db_path, readonly=True, lock=False)
                with env.begin() as txn:
                    for key, value in txn.cursor():
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
        for c in range(N_CLASSES):
            print(f"  Class {c} ({CLASS_NAMES[c]}): {len(self.class_indices[c])}")'''

new_class = '''# Load clean endotype CSV to filter samples
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
            print(f"  Class {c} ({CLASS_NAMES[c]}): {len(self.class_indices[c])}")'''

if old_class in content:
    content = content.replace(old_class, new_class)
    print("Patched LMDBDataset to filter to clean endotype samples")
else:
    print("ERROR: Could not find LMDBDataset class to patch")

with open(OUTPUT_FILE, "w") as f:
    f.write(content)

print(f"Saved patched file to {OUTPUT_FILE}")
