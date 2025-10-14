#!/usr/bin/env python3
"""
prepare_training_manifests.py
---------------------------------
Merges, validates, and splits filtered swipe datasets
into train/val manifests compatible with train_rnnt_personalized.py.

Usage (one-liner):
uv run python prepare_training_manifests.py filtered/*.jsonl data/train_final_train.jsonl data/train_final_val.jsonl
"""

import json, sys, random
from pathlib import Path
from tqdm import tqdm


def load_jsonl(path: Path):
    """Load all valid JSON lines from a file."""
    samples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if isinstance(obj, dict) and "word" in obj and "points" in obj:
                    samples.append(obj)
            except Exception:
                continue
    return samples


def main():
    if len(sys.argv) < 4:
        print(
            "Usage: python prepare_training_manifests.py <input1.jsonl> [<input2.jsonl> ...] <train_out.jsonl> <val_out.jsonl>"
        )
        sys.exit(1)

    *input_files, train_out, val_out = map(Path, sys.argv[1:])
    all_samples = []

    for path in input_files:
        if not path.exists():
            print(f"⚠️ Skipping missing file: {path}")
            continue
        data = load_jsonl(path)
        print(f"Loaded {len(data):,} samples from {path.name}")
        all_samples.extend(data)

    random.shuffle(all_samples)
    total = len(all_samples)
    if total == 0:
        print("❌ No valid samples found.")
        sys.exit(1)

    val_ratio = 0.1 if total > 1000 else 0.2
    split_idx = int(total * (1 - val_ratio))
    train_set = all_samples[:split_idx]
    val_set = all_samples[split_idx:]

    Path(train_out).parent.mkdir(parents=True, exist_ok=True)
    with open(train_out, "w", encoding="utf-8") as f:
        for s in train_set:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    with open(val_out, "w", encoding="utf-8") as f:
        for s in val_set:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\n✅ Prepared manifests:")
    print(f"  Train: {len(train_set):,} → {train_out}")
    print(f"  Val:   {len(val_set):,} → {val_out}")
    print(f"  Total merged: {total:,}")


if __name__ == "__main__":
    main()
