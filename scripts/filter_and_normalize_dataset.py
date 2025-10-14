#!/usr/bin/env python3
"""
filter_and_normalize_swipes.py
---------------------------------
Hybrid FUTO + normalization filter for RNNT training.

Combines:
✅ FUTO metadata filtering (orientation, canvas sanity, valid dictionary)
✅ My coordinate normalization ([-1,1], overshoot clamp, motion filters)
✅ Robust word validation (NLTK + wordfreq top-N hybrid)
✅ Detailed logging and stats reporting

Usage (one-liner):
uv run python filter_and_normalize_swipes.py futo/train.jsonl filtered/futo_filtered_norm.jsonl
"""

import json
import re
import sys
import nltk
import numpy as np
from pathlib import Path
from tqdm import tqdm
from wordfreq import top_n_list, word_frequency
from collections import Counter

# ----------------- CONFIG -----------------
MIN_WORD_LEN, MAX_WORD_LEN = 2, 20
MIN_POINTS, MAX_POINTS = 8, 512
MIN_DURATION_MS, MAX_DURATION_MS = 40, 4000
MIN_SPEED = 0.002
MAX_OVERSHOOT = 0.15
OUT_CLAMP = 1.5
MAX_CANVAS_WIDTH = 900
MIN_WORD_FREQ = 3


# ----------------- WORD VALIDATION -----------------
def build_valid_word_set(max_words=400000):
    """Build hybrid set using NLTK + top wordfreq words."""
    try:
        from nltk.corpus import words

        valid_words = set(w.lower() for w in words.words())
    except LookupError:
        print("Downloading NLTK words corpus...")
        nltk.download("words", quiet=True)
        from nltk.corpus import words

        valid_words = set(w.lower() for w in words.words())

    wf_set = set()
    for w in top_n_list("en", max_words):
        base = w.lower().replace("'", "").replace("-", "")
        if re.fullmatch(r"[a-z]{2,20}", base):
            wf_set.add(base)
            wf_set.add(w.lower())
    print(f"Loaded {len(valid_words):,} NLTK words and {len(wf_set):,} wordfreq words.")
    return valid_words | wf_set


def canonicalize_word(word: str) -> str:
    """Lowercase and strip apostrophes for model vocabulary."""
    return word.lower().replace("'", "")


def is_valid_word(word: str, valid_set) -> bool:
    """Check if a word is valid (letters/apostrophes, in dictionary)."""
    if not re.match(r"^[a-zA-Z']+$", word):
        return False
    if len(word) < MIN_WORD_LEN or len(word) > MAX_WORD_LEN:
        return False
    clean = canonicalize_word(word)
    return clean in valid_set


# ----------------- FREQ COUNTER (NEW) -----------------
def build_frequency_map(input_path: Path) -> Counter:
    """Scan dataset once to count occurrences of canonical words."""
    freq = Counter()
    with input_path.open("r") as fin:
        for line in fin:
            try:
                sample = json.loads(line)
                word = canonicalize_word(sample.get("word", ""))
                if len(word) >= MIN_WORD_LEN:
                    freq[word] += 1
            except Exception:
                continue
    print(f"Counted {len(freq):,} unique words in dataset.")
    return freq


# ----------------- GESTURE NORMALIZATION -----------------
def normalize_points(points):
    if len(points) < MIN_POINTS or len(points) > MAX_POINTS:
        return None

    t = np.array([p["t"] for p in points], dtype=np.float64)
    x = np.array([p["x"] for p in points], dtype=np.float64)
    y = np.array([p["y"] for p in points], dtype=np.float64)

    order = np.argsort(t)
    t, x, y = t[order], x[order], y[order]
    _, uniq_idx = np.unique(t, return_index=True)
    t, x, y = t[uniq_idx], x[uniq_idx], y[uniq_idx]
    if len(t) < MIN_POINTS:
        return None

    t -= t[0]
    duration = t[-1]
    if duration < MIN_DURATION_MS or duration > MAX_DURATION_MS:
        return None

    dx, dy, dt = np.diff(x), np.diff(y), np.diff(t)
    dt[dt == 0] = 1e-3
    mean_speed = np.mean(np.sqrt(dx**2 + dy**2) / dt)
    if mean_speed < MIN_SPEED:
        return None

    x = np.clip(x, -MAX_OVERSHOOT, 1 + MAX_OVERSHOOT)
    y = np.clip(y, -MAX_OVERSHOOT, 1 + MAX_OVERSHOOT)
    x = (x * 2) - 1
    y = (y * 2) - 1
    x = np.clip(x, -OUT_CLAMP, OUT_CLAMP)
    y = np.clip(y, -OUT_CLAMP, OUT_CLAMP)

    return [
        {"t": float(tt), "x": float(xx), "y": float(yy)} for tt, xx, yy in zip(t, x, y)
    ]


# ----------------- MAIN FILTER -----------------
def filter_and_normalize(input_path: Path, output_path: Path, valid_set, freq_map):
    stats = {
        "total": 0,
        "kept": 0,
        "invalid_sentence": 0,
        "invalid_word": 0,
        "too_rare": 0,  # NEW
        "not_portrait": 0,
        "wrong_dimensions": 0,
        "width_too_large": 0,
        "trace_invalid": 0,
    }

    with input_path.open("r") as fin, output_path.open("w") as fout:
        for line in tqdm(fin, desc=f"Filtering {input_path.name}"):
            stats["total"] += 1
            try:
                sample = json.loads(line)
                if sample.get("potentially_invalid_sentence", False):
                    stats["invalid_sentence"] += 1
                    continue

                word = sample.get("word", "")
                canon = canonicalize_word(word)

                if not is_valid_word(word, valid_set):
                    stats["invalid_word"] += 1
                    continue

                # NEW: frequency filter
                if freq_map[canon] < MIN_WORD_FREQ:
                    stats["too_rare"] += 1
                    continue

                orientation = sample.get("orientation", "")
                if orientation != "portrait-primary":
                    stats["not_portrait"] += 1
                    continue

                w, h = sample.get("canvas_width", 0), sample.get("canvas_height", 0)
                if w <= h:
                    stats["wrong_dimensions"] += 1
                    continue
                if w > MAX_CANVAS_WIDTH:
                    stats["width_too_large"] += 1
                    continue

                norm_points = normalize_points(sample.get("data", []))
                if norm_points is None:
                    stats["trace_invalid"] += 1
                    continue

                out = {
                    "word": canonicalize_word(word),
                    "points": norm_points,
                    "id": sample.get("id"),
                    "session": sample.get("session"),
                    "timestamp": sample.get("timestamp"),
                }
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                stats["kept"] += 1

            except Exception as e:
                stats["trace_invalid"] += 1
                continue

    # Save log
    log_path = output_path.with_name(output_path.stem + "_stats.json")
    with log_path.open("w") as f:
        json.dump(stats, f, indent=2)
    print("\n=== STATS ===")
    for k, v in stats.items():
        print(f"{k:20s}: {v}")
    print(f"\nFiltered dataset saved to: {output_path}")
    print(f"Stats saved to: {log_path}")


# ----------------- ENTRY -----------------
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "Usage: python filter_and_normalize_swipes.py <input.jsonl> <output.jsonl>"
        )
        sys.exit(1)
    valid_set = build_valid_word_set()
    input_path = Path(sys.argv[1])
    freq_map = build_frequency_map(input_path)  # NEW PREPASS
    filter_and_normalize(input_path, Path(sys.argv[2]), valid_set, freq_map)
