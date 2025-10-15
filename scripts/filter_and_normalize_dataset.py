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
MIN_SPEED = 0.001
MAX_OVERSHOOT = 0.15
OUT_CLAMP = 1.5
MAX_CANVAS_WIDTH = 900
MIN_WORD_FREQ = 3
MAX_SPEED = 0.01


# ----------------- WORD VALIDATION -----------------
def build_valid_word_set(max_words=400000):
    """Build hybrid set using NLTK + top wordfreq words."""
    try:
        from nltk.corpus import words

        valid_words = set(canonicalize_word(w) for w in words.words())
    except LookupError:
        print("Downloading NLTK words corpus...")
        nltk.download("words", quiet=True)
        from nltk.corpus import words

        valid_words = set(canonicalize_word(w) for w in words.words())

    wf_set = set()
    for w in top_n_list("en", max_words):
        base = canonicalize_word(w)
        if re.fullmatch(r"[a-z]{2,20}", base):
            wf_set.add(base)
    print(f"Loaded {len(valid_words):,} NLTK words and {len(wf_set):,} wordfreq words.")
    return valid_words | wf_set


def canonicalize_word(word: str) -> str:
    """Lowercase and strip apostrophes for model vocabulary."""
    return (
        word.lower()
        .replace("'", "")
        .replace(".", "")
        .replace(",", "")
        .replace(";", "")
        .replace(":", "")
        .replace("!", "")
        .replace("?", "")
        .replace("(", "")
        .replace(")", "")
        # .replace("[", "")
        # .replace("]", "")
        # .replace("{", "")
        # .replace("}", "")
        # .replace(" ", "")
    )


def is_valid_word(word: str, valid_set) -> bool:
    """Check if a word is valid (letters/apostrophes, in dictionary)."""
    if not re.match(r"^[a-zA-Z']+$", word):
        stats["invalid_word"] += 1
        return False
    if len(word) < MIN_WORD_LEN or len(word) > MAX_WORD_LEN:
        stats["invalid_length"] += 1
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
    if len(points) < MIN_POINTS:
        stats["trace_too_short"] += 1
        return None

    if len(points) > MAX_POINTS:
        stats["trace_too_long"] += 1
        return None

    t = np.array([p["t"] for p in points], dtype=np.float64)
    x = np.array([p["x"] for p in points], dtype=np.float64)
    y = np.array([p["y"] for p in points], dtype=np.float64)

    order = np.argsort(t)
    t, x, y = t[order], x[order], y[order]
    _, uniq_idx = np.unique(t, return_index=True)
    t, x, y = t[uniq_idx], x[uniq_idx], y[uniq_idx]
    if len(t) < MIN_POINTS:
        stats["trace_too_short"] += 1
        return None

    t -= t[0]
    duration = t[-1]
    if duration < MIN_DURATION_MS:
        stats["too_short_duration"] += 1
        return None
    if duration > MAX_DURATION_MS:
        stats["too_long_duration"] += 1
        return None

    dx, dy, dt = np.diff(x), np.diff(y), np.diff(t)
    dt[dt == 0] = 1e-3
    mean_speed = np.mean(np.sqrt(dx**2 + dy**2) / dt)
    if mean_speed < MIN_SPEED:
        stats["too_slow_speed"] += 1
        return None
    if mean_speed > MAX_SPEED:
        stats["too_fast_speed"] += 1
        return None

    # x = np.clip(x, -MAX_OVERSHOOT, 1 + MAX_OVERSHOOT)
    # y = np.clip(y, -MAX_OVERSHOOT, 1 + MAX_OVERSHOOT)
    # x = (x * 2) - 1
    # y = (y * 2) - 1
    x = np.clip(x, -OUT_CLAMP, OUT_CLAMP)
    y = np.clip(y, -OUT_CLAMP, OUT_CLAMP)

    return [
        {"t": float(tt), "x": float(xx), "y": float(yy)} for tt, xx, yy in zip(t, x, y)
    ]


stats = {
    "total": 0,
    "kept": 0,
    "invalid_sentence": 0,
    "invalid_word": 0,
    "invalid_length": 0,
    "too_rare": 0,  # NEW
    "not_portrait": 0,
    "canvas_wrong_dimensions": 0,
    "canvas_width_too_large": 0,
    "too_short_duration": 0,
    "too_long_duration": 0,
    "too_slow_speed": 0,
    "too_fast_speed": 0,
    "trace_too_short": 0,
    "trace_too_long": 0,
    "trace_error": 0,
}


# ----------------- MAIN FILTER -----------------
def filter_and_normalize(input_path: Path, output_path: Path, valid_set, freq_map):
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

                if not is_valid_word(canon, valid_set):
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
                    stats["canvas_wrong_dimensions"] += 1
                    continue
                if w > MAX_CANVAS_WIDTH:
                    stats["canvas_width_too_large"] += 1
                    continue

                norm_points = normalize_points(sample.get("data", []))
                if norm_points is None:
                    continue

                out = {
                    "word": canon,
                    "points": norm_points,
                    "id": sample.get("id"),
                    "session": sample.get("session"),
                    "timestamp": sample.get("timestamp"),
                }
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                stats["kept"] += 1

            except Exception as e:
                stats["trace_error"] += 1
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


# leonweber:
# {"id": 65, "session": "anon-session-6f52996c-e179-4224-b07c-9da6df04c98e", "timestamp": 1724390287154, "word": "The", "canvas_width": 422.0, "canvas_height": 170.3125, "orientation": "portrait-primary", "data": [{"t": 1724390286378, "x": 0.45023696682464454, "y": 0.18247706422018348}, {"t": 1724390286507, "x": 0.4786729857819905, "y": 0.2177064220183486}, {"t": 1724390286521, "x": 0.5071090047393365, "y": 0.27055045871559635}, {"t": 1724390286537, "x": 0.5308056872037915, "y": 0.3175229357798165}, {"t": 1724390286555, "x": 0.54739336492891, "y": 0.35862385321100915}, {"t": 1724390286571, "x": 0.5592417061611374, "y": 0.3938532110091743}, {"t": 1724390286588, "x": 0.5687203791469194, "y": 0.4232110091743119}, {"t": 1724390286604, "x": 0.5734597156398105, "y": 0.4408256880733945}, {"t": 1724390286623, "x": 0.5758293838862559, "y": 0.45256880733944954}, {"t": 1724390286672, "x": 0.5592417061611374, "y": 0.45256880733944954}, {"t": 1724390286689, "x": 0.5308056872037915, "y": 0.4232110091743119}, {"t": 1724390286706, "x": 0.48578199052132703, "y": 0.37036697247706424}, {"t": 1724390286722, "x": 0.42890995260663506, "y": 0.311651376146789}, {"t": 1724390286740, "x": 0.3767772511848341, "y": 0.2529357798165138}, {"t": 1724390286756, "x": 0.34360189573459715, "y": 0.20596330275229358}, {"t": 1724390286773, "x": 0.32701421800947866, "y": 0.18247706422018348}, {"t": 1724390286790, "x": 0.31990521327014215, "y": 0.16486238532110092}, {"t": 1724390286806, "x": 0.3127962085308057, "y": 0.15311926605504586}, {"t": 1724390286823, "x": 0.3056872037914692, "y": 0.14724770642201834}, {"t": 1724390286840, "x": 0.2985781990521327, "y": 0.14724770642201834}, {"t": 1724390286856, "x": 0.2890995260663507, "y": 0.14724770642201834}, {"t": 1724390286873, "x": 0.28199052132701424, "y": 0.14724770642201834}, {"t": 1724390286889, "x": 0.27488151658767773, "y": 0.14724770642201834}, {"t": 1724390286905, "x": 0.2725118483412322, "y": 0.14724770642201834}, {"t": 1724390287022, "x": 0.27014218009478674, "y": 0.15311926605504586}, {"t": 1724390287039, "x": 0.2677725118483412, "y": 0.15311926605504586}, {"t": 1724390287055, "x": 0.2677725118483412, "y": 0.1589908256880734}], "sentence": "The", "word_idx": 0, "potentially_invalid_sentence": true, "distance": 17.02327802608763, "masked_sentence": "<start> TYYYTYYJJGGGGRRRFERRREEEERDRRE <end>", "trajectory_sampled": [[0.46979628212661917, 0.1389662322464908], [0.5507752845606483, -0.08036003231824557], [0.5274100827967874, 0.08098197581495731], [0.5174478729525049, 0.309830016804462], [0.4791009519754161, 0.32553815799224933], [0.5240355170625893, 0.09609863649789599], [0.5732522424910671, 0.30552105072651436], [0.6226553016216716, 0.534040080375402], [0.606016478661277, 0.569658040331909], [0.49477231088218315, 0.5406883097486168], [0.4795596611302364, 0.351606236871874], [0.48522446435381017, 0.4426706409138568], [0.46300722207911055, 0.3524408289167372], [0.3991416300207672, 0.1275346366627923], [0.342769758313922, -0.04682783931786305], [0.3469072435688966, 0.18693377082994594], [0.3454993890560092, 0.38868164222567153], [0.2864113228100569, 0.22223472643888403], [0.3188952583045957, 0.2614895296031712], [0.3219118244791888, 0.027710767754184407], [0.3024982814804404, -0.023740592673447553], [0.24876301403149922, 0.20379870106572653], [0.23755542907516655, 0.026131968506602266], [0.23223266556039138, -0.20760565651704932], [0.2700110563202872, -0.025395084969895343], [0.31272724660874884, 0.20446777903899782], [0.3003771424173545, 0.3352772693741653], [0.3373190740803588, 0.3228768953083935], [0.3043474874418134, 0.09274558215328663], [0.2888118393355659, -0.1405359061844084]], "trajectory_sampled_keys": ["T", "Y", "Y", "Y", "T", "Y", "Y", "J", "J", "G", "G", "G", "G", "R", "R", "R", "F", "E", "R", "R", "R", "E", "E", "E", "E", "R", "D", "R", "R", "E"], "trajectory_word": "TYYYTYYJJGGGGRRRFERRREEEERDRRE"}

# futo:
# {"id":65,"session":"anon-session-6f52996c-e179-4224-b07c-9da6df04c98e","timestamp":1724390287154,"word":"The","canvas_width":422,"canvas_height":170.3125,"orientation":"portrait-primary","data":[{"t":1724390286378,"x":0.45023696682464454,"y":0.18247706422018348},{"t":1724390286507,"x":0.4786729857819905,"y":0.2177064220183486},{"t":1724390286521,"x":0.5071090047393365,"y":0.27055045871559635},{"t":1724390286537,"x":0.5308056872037915,"y":0.3175229357798165},{"t":1724390286555,"x":0.54739336492891,"y":0.35862385321100915},{"t":1724390286571,"x":0.5592417061611374,"y":0.3938532110091743},{"t":1724390286588,"x":0.5687203791469194,"y":0.4232110091743119},{"t":1724390286604,"x":0.5734597156398105,"y":0.4408256880733945},{"t":1724390286623,"x":0.5758293838862559,"y":0.45256880733944954},{"t":1724390286672,"x":0.5592417061611374,"y":0.45256880733944954},{"t":1724390286689,"x":0.5308056872037915,"y":0.4232110091743119},{"t":1724390286706,"x":0.48578199052132703,"y":0.37036697247706424},{"t":1724390286722,"x":0.42890995260663506,"y":0.311651376146789},{"t":1724390286740,"x":0.3767772511848341,"y":0.2529357798165138},{"t":1724390286756,"x":0.34360189573459715,"y":0.20596330275229358},{"t":1724390286773,"x":0.32701421800947866,"y":0.18247706422018348},{"t":1724390286790,"x":0.31990521327014215,"y":0.16486238532110092},{"t":1724390286806,"x":0.3127962085308057,"y":0.15311926605504586},{"t":1724390286823,"x":0.3056872037914692,"y":0.14724770642201834},{"t":1724390286840,"x":0.2985781990521327,"y":0.14724770642201834},{"t":1724390286856,"x":0.2890995260663507,"y":0.14724770642201834},{"t":1724390286873,"x":0.28199052132701424,"y":0.14724770642201834},{"t":1724390286889,"x":0.27488151658767773,"y":0.14724770642201834},{"t":1724390286905,"x":0.2725118483412322,"y":0.14724770642201834},{"t":1724390287022,"x":0.27014218009478674,"y":0.15311926605504586},{"t":1724390287039,"x":0.2677725118483412,"y":0.15311926605504586},{"t":1724390287055,"x":0.2677725118483412,"y":0.1589908256880734}],"sentence":"The","word_idx":0,"potentially_invalid_sentence":true,"distance":17.02327802608763}
