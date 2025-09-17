#!/usr/bin/env python3
"""Build a word-frequency array aligned to web-demo/words.txt.

Each entry in words.txt receives a natural-log frequency value.
Data is sourced from swipe_vocabulary.json but kept in the exact order
and spelling (lowercase + apostrophes) found in words.txt.
Words with no matching frequency fall back to a tiny prior (1e-8).
"""
from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Dict, Tuple

ROOT = Path(__file__).resolve().parents[1]
WORDS_PATH = ROOT / "web-demo" / "words.txt"
FREQ_JSON_PATH = ROOT / "web-demo" / "swipe_vocabulary.json"
OUT_PATH = ROOT / "web-demo" / "word_frequencies_aligned.json"

FALLBACK_FREQ = 1e-8


def sanitize(word: str) -> str:
    """Normalize word to lowercase ASCII letters plus apostrophes."""
    word = word.strip().lower()
    word = unicodedata.normalize("NFKD", word)
    return "".join(ch for ch in word if ch == "'" or ('a' <= ch <= 'z'))


def load_words() -> Tuple[list[str], Dict[str, int]]:
    words = [line.strip() for line in WORDS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    index_map = {word: idx for idx, word in enumerate(words)}
    return words, index_map


def load_freq_map() -> Dict[str, float]:
    data = json.loads(FREQ_JSON_PATH.read_text(encoding="utf-8"))
    raw_freq = data.get("word_frequencies", {})
    normalized: Dict[str, float] = {}
    for raw_word, freq in raw_freq.items():
        key = sanitize(raw_word)
        if not key:
            continue
        existing = normalized.get(key)
        if existing is None or freq > existing:
            normalized[key] = freq
    return normalized


def main() -> None:
    words, _ = load_words()
    freq_map = load_freq_map()

    log_freqs = []
    fallback_words = []
    stats = {
        "total_words": len(words),
        "freq_entries_available": len(freq_map),
        "matched": 0,
        "fallback": 0,
    }

    for word in words:
        sanitized = sanitize(word)
        freq = freq_map.get(sanitized)
        if freq is None or freq <= 0.0:
            freq = FALLBACK_FREQ
            fallback_words.append(word)
            stats["fallback"] += 1
        else:
            stats["matched"] += 1
        log_freqs.append(float(math_log(freq)))

    out = {
        "description": "Natural-log frequencies aligned with words.txt order",
        "words": len(words),
        "fallback_entries": len(fallback_words),
        "log_frequencies": log_freqs,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"Wrote aligned frequencies to {OUT_PATH}")
    print(json.dumps(stats, indent=2))
    if fallback_words:
        preview = ", ".join(fallback_words[:20])
        print(f"Example fallback words ({len(fallback_words)} total): {preview} ...")


def math_log(x: float) -> float:
    import math
    return math.log(x if x > 0.0 else FALLBACK_FREQ)


if __name__ == "__main__":
    main()
