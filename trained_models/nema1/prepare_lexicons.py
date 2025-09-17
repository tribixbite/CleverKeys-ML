#!/usr/bin/env python3
"""Generate curated CleverKeys lexicons for export packaging."""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOG = logging.getLogger("prepare_lexicons")

ALLOWED_CHARS = set("abcdefghijklmnopqrstuvwxyz'")
REPEAT_PATTERN = re.compile(r"([a-z'])\1{3,}")  # four repeats in a row
VOWELS = set("aeiouy")
ALLOWED_CONTRACTIONS = {"'s", "'d", "'ll", "'re", "'ve", "'m", "'t", "'em"}
PRIORITY_WORDS = list(ALLOWED_CONTRACTIONS)


def load_word_frequencies(path: Path) -> Dict[str, float]:
    with path.open() as handle:
        data = json.load(handle)
    freqs = data.get("word_frequencies")
    if not isinstance(freqs, dict):
        raise ValueError(f"word_frequencies missing in {path}")
    return {word: float(freq) for word, freq in freqs.items()}


def load_wordlist(path: Path) -> List[str]:
    with path.open() as handle:
        return [line.strip() for line in handle if line.strip()]


def is_reasonable_word(word: str, max_length: int, allow_apostrophe_start: bool = True) -> bool:
    if len(word) > max_length:
        return False
    if word.count("'") > 1 and word.endswith("''"):
        return False
    if not allow_apostrophe_start and word.startswith("'") and len(word) > 1:
        return False
    if not set(word) <= ALLOWED_CHARS:
        return False
    if REPEAT_PATTERN.search(word):
        return False
    letters = [ch for ch in word if ch.isalpha()]
    if letters and not any(ch in VOWELS for ch in letters):
        if word not in ALLOWED_CONTRACTIONS:
            return False
    return True


def filter_and_rank(
    words: Iterable[str],
    freqs: Dict[str, float],
    *,
    max_length: int,
    min_frequency: float,
    max_items: int | None,
) -> List[Tuple[str, float]]:
    seen = set()
    ranked: List[Tuple[str, float]] = []
    for word in words:
        if word in seen:
            continue
        seen.add(word)
        freq = freqs.get(word, 0.0)
        if freq < min_frequency:
            continue
        if not is_reasonable_word(word, max_length=max_length):
            continue
        ranked.append((word, freq))
    ranked.sort(key=lambda wf: (-wf[1], wf[0]))
    if max_items is not None:
        ranked = ranked[:max_items]
    return ranked


def write_wordlist(path: Path, entries: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for word in entries:
            handle.write(f"{word}\n")


def summarise(words: List[str], freqs: Dict[str, float]) -> Dict[str, float]:
    total_freq = sum(freqs.get(w, 0.0) for w in words)
    return {
        "count": len(words),
        "estimated_frequency_mass": total_freq,
        "max_frequency": max((freqs.get(w, 0.0) for w in words), default=0.0),
        "min_frequency": min((freqs.get(w, 0.0) for w in words), default=0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare curated lexicons for CleverKeys exports")
    parser.add_argument("--wordlist", default="trained_models/nema1/words.txt", help="Base word list")
    parser.add_argument(
        "--frequencies",
        default="web-demo/swipe_vocabulary.json",
        help="JSON file with word_frequencies mapping",
    )
    parser.add_argument("--output-dir", default="vocab/lexicons", help="Directory for generated lexicons")
    parser.add_argument("--small-size", type=int, default=15000, help="Number of entries in the small lexicon")
    parser.add_argument("--full-size", type=int, default=75000, help="Number of entries in the full lexicon")
    parser.add_argument("--max-length", type=int, default=24, help="Maximum word length")
    parser.add_argument("--min-frequency", type=float, default=1e-8, help="Minimum frequency threshold")
    args = parser.parse_args()

    wordlist_path = Path(args.wordlist)
    freq_path = Path(args.frequencies)
    output_dir = Path(args.output_dir)

    LOG.info("Loading frequencies from %s", freq_path)
    freqs = load_word_frequencies(freq_path)
    LOG.info("Loaded %d frequency entries", len(freqs))

    LOG.info("Loading base wordlist from %s", wordlist_path)
    base_words = load_wordlist(wordlist_path)
    LOG.info("Loaded %d candidate words", len(base_words))

    filtered = filter_and_rank(
        base_words,
        freqs,
        max_length=args.max_length,
        min_frequency=args.min_frequency,
        max_items=None,
    )
    LOG.info("Retained %d words after filtering", len(filtered))

    if args.full_size and args.full_size < len(filtered):
        full_entries = filtered[: args.full_size]
    else:
        full_entries = filtered

    if args.small_size:
        small_entries = filtered[: min(args.small_size, len(filtered))]
    else:
        small_entries = filtered

    def ensure_priority(entries: List[Tuple[str, float]], limit: int | None) -> List[Tuple[str, float]]:
        selected = list(entries)
        existing = {word for word, _ in selected}
        if selected:
            floor_index = min(len(selected), limit or len(selected)) - 1
            floor_freq = selected[floor_index][1]
        else:
            floor_freq = args.min_frequency
        extras: List[Tuple[str, float]] = []
        for word in PRIORITY_WORDS:
            if word in existing or word not in base_words:
                continue
            if not is_reasonable_word(word, max_length=args.max_length):
                continue
            freq = freqs.get(word, floor_freq)
            extras.append((word, max(freq, floor_freq)))
        if not extras:
            return selected
        combined = selected + extras
        combined.sort(key=lambda wf: (-wf[1], wf[0]))
        if limit is not None:
            combined = combined[:limit]
        return combined

    full_entries = ensure_priority(full_entries, args.full_size if args.full_size else None)
    small_entries = ensure_priority(small_entries, args.small_size if args.small_size else None)

    small_path = output_dir / "lexicon_small.txt"
    full_path = output_dir / "lexicon_full.txt"

    write_wordlist(small_path, (word for word, _ in small_entries))
    write_wordlist(full_path, (word for word, _ in full_entries))

    stats = {
        "source_wordlist": str(wordlist_path),
        "frequency_file": str(freq_path),
        "max_length": args.max_length,
        "min_frequency": args.min_frequency,
        "small": summarise([w for w, _ in small_entries], freqs),
        "full": summarise([w for w, _ in full_entries], freqs),
    }

    stats_path = output_dir / "lexicon_stats.json"
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)

    LOG.info("Small lexicon -> %s (%d words)", small_path, len(small_entries))
    LOG.info("Full lexicon -> %s (%d words)", full_path, len(full_entries))
    LOG.info("Stats -> %s", stats_path)


if __name__ == "__main__":
    main()
