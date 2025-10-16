#!/usr/bin/env python3
"""Quick summary statistics for swipe datasets."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np


def load_records(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def summarize(manifest: Path, top_k: int = 25) -> None:
    word_counter: Counter[str] = Counter()
    length_counter: Counter[int] = Counter()
    point_counts: list[int] = []
    out_of_range = 0

    for record in load_records(manifest):
        word = record.get("word") or record.get("target")
        points = record.get("points") or []
        if not word or not points:
            continue
        word_counter[word] += 1
        length_counter[len(word)] += 1
        point_counts.append(len(points))

        for pt in points:
            x = pt.get("x")
            y = pt.get("y")
            if x is not None and not (0.0 <= float(x) <= 1.0):
                out_of_range += 1
                break
            if y is not None and not (0.0 <= float(y) <= 1.0):
                out_of_range += 1
                break

    total = sum(word_counter.values())
    print(f"manifest: {manifest}")
    print(f"samples: {total}")

    print(f"\nTop {top_k} words:")
    for word, freq in word_counter.most_common(top_k):
        print(f"  {word:<16} {freq:>7} ({freq/total:.2%})")

    print("\nWord length distribution:")
    for length, freq in length_counter.most_common():
        pct = freq / total
        if pct < 0.01:
            continue
        print(f"  len={length:<2} count={freq:>7} ({pct:.2%})")

    if point_counts:
        arr = np.asarray(point_counts)
        print("\nTrace length stats:")
        for label, value in (
            ("min", arr.min()),
            ("max", arr.max()),
            ("mean", arr.mean()),
            ("p90", np.percentile(arr, 90)),
            ("p99", np.percentile(arr, 99)),
        ):
            print(f"  {label:<4} {value:.2f}")

    print(f"\nSwipes with coordinates outside [0,1]: {out_of_range}")
    uniques = sum(1 for count in word_counter.values() if count == 1)
    print(f"Unique words occurring once: {uniques}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse swipe dataset manifests")
    parser.add_argument("manifest", type=Path, help="Path to JSONL manifest")
    parser.add_argument("--top-k", type=int, default=25, help="How many frequent words to display")
    args = parser.parse_args()

    summarize(args.manifest, args.top_k)


if __name__ == "__main__":
    main()
