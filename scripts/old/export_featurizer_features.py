#!/usr/bin/env python3
"""
Export Python-side 37-D swipe features for a small sample to aid web parity checks.

Usage:
  uv run python scripts/export_featurizer_features.py \
      --manifest data/train_final_val.jsonl --count 20 --out exports/py_features.json

The output JSON contains entries: { "word": str, "features": [[...],[...], ...], "length": int }
These can be compared to features computed in web-demo for the same samples.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from omegaconf import DictConfig

from new.train_transducer_personalized import (
    PersonalizedSwipeFeaturizer,
    clamp,
    determine_resample_target,
    resample_points,
)


def load_samples(manifest: Path, count: int) -> List[Dict[str, Any]]:
    out = []
    with manifest.open("r", encoding="utf-8") as f:
        for line in f:
            if len(out) >= count:
                break
            try:
                rec = json.loads(line)
                if rec.get("word") and rec.get("points"):
                    out.append(rec)
            except Exception:
                continue
    return out


def normalize_points(points: List[Dict[str, float]]) -> List[Dict[str, float]]:
    if not points:
        return []
    start_t = float(points[0].get("t", 0.0))
    out: List[Dict[str, float]] = []
    for idx, pt in enumerate(points):
        x01 = float(pt.get("x", 0.0))
        y01 = float(pt.get("y", 0.0))
        x = clamp(x01 * 2.0 - 1.0, -1.5, 1.5)
        y = clamp(y01 * 2.0 - 1.0, -1.5, 1.5)
        t = float(pt.get("t", idx * 10.0)) - start_t
        out.append({"x": x, "y": y, "t": max(0.0, t)})
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--count", type=int, default=20)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    samples = load_samples(Path(args.manifest), args.count)
    featurizer = PersonalizedSwipeFeaturizer(key_centers_path=None, mobile_features=False)
    cfg = DictConfig({
        "resample_short_target": 56,
        "resample_long_target": 96,
        "resample_short_threshold": 48,
        "resample_long_threshold": 112,
    })

    results: List[Dict[str, Any]] = []
    for rec in samples:
        word = rec["word"]
        points = normalize_points(rec["points"])[:256]
        target_len = determine_resample_target(len(points), cfg)
        processed = resample_points(points, target_len)
        feats = featurizer(processed).tolist()
        results.append({"word": word, "length": len(processed), "features": feats})

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results), encoding="utf-8")
    print(f"Wrote {len(results)} examples to {out_path}")


if __name__ == "__main__":
    main()

