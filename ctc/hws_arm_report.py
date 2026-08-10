#!/usr/bin/env python3
"""Per-source / per-englishLevel / contributor-overlap report for the I-B arms.

Consumes one ``eval_beam.py --out`` per-trace dump per arm (full val-9918,
row order == ``val_hwsfuto.jsonl``) and joins each row to:

* its **source** (futo | hws) — ``cache/holdout_source_tags.json``;
* for HWS rows, the contributor uid (the row's own ``session`` field) and that
  uid's **englishLevel** (``data/hws_uid_levels.json``);
* whether that contributor is **inside the arm's HWS training pool** — the
  T3-family tiers are contributor-dirty by design (PHASE_A §5: benchmark
  tier), and the level arms REMOVE some val contributors from training, so a
  per-level comparison that ignored overlap would confuse "less leak" with
  "worse data".  Both slices are reported.

Output: one markdown-ish table block per arm + a JSON blob for the doc.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_hws_arms import ARM_LEVELS, LEVELS  # noqa: E402
from paths import DEFAULT_WORKDIR, resolve  # noqa: E402


def acc(rows: List[dict], k: int = 1) -> float:
    n = sum(1 for _ in rows)
    if not n:
        return float("nan")
    hit = sum(1 for r in rows if 0 <= r["rank"] < k)
    return hit / n * 100


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    ap.add_argument("--dumps", required=True,
                    help="comma-separated arm=dump.jsonl pairs")
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()

    data_dir = resolve(args.workdir, Path("data"))
    tags = json.loads((resolve(args.workdir, Path("cache")) /
                       "holdout_source_tags.json").read_text())
    val_tags: List[str] = tags["val"] if isinstance(tags, dict) else tags
    levels: Dict[str, str] = json.loads((data_dir / "hws_uid_levels.json").read_text())

    sessions: List[Optional[str]] = []
    with open(data_dir / "val_hwsfuto.jsonl") as f:
        for line in f:
            sessions.append(str(json.loads(line).get("session")))

    report: Dict[str, dict] = {}
    for pair in args.dumps.split(","):
        arm, dump = pair.split("=", 1)
        rows = [json.loads(l) for l in open(dump)]
        if len(rows) != len(sessions):
            raise SystemExit(f"{arm}: dump has {len(rows)} rows, val has "
                             f"{len(sessions)}")
        # the arm's HWS training contributors
        keep_levels = ARM_LEVELS.get(arm)
        futo = [r for r, t in zip(rows, val_tags) if t == "futo"]
        hws = [(r, sessions[i]) for i, (r, t) in enumerate(zip(rows, val_tags))
               if t == "hws"]
        blk: Dict[str, object] = {
            "all_t1": round(acc(rows), 2), "all_t3": round(acc(rows, 3), 2),
            "all_t5": round(acc(rows, 5), 2),
            "futo_t1": round(acc(futo), 2),
            "hws_t1": round(acc([r for r, _ in hws]), 2),
        }
        by_level: Dict[str, dict] = {}
        for lv in LEVELS:
            sub = [r for r, s in hws if levels.get(s) == lv]
            if not sub:
                continue
            e = {"n": len(sub), "t1": round(acc(sub), 2)}
            if keep_levels is not None:
                e["in_train"] = lv in keep_levels
            by_level[lv] = e
        blk["hws_by_level"] = by_level
        if keep_levels is not None:
            inl = [r for r, s in hws if levels.get(s) in keep_levels]
            outl = [r for r, s in hws if levels.get(s) not in keep_levels]
            blk["hws_traincontrib_t1"] = round(acc(inl), 2)
            blk["hws_traincontrib_n"] = len(inl)
            blk["hws_heldoutcontrib_t1"] = round(acc(outl), 2)
            blk["hws_heldoutcontrib_n"] = len(outl)
        report[arm] = blk
        print(f"== {arm}")
        print(json.dumps(blk, indent=1))

    if args.out_json:
        args.out_json.write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
