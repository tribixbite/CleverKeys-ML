#!/usr/bin/env python3
"""
Aggregate WER metrics from comprehensive runner CSV logs.

Usage:
  uv run python scripts/metrics_aggregate.py --base ./9292025script/20251002 \
      [--baseline-profile validation_balanced] [--out-md SUMMARY.md]

Outputs per-profile best/latest WER and overall best. If a baseline profile is
provided, prints deltas to the best WER observed under the baseline.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import csv
from collections import defaultdict


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="Run base directory (e.g., ./9292025script/20251002)")
    p.add_argument("--baseline-profile", default=None, help="Profile name to use as baseline for WER deltas")
    p.add_argument("--out-md", default=None, help="Optional Markdown output file")
    return p.parse_args()


def load_rows(metrics_path: Path) -> list[dict]:
    rows: list[dict] = []
    with metrics_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def main() -> None:
    args = parse_args()
    base = Path(args.base)
    logs_dir = base / "training_logs"
    csvs = sorted(logs_dir.glob("metrics_*.csv"))
    if not csvs:
        print(f"No metrics CSV found under {logs_dir}")
        return

    rows: list[dict] = []
    for csv_path in csvs:
        rows.extend(load_rows(csv_path))

    # Per-profile best/latest
    best_by_profile: dict[str, tuple[float, dict]] = {}
    latest_by_profile: dict[str, dict] = {}

    def parse_float(x: str) -> float:
        try:
            return float(x)
        except Exception:
            return float("inf")

    for r in rows:
        profile = r.get("profile", "unknown")
        latest_by_profile[profile] = r
        wer = parse_float(r.get("wer", "inf"))
        if wer != float("inf"):
            cur = best_by_profile.get(profile)
            if cur is None or wer < cur[0]:
                best_by_profile[profile] = (wer, r)

    # Overall best
    overall_best = None
    for prof, (wer, r) in best_by_profile.items():
        if overall_best is None or wer < overall_best[0]:
            overall_best = (wer, r)

    # Print summary
    lines: list[str] = []

    def add(line: str) -> None:
        print(line)
        lines.append(line)

    add("Per-profile best WER:")
    baseline_best = None
    if args.baseline_profile and args.baseline_profile in best_by_profile:
        baseline_best = best_by_profile[args.baseline_profile][0]

    for prof in sorted(best_by_profile.keys()):
        wer, r = best_by_profile[prof]
        delta = ""
        if baseline_best is not None:
            delta = f"  (Δ vs {args.baseline_profile}: {wer - baseline_best:+.3f})"
        add(f"  {prof:20s}  best WER={wer:.3f}{delta}  ckpt={r.get('checkpoint','')}  epoch={r.get('epoch','')}")

    add("\nPer-profile latest:")
    for prof in sorted(latest_by_profile.keys()):
        r = latest_by_profile[prof]
        w = r.get("wer", "N/A")
        add(f"  {prof:20s}  latest WER={w}  ckpt={r.get('checkpoint','')}  epoch={r.get('epoch','')}")

    if overall_best:
        wer, r = overall_best
        add("\nOverall best:")
        add(f"  best WER={wer:.3f}  profile={r.get('profile','')}  ckpt={r.get('checkpoint','')}")

    if args.out_md:
        out = Path(args.out_md)
        out.parent.mkdir(parents=True, exist_ok=True)
        md = ["# WER Summary\n"] + [l + "\n" for l in lines]
        out.write_text("".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
