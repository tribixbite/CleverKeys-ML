#!/usr/bin/env python3
"""Sweep the K3 rerank blend weight w over a recorded eval_beam.py dump.

Input: a ``--out`` dump produced with ``--ranker-onnx`` at ``--rerank-weight 0``
(topk in BEAM order, with the aligned per-candidate ``ranker`` column). For each
w the candidates are re-sorted by ``beam_final + w * ranker`` and the five
campaign metrics recomputed — no beam re-decode, so the whole grid costs
seconds.

Protocol: the grid is scored on the TUNE half (rows [0:n/2)) and the chosen w
(min-margin against the five val bars, the §6.8b objective) is then read once
on the CONFIRM half. Both tables are printed; nothing is hidden.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

#: Phase-J §0 val bars (resbn192i 3-seed means): t1, t3, t5, <=3, 4+.
DEFAULT_BARS = (88.30, 92.60, 93.26, 91.27, 86.77)


def load_dump(path: Path):
    rows = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            rows.append((d["word"].lower(), d["topk"], d.get("ranker")))
    return rows


def metrics(rows, w: float):
    n = t1 = t3 = t5 = 0
    s_n = {True: 0, False: 0}
    s_t1 = {True: 0, False: 0}
    for gold, topk, rk in rows:
        if w == 0.0 or rk is None:
            order = [wd for wd, _ in topk]
        else:
            order = [wd for wd, _ in sorted(
                ((wd, s + w * r) for (wd, s), r in zip(topk, rk)),
                key=lambda t: t[1], reverse=True)]
        try:
            r = order.index(gold)
        except ValueError:
            r = -1
        n += 1
        short = len(gold) <= 3
        s_n[short] += 1
        if r == 0:
            t1 += 1
            s_t1[short] += 1
        if 0 <= r < 3:
            t3 += 1
        if 0 <= r < 5:
            t5 += 1
    return (t1 / n * 100, t3 / n * 100, t5 / n * 100,
            s_t1[True] / max(s_n[True], 1) * 100,
            s_t1[False] / max(s_n[False], 1) * 100)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--grid", default="0,0.05,0.1,0.15,0.2,0.3,0.5,0.75,1,1.5,2,3")
    ap.add_argument("--bars", default=",".join(str(b) for b in DEFAULT_BARS),
                    help="t1,t3,t5,le3,4p bars for the min-margin objective")
    args = ap.parse_args()
    rows = load_dump(Path(args.dump))
    bars = [float(b) for b in args.bars.split(",")]
    half = len(rows) // 2
    tune, confirm = rows[:half], rows[half:]
    print(f"{len(rows)} rows -> tune {len(tune)} / confirm {len(confirm)}")
    grid = [float(w) for w in args.grid.split(",")]
    best_w, best_key = None, None
    print(f"{'w':>6} | {'t1':>6} {'t3':>6} {'t5':>6} {'<=3':>6} {'4+':>6} | "
          f"minmargin  (TUNE half)")
    for w in grid:
        m = metrics(tune, w)
        margins = [m[i] - bars[i] for i in range(5)]
        key = (min(margins), sum(margins) / 5)
        tag = ""
        if best_key is None or key > best_key:
            best_w, best_key = w, key
            tag = " <-"
        print(f"{w:>6} | {m[0]:6.2f} {m[1]:6.2f} {m[2]:6.2f} {m[3]:6.2f} "
              f"{m[4]:6.2f} | {key[0]:+.2f}{tag}")
    print(f"\nchosen on tune: w = {best_w}  -> CONFIRM half:")
    for w in (0.0, best_w):
        m = metrics(confirm, w)
        print(f"{w:>6} | {m[0]:6.2f} {m[1]:6.2f} {m[2]:6.2f} {m[3]:6.2f} "
              f"{m[4]:6.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
