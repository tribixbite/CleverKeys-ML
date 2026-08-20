#!/usr/bin/env python3
"""Phase Q — paired comparison of two `eval_script --dump` JSONLs.

Both dumps must come from the SAME probe rows (they join on ``row``; any
asymmetric difference in the row sets is an error, not a warning), so the
discordant cells are a legitimate exact McNemar.  Reports in-dict top-1 and
greedy overall plus the ≤3 / 4+ strata — the G5-Q ship-gate read
(PHASE_Q.md §4) and the upper-bound read (§5.3) both come from here.

Usage::

  python3 ctc/phaseQ_paired.py --a evalQ/dump_ru_v2full.jsonl --a-name v2full \
      --b evalQ/dump_ru_v3.jsonl --b-name v3 --out evalQ/paired_v3_vs_v2.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase_n_decomp import mcnemar_p  # noqa: E402


def load(path: Path) -> Dict[int, dict]:
    out: Dict[int, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            out[o["row"]] = o
    return out


def cells(a: Dict[int, dict], b: Dict[int, dict], key: Callable[[dict], bool],
          rows) -> dict:
    n = only_a = only_b = both = 0
    for r in rows:
        x, y = key(a[r]), key(b[r])
        n += 1
        both += int(x and y)
        only_a += int(x and not y)
        only_b += int(y and not x)
    pa = (both + only_a) / max(n, 1) * 100
    pb = (both + only_b) / max(n, 1) * 100
    return {"n": n, "a_pct": round(pa, 2), "b_pct": round(pb, 2),
            "delta_b_minus_a": round(pb - pa, 2), "a_only": only_a,
            "b_only": only_b, "p_mcnemar": mcnemar_p(only_a, only_b)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--a", type=Path, required=True)
    ap.add_argument("--b", type=Path, required=True)
    ap.add_argument("--a-name", default="a")
    ap.add_argument("--b-name", default="b")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    A, B = load(args.a), load(args.b)
    if set(A) != set(B):
        raise SystemExit(f"row sets differ: {len(A)} vs {len(B)} rows, "
                         f"symmetric diff {len(set(A) ^ set(B))} — the two "
                         f"dumps are not the same probe")
    rows = sorted(A)
    rep = {
        "a": {"name": args.a_name, "dump": str(args.a)},
        "b": {"name": args.b_name, "dump": str(args.b)},
        "indict_t1": cells(A, B, lambda o: o["rank"] == 0, rows),
        "greedy": cells(A, B, lambda o: bool(o["greedy_hit"]), rows),
        "le3_t1": cells(A, B, lambda o: o["rank"] == 0,
                        [r for r in rows if len(A[r]["target"]) <= 3]),
        "ge4_t1": cells(A, B, lambda o: o["rank"] == 0,
                        [r for r in rows if len(A[r]["target"]) >= 4]),
    }
    for k in ("indict_t1", "greedy", "le3_t1", "ge4_t1"):
        c = rep[k]
        print(f"{k:<10} n {c['n']:>6}  {args.a_name} {c['a_pct']:>6.2f}  "
              f"{args.b_name} {c['b_pct']:>6.2f}  Δ {c['delta_b_minus_a']:+.2f}  "
              f"(a-only {c['a_only']}, b-only {c['b_only']}, "
              f"p {c['p_mcnemar']:.3g})")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rep, indent=1, ensure_ascii=False))
        print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
