#!/usr/bin/env python3
"""Phase P6 — paired P4 (90/10 donor split) vs P6 (full donor pool) per script.

Both dumps come from `eval_script --dump` on the SAME holdout npz (proved
bit-identical across the two regenerations by phaseP6_assert_holdout.py), so
rows join on `row` and the discordant cells are a legitimate exact McNemar.
Reports top-1 and greedy, plus the <=3 / 4+ strata, and writes the whole thing
to ~/ctc-train/phaseP6_paired.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/will/git/CleverKeys-ML/ctc")
from phase_n_decomp import mcnemar_p  # noqa: E402

O = Path.home() / "ctc-train/evalP6"


def load(path: Path) -> dict:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            out[o["row"]] = o
    return out


def cells(a: dict, b: dict, key, rows) -> dict:
    """b = P4-only wins, c = P6-only wins, on the given row subset."""
    n = bb = cc = both = 0
    for r in rows:
        x, y = key(a[r]), key(b[r])
        n += 1
        both += int(x and y)
        bb += int(x and not y)
        cc += int(y and not x)
    return {"n": n, "p4_only": bb, "p6_only": cc, "both": both,
            "p4_acc": round((both + bb) / max(n, 1) * 100, 2),
            "p6_acc": round((both + cc) / max(n, 1) * 100, 2),
            "delta": round((cc - bb) / max(n, 1) * 100, 2),
            "p": float(f"{mcnemar_p(bb, cc):.3g}")}


def main() -> int:
    report = {}
    for code in sys.argv[1:]:
        a, b = load(O / f"dump_{code}_p4.jsonl"), load(O / f"dump_{code}_p6.jsonl")
        rows = sorted(set(a) & set(b))
        assert len(rows) == len(a) == len(b), \
            f"[{code}] row sets differ: {len(a)} / {len(b)} / {len(rows)}"
        t1 = lambda o: o["rank"] == 0          # noqa: E731
        gr = lambda o: bool(o["greedy_hit"])   # noqa: E731
        le3 = [r for r in rows if len(a[r]["target"]) <= 3]
        ge4 = [r for r in rows if len(a[r]["target"]) >= 4]
        report[code] = {
            "indict_t1": cells(a, b, t1, rows),
            "greedy": cells(a, b, gr, rows),
            "le3_t1": cells(a, b, t1, le3),
            "ge4_t1": cells(a, b, t1, ge4),
        }
        d = report[code]["indict_t1"]
        g = report[code]["greedy"]
        print(f"{code}: t1 {d['p4_acc']:6.2f} -> {d['p6_acc']:6.2f} "
              f"({d['delta']:+.2f}, b/c {d['p4_only']}/{d['p6_only']}, p {d['p']})"
              f"   greedy {g['p4_acc']:6.2f} -> {g['p6_acc']:6.2f} "
              f"({g['delta']:+.2f}, p {g['p']})")
    out = Path.home() / "ctc-train/phaseP6_paired.json"
    out.write_text(json.dumps(report, indent=1))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
