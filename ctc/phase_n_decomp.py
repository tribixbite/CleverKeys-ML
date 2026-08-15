#!/usr/bin/env python3
"""Phase N gap decomposition — where exactly does FUTO's engine beat ours?

Joins per-row outputs of both engines on the same converted official split
(`PHASE_N.md` §6-N1.3) with the HF original rows (by ``id``) to recover the
metadata the converter drops (orientation, canvas), and stratifies paired
top-1 correctness by: word length, lexicon membership (OOV), orientation,
trace point count, and normalized path speed quartiles. Also reports the
paired win/loss cells and an exact two-sided McNemar per stratum.

Inputs accept both dump schemas:
  * ours  (``eval_beam --out``):   {idx, word, topk: [[w, s]…], rank}
  * FUTO  (``futo_decoder_*``):    {idx, word, in_vocab, preds: [w…]}

Rows are aligned by ``idx`` = line number of the converted split, which both
harnesses preserve.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scan_futo_sessions import normalize_word  # noqa: E402


def mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided binomial McNemar on discordant counts (b, c)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # two-sided: 2 * P(X <= k), capped at 1
    p = 0.0
    for i in range(k + 1):
        p += math.comb(n, i)
    p = 2.0 * p * (0.5 ** n)
    return min(1.0, p)


def load_ours(path: Path) -> dict:
    out = {}
    with open(path) as f:
        for line in f:
            o = json.loads(line)
            preds = [w for w, *_ in o.get("topk", [])]
            out[o["idx"]] = preds
    return out


def load_futo(path: Path) -> dict:
    out = {}
    with open(path) as f:
        for line in f:
            o = json.loads(line)
            out[o["idx"]] = list(o.get("preds", []))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", required=True, help="converted futo_{dev,test}.jsonl")
    ap.add_argument("--hf", required=True, help="original HF jsonl (for metadata)")
    ap.add_argument("--ours", required=True, help="our per-row dump")
    ap.add_argument("--futo", required=True, help="FUTO per-row dump")
    ap.add_argument("--vocab", required=True, help="wordlist for the OOV flag")
    ap.add_argument("--out", default="", help="optional json report path")
    args = ap.parse_args()

    vocab = set()
    with open(args.vocab, errors="replace") as f:
        for line in f:
            if line.startswith(" word="):
                w = normalize_word(line.split("word=")[1].split(",")[0])
                if w:
                    vocab.add(w)
            elif "=" not in line:
                w = normalize_word(line.strip())
                if w:
                    vocab.add(w)
    print(f"vocab: {len(vocab)} normalized words")

    meta = {}
    with open(args.hf) as f:
        for line in f:
            o = json.loads(line)
            meta[o["id"]] = o
    ours = load_ours(Path(args.ours))
    futo = load_futo(Path(args.futo))

    rows = []
    with open(args.split) as f:
        for idx, line in enumerate(f):
            o = json.loads(line)
            m = meta[o["id"]]
            tgt = normalize_word(o["word"])
            po = ours.get(idx)
            pf = futo.get(idx)
            if po is None or pf is None:
                continue
            pts = o["points"]
            xs = np.array([p["x"] for p in pts])
            ys = np.array([p["y"] for p in pts])
            ts = np.array([p["t"] for p in pts])
            dur = max(float(ts[-1] - ts[0]), 1.0)
            path_len = float(np.hypot(np.diff(xs), np.diff(ys)).sum())
            rows.append({
                "tgt": tgt,
                "len": len(tgt),
                "oov": tgt not in vocab,
                "orient": (m.get("orientation") or "?").split("-")[0],
                "npts": len(pts),
                "speed": path_len / dur * 1000.0,  # letter-area units / s
                "o1": bool(po) and normalize_word(po[0]) == tgt,
                "f1": bool(pf) and normalize_word(pf[0]) == tgt,
                "o3": tgt in [normalize_word(w) for w in po[:3]],
                "f3": tgt in [normalize_word(w) for w in pf[:3]],
            })
    n = len(rows)
    print(f"joined rows: {n}")

    def cell(sub, label):
        m = len(sub)
        if m == 0:
            return
        o1 = sum(r["o1"] for r in sub)
        f1 = sum(r["f1"] for r in sub)
        b = sum(1 for r in sub if r["o1"] and not r["f1"])   # we win
        c = sum(1 for r in sub if r["f1"] and not r["o1"])   # they win
        p = mcnemar_p(b, c)
        print(f"{label:<28} n={m:<6} ours t1 {100*o1/m:6.2f}  futo t1 {100*f1/m:6.2f} "
              f" Δ {100*(o1-f1)/m:+6.2f}  we-win {b:<5} they-win {c:<5} p={p:.2e}")
        return {"label": label, "n": m, "ours_t1": 100*o1/m, "futo_t1": 100*f1/m,
                "we_win": b, "they_win": c, "p": p}

    report = []
    report.append(cell(rows, "ALL"))
    report.append(cell([r for r in rows if not r["oov"]], "in-vocab"))
    report.append(cell([r for r in rows if r["oov"]], "OOV (both always miss)"))
    for lo, hi, lab in ((1, 3, "len<=3"), (4, 5, "len 4-5"), (6, 8, "len 6-8"),
                        (9, 99, "len>=9")):
        report.append(cell([r for r in rows if lo <= r["len"] <= hi], lab))
    for orient in sorted({r["orient"] for r in rows}):
        report.append(cell([r for r in rows if r["orient"] == orient],
                           f"orient={orient}"))
    iv = [r for r in rows if not r["oov"]]
    qs = np.percentile([r["speed"] for r in iv], [25, 50, 75])
    for i, (lo, hi) in enumerate(zip([-1e9, *qs], [*qs, 1e9])):
        report.append(cell([r for r in iv if lo < r["speed"] <= hi],
                           f"in-vocab speed q{i+1}"))
    qs = np.percentile([r["npts"] for r in iv], [25, 50, 75])
    for i, (lo, hi) in enumerate(zip([-1e9, *qs], [*qs, 1e9])):
        report.append(cell([r for r in iv if lo < r["npts"] <= hi],
                           f"in-vocab npts q{i+1}"))

    if args.out:
        Path(args.out).write_text(json.dumps([r for r in report if r], indent=1))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
