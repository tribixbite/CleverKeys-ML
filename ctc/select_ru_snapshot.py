#!/usr/bin/env python3
"""Offline beam-t1 checkpoint selection for Cyrillic runs (Phase J).

`train.py`'s in-process BeamValidator is a-z-hardcoded (vocab loader + warp
targets), so ru runs select on greedy — which Phase B measured as
ANTI-correlated with beam top-1. This tool restores beam-t1 selection without
touching the validator: it scores every retained ``snap_*.pt`` of a run by
lexicon-beam top-1 over a SYNTHETIC selection val (no real Cyrillic row is
touched — the no-corpus counterfactual stays intact for synth-only arms) and
reports the winner.

Usage:
  python select_ru_snapshot.py --run phaseJ-ru-synth-192 \
      --cache cache_ru_synth --rows 5000
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_cyrillic import build_trie  # noqa: E402
from futo_decoder_ceiling import futo_viterbi_beam  # noqa: E402
from futo_decoder_eval import load_layout  # noqa: E402
from model import MAX_KEYS, encoder_from_checkpoint, slice_head_torch  # noqa: E402
from paths import DEFAULT_WORKDIR, resolve  # noqa: E402

HERE = Path(__file__).resolve().parent
E1 = (1.05, 1.1, 0.2, 0.3734, 0.9882)

_TRIE = None
_LETTERS: List[str] = []
_EM = None


def _init(trie, letters, em):
    global _TRIE, _LETTERS, _EM
    _TRIE, _LETTERS, _EM = trie, letters, em


def _chunk(bounds):
    lo, hi, targets = bounds
    g, lam, b, gp, bp = E1
    n = len(_LETTERS)
    h1 = 0
    for i in range(lo, hi):
        cands = futo_viterbi_beam(_EM[i], _LETTERS, n, _TRIE, 100, 1,
                                  g, lam, b, gp, bp)
        if cands and cands[0][0] == targets[i - lo]:
            h1 += 1
    return h1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    ap.add_argument("--run", required=True)
    ap.add_argument("--cache", default="cache_ru_synth")
    ap.add_argument("--layout", default="ru_jcuken_default.json")
    ap.add_argument("--rows", type=int, default=5000)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--lexicon", default="app", choices=("app", "voc"))
    args = ap.parse_args()

    run_dir = resolve(args.workdir, Path("ckpt") / args.run)
    snaps = sorted(run_dir.glob("snap_*.pt"),
                   key=lambda p: int(p.stem.split("_")[1]))
    if not snaps:
        raise SystemExit(f"{run_dir}: no snap_*.pt (train with --snapshot-every)")
    letters, centers = load_layout(HERE / "layouts" / args.layout)
    nl = len(letters)
    with np.load(resolve(args.workdir, args.cache) / "val.npz") as d:
        feats = np.array(d["features"][: args.rows])
        words = [str(w) for w in d["words"][: args.rows]]
    trie, st = build_trie(args.lexicon,
                          Path.home() / "ctc-train" / "data" / "yandex_cup")
    print(f"{len(snaps)} snapshots; {len(feats)} synth selection rows; "
          f"lexicon {args.lexicon} {st['distinct']} words")
    keep = [i for i, w in enumerate(words) if trie.contains(w)]
    print(f"in-dict selection rows: {len(keep)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    keys = np.zeros((MAX_KEYS, 2), np.float32)
    keys[:nl] = centers
    kmask = np.zeros((MAX_KEYS,), bool)
    kmask[:nl] = True
    kt = torch.from_numpy(keys)[None].to(device)
    mt = torch.from_numpy(kmask)[None].to(device)
    ft = torch.from_numpy(feats[keep]).to(device)
    targets = [words[i] for i in keep]

    shm = mp.RawArray("f", len(keep) * 32 * (nl + 1))
    em = np.frombuffer(shm, np.float32).reshape(len(keep), 32, nl + 1)
    ctx = mp.get_context("fork")
    pool = ctx.Pool(args.jobs, initializer=_init, initargs=(trie, letters, em))
    step = max(1, (len(keep) + args.jobs - 1) // args.jobs)
    chunks = [(a, min(a + step, len(keep)), targets[a:min(a + step, len(keep))])
              for a in range(0, len(keep), step)]

    ledger = []
    best = (-1.0, None)
    for p in snaps:
        ck = torch.load(p, map_location=device, weights_only=True)
        model = encoder_from_checkpoint(ck).to(device).eval()
        model.load_state_dict(ck["model"])
        t0 = time.time()
        with torch.no_grad():
            for a in range(0, len(keep), 512):
                f = ft[a:a + 512]
                log_e = model(f, kt.expand(f.shape[0], -1, -1),
                              mt.expand(f.shape[0], -1))[0]
                em[a:a + f.shape[0]] = slice_head_torch(log_e, nl).cpu().numpy()
        h1 = sum(pool.map(_chunk, chunks))
        t1 = h1 / len(keep) * 100
        greedy = float(ck.get("val_greedy", float("nan")))
        print(f"{p.name}: beam t1 {t1:5.2f}  (train-logged greedy "
              f"{greedy * 100:.2f})  {time.time() - t0:.0f}s")
        ledger.append({"snap": p.name, "beam_t1": t1, "greedy": greedy})
        if t1 > best[0]:
            best = (t1, p)
    pool.close()
    pool.join()
    print(f"\nWINNER: {best[1].name} beam t1 {best[0]:.2f}")
    (run_dir / "ru_snapshot_selection.json").write_text(
        json.dumps({"ledger": ledger, "winner": best[1].name,
                    "winner_t1": best[0]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
