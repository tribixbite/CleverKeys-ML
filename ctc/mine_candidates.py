#!/usr/bin/env python3
"""Phase-K3 miner: run the campaign beam over TRAIN-set emissions to harvest
(trace, gold, confusable-candidates) slates for the discriminative rescorer.

License hygiene: the slates are SELF-MINED — our own encoder's emissions on our
own training rows, decoded by our own beam port. No FUTO decoder output is read
or stored (the FUTO corpus rows themselves are MIT).

Selection: ALL rows whose gold word is short (``len <= --short-max``) are kept
(the ≤3 stratum is the K3 target and is a minority of the mix); longer-word rows
are subsampled at ``--long-frac``; a global ``--max-rows`` cap applies last.
Selection is deterministic in ``--seed``.

Output: npz shards of ``--shard-size`` rows with, per row: gold word, sliced
emissions (fp16, ``[T', num_letters+1]``), the top-k beam slate (words + final
scores at the E1 preset), the gold's rank in it (-1 = absent), and the
``ranker_features.slate_features`` matrix — computed HERE, by the same module
eval-time reranking uses.

Usage:
  python mine_candidates.py --ckpt ckpt/phaseJ-sw2345/best.pt \
      --npz train_t3.npz,train_t3hws.npz,tier_sw234.npz,tier_sw5q.npz \
      --out-prefix phaseK/mined_sw2345
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from futo_decoder_ceiling import futo_viterbi_beam, slice_emissions  # noqa: E402
from futo_decoder_eval import load_layout  # noqa: E402
import lexicon  # noqa: E402
from model import MAX_KEYS, encoder_from_checkpoint  # noqa: E402
from paths import DEFAULT_LAYOUT, DEFAULT_WORKDIR, resolve  # noqa: E402
from ranker_features import NUM_FEATURES, slate_features  # noqa: E402

#: The E1 scoring preset — identical to the eval battery (phaseJ_eval.sh).
E1 = (1.05, 1.1, 0.2, 0.3734, 0.9882)

_G = {}


def _init_worker(vocab_path: str, vocab_kind: str, layout_path: str,
                 beam_width: int, top_k: int, preset) -> None:
    letters, _ = load_layout(Path(layout_path))
    _G["letters"] = letters
    _G["trie"] = lexicon.load_vocab(Path(vocab_path), vocab_kind)
    _G["beam_width"] = beam_width
    _G["top_k"] = top_k
    _G["preset"] = preset


def _decode_chunk(args):
    """(idx_array, lp_fp16 [n,T,K+1], words [n]) -> per-row slate records."""
    idxs, lps, words = args
    letters, trie = _G["letters"], _G["trie"]
    num_letters = len(letters)
    gamma, lam_w, beta, gp, bp = _G["preset"]
    out = []
    for i in range(len(idxs)):
        lp = lps[i].astype(np.float32)
        slate = futo_viterbi_beam(lp, letters, num_letters, trie,
                                  _G["beam_width"], _G["top_k"],
                                  gamma, lam_w, beta, gp, bp)
        feats = slate_features(lp, slate, trie, letters, num_letters, gamma)
        gold = words[i]
        rank = next((r for r, (w, _) in enumerate(slate) if w == gold), -1)
        out.append((idxs[i], [w for w, _ in slate],
                    [s for _, s in slate], rank, feats))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    ap.add_argument("--layout", type=Path, default=DEFAULT_LAYOUT)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--npz", required=True,
                    help="comma-separated cache npz names (resolved in "
                         "<workdir>/cache)")
    ap.add_argument("--vocab", default="data/futo_en_wordlist.combined")
    lexicon.add_argument(ap)
    ap.add_argument("--out-prefix", required=True, dest="out_prefix",
                    help="shard path prefix under workdir, e.g. phaseK/mined_sw2345")
    ap.add_argument("--short-max", type=int, default=4, dest="short_max",
                    help="keep ALL rows with len(gold) <= this")
    ap.add_argument("--long-frac", type=float, default=0.35, dest="long_frac",
                    help="subsample fraction for longer-word rows")
    ap.add_argument("--max-rows", type=int, default=600000, dest="max_rows")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--beam-width", type=int, default=100, dest="beam_width")
    ap.add_argument("--top-k", type=int, default=8, dest="top_k")
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=1024, help="GPU forward batch")
    ap.add_argument("--shard-size", type=int, default=100000, dest="shard_size")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    letters, centers = load_layout(args.layout)
    num_letters = len(letters)

    # ── gather + select rows ────────────────────────────────────────────────
    feats_all, words_all, src_all = [], [], []
    for name in args.npz.split(","):
        p = args.workdir / "cache" / name
        d = np.load(p, allow_pickle=True)
        F, W = d["features"], d["words"]
        wl = np.char.str_len(W)
        keep = (wl <= args.short_max) | (rng.random(len(W)) < args.long_frac)
        feats_all.append(F[keep])
        words_all.append(W[keep])
        src_all.append(np.full(int(keep.sum()), name.replace(".npz", ""), "<U24"))
        print(f"{name}: {len(W)} rows -> kept {int(keep.sum())} "
              f"({int((wl <= args.short_max).sum())} short)", flush=True)
    feats = np.concatenate(feats_all)
    words = np.concatenate(words_all)
    srcs = np.concatenate(src_all)
    if len(words) > args.max_rows:
        sel = rng.choice(len(words), args.max_rows, replace=False)
        sel.sort()
        feats, words, srcs = feats[sel], words[sel], srcs[sel]
    n = len(words)
    print(f"mining {n} rows total", flush=True)

    # ── GPU emissions ───────────────────────────────────────────────────────
    ck = torch.load(resolve(args.workdir, args.ckpt), map_location="cpu",
                    weights_only=True)
    model = encoder_from_checkpoint(ck).eval().to(args.device)
    model.load_state_dict(ck["model"])
    kp = np.zeros((MAX_KEYS, 2), np.float32)
    kp[:num_letters] = centers
    mp_mask = np.zeros((MAX_KEYS,), bool)
    mp_mask[:num_letters] = True
    t_out = getattr(model, "t_out", 32)
    lp_store = np.empty((n, t_out, num_letters + 1), np.float16)
    t0 = time.time()
    with torch.no_grad():
        keys_t = torch.from_numpy(kp[None]).to(args.device)
        mask_t = torch.from_numpy(mp_mask[None]).to(args.device)
        for i in range(0, n, args.batch):
            fb = torch.from_numpy(feats[i:i + args.batch]).to(args.device)
            b = fb.shape[0]
            log_e, _, _ = model(fb, keys_t.expand(b, -1, -1),
                                mask_t.expand(b, -1))
            full = log_e.float().cpu().numpy()               # [b, T', 65]
            for j in range(b):
                lp_store[i + j] = slice_emissions(full[j], num_letters,
                                                  MAX_KEYS).astype(np.float16)
            if (i // args.batch) % 50 == 0:
                print(f"  emissions {i}/{n} ({(i + b) / (time.time() - t0):.0f}/s)",
                      flush=True)
    print(f"emissions done in {time.time() - t0:.0f}s", flush=True)

    # ── parallel beam + features, shard by shard ────────────────────────────
    out_dir = (args.workdir / args.out_prefix).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = str(resolve(args.workdir, args.vocab))
    chunk = 200
    t0 = time.time()
    done = 0
    with mp.Pool(args.jobs, initializer=_init_worker,
                 initargs=(vocab_path, args.vocab_kind, str(args.layout),
                           args.beam_width, args.top_k, E1)) as pool:
        for s0 in range(0, n, args.shard_size):
            s1 = min(s0 + args.shard_size, n)
            m = s1 - s0
            sw = np.zeros((m, args.top_k), "<U27")
            ss = np.full((m, args.top_k), -1e30, np.float32)
            sl = np.zeros((m,), np.uint8)
            gr = np.full((m,), -1, np.int8)
            fx = np.zeros((m, args.top_k, NUM_FEATURES), np.float32)
            tasks = [(np.arange(c0, min(c0 + chunk, s1)),
                      lp_store[c0:min(c0 + chunk, s1)],
                      words[c0:min(c0 + chunk, s1)])
                     for c0 in range(s0, s1, chunk)]
            for res in pool.imap_unordered(_decode_chunk, tasks):
                for idx, ws, scs, rank, ft in res:
                    r = idx - s0
                    k = len(ws)
                    sw[r, :k] = ws
                    ss[r, :k] = scs
                    sl[r] = k
                    gr[r] = rank
                    fx[r, :k] = ft
                    done += 1
                if done % 20000 < chunk:
                    print(f"  beam {done}/{n} ({done / (time.time() - t0):.0f}/s)",
                          flush=True)
            shard = args.workdir / f"{args.out_prefix}_{s0:07d}.npz"
            np.savez_compressed(
                shard, words=words[s0:s1], srcs=srcs[s0:s1],
                emissions=lp_store[s0:s1], slate_words=sw, slate_scores=ss,
                slate_len=sl, gold_rank=gr, features=fx,
                meta=np.array([f"ckpt={args.ckpt} preset=E1 "
                               f"width={args.beam_width} k={args.top_k} "
                               f"seed={args.seed} short_max={args.short_max} "
                               f"long_frac={args.long_frac}"]))
            print(f"wrote {shard} ({m} rows)", flush=True)
    print(f"done: {n} rows in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
