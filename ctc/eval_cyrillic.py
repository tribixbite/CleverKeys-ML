#!/usr/bin/env python3
"""Cyrillic (ЙЦУКЕН) decode evaluation through the exported ONNX encoder.

The first non-Latin measurement of the campaign's CTC encoders.  Everything
runs through the same code path as `eval_altlayout.py` — `featurize`,
`futo_viterbi_beam`, `LexTrie`, `Tally` — with the two a-z assumptions
replaced at the call sites they live in:

* the emission gather takes columns `0..n_letters-1` of the exported `[32,65]`
  head plus the blank at `MAX_KEYS` (the model's emission column `c` is
  whatever key sits in slot `c` — the alphabet is layout-defined, so a
  31-letter layout simply uses 31 columns);
* the word projection is `prepare_yandex.project_ru` (lowercase, strip
  `-`/`'`, ё→е, ъ→ь; NO NFD — it would decompose й) applied to targets and
  lexicon identically, the ALT_LAYOUT_EVAL §3 policy.

Lexicons:
  ``app``  the app's bundled langpack-ru CKDT v2 (50 k, freq = 255 − rank —
           same magnitude scale as every other CKDT lexicon in the campaign);
  ``voc``  the Yandex competition vocabulary (503,598 words, no frequencies —
           every word gets freq 1, so the λ term is inert).

Protocols: in-dict (OOV excluded from the denominator, the geometric-engine
comparison convention) and all-rows, both reported.  test-2400 is not touched
— this file cannot even read it (English splits are not inputs here).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_altlayout import OnnxEncoder, read_ckdt_v2  # noqa: E402
from futo_decoder_ceiling import futo_viterbi_beam  # noqa: E402
from futo_decoder_eval import (LexTrie, Tally, featurize, greedy_ctc,  # noqa: E402
                               len_stratum, load_layout, rank_of)
from model import MAX_KEYS  # noqa: E402
from paths import DEFAULT_WORKDIR, resolve  # noqa: E402
from prepare_yandex import project_ru  # noqa: E402

HERE = Path(__file__).resolve().parent
E1 = (1.05, 1.1, 0.2, 0.3734, 0.9882)
RU_ZIP = Path("/home/will/git/swype/CleverKeys/scripts/dictionaries/langpack-ru.zip")


def build_trie(kind: str, yandex_dir: Path) -> Tuple[LexTrie, Dict[str, int]]:
    trie = LexTrie()
    st = {"records": 0, "kept": 0}
    if kind == "app":
        with zipfile.ZipFile(RU_ZIP) as zf:
            blob = zf.read("dictionary.bin")
        tmp = Path("/tmp/ru_dictionary_ckdt_eval.bin")
        tmp.write_bytes(blob)
        for word, rank in read_ckdt_v2(tmp):
            st["records"] += 1
            p = project_ru(word)
            if p is None:
                continue
            trie.insert(p, float(max(1, 255 - rank)))
            st["kept"] += 1
    elif kind == "voc":
        for line in open(yandex_dir / "voc.txt", encoding="utf-8"):
            st["records"] += 1
            p = project_ru(line.strip())
            if p is None:
                continue
            trie.insert(p, 1.0)
            st["kept"] += 1
    else:
        raise SystemExit(f"unknown lexicon {kind!r}")
    st["distinct"] = trie.num_words
    return trie, st


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--val", type=Path, default=None,
                    help="canonical ru jsonl (default data/yandex_val10k.jsonl)")
    ap.add_argument("--layout", default="ru_jcuken_default.json")
    ap.add_argument("--lexicon", default="app", choices=("app", "voc"))
    ap.add_argument("--yandex-dir", type=Path,
                    default=Path.home() / "ctc-train" / "data" / "yandex_cup")
    ap.add_argument("--preset", default=",".join(str(v) for v in E1))
    ap.add_argument("--beam-width", type=int, default=100)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--progress", type=int, default=1000)
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()

    preset = tuple(float(v) for v in args.preset.split(","))
    gamma, lam_w, beta, gamma_prune, beta_prune = preset
    val = args.val or resolve(args.workdir, Path("data")) / "yandex_val10k.jsonl"

    letters, centers = load_layout(HERE / "layouts" / args.layout)
    n_letters = len(letters)
    keys = np.zeros((1, MAX_KEYS, 2), np.float32)
    keys[0, :n_letters] = centers
    mask = np.zeros((1, MAX_KEYS), bool)
    mask[0, :n_letters] = True

    trie, lex_st = build_trie(args.lexicon, args.yandex_dir)
    print(f"lexicon {args.lexicon}: {lex_st}")

    enc = OnnxEncoder(args.onnx)
    g_tal, b_tal = Tally(), Tally()
    strat = {"<=3": Tally(), "4+": Tally()}
    oov = unproj = n_seen = 0
    t0 = time.time()
    all_rows_misses = 0          # OOV+unprojectable count as top-1 misses here
    with open(val, encoding="utf-8") as f:
        for line in f:
            if args.limit and n_seen >= args.limit:
                break
            o = json.loads(line)
            n_seen += 1
            target = project_ru(o["word"])
            if target is None:
                unproj += 1
                all_rows_misses += 1
                continue
            if not trie.contains(target):
                oov += 1
                all_rows_misses += 1
                continue
            pts = o["points"]
            feats = featurize([p["x"] for p in pts], [p["y"] for p in pts],
                              [p["t"] for p in pts])
            full = enc.forward(feats, keys, mask)                # [32, 65]
            lp = np.empty((full.shape[0], n_letters + 1), np.float32)
            lp[:, :n_letters] = full[:, :n_letters]
            lp[:, n_letters] = full[:, MAX_KEYS]
            greedy = greedy_ctc(lp, letters, n_letters)
            beam = futo_viterbi_beam(lp, letters, n_letters, trie,
                                     args.beam_width, args.top_k, gamma, lam_w,
                                     beta, gamma_prune, beta_prune)
            words = [w for w, _ in beam]
            g_tal.add(0 if greedy == target else -1)
            r = rank_of(target, words)
            b_tal.add(r)
            strat[len_stratum(target)].add(r)
            if args.progress and b_tal.n % args.progress == 0:
                print(f"  {b_tal.n} decoded  t1 {b_tal.t1 / b_tal.n * 100:.2f} "
                      f"t3 {b_tal.t3 / b_tal.n * 100:.2f} "
                      f"t5 {b_tal.t5 / b_tal.n * 100:.2f}  greedy "
                      f"{g_tal.t1 / g_tal.n * 100:.2f}  "
                      f"({b_tal.n / (time.time() - t0):.1f} tr/s)", flush=True)

    res = {
        "onnx": str(args.onnx), "layout": args.layout, "lexicon": args.lexicon,
        "preset": preset, "rows": n_seen, "decoded": b_tal.n,
        "unprojectable": unproj, "oov": oov,
        "greedy_t1": round(g_tal.t1 / max(g_tal.n, 1) * 100, 2),
        "indict_t1": round(b_tal.t1 / max(b_tal.n, 1) * 100, 2),
        "indict_t3": round(b_tal.t3 / max(b_tal.n, 1) * 100, 2),
        "indict_t5": round(b_tal.t5 / max(b_tal.n, 1) * 100, 2),
        "allrows_t1": round(b_tal.t1 / max(n_seen, 1) * 100, 2),
        "allrows_t3": round(b_tal.t3 / max(n_seen, 1) * 100, 2),
        "allrows_t5": round(b_tal.t5 / max(n_seen, 1) * 100, 2),
        "le3_n": strat["<=3"].n,
        "le3_t1": round(strat["<=3"].t1 / max(strat["<=3"].n, 1) * 100, 2),
        "ge4_n": strat["4+"].n,
        "ge4_t1": round(strat["4+"].t1 / max(strat["4+"].n, 1) * 100, 2),
        "seconds": round(time.time() - t0, 1),
    }
    print(json.dumps(res, indent=1))
    if args.out_json:
        args.out_json.write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
