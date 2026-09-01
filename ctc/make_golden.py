#!/usr/bin/env python3
"""Freeze model-backed golden cases for the Kotlin parity test.

Synthetic paths -> features -> ONNX emissions -> harness greedy + beam. The
output matches the ``ctc_golden.json`` schema that CleverKeys' ``CtcParityTest``
reads — both its ``"featurize"`` cases (pure resampler branch probes, asserted
bit-identical) and its ``"beam"`` cases — plus the raw
``points -> features -> emissions`` pairs an ONNX-backed ``CtcEmissionModel``
parity test needs.

Emissions are stored as the sliced ``[32,27]`` contract view (blank relocated
from full-head column 64 to column 26), which is exactly what
``CtcEmissions.sliceFromHead`` hands ``CtcBeamDecoder``.

Audit fix #16: --workdir pathing; layout defaults to the script's directory.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent))
from futo_decoder_eval import featurize, greedy_ctc, load_layout, LexTrie  # noqa: E402
from futo_decoder_ceiling import (ENC_BETA, ENC_BETA_PRUNE, ENC_GAMMA,  # noqa: E402
                                  ENC_GAMMA_PRUNE, ENC_LAMBDA, futo_viterbi_beam,
                                  slice_emissions)
from model import MAX_KEYS, T_OUT  # noqa: E402
from paths import (DEFAULT_LAYOUT, DEFAULT_WORKDIR, resolve,  # noqa: E402
                   sha256_file)

LEXICON = [("cat", 150.0), ("car", 180.0), ("cart", 120.0), ("care", 140.0),
           ("the", 250.0), ("hello", 160.0), ("keyboard", 110.0)]
WORDS = ["cat", "the", "hello", "keyboard"]

#: Per-script fixture vocabularies. Each is ``(lexicon, decoded words)`` and each
#: mirrors the ``en`` structure exactly — a four-member shared-prefix family so
#: the beam's trie branching is actually exercised, the script's highest-frequency
#: short word, a greeting, and one long word — so a new script's fixture probes
#: the same decode behaviours the English one does.
#:
#: ``ru`` frequencies are the app's own CKDT scale (``255 - rank`` from
#: ``langpack-ru.zip``'s ``dictionary.bin``, the identical formula
#: ``eval_cyrillic.build_trie`` uses), NOT invented weights: the λ term is scale
#: sensitive (PHASE_J §6.9 — the CKDT scale is exactly why ru wants λ = 2.0 and
#: not E1's 1.1), so a fixture on a different frequency scale would validate a
#: ranking the app never performs.
VOCABS = {
    "en": (LEXICON, WORDS),
    "ru": ([("кот", 137.0), ("кота", 122.0), ("коты", 96.0), ("котик", 93.0),
            ("что", 249.0), ("привет", 169.0), ("клавиатура", 88.0)],
           ["кот", "что", "привет", "клавиатура"]),
    # ── Phase O. Weights below are read out of each script's real lexicon at
    # the same 255 - rank CKDT scale (script_registry.ScriptSpec.load_lexicon),
    # so the fixture ranks words exactly as the shipped decode would. For the
    # wordfreq-sourced scripts (uk/bg/mk/he) the rank comes from the app's own
    # build_dictionary.frequency_to_rank, replicated byte-exactly.
    #
    # el: note the family members end in ς, not σ. That is the whole point of
    # the final-sigma repair (PHASE_O.md §1.3/§2.2) — a σ-final fixture would
    # freeze a decode against the wrong key, in the wrong row.
    "el": ([("γεια", 157.0), ("και", 255.0), ("καινουρια", 134.0),
            ("καιρο", 167.0), ("καιρος", 156.0), ("υπολογιστης", 121.0)],
           ["καιρος", "και", "γεια", "υπολογιστης"]),
    "uk": ([("клавіатура", 90.0), ("на", 254.0), ("привіт", 162.0),
            ("про", 230.0), ("просто", 209.0), ("проти", 195.0),
            ("протягом", 179.0)],
           ["про", "на", "привіт", "клавіатура"]),
    "bg": ([("здравей", 151.0), ("клавиатура", 105.0), ("на", 255.0),
            ("това", 218.0), ("товар", 118.0), ("товара", 105.0),
            ("товари", 123.0)],
           ["това", "на", "здравей", "клавиатура"]),
    "mk": ([("дека", 225.0), ("декада", 104.0), ("декември", 161.0),
            ("декрет", 110.0), ("здраво", 155.0), ("на", 255.0),
            ("тастатура", 94.0)],
           ["дека", "на", "здраво", "тастатура"]),
    # he: authored in logical (memory) order, which is what the decoder and the
    # trie see. The app has no RTL branch anywhere in the geometry or swipe path
    # (APP_INTEGRATION_AUDIT §5), so display direction never enters the fixture.
    "he": ([("את", 255.0), ("הוא", 238.0), ("הואיל", 123.0), ("הוארך", 104.0),
            ("הואשם", 112.0), ("מקלדת", 115.0), ("שלום", 187.0)],
           ["הוא", "את", "שלום", "מקלדת"]),
}
BEAM_WIDTH = 32
TOP_K = 4

#: Pure-featurizer parity cases (no model involved): each exercises a distinct
#: resampler branch of ``featurize`` so the Kotlin ``CtcFeaturizer`` port can be
#: asserted bit-identical. ``CtcParityTest.featurizer_matchesPythonPort_bitIdentical``
#: requires at least one ``"kind": "featurize"`` case in the fixture, which the
#: model-backed beam cases alone do not provide.
FEATURIZE_CASES = [
    # n == 1: both resample stages take their single-point degenerate branch.
    ("feat_single_point", [0.42], [0.31], [0.0]),
    # duration <= EPS: stage 1 returns the first/last point pair unchanged.
    ("feat_zero_duration", [0.1, 0.2, 0.3], [0.5, 0.5, 0.5], [10.0, 10.0, 10.0]),
    # Two points over 2 s: stage 1 emits ~121 samples (round-half-even path).
    ("feat_two_point_long", [0.05, 0.95], [0.9, 0.1], [0.0, 2000.0]),
    # Irregular timestamps: lower-bound segment lerp with unequal segment lengths.
    ("feat_nonuniform_t",
     [0.1, 0.3, 0.35, 0.8], [0.2, 0.6, 0.61, 0.4], [0.0, 12.0, 13.5, 200.0]),
    # Out-of-frame inputs: stage 2's [0,1] clamp is the only clamp in the chain.
    ("feat_clamp_out_of_range",
     [-0.2, 0.5, 1.3], [1.2, 0.5, -0.1], [0.0, 40.0, 90.0]),
]


def ideal_path(by_letter, word: str, pts_per_seg: int = 12
               ) -> Tuple[List[float], List[float], List[float]]:
    """Deterministic straight-line path through the word's key centers, 60 Hz stamps."""
    cs = [by_letter[c] for c in word]
    xs: List[float] = []
    ys: List[float] = []
    for a, b in zip(cs[:-1], cs[1:]):
        for j in range(pts_per_seg):
            f = j / pts_per_seg
            xs.append(float(a[0] + f * (b[0] - a[0])))
            ys.append(float(a[1] + f * (b[1] - a[1])))
    xs.append(float(cs[-1][0]))
    ys.append(float(cs[-1][1]))
    ts = [i * (1000.0 / 60.0) for i in range(len(xs))]
    return xs, ys, ts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    ap.add_argument("--layout", type=Path, default=DEFAULT_LAYOUT)
    ap.add_argument("--onnx", default="ctc_swipe_encoder.onnx",
                    help="exported encoder; comma-separate several to freeze "
                         "an emission-averaging ensemble fixture (Phase K)")
    ap.add_argument("--ens-avg", choices=["logprob", "prob"], default="prob",
                    dest="ens_avg",
                    help="averaging mode when --onnx lists >1 model")
    ap.add_argument("--out", default="ctc_model_golden.json")
    ap.add_argument("--vocab", default="en", choices=sorted(VOCABS),
                    help="which script's fixture vocabulary to freeze (see "
                         "VOCABS). Must match --layout: the ideal paths are "
                         "traced through the layout's own key centers, so a word "
                         "with a letter the layout lacks is a KeyError. Default "
                         "'en' keeps every pre-existing fixture byte-identical.")
    ap.add_argument("--preset", default="",
                    help="scoring preset the beam cases are generated at, as "
                         "'gamma,lambda,beta,gammaPrune,betaPrune'. Defaults to the "
                         "published encoderOnly preset. The fixture must be "
                         "generated at whatever preset the app ships in "
                         "CtcScoringParams, or the parity test asserts against a "
                         "configuration that is never run.")
    args = ap.parse_args()

    gamma, lam, beta, gamma_prune, beta_prune = (
        ENC_GAMMA, ENC_LAMBDA, ENC_BETA, ENC_GAMMA_PRUNE, ENC_BETA_PRUNE)
    if args.preset:
        vals = [float(v) for v in args.preset.split(",")]
        if len(vals) != 5:
            raise SystemExit("--preset needs 5 floats: "
                             "gamma,lambda,beta,gammaPrune,betaPrune")
        gamma, lam, beta, gamma_prune, beta_prune = vals
    print(f"beam cases at gamma={gamma} lambda={lam} beta={beta} "
          f"gammaPrune={gamma_prune} betaPrune={beta_prune}")

    letters, centers = load_layout(args.layout)
    num_letters = len(letters)
    by_letter = {l: centers[i] for i, l in enumerate(letters)}
    # Phase K: comma-separated --onnx freezes an emission-averaging ensemble's
    # fixture — the emissions stored are the AVERAGED ones the app-side dual
    # session must reproduce (see eval_beam.ensemble_average).
    from eval_beam import ensemble_average
    sessions = [ort.InferenceSession(str(resolve(args.workdir, p)),
                                     providers=["CPUExecutionProvider"])
                for p in str(args.onnx).split(",")]
    ens_avg = getattr(args, "ens_avg", "prob")

    lexicon, words = VOCABS[args.vocab]
    missing = sorted({c for w, _ in lexicon for c in w} - set(letters))
    if missing:
        raise SystemExit(f"--vocab {args.vocab} needs letters {missing!r} that "
                         f"{args.layout.name} does not have")
    trie = LexTrie()
    for w, f in lexicon:
        trie.insert(w, f)

    keys = np.zeros((MAX_KEYS, 2), np.float32)
    keys[:num_letters] = centers
    mask = np.zeros((MAX_KEYS,), bool)
    mask[:num_letters] = True

    cases = []
    # Featurizer-only cases first: the fixed synthetic branch probes, plus one
    # realistic multi-segment word path built from the layout's own key centers.
    feat_cases = list(FEATURIZE_CASES)
    fx, fy, ft = ideal_path(by_letter, words[0])
    feat_cases.append((f"feat_word_path_{words[0]}", fx, fy, ft))
    for name, xs, ys, ts in feat_cases:
        feats = featurize(xs, ys, ts)                                 # [2,64]
        cases.append({
            "kind": "featurize", "name": name,
            "points": {"x": [float(v) for v in xs], "y": [float(v) for v in ys],
                       "t": [float(v) for v in ts]},
            "features": [float(v) for v in feats.reshape(-1)],        # [128]
        })
    for word in words:
        xs, ys, ts = ideal_path(by_letter, word)
        feats = featurize(xs, ys, ts)                                 # [2,64]
        feed = {"features": feats[None], "layout_keys": keys[None],
                "layout_mask": mask[None]}
        fulls = [s.run(["log_emissions"], feed)[0][0] for s in sessions]
        full = fulls[0] if len(fulls) == 1 else \
            ensemble_average(fulls, ens_avg)                          # [T',65]
        lp = slice_emissions(full, num_letters, MAX_KEYS)             # [T',27]
        greedy = greedy_ctc(lp, letters, num_letters)
        topk = futo_viterbi_beam(lp, letters, num_letters, trie, BEAM_WIDTH, TOP_K,
                                 gamma, lam, beta, gamma_prune, beta_prune)
        cases.append({
            "kind": "beam", "name": f"model_{word}",
            "alphabet": "".join(letters), "frames": int(lp.shape[0]),
            "numClasses": num_letters + 1,
            "points": {"x": xs, "y": ys, "t": ts},
            "features": [float(v) for v in feats.reshape(-1)],        # [128] x-row then y-row
            "emissions": [[float(v) for v in row] for row in lp],
            "lexicon": [[w, f] for w, f in lexicon],
            "params": {"gamma": gamma, "lambda": lam, "beta": beta,
                       "alpha": 0.0, "gammaPrune": gamma_prune,
                       "betaPrune": beta_prune, "beamWidth": BEAM_WIDTH,
                       "topK": TOP_K},
            "greedy": greedy,
            "topk": [[w, s] for w, s in topk],
        })
    out_path = resolve(args.workdir, args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        # LOW-6 / ARC-061: basenames only — the fixture ships into the app repo,
        # and an absolute path would leak the generating host's home directory
        # (both shipped ctc_golden.json copies carried /home/will/…). Identity
        # is pinned by source_onnx_sha256; the basename is enough provenance.
        {"source_onnx": ",".join(Path(p).name for p in str(args.onnx).split(",")),
         "source_onnx_sha256": [sha256_file(resolve(args.workdir, p))
                                for p in str(args.onnx).split(",")],
         "preset": [gamma, lam, beta, gamma_prune, beta_prune],
         # The exact layout tensors the emissions were generated against, so an
         # app-side model-backed parity test can rebuild layout_keys/layout_mask
         # (CtcLayout.of + buildPaddedLayout) without carrying en_qwerty.json.
         "layout": {"letters": "".join(letters),
                    "cx": [float(c[0]) for c in centers],
                    "cy": [float(c[1]) for c in centers]},
         "cases": cases}, indent=1))
    print(f"wrote {out_path} ({len(cases)} cases)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
