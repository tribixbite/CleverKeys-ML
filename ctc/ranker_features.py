#!/usr/bin/env python3
"""Candidate features for the Phase-K discriminative rescorer (K3).

One place computes the per-(trace, candidate) feature vector used by BOTH the
miner (``mine_candidates.py``) and the eval-time reranker (``eval_beam.py
--ranker-onnx``), so train and inference cannot drift.

The alignment features replay the exact state machine of
``futo_decoder_ceiling.futo_viterbi_beam`` for ONE word: states are
``(letters_consumed, blank_ended)``, transitions are blank / extend / repeat,
Viterbi-max merge — so ``forced_viterbi`` equals the raw path score the beam
would carry for that word, whether or not the beam kept it. ``ctc_forward``
runs the same machine with logsumexp (total alignment mass).

Feature vector (F = 14), in order:
  0  forced_viterbi / len^gamma      (the beam's length-normalized CTC term)
  1  forced_viterbi / T              (per-frame path quality)
  2  ctc_forward − forced_viterbi    (alignment-mass spread beyond the best path)
  3  len(word)
  4  log_freq                        (trie log-frequency, beam's λ term input)
  5  beam final score                (as produced by the beam at its preset)
  6  gap to the slate's rank-1 final score  (0 for rank 1)
  7  rank in the slate (0-based)
  8  min over letters of (max-frame letter log-emission)   (weakest evidence)
  9  mean over letters of (max-frame letter log-emission)
 10  mean blank log-emission over frames
 11  1.0 if len(word) <= 3 else 0.0  (the target stratum, explicit)
 12  forced_viterbi (raw)
 13  T (emission frames; 32 or 64 — carries the contract into the features)
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

NUM_FEATURES = 14
NEG_INF = -1e30


def word_log_freq(trie, word: str) -> float:
    """Walk the LexTrie for *word*'s stored log-frequency (NEG_INF if absent)."""
    node = trie.root
    for ch in word:
        node = node.children.get(ch)
        if node is None:
            return NEG_INF
    return node.log_freq if node.is_word else NEG_INF


def forced_scores(lp: np.ndarray, word: str, letters_index: dict,
                  blank_idx: int) -> Tuple[float, float]:
    """(viterbi_max, logsumexp_forward) of *word* through ``lp [T, K+1]``.

    Exact single-word replay of the beam's transition system:
      blank:  (i, be) -> (i, True)   + lp[t, blank]
      extend: (i, .)  -> (i+1, False)+ lp[t, word[i]]
      repeat: (i>0, False) -> (i, False) + lp[t, word[i-1]]
    """
    T = lp.shape[0]
    L = len(word)
    idx = np.fromiter((letters_index[c] for c in word), np.int64, L)
    # v[i, be] / f[i, be]
    v = np.full((L + 1, 2), NEG_INF)
    f = np.full((L + 1, 2), NEG_INF)
    v[0, 0] = f[0, 0] = 0.0
    for t in range(T):
        row = lp[t]
        blank = float(row[blank_idx])
        nv = np.full((L + 1, 2), NEG_INF)
        nf = np.full((L + 1, 2), NEG_INF)
        # blank transition (vectorized over i): to (i, True) from best of both be
        vb = np.maximum(v[:, 0], v[:, 1]) + blank
        fb = np.logaddexp(f[:, 0], f[:, 1]) + blank
        nv[:, 1] = vb
        nf[:, 1] = fb
        # extend: (i, be) -> (i+1, False)
        if L:
            step = row[idx]                                     # [L]
            ve = np.maximum(v[:L, 0], v[:L, 1]) + step
            fe = np.logaddexp(f[:L, 0], f[:L, 1]) + step
            # repeat: (i, False) -> (i, False), i >= 1
            vr = v[1:, 0] + step
            fr = f[1:, 0] + step
            nv[1:, 0] = np.maximum(ve, vr)
            nf[1:, 0] = np.logaddexp(fe, fr)
        v, f = nv, nf
    vit = float(max(v[L, 0], v[L, 1]))
    fwd = float(np.logaddexp(f[L, 0], f[L, 1]))
    return vit, fwd


def slate_features(lp: np.ndarray, slate: Sequence[Tuple[str, float]],
                   trie, letters: List[str], blank_idx: int,
                   gamma: float) -> np.ndarray:
    """Features for every candidate of one slate -> ``[len(slate), F]`` f32.

    *lp* is the SLICED log-emission matrix ``[T, num_letters+1]`` the beam ran
    on; *slate* is the beam output ``[(word, final_score), ...]`` in rank order;
    *gamma* is the decode preset's length exponent (feature 0 mirrors the
    beam's own normalization).
    """
    letters_index = {c: i for i, c in enumerate(letters)}
    T = lp.shape[0]
    blank_mean = float(lp[:, blank_idx].mean())
    max_per_class = lp.max(axis=0)                              # [K+1]
    top_score = slate[0][1] if slate else 0.0
    out = np.empty((len(slate), NUM_FEATURES), np.float32)
    for r, (word, score) in enumerate(slate):
        vit, fwd = forced_scores(lp, word, letters_index, blank_idx)
        L = max(len(word), 1)
        letter_max = np.fromiter((max_per_class[letters_index[c]] for c in word),
                                 np.float64, len(word))
        out[r] = (
            vit / (L ** gamma),
            vit / T,
            fwd - vit,
            float(len(word)),
            word_log_freq(trie, word),
            score,
            top_score - score,
            float(r),
            float(letter_max.min()) if len(word) else NEG_INF,
            float(letter_max.mean()) if len(word) else NEG_INF,
            blank_mean,
            1.0 if len(word) <= 3 else 0.0,
            vit,
            float(T),
        )
    return out
