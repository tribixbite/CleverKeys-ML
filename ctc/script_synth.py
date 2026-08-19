#!/usr/bin/env python3
"""Residual-transplant synthesis for ANY script — the Phase-O generalization of
``cyrillic_synth.py``.

The counterfactual is unchanged: **a script with no swipe corpus at all.** Real
human deviation (undershoot, corner-cutting, jitter — the tangent/normal
residual decomposition ``layout_aug.warp_path`` computes) is sampled from
ENGLISH traces and re-anchored onto the ideal polylines of the target script's
words on the target script's own layout.  ``cyrillic_synth.py`` is left in place
as the historical record of the ru run; this file is the same mechanism with the
three ru-specific facts (alphabet, projection, lexicon) lifted into
:mod:`script_registry`.

What is new here, and why
-------------------------
Phase I-B could validate its synthesis against a real Yandex corpus.  No other
script has one, so Phase O has to build its own honest holdout — and a holdout
drawn from the same generator is worth nothing unless it is disjoint in the
things that could leak.  Two disjointness rules are enforced:

* **Donor split.** The English donor pool is partitioned by a deterministic hash
  of the donor row index; ``--donor-side train`` and ``--donor-side holdout``
  draw from disjoint halves.  A holdout trace therefore carries motor noise from
  a human trace the training set never saw.
* **Independent draws.** Separate RNG seeds for train / selection-val / holdout,
  so the word draws are independent too.

What that holdout can and cannot establish is stated once, here, and repeated in
every table it feeds: it measures **generalization to fresh samples of this
generator**, not to real human swipes in the target script.  The size of *that*
gap is knowable for exactly one script — Russian, which has both a synthesis
holdout and a real corpus — and Phase O measures it there and carries it as the
calibration for everything else (PHASE_O.md §2.1).

Generator v2 (Phase P)
----------------------
``SYNTH_V2_DESIGN.md`` measured what v1 gets wrong and
``SYNTH_V2_RESEARCH_AUDIT.md`` built and scored the proposed repairs before any
of them were adopted; where the two disagree the audit is authoritative.  The
v2 pipeline, stage by stage, with the flag letter that switches each one:

===== ===========================================================  ====
stage what it does                                                 flag
===== ===========================================================  ====
S0    lexicon load (unchanged) + **wordfreq token mass** as the      ``a``
      draw weight instead of the compressed CKDT ``255 − rank``.
      The mass is accumulated over every wordfreq token that
      *projects* to a lexicon entry, so accent-stripping and the
      Greek final-sigma repair do not silently zero a word.
S1    donor index by ``(vertex_count, ⌊log1.25 L_polyline⌋)``,      ``c``/``d``
      plus a per-contributor sub-index over the donor bank's
      ``group`` column.
S2    word ~ token mass; donor = argmin over ``k = 16`` reservoir   ``c``/``d``
      candidates of ``Σ_seg |log(L_dst/L_src)|``, optionally
      restricted to one contributor for a block of rows.
S3    ``layout_aug.warp_path`` — UNCHANGED, every Phase-H
      invariant intact.
S4    **vertex-aligned per-segment re-timing** at ``alpha = 0.5``   ``b``
      (``layout_aug.retime_segment``).  Replaces the design's
      global-arc-fraction form, which is measurably worse on
      stroke count, speed autocorrelation and the gate.
S5    **acquisition-bandwidth matching**: a target duration        ``s5``
      predicted from the donor's own duration and the target/donor
      ideal-length ratio, with the exponent fit on MIT English
      only, then a re-featurization through the real 60 Hz chain.
S6    clip + write, plus per-row provenance (donor row, donor
      contributor, drawn duration).
===== ===========================================================  ====

``--generator v1`` (equivalently ``--stages ''``) reproduces the Phase-O
mechanism **bit-exactly**, including its RNG call sequence, so the paired
ablation is a true control rather than a re-implementation.
``--selftest-v1 <npz>`` asserts it against an existing cache.

Outputs (under ``--workdir``, default ``~/ctc-train``)::

  cache_<code>/train_synth.npz   [N,2,64] features + targets   (--rows, seed 1234)
  cache_<code>/val.npz           5,000 rows for greedy checkpoint selection (seed 999)
  cache_<code>/holdout.npz       10,000 rows for the final decode read  (seed 777)
  cache_<code>/synth_stats.json  generation + endpoint-proximity + control
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import script_registry as SR  # noqa: E402
from futo_decoder_eval import load_layout  # noqa: E402
from layout_aug import resample_bandwidth, retime_segment, warp_path  # noqa: E402
from paths import DEFAULT_WORKDIR, resolve  # noqa: E402
from prepare_data import T_OUT, adjacent_repeats  # noqa: E402

HERE = Path(__file__).resolve().parent

#: Fraction of the English donor pool reserved for holdout synthesis.
DONOR_HOLDOUT_FRAC = 0.1

#: Seeds, one per split — independent word draws by construction.
SEEDS = {"train": 1234, "val": 999, "holdout": 777}

#: Cross-segment sample-budget exponent for S4.  Not a tuned parameter: the
#: within-trace regression of per-segment time share on per-segment ideal-length
#: share reads 0.460 (HWS) / 0.493 (FUTO) / 0.447 (real ru) — an isochrony-type
#: invariant that transfers across scripts, so it is fit on MIT English and only
#: *checked* on Russian (`synth_retime_probe.py --stage law`).
RETIME_ALPHA = 0.5

#: Fix C reservoir size (SYNTH_V2_RESEARCH_AUDIT §1.3: k = 16 takes the
#: per-segment dst/src length-ratio p95 from 3.63 to 1.72).
K_CANDIDATES = 16

#: Fix D contributor-block length range, in rows.
GROUP_BLOCK = (50, 200)

#: The default v2 stage set.  ``d`` is implemented but OFF by default: the audit
#: (§1.4) shows it can neither pass nor fail the acceptance criteria — the CTC
#: encoder consumes single traces, so per-contributor coherence has no
#: first-order channel to the loss — while it does cost donor diversity inside a
#: block.  Leaving it off keeps every measured v2 delta attributable to A/C/B′/S5.
DEFAULT_STAGES = "a,c,b,s5"


def collapse(seq: np.ndarray) -> np.ndarray:
    """Adjacent-repeat collapse — same rule as ``layout_aug._polyline_letters``."""
    if len(seq) <= 1:
        return seq
    keep = np.ones(len(seq), bool)
    keep[1:] = seq[1:] != seq[:-1]
    return seq[keep]


@dataclass
class DonorBank:
    """One side of the English donor pool, indexed for every v2 draw policy."""

    feats: np.ndarray                       # [N,2,64] float32
    seqs: List[np.ndarray]                  # collapsed a-z polyline vertices
    by_count: Dict[int, np.ndarray]         # vertex count -> donor rows
    duration_ms: np.ndarray                 # [N] float32, 0 when unknown
    group: np.ndarray                       # [N] int32, -1 when unknown
    #: Per-donor ideal-polyline segment lengths on QWERTY (fix C's cost term).
    seg_len: List[np.ndarray]
    #: contributor -> vertex count -> donor rows (fix D).
    by_group_count: Dict[int, Dict[int, np.ndarray]] = field(default_factory=dict)
    #: The S5 duration law, fit on this bank (MIT English) — see `fit_duration_law`.
    dur_exponent: float = 0.0
    dur_fit: Dict[str, float] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.feats)


def build_donor_index(npz_paths: Sequence[Path], side: str,
                      qwerty: Optional[np.ndarray] = None,
                      need_groups: bool = False) -> DonorBank:
    """English donors -> a :class:`DonorBank` for *side*.

    *side* is ``"train"``, ``"holdout"`` or ``"all"``.  The split is a
    deterministic stride over the global donor row index, so it does not depend
    on load order beyond the fixed ``--donors`` list, and the two sides are
    exactly complementary.

    ``duration_ms``/``group`` come from the Phase-P donor-bank columns; a cache
    predating them yields zeros/``-1`` and the S5 and D stages refuse to run
    rather than silently degrading.
    """
    stride = int(round(1.0 / DONOR_HOLDOUT_FRAC))
    feats_all: List[np.ndarray] = []
    dur_all: List[np.ndarray] = []
    grp_all: List[np.ndarray] = []
    seqs: List[np.ndarray] = []
    by_count: Dict[int, List[int]] = defaultdict(list)
    kept = 0
    row = 0
    for p in npz_paths:
        with np.load(p) as d:
            f = np.array(d["features"])
            words = [str(w) for w in d["words"]]
            dur = (np.asarray(d["duration_ms"], np.float32) if "duration_ms" in d
                   else np.zeros(len(words), np.float32))
            grp = (np.asarray(d["group"], np.int32) if "group" in d
                   else np.full(len(words), -1, np.int32))
        keep_mask = np.zeros(len(words), bool)
        for i in range(len(words)):
            is_holdout = (row + i) % stride == 0
            keep_mask[i] = is_holdout if side == "holdout" else (
                True if side == "all" else not is_holdout)
        feats_all.append(f[keep_mask])
        dur_all.append(dur[keep_mask])
        # Contributor ids are per-corpus; offset them so two corpora cannot
        # collide on id 0.
        base = (max((g.max() + 1 for g in grp_all if len(g) and g.max() >= 0),
                    default=0))
        grp_all.append(np.where(grp[keep_mask] >= 0, grp[keep_mask] + base, -1))
        for i, w in enumerate(words):
            if not keep_mask[i]:
                continue
            t = np.frombuffer(w.encode("ascii"), np.uint8).astype(np.int64) - 97
            seq = collapse(t)
            seqs.append(seq)
            by_count[len(seq)].append(kept)
            kept += 1
        row += len(words)
    feats = np.concatenate(feats_all) if len(feats_all) > 1 else feats_all[0]
    assert len(feats) == kept, (len(feats), kept)
    bank = DonorBank(
        feats=feats, seqs=seqs,
        by_count={k: np.asarray(v) for k, v in by_count.items()},
        duration_ms=np.concatenate(dur_all) if len(dur_all) > 1 else dur_all[0],
        group=np.concatenate(grp_all) if len(grp_all) > 1 else grp_all[0],
        seg_len=[])
    if qwerty is not None:
        bank.seg_len = [np.hypot(*np.diff(qwerty[s].astype(np.float64), axis=0).T)
                        if len(s) >= 2 else np.zeros(0) for s in bank.seqs]
    if need_groups:
        nest: Dict[int, Dict[int, List[int]]] = defaultdict(lambda: defaultdict(list))
        for i, g in enumerate(bank.group):
            if g >= 0:
                nest[int(g)][len(bank.seqs[i])].append(i)
        bank.by_group_count = {g: {c: np.asarray(v) for c, v in cs.items()}
                               for g, cs in nest.items()}
    return bank


def fit_duration_law(bank: DonorBank) -> Tuple[float, Dict[str, float]]:
    """Fit ``log T = a + b·log L_ideal + c·log S`` on the donor bank (MIT English).

    S5 needs to know how a trace's *duration* responds to its geometry, because
    the acquisition bandwidth (``n60 = round(T/16.667)+1``) is what band-limits
    the trace before the 64-point resample.  The transplant keeps the donor's own
    tempo and rescales it only for the geometry change:

    ``T_target = T_donor · (L_dst / L_src)^b``

    with ``b`` the partial elasticity of duration in ideal path length at fixed
    vertex count — the right estimand precisely because the donor draw is
    vertex-count matched, so ``S`` is identical on both sides and its term
    cancels.  Fitting on English and applying cross-script is the same discipline
    the S4 exponent follows, and for the same reason: **no parameter of the
    generator may be fit against Yandex statistics.**
    """
    L = np.array([s.sum() for s in bank.seg_len])
    # `seg_len` holds SEGMENT lengths, so its length is (vertices - 1);
    # S is therefore the polyline complexity, a bijection of vertex count.
    S = np.array([max(len(s), 1) for s in bank.seg_len], np.float64)
    T = np.asarray(bank.duration_ms, np.float64)
    ok = (L > 1e-6) & (T > 1.0) & (S >= 1)
    if ok.sum() < 1000:
        raise SystemExit(
            "S5 needs donor durations: the donor cache has none (rebuild it with "
            "`prepare_data.py --extra-train ...`, which records duration_ms)")
    X = np.stack([np.ones(int(ok.sum())), np.log(L[ok]), np.log(S[ok])], 1)
    y = np.log(T[ok])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    r2 = 1.0 - float((resid ** 2).sum()) / max(float(((y - y.mean()) ** 2).sum()), 1e-12)
    # The length-only elasticity is reported alongside because it is the number
    # comparable to CLC's T_line = 68.8·L^0.469 (Cao & Zhai CHI '07).
    X1 = np.stack([np.ones(int(ok.sum())), np.log(L[ok])], 1)
    coef1, *_ = np.linalg.lstsq(X1, y, rcond=None)
    fit = {"n": int(ok.sum()), "b_length_at_fixed_S": float(coef[1]),
           "c_vertices": float(coef[2]), "intercept": float(coef[0]), "r2": r2,
           "b_length_only": float(coef1[1]),
           "median_duration_ms": float(np.median(T[ok]))}
    return float(coef[1]), fit


@dataclass
class GenOpts:
    """Which v2 stages are live.  All-off is the bit-exact v1 mechanism."""

    freq_draw: bool = False        # A  — wordfreq token mass as the draw weight
    geo_donor: bool = False        # C  — k=16 geometry-matched donor selection
    group_block: bool = False      # D  — contributor-coherent donor blocks
    retime: bool = False           # B′ — vertex-aligned per-segment re-timing
    bandwidth: bool = False        # S5 — acquisition-bandwidth matching

    @property
    def is_v1(self) -> bool:
        return not (self.freq_draw or self.geo_donor or self.group_block
                    or self.retime or self.bandwidth)

    @property
    def mask(self) -> str:
        return ",".join(k for k, v in (("a", self.freq_draw), ("c", self.geo_donor),
                                       ("d", self.group_block), ("b", self.retime),
                                       ("s5", self.bandwidth)) if v) or "v1"

    @staticmethod
    def parse(spec: str) -> "GenOpts":
        want = {s.strip().lower() for s in spec.split(",") if s.strip()}
        known = {"a", "c", "d", "b", "s5"}
        if want - known:
            raise SystemExit(f"unknown generator stage(s) {sorted(want - known)}; "
                             f"have {sorted(known)}")
        return GenOpts(freq_draw="a" in want, geo_donor="c" in want,
                       group_block="d" in want, retime="b" in want,
                       bandwidth="s5" in want)


def token_mass(spec: SR.ScriptSpec, lang: str,
               keys: Sequence[str]) -> Tuple[Dict[str, float], Dict[str, object]]:
    """Fix A's draw weight: wordfreq token mass, accumulated **after projection**.

    Querying ``word_frequency`` with the *projected* lexicon form is wrong and
    silently catastrophic for two of the six scripts: Greek's projection strips
    accents and restores word-final ς, and wordfreq's Greek list carries the
    accented, σ-final casefolded forms, so a direct lookup returns 0 for 90 % of
    the pack and collapses the draw onto 2-3-letter words (measured).  Walking
    wordfreq's own list and summing each token's frequency into *its projection*
    is the correct estimand — the token mass of the projected equivalence class —
    and it reads 0 zero-frequency lexicon entries on every Phase-O script.
    """
    from wordfreq import iter_wordlist, word_frequency
    want = set(keys)
    mass: Dict[str, float] = defaultdict(float)
    n_tok = n_hit = 0
    for w in iter_wordlist(lang, "best"):
        n_tok += 1
        p = spec.project(w)
        if p is None or p not in want:
            continue
        mass[p] += word_frequency(w, lang)
        n_hit += 1
    zero = [w for w in want if mass.get(w, 0.0) <= 0.0]
    st = {"wordfreq_lang": lang, "tokens_walked": n_tok, "tokens_matched": n_hit,
          "lexicon_words": len(want), "zero_frequency_words": len(zero),
          "zero_frequency_examples": sorted(zero)[:10]}
    return dict(mass), st


def _pick_donor_geo(cand: np.ndarray, bank: DonorBank,
                    dst_seg: np.ndarray) -> int:
    """Fix C: the reservoir candidate minimising ``Σ_seg |log(L_dst/L_src)|``."""
    best, best_cost = int(cand[0]), np.inf
    for ci in cand:
        src_seg = bank.seg_len[int(ci)]
        cost = float(np.abs(np.log(np.maximum(dst_seg, 1e-6) /
                                   np.maximum(src_seg, 1e-6))).sum())
        if cost < best_cost:
            best, best_cost = int(ci), cost
    return best


class _Draw:
    """Donor-draw state: fix C's reservoir and fix D's contributor block."""

    __slots__ = ("bank", "opts", "cur_group", "block_left")

    def __init__(self, bank: DonorBank, opts: GenOpts) -> None:
        self.bank, self.opts = bank, opts
        self.cur_group, self.block_left = -1, 0

    def donor(self, rng: np.random.Generator, S: int, dst_seg: np.ndarray,
              st: Dict[str, object]) -> int:
        """Draw one donor row for a target of *S* vertices.

        The v1 branch is a single ``rng.integers`` on the count-matched pool, in
        exactly that position of the stream, so ``--generator v1`` reproduces the
        Phase-O byte stream.
        """
        pool = self.bank.by_count[S]
        if self.opts.is_v1:
            return int(pool[rng.integers(len(pool))])
        if self.opts.group_block:
            if self.block_left <= 0:
                gids = list(self.bank.by_group_count)
                self.cur_group = gids[int(rng.integers(len(gids)))]
                self.block_left = int(rng.integers(*GROUP_BLOCK))
            self.block_left -= 1
            gpool = self.bank.by_group_count.get(self.cur_group, {}).get(S)
            if gpool is None or len(gpool) == 0:
                st["group_fallback"] = int(st["group_fallback"]) + 1
            else:
                pool = gpool
        if self.opts.geo_donor:
            cand = pool[rng.integers(len(pool), size=min(K_CANDIDATES, len(pool)))]
            return _pick_donor_geo(cand, self.bank, dst_seg)
        return int(pool[rng.integers(len(pool))])


def transplant(bank: DonorBank, di: int, seq: np.ndarray, dst_seg: np.ndarray,
               qwerty: np.ndarray, centers: np.ndarray, opts: GenOpts,
               st: Dict[str, object]) -> Tuple[np.ndarray, float]:
    """S3–S5 for one row: warp, re-time, band-limit.  ``-> (path [2,64], T_ms)``.

    This is the whole generator mechanism; both the lexicon-draw driver
    (:func:`synthesize`) and the word-matched driver used by the gate battery
    (:func:`synthesize_matched`) call it, so the gated mechanism *is* the shipped
    mechanism rather than a re-implementation of it.
    """
    S = len(seq)
    dseq = bank.seqs[di]
    # Virtual per-vertex alphabet: id i is vertex i of BOTH polylines, so
    # warp_path's monotone-DP correspondence, endpoint pins, arc remap and
    # movement-frame residual transfer run byte-for-byte unchanged and every
    # Phase-H exactness invariant carries over. Letter identity never enters.
    src_virtual = qwerty[dseq]                        # [S,2]
    dst_virtual = centers[seq]                        # [S,2]
    target = np.arange(S, dtype=np.int64)
    if opts.retime and S >= 2:
        warped, jseg = warp_path(bank.feats[di], target, src_virtual,
                                 dst_virtual, return_assign=True)
        np.clip(warped, 0.0, 1.0, out=warped)
        warped = np.clip(retime_segment(warped, bank.feats[di], jseg,
                                        RETIME_ALPHA), 0.0, 1.0)
        st["retimed"] = int(st["retimed"]) + 1
    else:
        warped = warp_path(bank.feats[di], target, src_virtual, dst_virtual)
        np.clip(warped, 0.0, 1.0, out=warped)         # the training loop's
    t_target = 0.0                                    # post-noise clip range
    if opts.bandwidth and S >= 2:
        t_donor = float(bank.duration_ms[di])
        l_src = float(bank.seg_len[di].sum())
        l_dst = float(dst_seg.sum())
        if t_donor > 1.0 and l_src > 1e-6 and l_dst > 1e-6:
            t_target = t_donor * (l_dst / l_src) ** bank.dur_exponent
            before = warped
            warped = resample_bandwidth(warped, t_target)
            if warped is not before:
                st["band_limited"] = int(st["band_limited"]) + 1
    return warped, t_target


def _blank_stats() -> Dict[str, object]:
    return dict(drawn=0, no_donor=0, made=0, retimed=0, band_limited=0,
                group_fallback=0)


def synthesize(n_rows: int, rng: np.random.Generator,
               lexicon: Sequence[Tuple[str, float]], letters: str,
               bank: DonorBank, qwerty: np.ndarray, centers: np.ndarray,
               opts: GenOpts, draw_weights: Optional[np.ndarray] = None,
               progress: int = 100_000,
               ) -> Tuple[np.ndarray, List[str], Dict[str, object], Dict[str, np.ndarray]]:
    """Draw words, transplant donor residuals, re-time, band-limit.

    :param draw_weights: fix-A weights aligned to *lexicon*; ``None`` keeps the
        CKDT ``255 - rank`` weights the lexicon carries.
    :returns: ``(features [N,2,64], words, stats, per-row provenance arrays)``.
    """
    idx = {c: i for i, c in enumerate(letters)}
    words_arr = np.array([w for w, _ in lexicon])
    weights = (np.array([f for _, f in lexicon], np.float64)
               if draw_weights is None else np.asarray(draw_weights, np.float64))
    weights = weights / weights.sum()
    seqs = [collapse(np.array([idx[c] for c in w], np.int64)) for w in words_arr]
    dst_segs = [np.hypot(*np.diff(centers[s].astype(np.float64), axis=0).T)
                if len(s) >= 2 else np.zeros(0) for s in seqs]
    st = _blank_stats()
    draw = _Draw(bank, opts)
    out_feats = np.empty((n_rows, 2, 64), np.float32)
    out_words: List[str] = []
    prov_donor = np.full(n_rows, -1, np.int32)
    prov_group = np.full(n_rows, -1, np.int32)
    prov_dur = np.zeros(n_rows, np.float32)
    t0 = time.time()
    while len(out_words) < n_rows:
        k = rng.choice(len(words_arr), p=weights)
        st["drawn"] = int(st["drawn"]) + 1
        seq = seqs[k]
        pool = bank.by_count.get(len(seq))
        if pool is None or len(pool) == 0:
            st["no_donor"] = int(st["no_donor"]) + 1
            continue
        di = draw.donor(rng, len(seq), dst_segs[k], st)
        warped, t_target = transplant(bank, di, seq, dst_segs[k], qwerty,
                                      centers, opts, st)
        i = len(out_words)
        out_feats[i] = warped
        prov_donor[i], prov_group[i], prov_dur[i] = di, int(bank.group[di]), t_target
        out_words.append(str(words_arr[k]))
        st["made"] = int(st["made"]) + 1
        if progress and int(st["made"]) % progress == 0:
            print(f"  {st['made']}/{n_rows} "
                  f"({int(st['made']) / (time.time() - t0):.0f}/s)", flush=True)
    rows = {"donor_row": prov_donor, "donor_group": prov_group,
            "drawn_duration_ms": prov_dur}
    st["seconds"] = round(time.time() - t0, 1)
    return out_feats, out_words, st, rows


def synthesize_matched(words: Sequence[str], rng: np.random.Generator,
                       letters: str, bank: DonorBank, qwerty: np.ndarray,
                       centers: np.ndarray, opts: GenOpts,
                       ) -> Tuple[np.ndarray, np.ndarray, Dict[str, object],
                                  Dict[str, np.ndarray]]:
    """One synthetic trace per GIVEN word — the gate battery's word-matched arm.

    The word draw is the one thing this driver does not do: the words come from
    the real Yandex partners, so every difference between the two sets is
    generator error with no word, layout or lexicon confound
    (``SYNTH_V2_DESIGN.md`` §1.1).  Rows whose vertex count has no donor are
    reported in ``ok`` rather than dropped, so the arrays stay row-aligned with
    the real set.
    """
    idx = {c: i for i, c in enumerate(letters)}
    st = _blank_stats()
    draw = _Draw(bank, opts)
    out = np.zeros((len(words), 2, 64), np.float32)
    ok = np.zeros(len(words), bool)
    prov_dur = np.zeros(len(words), np.float32)
    prov_donor = np.full(len(words), -1, np.int32)
    for i, w in enumerate(words):
        seq = collapse(np.array([idx[c] for c in w], np.int64))
        st["drawn"] = int(st["drawn"]) + 1
        pool = bank.by_count.get(len(seq))
        if pool is None or len(pool) == 0:
            st["no_donor"] = int(st["no_donor"]) + 1
            continue
        dst_seg = (np.hypot(*np.diff(centers[seq].astype(np.float64), axis=0).T)
                   if len(seq) >= 2 else np.zeros(0))
        di = draw.donor(rng, len(seq), dst_seg, st)
        out[i], prov_dur[i] = transplant(bank, di, seq, dst_seg, qwerty, centers,
                                         opts, st)
        prov_donor[i] = di
        ok[i] = True
        st["made"] = int(st["made"]) + 1
    return out, ok, st, {"donor_row": prov_donor, "drawn_duration_ms": prov_dur}


def endpoint_stats(feats: np.ndarray, words: Sequence[str], letters: str,
                   centers: np.ndarray) -> Dict[str, float]:
    """PHASE_H §2.3 endpoint proximity, generalized off the a-z assumption."""
    idx = {c: i for i, c in enumerate(letters)}
    hit_s = hit_e = n = 0
    dsum_s = dsum_e = 0.0
    for f, w in zip(feats, words):
        if not w:
            continue
        n += 1
        for px, py, letter, first in ((f[0][0], f[1][0], w[0], True),
                                      (f[0][-1], f[1][-1], w[-1], False)):
            d = np.hypot(centers[:, 0] - px, centers[:, 1] - py)
            li = idx[letter]
            hit = int(d.argmin()) == li
            if first:
                dsum_s += float(d[li]); hit_s += hit
            else:
                dsum_e += float(d[li]); hit_e += hit
    return {"n": n, "start_hit": round(hit_s / max(n, 1), 4),
            "end_hit": round(hit_e / max(n, 1), 4),
            "start_d": round(dsum_s / max(n, 1), 4),
            "end_d": round(dsum_e / max(n, 1), 4)}


def wrong_geometry(letters: str, rng: np.random.Generator) -> np.ndarray:
    """The PHASE_I_B falsification control: a deliberately wrong geometry.

    Key slots are randomly permuted, so every key sits on a real key position
    but the WRONG one.  Endpoint proximity against it must collapse to
    near-chance (ru measured 0.008 start-hit against 0.917 for the true frame);
    a frame that scores well under this control is not being validated by the
    endpoint test at all.
    """
    _, qwerty = load_layout(HERE / "en_qwerty.json")
    n = len(letters)
    src = qwerty[rng.permutation(len(qwerty))]
    return np.array([src[i % len(qwerty)] for i in range(n)], np.float32)


def write_split(cache: Path, name: str, feats: np.ndarray, words: List[str],
                letters: str, prov: Dict[str, object],
                rows: Optional[Dict[str, np.ndarray]] = None) -> Path:
    idx = {c: i for i, c in enumerate(letters)}
    tgt_flat = np.concatenate([np.array([idx[c] for c in w], np.int64)
                               for w in words])
    tgt_len = np.array([len(w) for w in words], np.int64)
    out = cache / name
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, features=feats, targets=tgt_flat,
                        target_lengths=tgt_len, words=np.array(words),
                        provenance=np.array(json.dumps(prov, sort_keys=True)),
                        **(rows or {}))
    return out


def selftest_v1(reference: Path, feats: np.ndarray, words: List[str]) -> None:
    """Assert the v1-compat path reproduces an existing cache **bit-exactly**.

    A paired ablation is only a control if the control arm is the thing it
    claims to be.  The v2 rewrite keeps v1's RNG call sequence (one
    ``rng.choice`` for the word, one ``rng.integers`` for the donor, in that
    order, with the CKDT weights) precisely so this holds.
    """
    with np.load(reference, allow_pickle=False) as d:
        ref_f = np.asarray(d["features"])[:len(feats)]
        ref_w = [str(w) for w in d["words"][:len(words)]]
    if ref_w != list(words):
        bad = next(i for i, (a, b) in enumerate(zip(ref_w, words)) if a != b)
        raise SystemExit(f"v1 selftest: word stream diverges at row {bad} "
                         f"({ref_w[bad]!r} vs {words[bad]!r})")
    delta = float(np.abs(ref_f - feats).max())
    if delta != 0.0:
        raise SystemExit(f"v1 selftest: max|Δ| = {delta:.3e} vs {reference} — "
                         f"the v1-compat path is NOT the Phase-O mechanism")
    print(f"v1 selftest: {len(feats)} rows bit-identical to {reference} "
          f"(max|Δ| = 0.00e+00)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--code", required=True, help="script code (script_registry)")
    ap.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    ap.add_argument("--rows", type=int, default=1_000_000, help="training rows")
    ap.add_argument("--val-rows", type=int, default=5_000,
                    help="greedy-selection rows (disjoint draw, same donor side)")
    ap.add_argument("--holdout-rows", type=int, default=10_000,
                    help="final-read rows (disjoint draw AND disjoint donor half)")
    ap.add_argument("--donors", default="train_t3futo.npz,train_t3hws.npz",
                    help="comma-separated cache npz names for the donor pool")
    ap.add_argument("--cache", default="", help="cache dir name (default: "
                    "cache_<code>, except ru -> cache_ru_phaseO because "
                    "cache_ru already holds Phase I-B's real-data caches)")
    ap.add_argument("--splits", default="train,val,holdout")
    ap.add_argument("--force", action="store_true",
                    help="allow overwriting existing split files. Off by "
                         "default: the first Phase-O ru run silently clobbered "
                         "Phase I-B's cache_ru/val.npz (the real-data selection "
                         "val) before this guard existed")
    ap.add_argument("--validate-rows", type=int, default=2000)
    ap.add_argument("--train-donor-side", choices=["train", "all"], default="train",
                    help="donor half the train/val splits draw from. 'train' is "
                         "Phase O's 90/10 discipline, which keeps the synthesis "
                         "HOLDOUT donor-disjoint. 'all' is Phase I-B's footing "
                         "(cyrillic_synth.py predates the split) and is what the "
                         "77.41 ru real-probe baseline was trained on, so it is "
                         "the only footing on which a v2 number is comparable to "
                         "that baseline — at the cost of a holdout that shares "
                         "donors with training, which for ru does not matter "
                         "because ru is read on a REAL probe")
    ap.add_argument("--generator", choices=["v1", "v2"], default="v2",
                    help="v1 = the Phase-O mechanism, bit-exact (the paired "
                         "ablation control); v2 = the amended generator")
    ap.add_argument("--stages", default="",
                    help=f"explicit v2 stage mask (default {DEFAULT_STAGES!r}); "
                         "letters a=wordfreq draw, c=geometry-matched donor, "
                         "d=contributor blocks, b=vertex-aligned re-timing, "
                         "s5=acquisition-bandwidth matching")
    ap.add_argument("--wf-lang", default="",
                    help="wordfreq language for the fix-A draw (default: the "
                         "script's own code, or its wordfreq lexicon's lang)")
    ap.add_argument("--selftest-v1", default="",
                    help="assert the v1-compat path reproduces this npz "
                         "bit-exactly, then exit")
    args = ap.parse_args()

    opts = (GenOpts() if args.generator == "v1"
            else GenOpts.parse(args.stages or DEFAULT_STAGES))
    if args.generator == "v1" and args.stages:
        raise SystemExit("--generator v1 takes no --stages (v1 IS the empty mask)")

    spec = SR.get(args.code)
    cache_en = resolve(args.workdir, Path("cache"))
    default_cache = "cache_ru_phaseO" if spec.code == "ru" else f"cache_{spec.code}"
    cache = resolve(args.workdir, Path(args.cache or default_cache))
    cache.mkdir(parents=True, exist_ok=True)
    want_files = {"train": "train_synth.npz", "val": "val.npz",
                  "holdout": "holdout.npz"}
    if not args.force:
        clash = [want_files[s] for s in
                 [x.strip() for x in args.splits.split(",") if x.strip()]
                 if (cache / want_files[s]).exists()]
        if clash:
            raise SystemExit(
                f"{cache} already holds {clash} — refusing to overwrite. Pass "
                f"--force if that is really what you want, or --cache <dir>.")

    letters, centers = load_layout(HERE / spec.layout_json)
    if "".join(letters) != spec.letters:
        raise SystemExit(f"{spec.layout_json}: letters {''.join(letters)!r} != "
                         f"registry {spec.letters!r}")
    _, qwerty = load_layout(HERE / "en_qwerty.json")

    lexicon, lex_st = spec.load_lexicon(T_OUT)
    print(f"[{spec.code}] lexicon {lex_st}", flush=True)
    if not lexicon:
        raise SystemExit(f"[{spec.code}] empty lexicon — nothing to synthesize")

    # S0 / fix A — the draw weight.  The lexicon itself is unchanged (same words,
    # same CKDT weights in the trie); only which words the GENERATOR emphasises
    # moves, so the no-corpus counterfactual is intact: token frequencies are
    # lexicon-tier knowledge, not swipe-corpus knowledge.
    draw_weights = None
    freq_st: Dict[str, object] = {}
    if opts.freq_draw:
        lang = args.wf_lang or spec.lexicon.lang or spec.code
        mass, freq_st = token_mass(spec, lang, [w for w, _ in lexicon])
        draw_weights = np.array([mass.get(w, 0.0) for w, _ in lexicon], np.float64)
        if draw_weights.sum() <= 0:
            raise SystemExit(f"[{spec.code}] wordfreq '{lang}' gives the whole "
                             f"lexicon zero mass — fix A cannot run")
        print(f"[{spec.code}] fix A draw: {freq_st}", flush=True)

    report: Dict[str, object] = {
        "script": spec.code, "layout": spec.layout_json,
        "letters": spec.letters, "lexicon": lex_st,
        "lexicon_tier": spec.lexicon.tier,
        "donor_holdout_frac": DONOR_HOLDOUT_FRAC,
        "generator_version": args.generator,
        "stage_mask": opts.mask,
        "retime_alpha": RETIME_ALPHA if opts.retime else None,
        "wordfreq_draw": freq_st or None,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    donor_paths = [cache_en / p.strip() for p in args.donors.split(",")]
    want = [s.strip() for s in args.splits.split(",") if s.strip()]

    # Vertex-count coverage of the lexicon against each donor side.
    side_of = {"train": args.train_donor_side, "val": args.train_donor_side,
               "holdout": "holdout"}
    for side in sorted({side_of[s] for s in want}):
        bank = build_donor_index(donor_paths, side, qwerty=qwerty,
                                 need_groups=opts.group_block)
        by_count = bank.by_count
        print(f"[{spec.code}] donor side {side}: {len(bank)} traces, "
              f"vertex counts {min(by_count)}..{max(by_count)}", flush=True)
        if opts.bandwidth:
            bank.dur_exponent, bank.dur_fit = fit_duration_law(bank)
            report[f"duration_law_{side}"] = bank.dur_fit
            print(f"[{spec.code}]   S5 duration law: T ∝ L^"
                  f"{bank.dur_exponent:.3f} at fixed vertex count "
                  f"(R² {bank.dur_fit['r2']:.3f}, n {bank.dur_fit['n']}, "
                  f"length-only exponent {bank.dur_fit['b_length_only']:.3f}, "
                  f"donor median {bank.dur_fit['median_duration_ms']:.0f} ms)",
                  flush=True)
        if opts.group_block:
            cov = float((bank.group >= 0).mean())
            report[f"donor_groups_{side}"] = {
                "distinct": len(bank.by_group_count), "row_coverage": round(cov, 4)}
            print(f"[{spec.code}]   contributor ids: {len(bank.by_group_count)} "
                  f"groups over {cov:.1%} of the side", flush=True)
        idx = {c: i for i, c in enumerate(spec.letters)}
        need: Dict[int, int] = defaultdict(int)
        for w, _ in lexicon:
            need[len(collapse(np.array([idx[c] for c in w])))] += 1
        uncovered = {k: v for k, v in need.items() if k not in by_count}
        report[f"donor_{side}"] = {
            "traces": int(len(bank)),
            "uncovered_vertex_counts": uncovered or None,
            "uncovered_words": int(sum(uncovered.values())),
        }
        print(f"[{spec.code}]   uncovered vertex counts: {uncovered or 'NONE'}",
              flush=True)

        for split in want:
            if side_of[split] != side:
                continue
            n = {"train": args.rows, "val": args.val_rows,
                 "holdout": args.holdout_rows}[split]
            rng = np.random.default_rng(SEEDS[split])
            feats, words, st, rows = synthesize(
                n, rng, lexicon, spec.letters, bank, qwerty, centers, opts,
                draw_weights=draw_weights)
            if args.selftest_v1:
                selftest_v1(resolve(args.workdir, Path(args.selftest_v1)),
                            feats, words)
                return 0
            fname = {"train": "train_synth.npz", "val": "val.npz",
                     "holdout": "holdout.npz"}[split]
            prov = dict(generator="script_synth.py", script=spec.code,
                        generator_version=args.generator, stage_mask=opts.mask,
                        retime_alpha=RETIME_ALPHA if opts.retime else None,
                        duration_exponent=(bank.dur_exponent if opts.bandwidth
                                           else None),
                        split=split, seed=SEEDS[split], rows=n,
                        donor_side=side, donors=args.donors,
                        layout=spec.layout_json, stats=st,
                        lexicon=spec.lexicon.tier)
            out = write_split(cache, fname, feats, words, spec.letters, prov,
                              rows=rows)
            report[f"gen_{split}"] = st
            print(f"[{spec.code}] {split}: {st} -> {out}", flush=True)

            if split in ("train", "holdout"):
                v = args.validate_rows
                ep = endpoint_stats(feats[:v], words[:v], spec.letters, centers)
                ctrl = endpoint_stats(
                    feats[:v], words[:v], spec.letters,
                    wrong_geometry(spec.letters, np.random.default_rng(4242)))
                report[f"endpoints_{split}"] = ep
                report[f"endpoints_{split}_wrong_geo_control"] = ctrl
                print(f"[{spec.code}] {split} endpoints {ep}", flush=True)
                print(f"[{spec.code}] {split} wrong-geo control {ctrl}", flush=True)

    (cache / "synth_stats.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1))
    print(f"[{spec.code}] wrote {cache / 'synth_stats.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
