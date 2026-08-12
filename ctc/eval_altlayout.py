#!/usr/bin/env python3
"""Cross-layout / cross-language evaluation of the CTC swipe encoders.

The encoders take the keyboard geometry as an INPUT (``layout_keys [1,64,2]`` key
centers + ``layout_mask [1,64]``; emission column ``c`` is the key sitting in slot
``c``) and were trained with slot-permutation + affine-jitter augmentation on
**en_qwerty only**. They are therefore layout-agnostic *by construction* and had
been validated on *nothing but* en_qwerty. This script supplies the missing
evidence: REAL human swipes on five non-QWERTY layouts.

Corpora
    ``futo-org/swipe.futo.org`` config ``swipe-5`` (MIT), single-finger,
    language-matched, regenerated locally by ``fetch_futo_multilayout.mjs`` into
    ``~/.cache/cleverkeys-test/futo_swipe5_<layout>.jsonl.gz``. Rows are
    ``{"word","w","h","pts":[[x,y,t],...]}`` with x,y ALREADY normalized over the
    [0,1]^2 letter area — the same frame the committed layout geometries and our
    en_qwerty training data use, so the frame mapping is the IDENTITY. That claim
    is not taken on faith: ``--sanity`` measures endpoint-key proximity per layout
    and also under a deliberately WRONG geometry, so a broken frame would show up
    before any accuracy number is read.

Alphabet policy (documented, applied consistently to lexicon AND targets)
    Emissions are a-z, as trained. A word is projected by NFD-decomposing,
    dropping combining marks, then removing ``'`` and ``-`` (the STRIP convention
    the en trie already uses). A word whose projection still contains a non-a-z
    character (German ``ß``, French ``œ``/``æ``) has NO a-z spelling, is
    UNTYPEABLE on an a-z emission alphabet, and is counted separately rather than
    mangled.

Key-slot arms
    ``az26``  the 26 a-z keys only (mask 26) — matches the training regime exactly.
    ``full``  the a-z keys in slots 0..25 PLUS the layout's extra letter keys
              (``'``, ä, ö, ü, ñ) in slots 26.. with mask true. The emission slice
              still reads columns 0..25 + blank, so the extra keys only tell the
              model "a key exists here", which is geometrically faithful to what
              the user's finger actually travelled over.

Usage
    python eval_altlayout.py --onnx artifacts/ch128_s1234.onnx --sanity
    python eval_altlayout.py --onnx artifacts/ch128_s1234.onnx --arm az26 --arm full
"""
from __future__ import annotations

import argparse
import gzip
import json
import struct
import sys
import time
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from futo_decoder_ceiling import futo_viterbi_beam, slice_emissions  # noqa: E402
from futo_decoder_eval import (LexTrie, Tally, featurize, greedy_ctc,  # noqa: E402
                               len_stratum, load_combined_vocab, rank_of)
from model import MAX_KEYS  # noqa: E402
from paths import DEFAULT_WORKDIR, resolve  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
LAYOUT_DIR = SCRIPT_DIR / "layouts"

#: The E1 scoring preset (PHASE_E.md §1) — every accuracy number in the campaign.
E1_PRESET = (1.05, 1.1, 0.2, 0.3734, 0.9882)

#: layout -> (corpus language, lexicon key). ``en_aosp`` is the 146,964-word STRIP
#: trie the en_qwerty control uses, so dvorak changes the GEOMETRY and nothing else.
LAYOUTS: Dict[str, Tuple[str, str]] = {
    "qwerty": ("en", "en_aosp"),
    "dvorak": ("en", "en_aosp"),
    "azerty": ("fr", "fr"),
    "qwertz": ("de", "de"),
    "german": ("de", "de"),
    "spanish": ("es", "es"),
    # Phase J realalt heldouts (swipe-5, session-disjoint 20 % splits built by
    # build_swipe2345.py) — English text on layouts no committed eval used.
    "clearflow": ("en", "en_aosp"),
    "kasroz": ("en", "en_aosp"),
}

#: ``--lexicon LAYOUT=KIND`` overrides. ``en`` selects the app's bundled 98,140-word
#: ``en_enhanced.bin``, which is the trie the geometric-engine anchors were measured
#: against — the apples-to-apples choice when comparing dvorak to that anchor.
LEXICON_KINDS = ("en_aosp", "en", "fr", "de", "es")

#: Bundled CKDT-v2 dictionaries in the app repo (read-only reference).
APP_DICT_DIR = Path("/home/will/git/swype/CleverKeys/src/main/assets/dictionaries")


# ── word projection onto the a-z emission alphabet ──────────────────────────────

def project_az(word: str) -> Optional[str]:
    """Project *word* onto a-z, or ``None`` when it has no a-z spelling.

    NFD + combining-mark removal folds é→e, ä→a, ñ→n; ``'`` and ``-`` are removed
    (the STRIP convention). ``ß``/``œ``/``æ`` survive both steps as non-a-z
    characters and yield ``None`` — they are untypeable, not mangled.
    """
    s = unicodedata.normalize("NFD", word.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace("'", "").replace("’", "").replace("-", "")
    if not s:
        return None
    for c in s:
        if not ("a" <= c <= "z"):
            return None
    return s


def offending_chars(word: str) -> List[str]:
    """The non-a-z characters that made :func:`project_az` fail (for reporting)."""
    s = unicodedata.normalize("NFD", word.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace("'", "").replace("’", "").replace("-", "")
    return sorted({c for c in s if not ("a" <= c <= "z")})


# ── CKDT v2 reader (format: scripts/build_dictionary.py::write_v2_binary) ───────

def read_ckdt_v2(path: Path) -> List[Tuple[str, int]]:
    """Read a bundled ``<lang>_enhanced.bin`` -> ``[(canonical_word, rank)]``.

    Header (48 B, little-endian): magic ``CKDT``, version 2, 4-byte language tag,
    word count, canonical/normalized/accent-map offsets, 20 reserved bytes. The
    canonical section is ``count`` records of ``uint16 len | utf-8 bytes | uint8
    rank`` where **rank 0 = most frequent, 255 = least**.
    """
    buf = path.read_bytes()
    magic, version = buf[:4], struct.unpack_from("<I", buf, 4)[0]
    if magic != b"CKDT" or version != 2:
        raise SystemExit(f"{path}: expected CKDT v2, got {magic!r} v{version}")
    count, canon_off = struct.unpack_from("<I", buf, 12)[0], struct.unpack_from("<I", buf, 16)[0]
    out: List[Tuple[str, int]] = []
    off = canon_off
    for _ in range(count):
        (n,) = struct.unpack_from("<H", buf, off)
        off += 2
        word = buf[off:off + n].decode("utf-8", "replace")
        off += n
        rank = buf[off]
        off += 1
        out.append((word, rank))
    return out


def ckdt_trie(path: Path) -> Tuple[LexTrie, Dict[str, int]]:
    """Build an a-z :class:`LexTrie` from a CKDT v2 dictionary.

    ``freq = 255 - rank`` puts the CKDT rank on the SAME 1..255 magnitude scale as
    the AOSP ``f=`` field the en trie is built from, so ``log_freq = log(freq)``
    — and therefore the E1 ``lambda`` weight — means the same thing across
    lexicons. Both scales are themselves log-compressed frequencies, so the
    transformation is like-for-like rather than merely numeric.
    """
    trie = LexTrie()
    stats = {"records": 0, "untypeable": 0, "kept": 0}
    for word, rank in read_ckdt_v2(path):
        stats["records"] += 1
        key = project_az(word)
        if key is None:
            stats["untypeable"] += 1
            continue
        trie.insert(key, float(max(1, 255 - rank)))
        stats["kept"] += 1
    stats["distinct"] = trie.num_words
    return trie, stats


def load_lexicon(kind: str, workdir: Path) -> Tuple[LexTrie, Dict[str, int]]:
    """``en_aosp`` -> the campaign's STRIP trie; everything else -> a CKDT v2 bin."""
    if kind == "en_aosp":
        trie = load_combined_vocab(resolve(workdir, "data/futo_en_wordlist.combined"))
        return trie, {"records": -1, "untypeable": -1, "kept": -1,
                      "distinct": trie.num_words}
    return ckdt_trie(APP_DICT_DIR / f"{kind}_enhanced.bin")


# ── layout geometry ─────────────────────────────────────────────────────────────

class Layout:
    """A FUTO layout json reduced to the model's slot contract.

    ``az_centers`` are the 26 a-z key centers in a..z order, so emission column
    ``c`` is letter ``chr(ord('a')+c)`` for both arms. ``extra`` holds the layout's
    remaining letter keys (``'``, ä, ö, ü, ñ) which the ``full`` arm appends to
    slots 26.. — present in the geometry, absent from the emission slice.
    """

    def __init__(self, path: Path) -> None:
        obj = json.loads(path.read_text())
        self.name = obj["name"]
        by_letter = {k["letter"]: (float(k["cx"]), float(k["cy"])) for k in obj["keys"]}
        missing = [chr(ord("a") + i) for i in range(26)
                   if chr(ord("a") + i) not in by_letter]
        if missing:
            raise SystemExit(f"{path}: layout lacks a-z keys {missing}")
        self.az_centers = np.array(
            [by_letter[chr(ord("a") + i)] for i in range(26)], np.float32)   # [26,2]
        self.extra_letters = [k["letter"] for k in obj["keys"]
                              if not ("a" <= k["letter"] <= "z")]
        self.extra_centers = np.array(
            [by_letter[c] for c in self.extra_letters], np.float32
        ).reshape(-1, 2)

    def slots(self, arm: str, perm_seed: int = -1
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """-> padded ``keys [64,2]``, ``mask [64]`` and the emission columns to read.

        With ``perm_seed >= 0`` the 26 a-z keys are scattered into RANDOM slots of
        the 64 — exactly the train-time slot permutation, applied at inference —
        and ``cols[i]`` is the slot holding letter ``i``, so the emission slice
        follows the keys. This is the probe that separates *slot* invariance (which
        the augmentation trained, at p=0.5) from *geometry* generalization (which
        it did not).
        """
        keys = np.zeros((MAX_KEYS, 2), np.float32)
        mask = np.zeros((MAX_KEYS,), bool)
        n_extra = len(self.extra_letters) if arm == "full" else 0
        if perm_seed >= 0:
            rng = np.random.default_rng(perm_seed)
            slots = rng.permutation(MAX_KEYS)[: 26 + n_extra]
        else:
            slots = np.arange(26 + n_extra)
        keys[slots[:26]] = self.az_centers
        mask[slots[:26]] = True
        if n_extra:
            keys[slots[26:]] = self.extra_centers
            mask[slots[26:]] = True
        return keys, mask, slots[:26].astype(np.int64)

    def affine(self, sx: float, sy: float, tx: float, ty: float) -> "Layout":
        """A copy with every key center pushed through ``(c-0.5)*s + 0.5 + t``.

        The train-time jitter applies one such affine to the path AND the centers
        together, so replaying it here (with the same transform applied to the
        path) measures accuracy INSIDE vs OUTSIDE the augmentation envelope.
        """
        out = object.__new__(Layout)
        out.name = f"{self.name}@affine({sx},{sy},{tx},{ty})"
        out.extra_letters = list(self.extra_letters)
        for src, dst in ((self.az_centers, "az_centers"),
                         (self.extra_centers, "extra_centers")):
            c = src.copy()
            if c.size:
                c[:, 0] = (c[:, 0] - 0.5) * sx + 0.5 + tx
                c[:, 1] = (c[:, 1] - 0.5) * sy + 0.5 + ty
            setattr(out, dst, c)
        return out


def affine_rows(rows, sx: float, sy: float, tx: float, ty: float):
    """Apply the same affine to every trace point (paired with :meth:`Layout.affine`)."""
    out = []
    for word, xs, ys, ts in rows:
        out.append((word,
                    [min(1.0, max(0.0, (x - 0.5) * sx + 0.5 + tx)) for x in xs],
                    [min(1.0, max(0.0, (y - 0.5) * sy + 0.5 + ty)) for y in ys],
                    ts))
    return out


# ── corpus ──────────────────────────────────────────────────────────────────────

def load_corpus(path: Path) -> List[Tuple[str, List[float], List[float], List[float]]]:
    """Read a ``futo_swipe5_<layout>.jsonl.gz`` sample -> ``(word, xs, ys, ts)``."""
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            pts = o["pts"]
            rows.append((o["word"],
                         [float(p[0]) for p in pts],
                         [float(p[1]) for p in pts],
                         [float(p[2]) for p in pts]))
    return rows


def load_qwerty_control(path: Path, limit: int
                        ) -> List[Tuple[str, List[float], List[float], List[float]]]:
    """Read the en_qwerty val split (``{word, points:[{t,x,y}]}``) as a corpus."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            pts = o["points"]
            rows.append((o["word"],
                         [float(p["x"]) for p in pts],
                         [float(p["y"]) for p in pts],
                         [float(p["t"]) for p in pts]))
            if limit and len(rows) >= limit:
                break
    return rows


# ── frame-mapping sanity metric ─────────────────────────────────────────────────

def endpoint_proximity(rows, centers: np.ndarray) -> Dict[str, float]:
    """Fraction of traces whose first/last point's nearest key IS the first/last
    letter of the target word, plus the mean endpoint→key distance.

    This is the frame-mapping check. Under a correct mapping a swipe starts on its
    word's first key and ends on its last, so both fractions run high (~0.9+); a
    wrong normalization, a transposed axis or a mismatched layout collapses them
    toward chance (~1/26) while accuracy would still print a plausible-looking
    number. Words whose endpoints are not a-z (unprojectable) are skipped.
    """
    hit_s = hit_e = n = 0
    dsum_s = dsum_e = 0.0
    for word, xs, ys, _ts in rows:
        w = project_az(word)
        if not w:
            continue
        n += 1
        for px, py, letter, which in ((xs[0], ys[0], w[0], "s"),
                                      (xs[-1], ys[-1], w[-1], "e")):
            d = np.hypot(centers[:, 0] - px, centers[:, 1] - py)
            j = int(d.argmin())
            if which == "s":
                dsum_s += float(d[ord(letter) - 97])
                hit_s += (j == ord(letter) - 97)
            else:
                dsum_e += float(d[ord(letter) - 97])
                hit_e += (j == ord(letter) - 97)
    if not n:
        return {"n": 0, "start_hit": 0.0, "end_hit": 0.0, "start_d": 0.0, "end_d": 0.0}
    return {"n": n, "start_hit": hit_s / n, "end_hit": hit_e / n,
            "start_d": dsum_s / n, "end_d": dsum_e / n}


# ── encoder ─────────────────────────────────────────────────────────────────────

class OnnxEncoder:
    """The exported ONNX graph — the on-device path (same class as eval_beam.py).

    ``onnx_path`` may name several comma-separated exports (Phase K1
    seed-ensemble): log-emissions are averaged across models BEFORE the beam
    (see ``eval_beam.ensemble_average`` for the two modes and why 'logprob'
    renormalizes)."""

    def __init__(self, onnx_path, avg: str = "logprob") -> None:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        paths = onnx_path if isinstance(onnx_path, list) else [onnx_path]
        self.sessions = [ort.InferenceSession(str(p), so,
                                              providers=["CPUExecutionProvider"])
                         for p in paths]
        self.avg = avg

    def forward(self, feats: np.ndarray, keys: np.ndarray, mask: np.ndarray):
        feed = {"features": feats[None], "layout_keys": keys[None],
                "layout_mask": mask[None]}
        outs = [s.run(["log_emissions"], feed)[0][0] for s in self.sessions]
        if len(outs) == 1:
            return outs[0]                                      # [T', 65]
        from eval_beam import ensemble_average
        return ensemble_average(outs, self.avg)


# ── decode ──────────────────────────────────────────────────────────────────────

LETTERS_AZ = [chr(ord("a") + i) for i in range(26)]


def decode_corpus(enc: OnnxEncoder, rows, keys: np.ndarray, mask: np.ndarray,
                  cols: np.ndarray, trie: LexTrie, preset, beam_width: int,
                  top_k: int, progress: int, label: str,
                  out_f=None) -> Dict[str, object]:
    """Decode every projectable, in-vocabulary row; return the tally + buckets.

    OOV and untypeable rows are EXCLUDED from the accuracy denominator and
    reported separately, matching the geometric engine's in-dict replay protocol
    (which is what the comparison anchors were measured under).
    """
    gamma, lam_w, beta, gamma_prune, beta_prune = preset
    g_tal, b_tal = Tally(), Tally()
    strat = {"<=3": Tally(), "4+": Tally()}
    untypeable, oov = 0, 0
    untypeable_chars: Dict[str, int] = {}
    t0 = time.time()
    n_seen = 0
    for word, xs, ys, ts in rows:
        n_seen += 1
        target = project_az(word)
        if target is None:
            untypeable += 1
            for c in offending_chars(word):
                untypeable_chars[c] = untypeable_chars.get(c, 0) + 1
            continue
        if not trie.contains(target):
            oov += 1
            continue
        feats = featurize(xs, ys, ts)
        full = enc.forward(feats, keys, mask)                   # [32, 65]
        # Gather the 26 columns holding a-z (identity slots -> the plain slice) and
        # append the blank at MAX_KEYS, matching slice_emissions' layout exactly.
        lp = np.empty((full.shape[0], 27), np.float32)
        lp[:, :26] = full[:, cols]
        lp[:, 26] = full[:, MAX_KEYS]
        greedy = greedy_ctc(lp, LETTERS_AZ, 26)
        beam = futo_viterbi_beam(lp, LETTERS_AZ, 26, trie, beam_width, top_k,
                                 gamma, lam_w, beta, gamma_prune, beta_prune)
        words = [w for w, _ in beam]
        g_tal.add(0 if greedy == target else -1)
        r = rank_of(target, words)
        b_tal.add(r)
        strat[len_stratum(target)].add(r)
        if out_f is not None:
            out_f.write(json.dumps({"word": word, "target": target,
                                    "greedy": greedy, "topk": words[:5],
                                    "rank": r}) + "\n")
        if progress and b_tal.n % progress == 0:
            print(f"  [{label}] {b_tal.n} decoded  {b_tal.row()}  "
                  f"({b_tal.n / (time.time() - t0):.1f} tr/s)", flush=True)
    return {
        "rows": n_seen, "decoded": b_tal.n, "untypeable": untypeable, "oov": oov,
        "untypeable_chars": untypeable_chars,
        "greedy_t1": g_tal.t1 / max(g_tal.n, 1) * 100,
        "t1": b_tal.t1 / max(b_tal.n, 1) * 100,
        "t3": b_tal.t3 / max(b_tal.n, 1) * 100,
        "t5": b_tal.t5 / max(b_tal.n, 1) * 100,
        "le3_n": strat["<=3"].n,
        "le3_t1": strat["<=3"].t1 / max(strat["<=3"].n, 1) * 100,
        "ge4_n": strat["4+"].n,
        "ge4_t1": strat["4+"].t1 / max(strat["4+"].n, 1) * 100,
        "seconds": time.time() - t0,
    }


# ── main ────────────────────────────────────────────────────────────────────────

def cache_dir() -> Path:
    import os
    return Path(os.environ.get("CLEVERKEYS_TEST_CACHE",
                               str(Path.home() / ".cache" / "cleverkeys-test")))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    ap.add_argument("--onnx", default="artifacts/ch128_s1234.onnx",
                    help="exported encoder; relative paths resolve against this "
                         "script; comma-separate several for a K1 ensemble")
    ap.add_argument("--ens-avg", choices=["logprob", "prob"], default="logprob",
                    dest="ens_avg",
                    help="ensemble averaging mode when --onnx lists >1 model")
    ap.add_argument("--layouts", default="dvorak,azerty,qwertz,german,spanish")
    ap.add_argument("--arm", action="append", default=[], choices=["az26", "full"],
                    help="key-slot arm (repeatable); default az26")
    ap.add_argument("--preset", default="", help="gamma,lambda,beta,gammaPrune,betaPrune "
                                                 "(default: the E1 preset)")
    ap.add_argument("--beam-width", type=int, default=100, dest="beam_width")
    ap.add_argument("--top-k", type=int, default=8, dest="top_k")
    ap.add_argument("--limit", type=int, default=0, help="cap rows per layout")
    ap.add_argument("--progress", type=int, default=500)
    ap.add_argument("--sanity", action="store_true",
                    help="print the endpoint-proximity frame check (incl. the "
                         "wrong-geometry falsification control) and exit")
    ap.add_argument("--qwerty-control", default="", dest="qwerty_control",
                    help="also run this en_qwerty jsonl through the same code path")
    ap.add_argument("--lexicon", action="append", default=[], metavar="LAYOUT=KIND",
                    help="override a layout's lexicon (repeatable), e.g. "
                         "'dvorak=en' to decode dvorak against the app's bundled "
                         "98k en trie — the one the geometric anchors used")
    ap.add_argument("--perm-seed", type=int, default=-1, dest="perm_seed",
                    help="scatter the keys into RANDOM slots of the 64 with this "
                         "seed (the train-time slot permutation, applied at "
                         "inference). -1 = identity slots. Probes slot invariance "
                         "in isolation from geometry generalization")
    ap.add_argument("--affine", default="",
                    help="'sx,sy,tx,ty' applied to BOTH the key centers and the "
                         "trace points, i.e. the train-time shared affine replayed "
                         "at inference. Train range: scale 0.85-1.15, |t| <= 0.05")
    ap.add_argument("--json-out", default="", dest="json_out")
    ap.add_argument("--dump", default="", help="per-trace JSONL dump dir")
    args = ap.parse_args()

    preset = E1_PRESET
    if args.preset:
        vals = [float(v) for v in args.preset.split(",")]
        if len(vals) != 5:
            raise SystemExit("--preset needs 5 floats")
        preset = tuple(vals)
    arms = args.arm or ["az26"]
    names = [s.strip() for s in args.layouts.split(",") if s.strip()]
    # Per-layout lexicon overrides; the table is copied so LAYOUTS stays the
    # documented default and a run's actual choice is recorded in the json-out.
    layout_lex = {n: LAYOUTS[n] for n in LAYOUTS}
    for spec in args.lexicon:
        lay, _, kind = spec.partition("=")
        if lay not in layout_lex or kind not in LEXICON_KINDS:
            raise SystemExit(f"--lexicon {spec!r}: expected LAYOUT=KIND with KIND "
                             f"in {LEXICON_KINDS}")
        layout_lex[lay] = (layout_lex[lay][0], kind)

    aff = None
    if args.affine:
        vals = [float(v) for v in args.affine.split(",")]
        if len(vals) != 4:
            raise SystemExit("--affine needs 4 floats: sx,sy,tx,ty")
        aff = tuple(vals)

    layouts = {n: Layout(LAYOUT_DIR / f"futo_{n}.json") for n in names}
    if aff:
        layouts = {n: L.affine(*aff) for n, L in layouts.items()}
    corpora = {}
    for n in names:
        f = cache_dir() / f"futo_swipe5_{n}.jsonl.gz"
        if not f.exists():
            print(f"[skip] {n}: no corpus at {f} — run fetch_futo_multilayout.mjs")
            continue
        rows = load_corpus(f)
        rows = rows[: args.limit] if args.limit else rows
        corpora[n] = affine_rows(rows, *aff) if aff else rows

    # ── frame sanity ─────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("FRAME-MAPPING SANITY — nearest-key hit rate at the trace endpoints")
    print("=" * 78)
    print(f"{'corpus':<10} {'geometry':<10} {'n':>6} {'start-hit':>10} {'end-hit':>9} "
          f"{'start-d':>8} {'end-d':>7}")
    sanity: Dict[str, object] = {}
    ctrl_rows = None
    if args.qwerty_control:
        ctrl_rows = load_qwerty_control(resolve(args.workdir, args.qwerty_control),
                                        args.limit or 2000)
        qw = Layout(LAYOUT_DIR / "futo_qwerty.json")
        if aff:
            qw, ctrl_rows = qw.affine(*aff), affine_rows(ctrl_rows, *aff)
        s = endpoint_proximity(ctrl_rows, qw.az_centers)
        sanity["qwerty/qwerty"] = s
        print(f"{'qwerty':<10} {'qwerty':<10} {s['n']:>6} {s['start_hit']:>10.3f} "
              f"{s['end_hit']:>9.3f} {s['start_d']:>8.4f} {s['end_d']:>7.4f}")
    qwerty_geo = (qw if args.qwerty_control else
                  Layout(LAYOUT_DIR / "futo_qwerty.json")).az_centers
    for n, rows in corpora.items():
        s = endpoint_proximity(rows, layouts[n].az_centers)
        sanity[f"{n}/{n}"] = s
        print(f"{n:<10} {n:<10} {s['n']:>6} {s['start_hit']:>10.3f} "
              f"{s['end_hit']:>9.3f} {s['start_d']:>8.4f} {s['end_d']:>7.4f}")
        # Falsification control: the SAME traces read against QWERTY geometry.
        # If the metric could not tell a right frame from a wrong one it would be
        # worthless as evidence, so the wrong-geometry row is printed too.
        w = endpoint_proximity(rows, qwerty_geo)
        sanity[f"{n}/qwerty"] = w
        print(f"{'':<10} {'qwerty*':<10} {w['n']:>6} {w['start_hit']:>10.3f} "
              f"{w['end_hit']:>9.3f} {w['start_d']:>8.4f} {w['end_d']:>7.4f}")
    print("* wrong-geometry falsification control (expected to collapse)")
    if args.sanity:
        if args.json_out:
            Path(args.json_out).write_text(json.dumps({"sanity": sanity}, indent=2))
        return 0

    # ── lexicons ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("LEXICONS")
    print("=" * 78)
    tries: Dict[str, LexTrie] = {}
    lex_stats: Dict[str, Dict[str, int]] = {}
    needed = {layout_lex[n][1] for n in corpora}
    if args.qwerty_control:
        needed.add("en_aosp")
    for kind in sorted(needed):
        tries[kind], lex_stats[kind] = load_lexicon(kind, args.workdir)
        st = lex_stats[kind]
        print(f"  {kind:<8} distinct a-z words={st['distinct']:>7}"
              + ("" if st["records"] < 0 else
                 f"  (records={st['records']}, untypeable dropped={st['untypeable']})"))

    enc = OnnxEncoder([resolve(SCRIPT_DIR, p) for p in str(args.onnx).split(",")],
                      args.ens_avg)
    if len(str(args.onnx).split(",")) > 1:
        print(f"ENSEMBLE of {len(str(args.onnx).split(','))} models, "
              f"avg={args.ens_avg}")
    dump_dir = Path(args.dump) if args.dump else None
    if dump_dir:
        dump_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Dict[str, object]] = {}

    if ctrl_rows is not None:
        keys, mask, cols = qw.slots("az26", args.perm_seed)
        print(f"\n[control] en_qwerty n={len(ctrl_rows)} arm=az26")
        r = decode_corpus(enc, ctrl_rows, keys, mask, cols, tries["en_aosp"], preset,
                          args.beam_width, args.top_k, args.progress, "qwerty/az26")
        results["qwerty|az26"] = r
        print(f"  qwerty az26  n={r['decoded']}  t1={r['t1']:.2f} t3={r['t3']:.2f} "
              f"t5={r['t5']:.2f}  greedy={r['greedy_t1']:.2f}")

    for n, rows in corpora.items():
        lang, lex = layout_lex[n]
        for arm in arms:
            keys, mask, cols = layouts[n].slots(arm, args.perm_seed)
            n_extra = int(mask.sum()) - 26
            print(f"\n[{n}/{lang}] arm={arm} keys={int(mask.sum())} "
                  f"(+{n_extra} non-az) rows={len(rows)}")
            out_f = open(dump_dir / f"{n}_{arm}.jsonl", "w") if dump_dir else None
            r = decode_corpus(enc, rows, keys, mask, cols, tries[lex], preset,
                              args.beam_width, args.top_k, args.progress,
                              f"{n}/{arm}", out_f)
            if out_f:
                out_f.close()
            r.update({"layout": n, "lang": lang, "lexicon": lex, "arm": arm,
                      "active_keys": int(mask.sum())})
            results[f"{n}|{arm}"] = r
            print(f"  n={r['decoded']} oov={r['oov']} untypeable={r['untypeable']} "
                  f"{r['untypeable_chars']}")
            print(f"  t1={r['t1']:.2f} t3={r['t3']:.2f} t5={r['t5']:.2f}  "
                  f"greedy={r['greedy_t1']:.2f}")

    print("\n" + "=" * 78)
    print(f"{'layout':<10} {'arm':<6} {'n':>6} {'OOV%':>6} {'t1':>7} {'t3':>7} {'t5':>7}")
    print("=" * 78)
    for k, r in results.items():
        rows_n = max(r["rows"], 1)
        print(f"{k.split('|')[0]:<10} {k.split('|')[1]:<6} {r['decoded']:>6} "
              f"{r['oov'] / rows_n * 100:>6.1f} {r['t1']:>7.2f} {r['t3']:>7.2f} "
              f"{r['t5']:>7.2f}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"onnx": str(args.onnx), "preset": list(preset), "sanity": sanity,
             "perm_seed": args.perm_seed, "affine": args.affine,
             "lexicons": lex_stats, "results": results}, indent=2, default=str))
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
