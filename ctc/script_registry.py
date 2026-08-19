#!/usr/bin/env python3
"""Per-script definitions for the Phase-O multi-script CTC models.

One place that answers, for every script the app can serve: *what is the
alphabet*, *how does a word project onto it*, and *where does the lexicon come
from and on what frequency scale*.  Consumed by :mod:`script_synth`,
:mod:`eval_script` and :mod:`make_golden`, so the three can never disagree —
the Phase-I-B Cyrillic path had the same three facts spread across
``prepare_yandex.py``, ``cyrillic_synth.py`` and ``eval_cyrillic.py``.

Three invariants every entry must satisfy
-----------------------------------------
1. **The alphabet is the layout's centre-key letters, nothing else.**  A letter
   that only exists on a *corner* slot (``key1``..``key8``) is typeable by a
   directional flick but NOT by a swipe, and the app's own alias table gives it
   the host key's centroid — two letters at one coordinate, which no
   geometry-conditioned encoder can separate.  Corner-only letters are
   therefore *excluded from the alphabet* and the words that need them are
   rejected by the projection, with the rejection rate measured and reported
   (``script_synth.py --stats``), never assumed to be negligible.
2. **The projection is applied identically to targets and to the lexicon.**
   The ALT_LAYOUT_EVAL §3 policy; anything else silently scores a different
   task than it decodes.
3. **Lexicon weights live on the CKDT ``255 − rank`` scale**, the same
   magnitude scale every other lexicon in this campaign uses, because the
   decode λ term is scale-sensitive (PHASE_J §6.9: the CKDT scale is exactly
   why ru wants λ = 2.0 rather than E1's 1.1).  For wordfreq-derived lexicons
   the rank is produced by a **byte-exact replica of the app's own**
   ``build_dictionary.frequency_to_rank``, so a pack the app builds later lands
   on the identical scale.
"""
from __future__ import annotations

import math
import struct
import unicodedata
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

APP_REPO = Path("/home/will/git/swype/CleverKeys")
APP_DICTS = APP_REPO / "scripts" / "dictionaries"

HERE = Path(__file__).resolve().parent


# ── projection primitives ────────────────────────────────────────────────────

#: Characters that are stripped from every word before the alphabet test: the
#: word separators the app's own projections drop (``prepare_yandex.project_ru``
#: strips ``-`` and the two apostrophes).
_STRIP = "-'’ʼ‘`"


def strip_marks(s: str) -> str:
    """NFD → drop combining marks (category ``Mn``) → NFC.

    Safe *only* for scripts where no letter's identity depends on a combining
    mark.  It is correct for Greek (accents and diaeresis are not separate
    keys), for Hebrew (niqqud are not keys) and for the Cyrillic layouts here
    (й and ї are precomposed **and on their own keys**, so they must NOT be
    decomposed — those scripts use :func:`no_marks` instead, which asserts the
    word is already mark-free).  ``PHASE_I_DATA`` §4 records the ru case: NFD
    would decompose й into и + breve and silently change the alphabet.
    """
    return unicodedata.normalize(
        "NFC", "".join(c for c in unicodedata.normalize("NFD", s)
                       if not unicodedata.combining(c)))


def _base_project(word: str, alphabet: str, pre: Optional[Callable[[str], str]] = None,
                  post: Optional[Callable[[str], str]] = None) -> Optional[str]:
    """lowercase → strip separators → *pre* → *post* → alphabet test.

    Returns ``None`` when the word has no spelling in *alphabet* — counted as
    untypeable, never mangled.
    """
    s = word.strip().lower()
    for ch in _STRIP:
        s = s.replace(ch, "")
    if pre is not None:
        s = pre(s)
    if post is not None:
        s = post(s)
    if not s:
        return None
    for c in s:
        if c not in alphabet:
            return None
    return s


def _final_sigma(s: str) -> str:
    """Restore Greek word-final sigma: ``…σ`` → ``…ς``.

    Modern Greek orthography is fully deterministic here — σ everywhere except
    word-final, where ς is written — so this is a *lossless* repair, not a
    guess.  It is needed because the app's shipped ``langpack-el`` carries
    σ-final display forms: ``build_wordlist.py::load_aosp`` documents that
    wordfreq casefolds its corpus and Python casefolding maps ς→σ, and calls the
    result "a documented wordfreq-status-quo caveat, not a fix target this
    round".  For *tapping* that is a cosmetic misspelling; for *swiping* it
    would be fatal, because ς and σ are different keys in different rows of
    ``grek_qwerty.xml`` (ς at the QWERTY-w position, σ at the s position), so a
    σ-final lexicon would train and score the wrong endpoint for a very large
    share of Greek words.  See PHASE_O.md §2.2.
    """
    return s[:-1] + "ς" if s.endswith("σ") else s


# ── lexicon sources ──────────────────────────────────────────────────────────

def read_ckdt_v2(path: Path) -> List[Tuple[str, int]]:
    """``[(canonical_word, rank)]`` from a CKDT-v2 blob (rank 0 = most frequent).

    Duplicated from :mod:`eval_altlayout` on purpose: that module imports
    onnxruntime and a decoder stack, and the synthesis path must not.
    """
    buf = path.read_bytes()
    magic, version = buf[:4], struct.unpack_from("<I", buf, 4)[0]
    if magic != b"CKDT" or version != 2:
        raise SystemExit(f"{path}: expected CKDT v2, got {magic!r} v{version}")
    count = struct.unpack_from("<I", buf, 12)[0]
    off = struct.unpack_from("<I", buf, 16)[0]
    out: List[Tuple[str, int]] = []
    for _ in range(count):
        (n,) = struct.unpack_from("<H", buf, off)
        off += 2
        out.append((buf[off:off + n].decode("utf-8", "replace"), buf[off + n]))
        off += n + 1
    return out


def frequency_to_rank(frequency: float, max_freq: float) -> int:
    """Byte-exact replica of the app's ``build_dictionary.frequency_to_rank``."""
    if frequency <= 0 or max_freq <= 0:
        return 255
    rank = int((1.0 - math.log(frequency + 1) / math.log(max_freq + 1)) * 255)
    return max(0, min(255, rank))


def _wordfreq_scaled(word: str, lang: str) -> float:
    """The app's ``get_word_frequency`` for the wordfreq branch (``freq * 1e8``)."""
    import wordfreq
    f = wordfreq.word_frequency(word, lang)
    return f * 1e8 if f > 0 else 1.0


@dataclass(frozen=True)
class Lexicon:
    """Where a script's words come from, and on what footing."""

    kind: str                      # "ckdt" | "wordfreq"
    #: CKDT: the app langpack zip (``dictionary.bin`` inside) or a bare .bin.
    path: Optional[Path] = None
    #: wordfreq: the language code and how deep to take its ranked list.
    lang: Optional[str] = None
    top_n: int = 80_000
    limit: int = 50_000
    #: Evidence tier for anything measured against this lexicon.
    tier: str = ""


@dataclass(frozen=True)
class ScriptSpec:
    """Everything Phase O needs to know about one script."""

    code: str                      # "el"
    name: str                      # "Greek"
    app_xml: str                   # layout file under the app's src/main/layouts
    layout_json: str               # vendored geometry, relative to ctc/
    letters: str                   # alphabet == emission slot order
    lexicon: Lexicon
    #: Letters the layout only offers on corner slots — NOT swipe-typeable.
    corner_only: str = ""
    #: NFD-strip combining marks before the alphabet test (see strip_marks).
    strip_combining: bool = False
    #: Extra character folds applied after mark stripping, ``{src: dst}``.
    folds: Dict[str, str] = field(default_factory=dict)
    #: Script-specific final touch-up (Greek final sigma).
    post: Optional[Callable[[str], str]] = None
    notes: str = ""

    def project(self, word: str) -> Optional[str]:
        """Project a target/lexicon word onto this script's alphabet."""
        def pre(s: str) -> str:
            if self.strip_combining:
                s = strip_marks(s)
            for a, b in self.folds.items():
                s = s.replace(a, b)
            return s
        return _base_project(word, self.letters, pre, self.post)

    def load_lexicon(self, max_len: int = 32) -> Tuple[List[Tuple[str, float]], Dict[str, int]]:
        """``([(projected_word, weight)], stats)`` on the CKDT ``255 − rank`` scale.

        *max_len* rejects words whose CTC target (letters + adjacent repeats)
        cannot fit the ``T_OUT`` frame budget, the same guard
        ``cyrillic_synth.load_ru_lexicon`` applies.
        """
        lx = self.lexicon
        st = {"records": 0, "unprojectable": 0, "too_long": 0, "too_short": 0}
        raw: List[Tuple[str, float]] = []
        if lx.kind == "ckdt":
            assert lx.path is not None
            if lx.path.suffix == ".zip":
                with zipfile.ZipFile(lx.path) as zf:
                    blob = zf.read("dictionary.bin")
                tmp = Path("/tmp") / f"ckdt_{self.code}.bin"
                tmp.write_bytes(blob)
                recs = read_ckdt_v2(tmp)
            else:
                recs = read_ckdt_v2(lx.path)
            raw = [(w, float(max(1, 255 - r))) for w, r in recs]
        elif lx.kind == "wordfreq":
            from wordfreq import iter_wordlist
            assert lx.lang is not None
            # Walk wordfreq's frequency-ordered list to a depth of top_n
            # candidates and keep the first `limit` that project — the same
            # shape as the app's build_wordlist (`--top` depth, `--limit` cap),
            # minus its spell-check oracles and allow/block lists.
            cands: List[str] = []
            seen_depth = 0
            for w in iter_wordlist(lx.lang, "best"):
                seen_depth += 1
                if seen_depth > lx.top_n:
                    break
                if self.project(w) is not None:
                    cands.append(w)
                    if len(cands) >= lx.limit:
                        break
            st["wordfreq_depth"] = seen_depth
            freqs = {w: _wordfreq_scaled(w, lx.lang) for w in cands}
            max_freq = max(freqs.values()) if freqs else 1.0
            raw = [(w, float(max(1, 255 - frequency_to_rank(freqs[w], max_freq))))
                   for w in cands]
        else:
            raise SystemExit(f"unknown lexicon kind {lx.kind!r}")

        out: Dict[str, float] = {}
        for word, weight in raw:
            st["records"] += 1
            p = self.project(word)
            if p is None:
                st["unprojectable"] += 1
                continue
            if len(p) < 2:
                st["too_short"] += 1
                continue
            if len(p) + _adjacent_repeats(p) > max_len:
                st["too_long"] += 1
                continue
            if p not in out or weight > out[p]:
                out[p] = weight
        st["distinct"] = len(out)
        return sorted(out.items()), st


def _adjacent_repeats(word: str) -> int:
    return sum(1 for a, b in zip(word, word[1:]) if a == b)


# ── the registry ─────────────────────────────────────────────────────────────

#: Greek — the one non-Latin script besides Russian with a **bundled app
#: lexicon**, and a perfect 1:1 layout fit (25 centre keys, zero corner-only
#: letters).  Alphabet order is codepoint-sorted, which puts ς (U+03C2) between
#: ρ and σ; the order is arbitrary but must be stable, because it *is* the
#: emission-column assignment.
EL = ScriptSpec(
    code="el", name="Greek",
    app_xml="grek_qwerty.xml", layout_json="layouts/el_qwerty.json",
    letters="αβγδεζηθικλμνξοπρςστυφχψω",
    corner_only="",
    strip_combining=True,
    post=_final_sigma,
    lexicon=Lexicon(kind="ckdt", path=APP_DICTS / "langpack-el.zip",
                    tier="app-bundled langpack-el (CKDT v2, 39,860 words)"),
    notes="accents/diaeresis stripped (no accented key exists; the layout's "
          "accent_aigu/trema/grave are dead keys); word-final σ restored to ς.",
)

#: Ukrainian — layout present, **no app lexicon**; wordfreq supplies one on the
#: app's own rank scale.  ї and ґ live on ``loc`` corner slots of the ЙЦУКЕН-uk
#: grid, so they are not swipe-typeable and words needing them are rejected.
UK = ScriptSpec(
    code="uk", name="Ukrainian",
    app_xml="cyrl_jcuken_uk.xml", layout_json="layouts/uk_jcuken.json",
    letters="абвгдежзийклмнопрстуфхцчшщьюяєі",
    corner_only="їґ",
    lexicon=Lexicon(kind="wordfreq", lang="uk",
                    tier="wordfreq top-50k, app rank formula — NOT the app's "
                         "curated build_wordlist output (no spell oracles)"),
    notes="ї/ґ are corner-only on this layout: rejected, cost measured.",
)

#: Bulgarian (БДС) — 30 centre keys, exactly the Bulgarian alphabet.
BG = ScriptSpec(
    code="bg", name="Bulgarian",
    app_xml="cyrl_ueishsht.xml", layout_json="layouts/bg_bds.json",
    letters="абвгдежзийклмнопрстуфхцчшщъьюя",
    corner_only="ѝ",
    folds={"ѝ": "и"},
    lexicon=Lexicon(kind="wordfreq", lang="bg",
                    tier="wordfreq top-50k, app rank formula"),
    notes="ѝ (grave-и, a disambiguating pronoun spelling) folds to и.",
)

#: Macedonian — 31 centre keys, exactly the Macedonian alphabet.
MK = ScriptSpec(
    code="mk", name="Macedonian",
    app_xml="cyrl_lynyertdz_mk.xml", layout_json="layouts/mk_lynyertdz.json",
    letters="абвгдежзиклмнопрстуфхцчшѓѕјљњќџ",
    corner_only="ѐѝ",
    folds={"ѐ": "е", "ѝ": "и"},
    lexicon=Lexicon(kind="wordfreq", lang="mk",
                    tier="wordfreq top-50k, app rank formula"),
    notes="ѐ/ѝ (accented disambiguators) fold to е/и.",
)

#: Serbian — 30 centre keys, exactly the Serbian Cyrillic alphabet, zero
#: corner-only letters.  wordfreq has no ``sr``; its Serbo-Croatian list
#: (``sh``) is mostly Latin script, so the Cyrillic words are taken from it by
#: projection and the yield is reported.
SR = ScriptSpec(
    code="sr", name="Serbian",
    app_xml="cyrl_lynyertz_sr.xml", layout_json="layouts/sr_lynyertz.json",
    letters="абвгдежзиклмнопрстуфхцчшђјљњћџ",
    corner_only="",
    lexicon=Lexicon(kind="wordfreq", lang="sh",
                    tier="wordfreq 'sh' (Serbo-Croatian) Cyrillic subset, app "
                         "rank formula"),
    notes="wordfreq has no 'sr'; 'sh' is predominantly Latin — Cyrillic yield "
          "is measured, not assumed.",
)

#: Hebrew — 27 centre keys (22 letters + 5 final forms), zero corner-only
#: letters.  An abjad and RTL, neither of which the geometry cares about: the
#: XML is authored in visual left-to-right key order, so the coordinate walk is
#: unchanged (app audit §5 — there is no RTL branch anywhere in the geometry or
#: swipe path).
HE = ScriptSpec(
    code="he", name="Hebrew",
    app_xml="hebr_1_il.xml", layout_json="layouts/he_1.json",
    letters="אבגדהוזחטיךכלםמןנסעףפץצקרשת",
    corner_only="",
    strip_combining=True,
    lexicon=Lexicon(kind="wordfreq", lang="he",
                    tier="wordfreq top-50k, app rank formula"),
    notes="niqqud stripped (not keys); final forms are distinct keys and are "
          "kept distinct, exactly as Hebrew orthography requires.",
)

#: Russian — **already delivered** (Phase I-B / J, exported 2026-08-18).  It is
#: reproduced here only so Phase O can run its synthesis holdout through the
#: *identical* code path as the new scripts and thereby measure the one number
#: nothing else can supply: how far a synthesis-holdout score sits from the same
#: model's score on REAL swipes.  Alphabet, layout and projection are
#: byte-for-byte what ``prepare_yandex.project_ru`` and
#: ``layouts/ru_jcuken_default.json`` already use (no NFD — it would decompose
#: й into и + breve).
RU = ScriptSpec(
    code="ru", name="Russian",
    app_xml="cyrl_jcuken_ru.xml", layout_json="layouts/ru_jcuken_default.json",
    letters="абвгдежзийклмнопрстуфхцчшщыьэюя",
    corner_only="ъёєіїўґ",
    folds={"ё": "е", "ъ": "ь"},
    lexicon=Lexicon(kind="ckdt", path=APP_DICTS / "langpack-ru.zip",
                    tier="app-importable langpack-ru (CKDT v2, 50,000 words)"),
    notes="the Phase-O calibration anchor: the only script with BOTH a "
          "synthesis holdout and a real eval corpus.",
)

SCRIPTS: Dict[str, ScriptSpec] = {s.code: s for s in (RU, EL, UK, BG, MK, SR, HE)}


def get(code: str) -> ScriptSpec:
    if code not in SCRIPTS:
        raise SystemExit(f"unknown script {code!r}; have {sorted(SCRIPTS)}")
    return SCRIPTS[code]


def main() -> int:
    import argparse
    import json
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--code", default="", help="one script, or empty for all")
    ap.add_argument("--max-len", type=int, default=32)
    args = ap.parse_args()
    codes = [args.code] if args.code else sorted(SCRIPTS)
    for c in codes:
        s = get(c)
        lex, st = s.load_lexicon(args.max_len)
        top = sorted(lex, key=lambda kv: -kv[1])[:8]
        print(json.dumps({
            "code": s.code, "name": s.name, "n_letters": len(s.letters),
            "letters": s.letters, "corner_only": s.corner_only,
            "lexicon_kind": s.lexicon.kind, "tier": s.lexicon.tier,
            "stats": st, "top": top,
        }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
