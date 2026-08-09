#!/usr/bin/env python3
"""Lexicon selection for the eval harnesses — the AOSP tuning trie vs the app trie.

Every accuracy number in phases A–F was decoded against **one** lexicon: the
AOSP ``en_wordlist.combined`` normalized a–z with the STRIP policy, which yields
a **146,964-word** trie whose raw frequencies span ``1..222`` (``log_freq`` in
``[0.0, 5.40]``). The E1 preset's ``lambda = 1.1`` was fitted against that scale.

The app ships something different (`APP_INTEGRATION_PLAN.md` D4/O3): the bundled
``dictionaries/en_enhanced.json``, a flat ``{word: int_freq}`` map of **98,140**
entries whose values are already compressed onto a **134..255** byte scale
(``log_freq`` in ``[4.898, 5.541]``). Two things therefore change at once when the
app trie is substituted — **coverage** (fewer words: more OOV, but also fewer
confusables) and **frequency scale** (the spread of ``log_freq`` collapses from
5.40 to 0.64, so the same ``lambda`` buys ~8× less ranking signal).

This module is the single place that maps a ``--vocab-kind`` onto a loader, so
``eval_beam.py`` and ``sweep_scoring.py`` (and anything downstream of them) cannot
drift apart on which trie a number was measured against.

Kinds
-----
``combined``
    :func:`futo_decoder_eval.load_combined_vocab` — the AOSP wordlist, STRIP
    policy. The tuning/reporting trie for every committed number.
``json``
    :func:`futo_decoder_eval.load_flat_json_vocab` — the vendored flat-JSON
    loader, **DROP** policy (a word containing any non-a–z character is skipped).
    This is the ``--vocab-json`` semantics named in O3 and the exact policy of
    the app's ``CtcLexiconTrie.loadFromFrequencyMap``.
``json-strip``
    Same file, **STRIP** policy — the app's
    ``CtcLexiconTrie.loadStrippingNonAlphabet``, which `APP_INTEGRATION_PLAN.md`
    D4 selects. On ``en_enhanced.json`` the two differ by 148 words: the file
    carries no apostrophes (it is already a–z-aliased), so stripping only turns
    207 accented entries into 148 unique consonant skeletons (``café`` → ``caf``).
    Both are provided so the choice is measured rather than assumed.

The frequency semantics are identical in all three and identical to the Kotlin
port: ``log_freq = ln(freq + 1e-10)``, retained as the max over surface forms
that normalize to the same a–z string, with the ``0.0``-sentinel overwrite quirk
(`CtcLexiconTrie.insert`).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from futo_decoder_eval import (LexTrie, load_combined_vocab,  # noqa: E402
                               load_flat_json_vocab)

#: Accepted ``--vocab-kind`` values, in the order argparse should show them.
VOCAB_KINDS = ("combined", "json", "json-strip")

#: Default for every harness: the trie every committed number was measured on.
DEFAULT_VOCAB_KIND = "combined"


def load_flat_json_vocab_stripping(path: Path) -> LexTrie:
    """Flat ``{word: freq}`` JSON → trie, STRIPPING out-of-alphabet characters.

    The app-side policy (`CtcLexiconTrie.loadStrippingNonAlphabet`): a word is
    reduced to its a–z characters rather than skipped, so apostrophe/hyphen
    surface forms stay reachable for an a–z-only CTC model. Words that strip to
    the empty string are dropped; ``freq <= 0`` is floored to ``1.0``, matching
    :func:`futo_decoder_eval.load_flat_json_vocab`.
    """
    trie = LexTrie()
    data = json.loads(Path(path).read_text())
    for word, freq in data.items():
        wl = "".join(c for c in str(word).lower() if "a" <= c <= "z")
        if wl:
            f = float(freq)
            trie.insert(wl, f if f > 0 else 1.0)
    return trie


def load_vocab(path: Path, kind: str = DEFAULT_VOCAB_KIND) -> LexTrie:
    """Build the trie named by *kind* from *path*.

    :raises SystemExit: on an unknown *kind* (argparse normally prevents this,
        but the helper is also called programmatically).
    """
    if kind == "combined":
        return load_combined_vocab(path)
    if kind == "json":
        return load_flat_json_vocab(path)
    if kind == "json-strip":
        return load_flat_json_vocab_stripping(path)
    raise SystemExit(f"unknown --vocab-kind {kind!r}; expected one of {VOCAB_KINDS}")


def add_argument(ap) -> None:
    """Register the shared ``--vocab-kind`` flag on a script's parser."""
    ap.add_argument("--vocab-kind", choices=VOCAB_KINDS, default=DEFAULT_VOCAB_KIND,
                    dest="vocab_kind",
                    help="lexicon format/policy for --vocab: 'combined' = the AOSP "
                         "wordlist STRIP trie every committed number uses; 'json' = "
                         "the app's flat en_enhanced.json, DROP policy; 'json-strip' "
                         "= the same file with the app's STRIP policy (O3)")
