#!/usr/bin/env python3
"""Extract per-key geometry from a CleverKeys app layout XML into the campaign
layout-json schema (``en_qwerty.json`` / ``layouts/ru_jcuken_default.json``).

Why this file exists
--------------------
Phase I-B's Cyrillic geometry came from the **Yandex corpus' own embedded
grids** — the corpus told us where the keys were.  Phase O generalises to
scripts with *no corpus at all*, so the only authority on key positions is the
app itself.  That is in fact the better authority: at runtime the app feeds the
model ``layout_keys`` computed from exactly these XML files, so a layout json
extracted here is **deployment-exact** rather than merely plausible.

Geometry — a line-for-line replica of the app's own hit-test walk
-----------------------------------------------------------------
``a11y/KeyboardGeometry.computeKeyRects`` (itself a verbatim transplant of
``Keyboard2View.getKeyAtPosition``) is the single source of truth in the app for
where a key is, and ``swipe/CtcEngineAdapter.buildMappedLayout`` is what turns
those rects into the model's ``layout_keys``.  Both are replicated here:

* x cursor starts at ``marginLeft``; per key ``xLeft = x + key.shift*keyWidth``,
  ``xRight = xLeft + key.width*keyWidth``, then ``x = xRight``;
* y cursor starts at ``marginTop``; per row ``yTop = y + row.shift*rowHeight``,
  ``yBottom = y + (row.shift + row.height)*rowHeight``, then ``y = yBottom``;
* attribute defaults, from ``KeyboardData.Row.parse`` / ``Key.parse``:
  row ``height`` 1 (clamped ``max(h, 0.5)``), row ``shift`` 0 (clamped ``>= 0``),
  row ``scale`` 0 = off (when > 0 every key width is multiplied by
  ``scale / row_keys_width``, i.e. ``Row.updateWidth``), key ``width`` 1
  (clamped ``>= 0``), key ``shift`` 0 (clamped ``>= 0``);
* the centre value of a key is attribute ``key0`` *or* its synonym ``c``
  (``Key.parse`` reads ``get_key_attr(parser, "key0", "c")``);
* **normalisation is over the bounding box of the LETTER keys only** —
  ``buildMappedLayout`` accumulates ``left/top/right/bottom`` while walking the
  letter rects and divides by that box, it does not use the keyboard width.
  Because every letter row here has the same height, the row centres land on the
  ``(2r+1)/2R`` family (0.167/0.5/0.833 for three rows) — the same frame every
  trained geometry in this campaign lives in (``PHASE_I_DATA.md`` §4).

Since margins and the unit sizes cancel in that normalisation, they are fixed at
``marginLeft = marginTop = 0``, ``keyWidth = rowHeight = 1``.

What is deliberately NOT modelled (and why it cannot matter here)
-----------------------------------------------------------------
* the built-in **bottom row** (``bottom_row="true"`` by default) and the
  **numpad**: they carry no letter keys, and the normalisation box is built from
  letter keys only, so they cancel;
* ``embedded_number_row``: adds a *digit* row above — same argument;
* ``locale_extra_keys`` / ``loc `` -prefixed values: keys the app may add later
  for a specific locale.  A ``loc `` value is skipped (it is not on the board by
  default), which is the conservative choice — see ``--include-loc`` to measure
  the difference;
* ``<modmap>``: it rewrites the shift/fn *layers*, never ``key0``.

Usage::

    python3 app_layout.py --xml grek_qwerty.xml --out layouts/el_qwerty.json
    python3 app_layout.py --census            # every non-Latin app layout
"""
from __future__ import annotations

import argparse
import json
import os
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

#: The app repo is a READ-ONLY reference for this campaign.  Overridable via
#: ``CK_APP_REPO`` so the extractor runs on any checkout of the app (the
#: Termux device keeps it at ``~/git/swype/cleverkeys``); the default is the
#: original WSL path every Phase-O json was generated against.
APP_REPO = Path(os.environ.get("CK_APP_REPO",
                               "/home/will/git/swype/CleverKeys")).expanduser()

#: The layout dir that actually SHIPS.  ``build.gradle``'s ``copyLayoutDefinitions``
#: copies ``src/main/layouts/*.xml`` into ``build/generated/layouts/res/raw``;
#: ``srcs/layouts`` is the upstream-style source dir the *tests* read.  The two
#: are byte-identical for every non-Latin layout except ``grek_qwerty.xml``,
#: which differs only in its ``script`` attribute (``greek`` in ``srcs``,
#: ``latin`` in the shipped copy — an app bug, PHASE_O.md §1.3).  Geometry is
#: unaffected, but this module reads the shipped copy on principle.
APP_LAYOUTS = APP_REPO / "src" / "main" / "layouts"

#: The eight corner slots, in ``KeyboardData.Key.parse`` order (with the compass
#: synonyms the XML may use instead).  A corner value that is a single letter is
#: a *typeable* letter of the layout — by a directional flick off the host key —
#: but it is **NOT swipe-typeable and must never become an emission slot**: the
#: app's own alias table (`GeoLayoutFixtures`) gives such a letter the host key's
#: centroid, so two letters would share one coordinate and the encoder could not
#: separate them.  :func:`corner_letters` reports them so a script's projection
#: can fold or reject the words that need them, with the cost measured rather
#: than assumed (PHASE_O.md §1.2).
CORNER_SLOTS = (("key1", "nw"), ("key2", "ne"), ("key3", "sw"), ("key4", "se"),
                ("key5", "w"), ("key6", "e"), ("key7", "n"), ("key8", "s"))

HERE = Path(__file__).resolve().parent


def _fattr(el: ET.Element, name: str, default: float) -> float:
    """``KeyboardData.attribute_float`` — missing attribute yields *default*."""
    v = el.get(name)
    if v is None:
        return default
    return float(v)


def _centre_value(key: ET.Element) -> Optional[str]:
    """``key0`` or its synonym ``c`` (``Key.parse`` accepts both spellings)."""
    v = key.get("key0")
    if v is None:
        v = key.get("c")
    return v


def is_letter(value: Optional[str], include_loc: bool = False) -> Optional[str]:
    """The single letter a centre value denotes, or ``None``.

    Named keys (``shift``, ``backspace``, ``accent_aigu``, …) are multi-character
    and rejected.  Punctuation and digits are rejected by the unicode category
    test.  ``loc `` -prefixed values name a key that is *not* on the board unless
    the locale adds it, so they are skipped unless *include_loc*.
    """
    if value is None:
        return None
    if value.startswith("loc "):
        if not include_loc:
            return None
        value = value[4:]
    if len(value) != 1:
        return None
    if not unicodedata.category(value).startswith("L"):
        return None
    return value.lower()


class KeyRect:
    """One key's hit-test cell in layout units (``KeyboardGeometry.KeyRect``)."""

    __slots__ = ("value", "left", "top", "right", "bottom")

    def __init__(self, value: Optional[str], left: float, top: float,
                 right: float, bottom: float) -> None:
        self.value, self.left, self.top = value, left, top
        self.right, self.bottom = right, bottom

    @property
    def cx(self) -> float:
        return (self.left + self.right) / 2.0

    @property
    def cy(self) -> float:
        return (self.top + self.bottom) / 2.0


def compute_key_rects(xml_path: Path) -> Tuple[ET.Element, List[KeyRect]]:
    """Parse a layout XML and walk it exactly as ``computeKeyRects`` does."""
    root = ET.parse(xml_path).getroot()
    if root.tag != "keyboard":
        raise SystemExit(f"{xml_path}: root tag {root.tag!r}, expected 'keyboard'")
    rects: List[KeyRect] = []
    y = 0.0
    for row in root:
        if row.tag != "row":          # <modmap> and comments carry no geometry
            continue
        height = max(_fattr(row, "height", 1.0), 0.5)
        shift = max(_fattr(row, "shift", 0.0), 0.0)
        scale = _fattr(row, "scale", 0.0)
        keys = [k for k in row if k.tag == "key"]
        widths = [max(_fattr(k, "width", 1.0), 0.0) for k in keys]
        shifts = [max(_fattr(k, "shift", 0.0), 0.0) for k in keys]
        if scale > 0.0:               # Row.updateWidth: scale key widths only
            row_w = sum(w + s for w, s in zip(widths, shifts))
            if row_w > 0.0:
                widths = [w * (scale / row_w) for w in widths]
                shifts = [s * (scale / row_w) for s in shifts]
        y_top, y_bottom = y + shift, y + shift + height
        x = 0.0
        for key, w, s in zip(keys, widths, shifts):
            x_left = x + s
            x_right = x_left + w
            rects.append(KeyRect(_centre_value(key), x_left, y_top, x_right, y_bottom))
            x = x_right
        y = y_bottom
    return root, rects


def extract(xml_path: Path, include_loc: bool = False,
            letters_order: Optional[Sequence[str]] = None) -> Dict[str, object]:
    """One app layout XML -> the campaign layout-json dict.

    First occurrence of a letter wins, in row-major order — the same
    ``if (seen[i]) continue`` rule ``buildMappedLayout`` applies.
    """
    root, rects = compute_key_rects(xml_path)
    seen: Dict[str, KeyRect] = {}
    for r in rects:
        letter = is_letter(r.value, include_loc)
        if letter is None or letter in seen:
            continue
        seen[letter] = r
    if not seen:
        raise SystemExit(f"{xml_path}: no letter keys found")
    left = min(r.left for r in seen.values())
    right = max(r.right for r in seen.values())
    top = min(r.top for r in seen.values())
    bottom = max(r.bottom for r in seen.values())
    w, h = right - left, bottom - top
    if w <= 0 or h <= 0:
        raise SystemExit(f"{xml_path}: degenerate letter box {w}x{h}")
    order = list(letters_order) if letters_order is not None else sorted(seen)
    missing = [c for c in order if c not in seen]
    if missing:
        raise SystemExit(f"{xml_path}: requested letters missing from layout: "
                         f"{''.join(missing)}")
    keys = [{
        "letter": c,
        "cx": (seen[c].cx - left) / w,
        "cy": (seen[c].cy - top) / h,
        "rx": (seen[c].right - seen[c].left) / 2.0 / w,
        "ry": (seen[c].bottom - seen[c].top) / 2.0 / h,
    } for c in order]
    return {
        "name": Path(xml_path).stem,
        "letters": "".join(order),
        "source": {
            "app_xml": str(Path(xml_path).relative_to(APP_REPO)),
            "keyboard_name": root.get("name"),
            "script": root.get("script"),
            "include_loc": include_loc,
            "letter_box_units": [round(w, 6), round(h, 6)],
        },
        "keys": keys,
    }


def corner_letters(xml_path: Path, include_loc: bool = False) -> Dict[str, str]:
    """``{letter: host_centre_letter_or_value}`` for letters only on corner slots.

    These are typeable by a flick but not by a swipe (see :data:`CORNER_SLOTS`).
    Only letters that have no centre key of their own are reported.
    """
    root = ET.parse(xml_path).getroot()
    centres = set()
    corners: Dict[str, str] = {}
    for row in root:
        if row.tag != "row":
            continue
        for key in row:
            if key.tag != "key":
                continue
            c = is_letter(_centre_value(key), include_loc)
            if c is not None:
                centres.add(c)
    for row in root:
        if row.tag != "row":
            continue
        for key in row:
            if key.tag != "key":
                continue
            host = _centre_value(key) or "?"
            for a, b in CORNER_SLOTS:
                v = key.get(a)
                if v is None:
                    v = key.get(b)
                # corner letters are usually 'loc '-prefixed placeholders, so
                # they are read with include_loc forced on
                letter = is_letter(v, include_loc=True)
                if letter is not None and letter not in centres:
                    corners.setdefault(letter, host)
    return corners


def census(include_loc: bool = False) -> List[Dict[str, object]]:
    """Every app layout, with its script and letter inventory (the O1 table)."""
    rows: List[Dict[str, object]] = []
    for xml in sorted(APP_LAYOUTS.glob("*.xml")):
        try:
            root, rects = compute_key_rects(xml)
        except Exception as exc:                      # malformed layout
            rows.append({"file": xml.name, "error": str(exc)})
            continue
        letters: List[str] = []
        for r in rects:
            c = is_letter(r.value, include_loc)
            if c is not None and c not in letters:
                letters.append(c)
        corners = corner_letters(xml, include_loc)
        rows.append({
            "file": xml.name,
            "script": root.get("script"),
            "name": root.get("name"),
            "n_letters": len(letters),
            "letters": "".join(sorted(letters)),
            "rows": sum(1 for el in root if el.tag == "row"),
            "corner_only": "".join(sorted(corners)),
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--xml", help="layout file name under the app's srcs/layouts "
                                  "(or an absolute path)")
    ap.add_argument("--out", type=Path, help="write the layout json here")
    ap.add_argument("--letters", default="",
                    help="explicit alphabet/slot order; default = sorted")
    ap.add_argument("--include-loc", action="store_true",
                    help="also take 'loc '-prefixed centre values (keys the app "
                         "only adds for some locales)")
    ap.add_argument("--census", action="store_true",
                    help="print the per-layout letter inventory as JSON")
    ap.add_argument("--non-latin-only", action="store_true")
    args = ap.parse_args()

    if args.census:
        rows = census(args.include_loc)
        if args.non_latin_only:
            rows = [r for r in rows if r.get("script") not in ("latin", None)]
        print(json.dumps(rows, ensure_ascii=False, indent=1))
        return 0

    if not args.xml:
        ap.error("--xml or --census required")
    path = Path(args.xml)
    if not path.is_absolute():
        path = APP_LAYOUTS / path
    obj = extract(path, args.include_loc, list(args.letters) or None)
    text = json.dumps(obj, ensure_ascii=False, indent=1)
    if args.out:
        out = args.out if args.out.is_absolute() else HERE / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
        print(f"{path.name}: {len(obj['letters'])} letters -> {out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
