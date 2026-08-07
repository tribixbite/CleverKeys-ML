#!/usr/bin/env python3
"""Fetch and verify the COMPLETE "How We Swipe" shape-writing dataset release.

Reference
---------
L. A. Leiva, S. Kim, W. Cui, X. Bi, A. Oulasvirta.
"How We Swipe: A Large-scale Shape-writing Dataset and Empirical Findings."
Proc. MobileHCI, 2021.

Canonical source
----------------
OSF project ``sj67f`` -- "How-We-Swipe Shape-writing Dataset"
(https://osf.io/sj67f/).  ``wy9q8`` is *not* the project id: it is the OSF GUID
of the ``metadata.tsv`` **file** inside that project, which is why
``api.osf.io/v2/nodes/wy9q8/`` 404s.

License: **MIT** (declared on the OSF node; copyright 2021 L. A. Leiva,
S. Kim, W. Cui, X. Bi, A. Oulasvirta).  Permissive -- research and commercial
use, modification and redistribution allowed with attribution.

The GitHub repo https://github.com/luileito/swipetest is the *study software*
(PHP/JS web experiment + processing scripts), NOT a data mirror -- it is 818 kB
and contains no logs.  So OSF is the only source for the data; there is no
faster mirror.  Direct per-file endpoints ``https://osf.io/download/<guid>/``
are used here (fast, ~10 s for the whole 83 MB release) rather than the OSF
zip-the-whole-node endpoint, which is slow and rate-limited.

Release contents (5 files + project wiki)
-----------------------------------------
======================  ======  ==========  =========================================
file                    guid    size        contents
======================  ======  ==========  =========================================
swipelogs.zip           x4yjn   69.9 MB     1338 ``<uid>.log`` + 1338 ``<uid>.json``
metadata.tsv            wy9q8   103 kB      909-user filtered subset used in the paper
stats-words.tsv         vpc5a   10.4 MB     per-word length/time/DTW stats
stats-sentences.tsv     ujpdc   3.3 MB      per-sentence WPM/WER stats
wordfreq.txt            efzmp   177 kB      word frequency list of the phrase sets
wiki (README)           pk9qd   13.5 kB     full column-level data dictionary
======================  ======  ==========  =========================================

Note the two participant populations, which are easy to conflate:

* **1338** users have a ``.log`` + ``.json`` in ``swipelogs.zip``  (the raw release).
* **909** users are in ``metadata.tsv`` -- the paper's *filtered* subset, kept by
  the authors' criteria: >=5 sentences submitted, mobile device
  (``maxTouchPoints > 2``, ``width < 600``, ``500 < height < 900``), and no
  ``NA`` in the demographics.

The per-user ``.json`` carries ``englishLevel`` for **all 1338** users, so
native-speaker filtering does not require ``metadata.tsv``.

Usage
-----
    python fetch_hws_full.py                     # download + extract + verify
    python fetch_hws_full.py --analyze           # + participant/yield breakdown
    python fetch_hws_full.py --dest DIR --force  # re-download over an existing tree
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import urllib.request
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #

OSF_NODE = "sj67f"
OSF_API_NODE = f"https://api.osf.io/v2/nodes/{OSF_NODE}/"
OSF_API_FILES = f"https://api.osf.io/v2/nodes/{OSF_NODE}/files/osfstorage/?page%5Bsize%5D=100"
OSF_WIKI_CONTENT = "https://api.osf.io/v2/wikis/pk9qd/content/"


@dataclass(frozen=True)
class Asset:
    """One downloadable file of the release."""

    name: str
    guid: str
    size: int  # bytes, as published by the OSF API on 2026-08-07

    @property
    def url(self) -> str:
        return f"https://osf.io/download/{self.guid}/"


ASSETS: Tuple[Asset, ...] = (
    Asset("swipelogs.zip", "x4yjn", 69_925_564),
    Asset("metadata.tsv", "wy9q8", 103_584),
    Asset("stats-words.tsv", "vpc5a", 10_433_773),
    Asset("stats-sentences.tsv", "ujpdc", 3_284_042),
    Asset("wordfreq.txt", "efzmp", 177_062),
)

EXPECTED_PARTICIPANTS = 1338
EXPECTED_METADATA_ROWS = 909
DEFAULT_DEST = Path.home() / "ctc-train" / "data" / "hws_full"

# Column order of the space-separated .log files (documented in the OSF wiki).
LOG_COLUMNS = (
    "sentence timestamp keyb_width keyb_height event "
    "x_pos y_pos x_radius y_radius angle word is_err"
).split()
COL_EVENT, COL_X, COL_Y, COL_WORD, COL_IS_ERR = 4, 5, 6, 10, 11


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "cleverkeys-ml/fetch_hws_full"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return resp.read()


def resolve_assets_live() -> List[Asset]:
    """Re-resolve the asset list from the OSF API (guards against a re-upload)."""
    payload = json.loads(_get(OSF_API_FILES))
    live = [
        Asset(f["attributes"]["name"], f["attributes"]["guid"], f["attributes"]["size"])
        for f in payload["data"]
        if f["attributes"]["kind"] == "file"
    ]
    known = {a.name: a for a in ASSETS}
    for a in live:
        pinned = known.get(a.name)
        if pinned and (pinned.guid != a.guid or pinned.size != a.size):
            print(
                f"  ! {a.name} changed upstream: guid {pinned.guid}->{a.guid}, "
                f"size {pinned.size}->{a.size}",
                file=sys.stderr,
            )
    return live


def download(dest: Path, force: bool = False, live: bool = True) -> None:
    """Download every release file plus the wiki README into ``dest``."""
    dest.mkdir(parents=True, exist_ok=True)
    assets = resolve_assets_live() if live else list(ASSETS)

    for asset in assets:
        target = dest / asset.name
        if target.exists() and target.stat().st_size == asset.size and not force:
            print(f"  = {asset.name} ({asset.size:,} B) already present")
            continue
        print(f"  > {asset.name} <- {asset.url}")
        blob = _get(asset.url)
        if len(blob) != asset.size:
            raise RuntimeError(
                f"{asset.name}: expected {asset.size:,} bytes, got {len(blob):,}"
            )
        target.write_bytes(blob)
        print(f"    sha256 {hashlib.sha256(blob).hexdigest()[:16]}...  {len(blob):,} B")

    readme = dest / "README_osf_wiki.md"
    if force or not readme.exists():
        print(f"  > README_osf_wiki.md <- {OSF_WIKI_CONTENT}")
        readme.write_bytes(_get(OSF_WIKI_CONTENT))

    license_note = dest / "LICENSE.txt"
    if force or not license_note.exists():
        node = json.loads(_get(OSF_API_NODE))["data"]["attributes"]
        lic = node["node_license"]
        license_note.write_text(
            "How-We-Swipe Shape-writing Dataset\n"
            f"Source: https://osf.io/{OSF_NODE}/\n"
            "License (as declared on the OSF node): MIT License\n"
            f"Copyright (c) {lic['year']} {', '.join(lic['copyright_holders'])}\n\n"
            "Permission is hereby granted, free of charge, to any person obtaining a copy\n"
            "of this software and associated documentation files (the \"Software\"), to deal\n"
            "in the Software without restriction, including without limitation the rights\n"
            "to use, copy, modify, merge, publish, distribute, sublicense, and/or sell\n"
            "copies of the Software, and to permit persons to whom the Software is\n"
            "furnished to do so, subject to the following conditions:\n\n"
            "The above copyright notice and this permission notice shall be included in\n"
            "all copies or substantial portions of the Software.\n\n"
            "THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\n"
            "IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\n"
            "FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\n"
            "AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\n"
            "LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\n"
            "OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN\n"
            "THE SOFTWARE.\n\n"
            "Cite: L. A. Leiva, S. Kim, W. Cui, X. Bi, A. Oulasvirta. "
            "How We Swipe: A Large-scale Shape-writing Dataset and Empirical Findings. "
            "Proc. MobileHCI, 2021.\n"
        )


def extract(dest: Path, force: bool = False) -> Path:
    """Unzip ``swipelogs.zip`` into ``dest/swipelogs``."""
    logs_dir = dest / "swipelogs"
    if logs_dir.is_dir() and not force:
        n = sum(1 for _ in logs_dir.glob("*.log"))
        if n == EXPECTED_PARTICIPANTS:
            print(f"  = swipelogs/ already extracted ({n} logs)")
            return logs_dir
        shutil.rmtree(logs_dir)
    elif logs_dir.is_dir():
        shutil.rmtree(logs_dir)
    print("  > extracting swipelogs.zip")
    with zipfile.ZipFile(dest / "swipelogs.zip") as zf:
        zf.extractall(dest)
    return logs_dir


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #


def load_user_meta(logs_dir: Path) -> Dict[str, dict]:
    """Per-user ``.json`` demographics, keyed by uid. Available for all 1338 users."""
    meta: Dict[str, dict] = {}
    for path in logs_dir.glob("*.json"):
        with path.open(errors="replace") as fh:
            meta[path.stem] = json.load(fh)
    return meta


def load_metadata_tsv(dest: Path) -> Dict[str, dict]:
    """The paper's 909-user filtered subset, keyed by uid."""
    with (dest / "metadata.tsv").open() as fh:
        return {r["uid"]: r for r in csv.DictReader(fh, delimiter="\t")}


def english_level(user_json: dict) -> str:
    """Normalized self-reported English level: native|advanced|intermediate|beginner|na."""
    return (user_json.get("englishLevel") or "NA").strip().lower()


def verify(dest: Path, logs_dir: Path) -> Dict[str, object]:
    """Structural completeness check. Raises on a short/partial download."""
    logs = {p.stem for p in logs_dir.glob("*.log")}
    metas = {p.stem for p in logs_dir.glob("*.json")}
    tsv = load_metadata_tsv(dest)

    problems: List[str] = []
    if len(logs) != EXPECTED_PARTICIPANTS:
        problems.append(f"{len(logs)} .log files, expected {EXPECTED_PARTICIPANTS}")
    if logs != metas:
        problems.append(f"{len(logs ^ metas)} uids lack a matching .log/.json pair")
    if len(tsv) != EXPECTED_METADATA_ROWS:
        problems.append(f"metadata.tsv has {len(tsv)} rows, expected {EXPECTED_METADATA_ROWS}")
    if not set(tsv).issubset(logs):
        problems.append(f"{len(set(tsv) - logs)} metadata.tsv uids have no log file")
    if problems:
        raise RuntimeError("incomplete release: " + "; ".join(problems))

    print(f"  ok  {len(logs)} participants, .log/.json paired 1:1")
    print(f"  ok  metadata.tsv {len(tsv)} rows, all uids present in swipelogs/")
    return {"logs": logs, "tsv": tsv}


# --------------------------------------------------------------------------- #
# Yield analysis -- exact replica of the existing HWS ingestion filter
# --------------------------------------------------------------------------- #


def count_usable_traces(log_path: Path) -> Tuple[int, int]:
    """Count traces a log yields under the *existing* CleverKeys HWS filter.

    Mirrors ``neural-swipe-typing/process_swipe_logs.py`` exactly:
      * drop malformed rows (<12 space-separated fields);
      * drop every row with ``is_err == 1``;
      * a trace runs from a ``touchstart`` to its ``touchend`` (or to the next
        ``touchstart``), and needs >= 3 points to be kept.
    Then the downstream ``len(word) == 1`` drop is applied.

    Returns ``(traces_before_len1_drop, traces_after_len1_drop)``.
    """
    before = after = 0
    current: List[str] = []  # words are constant within a trace; only length matters
    word: Optional[str] = None

    def flush() -> None:
        nonlocal before, after, current
        if len(current) >= 3 and word is not None:
            before += 1
            if len(word) > 1:
                after += 1
        current = []

    with log_path.open(errors="replace") as fh:
        fh.readline()  # header
        for line in fh:
            parts = line.split()
            if len(parts) < 12:
                continue
            try:
                if int(parts[COL_IS_ERR]) == 1:
                    continue
            except ValueError:
                continue
            event = parts[COL_EVENT]
            if event == "touchstart":
                flush()
                current = [event]
                word = parts[COL_WORD]
            elif event in ("touchmove", "touchend") and current:
                current.append(event)
                if event == "touchend":
                    flush()
    flush()
    return before, after


def analyze(dest: Path, logs_dir: Path, local_dir: Optional[Path]) -> None:
    """Participant breakdown by English level + trace yield per group."""
    meta = load_user_meta(logs_dir)
    tsv = load_metadata_tsv(dest)
    all_uids = set(meta)

    local_uids: Set[str] = set()
    if local_dir and local_dir.is_dir():
        local_uids = {p.stem for p in local_dir.glob("*.log")}

    print("\n=== participants by self-reported English level (per-user .json) ===")
    counts = Counter(english_level(m) for m in meta.values())
    for level in ("native", "advanced", "intermediate", "beginner", "na"):
        if counts[level]:
            print(f"  {level:13s} {counts[level]:5d}")

    yields: Dict[str, Tuple[int, int]] = {}
    for uid in sorted(all_uids):
        yields[uid] = count_usable_traces(logs_dir / f"{uid}.log")

    groups: Dict[str, Set[str]] = {
        "ALL (release)": all_uids,
        "native": {u for u in all_uids if english_level(meta[u]) == "native"},
        "native+advanced": {
            u for u in all_uids if english_level(meta[u]) in ("native", "advanced")
        },
        "intermediate": {u for u in all_uids if english_level(meta[u]) == "intermediate"},
        "beginner": {u for u in all_uids if english_level(meta[u]) == "beginner"},
        "paper subset (909)": set(tsv) & all_uids,
    }
    if local_uids:
        groups["local pool"] = local_uids & all_uids
        groups["local native+adv"] = {
            u
            for u in local_uids & all_uids
            if english_level(meta[u]) in ("native", "advanced")
        }
        groups["NEW (not local)"] = all_uids - local_uids
        groups["NEW native+adv"] = {
            u
            for u in all_uids - local_uids
            if english_level(meta[u]) in ("native", "advanced")
        }

    print("\n=== trace yield under the existing HWS filter ===")
    print(f"{'group':22s} {'users':>6s} {'is_err=0':>10s} {'+len>1':>10s}")
    for name, uids in groups.items():
        b = sum(yields[u][0] for u in uids)
        a = sum(yields[u][1] for u in uids)
        print(f"{name:22s} {len(uids):6d} {b:10,d} {a:10,d}")

    if local_uids:
        missing = local_uids - all_uids
        print(f"\n  local logs: {len(local_uids)}; contained in release: "
              f"{len(local_uids & all_uids)}; absent from release: {len(missing)}")

    total = sum(p.stat().st_size for p in dest.rglob("*") if p.is_file())
    print(f"\n  disk usage: {total / 1e9:.2f} GB under {dest}")


# --------------------------------------------------------------------------- #


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="download directory")
    ap.add_argument("--force", action="store_true", help="re-download / re-extract")
    ap.add_argument("--analyze", action="store_true", help="print the yield breakdown")
    ap.add_argument(
        "--local-logs",
        type=Path,
        default=Path("/home/will/git/swype/neural-swipe-typing/data/how_we_swipe_logs/swipetraces"),
        help="existing partial HWS log dir, for overlap reporting",
    )
    ap.add_argument("--offline", action="store_true", help="skip the live OSF API re-resolve")
    args = ap.parse_args(argv)

    print(f"How-We-Swipe full release -> {args.dest}")
    download(args.dest, force=args.force, live=not args.offline)
    logs_dir = extract(args.dest, force=args.force)
    verify(args.dest, logs_dir)
    if args.analyze:
        analyze(args.dest, logs_dir, args.local_logs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
