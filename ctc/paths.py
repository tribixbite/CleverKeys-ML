#!/usr/bin/env python3
"""Shared path/workdir resolution for the CTC swipe pipeline (audit fix #16).

Every script accepts ``--workdir`` (default ``~/ctc-train``) and resolves
``data/``, ``cache/`` and ``ckpt/`` beneath it, so the scripts can be launched
from any current working directory. The canonical layout defaults to the
``en_qwerty.json`` sitting next to the scripts themselves.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

#: Directory holding the vendored harness + layout (this file's directory).
SCRIPT_DIR = Path(__file__).resolve().parent

#: Default runtime workdir holding data/, cache/ and ckpt/.
DEFAULT_WORKDIR = Path.home() / "ctc-train"

#: Default canonical layout, shipped next to the scripts.
DEFAULT_LAYOUT = SCRIPT_DIR / "en_qwerty.json"


def add_common_args(ap: argparse.ArgumentParser) -> None:
    """Register ``--workdir`` / ``--layout`` on an argument parser."""
    ap.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR,
                    help="runtime dir holding data/, cache/, ckpt/ (default: ~/ctc-train)")
    ap.add_argument("--layout", type=Path, default=DEFAULT_LAYOUT,
                    help="canonical layout json (default: en_qwerty.json next to this script)")


def resolve(workdir: Path, candidate: Path | str) -> Path:
    """Resolve *candidate* against *workdir* unless it is already absolute."""
    p = Path(candidate).expanduser()
    return p if p.is_absolute() else (Path(workdir).expanduser() / p)


def sha256_file(path: Path) -> str:
    """Hex sha256 of a file's bytes (used for cache provenance, audit fix #17)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
