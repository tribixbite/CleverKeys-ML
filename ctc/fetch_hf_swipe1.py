#!/usr/bin/env python3
"""Download the HF ``futo-org/swipe.futo.org`` ``swipe-1`` config (MIT).

``swipe-1`` is the ~1 M-swipe main collection run, served as three JSONL files in
the repo root (the dataset card maps them to the ``swipe-1`` config splits):

    train.jsonl  5.16 GB   ~939 k rows   <- the scale-up corpus
    dev.jsonl    0.30 GB   ~54 k rows    } downloaded for the session-leak scan
    test.jsonl   0.27 GB   ~50 k rows    } only, NEVER used for training or eval

We keep our canonical local ``{val,test}_hwsfuto.jsonl`` as the evaluation sets —
every committed baseline is measured on those — so HF's own dev/test splits are
pulled purely so that contamination control can find which *sessions* our
canonical val/test rows belong to (see ``convert_hf_swipe1.py``).

Downloads land in ``<workdir>/data/hf/`` and are resumable; re-running is a no-op
once the files are present and their sha256 matches the hub.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import DEFAULT_WORKDIR, resolve  # noqa: E402

REPO = "futo-org/swipe.futo.org"
#: Repo-root files making up the swipe-1 config, and the layout definition.
FILES = ("train.jsonl", "dev.jsonl", "test.jsonl")
LAYOUT = "swipe-5/layouts/qwerty.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    ap.add_argument("--dest", type=Path, default=Path("data/hf"),
                    help="download dir, relative to --workdir unless absolute")
    ap.add_argument("--files", default=",".join(FILES),
                    help="comma-separated repo files to fetch")
    ap.add_argument("--no-layout", action="store_true", dest="no_layout")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download

    dest = resolve(args.workdir, args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    wanted = [f for f in args.files.split(",") if f]
    if not args.no_layout:
        wanted.append(LAYOUT)

    for name in wanted:
        print(f"[fetch] {REPO}:{name} -> {dest}", flush=True)
        p = hf_hub_download(REPO, name, repo_type="dataset",
                            local_dir=str(dest))
        size = Path(p).stat().st_size
        print(f"        {size / 1e9:.2f} GB  {p}", flush=True)
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
