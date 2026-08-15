#!/usr/bin/env python3
"""Convert FUTO's official swipe-1 dev/test splits to the canonical eval schema.

Phase N (`PHASE_N.md` §1.1, §6-N0). The HF row schema
(`id, session, timestamp, word, canvas_*, orientation, data[{t,x,y}], …`) is
mapped to the canonical eval schema used by ``eval_beam.py`` and the FUTO
harness (`word, points[{t,x,y}], id, session, timestamp, source`):

* ``x, y`` copied **verbatim** — the HF corpus frame *is* the canonical
  letter-area frame, proven bit-exact in ``DATA_TIERS.md`` §1;
* ``t → t − t[0]`` (float ms, exactly the canonical build's transform);
* ``word → word.lower()``.

Drop rules — the ONLY filtering, each counted exactly (`PHASE_N.md` §1.1):

* a row whose a–z-normalized target is empty (it could never be scored);
* a row with fewer than 2 trace points (degenerate for every featurizer).

Everything else ships as FUTO shipped it: landscape traces, speed outliers,
dictionary-junk words (OOV-as-miss handles those symmetrically for both
engines). The stats sidecar records input sha256, row accounting, stratum
n's (≤3 / 4+ on the normalized target length) and session counts, so the
benchmark's provenance is a committed artifact rather than a recollection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import DEFAULT_WORKDIR, resolve  # noqa: E402
from scan_futo_sessions import normalize_word  # noqa: E402


def sha256_file(path: Path, bufsize: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(bufsize):
            h.update(chunk)
    return h.hexdigest()


def convert(src: Path, dst: Path, source_tag: str) -> dict:
    t0 = time.time()
    n_in = n_out = 0
    drop_empty_word = drop_short_trace = drop_malformed = 0
    n_le3 = n_ge4 = 0
    sessions = set()
    with open(src) as fin, open(dst, "w") as fout:
        for line in fin:
            n_in += 1
            try:
                o = json.loads(line)
                data = o["data"]
                word = str(o["word"]).lower()
            except Exception:  # noqa: BLE001  malformed line — counted, not hidden
                drop_malformed += 1
                continue
            if len(data) < 2:
                drop_short_trace += 1
                continue
            norm = normalize_word(word)
            if not norm:
                drop_empty_word += 1
                continue
            t_first = float(data[0]["t"])
            points = [{"t": float(p["t"]) - t_first, "x": p["x"], "y": p["y"]}
                      for p in data]
            row = {
                "word": word,
                "points": points,
                "id": o.get("id"),
                "session": o.get("session") or "",
                "timestamp": o.get("timestamp"),
                "source": source_tag,
            }
            fout.write(json.dumps(row) + "\n")
            n_out += 1
            sessions.add(row["session"])
            if len(norm) <= 3:
                n_le3 += 1
            else:
                n_ge4 += 1
    return {
        "src": str(src),
        "src_sha256": sha256_file(src),
        "dst": str(dst),
        "dst_sha256": sha256_file(dst),
        "rows_in": n_in,
        "rows_out": n_out,
        "drop_malformed": drop_malformed,
        "drop_short_trace": drop_short_trace,
        "drop_empty_normalized_word": drop_empty_word,
        "stratum_le3": n_le3,
        "stratum_4plus": n_ge4,
        "sessions": len(sessions),
        "elapsed_s": round(time.time() - t0, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    ap.add_argument("--hf-dir", default="data/hf",
                    help="directory holding the official dev.jsonl / test.jsonl")
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args()

    hf = resolve(args.workdir, args.hf_dir)
    out = resolve(args.workdir, args.out_dir)
    all_stats = {}
    for split in ("dev", "test"):
        src = hf / f"{split}.jsonl"
        dst = out / f"futo_{split}.jsonl"
        s = convert(src, dst, source_tag=f"futo-official-{split}")
        all_stats[split] = s
        print(f"[{split}] {s['rows_in']} -> {s['rows_out']} rows "
              f"(malformed {s['drop_malformed']}, <2pts {s['drop_short_trace']}, "
              f"empty-word {s['drop_empty_normalized_word']}); "
              f"<=3: {s['stratum_le3']}  4+: {s['stratum_4plus']}  "
              f"sessions {s['sessions']}  {s['elapsed_s']}s", flush=True)
    stats_path = out / "futo_official_convert.stats.json"
    stats_path.write_text(json.dumps(all_stats, indent=2) + "\n")
    print(f"stats -> {stats_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
