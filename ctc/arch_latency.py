#!/usr/bin/env python3
"""Latency of candidate encoder architectures at random init (Phase F, F2 planning).

Latency is a property of the graph, not of the weights, so the whole
width/depth/block-type grid can be priced **before** any of it is trained. Each
spec is built, BN-folded, exported with the production
:mod:`export_onnx` settings (fixed shapes, opset 17, constant folding), timed with
the :mod:`bench_latency` protocol, and — optionally — statically quantized and timed
again, so the table shows the fp32 and int8 cost of every candidate side by side.

A spec is ``block:ch:dilations[:embed_hid]``, e.g. ``dwsep:128:1,2,4,8`` or
``dwsep:96:1,2,4,8,1,2,4,8:64``.

Usage:
  python arch_latency.py --spec dwsep:128:1,2,4,8 --spec dwsep:96:1,2,4,8 --quant
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_latency import make_feed, measure, session  # noqa: E402
from futo_decoder_eval import load_layout  # noqa: E402
from model import MAX_KEYS, T_IN, CtcSwipeEncoder  # noqa: E402
from paths import DEFAULT_LAYOUT, DEFAULT_WORKDIR, resolve  # noqa: E402


def parse_spec(spec: str) -> Dict[str, object]:
    """``block:ch:d1,d2,...[:embed_hid]`` -> encoder kwargs."""
    parts = spec.split(":")
    if len(parts) not in (3, 4):
        raise SystemExit(f"bad --spec {spec!r}: want block:ch:dilations[:embed_hid]")
    return {"block": parts[0], "ch": int(parts[1]),
            "dilations": tuple(int(d) for d in parts[2].split(",")),
            "embed_hid": int(parts[3]) if len(parts) == 4 else 96}


def export(kwargs: Dict[str, object], out: Path, num_letters: int) -> int:
    """Build at random init, fold BN, export; -> parameter count (pre-fold)."""
    model = CtcSwipeEncoder(**kwargs).eval()          # type: ignore[arg-type]
    params = sum(p.numel() for p in model.parameters())
    model.fold_bn()
    feats = torch.rand(1, 2, T_IN)
    keys = torch.rand(1, MAX_KEYS, 2)
    mask = torch.zeros(1, MAX_KEYS, dtype=torch.bool)
    mask[:, :num_letters] = True
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, (feats, keys, mask), str(out),
                      input_names=["features", "layout_keys", "layout_mask"],
                      output_names=["log_emissions", "coefficients", "lambda"],
                      opset_version=17, dynamic_axes=None,
                      do_constant_folding=True, dynamo=False)
    return params


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    ap.add_argument("--layout", type=Path, default=DEFAULT_LAYOUT)
    ap.add_argument("--spec", action="append", required=True)
    ap.add_argument("--dir", default="cache/archsweep",
                    help="where the throwaway .onnx files are written")
    ap.add_argument("--runs", type=int, default=300)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--quant", action="store_true",
                    help="also statically quantize each candidate and time that")
    ap.add_argument("--calib-npz", default="cache/train_t3hws.npz", dest="calib_npz")
    ap.add_argument("--calib-rows", type=int, default=256, dest="calib_rows")
    ap.add_argument("--out", default="cache/phase_f_arch_latency.json")
    args = ap.parse_args()

    letters, _ = load_layout(args.layout)
    feed = make_feed(args.layout)
    root = resolve(args.workdir, args.dir)
    results: Dict[str, object] = {}
    print(f"{'spec':<30} {'params':>9} {'fp32 ms':>9} {'p90':>7} "
          f"{'int8 ms':>9} {'p90':>7} {'bytes':>9}")
    for spec in args.spec:
        kw = parse_spec(spec)
        path = root / (spec.replace(":", "_").replace(",", "-") + ".onnx")
        params = export(kw, path, len(letters))
        sess = session(path)
        mean, p90, _ = measure(sess, feed, ["log_emissions"], args.warmup,
                               args.rounds, args.runs)
        rec: Dict[str, object] = {"params": params, "fp32_ms": mean,
                                  "fp32_p90": p90, "bytes": path.stat().st_size}
        qm = qp = float("nan")
        if args.quant:
            import subprocess
            qpath = path.with_name(path.stem + "_int8.onnx")
            subprocess.run([sys.executable, str(Path(__file__).with_name(
                "quantize_onnx.py")), "--onnx", str(path), "--out", str(qpath),
                "--mode", "static", "--calib-npz", args.calib_npz,
                "--calib-rows", str(args.calib_rows)],
                check=True, capture_output=True)
            qsess = session(qpath)
            qm, qp, _ = measure(qsess, feed, ["log_emissions"], args.warmup,
                                args.rounds, args.runs)
            rec.update({"int8_ms": qm, "int8_p90": qp,
                        "int8_bytes": qpath.stat().st_size})
        print(f"{spec:<30} {params:>9} {mean:>9.3f} {p90:>7.3f} "
              f"{qm:>9.3f} {qp:>7.3f} {rec['bytes']:>9}")
        results[spec] = rec
    p = resolve(args.workdir, args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(results, indent=1))
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
