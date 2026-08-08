#!/usr/bin/env python3
"""Single-thread batch-1 CPU latency + op-level profile of an exported encoder.

The Phase-F measurement instrument. It implements exactly the protocol
`AUDIT_PREDECODE.md` §7 used, so every number it prints is comparable with the
0.455 ms (ch 128) / 0.877 ms (ch 192) figures already on record:

* ONNX Runtime ``CPUExecutionProvider``, ``intra_op = inter_op = 1``;
* batch 1, fixed shapes, the canonical en_qwerty layout in ``layout_keys`` /
  ``layout_mask`` (a random layout would change nothing — the shapes are static —
  but the real one keeps the graph's mask branch on the path it takes on device);
* 50 warmup calls, then ``rounds`` x ``runs`` timed calls; the reported mean and
  p90 are those of the **best round** (lowest mean), which is the standard
  defence against a scheduler hiccup landing inside a measurement window.

``--profile`` re-runs the model with ORT's profiler on and aggregates the trace
by op type and by node, which is what tells you where the milliseconds go before
you try to remove them. Profiling instruments every kernel, so its absolute
numbers run high; read the *shares*, not the totals.

``--optimize-out`` serializes the ``ORT_ENABLE_ALL`` graph so the fusions ORT
does at session-load time can be inspected (and their cost paid offline).

Usage:
  python bench_latency.py --onnx a.onnx --onnx b.onnx --label a --label b
  python bench_latency.py --onnx a.onnx --profile --optimize-out a.opt.onnx
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import DEFAULT_LAYOUT, DEFAULT_WORKDIR, resolve  # noqa: E402

WARMUP = 50
ROUNDS = 3
RUNS = 300


def make_feed(layout: Path, seed: int = 0) -> Dict[str, np.ndarray]:
    """The fixed-shape input triple the IME hands the encoder once per swipe."""
    from futo_decoder_eval import load_layout
    letters, centers = load_layout(layout)
    rng = np.random.default_rng(seed)
    keys = np.zeros((1, 64, 2), np.float32)
    keys[0, : len(letters)] = centers
    mask = np.zeros((1, 64), bool)
    mask[0, : len(letters)] = True
    return {"features": rng.random((1, 2, 64), dtype=np.float32),
            "layout_keys": keys, "layout_mask": mask}


def session(onnx: Path, optimize_out: Path | None = None,
            profile_prefix: str | None = None,
            opt_level: str = "all"):
    """A single-threaded CPU session, optionally profiling / serializing."""
    import onnxruntime as ort
    levels = {"all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
              "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
              "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
              "none": ort.GraphOptimizationLevel.ORT_DISABLE_ALL}
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = levels[opt_level]
    if optimize_out is not None:
        opts.optimized_model_filepath = str(optimize_out)
    if profile_prefix is not None:
        opts.enable_profiling = True
        opts.profile_file_prefix = profile_prefix
    return ort.InferenceSession(str(onnx), opts, providers=["CPUExecutionProvider"])


def measure(sess, feed: Dict[str, np.ndarray], outputs: Sequence[str],
            warmup: int = WARMUP, rounds: int = ROUNDS,
            runs: int = RUNS) -> Tuple[float, float, List[float]]:
    """-> ``(best-round mean ms, that round's p90 ms, per-round means)``."""
    out = list(outputs)
    for _ in range(warmup):
        sess.run(out, feed)
    means: List[float] = []
    best: Tuple[float, float] = (float("inf"), float("inf"))
    for _ in range(rounds):
        ts = np.empty(runs)
        for i in range(runs):
            t = time.perf_counter()
            sess.run(out, feed)
            ts[i] = (time.perf_counter() - t) * 1000.0
        m = float(ts.mean())
        means.append(m)
        if m < best[0]:
            best = (m, float(np.percentile(ts, 90)))
    return best[0], best[1], means


def profile(onnx: Path, feed: Dict[str, np.ndarray], outputs: Sequence[str],
            runs: int, workdir: Path) -> Dict[str, object]:
    """Run with ORT profiling on; aggregate the trace by op type and by node."""
    prefix = str(workdir / f"prof_{onnx.stem}")
    sess = session(onnx, profile_prefix=prefix)
    out = list(outputs)
    for _ in range(20):
        sess.run(out, feed)
    for _ in range(runs):
        sess.run(out, feed)
    path = Path(sess.end_profiling())
    events = json.loads(path.read_text())
    by_type: Dict[str, float] = collections.defaultdict(float)
    by_node: Dict[str, Tuple[float, str]] = {}
    total = 0.0
    for e in events:
        if e.get("cat") != "Node" or not e.get("name", "").endswith("_kernel_time"):
            continue
        dur = float(e["dur"])           # microseconds
        op = e["args"].get("op_name", "?")
        node = e["name"][: -len("_kernel_time")]
        by_type[op] += dur
        prev = by_node.get(node, (0.0, op))
        by_node[node] = (prev[0] + dur, op)
        total += dur
    return {"trace": str(path), "total_us": total, "runs": runs,
            "by_type": dict(sorted(by_type.items(), key=lambda kv: -kv[1])),
            "by_node": dict(sorted(by_node.items(), key=lambda kv: -kv[1][0]))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    ap.add_argument("--layout", type=Path, default=DEFAULT_LAYOUT)
    ap.add_argument("--onnx", action="append", required=True,
                    help="model to time; repeatable")
    ap.add_argument("--label", action="append", default=[],
                    help="display name per --onnx (default: file stem)")
    ap.add_argument("--outputs", default="log_emissions",
                    help="comma-separated output names to fetch")
    ap.add_argument("--warmup", type=int, default=WARMUP)
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    ap.add_argument("--runs", type=int, default=RUNS)
    ap.add_argument("--opt-level", default="all", dest="opt_level",
                    choices=("all", "extended", "basic", "none"))
    ap.add_argument("--profile", action="store_true",
                    help="also emit an op-type / per-node breakdown")
    ap.add_argument("--profile-runs", type=int, default=200, dest="profile_runs")
    ap.add_argument("--optimize-out", default="", dest="optimize_out",
                    help="serialize the optimized graph of the FIRST --onnx here")
    ap.add_argument("--top-nodes", type=int, default=15, dest="top_nodes")
    ap.add_argument("--out", default="", help="write the results JSON here")
    args = ap.parse_args()

    outputs = [o for o in args.outputs.split(",") if o]
    feed = make_feed(args.layout)
    results: Dict[str, object] = {"protocol": {"warmup": args.warmup,
                                               "rounds": args.rounds,
                                               "runs": args.runs,
                                               "intra_op": 1, "inter_op": 1,
                                               "opt_level": args.opt_level,
                                               "outputs": outputs}}
    for i, spec in enumerate(args.onnx):
        p = resolve(args.workdir, spec)
        label = args.label[i] if i < len(args.label) else p.stem
        opt_out = (resolve(args.workdir, args.optimize_out)
                   if args.optimize_out and i == 0 else None)
        sess = session(p, optimize_out=opt_out, opt_level=args.opt_level)
        mean, p90, means = measure(sess, feed, outputs, args.warmup,
                                   args.rounds, args.runs)
        size = p.stat().st_size
        rec: Dict[str, object] = {"path": str(p), "bytes": size,
                                  "mean_ms": mean, "p90_ms": p90,
                                  "round_means": means}
        print(f"{label:<34} mean {mean:6.3f} ms   p90 {p90:6.3f} ms   "
              f"rounds {' '.join(f'{m:.3f}' for m in means)}   "
              f"{size / 1024:.0f} KiB")
        if args.profile:
            pr = profile(p, feed, outputs, args.profile_runs, args.workdir)
            rec["profile"] = pr
            tot = pr["total_us"] or 1.0
            print(f"  op-type breakdown ({pr['runs']} profiled runs, "
                  f"{tot / pr['runs']:.0f} us/run instrumented):")
            for op, us in pr["by_type"].items():
                print(f"    {op:<28} {us / tot * 100:5.1f}%  "
                      f"{us / pr['runs']:7.1f} us/run")
            print(f"  slowest {args.top_nodes} nodes:")
            for node, (us, op) in list(pr["by_node"].items())[: args.top_nodes]:
                print(f"    {node[:52]:<52} {op:<18} {us / tot * 100:5.1f}%  "
                      f"{us / pr['runs']:7.1f} us/run")
        results[label] = rec
    if args.out:
        q = resolve(args.workdir, args.out)
        q.parent.mkdir(parents=True, exist_ok=True)
        q.write_text(json.dumps(results, indent=1))
        print(f"wrote {q}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
