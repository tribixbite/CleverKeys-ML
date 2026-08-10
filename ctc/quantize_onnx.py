#!/usr/bin/env python3
"""Post-training quantization / weight-storage compression of an exported CTC
swipe encoder (Phase F F1: activation quantization; Phase I I2: weight-only
storage).

Phase I adds two **storage-only** modes that keep every activation and every
matmul/conv in float32 — sidestepping the activation-quantization failure Phase F
measured (the ``MASK_NEG`` −1e4 pad columns and the softplus gate are exactly the
tensors an 8-bit affine scale handles worst; int8 activations lost t5 at every
size):

* ``--mode fp16w`` — every large float32 initializer is stored as float16 and a
  ``Cast`` node restores float32 at load. ORT's constant folding runs the Cast
  once at session-init, so steady-state compute AND memory are identical to the
  fp32 graph; only the file (and the fp16 rounding of the weights, rel ~5e-4)
  changes. Halves the artifact.
* ``--mode int8w`` — per-output-channel symmetric int8 weight storage restored
  by ``DequantizeLinear`` (opset-13 axis form), same constant-folding argument.
  Quarters the artifact; weight rounding is the only numeric change.

Both modes print a parity check against the source graph (random draws, sliced
[T',27] contract view, max|Δ| + argmax agreement — the export_onnx.py protocol).

Phase F's two activation arms, both leaving the ONNX **I/O contract untouched** — the graph still takes
``features [1,2,64] float32 / layout_keys [1,64,2] float32 / layout_mask [1,64] bool``
and still returns ``log_emissions [1,32,65] / coefficients [1,32,64] / lambda [1,32,1]``
in float32, because quantization is inserted strictly between the boundary
Quantize/Dequantize pairs:

* ``--mode dynamic`` — weights are stored int8 and activations are quantized on the
  fly. ORT's dynamic path only covers ``MatMul``/``Gemm``/``Attention``/``LSTM``, so on
  a graph whose cost is 66 % ``Conv`` it is expected to be nearly free of *both* cost
  and benefit; it is measured anyway so the report can say so with a number.
* ``--mode static`` — QDQ format with a real calibration set drawn from the training
  cache, which does cover ``Conv``. Weights are per-channel int8, activations uint8
  (the combination x86 ``ConvInteger``/``QLinearConv`` kernels are written for).

``--exclude-tail`` keeps the numerically sensitive end of the graph in float32: the
per-key embedding MLP, the coefficient/lambda/blank heads, the key-scoring MatMul and
the log_softmax. Those carry the ``MASK_NEG`` pad columns (≈ -1e4) and a softplus gate,
which is exactly the shape of tensor an 8-bit affine scale handles worst.

Usage:
  python quantize_onnx.py --onnx ckpt/X/ctc_swipe_encoder.onnx --mode static \
      --calib-npz cache/train_t3.npz --calib-rows 1024 --out ckpt/X/enc_int8.onnx
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import DEFAULT_LAYOUT, DEFAULT_WORKDIR, resolve  # noqa: E402

#: Substrings of node names whose output feeds the contract tail. ``--exclude-tail``
#: turns these into a `nodes_to_exclude` list by prefix match on the node name.
TAIL_PATTERNS = ("key_embed", "coeff_head", "lambda_head", "blank_head",
                 "LogSoftmax", "Softplus", "/MatMul", "Where", "Concat")


def layout_inputs(layout: Path) -> Dict[str, np.ndarray]:
    """The two static layout tensors, canonical en_qwerty."""
    from futo_decoder_eval import load_layout
    letters, centers = load_layout(layout)
    keys = np.zeros((1, 64, 2), np.float32)
    keys[0, : len(letters)] = centers
    mask = np.zeros((1, 64), bool)
    mask[0, : len(letters)] = True
    return {"layout_keys": keys, "layout_mask": mask}


class NpzCalibrationReader:
    """Feeds real training features through the graph, one row at a time.

    Calibration uses the **canonical** layout, not the slot-permuted one training
    saw: the exported graph is only ever called with a real layout, and the
    activation ranges that matter are the ones inference produces.
    """

    def __init__(self, features: np.ndarray, layout: Dict[str, np.ndarray]) -> None:
        self._rows = features
        self._layout = layout
        self._i = 0

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        if self._i >= len(self._rows):
            return None
        f = self._rows[self._i : self._i + 1].astype(np.float32)
        self._i += 1
        return {"features": f, **self._layout}

    def rewind(self) -> None:
        self._i = 0


def node_names(onnx_path: Path) -> List[str]:
    import onnx
    return [n.name for n in onnx.load(str(onnx_path)).graph.node]


#: Initializers smaller than this stay fp32 in the weight-only modes — shape
#: constants and per-op scalars are not worth a Cast node each.
WEIGHT_MIN_ELEMS = 64


def shrink_weights(src: Path, dst: Path, mode: str,
                   int8_scope: str = "all") -> None:
    """Store large fp32 initializers as fp16 (``fp16w``) or per-channel int8
    (``int8w``), restored to fp32 in-graph so all compute stays float32.

    ORT constant-folds the restore ops at session load (verified by the caller's
    latency path being unchanged), so this is a pure disk-bytes lever.

    :param int8_scope: ``int8w`` only. ``all`` quantizes every large tensor;
        ``trunk`` quantizes only the trunk convolutions (initializer names
        ``blocks.*`` / ``stem.*`` — ~92 % of the parameters at ch 192+) and
        stores the numerically sensitive rest (key-embed MLP, heads) as fp16.
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    m = onnx.load(str(src))
    g = m.graph
    graph_inputs = {i.name for i in g.input}
    new_inits, new_nodes = [], []
    n_shrunk = elems = 0
    for init in g.initializer:
        arr = numpy_helper.to_array(init)
        if (init.data_type != TensorProto.FLOAT or arr.size < WEIGHT_MIN_ELEMS
                or init.name in graph_inputs):
            new_inits.append(init)
            continue
        n_shrunk += 1
        elems += arr.size
        as_fp16 = mode == "fp16w" or (
            mode == "int8w" and int8_scope == "trunk"
            and not (init.name.startswith("blocks.")
                     or init.name.startswith("stem.")))
        if as_fp16:
            half = numpy_helper.from_array(arr.astype(np.float16),
                                           init.name + "_fp16")
            new_inits.append(half)
            new_nodes.append(helper.make_node(
                "Cast", [half.name], [init.name], name=init.name + "_upcast",
                to=TensorProto.FLOAT))
        else:                                   # int8w — symmetric weight storage
            if arr.ndim > 1:                    # per-output-channel, axis 0
                amax = np.abs(arr.reshape(arr.shape[0], -1)).max(axis=1)
                scale = np.where(amax > 0, amax / 127.0, 1.0).astype(np.float32)
                shp = (-1,) + (1,) * (arr.ndim - 1)
                q = np.clip(np.rint(arr / scale.reshape(shp)), -127, 127) \
                    .astype(np.int8)
                dq_kwargs = {"axis": 0}
            else:                               # 1-D (bias etc): per-tensor
                amax = float(np.abs(arr).max())
                scale = np.float32(amax / 127.0 if amax > 0 else 1.0)
                q = np.clip(np.rint(arr / scale), -127, 127).astype(np.int8)
                scale = np.array(scale, np.float32)      # scalar tensor
                dq_kwargs = {}
            new_inits.append(numpy_helper.from_array(q, init.name + "_q8"))
            new_inits.append(numpy_helper.from_array(scale, init.name + "_q8s"))
            new_nodes.append(helper.make_node(
                "DequantizeLinear", [init.name + "_q8", init.name + "_q8s"],
                [init.name], name=init.name + "_dq", **dq_kwargs))
    del g.initializer[:]
    g.initializer.extend(new_inits)
    # Restore ops depend only on initializers, so prepending keeps the graph
    # topologically sorted.
    old_nodes = list(g.node)
    del g.node[:]
    g.node.extend(new_nodes + old_nodes)
    onnx.checker.check_model(m)
    onnx.save(m, str(dst))
    print(f"{mode}: {n_shrunk} initializers ({elems:,} elems) stored "
          f"{'fp16' if mode == 'fp16w' else 'int8+scale'}")


def parity_vs_source(src: Path, dst: Path, layout: Path, trials: int = 100
                     ) -> None:
    """export_onnx.py's sliced-view parity, shrunk graph vs its fp32 source."""
    import onnxruntime as ort
    from futo_decoder_eval import load_layout

    letters, _ = load_layout(layout)
    k = len(letters)
    a = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"])
    b = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(7)
    mask = np.zeros((1, 64), bool)
    mask[:, :k] = True
    worst, agree = 0.0, 0
    for _ in range(trials):
        feed = {"features": rng.random((1, 2, 64), dtype=np.float32),
                "layout_keys": rng.random((1, 64, 2), dtype=np.float32),
                "layout_mask": mask}
        ea = a.run(["log_emissions"], feed)[0][0]
        eb = b.run(["log_emissions"], feed)[0][0]
        sa = np.concatenate([ea[:, :k], ea[:, 64:65]], axis=1)
        sb = np.concatenate([eb[:, :k], eb[:, 64:65]], axis=1)
        worst = max(worst, float(np.abs(sa - sb).max()))
        agree += int((sa.argmax(-1) == sb.argmax(-1)).all())
    print(f"parity vs source: sliced max|Δ| {worst:.2e}  argmax agreement "
          f"{agree}/{trials}")


def tail_nodes(onnx_path: Path, patterns: Sequence[str] = TAIL_PATTERNS) -> List[str]:
    """Node names matching any tail pattern — the float32 exclusion list."""
    return [n for n in node_names(onnx_path)
            if any(p.lower() in n.lower() for p in patterns)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    ap.add_argument("--layout", type=Path, default=DEFAULT_LAYOUT)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="static",
                    choices=("dynamic", "static", "fp16w", "int8w"))
    ap.add_argument("--int8-scope", default="all", choices=("all", "trunk"),
                    dest="int8_scope",
                    help="int8w only: 'trunk' = int8 for stem+blocks convs, "
                         "fp16 for the sensitive tail (key-embed, heads)")
    ap.add_argument("--calib-npz", default="cache/train_t3.npz", dest="calib_npz")
    ap.add_argument("--calib-rows", type=int, default=1024, dest="calib_rows")
    ap.add_argument("--calib-seed", type=int, default=20260808, dest="calib_seed")
    ap.add_argument("--per-channel", type=int, default=1, dest="per_channel")
    ap.add_argument("--exclude-tail", action="store_true", dest="exclude_tail",
                    help="keep the key-embed / heads / log_softmax tail in float32")
    ap.add_argument("--exclude", default="",
                    help="extra comma-separated node-name substrings to exclude")
    ap.add_argument("--calib-method", default="minmax", dest="calib_method",
                    choices=("minmax", "entropy", "percentile"))
    ap.add_argument("--op-types", default="", dest="op_types",
                    help="comma-separated op types to quantize (default: all "
                         "supported; dynamic mode defaults to MatMul,Gemm because "
                         "this ORT build has no runnable ConvInteger kernel)")
    args = ap.parse_args()

    src = resolve(args.workdir, args.onnx)
    dst = resolve(args.workdir, args.out)
    dst.parent.mkdir(parents=True, exist_ok=True)

    if args.mode in ("fp16w", "int8w"):
        shrink_weights(src, dst, args.mode, args.int8_scope)
        parity_vs_source(src, dst, args.layout)
        print(f"wrote {dst} ({dst.stat().st_size} bytes; source "
              f"{src.stat().st_size} bytes, "
              f"{dst.stat().st_size / src.stat().st_size * 100:.1f}%)")
        return 0

    from onnxruntime.quantization import (CalibrationMethod, QuantFormat, QuantType,
                                          quantize_dynamic, quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    pre = dst.with_suffix(".pre.onnx")
    quant_pre_process(str(src), str(pre), skip_symbolic_shape=False)

    exclude: List[str] = []
    if args.exclude_tail:
        exclude += tail_nodes(pre)
    if args.exclude:
        pats = [p for p in args.exclude.split(",") if p]
        exclude += [n for n in node_names(pre)
                    if any(p.lower() in n.lower() for p in pats)]
    exclude = sorted(set(exclude))
    if exclude:
        print(f"excluding {len(exclude)} nodes from quantization")

    op_types = [t for t in args.op_types.split(",") if t]
    if args.mode == "dynamic":
        # ORT emits ConvInteger for a dynamically quantized Conv, and this build has
        # no CPU kernel registered for it (session load fails outright), so the
        # dynamic arm is MatMul/Gemm-only unless --op-types says otherwise.
        quantize_dynamic(str(pre), str(dst), weight_type=QuantType.QInt8,
                         per_channel=bool(args.per_channel),
                         nodes_to_exclude=exclude,
                         op_types_to_quantize=op_types or ["MatMul", "Gemm"],
                         extra_options={"MatMulConstBOnly": True})
    else:
        with np.load(resolve(args.workdir, args.calib_npz)) as d:
            feats = d["features"]
            rng = np.random.default_rng(args.calib_seed)
            idx = rng.choice(len(feats), size=min(args.calib_rows, len(feats)),
                             replace=False)
            rows = np.array(feats[np.sort(idx)])
        print(f"calibration: {len(rows)} rows from {args.calib_npz}")
        reader = NpzCalibrationReader(rows, layout_inputs(args.layout))
        methods = {"minmax": CalibrationMethod.MinMax,
                   "entropy": CalibrationMethod.Entropy,
                   "percentile": CalibrationMethod.Percentile}
        quantize_static(str(pre), str(dst), reader,
                        quant_format=QuantFormat.QDQ,
                        activation_type=QuantType.QUInt8,
                        weight_type=QuantType.QInt8,
                        per_channel=bool(args.per_channel),
                        calibrate_method=methods[args.calib_method],
                        nodes_to_exclude=exclude,
                        extra_options={"ActivationSymmetric": False,
                                       "WeightSymmetric": True})
    pre.unlink(missing_ok=True)
    for junk in ("augmented_model.onnx",):
        Path(junk).unlink(missing_ok=True)
    print(f"wrote {dst} ({dst.stat().st_size} bytes; source "
          f"{src.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
