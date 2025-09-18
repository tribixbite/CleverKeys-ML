#!/usr/bin/env python3
"""Profile personalized RNNT ONNX exports (latency + accuracy)."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import onnxruntime as ort

# ----------------------------- Feature pipeline -----------------------------

RESAMPLE_CFG = {
    "short_target": 56,
    "long_target": 96,
    "short_threshold": 48,
    "long_threshold": 112,
}

KEY_LAYOUT = [
    "qwertyuiop",
    "asdfghjkl",
    "zxcvbnm",
]


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def build_key_centers() -> List[Tuple[str, float, float]]:
    centers: List[Tuple[str, float, float]] = []
    for row_idx, row in enumerate(KEY_LAYOUT):
        for col_idx, char in enumerate(row):
            x01 = (col_idx + 0.5) / 10.0
            y01 = (row_idx + 0.5) / 3.0
            centers.append((char, x01 * 2.0 - 1.0, y01 * 2.0 - 1.0))
    return centers

KEY_CENTERS = build_key_centers()


def normalize_points(points: List[Dict]) -> List[Dict[str, float]]:
    if not points:
        return []
    start_t = float(points[0].get("t", 0.0))
    norm: List[Dict[str, float]] = []
    for idx, pt in enumerate(points):
        x = clamp(float(pt.get("x", 0.5)) * 2.0 - 1.0, -1.0, 1.0)
        y = clamp(float(pt.get("y", 0.5)) * 2.0 - 1.0, -1.0, 1.0)
        t = float(pt.get("t", idx * 10.0)) - start_t
        norm.append({"x": x, "y": y, "t": max(0.0, t)})
    return norm


def determine_target(length: int) -> int:
    if length <= 1:
        return max(length, RESAMPLE_CFG["short_target"])
    if length <= RESAMPLE_CFG["short_threshold"]:
        return max(length, RESAMPLE_CFG["short_target"])
    if length >= RESAMPLE_CFG["long_threshold"]:
        return RESAMPLE_CFG["long_target"]
    return length


def resample_points(points: List[Dict[str, float]], target_count: int) -> List[Dict[str, float]]:
    if target_count <= 0 or not points:
        return []
    if len(points) == target_count:
        return [dict(p) for p in points]
    out: List[Dict[str, float]] = []
    first = points[0]["t"]
    last = points[-1]["t"]
    duration = max(last - first, 1.0)
    step = duration / max(target_count - 1, 1)
    src_idx = 0
    for i in range(target_count):
        target_t = last if i == target_count - 1 else first + step * i
        while src_idx < len(points) - 2 and points[src_idx + 1]["t"] < target_t:
            src_idx += 1
        p1 = points[src_idx]
        p2 = points[min(src_idx + 1, len(points) - 1)]
        span = max(p2["t"] - p1["t"], 1.0)
        alpha = clamp((target_t - p1["t"]) / span, 0.0, 1.0)
        x = p1["x"] + (p2["x"] - p1["x"]) * alpha
        y = p1["y"] + (p2["y"] - p1["y"]) * alpha
        out.append({"x": x, "y": y, "t": target_t})
    return out


def compute_feature(points: List[Dict[str, float]], idx: int) -> np.ndarray:
    total = len(points)
    curr = points[idx]
    prev = points[idx - 1] if idx > 0 else None
    prev2 = points[idx - 2] if idx > 1 else None

    x = clamp(curr["x"], -1.0, 1.0)
    y = clamp(curr["y"], -1.0, 1.0)
    t_sec = curr["t"] / 1000.0

    vx = vy = speed = 0.0
    if prev:
        dt = max((curr["t"] - prev["t"]) / 1000.0, 0.001)
        prev_x = clamp(prev["x"], -1.0, 1.0)
        prev_y = clamp(prev["y"], -1.0, 1.0)
        vx = (x - prev_x) / dt
        vy = (y - prev_y) / dt
        speed = math.hypot(vx, vy)

    ax = ay = acc = 0.0
    if prev and prev2:
        dt1 = max((curr["t"] - prev["t"]) / 1000.0, 0.001)
        dt2 = max((prev["t"] - prev2["t"]) / 1000.0, 0.001)
        prev_x = clamp(prev["x"], -1.0, 1.0)
        prev_y = clamp(prev["y"], -1.0, 1.0)
        prev2_x = clamp(prev2["x"], -1.0, 1.0)
        prev2_y = clamp(prev2["y"], -1.0, 1.0)
        vx_prev = (prev_x - prev2_x) / dt2
        vy_prev = (prev_y - prev2_y) / dt2
        ax = (vx - vx_prev) / dt1
        ay = (vy - vy_prev) / dt1
        acc = math.hypot(ax, ay)

    angle = math.atan2(vy, vx) if prev else 0.0
    angle_sin = math.sin(angle)
    angle_cos = math.cos(angle)

    curvature = 0.0
    if prev and prev2:
        prev_x = clamp(prev["x"], -1.0, 1.0)
        prev_y = clamp(prev["y"], -1.0, 1.0)
        prev2_x = clamp(prev2["x"], -1.0, 1.0)
        prev2_y = clamp(prev2["y"], -1.0, 1.0)
        prev_angle = math.atan2(prev_y - prev2_y, prev_x - prev2_x)
        curvature = angle - prev_angle
        while curvature > math.pi:
            curvature -= 2 * math.pi
        while curvature < -math.pi:
            curvature += 2 * math.pi

    distances = [math.hypot(x - kx, y - ky) for _, kx, ky in KEY_CENTERS]
    distances.sort()
    while len(distances) < 5:
        distances.append(1.0)
    key_dist = distances[:5]

    progress = idx / max(total - 1, 1)
    is_start = 1.0 if idx == 0 else 0.0
    is_end = 1.0 if idx == total - 1 else 0.0

    window = points[max(0, idx - 2): min(total, idx + 3)]
    if len(window) > 1:
        xs = [clamp(p["x"], -1.0, 1.0) for p in window]
        ys = [clamp(p["y"], -1.0, 1.0) for p in window]
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)
        std_x = math.sqrt(sum((v - mean_x) ** 2 for v in xs) / len(xs))
        std_y = math.sqrt(sum((v - mean_y) ** 2 for v in ys) / len(ys))
        range_x = max(xs) - min(xs)
        range_y = max(ys) - min(ys)
    else:
        mean_x = x
        mean_y = y
        std_x = std_y = 0.0
        range_x = range_y = 0.0

    features = [
        x, y, t_sec,
        vx, vy, speed,
        ax, ay, acc,
        angle, angle_sin, angle_cos,
        curvature,
        *key_dist,
        progress,
        is_start, is_end,
        mean_x, std_x,
        mean_y, std_y,
        range_x, range_y,
    ]
    while len(features) < 37:
        features.append(0.0)
    return np.array(features[:37], dtype=np.float32)


def featurize(points: List[Dict[str, float]]) -> np.ndarray:
    return np.stack([compute_feature(points, i) for i in range(len(points))])

# ---------------------------- Lexicon beam search ---------------------------


def build_trie(words: List[str], char_to_id: Dict[str, int]):
    node = lambda: {"ch": {}, "is": False, "wid": -1}
    root = node()
    kept = 0
    for wid, word in enumerate(words):
        w = word.lower()
        if any(ch not in char_to_id for ch in w):
            continue
        cur = root
        for ch in w:
            cid = char_to_id[ch]
            cur = cur["ch"].setdefault(cid, node())
        cur["is"] = True
        cur["wid"] = wid
        kept += 1
    return root, kept


def slice_state(state: np.ndarray, idx: int) -> np.ndarray:
    return state[:, idx:idx + 1, :].astype(np.float32, copy=True)


def rnnt_word_beam(
    step_sess: ort.InferenceSession,
    frames: np.ndarray,
    blank_id: int,
    trie_root,
    words: List[str],
    beam: int,
    prune: int,
    max_symbols: int,
    topk: int,
    default_layers: int,
    default_hidden: int,
) -> Tuple[List[Tuple[str, float]], int]:
    T_out, D = frames.shape
    outputs = step_sess.get_outputs()
    logits_name = outputs[0].name
    h1_name = outputs[1].name
    c1_name = outputs[2].name

    batch_blank = np.array([blank_id], dtype=np.int64)

    # Determine L and H from input shape if available
    inputs = step_sess.get_inputs()
    layer_dim = inputs[1].shape[0]
    hidden_dim = inputs[1].shape[2]
    lstm_layers = layer_dim if isinstance(layer_dim, int) and layer_dim > 0 else default_layers
    hidden_size = hidden_dim if isinstance(hidden_dim, int) and hidden_dim > 0 else default_hidden

    beams = [{
        "y": blank_id,
        "h": np.zeros((lstm_layers, 1, hidden_size), dtype=np.float32),
        "c": np.zeros((lstm_layers, 1, hidden_size), dtype=np.float32),
        "tr": trie_root,
        "lp": 0.0,
    }]

    step_calls = 0

    for t in range(T_out):
        frame = frames[t:t + 1, :].astype(np.float32)
        for _ in range(max_symbols):
            beams.sort(key=lambda b: b["lp"], reverse=True)
            active = beams[:beam]
            N = len(active)
            y_prev = np.array([b["y"] for b in active], dtype=np.int64)
            h0 = np.concatenate([b["h"] for b in active], axis=1)
            c0 = np.concatenate([b["c"] for b in active], axis=1)
            enc_batch = np.repeat(frame, N, axis=0)

            run_outputs = step_sess.run(
                [logits_name, h1_name, c1_name],
                {"y_prev": y_prev, "h0": h0, "c0": c0, "enc_t": enc_batch},
            )
            logits, h1, c1 = run_outputs
            step_calls += 1

            nxt = []
            for i, beam_entry in enumerate(active):
                log_row = logits[i].reshape(-1)
                blank_lp = float(log_row[blank_id])
                nxt.append({
                    "y": blank_id,
                    "h": slice_state(h1, i),
                    "c": slice_state(c1, i),
                    "tr": beam_entry["tr"],
                    "lp": beam_entry["lp"] + blank_lp,
                    "chars": beam_entry.get("chars", [])[:],
                })

                allowed = list(beam_entry["tr"]["ch"].keys())
                allowed.sort(key=lambda cid: float(log_row[cid]), reverse=True)
                for cid in allowed[:min(prune, len(allowed))]:
                    child = beam_entry["tr"]["ch"][cid]
                    nxt.append({
                        "y": cid,
                        "h": slice_state(h1, i),
                        "c": slice_state(c1, i),
                        "tr": child,
                        "lp": beam_entry["lp"] + float(log_row[cid]),
                        "chars": beam_entry.get("chars", []) + [cid],
                    })

            nxt.sort(key=lambda b: b["lp"], reverse=True)
            beams = nxt[:beam]
            if beams and beams[0]["y"] == blank_id:
                break

    candidates: List[Tuple[str, float]] = []
    seen = set()
    for beam_entry in beams:
        node = beam_entry["tr"]
        wid = node.get("wid", -1)
        if node.get("is") and wid >= 0 and wid not in seen:
            seen.add(wid)
            candidates.append((words[wid], beam_entry["lp"]))
            if len(candidates) >= topk:
                break

    if not candidates:
        # Fallback to character sequence if no lexicon hit
        best = max(beams, key=lambda b: b["lp"])
        seq = best.get("chars", [])
        word = "".join(words[cid] if cid < len(words) else "" for cid in seq)
        candidates.append((word, best["lp"]))

    return candidates, step_calls

# ----------------------------- Profiling driver -----------------------------


def load_manifest(path: Path, limit: int | None) -> List[Dict]:
    samples: List[Dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            item = json.loads(line)
            if not item.get("points") or not item.get("word"):
                continue
            samples.append(item)
            if limit is not None and len(samples) >= limit:
                break
    return samples


def load_meta(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        meta = json.load(fh)
    blank_id = int(meta.get("blank_id", 0))
    unk_id = int(meta.get("unk_id", -1))
    char_to_id = {k: int(v) for k, v in meta.get("char_to_id", {}).items()}
    id_to_char = {int(k): v for k, v in meta.get("id_to_char", {}).items()}
    return blank_id, unk_id, char_to_id, id_to_char


def ensure_frames(encoded: np.ndarray, actual_len: int) -> np.ndarray:
    raw = encoded[0]
    if raw.shape[0] == actual_len:
        frames = raw
    elif raw.shape[1] == actual_len:
        frames = raw.T
    else:
        frames = raw.T if raw.shape[1] < raw.shape[0] else raw
        actual_len = frames.shape[0]
    return frames[:actual_len]


def tokens_to_word(chars: List[int], blank_id: int, id_to_char: Dict[int, str]) -> str:
    out = []
    for cid in chars:
        if cid == blank_id:
            continue
        ch = id_to_char.get(cid)
        if not ch or ch.startswith("<"):
            continue
        out.append(ch)
    return "".join(out)


def profile(args: argparse.Namespace):
    samples = load_manifest(Path(args.manifest), args.limit)
    if not samples:
        raise RuntimeError("No usable samples found")

    blank_id, _, char_to_id, id_to_char = load_meta(Path(args.meta))
    words = [w.strip() for w in Path(args.lexicon).read_text(encoding="utf-8").splitlines() if w.strip()]
    trie_root, kept = build_trie(words, char_to_id)

    providers = [args.provider]
    encoder_sess = ort.InferenceSession(str(Path(args.encoder)), providers=providers)
    decoder_sess = ort.InferenceSession(str(Path(args.decoder)), providers=providers)

    encode_times: List[float] = []
    decode_times: List[float] = []
    total_times: List[float] = []
    frame_counts: List[int] = []
    step_calls: List[int] = []
    matches = 0
    reports: List[Dict] = []

    for sample in samples:
        raw_points = sample["points"][: args.max_trace_len]
        norm = normalize_points(raw_points)
        target = determine_target(len(norm))
        processed = resample_points(norm, target)
        if not processed:
            continue
        feats = featurize(processed)
        feats_bft = np.expand_dims(feats.T, axis=0).astype(np.float32)
        lengths = np.array([feats_bft.shape[2]], dtype=np.int32)

        t_total_start = time.perf_counter()
        t_enc_start = time.perf_counter()
        enc_btf, enc_len = encoder_sess.run(None, {"features_bft": feats_bft, "lengths": lengths})
        encode_ms = (time.perf_counter() - t_enc_start) * 1000.0

        T_out = int(enc_len[0]) if enc_len.size else feats_bft.shape[2]
        frames = ensure_frames(enc_btf, T_out)

        t_dec_start = time.perf_counter()
        cands, steps = rnnt_word_beam(
            decoder_sess,
            frames,
            blank_id,
            trie_root,
            words,
            args.beam,
            args.prune,
            args.max_symbols,
            args.topk,
            args.lstm_layers,
            args.hidden,
        )
        decode_ms = (time.perf_counter() - t_dec_start) * 1000.0
        total_ms = (time.perf_counter() - t_total_start) * 1000.0
        best_word = cands[0][0] if cands else ""

        if best_word == sample["word"].lower():
            matches += 1

        encode_times.append(encode_ms)
        decode_times.append(decode_ms)
        total_times.append(total_ms)
        frame_counts.append(T_out)
        step_calls.append(steps)
        reports.append({
            "word": sample["word"],
            "prediction": best_word,
            "frames": T_out,
            "step_calls": steps,
            "encode_ms": encode_ms,
            "decode_ms": decode_ms,
            "total_ms": total_ms,
            "candidates": cands,
        })

    n = len(reports)
    accuracy = matches / n if n else 0.0
    summary = {
        "samples": n,
        "accuracy": accuracy,
        "encode_ms_mean": statistics.mean(encode_times) if encode_times else 0.0,
        "encode_ms_median": statistics.median(encode_times) if encode_times else 0.0,
        "decode_ms_mean": statistics.mean(decode_times) if decode_times else 0.0,
        "decode_ms_median": statistics.median(decode_times) if decode_times else 0.0,
        "total_ms_mean": statistics.mean(total_times) if total_times else 0.0,
        "total_ms_median": statistics.median(total_times) if total_times else 0.0,
        "frames_mean": statistics.mean(frame_counts) if frame_counts else 0.0,
        "frames_median": statistics.median(frame_counts) if frame_counts else 0.0,
        "step_calls_mean": statistics.mean(step_calls) if step_calls else 0.0,
        "step_calls_median": statistics.median(step_calls) if step_calls else 0.0,
    }

    print(f"Samples profiled: {summary['samples']}")
    print(f"Exact-match accuracy: {summary['accuracy']*100:.2f}% ({matches}/{summary['samples']})")
    print("Latency (ms):")
    print(f"  Encoder mean/median: {summary['encode_ms_mean']:.2f} / {summary['encode_ms_median']:.2f}")
    print(f"  Decoder mean/median: {summary['decode_ms_mean']:.2f} / {summary['decode_ms_median']:.2f}")
    print(f"  Total   mean/median: {summary['total_ms_mean']:.2f} / {summary['total_ms_median']:.2f}")
    print("Sequence stats:")
    print(f"  Frames mean/median: {summary['frames_mean']:.1f} / {summary['frames_median']:.1f}")
    print(f"  Decoder calls mean/median: {summary['step_calls_mean']:.1f} / {summary['step_calls_median']:.1f}")

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"summary": summary, "samples": reports}
        report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Report written to {report_path}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile personalized RNNT ONNX exports")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--decoder", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--lexicon", required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--provider", default="CPUExecutionProvider")
    parser.add_argument("--beam", type=int, default=16)
    parser.add_argument("--prune", type=int, default=6)
    parser.add_argument("--max_symbols", type=int, default=20)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--max_trace_len", type=int, default=256)
    parser.add_argument("--lstm_layers", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=320)
    parser.add_argument("--report")
    profile(parser.parse_args())
