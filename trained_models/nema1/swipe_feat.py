"""
Feature extraction pipeline for personalized swipe models.
This is shared between training, export, and testing.
"""

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple
import numpy as np

def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))

def determine_resample_target(length: int, cfg: Dict[str, Any]) -> int:
    if length <= 1:
        return length
    short_target = cfg.get("resample_short_target", 56)
    long_target = cfg.get("resample_long_target", 96)
    short_thresh = cfg.get("resample_short_threshold", 48)
    long_thresh = cfg.get("resample_long_threshold", 112)

    if length <= short_thresh:
        return max(length, short_target)
    if length >= long_thresh:
        return long_target
    return length

def resample_points(points: List[Dict[str, float]], target_count: int) -> List[Dict[str, float]]:
    if target_count <= 0 or len(points) == 0:
        return []
    if len(points) == target_count:
        return [dict(p) for p in points]

    resampled: List[Dict[str, float]] = []
    first_time = points[0]["t"]
    last_time = points[-1]["t"]
    duration = max(last_time - first_time, 1.0)
    step = duration / max(target_count - 1, 1)
    src_idx = 0

    for i in range(target_count):
        target_time = last_time if i == target_count - 1 else first_time + step * i
        while src_idx < len(points) - 2 and points[src_idx + 1]["t"] < target_time:
            src_idx += 1
        p1 = points[src_idx]
        p2 = points[min(src_idx + 1, len(points) - 1)]
        span = max(p2["t"] - p1["t"], 1.0)
        alpha = clamp((target_time - p1["t"]) / span, 0.0, 1.0)
        x = p1["x"] + (p2["x"] - p1["x"]) * alpha
        y = p1["y"] + (p2["y"] - p1["y"]) * alpha
        resampled.append({
            "x": x,
            "y": y,
            "t": target_time,
        })
    return resampled

def build_default_key_centers() -> List[Tuple[str, float, float]]:
    layout = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
    centers: List[Tuple[str, float, float]] = []
    for row_idx, row in enumerate(layout):
        for col_idx, char in enumerate(row):
            x01 = (col_idx + 0.5) / 10.0
            y01 = (row_idx + 0.5) / 3.0
            centers.append((char, x01 * 2.0 - 1.0, y01 * 2.0 - 1.0))
    return centers

KEY_CENTERS_CENTERED: List[Tuple[str, float, float]] = build_default_key_centers()

def normalize_points(points: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    if not points:
        return []
    start_t = float(points[0].get("t", 0.0))
    normalized: List[Dict[str, float]] = []
    for idx, pt in enumerate(points):
        raw_x = float(pt.get("x", 0.5))
        raw_y = float(pt.get("y", 0.5))
        centered_x = clamp(raw_x * 2.0 - 1.0, -1.0, 1.0)
        centered_y = clamp(raw_y * 2.0 - 1.0, -1.0, 1.0)
        raw_t = float(pt.get("t", idx * 10.0))
        normalized.append({
            "x": centered_x,
            "y": centered_y,
            "t": max(0.0, raw_t - start_t),
        })
    return normalized

class PersonalizedSwipeFeaturizer:
    """Feature generator mirroring the web demo pipeline."""

    def __init__(self, key_centers: Optional[List[Tuple[str, float, float]]] = None):
        self.key_centers = key_centers or KEY_CENTERS_CENTERED

    def __call__(self, points: Iterable[Dict[str, float]]) -> np.ndarray:
        pts = list(points)
        if not pts or len(pts) < 2:
            return np.zeros((0, 37), dtype=np.float32)

        vectors: List[np.ndarray] = []
        for idx in range(len(pts)):
            vectors.append(self._compute_feature_vector(pts, idx))
        return np.stack(vectors, axis=0).astype(np.float32)

    def _compute_feature_vector(self, points: List[Dict[str, float]], idx: int) -> np.ndarray:
        total = len(points)
        curr = points[idx]
        prev = points[idx - 1] if idx > 0 else None
        prev2 = points[idx - 2] if idx > 1 else None

        x = clamp(float(curr.get("x", 0.0)), -1.0, 1.0)
        y = clamp(float(curr.get("y", 0.0)), -1.0, 1.0)
        t_ms = float(curr.get("t", idx * 10.0))
        t_seconds = t_ms / 1000.0

        vx = vy = speed = 0.0
        if prev is not None:
            prev_t = float(prev.get("t", (idx - 1) * 10.0))
            dt = max((t_ms - prev_t) / 1000.0, 0.001)
            prev_x = clamp(float(prev.get("x", x)), -1.0, 1.0)
            prev_y = clamp(float(prev.get("y", y)), -1.0, 1.0)
            vx = (x - prev_x) / dt
            vy = (y - prev_y) / dt
            speed = math.hypot(vx, vy)

        ax = ay = acc = 0.0
        if prev is not None and prev2 is not None:
            prev_t = float(prev.get("t", (idx - 1) * 10.0))
            prev2_t = float(prev2.get("t", (idx - 2) * 10.0))
            dt1 = max((t_ms - prev_t) / 1000.0, 0.001)
            dt2 = max((prev_t - prev2_t) / 1000.0, 0.001)
            prev_x = clamp(float(prev.get("x", x)), -1.0, 1.0)
            prev_y = clamp(float(prev.get("y", y)), -1.0, 1.0)
            prev2_x = clamp(float(prev2.get("x", prev_x)), -1.0, 1.0)
            prev2_y = clamp(float(prev2.get("y", prev_y)), -1.0, 1.0)
            vx_prev = (prev_x - prev2_x) / dt2
            vy_prev = (prev_y - prev2_y) / dt2
            ax = (vx - vx_prev) / dt1
            ay = (vy - vy_prev) / dt1
            acc = math.hypot(ax, ay)

        angle = math.atan2(vy, vx) if prev is not None else 0.0
        angle_sin = math.sin(angle)
        angle_cos = math.cos(angle)

        curvature = 0.0
        if prev is not None and prev2 is not None:
            prev_x = clamp(float(prev.get("x", x)), -1.0, 1.0)
            prev_y = clamp(float(prev.get("y", y)), -1.0, 1.0)
            prev2_x = clamp(float(prev2.get("x", prev_x)), -1.0, 1.0)
            prev2_y = clamp(float(prev2.get("y", prev_y)), -1.0, 1.0)
            prev_angle = math.atan2(prev_y - prev2_y, prev_x - prev2_x)
            curvature = angle - prev_angle
            while curvature > math.pi:
                curvature -= 2 * math.pi
            while curvature < -math.pi:
                curvature += 2 * math.pi

        distances = []
        for _, kx, ky in self.key_centers:
            distances.append(math.hypot(x - kx, y - ky))
        distances.sort()
        key_distances = distances[:5]
        while len(key_distances) < 5:
            key_distances.append(1.0)

        progress = idx / max(total - 1, 1)
        is_start = 1.0 if idx == 0 else 0.0
        is_end = 1.0 if idx == total - 1 else 0.0

        window_size = 5
        half = window_size // 2
        win_pts = points[max(0, idx - half): min(total, idx + half + 1)]
        if len(win_pts) > 1:
            xs = [clamp(float(p.get("x", x)), -1.0, 1.0) for p in win_pts]
            ys = [clamp(float(p.get("y", y)), -1.0, 1.0) for p in win_pts]
            mean_x, std_x = np.mean(xs), np.std(xs)
            mean_y, std_y = np.mean(ys), np.std(ys)
            range_x, range_y = max(xs) - min(xs), max(ys) - min(ys)
        else:
            mean_x, std_x, mean_y, std_y, range_x, range_y = x, 0.0, y, 0.0, 0.0, 0.0

        features = [
            x, y, t_seconds, vx, vy, speed, ax, ay, acc,
            angle, angle_sin, angle_cos, curvature,
            *key_distances,
            progress, is_start, is_end,
            mean_x, std_x, mean_y, std_y, range_x, range_y,
        ]
        return np.array(features + [0.0] * (37 - len(features)), dtype=np.float32)[:37]
