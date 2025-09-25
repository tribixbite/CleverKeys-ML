
import json
import math
import numpy as np
from typing import Any, Dict, Iterable, List, Optional, Tuple

# --- Copied from new/train_transducer_personalized.py for direct comparison ---

def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))

def determine_resample_target(length: int, cfg: Dict[str, Any]) -> int:
    if length <= 1:
        return length
    short_target = cfg["resample_short_target"]
    long_target = cfg["resample_long_target"]
    short_thresh = cfg["resample_short_threshold"]
    long_thresh = cfg["resample_long_threshold"]
    if length <= short_thresh:
        return short_target
    if length >= long_thresh:
        return long_target
    # Linearly interpolate between short and long targets for intermediate lengths
    progress = (length - short_thresh) / (long_thresh - short_thresh)
    value = short_target + progress * (long_target - short_target)
    print(f"DEBUG: length={length}, progress={progress}, value={value}, int(value)={int(value)}")
    return int(value)

def resample_points(
    points: List[Dict[str, float]], target_count: int
) -> List[Dict[str, float]]:
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
        resampled.append({"x": x, "y": y, "t": target_time})
    return resampled

def load_key_centers(path: Optional[str]) -> List[Tuple[str, float, float]]:
    layout = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
    centers: List[Tuple[str, float, float]] = []
    for row_idx, row in enumerate(layout):
        for col_idx, char in enumerate(row):
            x01 = (col_idx + 0.5) / 10.0
            y01 = (row_idx + 0.5) / 3.0
            centers.append((char, x01 * 2.0 - 1.0, y01 * 2.0 - 1.0))
    return centers

class PersonalizedSwipeFeaturizer:
    FEATURE_NAMES = [
        "x", "y", "t_seconds", "vx", "vy", "speed", "ax", "ay", "acc",
        "angle", "angle_sin", "angle_cos", "curvature", "dist_key1",
        "dist_key2", "dist_key3", "dist_key4", "dist_key5", "progress",
        "is_start", "is_end", "win_mean_x", "win_std_x", "win_mean_y",
        "win_std_y", "win_range_x", "win_range_y",
    ]
    FINAL_FEATURE_COUNT = 37

    def __init__(self, key_centers_path: Optional[str] = None):
        self.key_centers = load_key_centers(key_centers_path)
        self.feature_dim = len(self.FEATURE_NAMES)

    def __call__(self, points: Iterable[Dict[str, float]]) -> np.ndarray:
        pts = list(points)
        if not pts:
            return np.zeros((1, self.FINAL_FEATURE_COUNT), dtype=np.float32)
        vectors: List[np.ndarray] = []
        for idx in range(len(pts)):
            vectors.append(self._compute_feature_vector(pts, idx))
        return np.stack(vectors, axis=0).astype(np.float32)

    def _compute_feature_vector(
        self, points: List[Dict[str, float]], idx: int
    ) -> np.ndarray:
        total = len(points)
        curr = points[idx]
        prev = points[idx - 1] if idx > 0 else None
        prev2 = points[idx - 2] if idx > 1 else None
        x = clamp(float(curr.get("x", 0.0)), -1.0, 1.0)
        y = clamp(float(curr.get("y", 0.0)), -1.0, 1.0)
        t_ms = float(curr.get("t", idx * 10.0))
        t_seconds = t_ms / 1000.0
        vx = vy = speed = 0.0
        if prev:
            dt = max((t_ms - float(prev.get("t", 0.0))) / 1000.0, 1e-6)
            vx = (x - float(prev.get("x", x))) / dt
            vy = (y - float(prev.get("y", y))) / dt
            speed = math.hypot(vx, vy)
        ax = ay = acc = 0.0
        if prev and prev2:
            dt1 = max((t_ms - float(prev.get("t", 0.0))) / 1000.0, 1e-6)
            dt2 = max(
                (float(prev.get("t", 0.0)) - float(prev2.get("t", 0.0))) / 1000.0, 1e-6
            )
            vx_prev = (float(prev.get("x", 0.0)) - float(prev2.get("x", 0.0))) / dt2
            vy_prev = (float(prev.get("y", 0.0)) - float(prev2.get("y", 0.0))) / dt2
            ax = (vx - vx_prev) / dt1
            ay = (vy - vy_prev) / dt1
            acc = math.hypot(ax, ay)
        angle = math.atan2(vy, vx) if prev else 0.0
        curvature = 0.0
        if prev and prev2:
            prev_angle = math.atan2(
                float(prev.get("y", 0.0)) - float(prev2.get("y", 0.0)),
                float(prev.get("x", 0.0)) - float(prev2.get("x", 0.0)),
            )
            curvature = angle - prev_angle
            while curvature > math.pi:
                curvature -= 2 * math.pi
            while curvature < -math.pi:
                curvature += 2 * math.pi
        key_distances = sorted(
            [math.hypot(x - kx, y - ky) for _, kx, ky in self.key_centers]
        )[:5]
        while len(key_distances) < 5:
            key_distances.append(1.0)
        progress = idx / max(total - 1, 1)
        is_start = 1.0 if idx == 0 else 0.0
        is_end = 1.0 if idx == total - 1 else 0.0
        win_pts = points[max(0, idx - 2) : min(total, idx + 3)]
        if len(win_pts) > 1:
            xs = [p["x"] for p in win_pts]
            ys = [p["y"] for p in win_pts]
            win_mean_x, win_std_x = float(np.mean(xs)), float(np.std(xs))
            win_mean_y, win_std_y = float(np.mean(ys)), float(np.std(ys))
            win_range_x, win_range_y = max(xs) - min(xs), max(ys) - min(ys)
        else:
            win_mean_x, win_std_x, win_mean_y, win_std_y, win_range_x, win_range_y = (
                x, 0.0, y, 0.0, 0.0, 0.0,
            )
        feature_dict = {
            "x": x, "y": y, "t_seconds": t_seconds, "vx": vx, "vy": vy, "speed": speed,
            "ax": ax, "ay": ay, "acc": acc, "angle": angle, "angle_sin": math.sin(angle),
            "angle_cos": math.cos(angle), "curvature": curvature, "dist_key1": key_distances[0],
            "dist_key2": key_distances[1], "dist_key3": key_distances[2],
            "dist_key4": key_distances[3], "dist_key5": key_distances[4],
            "progress": progress, "is_start": is_start, "is_end": is_end,
            "win_mean_x": win_mean_x, "win_std_x": win_std_x, "win_mean_y": win_mean_y,
            "win_std_y": win_std_y, "win_range_x": win_range_x, "win_range_y": win_range_y,
        }
        feature_vector = [feature_dict.get(name, 0.0) for name in self.FEATURE_NAMES]
        padding = [0.0] * (self.FINAL_FEATURE_COUNT - len(feature_vector))
        return np.array(feature_vector + padding, dtype=np.float32)

def _prepare_points(points: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    if not points:
        return []
    start_t = float(points[0].get("t", 0.0))
    prepared: List[Dict[str, float]] = []
    for idx, pt in enumerate(points):
        raw_x = float(pt.get("x", 0.0))
        raw_y = float(pt.get("y", 0.0))
        centered_x = clamp(raw_x, -1.0, 1.0)
        centered_y = clamp(raw_y, -1.0, 1.0)
        raw_t = float(pt.get("t", idx * 10.0))
        prepared.append(
            {"x": centered_x, "y": centered_y, "t": max(0.0, raw_t - start_t)}
        )
    return prepared

# --- Main Debug Logic ---

def main():
    test_data = {"word": "raped", "points": [{"x": 0.377898441745, "y": 0.309550308126, "t": 0}, {"x": 0.377898441745, "y": 0.309550308126, "t": 20}, {"x": 0.374141490448, "y": 0.313618343637, "t": 37}, {"x": 0.361931347031, "y": 0.32582209188, "t": 53}, {"x": 0.307455243013, "y": 0.360400393726, "t": 70}, {"x": 0.252979159676, "y": 0.407182443815, "t": 86}, {"x": 0.193806876878, "y": 0.427522621371, "t": 103}, {"x": 0.161872635748, "y": 0.439726727905, "t": 120}, {"x": 0.156237157101, "y": 0.443794763416, "t": 136}, {"x": 0.160933377243, "y": 0.447862798928, "t": 153}, {"x": 0.184414436593, "y": 0.451930834439, "t": 169}, {"x": 0.237951271766, "y": 0.453964493904, "t": 186}, {"x": 0.311212235672, "y": 0.47430467146, "t": 203}, {"x": 0.39574407415, "y": 0.500746902284, "t": 219}, {"x": 0.481215160792, "y": 0.525155115351, "t": 236}, {"x": 0.55541531082, "y": 0.523120739305, "t": 252}, {"x": 0.617405358794, "y": 0.514984668282, "t": 269}, {"x": 0.664367477494, "y": 0.510916632771, "t": 285}, {"x": 0.707572644898, "y": 0.504814937795, "t": 303}, {"x": 0.750777895025, "y": 0.490577171796, "t": 319}, {"x": 0.784590580782, "y": 0.468202618193, "t": 335}, {"x": 0.808071702175, "y": 0.443794763416, "t": 352}, {"x": 0.82873499635, "y": 0.425488603616, "t": 369}, {"x": 0.844702132426, "y": 0.413284855373, "t": 385}, {"x": 0.851276869579, "y": 0.411250479326, "t": 402}, {"x": 0.851276869579, "y": 0.409216819861, "t": 418}, {"x": 0.846580628755, "y": 0.403114408304, "t": 435}, {"x": 0.829674244515, "y": 0.380740571282, "t": 452}, {"x": 0.778955091794, "y": 0.335992180658, "t": 468}, {"x": 0.695362501481, "y": 0.283108077302, "t": 485}, {"x": 0.593924278763, "y": 0.252598169259, "t": 502}, {"x": 0.515027887911, "y": 0.252598169259, "t": 519}, {"x": 0.467126521046, "y": 0.272938346815, "t": 535}, {"x": 0.430496090795, "y": 0.299380219348, "t": 551}, {"x": 0.396683322314, "y": 0.32175441466, "t": 568}, {"x": 0.367566794657, "y": 0.340060216169, "t": 585}, {"x": 0.345024962821, "y": 0.356278211345, "t": 601}, {"x": 0.331686319254, "y": 0.368481959588, "t": 618}, {"x": 0.3289048159, "y": 0.380685707831, "t": 635}, {"x": 0.3289048159, "y": 0.39492347383, "t": 651}, {"x": 0.3289048159, "y": 0.41322963358, "t": 668}, {"x": 0.3289048159, "y": 0.435603470597, "t": 685}, {"x": 0.3289048159, "y": 0.46204534313, "t": 701}, {"x": 0.3289048159, "y": 0.490521233418, "t": 718}, {"x": 0.3289048159, "y": 0.525099535254, "t": 734}, {"x": 0.3289048159, "y": 0.561729965505, "t": 751}, {"x": 0.33078330868, "y": 0.596308267341, "t": 768}, {"x": 0.335379549506, "y": 0.632938697592, "t": 784}, {"x": 0.338097744945, "y": 0.657346910659, "t": 801}]}
    
    preprocess_cfg = {
        "resample_short_target": 56,
        "resample_long_target": 96,
        "resample_short_threshold": 48,
        "resample_long_threshold": 112,
    }

    featurizer = PersonalizedSwipeFeaturizer()

    print("--- DEBUG: Feature Extraction ---")
    
    normalized_points = _prepare_points(test_data["points"])
    target_length = determine_resample_target(len(normalized_points), preprocess_cfg)
    resampled_points = resample_points(normalized_points, target_length)
    feature_matrix = featurizer(resampled_points)

    print("Normalized Points (first 5):", json.dumps(normalized_points[:5], indent=2))
    print("Target Length:", target_length)
    print("Resampled Points (first 5):", json.dumps(resampled_points[:5], indent=2))
    
    # Log the first 2 full feature vectors for comparison
    feature_matrix_for_log = [row.tolist() for row in feature_matrix[:2]]
    print("Feature Matrix (first 2 rows):", json.dumps(feature_matrix_for_log, indent=2))
    print("-----------------------------------")

if __name__ == "__main__":
    main()
