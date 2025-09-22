#!/usr/bin/env python3
"""
Personalized RNN-T training script for gesture swipe models.

This variant bakes in the end-to-end preprocessing pipeline expected by the
updated web demo and on-device stacks. It ingests raw swipe traces as
(x, y, t_ms) where coordinates are in [-1, 1] with 0,0 at the keyboard origin
and t=0 at gesture start. The script performs adaptive resampling, feature
extraction aligned with the web runtime, and optional teacher-student
distillation for smaller deployment footprints.

Key capabilities beyond the baseline trainer:
  * Adaptive resampling towards 56–96 frames depending on trace length.
  * Feature extraction mirroring the JavaScript beam-search frontend.
  * Optional knowledge distillation from a larger teacher RNNT checkpoint.
  * Configurable character-hypothesis budget for downstream decoders.
  * GPU/CPU auto-fallback with sensible Lightning defaults for a 4090M.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import WeightedRandomSampler
from omegaconf import DictConfig
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from collections import Counter
import re

import nemo.collections.asr as nemo_asr

try:
    from nemo.collections.asr.data.audio_to_text import DALIOutputs  # type: ignore
except ImportError:  # pragma: no cover - DALI optional dependency
    DALIOutputs = type(
        "DALIOutputsPlaceholder", (object,), {"has_processed_signal": False}
    )

# Local script dependencies are now expected in the same directory.
from swipe_data_utils import collate_fn

# Import augmentation and progressive unfreezing
try:
    from data_augmentation import SwipeAugmentation, create_augmentation_pipeline

    AUGMENTATION_AVAILABLE = True
except ImportError:
    AUGMENTATION_AVAILABLE = False
    print(
        "Warning: data_augmentation module not available. Augmentation features will be disabled."
    )

try:
    from progressive_unfreezing import ProgressiveUnfreezingCallback

    UNFREEZING_AVAILABLE = True
except ImportError:
    UNFREEZING_AVAILABLE = False
    print(
        "Warning: progressive_unfreezing module not available. Unfreezing features will be disabled."
    )

SCRIPT_DIR = Path(__file__).resolve().parent
runtime_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# This configuration dictionary defines all hyperparameters for the training run.
# Inline comments explain the role and rationale for each parameter, focusing on
# creating a robust, on-device model.

CONFIG: Dict[str, Any] = {
    # --- Data and Vocabulary Paths ---
    # These are now default values, overrideable via command-line arguments
    # for better portability and experiment management.
    "data": {
        "train_manifest": "/home/will/git/swype/cleverkeys/data/train_final_train.jsonl",
        "val_manifest": "/home/will/git/swype/cleverkeys/data/train_final_val.jsonl",
        "vocab_path": "/home/will/git/swype/cleverkeys/data/vocab.txt",
        "key_centers_path": None,  # Optional: Path to a JSON file defining keyboard layout for featurization.
        "max_trace_len": 256,  # Safety limit to prevent excessively long traces from consuming too much memory.
    },
    # --- Training Hyperparameters ---
    "training": {
        "batch_size": 1000,  # Maximize GPU utilization on a 16GB+ card. Larger batches provide more stable gradients.
        "num_workers": 10,  # Use multiple workers to pre-fetch data and keep the GPU saturated. Set based on CPU cores.
        "learning_rate": 2e-4,  # A conservative learning rate for the AdamW optimizer, good for stable convergence.
        "max_epochs": 220,  # Total number of training epochs.
        "gradient_accumulation": 1,  # Accumulate gradients over multiple batches. Useful for simulating larger batch sizes on smaller GPUs.
        "accelerator": "gpu",  # Use 'gpu' if available, will auto-fallback to 'cpu'.
        "devices": 1,  # Number of GPUs to use.
        "precision": "bf16-mixed",  # Best performance on modern (Ampere+) GPUs. Avoids CUDA graph issues seen with fp16.
        "warmup_steps": 1500,  # Number of initial steps to linearly increase the learning rate, preventing instability at the start.
        "teacher_checkpoint": None,  # For Knowledge Distillation: path to a larger "teacher" model.
        "kd_lambda": 0.05,  # Weight of the distillation loss. Balances learning from the teacher vs. the true labels.
        "kd_temperature": 1.5,  # Softens the teacher's output distribution, providing richer training signals for the student.
    },
    # --- Core Model Architecture (Conformer-RNNT) ---
    "model": {
        "encoder": {
            "feat_in": 37,  # Input feature dimension from PersonalizedSwipeFeaturizer. MUST match feature output.
            "d_model": 256,  # The main hidden dimension of the Conformer model. A balance between capacity and on-device performance.
            "n_heads": 4,  # Number of attention heads in the multi-head self-attention layers.
            "num_layers": 6,  # Number of Conformer blocks. More layers increase accuracy but also model size and inference time.
            "conv_kernel_size": 31,  # Kernel size for the convolution module within each Conformer block. Captures local patterns.
            "subsampling_factor": 2,  # Reduces the sequence length early in the model, saving computation.
        },
        "decoder": {  # The "Prediction Network" in RNN-T
            "pred_hidden": 320,  # Hidden size of the LSTM decoder.
            "pred_rnn_layers": 2,  # Two LSTM layers provide sufficient power to model character-level dependencies (e.g., 'q' -> 'u').
        },
        "joint": {  # Combines encoder and decoder outputs
            "joint_hidden": 512,  # Hidden size of the feed-forward network in the joint.
            "activation": "relu",
            "dropout": 0.1,
        },
    },
    # --- Preprocessing ---
    "preprocess": {
        # Adaptive resampling normalizes gesture length for robustness and efficient batching.
        "resample_short_target": 56,  # Target frame count for shorter swipes. Ensures a minimum information density.
        "resample_long_target": 96,  # Target frame count for longer swipes. Compresses long gestures to a manageable length.
        "resample_short_threshold": 48,  # Gestures shorter than this are considered "short".
        "resample_long_threshold": 112,  # Gestures longer than this are considered "long".
    },
    # --- Weighted Sampling Strategy ---
    # This is critical for building a robust model that handles rare words well.
    "sampling": {
        "strategy": "inverse_sqrt_freq",  # Oversamples rare words to help the model learn the long-tail of the vocabulary.
        "freq_power": 0.55,  # Exponent for frequency weighting. Higher values give more weight to the rarest words.
        "length_power": 0.8,  # Exponent for word length weighting. Higher values focus more on longer words.
        "rare_frequency_threshold": 25,  # Words seen fewer than this many times are considered "rare".
        "rare_word_boost": 3.5,  # A direct multiplier for rare words. A key parameter for improving tail-end accuracy.
        "max_weight_factor": 12.0,  # Caps the maximum weight of any sample to prevent extreme values from destabilizing training.
        "min_word_length": 1,  # Include all words by default to avoid biasing against short words.
        "max_frequency": 3000,  # Can be used to ignore very common words to focus on the middle/rare part of the vocabulary.
    },
    # --- Validation Parameters ---
    "validation": {
        "strategy": "inverse_sqrt_freq",  # Can also apply sampling to validation to focus evaluation on challenging cases.
        "freq_power": 0.75,
        "length_power": 1.3,
        "min_word_length": 2,
        "rare_frequency_threshold": 40,
        "rare_word_boost": 6.0,
        "max_weight_factor": 28.0,
        "batch_size_factor": 0.35,  # Use a smaller batch size for validation.
        "limit_batches": 0.15,  # To speed up validation, run on a random 15% subset of the validation set each time.
        "check_interval": 0.5,  # Run validation 3 times per training epoch.
        "max_samples": 1500,  # Limit the number of validation samples used.
        "log_error_batches": 3,  # Print mispredictions from the first 3 validation batches for qualitative analysis.
    },
    # --- Data Augmentation ---
    # Creates more diverse training data to make the model more robust to real-world gesture variations.
    "augmentation": {
        "enabled": False,  # Master switch, can be enabled with the --augment flag.
        "noise_std": 0.02,
        "time_warp_factor": 0.1,
        "spatial_shift_range": 0.05,
        "speed_variation": 0.2,
        "enable_for_rare_only": True,  # CRITICAL: Only apply augmentation to rare words, efficiently boosting data where it's needed most.
        "rare_threshold": 50,  # Definition of "rare" for the purpose of augmentation.
        "augmentation_prob": 0.5,  # Apply augmentation to 50% of eligible (rare) samples.
    },
    # --- Progressive Unfreezing ---
    # An advanced fine-tuning technique to improve accuracy when resuming from a checkpoint.
    "unfreezing": {
        "enabled": False,  # Master switch, can be enabled with the --unfreeze flag.
        "warmup_epochs": 2,  # Number of epochs to train only the model head before starting to unfreeze layers.
        "discriminative_lr": True,  # Use smaller learning rates for earlier layers of the model.
        "lr_decay_factor": 0.95,  # Decay factor for discriminative learning rates.
    },
}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _has_usable_cuda() -> bool:
    """Robust CUDA availability check that tolerates driver hiccups."""
    try:
        if not torch.cuda.is_available():
            return False
        if torch.cuda.device_count() <= 0:
            return False
        _ = torch.cuda.current_device()
        return True
    except Exception as exc:  # pragma: no cover - defensive guard
        print(f"CUDA probe failed ({exc}); falling back to CPU")
        return False


def _resolve_path(path_str: str) -> str:
    """Resolves a path relative to the script's directory if not absolute."""
    path = Path(path_str)
    if not path.is_absolute():
        path = (SCRIPT_DIR / path).resolve()
    return str(path)


def clamp(value: float, minimum: float, maximum: float) -> float:
    """Clamps a value to a specified min/max range."""
    return max(minimum, min(value, maximum))


def determine_resample_target(length: int, cfg: Dict[str, Any]) -> int:
    """
    Determines the target number of frames for adaptive resampling.
    This version uses linear interpolation for a smoother transition, which can
    lead to more stable training than a hard-threshold approach.
    """
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
    return int(short_target + progress * (long_target - short_target))


def resample_points(
    points: List[Dict[str, float]], target_count: int
) -> List[Dict[str, float]]:
    """Resamples a sequence of points to a target count using linear interpolation."""
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
    """Loads key centers from a JSON file or returns a default QWERTY layout."""
    if path:
        try:
            with open(_resolve_path(path), "r") as f:
                return json.load(f)
        except Exception as e:
            print(
                f"Warning: Could not load key centers from {path}. Using default. Error: {e}"
            )

    layout = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
    centers: List[Tuple[str, float, float]] = []
    for row_idx, row in enumerate(layout):
        for col_idx, char in enumerate(row):
            x01 = (col_idx + 0.5) / 10.0
            y01 = (row_idx + 0.5) / 3.0
            centers.append((char, x01 * 2.0 - 1.0, y01 * 2.0 - 1.0))
    return centers


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


class PersonalizedSwipeFeaturizer:
    """
    Feature generator mirroring the web demo pipeline.
    This class is responsible for turning a sequence of points into a rich,
    37-dimensional feature vector that the model can learn from.
    """

    FEATURE_NAMES = [
        "x",
        "y",
        "t_seconds",
        "vx",
        "vy",
        "speed",
        "ax",
        "ay",
        "acc",
        "angle",
        "angle_sin",
        "angle_cos",
        "curvature",
        "dist_key1",
        "dist_key2",
        "dist_key3",
        "dist_key4",
        "dist_key5",
        "progress",
        "is_start",
        "is_end",
        "win_mean_x",
        "win_std_x",
        "win_mean_y",
        "win_std_y",
        "win_range_x",
        "win_range_y",
    ]
    # Pad to 37 features for model compatibility
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
        # This method computes a single feature vector for one point in the swipe.
        # It captures kinematics, trajectory shape, and spatial/temporal context.
        total = len(points)
        curr = points[idx]
        prev = points[idx - 1] if idx > 0 else None
        prev2 = points[idx - 2] if idx > 1 else None

        # --- Basic Kinematics ---
        x = clamp(float(curr.get("x", 0.0)), -1.0, 1.0)
        y = clamp(float(curr.get("y", 0.0)), -1.0, 1.0)
        t_ms = float(curr.get("t", idx * 10.0))
        t_seconds = t_ms / 1000.0

        # --- Velocity ---
        vx = vy = speed = 0.0
        if prev:
            dt = max((t_ms - float(prev.get("t", 0.0))) / 1000.0, 1e-6)
            vx = (x - float(prev.get("x", x))) / dt
            vy = (y - float(prev.get("y", y))) / dt
            speed = math.hypot(vx, vy)

        # --- Acceleration ---
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

        # --- Trajectory Shape ---
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

        # --- Spatial Context: Distances to 5 nearest keyboard keys ---
        key_distances = sorted(
            [math.hypot(x - kx, y - ky) for _, kx, ky in self.key_centers]
        )[:5]
        while len(key_distances) < 5:
            key_distances.append(1.0)

        # --- Temporal Context ---
        progress = idx / max(total - 1, 1)
        is_start = 1.0 if idx == 0 else 0.0
        is_end = 1.0 if idx == total - 1 else 0.0

        # --- Windowed Stats: Captures local trajectory stability and shape ---
        win_pts = points[max(0, idx - 2) : min(total, idx + 3)]
        if len(win_pts) > 1:
            xs = [p["x"] for p in win_pts]
            ys = [p["y"] for p in win_pts]
            win_mean_x, win_std_x = float(np.mean(xs)), float(np.std(xs))
            win_mean_y, win_std_y = float(np.mean(ys)), float(np.std(ys))
            win_range_x, win_range_y = max(xs) - min(xs), max(ys) - min(ys)
        else:
            win_mean_x, win_std_x, win_mean_y, win_std_y, win_range_x, win_range_y = (
                x,
                0.0,
                y,
                0.0,
                0.0,
                0.0,
            )

        # --- Assemble Feature Vector ---
        # This dictionary-based approach is more robust than a simple list.
        feature_dict = {
            "x": x,
            "y": y,
            "t_seconds": t_seconds,
            "vx": vx,
            "vy": vy,
            "speed": speed,
            "ax": ax,
            "ay": ay,
            "acc": acc,
            "angle": angle,
            "angle_sin": math.sin(angle),
            "angle_cos": math.cos(angle),
            "curvature": curvature,
            "dist_key1": key_distances[0],
            "dist_key2": key_distances[1],
            "dist_key3": key_distances[2],
            "dist_key4": key_distances[3],
            "dist_key5": key_distances[4],
            "progress": progress,
            "is_start": is_start,
            "is_end": is_end,
            "win_mean_x": win_mean_x,
            "win_std_x": win_std_x,
            "win_mean_y": win_mean_y,
            "win_std_y": win_std_y,
            "win_range_x": win_range_x,
            "win_range_y": win_range_y,
        }

        feature_vector = [feature_dict.get(name, 0.0) for name in self.FEATURE_NAMES]

        # Pad with zeros to match the model's expected input dimension (37).
        padding = [0.0] * (self.FINAL_FEATURE_COUNT - len(feature_vector))
        return np.array(feature_vector + padding, dtype=np.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class PersonalizedSwipeDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        vocab: Dict[str, int],
        max_trace_len: int,
        preprocess_cfg: Dict[str, Any],
        featurizer: PersonalizedSwipeFeaturizer,
        augmenter: Optional["SwipeAugmentation"] = None,
        is_training: bool = False,
    ) -> None:
        super().__init__()
        self.manifest_path = manifest_path
        self.vocab = vocab
        self.max_trace_len = max_trace_len
        self.preprocess_cfg = preprocess_cfg
        self.featurizer = featurizer
        self.augmenter = augmenter
        self.is_training = is_training
        self.samples: List[Dict[str, Any]] = []

        try:
            with open(manifest_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    payload = json.loads(line)
                    if not payload.get("word") or not payload.get("points"):
                        continue
                    self.samples.append(payload)
        except FileNotFoundError:
            print(f"FATAL: Manifest file not found at {manifest_path}")
            raise

        self.word_counts = Counter(sample["word"] for sample in self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        item = self.samples[index]

        if self.is_training and self.augmenter:
            item = self.augmenter.augment(item)

        raw_points = item["points"][: self.max_trace_len]

        normalized = self._prepare_points(raw_points)
        target_len = determine_resample_target(len(normalized), self.preprocess_cfg)
        processed = resample_points(normalized, target_len)

        features = self.featurizer(processed)
        features_tensor = torch.from_numpy(features).float()

        tokens = [self.vocab.get(ch, self.vocab.get("<unk>", 0)) for ch in item["word"]]
        tokens_tensor = torch.tensor(tokens, dtype=torch.long)

        return (
            features_tensor,
            torch.tensor(features_tensor.shape[0], dtype=torch.long),
            tokens_tensor,
            torch.tensor(len(tokens), dtype=torch.long),
        )

    def compute_sampling_weights(
        self, sampling_cfg: Dict[str, Any]
    ) -> Optional[np.ndarray]:
        strategy = sampling_cfg.get("strategy", "none")
        if strategy == "none" or not self.samples:
            return None

        # Unpack all sampling parameters from config for clarity
        freq_power = float(sampling_cfg.get("freq_power", 0.5))
        length_power = float(sampling_cfg.get("length_power", 0.0))
        rare_freq_thresh = int(sampling_cfg.get("rare_frequency_threshold", 0))
        rare_boost = float(sampling_cfg.get("rare_word_boost", 1.0))
        max_weight = float(sampling_cfg.get("max_weight_factor", 10.0))
        min_len = int(sampling_cfg.get("min_word_length", 1))
        max_len = (
            int(sampling_cfg.get("max_word_length", 99))
            if sampling_cfg.get("max_word_length")
            else 99
        )
        max_freq = (
            int(sampling_cfg.get("max_frequency", 999999))
            if sampling_cfg.get("max_frequency")
            else 999999
        )

        weights: List[float] = []
        for sample in self.samples:
            word = sample["word"]
            freq = self.word_counts.get(word, 1)
            word_len = len(word)

            # Apply filters: skip samples that don't meet criteria
            if not (min_len <= word_len <= max_len and freq <= max_freq):
                weights.append(0.0)
                continue

            # Calculate base weight from frequency
            weight = freq ** -abs(freq_power)

            # Apply word length bonus
            if length_power != 0:
                weight *= max(word_len, 1) ** length_power

            # Apply boost for rare words
            if rare_freq_thresh and freq <= rare_freq_thresh:
                weight *= rare_boost

            weights.append(weight)

        if not weights or sum(weights) == 0:
            return None

        weights_arr = np.asarray(weights, dtype=np.float64)
        weights_arr /= weights_arr.mean()  # Normalize weights
        if max_weight > 0:
            weights_arr = np.clip(weights_arr, 1.0 / max_weight, max_weight)
        return weights_arr

    @staticmethod
    def _prepare_points(points: List[Dict[str, Any]]) -> List[Dict[str, float]]:
        """Prepares raw points by ensuring they are in [-1, 1] and time is relative."""
        if not points:
            return []
        start_t = float(points[0].get("t", 0.0))
        prepared: List[Dict[str, float]] = []
        for idx, pt in enumerate(points):
            raw_x = float(pt.get("x", 0.0))
            raw_y = float(pt.get("y", 0.0))

            # BUG FIX: The original implementation incorrectly assumed input coordinates were in [0, 1]
            # and applied a `* 2.0 - 1.0` transformation. The data is already in [-1, 1],
            # so we just need to clamp it to ensure it's within the expected bounds.
            centered_x = clamp(raw_x, -1.0, 1.0)
            centered_y = clamp(raw_y, -1.0, 1.0)

            raw_t = float(pt.get("t", idx * 10.0))
            prepared.append(
                {
                    "x": centered_x,
                    "y": centered_y,
                    "t": max(0.0, raw_t - start_t),
                }
            )
        return prepared


# ---------------------------------------------------------------------------
# Distillation-ready RNNT model
# ---------------------------------------------------------------------------


class PersonalizedRNNTModel(nemo_asr.models.EncDecRNNTModel):
    def __init__(
        self,
        cfg: DictConfig,
        kd_lambda: float = 0.0,
        kd_temperature: float = 1.0,
        teacher_checkpoint: Optional[str] = None,
    ):
        super().__init__(cfg=cfg)
        self.kd_lambda = kd_lambda
        self.kd_temperature = kd_temperature
        self.teacher = None
        if teacher_checkpoint:
            self._init_teacher(teacher_checkpoint)

        # Disable Nemo's random sample logging; we'll log mismatches manually
        if hasattr(self, "wer") and self.wer is not None:
            try:
                self.wer.log_prediction = False
            except AttributeError:
                pass

    def _init_teacher(self, checkpoint: str) -> None:
        """Initializes a frozen teacher model for knowledge distillation."""
        print(f"Loading teacher model from: {checkpoint}")
        self.teacher = nemo_asr.models.EncDecRNNTModel.restore_from(
            checkpoint, map_location="cpu"
        )
        self.teacher.freeze()
        self.teacher.eval()

    def forward(
        self,
        input_signal=None,
        input_signal_length=None,
        processed_signal=None,
        processed_signal_length=None,
    ):
        """Bypass the audio preprocessor; accept already-computed feature tensors."""
        if input_signal is not None and processed_signal is None:
            # NeMo's Conformer expects (batch, features, time), but our data is (batch, time, features).
            processed_signal = input_signal.transpose(1, 2)
            processed_signal_length = input_signal_length
        elif processed_signal is None:
            raise ValueError(
                "PersonalizedRNNTModel.forward requires input_signal or processed_signal"
            )

        encoded, encoded_len = self.encoder(
            audio_signal=processed_signal, length=processed_signal_length
        )
        return encoded, encoded_len

    def _compute_joint(
        self, model: nemo_asr.models.EncDecRNNTModel, batch, detach: bool = True
    ):
        """Helper to compute joint network output, used for teacher in KD."""
        signal, signal_len, transcript, transcript_len = batch
        encoded, encoded_len = model.forward(
            input_signal=signal, input_signal_length=signal_len
        )
        decoder, target_length, _ = model.decoder(
            targets=transcript, target_length=transcript_len
        )
        joint = model.joint(encoder_outputs=encoded, decoder_outputs=decoder)
        return joint.detach() if detach else joint

    def training_step(self, batch, batch_idx):
        signal, signal_len, transcript, transcript_len = batch
        encoded, encoded_len = self.forward(
            input_signal=signal, input_signal_length=signal_len
        )
        decoder, target_length, _ = self.decoder(
            targets=transcript, target_length=transcript_len
        )
        joint = self.joint(encoder_outputs=encoded, decoder_outputs=decoder)

        # --- Standard RNN-T Loss ---
        loss_value = self.loss(
            log_probs=joint,
            targets=transcript,
            input_lengths=encoded_len,
            target_lengths=target_length,
        )
        loss_value = self.add_auxiliary_losses(loss_value)

        # --- Knowledge Distillation Loss (Optional) ---
        kd_loss = None
        if self.teacher is not None and self.kd_lambda > 0:
            with torch.no_grad():
                teacher_joint = self._compute_joint(self.teacher, batch, detach=True)

            student_log_probs = torch.nn.functional.log_softmax(
                joint / self.kd_temperature, dim=-1
            )
            teacher_probs = torch.nn.functional.softmax(
                teacher_joint / self.kd_temperature, dim=-1
            )

            kd_loss = torch.nn.functional.kl_div(
                student_log_probs,
                teacher_probs,
                reduction="batchmean",
                log_target=False,
            ) * (self.kd_lambda * (self.kd_temperature**2))
            loss_value = loss_value + kd_loss

        # --- Logging ---
        logs = {
            "train_loss": loss_value,
            "learning_rate": self._optimizer.param_groups[0]["lr"],
        }
        if kd_loss is not None:
            logs["kd_loss"] = kd_loss.detach()

        # Log training WER periodically
        if (batch_idx + 1) % self.trainer.log_every_n_steps == 0:
            with torch.cuda.amp.autocast(enabled=False):  # Use fp32 for metrics
                self.wer.update(
                    predictions=encoded,
                    predictions_lengths=encoded_len,
                    targets=transcript,
                    targets_lengths=transcript_len,
                )
                _, scores, words = self.wer.compute()
                self.wer.reset()
                logs["training_batch_wer"] = scores.float() / words

        self.log_dict(logs)
        return {"loss": loss_value}

    def _decode_targets(
        self, transcript: torch.Tensor, transcript_len: torch.Tensor
    ) -> List[str]:
        """Helper to convert token tensors to strings."""
        references: List[str] = []
        for tokens, length in zip(transcript, transcript_len):
            token_list = tokens[: int(length.item())].detach().cpu().numpy().tolist()
            references.append(self.decoding.decode_tokens_to_str(token_list))
        return references

    def _log_batch_errors(self, batch, stage: str) -> None:
        """Prints mispredictions for a batch to the console for qualitative analysis."""
        try:
            with torch.no_grad():
                signal, signal_len, transcript, transcript_len = batch
                encoded, encoded_len = self.forward(
                    input_signal=signal, input_signal_length=signal_len
                )
                predictions = self.decoding.rnnt_decoder_predictions_tensor(
                    encoded, encoded_len
                )
                references = self._decode_targets(transcript, transcript_len)

                for ref, pred in zip(references, predictions):
                    pred_text = (
                        pred[0].text if isinstance(pred, list) and pred else str(pred)
                    )
                    if pred_text != ref:
                        print(
                            f"\033[1;33m[{stage} mispred]\033[0m ref='{ref}' pred='{pred_text}'"
                        )
                        return  # Log only the first error per batch
        except Exception as e:
            print(f"Could not log batch errors: {e}")

    def validation_step(self, *args, **kwargs):
        with torch.cuda.amp.autocast(
            enabled=False
        ):  # Run validation in fp32 for stability
            return super().validation_step(*args, **kwargs)

    def test_step(self, *args, **kwargs):
        with torch.cuda.amp.autocast(enabled=False):  # Run tests in fp32 for stability
            return super().test_step(*args, **kwargs)


# ---------------------------------------------------------------------------
# Training orchestration
# ---------------------------------------------------------------------------


class AnnounceCheckpoint(ModelCheckpoint):
    """Callback to print a clear message when a new checkpoint is saved."""

    def _save_model(self, trainer, filepath: str) -> None:
        super()._save_model(trainer, filepath)
        if getattr(trainer, "is_global_zero", True):
            print(f"\033[1;32m[saved checkpoint]\033[0m {filepath}")


class ValidationErrorLogger(pl.Callback):
    """Callback to log mispredictions during validation."""

    def __init__(self, max_batches: int = 1):
        self.max_batches = max_batches
        self._logged_this_epoch = 0

    def on_validation_epoch_start(self, trainer, pl_module):
        self._logged_this_epoch = 0

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if self._logged_this_epoch < self.max_batches:
            pl_module._log_batch_errors(batch, stage="validation")
            self._logged_this_epoch += 1


def load_vocab(vocab_path: str) -> Dict[str, int]:
    """Loads the vocabulary file into a dictionary."""
    vocab: Dict[str, int] = {}
    with open(vocab_path, "r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            token = line.strip()
            if token:
                vocab[token] = idx
    if "<unk>" not in vocab:
        vocab["<unk>"] = len(vocab)
    return vocab


def build_dataloaders(cfg: DictConfig, vocab: Dict[str, int]):
    """Builds and configures the training and validation DataLoaders."""
    featurizer = PersonalizedSwipeFeaturizer(cfg.data.get("key_centers_path"))

    augmenter = None
    if AUGMENTATION_AVAILABLE and cfg.augmentation.get("enabled", False):
        augmenter = create_augmentation_pipeline(cfg.augmentation)
        train_word_counts = Counter()
        with open(cfg.data.train_manifest, "r") as f:
            for line in f:
                train_word_counts[json.loads(line).get("word", "")] += 1
        augmenter.set_word_frequencies(dict(train_word_counts))
        print(
            f"Data augmentation enabled for rare words (threshold={cfg.augmentation.rare_threshold})"
        )

    train_ds = PersonalizedSwipeDataset(
        manifest_path=cfg.data.train_manifest,
        vocab=vocab,
        max_trace_len=cfg.data.max_trace_len,
        preprocess_cfg=cfg.preprocess,
        featurizer=featurizer,
        augmenter=augmenter,
        is_training=True,
    )
    val_ds = PersonalizedSwipeDataset(
        manifest_path=cfg.data.val_manifest,
        vocab=vocab,
        max_trace_len=cfg.data.max_trace_len,
        preprocess_cfg=cfg.preprocess,
        featurizer=featurizer,
        augmenter=None,
        is_training=False,
    )

    train_weights = train_ds.compute_sampling_weights(cfg.sampling)
    train_sampler = None
    if train_weights is not None:
        train_sampler = WeightedRandomSampler(
            weights=torch.from_numpy(train_weights),
            num_samples=len(train_weights),
            replacement=True,
        )
        print(
            f"Enabled '{cfg.sampling.strategy}' sampling (weight range {train_weights.min():.3f}–{train_weights.max():.3f})"
        )

    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=cfg.training.batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=cfg.training.num_workers,
        collate_fn=collate_fn,
        pin_memory=_has_usable_cuda(),
        drop_last=True,
        persistent_workers=cfg.training.num_workers > 0,
        prefetch_factor=4 if cfg.training.num_workers > 0 else 2,
    )

    val_sampler = None
    val_weights = val_ds.compute_sampling_weights(cfg.validation)
    if val_weights is not None:
        num_samples = min(
            len(val_weights), int(cfg.validation.get("max_samples", len(val_weights)))
        )
        val_sampler = WeightedRandomSampler(
            weights=torch.from_numpy(val_weights),
            num_samples=num_samples,
            replacement=False,
        )

    val_workers = max(0, cfg.training.num_workers // 2)
    val_batch_size = max(
        1, int(cfg.training.batch_size * cfg.validation.get("batch_size_factor", 0.5))
    )
    val_loader = DataLoader(
        dataset=val_ds,
        batch_size=val_batch_size,
        sampler=val_sampler,
        shuffle=val_sampler is None,
        num_workers=val_workers,
        collate_fn=collate_fn,
        pin_memory=_has_usable_cuda(),
        drop_last=False,  # drop_last=False is important for validation
        persistent_workers=val_workers > 0,
        prefetch_factor=2 if val_workers > 0 else None,
    )
    return train_loader, val_loader


def build_model_config(cfg: DictConfig, labels: List[str]) -> DictConfig:
    """Builds the full NeMo model configuration from the script's config."""
    return DictConfig(
        {
            "labels": labels,
            "sample_rate": 16000,
            "model_defaults": {
                "enc_hidden": cfg.model.encoder.d_model,
                "pred_hidden": cfg.model.decoder.pred_hidden,
            },
            "preprocessor": {
                "_target_": "nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor",
                "features": cfg.model.encoder.feat_in,
                "normalize": "per_feature",
                "sample_rate": 16000,
                "window_size": 0.025,
                "window_stride": 0.01,
                "n_fft": 512,
                "frame_splicing": 1,
                "dither": 0.0,
            },
            "encoder": {
                "_target_": "nemo.collections.asr.modules.ConformerEncoder",
                "feat_in": cfg.model.encoder.feat_in,
                "n_layers": cfg.model.encoder.num_layers,
                "d_model": cfg.model.encoder.d_model,
                "feat_out": -1,
                "subsampling": "striding",
                "subsampling_factor": cfg.model.encoder.subsampling_factor,
                "subsampling_conv_channels": -1,
                "ff_expansion_factor": 4,
                "self_attention_model": "rel_pos",
                "n_heads": cfg.model.encoder.n_heads,
                "att_context_size": [-1, -1],
                "conv_kernel_size": cfg.model.encoder.conv_kernel_size,
                "dropout": 0.1,
                "pos_emb_max_len": 4000,
            },
            "decoder": {
                "_target_": "nemo.collections.asr.modules.rnnt.RNNTDecoder",
                "prednet": {
                    "pred_hidden": cfg.model.decoder.pred_hidden,
                    "pred_rnn_layers": cfg.model.decoder.pred_rnn_layers,
                    "forget_gate_bias": 1.0,
                    "t_max": None,
                    "dropout": 0.1,
                },
                "vocab_size": len(labels),
                "blank_as_pad": True,
            },
            "joint": {
                "_target_": "nemo.collections.asr.modules.rnnt.RNNTJoint",
                "jointnet": {
                    "joint_hidden": cfg.model.joint.joint_hidden,
                    "activation": cfg.model.joint.activation,
                    "dropout": cfg.model.joint.dropout,
                },
                "num_classes": len(labels),
                "vocabulary": labels,
                "log_softmax": True,
            },
            "decoding": {
                "strategy": "greedy_batch",
                "use_cuda_graphs": False,
                "greedy": {"max_symbols": 15},
                "greedy_batch": {"max_symbols": 13, "enable_cuda_graphs": False},
            },
            "loss": {"_target_": "nemo.collections.asr.losses.rnnt_loss.RNNTLoss"},
            "optim": {
                "name": "adamw",
                "lr": cfg.training.learning_rate,
                "betas": [0.9, 0.98],
                "weight_decay": 1e-3,
                "sched": {
                    "name": "CosineAnnealing",
                    "warmup_steps": cfg.training.warmup_steps,
                    "last_epoch": -1,
                },
            },
        }
    )


def find_latest_checkpoint(prefer_checkpoint: Optional[str] = None) -> Optional[str]:
    """Finds the best checkpoint to resume from, prioritizing recency and epoch."""
    if prefer_checkpoint and Path(prefer_checkpoint).exists():
        print(f"Using specified checkpoint: {prefer_checkpoint}")
        return prefer_checkpoint

    # Search multiple common locations for checkpoints
    project_root = Path("/home/will/git/swype/cleverkeys")
    base_dirs = {SCRIPT_DIR.resolve(), Path.cwd().resolve(), project_root}
    patterns = ["rnnt_checkpoints_*/**/*.ckpt", "rnnt_logs_*/**/*.ckpt"]
    candidates = {
        p for base in base_dirs for pattern in patterns for p in base.glob(pattern)
    }
    if not candidates:
        return None

    date_pattern = re.compile(r"(\d{8}_\d{6})")
    epoch_pattern = re.compile(r"epoch=(\d+)")

    best_path, best_key = None, ("0", -1)  # (date_str, epoch)

    for path in candidates:
        date_match = date_pattern.search(str(path))
        date_str = date_match.group(1) if date_match else "0"

        epoch_match = epoch_pattern.search(path.name)
        epoch = int(epoch_match.group(1)) if epoch_match else -1

        if date_str > best_key[0] or (date_str == best_key[0] and epoch > best_key[1]):
            best_key = (date_str, epoch)
            best_path = path

    if best_path:
        print(
            f"Resuming from best checkpoint: {best_path.name} (run: {best_key[0]}, epoch: {best_key[1]})"
        )
        return str(best_path)
    return None


def load_sampling_profile(profile_name: Optional[str]) -> Optional[Dict[str, Any]]:
    """Loads a sampling profile by name from sampling_profiles.py."""
    if not profile_name:
        return None
    try:
        from sampling_profiles import get_profile

        return get_profile(profile_name)
    except (ImportError, ValueError) as e:
        print(
            f"Warning: Could not load sampling profile '{profile_name}'. Using default. Error: {e}"
        )
        return None


def main() -> None:
    cfg = DictConfig(CONFIG)

    # --- CLI Arguments ---
    parser = argparse.ArgumentParser(
        description="Train Personalized RNNT for CleverKeys"
    )
    # Feature toggles
    parser.add_argument(
        "--augment", action="store_true", help="Enable data augmentation"
    )
    parser.add_argument(
        "--unfreeze",
        action="store_true",
        help="Enable progressive unfreezing if available",
    )
    # Sampling profiles
    try:
        # Discover available profiles for better UX
        from sampling_profiles import SAMPLING_PROFILES as _SP  # type: ignore

        profile_choices = sorted(list(_SP.keys()))
    except Exception:
        profile_choices = None
    parser.add_argument(
        "--profile",
        type=str,
        choices=profile_choices,
        help="Sampling profile for TRAIN set (see new/sampling_profiles.py)",
    )
    parser.add_argument(
        "--val-profile",
        dest="val_profile",
        type=str,
        choices=profile_choices,
        help="Sampling profile for VALIDATION set",
    )
    # Training overrides
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Resume from specific checkpoint path",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Override training batch size"
    )
    parser.add_argument(
        "--num-workers", type=int, default=None, help="Override DataLoader worker count"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override optimizer learning rate",
    )
    args = parser.parse_args()

    # --- Resolve Paths and Set up Environment ---
    cfg.data.train_manifest = _resolve_path(cfg.data.train_manifest)
    cfg.data.val_manifest = _resolve_path(cfg.data.val_manifest)
    cfg.data.vocab_path = _resolve_path(cfg.data.vocab_path)

    if not _has_usable_cuda():
        cfg.training.accelerator, cfg.training.precision, cfg.training.num_workers = (
            "cpu",
            "32-true",
            0,
        )
    else:
        # Optimizations for modern GPUs (RTX 4090M)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # --- Apply CLI overrides ---
    if args.augment:
        cfg.augmentation.enabled = True
    if args.unfreeze and UNFREEZING_AVAILABLE:
        cfg.unfreezing.enabled = True
    if args.batch_size is not None and args.batch_size > 0:
        cfg.training.batch_size = int(args.batch_size)
    if args.num_workers is not None and args.num_workers >= 0:
        cfg.training.num_workers = int(args.num_workers)
    if args.learning_rate is not None and args.learning_rate > 0:
        cfg.training.learning_rate = float(args.learning_rate)

    # Load sampling profiles for training/validation when provided
    if args.profile:
        prof = load_sampling_profile(args.profile)
        if prof:
            cfg.sampling.update(prof)
            print(f"Applied training sampling profile: {args.profile}")
    if args.val_profile:
        vprof = load_sampling_profile(args.val_profile)
        if vprof:
            cfg.validation.update(vprof)
            print(f"Applied validation sampling profile: {args.val_profile}")

    # --- Build Model and DataLoaders ---
    vocab = load_vocab(cfg.data.vocab_path)
    train_loader, val_loader = build_dataloaders(cfg, vocab)
    nemo_cfg = build_model_config(cfg, list(vocab.keys()))

    model = PersonalizedRNNTModel(
        cfg=nemo_cfg,
        kd_lambda=cfg.training.kd_lambda,
        kd_temperature=cfg.training.kd_temperature,
        teacher_checkpoint=_resolve_path(cfg.training.teacher_checkpoint)
        if cfg.training.teacher_checkpoint
        else None,
    )

    # Optional: torch.compile for speedup (PyTorch 2.x)
    try:
        if hasattr(torch, "compile"):
            print("Compiling model with torch.compile()...")
            # Mode can be tuned; keep defaults for broad compatibility
            model = torch.compile(model)  # type: ignore[arg-type]
    except Exception as exc:
        print(
            f"torch.compile unavailable or failed ({exc}); continuing without compile."
        )

    # --- Callbacks ---
    checkpoint_callback = AnnounceCheckpoint(
        monitor="val_wer",
        mode="min",
        save_top_k=3,
        filename="epoch={epoch:02d}-wer={val_wer:.3f}",
        save_last=True,
        save_on_train_epoch_end=True,  # Save at epoch end to avoid resumption warnings
    )
    callbacks = [
        checkpoint_callback,
        ValidationErrorLogger(
            max_batches=int(cfg.validation.get("log_error_batches", 1))
        ),
    ]
    if UNFREEZING_AVAILABLE and cfg.unfreezing.get("enabled", False):
        schedule = None
        if args.profile:
            try:
                from progressive_unfreezing import (
                    create_unfreezing_schedule_for_profile,
                )

                schedule = create_unfreezing_schedule_for_profile(args.profile)
            except (ImportError, AttributeError):
                pass
        callbacks.append(
            ProgressiveUnfreezingCallback(
                unfreeze_schedule=schedule,
                warmup_epochs=cfg.unfreezing.get("warmup_epochs", 2),
            )
        )

    # --- Trainer ---
    # Find the latest checkpoint to resume from
    prefer_ckpt = _resolve_path(args.checkpoint) if args.checkpoint else None
    resume_from = find_latest_checkpoint(prefer_ckpt)

    profile_tag = args.profile if args.profile else "default"
    root_dir = f"./rnnt_checkpoints_{profile_tag}_{runtime_id}"
    trainer = pl.Trainer(
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        precision=cfg.training.precision,
        max_epochs=cfg.training.max_epochs,
        log_every_n_steps=20,
        gradient_clip_val=1.0,
        accumulate_grad_batches=cfg.training.gradient_accumulation,
        callbacks=callbacks,
        default_root_dir=root_dir,
        val_check_interval=cfg.validation.check_interval,
        limit_val_batches=cfg.validation.limit_batches,
        fast_dev_run=bool(int(os.environ.get("FAST_DEV_RUN", "0"))),
    )

    print(f"Starting trainer... Attempting to resume from: {resume_from}")
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=resume_from,
    )

    nemo_path = Path(f"{root_dir}/conformer_rnnt_final.nemo")
    model.save_to(str(nemo_path))
    print(f"Saved final NeMo checkpoint to {nemo_path}")


if __name__ == "__main__":
    main()
