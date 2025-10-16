#!/usr/bin/env python3
"""
Personalized RNN-T training script for gesture swipe models.

This variant bakes in the end-to-end preprocessing pipeline expected by the
updated web demo and on-device stacks. It ingests raw swipe traces as
(x, y, t_ms) where coordinates are in [0, 1] with (0,0) at top-left Q key,
then transforms them to [-1, 1] with (0,0) at keyboard center for model input.
The script performs adaptive resampling, feature extraction aligned with the
web runtime, and optional teacher-student distillation for smaller deployment.

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
import logging
from functools import lru_cache

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from numba import njit
from torch.utils.data import WeightedRandomSampler
from omegaconf import DictConfig
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from collections import Counter
import re

import nemo.collections.asr as nemo_asr
import torch.nn as nn

# (Helper functions clamp and load_key_centers are assumed to exist as in the original script)

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

# Model size presets for different deployment targets
MODEL_PRESETS = {
    "mobile": {  # Optimized for Android phones (< 20MB model, < 50ms inference)
        "encoder": {
            "d_model": 144,
            "n_heads": 4,
            "num_layers": 4,
            "conv_kernel_size": 15,
        },
        "decoder": {
            "pred_hidden": 192,
            "pred_rnn_layers": 1,
        },
        "joint": {
            "joint_hidden": 256,
        },
    },
    "tablet": {  # Balanced for tablets/high-end phones
        "encoder": {
            "d_model": 192,
            "n_heads": 4,
            "num_layers": 5,
            "conv_kernel_size": 21,
        },
        "decoder": {
            "pred_hidden": 256,
            "pred_rnn_layers": 2,
        },
        "joint": {
            "joint_hidden": 384,
        },
    },
    "server": {  # Maximum accuracy for cloud/research
        "encoder": {
            "d_model": 256,
            "n_heads": 8,
            "num_layers": 6,
            "conv_kernel_size": 31,
        },
        "decoder": {
            "pred_hidden": 320,
            "pred_rnn_layers": 2,
        },
        "joint": {
            "joint_hidden": 512,
        },
    },
}

# Select model size (can be overridden via CLI)
SELECTED_MODEL = "tablet"  # Default to mobile-optimized

CONFIG: Dict[str, Any] = {
    # --- Data and Vocabulary Paths ---
    # These are now default values, overrideable via command-line arguments
    # for better portability and experiment management.
    "data": {
        "train_manifest": "data/val_leon_filtered_norm.jsonl",
        "val_manifest": "data/train_leon_filtered_norm.jsonl",
        "vocab_path": None,  # Inline alphabet vocab by default; can be overridden via CLI
        "key_centers_path": None,  # Optional: Path to a JSON file defining keyboard layout for featurization.
        "max_trace_len": 512,  # Safety limit to prevent excessively long traces from consuming too much memory.
    },
    # --- Training Hyperparameters ---
    "training": {
        "batch_size": 384,  # Safer default for 16GB GPUs and RNNT
        "num_workers": 4,  # Safer default; can be overridden via CLI
        "learning_rate": 2e-4,  # A conservative learning rate for the AdamW optimizer, good for stable convergence.
        "max_epochs": 500,  # Total number of training epochs (increased for multi-day training).
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
    # Using preset configuration for selected model size
    "model": {
        "encoder": {
            "feat_in": 37,  # Input feature dimension from PersonalizedSwipeFeaturizer. MUST match feature output.
            **MODEL_PRESETS[SELECTED_MODEL]["encoder"],
            "subsampling_factor": 2,  # Reduces the sequence length early in the model, saving computation.
        },
        "decoder": MODEL_PRESETS[SELECTED_MODEL]["decoder"],
        "joint": {
            **MODEL_PRESETS[SELECTED_MODEL]["joint"],
            "activation": "relu",
            "dropout": 0.1,
        },
    },
    # --- Preprocessing ---
    "preprocess": {
        # Adaptive resampling normalizes gesture length for robustness and efficient batching.
        "resample_short_target": 64,  # Target frame count for shorter swipes. Ensures a minimum information density.
        "resample_long_target": 192,  # Target frame count for longer swipes. Compresses long gestures to a manageable length.
        "resample_short_threshold": 64,  # Gestures shorter than this are considered "short".
        "resample_long_threshold": 256,  # Gestures longer than this are considered "long".
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
        "strategy": "none",  # Keep validation deterministic by default
        "freq_power": 0.75,
        "length_power": 1.3,
        "min_word_length": 2,
        "rare_frequency_threshold": 40,
        "rare_word_boost": 6.0,
        "max_weight_factor": 28.0,
        "batch_size_factor": 0.25,  # Use a smaller batch size for validation.
        "limit_batches": 0.1,  # Validate on 10% of val set per run by default.
        "check_interval": 0.25,  # Run validation 4x per epoch (lighter each time).
        "max_samples": 500,  # Limit the number of validation samples used.
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
    """More conservative CUDA check to avoid lazy-init side effects.

    Uses device_count() and is_initialized() to decide if CUDA is already
    initialized and has visible devices, which helps skip first-touch lazy
    initialization quirks in some environments.
    """
    try:
        return bool(torch.cuda.device_count() > 0 and torch.cuda.is_initialized())
    except Exception as exc:  # pragma: no cover - defensive guard
        print(f"CUDA probe failed ({exc}); falling back to CPU")
        return False


def _supports_bf16() -> bool:
    """Checks whether the current CUDA device stack supports BF16 autocast."""
    try:
        return bool(
            torch.cuda.is_available()
            and hasattr(torch.cuda, "is_bf16_supported")
            and torch.cuda.is_bf16_supported()
        )
    except Exception:
        return False


def _resolve_path(path_str: str) -> str:
    """Resolve a path relative to CWD first, then script directory, if not absolute."""
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    # Prefer current working directory
    candidate = (Path.cwd() / path).resolve()
    if candidate.exists() or not (SCRIPT_DIR / path).exists():
        return str(candidate)
    # Fallback to script directory
    return str((SCRIPT_DIR / path).resolve())


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


@lru_cache(maxsize=1)
def load_key_centers(path: Optional[str]) -> List[Tuple[str, float, float]]:
    """Loads key centers from a JSON file or returns a default QWERTY layout."""
    if path:
        try:
            with open(_resolve_path(path), "r") as f:
                return json.load(f)
        except Exception as e:
            logging.getLogger("train_rnnt").warning(
                f"Could not load key centers from {path}. Using default. Error: {e}"
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
    feature vector that the model can learn from.

    For mobile deployment, we can reduce features to speed up inference.
    """

    # Full feature set for maximum accuracy
    FULL_FEATURE_NAMES = [
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

    # Reduced feature set for mobile (23 features)
    MOBILE_FEATURE_NAMES = [
        "x",
        "y",
        "vx",
        "vy",
        "speed",
        "ax",
        "ay",
        "angle_sin",
        "angle_cos",
        "curvature",
        "dist_key1",  # Only nearest key
        "dist_key2",
        "dist_key3",
        "progress",
        "is_start",
        "is_end",
        "win_mean_x",
        "win_mean_y",
        "win_range_x",
        "win_range_y",
    ]

    def __init__(
        self, key_centers_path: Optional[str] = None, mobile_features: bool = False
    ):
        self.key_centers = load_key_centers(key_centers_path)
        # Select feature names based on target footprint
        self.FEATURE_NAMES = (
            self.MOBILE_FEATURE_NAMES if mobile_features else self.FULL_FEATURE_NAMES
        )

    @property
    def feature_dim(self) -> int:
        return len(self.FEATURE_NAMES)

    def __call__(self, points: Iterable[Dict[str, float]]) -> np.ndarray:
        pts = list(points)
        if not pts:
            return np.zeros((1, self.feature_dim), dtype=np.float32)

        # Convert to arrays
        x = np.asarray(
            [clamp(float(p.get("x", 0.0)), -1.0, 1.0) for p in pts], dtype=np.float32
        )
        y = np.asarray(
            [clamp(float(p.get("y", 0.0)), -1.0, 1.0) for p in pts], dtype=np.float32
        )
        t_ms = np.asarray(
            [float(p.get("t", i * 10.0)) for i, p in enumerate(pts)], dtype=np.float32
        )
        t_seconds = t_ms / 1000.0

        # dt with safe epsilon
        dt = np.diff(t_seconds, prepend=t_seconds[0])
        dt[dt == 0] = 1e-6

        # Velocity
        dx = np.diff(x, prepend=x[0])
        dy = np.diff(y, prepend=y[0])
        vx = dx / dt
        vy = dy / dt
        speed = np.sqrt(vx * vx + vy * vy)

        # Acceleration
        dvx = np.diff(vx, prepend=vx[0])
        dvy = np.diff(vy, prepend=vy[0])
        ax = dvx / dt
        ay = dvy / dt
        acc = np.sqrt(ax * ax + ay * ay)

        # Angles and curvature
        angle = np.arctan2(vy, vx)
        angle_sin = np.sin(angle)
        angle_cos = np.cos(angle)
        prev_angle = np.roll(angle, 1)
        prev_angle[0] = angle[0]
        curvature = angle - prev_angle
        curvature = (curvature + np.pi) % (2 * np.pi) - np.pi

        # Distances to nearest keys (vectorized)
        if len(self.key_centers) > 0:
            kc = np.asarray(
                [[kx, ky] for _, kx, ky in self.key_centers], dtype=np.float32
            )  # (K,2)
            pts_xy = np.stack([x, y], axis=1)  # (N,2)
            diffs = pts_xy[:, None, :] - kc[None, :, :]  # (N,K,2)
            dists = np.sqrt(np.sum(diffs * diffs, axis=2))  # (N,K)
            nearest = np.sort(dists, axis=1)[:, :5]
            # Pad if fewer than 5 keys
            if nearest.shape[1] < 5:
                pad = np.full(
                    (nearest.shape[0], 5 - nearest.shape[1]), 1.0, dtype=np.float32
                )
                nearest = np.concatenate([nearest, pad], axis=1)
            dist_key1, dist_key2, dist_key3, dist_key4, dist_key5 = [
                nearest[:, i] for i in range(5)
            ]
        else:
            dist_key1 = dist_key2 = dist_key3 = dist_key4 = dist_key5 = np.ones_like(x)

        # Temporal context
        n = x.shape[0]
        if n > 1:
            progress = np.linspace(0.0, 1.0, n, dtype=np.float32)
        else:
            progress = np.zeros(1, dtype=np.float32)
        is_start = np.zeros(n, dtype=np.float32)
        is_end = np.zeros(n, dtype=np.float32)
        is_start[0] = 1.0
        is_end[-1] = 1.0

        # Windowed stats (size=5) using sliding windows
        try:
            from numpy.lib.stride_tricks import sliding_window_view

            win = 5
            pad = win // 2
            x_pad = np.pad(x, (pad, pad), mode="edge")
            y_pad = np.pad(y, (pad, pad), mode="edge")
            x_win = sliding_window_view(x_pad, win)  # (N,win)
            y_win = sliding_window_view(y_pad, win)
            win_mean_x = np.mean(x_win, axis=1)
            win_mean_y = np.mean(y_win, axis=1)
            win_std_x = np.std(x_win, axis=1)
            win_std_y = np.std(y_win, axis=1)
            win_range_x = np.max(x_win, axis=1) - np.min(x_win, axis=1)
            win_range_y = np.max(y_win, axis=1) - np.min(y_win, axis=1)
        except Exception:
            # Fallback: zeros if sliding_window_view unavailable
            win_mean_x = x
            win_mean_y = y
            win_std_x = np.zeros_like(x)
            win_std_y = np.zeros_like(y)
            win_range_x = np.zeros_like(x)
            win_range_y = np.zeros_like(y)

        # Build feature map and assemble in configured order
        fmap = {
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
            "angle_sin": angle_sin,
            "angle_cos": angle_cos,
            "curvature": curvature,
            "dist_key1": dist_key1,
            "dist_key2": dist_key2,
            "dist_key3": dist_key3,
            "dist_key4": dist_key4,
            "dist_key5": dist_key5,
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

        out_cols = []
        for name in self.FEATURE_NAMES:
            arr = fmap.get(name)
            if arr is None:
                arr = np.zeros_like(x)
            out_cols.append(arr.astype(np.float32, copy=False))
        return np.stack(out_cols, axis=1)


# ---------------------------------------------------------------------------
# Minimal Identity Preprocessor (bypass audio deps)
# ---------------------------------------------------------------------------


class IdentityPreprocessor(nn.Module):
    """No-op preprocessor to satisfy NeMo instantiation when feeding features directly."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, audio_signal, length):
        return audio_signal, length


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
        normalize_coords: bool = True,
    ) -> None:
        super().__init__()
        self.manifest_path = manifest_path
        self.vocab = vocab
        self.max_trace_len = max_trace_len
        self.preprocess_cfg = preprocess_cfg
        self.featurizer = featurizer
        self.augmenter = augmenter
        self.is_training = is_training
        self.normalize_coords = normalize_coords
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

        normalized = self._prepare_points(raw_points, self.normalize_coords)
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
    def _prepare_points(
        points: List[Dict[str, Any]], normalize_coords: bool = True
    ) -> List[Dict[str, float]]:
        """Prepares raw points by optionally transforming from [0, 1] to [-1, 1] and making time relative."""
        if not points:
            return []
        # Guard against rare non-monotonic timestamps by sorting by 't' when present
        try:
            if any(
                (
                    float(points[i + 1].get("t", (i + 1) * 10.0))
                    < float(points[i].get("t", i * 10.0))
                )
                for i in range(len(points) - 1)
            ):
                points = sorted(points, key=lambda p: float(p.get("t", 0.0)))
        except Exception:
            pass

        start_t = float(points[0].get("t", 0.0))
        prepared: List[Dict[str, float]] = []
        for idx, pt in enumerate(points):
            raw_x = float(pt.get("x", 0.0))
            raw_y = float(pt.get("y", 0.0))

            if normalize_coords:
                # Transform coordinates from [0, 1] to [-1, 1]
                # Dataset has (0,0) at top-left Q key, we need (0,0) at keyboard center
                centered_x = raw_x * 2.0 - 1.0
                centered_y = raw_y * 2.0 - 1.0

                # Allow for out-of-bounds gestures (finger went off keyboard)
                # but cap at reasonable limits to prevent extreme values
                centered_x = clamp(centered_x, -1.5, 1.5)
                centered_y = clamp(centered_y, -1.5, 1.5)
            else:
                # Assume data is already in [-1, 1] coordinate system
                # Still apply clamping for safety
                centered_x = clamp(raw_x, -1.5, 1.5)
                centered_y = clamp(raw_y, -1.5, 1.5)

            raw_t = float(pt.get("t", idx * 10.0))
            prepared.append(
                {
                    "x": centered_x,
                    "y": centered_y,
                    "t": max(0.0, raw_t - start_t),
                }
            )
        return prepared

    def state_dict(self) -> Dict[str, Any]:
        """Return state dict for dataloader resumption."""
        return {
            "manifest_path": self.manifest_path,
            "sample_count": len(self.samples),
            "epoch": getattr(self, "_epoch", 0),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load state dict for dataloader resumption."""
        if state.get("manifest_path") != self.manifest_path:
            print(f"Warning: Manifest path mismatch during resume")
        self._epoch = state.get("epoch", 0)


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
        use_autocast_bf16: bool = False,
    ):
        super().__init__(cfg=cfg)
        self.kd_lambda = kd_lambda
        self.kd_temperature = kd_temperature
        self.teacher = None
        self.use_autocast_bf16 = bool(use_autocast_bf16 and _supports_bf16())
        if teacher_checkpoint:
            p = Path(teacher_checkpoint)
            if p.exists():
                self._init_teacher(teacher_checkpoint)
            else:
                logging.getLogger("train_rnnt").warning(
                    f"Teacher checkpoint not found at {teacher_checkpoint}, skipping distillation"
                )

        # Disable Nemo's random sample logging; we'll log mismatches manually
        if hasattr(self, "wer") and self.wer is not None:
            try:
                self.wer.log_prediction = False
            except AttributeError:
                pass

    def _init_teacher(self, checkpoint: str) -> None:
        """Initializes a frozen teacher model for knowledge distillation."""
        logging.getLogger("train_rnnt").info(
            f"Loading teacher model from: {checkpoint}"
        )
        self.teacher = nemo_asr.models.EncDecRNNTModel.restore_from(
            checkpoint, map_location="cpu"
        )
        self.teacher.freeze()
        self.teacher.eval()

    # Handle state dicts saved from compiled modules (e.g., torch.compile) where
    # parameters are nested under "_orig_mod". Lightning calls this prior to
    # applying the state dict, letting us normalize keys safely.
    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:  # type: ignore[override]
        try:
            sd = checkpoint.get("state_dict", {})
            if not isinstance(sd, dict) or not sd:
                return
            needs_fix = any(
                "._orig_mod." in k or k.startswith("encoder._orig_mod")
                for k in sd.keys()
            )
            if not needs_fix:
                return
            new_sd = {}
            for k, v in sd.items():
                nk = k.replace("encoder._orig_mod.", "encoder.")
                nk = nk.replace("decoder._orig_mod.", "decoder.")
                nk = nk.replace("joint._orig_mod.", "joint.")
                nk = nk.replace("._orig_mod.", ".")
                new_sd[nk] = v
            checkpoint["state_dict"] = new_sd
            print("Normalized checkpoint keys by stripping _orig_mod wrappers")
        except Exception as e:
            print(f"Warning: could not normalize checkpoint keys: {e}")

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
        # Autocast forward path for speed on BF16-capable GPUs when enabled
        amp_ctx = torch.cuda.amp.autocast(
            enabled=self.use_autocast_bf16, dtype=torch.bfloat16
        )
        with amp_ctx:
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
                # Teacher forward can also benefit from autocast
                t_amp_ctx = torch.cuda.amp.autocast(
                    enabled=self.use_autocast_bf16, dtype=torch.bfloat16
                )
                with t_amp_ctx:
                    teacher_joint = self._compute_joint(
                        self.teacher, batch, detach=True
                    )

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
        # Validation forward can be autocast BF16 when enabled
        with torch.cuda.amp.autocast(
            enabled=self.use_autocast_bf16, dtype=torch.bfloat16
        ):
            return super().validation_step(*args, **kwargs)

    def test_step(self, *args, **kwargs):
        with torch.cuda.amp.autocast(
            enabled=self.use_autocast_bf16, dtype=torch.bfloat16
        ):
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


class PeriodicNeMoSaver(pl.Callback):
    """Save NeMo checkpoint every N epochs for robustness against system restarts."""

    def __init__(self, save_interval: int = 50, save_dir: str = None):
        self.save_interval = save_interval
        self.save_dir = save_dir or "./rnnt_checkpoints"
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if (epoch + 1) % self.save_interval == 0:
            nemo_path = Path(self.save_dir) / f"model_epoch{epoch + 1}.nemo"
            try:
                pl_module.save_to(str(nemo_path))
                print(
                    f"\033[1;32m✓ Saved NeMo checkpoint at epoch {epoch + 1}:\033[0m {nemo_path}"
                )
            except Exception as e:
                print(f"Warning: Could not save NeMo at epoch {epoch + 1}: {e}")


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


def load_vocab(vocab_path: Optional[str]) -> Dict[str, int]:
    """Loads vocab from file if provided, otherwise uses inline alphabet vocab."""
    if vocab_path:
        try:
            vocab: Dict[str, int] = {}
            with open(vocab_path, "r", encoding="utf-8") as fh:
                for idx, line in enumerate(fh):
                    token = line.strip()
                    if token:
                        vocab[token] = idx
            if "<unk>" not in vocab:
                vocab["<unk>"] = len(vocab)
            return vocab
        except FileNotFoundError:
            print(
                f"Warning: Vocab file not found at {vocab_path}. Falling back to inline alphabet."
            )
    # Inline alphabet vocab
    letters = list("abcdefghijklmnopqrstuvwxyz")
    vocab = {ch: i for i, ch in enumerate(letters)}
    vocab["<unk>"] = len(vocab)
    return vocab


def build_dataloaders(
    cfg: DictConfig,
    vocab: Dict[str, int],
    subset_train: int = 0,
    subset_val: int = 0,
    normalize_coords: bool = True,
):
    """Builds and configures the training and validation DataLoaders."""
    featurizer = PersonalizedSwipeFeaturizer(
        cfg.data.get("key_centers_path"),
        mobile_features=bool(cfg.model.get("mobile_features", False)),
    )

    augmenter = None
    if AUGMENTATION_AVAILABLE and cfg.augmentation.get("enabled", False):
        augmenter = create_augmentation_pipeline(cfg.augmentation)
        train_word_counts = Counter()
        with open(cfg.data.train_manifest, "r") as f:
            for line in f:
                train_word_counts[json.loads(line).get("word", "")] += 1
        augmenter.set_word_frequencies(dict(train_word_counts))
        logging.getLogger("train_rnnt").info(
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
        normalize_coords=normalize_coords,
    )
    val_ds = PersonalizedSwipeDataset(
        manifest_path=cfg.data.val_manifest,
        vocab=vocab,
        max_trace_len=cfg.data.max_trace_len,
        preprocess_cfg=cfg.preprocess,
        featurizer=featurizer,
        augmenter=None,
        is_training=False,
        normalize_coords=normalize_coords,
    )

    # Compute sampling weights only if training dataset is base dataset (not Subset)
    base_train_ds = train_ds.dataset if isinstance(train_ds, Subset) else train_ds
    train_weights = base_train_ds.compute_sampling_weights(cfg.sampling)
    train_sampler = None
    if train_weights is not None:
        train_sampler = WeightedRandomSampler(
            weights=torch.from_numpy(train_weights),
            num_samples=len(train_weights),
            replacement=True,
        )
        logging.getLogger("train_rnnt").info(
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
        prefetch_factor=(4 if cfg.training.num_workers > 0 else None),
    )

    val_sampler = None
    base_val_ds = val_ds.dataset if isinstance(val_ds, Subset) else val_ds
    val_weights = base_val_ds.compute_sampling_weights(cfg.validation)
    if val_weights is not None:
        num_samples = min(
            len(val_weights), int(cfg.validation.get("max_samples", len(val_weights)))
        )
        val_sampler = WeightedRandomSampler(
            weights=torch.from_numpy(val_weights),
            num_samples=num_samples,
            replacement=False,
        )

    # Use more workers for validation to avoid bottleneck
    # Use same number of workers as training or at least 4
    val_workers = max(4, cfg.training.num_workers)
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
    return train_loader, val_loader, featurizer.feature_dim


def build_model_config(
    cfg: DictConfig, labels: List[str], feature_dim: int
) -> DictConfig:
    """Builds the full NeMo model configuration from the script's config.

    Uses a dynamic input feature dimension from the featurizer to avoid
    brittle coupling with hardcoded values.
    """
    return DictConfig(
        {
            "labels": labels,
            "sample_rate": 16000,
            "model_defaults": {
                "enc_hidden": cfg.model.encoder.d_model,
                "pred_hidden": cfg.model.decoder.pred_hidden,
            },
            # Preprocessor config is present but effectively unused since we feed features directly.
            # Keeping this avoids NeMo instantiation errors.
            "preprocessor": {
                "_target_": "nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor",
                "sample_rate": 16000,
                "normalize": "per_feature",
                "window_size": 0.02,
                "window_stride": 0.01,
                "window": "hann",
                "features": int(feature_dim),
                "n_fft": 512,
                "frame_splicing": 1,
                "dither": 0.0,
                "stft_conv": False,
            },
            "encoder": {
                "_target_": "nemo.collections.asr.modules.SqueezeformerEncoder",
                "feat_in": int(feature_dim),
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
                # RNNT joint must include blank class in num_classes
                "num_classes": len(labels) + 1,
                "vocabulary": labels,
                "log_softmax": True,
            },
            "decoding": {
                "strategy": "greedy_batch",
                "use_cuda_graphs": False,
                "greedy": {"max_symbols": 13, "use_cuda_graph_decoder": False},
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
    """Finds the best .ckpt checkpoint to resume from under the run base."""
    if (
        prefer_checkpoint
        and Path(prefer_checkpoint).exists()
        and prefer_checkpoint.endswith(".ckpt")
    ):
        print(f"Using specified checkpoint: {prefer_checkpoint}")
        return prefer_checkpoint

    # Search within the standard runs base unless overridden
    run_base = Path(os.environ.get("CKS_RUN_BASE", "./runs")).resolve()
    base_dirs = {run_base}
    patterns = [
        "rnnt_checkpoints_*/**/*.ckpt",
        "**/checkpoints/*.ckpt",
    ]
    candidates = {
        p for base in base_dirs for pattern in patterns for p in base.glob(pattern)
    }
    if not candidates:
        return None

    date_pattern = re.compile(r"(\d{8}_\d{6})")
    epoch_pattern = re.compile(r"epoch=(\d+)")

    best_path, best_key = None, ("0", -1)
    for path in candidates:
        date_match = date_pattern.search(str(path))
        date_str = date_match.group(1) if date_match else "0"
        epoch_match = epoch_pattern.search(path.name)
        epoch = int(epoch_match.group(1)) if epoch_match else -1
        if date_str > best_key[0] or (date_str == best_key[0] and epoch > best_key[1]):
            best_key = (date_str, epoch)
            best_path = path

    if best_path:
        print(f"Resuming from best checkpoint: {best_path}")
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


def try_compile_model(model, resume_from: Optional[str]) -> None:
    """Attempt to compile model with torch.compile for speed if safe to do so."""
    import torch

    if resume_from or os.environ.get("DISABLE_COMPILE", "0") != "0":
        if resume_from:
            print(
                "Skipping torch.compile when resuming from checkpoint (avoids state dict issues)"
            )
        elif os.environ.get("DISABLE_COMPILE", "0") != "0":
            print("Skipping torch.compile (disabled via DISABLE_COMPILE env var)")
        return
    try:
        if hasattr(torch, "compile") and hasattr(torch, "_dynamo"):
            import torch._dynamo

            # More conservative settings to avoid memory corruption
            torch._dynamo.config.suppress_errors = True
            torch._dynamo.config.cache_size_limit = 64  # Reduced from 256
            torch._dynamo.config.capture_scalar_outputs = False  # Changed from True
            torch._dynamo.config.guard_nn_modules = True  # Changed from False

            # Skip compilation if CUDA graphs are enabled (potential conflict)
            if hasattr(model, "wer") and hasattr(model.wer, "use_cuda_graph_decoder"):
                print("Skipping torch.compile due to CUDA graphs compatibility")
                return

            compile_success = False
            # Only try encoder compilation with safer settings
            try:
                print("Attempting to compile encoder with safe settings...")
                model.encoder = torch.compile(
                    model.encoder,
                    mode="default",  # Changed from reduce-overhead
                    backend="inductor",
                    fullgraph=False,
                    dynamic=False,  # Changed from True
                )
                compile_success = True
                print("✓ Successfully compiled encoder")
            except Exception as e:
                print(f"Could not compile encoder: {e}")
                print("Continuing without torch.compile optimizations")
                return

            # Skip joint and full model compilation to avoid memory issues
            print("Skipping joint/full model compilation to avoid memory corruption")

    except ImportError:
        print("torch.compile not available (requires PyTorch 2.0+)")


def build_callbacks(
    cfg: DictConfig, args: argparse.Namespace, root_dir: str
) -> List[pl.Callback]:
    """Builds training callbacks for logging, checkpointing, and periodic saves."""
    checkpoint_callback = AnnounceCheckpoint(
        monitor="val_wer",
        mode="min",
        save_top_k=3,
        filename="epoch={epoch:02d}-wer={val_wer:.3f}",
        save_last=True,
        save_on_train_epoch_end=True,
    )

    class SamplePredictionsLogger(pl.Callback):
        def __init__(
            self, train_manifest: str, val_limit: int = 2, train_sample: int = 15
        ):
            self.train_manifest = train_manifest
            self.val_limit = val_limit
            self.train_sample = train_sample
            self._val_batches_logged = 0

        def on_validation_epoch_start(self, trainer, pl_module):
            self._val_batches_logged = 0

        def on_validation_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
        ):
            if self._val_batches_logged >= self.val_limit:
                return
            try:
                signal, signal_len, transcript, transcript_len = batch
                with torch.no_grad():
                    encoded, encoded_len = pl_module.forward(
                        input_signal=signal, input_signal_length=signal_len
                    )
                    hyps = pl_module.decoding.rnnt_decoder_predictions_tensor(
                        encoded, encoded_len
                    )
                for ref_tokens, ref_len, hyp in zip(transcript, transcript_len, hyps):
                    ids = (
                        ref_tokens[: int(ref_len.item())]
                        .detach()
                        .cpu()
                        .numpy()
                        .tolist()
                    )
                    ref_text = pl_module.decoding.decode_tokens_to_str(ids)
                    if isinstance(hyp, list) and hyp:
                        hyp = hyp[0]
                    hyp_text = hyp.text if hasattr(hyp, "text") else str(hyp)
                    mark = (
                        "\033[1;32m✓\033[0m"
                        if hyp_text == ref_text
                        else "\033[1;31m✗\033[0m"
                    )
                    print(f"  {mark} ref='{ref_text}' pred='{hyp_text}'")
                self._val_batches_logged += 1
            except Exception as e:
                print(f"Could not log val predictions: {e}")

        def on_validation_epoch_end(self, trainer, pl_module):
            try:
                words = []
                with open(self.train_manifest, "r", encoding="utf-8") as fh:
                    for i, line in enumerate(fh):
                        if i >= self.train_sample:
                            break
                        try:
                            w = json.loads(line).get("word", "")
                            if w:
                                words.append(w)
                        except Exception:
                            pass
                if words:
                    print("\033[1;36mtrain sample words:\033[0m ", ", ".join(words))
            except Exception as e:
                print(f"Could not read training manifest for samples: {e}")

    callbacks: List[pl.Callback] = [
        checkpoint_callback,
        ValidationErrorLogger(
            max_batches=int(cfg.validation.get("log_error_batches", 1))
        ),
        PeriodicNeMoSaver(save_interval=50, save_dir=root_dir),
        SamplePredictionsLogger(
            train_manifest=cfg.data.train_manifest, val_limit=2, train_sample=15
        ),
    ]

    class ValWERBanner(pl.Callback):
        def __init__(self):
            self.best = None

        def on_validation_epoch_end(self, trainer, pl_module):
            try:
                metrics = trainer.callback_metrics
                if "val_wer" in metrics:
                    cur = float(metrics["val_wer"])
                    if self.best is None or cur < self.best:
                        self.best = cur
                        print(f"\033[1;35m★★ BEST val_wer updated: {cur:.4f} ★★\033[0m")
            except Exception:
                pass

    callbacks.append(ValWERBanner())
    if UNFREEZING_AVAILABLE and cfg.unfreezing.get("enabled", False):
        schedule = None
        if args.profile:
            try:
                from progressive_unfreezing import (
                    create_unfreezing_schedule_for_profile,
                )

                schedule = create_unfreezing_schedule_for_profile(args.profile)
            except (ImportError, AttributeError):
                schedule = None
        callbacks.append(
            ProgressiveUnfreezingCallback(
                unfreeze_schedule=schedule,
                warmup_epochs=cfg.unfreezing.get("warmup_epochs", 2),
            )
        )
    return callbacks


def main() -> None:
    # Ensure torch is available in function scope
    import torch

    cfg = DictConfig(CONFIG)

    # --- CLI Arguments ---
    parser = argparse.ArgumentParser(
        description="Train Personalized RNNT for CleverKeys"
    )
    # Model size selection
    parser.add_argument(
        "--model-size",
        type=str,
        choices=["mobile", "tablet", "server"],
        default="tablet",
        help="Model size preset: mobile (fast, <20MB), tablet (balanced), server (accurate)",
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
        "--train-manifest",
        type=str,
        dest="train_manifest",
        default=None,
        help="Path to training JSONL manifest",
    )
    parser.add_argument(
        "--val-manifest",
        type=str,
        dest="val_manifest",
        default=None,
        help="Path to validation JSONL manifest",
    )
    parser.add_argument(
        "--vocab-path",
        type=str,
        dest="vocab_path",
        default=None,
        help="Path to vocab.txt",
    )
    # Fast test and subsetting
    parser.add_argument(
        "--subset-train", type=int, default=0, help="Use only first N training samples"
    )
    parser.add_argument(
        "--subset-val", type=int, default=0, help="Use only first N validation samples"
    )
    parser.add_argument(
        "--fast-test",
        action="store_true",
        help="Use small subset (2000/400) and 1 epoch for quick verification",
    )
    parser.add_argument(
        "--dry-run-first-batch",
        action="store_true",
        help="Build one train batch and run forward+loss, then exit",
    )
    parser.add_argument(
        "--val-limit-batches",
        type=float,
        default=None,
        help="Fraction [0,1] of validation set per run (overrides config)",
    )
    parser.add_argument(
        "--val-check-interval",
        type=float,
        default=None,
        help="Validation check interval as fraction of an epoch (overrides config)",
    )
    parser.add_argument(
        "--autocast-bf16",
        action="store_true",
        help="Wrap forward passes with torch.cuda.amp.autocast(dtype=bfloat16) when supported",
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
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Override maximum training epochs",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize coordinates from [0,1] to [-1,1]. Pass false if data is already in [-1,1]",
    )
    args = parser.parse_args()

    # --- Seeding & matmul precision ---
    seed = int(os.environ.get("PY_SEED", "1337"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Prefer file_system sharing to avoid BrokenPipe in multiprocessing resource sharer
    try:
        import torch.multiprocessing as mp

        mp.set_sharing_strategy("file_system")
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    # --- Apply model size preset ---
    if args.model_size != SELECTED_MODEL:
        # Update model configuration with selected preset
        cfg.model.encoder.update(MODEL_PRESETS[args.model_size]["encoder"])
        cfg.model.decoder = MODEL_PRESETS[args.model_size]["decoder"]
        cfg.model.joint.update(MODEL_PRESETS[args.model_size]["joint"])
        print(f"Using {args.model_size} model preset")
    cfg.model["mobile_features"] = bool(args.model_size == "mobile")

    # --- Resolve Paths and Set up Environment ---
    cfg.data.train_manifest = _resolve_path(cfg.data.train_manifest)
    cfg.data.val_manifest = _resolve_path(cfg.data.val_manifest)
    if cfg.data.vocab_path:
        cfg.data.vocab_path = _resolve_path(cfg.data.vocab_path)
    if args.train_manifest:
        cfg.data.train_manifest = _resolve_path(args.train_manifest)
    if args.val_manifest:
        cfg.data.val_manifest = _resolve_path(args.val_manifest)
    if args.vocab_path:
        cfg.data.vocab_path = _resolve_path(args.vocab_path)

    # Configure CUDA backends if available; avoid forcing CPU based on conservative probe
    if torch.cuda.is_available():
        try:
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

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
    if args.max_epochs is not None and args.max_epochs > 0:
        cfg.training.max_epochs = int(args.max_epochs)
        print(f"Overriding max_epochs to {args.max_epochs}")
    if args.val_limit_batches is not None:
        try:
            cfg.validation.limit_batches = float(args.val_limit_batches)
        except Exception:
            pass
    if args.val_check_interval is not None:
        try:
            cfg.validation.check_interval = float(args.val_check_interval)
        except Exception:
            pass

    # Load sampling profiles for training/validation when provided
    if args.profile:
        prof = load_sampling_profile(args.profile)
        if prof:
            cfg.sampling.update(prof)
            logging.getLogger("train_rnnt").info(
                f"Applied training sampling profile: {args.profile}"
            )
    if args.val_profile:
        vprof = load_sampling_profile(args.val_profile)
        if vprof:
            cfg.validation.update(vprof)
            logging.getLogger("train_rnnt").info(
                f"Applied validation sampling profile: {args.val_profile}"
            )

    # --- Fast test & subsetting overrides ---
    subset_train = int(args.subset_train or (2000 if args.fast_test else 0))
    subset_val = int(args.subset_val or (400 if args.fast_test else 0))
    if args.fast_test:
        cfg.training.max_epochs = 1
        if cfg.training.batch_size > 64:
            cfg.training.batch_size = 64
        cfg.training.num_workers = (
            0 if not _has_usable_cuda() else cfg.training.num_workers
        )
        # Lighter validation for quick passes
        try:
            cfg.validation.limit_batches = 0.1
        except Exception:
            pass

    # --- Build Model and DataLoaders ---
    vocab = load_vocab(cfg.data.vocab_path)
    train_loader, val_loader, feature_dim = build_dataloaders(
        cfg, vocab, subset_train=subset_train, subset_val=subset_val
    )
    nemo_cfg = build_model_config(cfg, list(vocab.keys()), feature_dim)
    # Avoid Numba JIT/caching issues in librosa on restricted environments
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    cache_dir = Path(
        os.environ.get("NUMBA_CACHE_DIR", str(Path("./.numba_cache").resolve()))
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = str(cache_dir)

    # --- Scheduler max_steps computation ---
    try:
        steps_per_epoch = len(train_loader)
        accum = int(cfg.training.get("gradient_accumulation", 1)) or 1
        optim_steps_per_epoch = (steps_per_epoch + accum - 1) // accum
        # In FAST_DEV_RUN, cap to 1 to match Lightning behavior
        if bool(int(os.environ.get("FAST_DEV_RUN", "0"))):
            max_steps = 1
        else:
            max_steps = max(optim_steps_per_epoch * int(cfg.training.max_epochs), 1)
        if "optim" in nemo_cfg and "sched" in nemo_cfg.optim:
            nemo_cfg.optim.sched["max_steps"] = int(max_steps)
    except Exception:
        # Fall back silently if any attribute missing
        pass

    # If manual autocast is requested, disable Lightning's mixed precision to avoid double contexts
    use_manual_autocast = bool(args.autocast_bf16 and _supports_bf16())
    if use_manual_autocast:
        cfg.training.precision = "32-true"
        print(
            "Enabling manual BF16 autocast for forward passes; Trainer precision set to 32-true"
        )

    model = PersonalizedRNNTModel(
        cfg=nemo_cfg,
        kd_lambda=cfg.training.kd_lambda,
        kd_temperature=cfg.training.kd_temperature,
        teacher_checkpoint=_resolve_path(cfg.training.teacher_checkpoint)
        if cfg.training.teacher_checkpoint
        else None,
        use_autocast_bf16=use_manual_autocast,
    )

    # Note: torch.compile will be attempted later if not resuming from checkpoint
    # Compilation must happen after checkpoint loading to avoid state dict mismatch

    # Prepare run directories and callbacks
    profile_tag = args.profile if args.profile else "default"
    run_base = os.environ.get("CKS_RUN_BASE", "./runs")
    root_dir = f"{run_base}/rnnt_checkpoints_{profile_tag}_{runtime_id}"

    # --- Callbacks ---
    callbacks = build_callbacks(cfg, args, root_dir)

    # Add a lightweight validation stats logger to surface metrics more often
    class QuickValStats(pl.Callback):
        def on_validation_epoch_start(self, trainer, pl_module):
            self._sum_T = 0
            self._sum_tokens = 0
            self._batches = 0

        def on_validation_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
        ):
            try:
                signal, signal_len, transcript, transcript_len = batch
                self._sum_T += int(signal_len.sum().item())
                self._sum_tokens += int(transcript_len.sum().item())
                self._batches += 1
            except Exception:
                pass

        def on_validation_epoch_end(self, trainer, pl_module):
            try:
                metrics = trainer.callback_metrics
                val_wer = metrics.get("val_wer", None)
                val_loss = metrics.get("val_loss", None)
                avg_T = (self._sum_T / max(self._batches, 1)) if self._batches else None
                avg_tokens = (
                    (self._sum_tokens / max(self._batches, 1))
                    if self._batches
                    else None
                )
                msg = "[val]"
                if val_wer is not None:
                    msg += f" wer={float(val_wer):.4f}"
                if val_loss is not None:
                    msg += f" loss={float(val_loss):.4f}"
                if avg_T is not None:
                    msg += f" avg_T={avg_T:.1f}"
                if avg_tokens is not None:
                    msg += f" avg_tokens={avg_tokens:.1f}"
                print(msg)
            except Exception:
                pass

    callbacks.append(QuickValStats())

    # --- Dry-run first batch (optional) ---
    if args.dry_run_first_batch:
        print("Performing dry-run on first training batch...")
        try:
            batch = next(iter(train_loader))
            signal, signal_len, transcript, transcript_len = batch
            with torch.no_grad():
                encoded, encoded_len = model.forward(
                    input_signal=signal, input_signal_length=signal_len
                )
                decoder, target_length, _ = model.decoder(
                    targets=transcript, target_length=transcript_len
                )
                joint = model.joint(encoder_outputs=encoded, decoder_outputs=decoder)
                loss_value = model.loss(
                    log_probs=joint,
                    targets=transcript,
                    input_lengths=encoded_len,
                    target_lengths=target_length,
                )
            print(
                f"[DRY] Batch shapes: signal={tuple(signal.shape)} tokens={tuple(transcript.shape)}; RNNT loss={float(loss_value)}"
            )
        except Exception as e:
            print(f"Dry-run failed: {e}")
        return

    # --- Trainer ---
    # Checkpoint selection with explicit preference for user-specified path
    ckpt_path = None
    if args.checkpoint:
        maybe = _resolve_path(args.checkpoint)
        if Path(maybe).exists():
            ckpt_path = maybe
            print(f"Using specified checkpoint: {ckpt_path}")
        else:
            print(
                f"Warning: specified checkpoint not found: {maybe}. Falling back to latest."
            )
    if ckpt_path is None:
        ckpt_path = find_latest_checkpoint(None)

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

    # Optional torch.compile
    try_compile_model(model, ckpt_path)

    print(f"Starting trainer... Attempting to resume from: {ckpt_path}")
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=ckpt_path,
    )

    nemo_path = Path(f"{root_dir}/conformer_rnnt_final.nemo")
    model.save_to(str(nemo_path))
    print(f"Saved final NeMo checkpoint to {nemo_path}")


if __name__ == "__main__":
    # Basic logging setup for CLI runs
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    main()
