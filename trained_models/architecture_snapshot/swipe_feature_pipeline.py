"""Shared swipe feature extraction utilities for training and deployment.

This module consolidates keyboard geometry, character tokenization, and
trajectory featurization so that the exact same logic can be reused by
training code, evaluation scripts, and client runtime exporters.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

DEFAULT_KEYBOARD_WIDTH = 360.0
DEFAULT_KEYBOARD_HEIGHT = 215.0
DEFAULT_KEYBOARD_CENTER = (DEFAULT_KEYBOARD_WIDTH / 2, DEFAULT_KEYBOARD_HEIGHT / 2)

# Approximate QWERTY layout derived from the web demo fallback.
_DEFAULT_KEY_POSITIONS: Dict[str, Tuple[float, float]] = {
    # Row 1
    "q": (18.0, 111.0),
    "w": (54.0, 111.0),
    "e": (90.0, 111.0),
    "r": (126.0, 111.0),
    "t": (162.0, 111.0),
    "y": (198.0, 111.0),
    "u": (234.0, 111.0),
    "i": (270.0, 111.0),
    "o": (306.0, 111.0),
    "p": (342.0, 111.0),
    # Row 2
    "a": (36.0, 167.0),
    "s": (72.0, 167.0),
    "d": (108.0, 167.0),
    "f": (144.0, 167.0),
    "g": (180.0, 167.0),
    "h": (216.0, 167.0),
    "j": (252.0, 167.0),
    "k": (288.0, 167.0),
    "l": (324.0, 167.0),
    # Row 3
    "z": (72.0, 223.0),
    "x": (108.0, 223.0),
    "c": (144.0, 223.0),
    "v": (180.0, 223.0),
    "b": (216.0, 223.0),
    "n": (252.0, 223.0),
    "m": (288.0, 223.0),
}

_SPECIAL_TOKEN_POSITIONS: Dict[str, Tuple[float, float]] = {
    "<unk>": DEFAULT_KEYBOARD_CENTER,
    "<pad>": (0.0, 0.0),
}


class KeyboardGeometry:
    """Utility for mapping screen coordinates to keyboard keys."""

    def __init__(
        self,
        grid_path: Optional[str] = "data/data_preprocessed/gridname_to_grid.json",
        layout_name: str = "qwerty_english",
        fallback_positions: Optional[Dict[str, Tuple[float, float]]] = None,
        fallback_dims: Tuple[float, float] = (DEFAULT_KEYBOARD_WIDTH, DEFAULT_KEYBOARD_HEIGHT),
    ) -> None:
        self.layout_name = layout_name
        positions: Dict[str, Tuple[float, float]] = {}
        width, height = fallback_dims

        grid_file = Path(grid_path) if grid_path else None
        if grid_file and grid_file.exists():
            try:
                with grid_file.open("r", encoding="utf-8") as fp:
                    grid_data = json.load(fp)
                layout = grid_data.get(layout_name, {})
                width = float(layout.get("width", width))
                height = float(layout.get("height", height))
                for key_info in layout.get("keys", []):
                    label = key_info.get("label")
                    hitbox = key_info.get("hitbox", {})
                    if not label or not hitbox:
                        continue
                    cx = float(hitbox.get("x", 0.0)) + float(hitbox.get("w", 0.0)) / 2.0
                    cy = float(hitbox.get("y", 0.0)) + float(hitbox.get("h", 0.0)) / 2.0
                    positions[label.lower()] = (cx, cy)
            except Exception as exc:  # pragma: no cover - defensive fallback
                print(f"[KeyboardGeometry] Failed to load {grid_file}: {exc}. Using fallback layout.")
                positions = {}

        if not positions:
            fallback_positions = fallback_positions or {}
            positions = {**_DEFAULT_KEY_POSITIONS, **fallback_positions}

        positions.update(_SPECIAL_TOKEN_POSITIONS)

        self.width = float(width)
        self.height = float(height)
        self.key_positions = positions
        self._labels = list(positions.keys())

    # ------------------------------------------------------------------
    def nearest_key(self, x: float, y: float) -> str:
        """Return the nearest key label for the given screen coordinates."""
        min_dist = float("inf")
        nearest = "<unk>"
        for label, (kx, ky) in self.key_positions.items():
            if label in {"<unk>", "<pad>"}:
                continue
            dx = x - kx
            dy = y - ky
            dist = dx * dx + dy * dy
            if dist < min_dist:
                min_dist = dist
                nearest = label
        return nearest

    # ------------------------------------------------------------------
    def nearest_key_indices(
        self, x: Sequence[float], y: Sequence[float], tokenizer: "CharTokenizer"
    ) -> np.ndarray:
        """Vectorised nearest-key lookup returning tokenizer indices."""
        indices = np.zeros(len(x), dtype=np.int64)
        for i, (xi, yi) in enumerate(zip(x, y)):
            key = self.nearest_key(xi, yi)
            indices[i] = tokenizer.char_to_idx.get(key, tokenizer.unk_idx)
        return indices


class CharTokenizer:
    """Character-level tokenizer shared between training and inference."""

    def __init__(self) -> None:
        chars = list("abcdefghijklmnopqrstuvwxyz")
        specials = ["<pad>", "<unk>", "<sos>", "<eos>"]
        self.vocab = specials + chars
        self.char_to_idx = {c: i for i, c in enumerate(self.vocab)}
        self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}

        self.pad_idx = self.char_to_idx["<pad>"]
        self.unk_idx = self.char_to_idx["<unk>"]
        self.sos_idx = self.char_to_idx["<sos>"]
        self.eos_idx = self.char_to_idx["<eos>"]
        self.vocab_size = len(self.vocab)

    def encode(self, word: str, add_special_tokens: bool = True) -> List[int]:
        tokens: List[int] = []
        if add_special_tokens:
            tokens.append(self.sos_idx)
        for char in word.lower():
            tokens.append(self.char_to_idx.get(char, self.unk_idx))
        if add_special_tokens:
            tokens.append(self.eos_idx)
        return tokens

    def decode(self, indices: Iterable[int]) -> str:
        chars: List[str] = []
        for idx in indices:
            if idx in (self.sos_idx, self.eos_idx, self.pad_idx):
                continue
            chars.append(self.idx_to_char.get(idx, "?"))
        return "".join(chars)


@dataclass
class TrajectoryFeatures:
    features: np.ndarray  # shape [seq_len, 6]
    nearest_keys: np.ndarray  # shape [seq_len]
    seq_len: int


class SwipeFeatureExtractor:
    """Converts raw swipe trajectories to model-ready tensors."""

    def __init__(
        self,
        keyboard: Optional[KeyboardGeometry] = None,
        tokenizer: Optional[CharTokenizer] = None,
        max_seq_len: int = 150,
        velocity_clip: float = 10.0,
        acceleration_clip: float = 10.0,
        dt_floor_ms: float = 1.0,
        default_dt_ms: float = 16.0,
        blend_uniform_dt_prob: float = 0.15,
        velocity_scale: float = 1.0,
        acceleration_scale: float = 1.0,
    ) -> None:
        self.keyboard = keyboard or KeyboardGeometry()
        self.tokenizer = tokenizer or CharTokenizer()
        self.max_seq_len = max_seq_len
        self.velocity_clip = velocity_clip
        self.acceleration_clip = acceleration_clip
        self.dt_floor_ms = dt_floor_ms
        self.default_dt_ms = default_dt_ms
        self.blend_uniform_dt_prob = blend_uniform_dt_prob
        self.velocity_scale = velocity_scale
        self.acceleration_scale = acceleration_scale

    # ------------------------------------------------------------------
    def _compute_dt(self, ts: Optional[Sequence[float]], length: int) -> np.ndarray:
        if ts is None or len(ts) == 0:
            return np.full(length, self.default_dt_ms, dtype=np.float32)

        ts_arr = np.asarray(ts, dtype=np.float32)
        if ts_arr.shape[0] != length:
            ts_arr = np.interp(
                np.linspace(0, len(ts_arr) - 1, num=length, endpoint=True),
                np.arange(len(ts_arr)),
                ts_arr,
            ).astype(np.float32)

        dt = np.diff(ts_arr, prepend=ts_arr[0])
        dt = np.maximum(dt, self.dt_floor_ms).astype(np.float32)

        if self.blend_uniform_dt_prob > 0.0 and np.random.rand() < self.blend_uniform_dt_prob:
            uniform_dt = np.full_like(dt, self.default_dt_ms)
            mix = np.random.rand()
            dt = mix * dt + (1.0 - mix) * uniform_dt

        return dt

    # ------------------------------------------------------------------
    def extract(
        self,
        xs: Sequence[float],
        ys: Sequence[float],
        ts: Optional[Sequence[float]] = None,
        truncate: bool = True,
    ) -> TrajectoryFeatures:
        if len(xs) == 0:
            raise ValueError("Swipe path is empty")

        xs_arr = np.asarray(xs, dtype=np.float32)
        ys_arr = np.asarray(ys, dtype=np.float32)
        length = len(xs_arr)
        if len(ys_arr) != length:
            raise ValueError("x and y must have identical lengths")

        dt = self._compute_dt(ts, length)

        xs_norm = xs_arr / self.keyboard.width
        ys_norm = ys_arr / self.keyboard.height

        vx = np.zeros(length, dtype=np.float32)
        vy = np.zeros(length, dtype=np.float32)
        vx[1:] = (xs_norm[1:] - xs_norm[:-1]) / dt[1:]
        vy[1:] = (ys_norm[1:] - ys_norm[:-1]) / dt[1:]
        vx *= self.velocity_scale
        vy *= self.velocity_scale
        if self.velocity_clip is not None:
            np.clip(vx, -self.velocity_clip, self.velocity_clip, out=vx)
            np.clip(vy, -self.velocity_clip, self.velocity_clip, out=vy)

        ax = np.zeros(length, dtype=np.float32)
        ay = np.zeros(length, dtype=np.float32)
        ax[1:] = (vx[1:] - vx[:-1]) / dt[1:]
        ay[1:] = (vy[1:] - vy[:-1]) / dt[1:]
        ax *= self.acceleration_scale
        ay *= self.acceleration_scale
        if self.acceleration_clip is not None:
            np.clip(ax, -self.acceleration_clip, self.acceleration_clip, out=ax)
            np.clip(ay, -self.acceleration_clip, self.acceleration_clip, out=ay)

        features = np.stack([xs_norm, ys_norm, vx, vy, ax, ay], axis=1)
        seq_len = length

        if truncate and length > self.max_seq_len:
            features = features[: self.max_seq_len]
            xs_arr = xs_arr[: self.max_seq_len]
            ys_arr = ys_arr[: self.max_seq_len]
            seq_len = self.max_seq_len

        if seq_len < self.max_seq_len:
            pad_len = self.max_seq_len - seq_len
            features = np.pad(features, ((0, pad_len), (0, 0)), mode="constant")

        nearest_keys = self.keyboard.nearest_key_indices(xs_arr, ys_arr, self.tokenizer)
        if seq_len < self.max_seq_len:
            pad_len = self.max_seq_len - seq_len
            nearest_keys = np.pad(nearest_keys, (0, pad_len), constant_values=self.tokenizer.pad_idx)
        elif seq_len > self.max_seq_len:
            nearest_keys = nearest_keys[: self.max_seq_len]

        return TrajectoryFeatures(features=features.astype(np.float32), nearest_keys=nearest_keys, seq_len=seq_len)

    # ------------------------------------------------------------------
    def to_tensors(
        self,
        xs: Sequence[float],
        ys: Sequence[float],
        ts: Optional[Sequence[float]],
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        traj = self.extract(xs, ys, ts)
        features = torch.tensor(traj.features, dtype=torch.float32, device=device)
        nearest = torch.tensor(traj.nearest_keys, dtype=torch.long, device=device)
        return features, nearest, traj.seq_len

    # ------------------------------------------------------------------
    def encode_word(
        self,
        word: str,
        max_word_len: int = 20,
    ) -> torch.Tensor:
        tokens = self.tokenizer.encode(word)
        if len(tokens) > max_word_len:
            tokens = tokens[: max_word_len - 1] + [self.tokenizer.eos_idx]
        pad_len = max_word_len - len(tokens)
        if pad_len > 0:
            tokens = tokens + [self.tokenizer.pad_idx] * pad_len
        return torch.tensor(tokens, dtype=torch.long)

    # ------------------------------------------------------------------
    def export_config(self) -> Dict[str, float]:
        return {
            "keyboard_width": self.keyboard.width,
            "keyboard_height": self.keyboard.height,
            "velocity_clip": self.velocity_clip,
            "acceleration_clip": self.acceleration_clip,
            "velocity_scale": self.velocity_scale,
            "acceleration_scale": self.acceleration_scale,
            "dt_floor_ms": self.dt_floor_ms,
            "default_dt_ms": self.default_dt_ms,
            "blend_uniform_dt_prob": self.blend_uniform_dt_prob,
            "max_seq_len": self.max_seq_len,
        }


__all__ = [
    "KeyboardGeometry",
    "CharTokenizer",
    "SwipeFeatureExtractor",
    "TrajectoryFeatures",
]
