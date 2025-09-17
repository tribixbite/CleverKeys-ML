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

import datetime as dt
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import WeightedRandomSampler
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from collections import Counter

import nemo.collections.asr as nemo_asr
try:
    from nemo.collections.asr.data.audio_to_text import DALIOutputs  # type: ignore
except ImportError:  # pragma: no cover - DALI optional dependency
    DALIOutputs = type('DALIOutputsPlaceholder', (object,), {'has_processed_signal': False})
from nemo.core.classes.mixins import AccessMixin

from swipe_data_utils import collate_fn

SCRIPT_DIR = Path(__file__).resolve().parent
runtime_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")


# ---------------------------------------------------------------------------
# Configuration ----------------------------------------------------------------

CONFIG: Dict[str, Any] = {
    "data": {
        "train_manifest": "../../data/train_final_train.jsonl",
        "val_manifest": "../../data/train_final_val.jsonl",
        "vocab_path": "../../data/vocab.txt",
        "max_trace_len": 256,
    },
    "training": {
        "batch_size": 320,
        "num_workers": 10,
        "learning_rate": 2e-4,
        "max_epochs": 120,
        "gradient_accumulation": 1,
        "accelerator": "gpu",
        "devices": 1,
        "precision": "bf16-mixed",
        "warmup_steps": 1500,
        "teacher_checkpoint": None,
        "kd_lambda": 0.05,
        "kd_temperature": 1.5,
    },
    "model": {
        "encoder": {
            "feat_in": 37,
            "d_model": 256,
            "n_heads": 4,
            "num_layers": 8,
            "conv_kernel_size": 31,
            "subsampling_factor": 2,
        },
        "decoder": {
            "pred_hidden": 320,
            "pred_rnn_layers": 2,
        },
        "joint": {
            "joint_hidden": 512,
            "activation": "relu",
            "dropout": 0.1,
        },
        "char_sequence_count": 4,
    },
    "preprocess": {
        "resample_short_target": 56,
        "resample_long_target": 96,
        "resample_short_threshold": 48,
        "resample_long_threshold": 112,
    },
    "sampling": {
        "strategy": "none",  # options: none, inverse_sqrt_freq
        "freq_power": 0.5,
        "length_power": 0.0,
        "rare_frequency_threshold": 5,
        "rare_word_boost": 2.0,
        "max_weight_factor": 10.0,
    },
}


# ---------------------------------------------------------------------------
# Utility helpers -------------------------------------------------------------


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
    path = Path(path_str)
    if not path.is_absolute():
        path = (SCRIPT_DIR / path).resolve()
    return str(path)


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def determine_resample_target(length: int, cfg: Dict[str, Any]) -> int:
    if length <= 1:
        return length
    short_target = cfg["resample_short_target"]
    long_target = cfg["resample_long_target"]
    if length <= cfg["resample_short_threshold"]:
        return max(length, short_target)
    if length >= cfg["resample_long_threshold"]:
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


# ---------------------------------------------------------------------------
# Feature extraction ---------------------------------------------------------


class PersonalizedSwipeFeaturizer:
    """Feature generator mirroring the web demo pipeline."""

    def __init__(self, key_centers: Optional[List[Tuple[str, float, float]]] = None):
        self.key_centers = key_centers or KEY_CENTERS_CENTERED

    def __call__(self, points: Iterable[Dict[str, float]]) -> np.ndarray:
        pts = list(points)
        if not pts:
            return np.zeros((1, 37), dtype=np.float32)
        if len(pts) == 1:
            return np.zeros((1, 37), dtype=np.float32)

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

        # Distances to nearest keys (top 5)
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
            mean_x = float(np.mean(xs))
            std_x = float(np.std(xs))
            mean_y = float(np.mean(ys))
            std_y = float(np.std(ys))
            range_x = max(xs) - min(xs)
            range_y = max(ys) - min(ys)
        else:
            mean_x = x
            std_x = 0.0
            mean_y = y
            std_y = 0.0
            range_x = 0.0
            range_y = 0.0

        features = [
            x,
            y,
            t_seconds,
            vx,
            vy,
            speed,
            ax,
            ay,
            acc,
            angle,
            angle_sin,
            angle_cos,
            curvature,
            *key_distances,
            progress,
            is_start,
            is_end,
            mean_x,
            std_x,
            mean_y,
            std_y,
            range_x,
            range_y,
        ]

        while len(features) < 37:
            features.append(0.0)

        return np.array(features[:37], dtype=np.float32)


# ---------------------------------------------------------------------------
# Dataset --------------------------------------------------------------------


class PersonalizedSwipeDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        vocab: Dict[str, int],
        max_trace_len: int,
        preprocess_cfg: Dict[str, Any],
        featurizer: Optional[PersonalizedSwipeFeaturizer] = None,
    ) -> None:
        super().__init__()
        self.manifest_path = manifest_path
        self.vocab = vocab
        self.max_trace_len = max_trace_len
        self.preprocess_cfg = preprocess_cfg
        self.featurizer = featurizer or PersonalizedSwipeFeaturizer()
        self.samples: List[Dict[str, Any]] = []

        with open(manifest_path, "r", encoding="utf-8") as fh:
            for line in fh:
                payload = json.loads(line)
                if not payload.get("word") or not payload.get("points"):
                    continue
                self.samples.append(payload)

        self.word_counts = Counter(sample["word"] for sample in self.samples)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.samples)

    def __getitem__(self, index: int):
        item = self.samples[index]
        raw_points = item["points"][: self.max_trace_len]

        normalized = self._normalize_points(raw_points)
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

    def compute_sampling_weights(self, sampling_cfg: Dict[str, Any]) -> Optional[np.ndarray]:
        strategy = sampling_cfg.get("strategy", "none")
        if strategy == "none" or not self.samples:
            return None

        freq_power = float(sampling_cfg.get("freq_power", 0.5))
        length_power = float(sampling_cfg.get("length_power", 0.0))
        rare_frequency_threshold = int(sampling_cfg.get("rare_frequency_threshold", 0))
        rare_word_boost = float(sampling_cfg.get("rare_word_boost", 1.0))
        max_weight_factor = float(sampling_cfg.get("max_weight_factor", 10.0))

        weights: List[float] = []
        for sample in self.samples:
            word = sample["word"]
            freq = max(self.word_counts.get(word, 1), 1)
            weight = 1.0

            if strategy == "inverse_sqrt_freq":
                exponent = -abs(freq_power)
                weight *= freq ** exponent

            if length_power:
                weight *= max(len(word), 1) ** length_power

            if rare_frequency_threshold and freq <= rare_frequency_threshold:
                weight *= rare_word_boost

            weights.append(weight)

        weights_arr = np.asarray(weights, dtype=np.float64)
        weights_arr /= weights_arr.mean()
        if max_weight_factor > 0:
            weights_arr = np.clip(weights_arr, 1.0 / max_weight_factor, max_weight_factor)
        return weights_arr.astype(np.float64)

    @staticmethod
    def _normalize_points(points: List[Dict[str, Any]]) -> List[Dict[str, float]]:
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


# ---------------------------------------------------------------------------
# Distillation-ready RNNT model ---------------------------------------------


class PersonalizedRNNTModel(nemo_asr.models.EncDecRNNTModel):
    def __init__(self, cfg: DictConfig, kd_lambda: float = 0.0, kd_temperature: float = 1.0,
                 teacher_checkpoint: Optional[str] = None):
        super().__init__(cfg=cfg)
        self.kd_lambda = kd_lambda
        self.kd_temperature = kd_temperature
        self.teacher = None
        if teacher_checkpoint:
            self._init_teacher(teacher_checkpoint)

    def _init_teacher(self, checkpoint: str) -> None:
        self.teacher = nemo_asr.models.EncDecRNNTModel.restore_from(checkpoint, map_location="cpu")
        self.teacher.freeze()
        self.teacher.eval()

    def forward(self, input_signal=None, input_signal_length=None, processed_signal=None, processed_signal_length=None):
        """Bypass the audio preprocessor; accept already-computed feature tensors."""
        if input_signal is not None and processed_signal is None:
            processed_signal = input_signal.transpose(1, 2)
            processed_signal_length = input_signal_length
        elif processed_signal is None:
            raise ValueError("PersonalizedRNNTModel.forward requires input_signal or processed_signal")

        encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length)
        return encoded, encoded_len

    def _compute_joint(self, model: nemo_asr.models.EncDecRNNTModel, batch, detach: bool = True):
        signal, signal_len, transcript, transcript_len = batch
        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            encoded, encoded_len = model.forward(processed_signal=signal, processed_signal_length=signal_len)
        else:
            encoded, encoded_len = model.forward(input_signal=signal, input_signal_length=signal_len)
        decoder, target_length, _ = model.decoder(targets=transcript, target_length=transcript_len)
        joint = model.joint(encoder_outputs=encoded, decoder_outputs=decoder)
        if detach:
            joint = joint.detach()
        return joint, encoded_len, target_length

    def training_step(self, batch, batch_idx):  # noqa: D401 - mirrors base class
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        signal, signal_len, transcript, transcript_len = batch

        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            encoded, encoded_len = self.forward(processed_signal=signal, processed_signal_length=signal_len)
        else:
            encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)

        decoder, target_length, _ = self.decoder(targets=transcript, target_length=transcript_len)

        if hasattr(self, '_trainer') and self._trainer is not None:
            log_every_n_steps = self._trainer.log_every_n_steps
            sample_id = self._trainer.global_step
        else:  # pragma: no cover - defensive guard
            log_every_n_steps = 1
            sample_id = batch_idx

        joint = self.joint(encoder_outputs=encoded, decoder_outputs=decoder)
        loss_value = self.loss(
            log_probs=joint,
            targets=transcript,
            input_lengths=encoded_len,
            target_lengths=target_length,
        )
        loss_value = self.add_auxiliary_losses(loss_value)

        kd_loss = None
        if self.teacher is not None and self.kd_lambda > 0:
            with torch.no_grad():
                teacher_joint, _, _ = self._compute_joint(self.teacher, batch, detach=True)
            student_log_probs = torch.nn.functional.log_softmax(joint / self.kd_temperature, dim=-1)
            teacher_probs = torch.nn.functional.softmax(teacher_joint / self.kd_temperature, dim=-1)
            kd_loss = torch.nn.functional.kl_div(
                student_log_probs,
                teacher_probs,
                reduction='batchmean',
            ) * (self.kd_lambda * (self.kd_temperature ** 2))
            loss_value = loss_value + kd_loss

        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        logs = {
            'train_loss': loss_value,
            'learning_rate': self._optimizer.param_groups[0]['lr'],
            'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
        }
        if kd_loss is not None:
            logs['kd_loss'] = kd_loss.detach()

        if (sample_id + 1) % log_every_n_steps == 0:
            # Disable autocast so metrics run in fp32 (avoids bf16 accumulation mismatch)
            with torch.cuda.amp.autocast(enabled=False):
                self.wer.update(
                    predictions=encoded,
                    predictions_lengths=encoded_len,
                    targets=transcript,
                    targets_lengths=transcript_len,
                )
            _, scores, words = self.wer.compute()
            self.wer.reset()
            logs['training_batch_wer'] = scores.float() / words

        self.log_dict(logs)
        return {'loss': loss_value}

    # ------------------------------------------------------------------
    # Validation/Test hooks -------------------------------------------------

    def validation_step(self, *args, **kwargs):  # type: ignore[override]
        # Run RNNT decoding/WER in fp32 to avoid dtype clashes with CUDA graphs
        with torch.cuda.amp.autocast(enabled=False):
            return super().validation_step(*args, **kwargs)

    def test_step(self, *args, **kwargs):  # type: ignore[override]
        with torch.cuda.amp.autocast(enabled=False):
            return super().test_step(*args, **kwargs)


# ---------------------------------------------------------------------------
# Training orchestration -----------------------------------------------------


def load_vocab(vocab_path: str) -> Dict[str, int]:
    vocab: Dict[str, int] = {}
    with open(vocab_path, "r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            token = line.strip()
            if token:
                vocab[token] = idx
    if '<unk>' not in vocab:
        vocab['<unk>'] = len(vocab)
    return vocab


def build_dataloaders(cfg: DictConfig, vocab: Dict[str, int]):
    train_ds = PersonalizedSwipeDataset(
        manifest_path=cfg.data.train_manifest,
        vocab=vocab,
        max_trace_len=cfg.data.max_trace_len,
        preprocess_cfg=cfg.preprocess,
    )

    val_ds = PersonalizedSwipeDataset(
        manifest_path=cfg.data.val_manifest,
        vocab=vocab,
        max_trace_len=cfg.data.max_trace_len,
        preprocess_cfg=cfg.preprocess,
    )

    collate = collate_fn

    train_weights = train_ds.compute_sampling_weights(cfg.sampling)
    train_sampler: Optional[WeightedRandomSampler] = None
    if train_weights is not None:
        train_sampler = WeightedRandomSampler(
            weights=torch.tensor(train_weights, dtype=torch.double),
            num_samples=len(train_weights),
            replacement=True,
        )
        rare_thresh = cfg.sampling.get('rare_frequency_threshold', 0)
        print(
            f"Sampling strategy '{cfg.sampling.strategy}' enabled: weight range "
            f"{train_weights.min():.3f}–{train_weights.max():.3f}, "
            f"rare_threshold={rare_thresh}"
        )

    train_loader_kwargs = dict(
        dataset=train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=train_sampler is None,
        num_workers=cfg.training.num_workers,
        collate_fn=collate,
        pin_memory=_has_usable_cuda(),
        drop_last=True,
        persistent_workers=cfg.training.num_workers > 0,
    )
    if train_sampler is not None:
        train_loader_kwargs['sampler'] = train_sampler
    if cfg.training.num_workers > 0:
        train_loader_kwargs['prefetch_factor'] = 4
    train_loader = DataLoader(**train_loader_kwargs)

    val_workers = max(0, cfg.training.num_workers // 2)
    val_loader_kwargs = dict(
        dataset=val_ds,
        batch_size=cfg.training.batch_size * 2,
        shuffle=False,
        num_workers=val_workers,
        collate_fn=collate,
        pin_memory=_has_usable_cuda(),
        drop_last=False,
        persistent_workers=val_workers > 0,
    )
    if val_workers > 0:
        val_loader_kwargs['prefetch_factor'] = 2
    val_loader = DataLoader(**val_loader_kwargs)

    return train_loader, val_loader


def build_model_config(cfg: DictConfig, labels: List[str]) -> DictConfig:
    nemo_cfg = DictConfig({
        'labels': labels,
        'sample_rate': 16000,
        'model_defaults': {
            'enc_hidden': cfg.model.encoder.d_model,
            'pred_hidden': cfg.model.decoder.pred_hidden,
        },
        'preprocessor': {
            '_target_': 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor',
            'sample_rate': 16000,
            'normalize': 'per_feature',
            'window_size': 0.025,
            'window_stride': 0.01,
            'features': cfg.model.encoder.feat_in,
            'n_fft': 512,
            'frame_splicing': 1,
            'dither': 0.0,
        },
        'encoder': {
            '_target_': 'nemo.collections.asr.modules.ConformerEncoder',
            'feat_in': cfg.model.encoder.feat_in,
            'n_layers': cfg.model.encoder.num_layers,
            'd_model': cfg.model.encoder.d_model,
            'feat_out': -1,
            'subsampling': 'striding',
            'subsampling_factor': cfg.model.encoder.subsampling_factor,
            'subsampling_conv_channels': -1,
            'ff_expansion_factor': 4,
            'self_attention_model': 'rel_pos',
            'att_context_size': [-1, -1],
            'conv_kernel_size': cfg.model.encoder.conv_kernel_size,
            'dropout': 0.1,
            'pos_emb_max_len': 4000,
        },
        'decoder': {
            '_target_': 'nemo.collections.asr.modules.rnnt.RNNTDecoder',
            'prednet': {
                'pred_hidden': cfg.model.decoder.pred_hidden,
                'pred_rnn_layers': cfg.model.decoder.pred_rnn_layers,
                'forget_gate_bias': 1.0,
                't_max': None,
                'dropout': 0.1,
            },
            'vocab_size': len(labels),
            'blank_as_pad': True,
        },
        'joint': {
            '_target_': 'nemo.collections.asr.modules.rnnt.RNNTJoint',
            'fuse_loss_wer': False,
            'jointnet': {
                'joint_hidden': cfg.model.joint.joint_hidden,
                'activation': cfg.model.joint.activation,
                'dropout': cfg.model.joint.dropout,
            },
            'num_classes': len(labels),
            'vocabulary': labels,
            'log_softmax': True,
            'preserve_memory': False,
        },
        'decoding': {
            'strategy': 'greedy_batch',
            'greedy': {
                'max_symbols': 15,
            },
            'greedy_batch': {
                'max_symbols': 13,
                'enable_cuda_graphs': False,
            },
            'use_cuda_graphs': False,
            'preserve_alignments': False,
            'preserve_word_confidence': False,
            'preserve_frame_confidence': False,
            'confidence_method_cfg': None,
        },
        'loss': {
            '_target_': 'nemo.collections.asr.losses.rnnt_loss.RNNTLoss',
            'warprnnt_numba_kwargs': {
                'numba_batch_size': 8,
            },
        },
        'optim': {
            'name': 'adamw',
            'lr': cfg.training.learning_rate,
            'betas': [0.9, 0.98],
            'weight_decay': 1e-3,
            'sched': {
                'name': 'CosineAnnealing',
                'warmup_steps': cfg.training.warmup_steps,
                'last_epoch': -1,
            },
        },
    })
    return nemo_cfg


def find_latest_checkpoint() -> Optional[str]:
    patterns = [
        './rnnt_checkpoints_*/conformer_rnnt/*/checkpoints/*.ckpt',
        './rnnt_logs_*/conformer_rnnt/*/checkpoints/*.ckpt',
        './rnnt_logs/conformer_rnnt/*/checkpoints/*.ckpt',
    ]
    candidates: List[str] = []
    for pattern in patterns:
        candidates.extend(Path().glob(pattern))
    if not candidates:
        return None
    return str(max(candidates, key=lambda p: p.stat().st_mtime))


def main() -> None:
    cfg = DictConfig(CONFIG)
    cfg.data.train_manifest = _resolve_path(cfg.data.train_manifest)
    cfg.data.val_manifest = _resolve_path(cfg.data.val_manifest)
    cfg.data.vocab_path = _resolve_path(cfg.data.vocab_path)

    if not _has_usable_cuda():
        cfg.training.accelerator = 'cpu'
        cfg.training.devices = 1
        cfg.training.precision = '32-true'
        cfg.training.num_workers = 0
        import types
        torch.cuda.is_available = types.MethodType(lambda self=None: False, torch.cuda)
        torch.cuda.device_count = types.MethodType(lambda self=None: 0, torch.cuda)
        torch.cuda.current_device = types.MethodType(lambda self=None: 0, torch.cuda)

    vocab = load_vocab(cfg.data.vocab_path)
    train_loader, val_loader = build_dataloaders(cfg, vocab)

    nemo_cfg = build_model_config(cfg, list(vocab.keys()))
    model = PersonalizedRNNTModel(
        cfg=nemo_cfg,
        kd_lambda=cfg.training.kd_lambda,
        kd_temperature=cfg.training.kd_temperature,
        teacher_checkpoint=_resolve_path(cfg.training.teacher_checkpoint)
        if cfg.training.teacher_checkpoint else None,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_wer',
        mode='min',
        save_top_k=3,
        filename='epoch={epoch:02d}-wer={val_wer:.3f}',
        save_last=True,
    )

    fast_dev = bool(int(os.environ.get("FAST_DEV_RUN", "0")))
    if fast_dev:
        print("FAST_DEV_RUN=1 -> running a single batch for smoke test")

    trainer = pl.Trainer(
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        precision=cfg.training.precision,
        max_epochs=cfg.training.max_epochs,
        log_every_n_steps=20,
        gradient_clip_val=1.0,
        accumulate_grad_batches=cfg.training.gradient_accumulation,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback],
        default_root_dir=f'./rnnt_checkpoints_{runtime_id}',
        enable_progress_bar=True,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0,
        fast_dev_run=fast_dev,
    )

    resume_from = find_latest_checkpoint()
    if resume_from:
        print(f"Resuming from {resume_from}")

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=resume_from)

    nemo_path = Path(f"conformer_rnnt_personalized_{runtime_id}.nemo")
    model.save_to(str(nemo_path))
    print(f"Saved final NeMo checkpoint to {nemo_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
