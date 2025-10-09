#!/usr/bin/env python3
"""
DistilHuBERT + CTC training for gesture typing (37D features).

This script fine-tunes a Hugging Face Wav2Vec2/Hubert-style CTC model to
predict characters from swipe-gesture features using the Trainer API.

Key points:
- Uses a custom featurizer to produce 37-dim features per timestep.
- Adapts the first Conv1d layer to accept 37 channels instead of audio mono.
- Uses CTC loss with a simple character vocabulary: a-z and apostrophe.
- Respects dataset paths used in train_squeeze2.py.

Usage:
  uv run python train_distilhubert_ctc.py \
    --output-dir distilhubert-ctc-gesture-model \
    --batch-size 128 --epochs 100

The script will automatically resume from the latest checkpoint in the
output directory if present (can be disabled with --no-auto-resume).
"""

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
import jiwer
import types

from transformers import (
    Trainer,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    HubertForCTC,
    AutoConfig,
)


# -----------------------------------------------------------------------------
# Config (dataset paths mirror train_squeeze2.py)
# -----------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "model_name": "distilhubert-ctc-gesture-model",
    # A HuBERT/DistilHuBERT-style backbone; allow size mismatches when adapting conv.
    # Prefer a widely available encoder; DistilHuBERT repos vary.
    # Use HuBERT base as stable default; still adapted to 37D inputs and CTC.
    "pretrained_model": "facebook/hubert-base-ls960",
    "data": {
        "train_manifest": "data/train_final_train.jsonl",
        "val_manifest": "data/train_final_val.jsonl",
        "vocab_chars": "abcdefghijklmnopqrstuvwxyz",
    },
    "training": {
        "batch_size": 128,  # per device
        "learning_rate": 3e-4,
        "max_epochs": 100,
        "warmup_steps": 1000,
        "fp16": torch.cuda.is_available(),
        "group_by_length": True,
        "logging_steps": 25,
        "save_total_limit": 2,
    },
    "preprocess": {
        "min_points": 10,
    },
}


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("training.log")],
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Feature extraction (aligned with train_squeeze2’s featurizer behavior)
# -----------------------------------------------------------------------------

class PersonalizedSwipeFeaturizer:
    """
    37-dimensional feature extractor for swipe gestures.
    Features: position, velocity, acceleration, keyboard proximity, curvature.
    """

    def __init__(self):
        self.feature_dim = 37
        self.key_positions = self._get_keyboard_layout()
        self._key_chars = list(self.key_positions.keys())
        self._keys_xy = np.array(
            [self.key_positions[c] for c in self._key_chars], dtype=np.float32
        )

    def _get_keyboard_layout(self) -> Dict[str, Tuple[float, float]]:
        layout: Dict[str, Tuple[float, float]] = {}
        rows = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
        for row_idx, row in enumerate(rows):
            y = row_idx * 0.33
            for col_idx, char in enumerate(row):
                x_offset = 0.05 * row_idx if row_idx > 0 else 0
                x = (col_idx / 10.0) + x_offset
                layout[char] = (x, y)
        # Apostrophe removed: training keyboard has no apostrophe key
        return layout

    def __call__(self, points: List[Dict]) -> np.ndarray:
        if len(points) < DEFAULT_CONFIG["preprocess"]["min_points"]:
            return np.zeros((0, self.feature_dim), dtype=np.float32)

        p = np.array([[pt.get("x", 0.0), pt.get("y", 0.0), pt.get("t", 0.0)] for pt in points], dtype=np.float32)
        x, y, t = p[:, 0], p[:, 1], p[:, 2]
        num_points = len(p)
        features = np.zeros((num_points, self.feature_dim), dtype=np.float32)

        # Position
        features[:, 0] = x
        features[:, 1] = y

        # Velocity (backward diff)
        dt_bwd = np.diff(t, prepend=t[0]) / 1000.0
        dt_bwd = np.maximum(dt_bwd, 1e-6)
        vx = np.diff(x, prepend=x[0]) / dt_bwd
        vy = np.diff(y, prepend=y[0]) / dt_bwd
        vx[0], vy[0] = 0.0, 0.0
        features[:, 2] = vx
        features[:, 3] = vy

        # Acceleration (central diff)
        vx_next = np.roll(vx, -1)
        vy_next = np.roll(vy, -1)
        t_prev = np.roll(t, 1)
        t_next = np.roll(t, -1)
        dt_total = (t_next - t_prev) / 1000.0
        dt_total = np.maximum(dt_total, 1e-6)
        ax = (vx_next - vx) / dt_total
        ay = (vy_next - vy) / dt_total
        ax[0] = ay[0] = ax[-1] = ay[-1] = 0.0
        features[:, 4] = ax
        features[:, 5] = ay

        # Speed and direction
        features[:, 6] = np.hypot(vx, vy)
        features[:, 7] = np.arctan2(vy, vx)

        # Keyboard proximity features
        points_xy = p[:, :2]
        diffs = points_xy[:, np.newaxis, :] - self._keys_xy[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        num_keys = self._keys_xy.shape[0]
        prox = np.exp(-dists * 5)
        end_idx = 8 + num_keys
        end_idx = min(end_idx, 36)  # guard
        features[:, 8:end_idx] = prox[:, : (end_idx - 8)]

        # Curvature
        p_prev = np.roll(points_xy, 1, axis=0)
        p_next = np.roll(points_xy, -1, axis=0)
        v1 = points_xy - p_prev
        v2 = p_next - points_xy
        norm_v1 = np.linalg.norm(v1, axis=1)
        norm_v2 = np.linalg.norm(v2, axis=1)
        dot_product = np.einsum("ij,ij->i", v1, v2)
        denom = norm_v1 * norm_v2
        cos_angle = np.zeros(num_points, dtype=np.float32)
        valid_mask = denom > 1e-6
        cos_angle[valid_mask] = np.clip(dot_product[valid_mask] / denom[valid_mask], -1.0, 1.0)
        curvature = np.arccos(cos_angle)
        curvature[0] = curvature[-1] = 0.0
        features[:, 36] = curvature

        return features


# -----------------------------------------------------------------------------
# Custom model to accept 37-channel inputs
# -----------------------------------------------------------------------------

class GestureWav2Vec2ForCTC(Wav2Vec2ForCTC):
    """Adapt first conv layer to accept 37 channels; transpose input time/features."""

    def __init__(self, config):
        super().__init__(config)
        # Replace first conv to support 37-channel inputs (fail fast if not possible)
        new_conv = nn.Conv1d(
            in_channels=37,
            out_channels=config.conv_dim[0],
            kernel_size=config.conv_kernel[0],
            stride=config.conv_stride[0],
            bias=False,
        )
        self.wav2vec2.feature_extractor.conv_layers[0].conv = new_conv

    def forward(
        self,
        input_values,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
    ):
        # Expect (batch, time, features) -> (batch, features, time)
        if input_values.dim() == 3:
            input_values = input_values.transpose(1, 2)

        return super().forward(
            input_values=input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            labels=labels,
        )


class GestureHubertForCTC(HubertForCTC):
    """Adapt first conv layer to accept 37 channels for HuBERT; transpose input."""

    def __init__(self, config):
        super().__init__(config)
        # Replace first conv to support 37-channel inputs (fail fast if not possible)
        new_conv = nn.Conv1d(
            in_channels=37,
            out_channels=config.conv_dim[0],
            kernel_size=config.conv_kernel[0],
            stride=config.conv_stride[0],
            bias=False,
        )
        self.hubert.feature_extractor.conv_layers[0].conv = new_conv

    def forward(
        self,
        input_values,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
    ):
        if input_values.dim() == 3:
            input_values = input_values.transpose(1, 2)
        return super().forward(
            input_values=input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            labels=labels,
        )


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def load_data_from_manifest(manifest_path: str) -> List[Dict]:
    data: List[Dict] = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (
                    isinstance(item, dict)
                    and "word" in item
                    and "points" in item
                    and len(item["points"]) >= DEFAULT_CONFIG["preprocess"]["min_points"]
                ):
                    data.append(item)
    return data


@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # Strict manual padding to (B, T, 37)
        xs: List[torch.Tensor] = []
        lengths: List[int] = []
        for f in features:
            arr = np.asarray(f["input_values"], dtype=np.float32)
            assert arr.ndim == 2 and arr.shape[1] == 37, f"Expected (T,37), got {arr.shape}"
            ten = torch.from_numpy(arr)
            xs.append(ten)
            lengths.append(ten.shape[0])

        max_len = max(lengths) if lengths else 0
        B = len(xs)
        feat_dim = 37
        input_values = torch.zeros((B, max_len, feat_dim), dtype=torch.float32)
        attention_mask = torch.zeros((B, max_len), dtype=torch.long)
        for i, t in enumerate(lengths):
            if t > 0:
                input_values[i, :t] = xs[i]
                attention_mask[i, :t] = 1

        # Labels
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            return_tensors="pt",
        )
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        return {"input_values": input_values, "attention_mask": attention_mask, "labels": labels}


def build_tokenizer_and_processor(vocab_chars: str, workdir: Path) -> Tuple[Wav2Vec2Processor, Wav2Vec2CTCTokenizer, Path]:
    vocab_dict = {char: i for i, char in enumerate(vocab_chars)}
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    tok_dir = workdir / "tokenizer"
    tok_dir.mkdir(parents=True, exist_ok=True)
    vocab_file = tok_dir / "vocab.json"
    with open(vocab_file, "w", encoding="utf-8") as f:
        json.dump(vocab_dict, f)

    # Instantiate tokenizer directly from vocab file to avoid incorrect class resolution.
    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_file=str(vocab_file), unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token=None
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=37, sampling_rate=100, padding_value=0.0, do_normalize=False
    )
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    return processor, tokenizer, tok_dir


def compute_metrics_builder(tokenizer: Wav2Vec2CTCTokenizer):
    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        # Replace -100 with pad token id for decoding
        label_ids = pred.label_ids.copy()
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(pred_ids)
        label_str = tokenizer.batch_decode(label_ids, group_tokens=False)

        # Track empty predictions, sanitize for CER
        empty_pred = sum(1 for s in pred_str if not isinstance(s, str) or len(s) == 0)
        pred_str = [s.lower() if isinstance(s, str) and len(s) > 0 else " " for s in pred_str]
        label_str = [s.lower() if isinstance(s, str) and len(s) > 0 else " " for s in label_str]

        transformation = jiwer.ToLowerCase()
        cer = jiwer.cer(label_str, pred_str, truth_transform=transformation, hypothesis_transform=transformation)
        return {"cer": cer, "empty_pred_rate": empty_pred / max(1, len(pred_str))}

    return compute_metrics


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train DistilHuBERT+CTC on gesture features")
    parser.add_argument("--train-manifest", type=str, default=DEFAULT_CONFIG["data"]["train_manifest"])
    parser.add_argument("--val-manifest", type=str, default=DEFAULT_CONFIG["data"]["val_manifest"])
    parser.add_argument("--pretrained-model", type=str, default=DEFAULT_CONFIG["pretrained_model"])
    parser.add_argument("--output-dir", type=str, default=DEFAULT_CONFIG["model_name"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["training"]["batch_size"])
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["training"]["max_epochs"])
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["training"]["learning_rate"])
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_CONFIG["training"]["warmup_steps"])
    parser.add_argument("--no-fp16", action="store_true", help="Disable fp16 even if CUDA is available")
    parser.add_argument("--no-auto-resume", action="store_true", help="Do not auto-resume from latest checkpoint")
    parser.add_argument("--subset-train", type=int, default=0, help="Use only the first N training samples (for quick smoke runs)")
    parser.add_argument("--subset-val", type=int, default=0, help="Use only the first N validation samples (for quick smoke runs)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build tokenizer/processor
    logger.info("Setting up tokenizer and processor...")
    processor, tokenizer, _ = build_tokenizer_and_processor(
        DEFAULT_CONFIG["data"]["vocab_chars"], output_dir
    )
    featurizer = PersonalizedSwipeFeaturizer()

    # Load datasets
    logger.info("Loading datasets...")
    train_data = load_data_from_manifest(args.train_manifest)
    val_data = load_data_from_manifest(args.val_manifest)
    train_ds = Dataset.from_list(train_data)
    val_ds = Dataset.from_list(val_data)

    # Preprocess: normalize points and featurize; tokenize labels
    allowed_chars = set(DEFAULT_CONFIG["data"]["vocab_chars"])

    def preprocess_function(batch):
        if not batch.get("points"):
            batch["input_values"] = np.zeros((0, 37), dtype=np.float32)
        else:
            start_t = float(batch["points"][0].get("t", 0.0))
            norm_points = [
                {
                    "x": p.get("x", 0.5) * 2.0 - 1.0,
                    "y": p.get("y", 0.5) * 2.0 - 1.0,
                    "t": max(0.0, float(p.get("t", 0.0)) - start_t),
                }
                for p in batch["points"]
            ]
            feats = featurizer(norm_points).astype(np.float32)
            batch["input_values"] = feats

        # Use processor's target processor for labels (letters only)
        clean_word = "".join(c for c in batch.get("word", "").lower() if c in allowed_chars)
        with processor.as_target_processor():
            batch["labels"] = processor.tokenizer(clean_word).input_ids
        # Provide length column for grouped sampling
        batch["input_length"] = int(batch["input_values"].shape[0]) if isinstance(batch["input_values"], np.ndarray) else 0
        return batch

    # Optional subsetting for quick pipeline checks (do this BEFORE map for speed)
    if args.subset_train and len(train_ds) > args.subset_train:
        logger.info(f"Subsetting train dataset to first {args.subset_train} samples (from {len(train_ds)})")
        train_ds = train_ds.select(range(args.subset_train))
    if args.subset_val and len(val_ds) > args.subset_val:
        logger.info(f"Subsetting val dataset to first {args.subset_val} samples (from {len(val_ds)})")
        val_ds = val_ds.select(range(args.subset_val))

    # Filter out samples with too few points or that clean to empty label
    def _filter_fn(example):
        has_points = example.get("points") and len(example["points"]) >= DEFAULT_CONFIG["preprocess"]["min_points"]
        word_clean = "".join(c for c in example.get("word", "").lower() if c in allowed_chars)
        return bool(has_points and len(word_clean) > 0)

    train_ds = train_ds.filter(_filter_fn)
    val_ds = val_ds.filter(_filter_fn)

    logger.info("Featurizing and tokenizing...")
    train_ds = train_ds.map(preprocess_function, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(preprocess_function, remove_columns=val_ds.column_names)
    logger.info(f"Training samples: {len(train_ds)} | Validation samples: {len(val_ds)}")

    # Model
    logger.info(f"Loading backbone: {args.pretrained_model}")
    cfg = AutoConfig.from_pretrained(args.pretrained_model)
    if cfg.model_type == "wav2vec2":
        model = GestureWav2Vec2ForCTC.from_pretrained(
            args.pretrained_model,
            pad_token_id=processor.tokenizer.pad_token_id,
            vocab_size=len(processor.tokenizer),
            ignore_mismatched_sizes=True,
        )
    elif cfg.model_type == "hubert":
        model = GestureHubertForCTC.from_pretrained(
            args.pretrained_model,
            pad_token_id=processor.tokenizer.pad_token_id,
            vocab_size=len(processor.tokenizer),
            ignore_mismatched_sizes=True,
        )
    else:
        raise ValueError(f"Unsupported encoder model_type for CTC adaptation: {cfg.model_type}")

    # Disable masking which expects longer sequences in HuBERT/Wav2Vec2 pretraining
    if hasattr(model, "config"):
        try:
            model.config.mask_time_prob = 0.0
            model.config.mask_feature_prob = 0.0
        except Exception as e:
            logger.warning(f"Could not disable masking: {e}")

    # Freeze low-level feature extractor (common fine-tuning practice)
    if hasattr(model, "freeze_feature_extractor"):
        try:
            model.freeze_feature_extractor()
            logger.info("Froze feature extractor successfully.")
        except Exception as e:
            logger.warning(f"Failed to freeze feature extractor: {e}")

    # Training args
    fp16 = DEFAULT_CONFIG["training"]["fp16"] and (not args.no_fp16)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        group_by_length=True,
        length_column_name="input_length",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        fp16=fp16,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        save_total_limit=DEFAULT_CONFIG["training"]["save_total_limit"],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        logging_steps=DEFAULT_CONFIG["training"]["logging_steps"],
        report_to=["tensorboard"],
    )

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    compute_metrics = compute_metrics_builder(tokenizer)

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    # Auto-resume from latest checkpoint
    resume_from_checkpoint = None
    if not args.no_auto_resume:
        ckpts = sorted(output_dir.glob("checkpoint-*"), key=os.path.getmtime, reverse=True)
        if ckpts:
            resume_from_checkpoint = str(ckpts[0])
            logger.info(f"Auto-resuming from: {resume_from_checkpoint}")

    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    best_dir = output_dir / "best_checkpoint"
    best_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(best_dir))
    processor.save_pretrained(str(best_dir))
    logger.info(f"Best model and processor saved to {best_dir}")


if __name__ == "__main__":
    main()
