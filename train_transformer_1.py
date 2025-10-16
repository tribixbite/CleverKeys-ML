#!/usr/bin/env python3
"""
Lightweight Transformer + CTC training for gesture typing.

This script trains a custom, small transformer model from scratch to predict
characters from swipe-gesture features. It is architecturally sound for this
task and produces a model small enough for on-device use.

Key Architectural Changes from the Original Script:
- No Pretrained Backbone: Trains a small transformer from a random initialization.
- No Convolutional Subsampling: Feeds gesture features directly to the
  transformer encoder, preserving the full temporal resolution of the swipe.
  This is critical for CTC loss stability and performance on this data type.
- Configurable Architecture: Model size (layers, hidden size) can be set
  via command-line arguments to find the best trade-off between performance
  and on-device feasibility.

Usage:
  # Train the default small model (4 layers, ~5M params)
  uv run python train_transformer_ctc.py \
    --output-dir gesture-transformer-small \
    --batch-size 256 \
    --epochs 100

  # Train a "tiny" model (~1.5M params)
  uv run python train_transformer_ctc.py \
    --output-dir gesture-transformer-tiny \
    --num-layers 3 --hidden-size 192 --ffn-size 768 \
    --batch-size 512 --epochs 100
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
from torch.utils.data import SequentialSampler

from transformers import (
    Trainer,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    Wav2Vec2Config,
)
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Encoder,
    Wav2Vec2PreTrainedModel,
)

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "model_name": "gesture-transformer-model",
    "data": {
        "train_manifest": "data/train_final_train.jsonl",
        "val_manifest": "data/train_final_val.jsonl",
        "vocab_chars": "abcdefghijklmnopqrstuvwxyz",
    },
    # Architecture for a "small" but capable model (~5M parameters)
    "model_arch": {
        "num_hidden_layers": 4,
        "hidden_size": 384,
        "num_attention_heads": 6,
        "ffn_dim": 1536,
        "dropout": 0.1,
    },
    "training": {
        "batch_size": 256,
        "learning_rate": 5e-4,
        "max_epochs": 100,
        "warmup_steps": 2000,
        "fp16": torch.cuda.is_available(),
        "logging_steps": 25,
        "save_total_limit": 2,
    },
    "preprocess": {
        "min_points": 120,
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
# Feature extraction
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
                x_offset = 0.05 * row_idx
                x = (col_idx / 10.0) + x_offset
                layout[char] = (x, y)
        return layout

    def __call__(self, points: List[Dict]) -> np.ndarray:
        if len(points) < DEFAULT_CONFIG["preprocess"]["min_points"]:
            return np.zeros((0, self.feature_dim), dtype=np.float32)

        p = np.array(
            [[pt.get("x", 0.0), pt.get("y", 0.0), pt.get("t", 0.0)] for pt in points],
            dtype=np.float32,
        )
        x, y, t = p[:, 0], p[:, 1], p[:, 2]
        num_points = len(p)
        features = np.zeros((num_points, self.feature_dim), dtype=np.float32)

        # Position
        features[:, 0] = x
        features[:, 1] = y

        # Velocity
        dt = np.diff(t, prepend=t[0]) / 1000.0
        dt = np.maximum(dt, 1e-6)
        vx = np.diff(x, prepend=x[0]) / dt
        vy = np.diff(y, prepend=y[0]) / dt
        vx[0], vy[0] = 0.0, 0.0
        features[:, 2] = vx
        features[:, 3] = vy

        # Acceleration
        ax = np.diff(vx, prepend=vx[0]) / dt
        ay = np.diff(vy, prepend=vy[0]) / dt
        ax[0], ay[0] = 0.0, 0.0
        features[:, 4] = ax
        features[:, 5] = ay

        # Speed and direction
        features[:, 6] = np.hypot(vx, vy)
        features[:, 7] = np.arctan2(vy, vx)

        # Keyboard proximity
        points_xy = p[:, :2]
        diffs = points_xy[:, np.newaxis, :] - self._keys_xy[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        num_keys = self._keys_xy.shape[0]
        prox = np.exp(-dists * 5)
        end_idx = 8 + num_keys
        features[:, 8:end_idx] = prox

        # Curvature
        v1 = np.diff(points_xy, n=1, axis=0, prepend=points_xy[[0]])
        v2 = np.diff(points_xy, n=1, axis=0, append=points_xy[[-1]])
        norm_v1 = np.linalg.norm(v1, axis=1)
        norm_v2 = np.linalg.norm(v2, axis=1)
        dot_product = np.einsum("ij,ij->i", v1, v2)
        denom = norm_v1 * norm_v2
        cos_angle = np.zeros(num_points, dtype=np.float32)
        valid_mask = denom > 1e-6
        cos_angle[valid_mask] = np.clip(
            dot_product[valid_mask] / denom[valid_mask], -1.0, 1.0
        )
        curvature = np.arccos(cos_angle)
        curvature[0] = curvature[-1] = 0.0
        features[:, 36] = curvature

        return features


# -----------------------------------------------------------------------------
# Custom Model (No Subsampling)
# -----------------------------------------------------------------------------


class GestureTransformerModel(Wav2Vec2PreTrainedModel):
    """
    Core transformer model that takes gesture features and encodes them.
    It replaces the convolutional feature extractor with a simple Linear layer.
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.input_projection = nn.Linear(37, config.hidden_size)
        self.dropout = nn.Dropout(config.feat_proj_dropout)
        self.encoder = Wav2Vec2Encoder(config)

    def forward(self, input_values, attention_mask=None, **kwargs):
        hidden_states = self.dropout(self.input_projection(input_values))

        if attention_mask is not None:
            extended_attention_mask = self.invert_attention_mask(attention_mask)
        else:
            extended_attention_mask = None

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=extended_attention_mask,
        )
        return encoder_outputs


class GestureTransformerForCTC(Wav2Vec2ForCTC):
    """
    The full model for CTC, using our custom GestureTransformerModel as the backbone.
    """

    def __init__(self, config):
        super().__init__(config)
        # We replace the default wav2vec2 backbone with our custom one
        self.wav2vec2 = GestureTransformerModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.init_weights()


# -----------------------------------------------------------------------------
# Data processing
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
                    and len(item["points"])
                    >= DEFAULT_CONFIG["preprocess"]["min_points"]
                ):
                    data.append(item)
    return data


@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        input_features = [f["input_values"] for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.pad(
            {"input_values": input_features},
            padding=self.padding,
            return_tensors="pt",
        )

        labels_batch = self.processor.tokenizer.pad(
            label_features, padding=self.padding, return_tensors="pt"
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels
        return batch


class CustomTrainer(Trainer):
    def _get_eval_sampler(self, eval_dataset: Dataset) -> Optional[torch.utils.data.Sampler]:
        return SequentialSampler(eval_dataset)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        attention_mask = inputs.get("attention_mask", None)
        outputs = model(input_values=inputs["input_values"], attention_mask=attention_mask)
        logits = outputs.logits  # (B, T, V)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1).transpose(0, 1)  # (T, B, V)
        if attention_mask is not None:
            input_lengths = attention_mask.sum(dim=1).to(dtype=torch.long)
        else:
            input_lengths = torch.full((logits.size(0),), logits.size(1), dtype=torch.long, device=logits.device)
        target_lengths = (labels != -100).sum(-1)
        targets = labels.masked_fill(labels == -100, model.config.pad_token_id)
        loss_fct = torch.nn.CTCLoss(blank=model.config.pad_token_id, zero_infinity=True)
        loss = loss_fct(log_probs, targets, input_lengths, target_lengths)
        if return_outputs:
            return loss, outputs
        return loss


def build_tokenizer_and_processor(vocab_chars: str, workdir: Path) -> Wav2Vec2Processor:
    vocab_dict = {char: i for i, char in enumerate(vocab_chars)}
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    tok_dir = workdir / "tokenizer"
    tok_dir.mkdir(parents=True, exist_ok=True)
    vocab_file = tok_dir / "vocab.json"
    with open(vocab_file, "w", encoding="utf-8") as f:
        json.dump(vocab_dict, f)

    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_file=str(vocab_file),
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="",  # Important for character-level models
        replace_word_delimiter_char=False,
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=37, sampling_rate=100, padding_value=0.0, do_normalize=False
    )
    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor, tokenizer=tokenizer
    )
    return processor


def compute_metrics_builder(processor: Wav2Vec2Processor):
    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        label_ids = pred.label_ids.copy()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.tokenizer.batch_decode(label_ids, group_tokens=False)

        cer = jiwer.cer(label_str, pred_str)
        return {"cer": cer}

    return compute_metrics


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Train a lightweight Transformer+CTC on gesture features"
    )

    # Data args
    parser.add_argument(
        "--train-manifest", type=str, default=DEFAULT_CONFIG["data"]["train_manifest"]
    )
    parser.add_argument(
        "--val-manifest", type=str, default=DEFAULT_CONFIG["data"]["val_manifest"]
    )
    parser.add_argument("--output-dir", type=str, default=DEFAULT_CONFIG["model_name"])

    # Model architecture args
    parser.add_argument(
        "--num-layers",
        type=int,
        default=DEFAULT_CONFIG["model_arch"]["num_hidden_layers"],
    )
    parser.add_argument(
        "--hidden-size", type=int, default=DEFAULT_CONFIG["model_arch"]["hidden_size"]
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=DEFAULT_CONFIG["model_arch"]["num_attention_heads"],
    )
    parser.add_argument(
        "--ffn-size", type=int, default=DEFAULT_CONFIG["model_arch"]["ffn_dim"]
    )

    # Training args
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_CONFIG["training"]["batch_size"]
    )
    parser.add_argument(
        "--epochs", type=int, default=DEFAULT_CONFIG["training"]["max_epochs"]
    )
    parser.add_argument(
        "--lr", type=float, default=DEFAULT_CONFIG["training"]["learning_rate"]
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=DEFAULT_CONFIG["training"]["warmup_steps"]
    )
    parser.add_argument("--no-fp16", action="store_true", help="Disable fp16 training")
    parser.add_argument(
        "--no-auto-resume",
        action="store_true",
        help="Do not auto-resume from latest checkpoint",
    )
    parser.add_argument("--subset-train", type=int, default=0, help="Use only first N training samples")
    parser.add_argument("--subset-val", type=int, default=0, help="Use only first N validation samples")
    parser.add_argument("--fast-test", action="store_true", help="Enable small subset + 1 epoch quick test")
    parser.add_argument("--dry-run-first-batch", action="store_true", help="Build 1 batch, run forward+loss, then exit")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup processor (tokenizer + feature extractor)
    logger.info("Setting up tokenizer and processor...")
    processor = build_tokenizer_and_processor(
        DEFAULT_CONFIG["data"]["vocab_chars"], output_dir
    )
    featurizer = PersonalizedSwipeFeaturizer()

    # Load and preprocess datasets
    logger.info("Loading and featurizing datasets...")
    train_data = load_data_from_manifest(args.train_manifest)
    val_data = load_data_from_manifest(args.val_manifest)
    train_ds = Dataset.from_list(train_data)
    val_ds = Dataset.from_list(val_data)

    allowed_chars = set(DEFAULT_CONFIG["data"]["vocab_chars"])

    def preprocess_function(batch):
        norm_points = [
            {"x": p.get("x", 0.0), "y": p.get("y", 0.0), "t": p.get("t", 0.0)}
            for p in batch["points"]
        ]
        batch["input_values"] = featurizer(norm_points).astype(np.float32)

        clean_word = "".join(c for c in batch["word"].lower() if c in allowed_chars)
        batch["labels"] = processor.tokenizer(clean_word).input_ids
        batch["input_length"] = len(batch["input_values"])
        return batch

    # Optional subsetting before map for speed
    if (args.fast_test or args.subset_train) and len(train_ds) > 0:
        n = args.subset_train or 2000
        train_ds = train_ds.select(range(min(n, len(train_ds))))
    if (args.fast_test or args.subset_val) and len(val_ds) > 0:
        n = args.subset_val or 400
        val_ds = val_ds.select(range(min(n, len(val_ds))))

    train_ds = train_ds.map(preprocess_function, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(preprocess_function, remove_columns=val_ds.column_names)
    logger.info(
        f"Training samples: {len(train_ds)} | Validation samples: {len(val_ds)}"
    )

    # Model Initialization (from scratch)
    logger.info("Initializing model from scratch with custom architecture...")
    config = Wav2Vec2Config(
        vocab_size=len(processor.tokenizer),
        pad_token_id=processor.tokenizer.pad_token_id,
        num_hidden_layers=args.num_layers,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_heads,
        intermediate_size=args.ffn_size,
        feat_proj_dropout=DEFAULT_CONFIG["model_arch"]["dropout"],
        layerdrop=DEFAULT_CONFIG["model_arch"]["dropout"],
        ctc_loss_reduction="mean",
        ctc_zero_infinity=True,
    )

    model = GestureTransformerForCTC(config)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Model architecture configured. Total trainable parameters: {num_params / 1e6:.2f}M"
    )

    # Fast test overrides
    if args.fast_test:
        args.epochs = 1
        if args.batch_size > 16:
            args.batch_size = 16
        args.no_auto_resume = True

    # Training Setup
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
        max_grad_norm=1.0,  # Gradient clipping for stability
        save_total_limit=DEFAULT_CONFIG["training"]["save_total_limit"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        logging_steps=DEFAULT_CONFIG["training"]["logging_steps"],
        report_to=["tensorboard"],
    )

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    compute_metrics = compute_metrics_builder(processor)

    trainer = CustomTrainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    if args.dry_run_first_batch:
        from torch.utils.data import DataLoader
        logger.info("Dry-running first batch...")
        dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, collate_fn=data_collator, num_workers=0)
        batch = next(iter(dl))
        for k, v in batch.items():
            if hasattr(v, "shape"):
                logger.info(f"[DRY] {k}.shape={tuple(v.shape)} {v.dtype}")
        model.eval()
        with torch.no_grad():
            outputs = model(input_values=batch["input_values"], attention_mask=(batch.get("attention_mask") if hasattr(batch, "get") else None))
            logits = outputs.logits
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1).transpose(0, 1)
            am = batch.get("attention_mask") if hasattr(batch, "get") else None
            if am is not None:
                input_lengths = am.sum(dim=1).to(dtype=torch.long)
            else:
                input_lengths = torch.full((logits.size(0),), logits.size(1), dtype=torch.long)
            labels = batch["labels"]
            target_lengths = (labels != -100).sum(-1)
            targets = labels.masked_fill(labels == -100, model.config.pad_token_id)
            loss_fct = torch.nn.CTCLoss(blank=model.config.pad_token_id, zero_infinity=True)
            loss = loss_fct(log_probs, targets, input_lengths, target_lengths)
        logger.info(f"[DRY] loss={float(loss)}")
        return

    # Auto-resume from latest checkpoint if available
    resume_from_checkpoint = None
    if not args.no_auto_resume:
        ckpts = sorted(
            output_dir.glob("checkpoint-*"), key=os.path.getmtime, reverse=True
        )
        if ckpts:
            resume_from_checkpoint = str(ckpts[0])
            logger.info(f"Auto-resuming from: {resume_from_checkpoint}")

    # Start Training
    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    best_dir = output_dir / "best_model"
    best_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(best_dir))
    processor.save_pretrained(str(best_dir))
    logger.info(f"Best model and processor saved to {best_dir}")


if __name__ == "__main__":
    main()
