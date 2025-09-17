#!/usr/bin/env python3
"""Improved full-data training script with shared preprocessing.

Implements architecture/documentation recommendations:
- Shared feature extractor used for both training and inference.
- Configurable model hyper-parameters with higher capacity.
- Automatic dataset fallback handling and export helpers.
- Optional mobile-friendly ONNX export with int32-compatible wrapper.
- Shape sanity checks to catch export regressions early.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Local snapshot modules
from swipe_feature_pipeline import (
    CharTokenizer,
    KeyboardGeometry,
    SwipeFeatureExtractor,
)

# Import model definition from original module copy for continuity
from train_character_model import CharacterLevelSwipeModel


# ---------------------------------------------------------------------------
def resolve_dataset_paths(root: Path) -> Tuple[Path, Path, Path]:
    """Return train/val/test dataset paths with graceful fallbacks."""
    primary = root / "data" / "combined_dataset"
    train_path = primary / "cleaned_english_swipes_train.jsonl"
    val_path = primary / "cleaned_english_swipes_val.jsonl"
    test_path = primary / "cleaned_english_swipes_test.jsonl"

    if not train_path.exists():
        # Fall back to raw converted dataset if cleaned train not available.
        alt_train = root / "data" / "raw_converted_english_swipes_train.jsonl"
        if alt_train.exists():
            train_path = alt_train
        else:
            raise FileNotFoundError(
                "Could not locate training dataset. Expected cleaned or raw conversion JSONL."
            )

    return train_path, val_path, test_path


# ---------------------------------------------------------------------------
class SwipeDatasetV2(Dataset):
    """Dataset that leverages the shared feature extractor."""

    def __init__(
        self,
        data_path: Path,
        extractor: SwipeFeatureExtractor,
        max_word_len: int = 20,
        max_samples: Optional[int] = None,
        use_time_stamps: bool = True,
        jitter_prob: float = 0.15,
        geom_noise_sigma: float = 0.75,
    ) -> None:
        self.data_path = data_path
        self.extractor = extractor
        self.max_word_len = max_word_len
        self.max_samples = max_samples
        self.use_time_stamps = use_time_stamps
        self.jitter_prob = jitter_prob
        self.geom_noise_sigma = geom_noise_sigma

        self.examples: List[Dict] = []
        with data_path.open("r", encoding="utf-8") as fp:
            for line in fp:
                sample = json.loads(line)
                if "curve" in sample and "word" in sample:
                    curve = sample["curve"]
                    xs = curve.get("x", [])
                    ys = curve.get("y", [])
                    ts = curve.get("t") if self.use_time_stamps else None
                elif "word_seq" in sample:
                    seq = sample["word_seq"]
                    xs = seq.get("x", [])
                    ys = seq.get("y", [])
                    ts = seq.get("time") if self.use_time_stamps else None
                else:
                    xs = sample.get("x", [])
                    ys = sample.get("y", [])
                    ts = sample.get("t") if self.use_time_stamps else None

                word = sample.get("word") or sample.get("target") or ""
                if not word:
                    continue

                self.examples.append({
                    "x": xs,
                    "y": ys,
                    "t": ts,
                    "word": word,
                })
                if self.max_samples and len(self.examples) >= self.max_samples:
                    break

        if not self.examples:
            raise RuntimeError(f"No swipe samples could be loaded from {data_path}")

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.examples)

    # ------------------------------------------------------------------
    def _apply_jitter(self, xs: List[float], ys: List[float]) -> Tuple[List[float], List[float]]:
        if self.geom_noise_sigma <= 0.0 or torch.rand(1).item() > self.jitter_prob:
            return xs, ys
        jitter_x = torch.normal(0.0, self.geom_noise_sigma, size=(len(xs),)).numpy()
        jitter_y = torch.normal(0.0, self.geom_noise_sigma, size=(len(ys),)).numpy()
        xs = (torch.tensor(xs) + jitter_x).tolist()
        ys = (torch.tensor(ys) + jitter_y).tolist()
        return xs, ys

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.examples[idx]
        xs, ys = sample["x"], sample["y"]
        ts = sample["t"] if self.use_time_stamps else None
        xs, ys = self._apply_jitter(xs, ys)

        features, nearest_keys, seq_len = self.extractor.to_tensors(xs, ys, ts)
        target = self.extractor.encode_word(sample["word"], max_word_len=self.max_word_len)

        return {
            "traj_features": features,
            "nearest_keys": nearest_keys,
            "target": target,
            "seq_len": torch.tensor(seq_len, dtype=torch.long),
            "word": sample["word"],
        }


# ---------------------------------------------------------------------------
def make_src_mask(seq_lens: torch.Tensor, max_len: int) -> torch.Tensor:
    mask = torch.ones((seq_lens.size(0), max_len), dtype=torch.bool, device=seq_lens.device)
    for i, length in enumerate(seq_lens.tolist()):
        mask[i, : length] = False
    return mask


# ---------------------------------------------------------------------------
def evaluate_word_accuracy(
    model: CharacterLevelSwipeModel,
    dataloader: DataLoader,
    tokenizer: CharTokenizer,
    device: torch.device,
    beam_size: int = 8,
) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            traj_features = batch["traj_features"].to(device)
            nearest_keys = batch["nearest_keys"].to(device)
            seq_lens = batch["seq_len"].to(device)
            words = batch["word"]

            src_mask = make_src_mask(seq_lens, traj_features.size(1))
            predictions = model.generate_beam(
                traj_features,
                nearest_keys,
                tokenizer,
                src_mask=src_mask,
                beam_size=beam_size,
                max_len=32,
            )
            for pred, true_word in zip(predictions, words):
                total += 1
                if pred == true_word:
                    correct += 1
    return correct / max(total, 1)


# ---------------------------------------------------------------------------
def export_mobile_ready_onnx(
    model: CharacterLevelSwipeModel,
    output_dir: Path,
    max_seq_len: int = 150,
    vocab_size: int = 30,
) -> None:
    """Export encoder+decoder ONNX graphs with int32-friendly inputs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    class EncoderWrapper(torch.nn.Module):
        def __init__(self, base_model: CharacterLevelSwipeModel) -> None:
            super().__init__()
            self.model = base_model

        def forward(self, traj_features: torch.Tensor, nearest_keys: torch.Tensor, src_mask: torch.Tensor):
            if nearest_keys.dtype != torch.long:
                nearest_keys = nearest_keys.to(torch.long)
            if src_mask.dtype != torch.bool:
                src_mask = src_mask.to(torch.bool)
            return self.model.encode_trajectory(traj_features, nearest_keys, src_mask)

    class DecoderWrapper(torch.nn.Module):
        def __init__(self, base_model: CharacterLevelSwipeModel, max_steps: int = 32) -> None:
            super().__init__()
            self.model = base_model
            self.max_steps = max_steps

        def forward(
            self,
            memory: torch.Tensor,
            target_tokens: torch.Tensor,
            src_mask: torch.Tensor,
            target_mask: torch.Tensor,
        ) -> torch.Tensor:
            if target_tokens.dtype != torch.long:
                target_tokens = target_tokens.to(torch.long)
            if src_mask.dtype != torch.bool:
                src_mask = src_mask.to(torch.bool)
            if target_mask.dtype != torch.bool:
                target_mask = target_mask.to(torch.bool)
            batch, tgt_len = target_tokens.shape
            embeddings = self.model.char_embedding(target_tokens) * math.sqrt(self.model.d_model)
            embeddings = embeddings + self.model.pe[:, :tgt_len, :]
            causal = torch.triu(torch.ones(tgt_len, tgt_len, device=embeddings.device), diagonal=1).bool()
            decoded = self.model.decoder(
                embeddings,
                memory,
                tgt_mask=causal,
                memory_key_padding_mask=src_mask,
                tgt_key_padding_mask=target_mask,
            )
            return self.model.output_proj(decoded)

    encoder = EncoderWrapper(model).eval()
    decoder = DecoderWrapper(model).eval()

    dummy_traj = torch.randn(1, max_seq_len, 6)
    dummy_keys = torch.randint(0, vocab_size, (1, max_seq_len), dtype=torch.int32)
    dummy_mask = torch.zeros(1, max_seq_len, dtype=torch.bool)

    encoder_path = output_dir / "swipe_encoder_mobile.onnx"
    torch.onnx.export(
        encoder,
        (dummy_traj, dummy_keys, dummy_mask),
        encoder_path,
        input_names=["trajectory_features", "nearest_keys", "src_mask"],
        output_names=["memory"],
        opset_version=14,
        dynamic_axes={
            "trajectory_features": {0: "batch", 1: "seq"},
            "nearest_keys": {0: "batch", 1: "seq"},
            "src_mask": {0: "batch", 1: "seq"},
            "memory": {0: "batch", 1: "seq"},
        },
    )

    memory = torch.randn(1, max_seq_len, model.d_model)
    target = torch.zeros(1, 32, dtype=torch.int32)
    src_mask = torch.zeros(1, max_seq_len, dtype=torch.bool)
    tgt_mask = torch.zeros(1, 32, dtype=torch.bool)

    decoder_path = output_dir / "swipe_decoder_mobile.onnx"
    torch.onnx.export(
        decoder,
        (memory, target, src_mask, tgt_mask),
        decoder_path,
        input_names=["memory", "target_tokens", "src_mask", "target_mask"],
        output_names=["logits"],
        opset_version=14,
        dynamic_axes={
            "memory": {0: "batch", 1: "enc_seq"},
            "target_tokens": {0: "batch", 1: "dec_seq"},
            "src_mask": {0: "batch", 1: "enc_seq"},
            "target_mask": {0: "batch", 1: "dec_seq"},
            "logits": {0: "batch", 1: "dec_seq"},
        },
    )


# ---------------------------------------------------------------------------
def sanity_check_shapes(model: CharacterLevelSwipeModel, tokenizer: CharTokenizer) -> None:
    model.eval()
    device = next(model.parameters()).device
    batch = 2
    seq_len = 150
    word_len = 20

    traj = torch.randn(batch, seq_len, 6, device=device)
    nearest = torch.randint(0, tokenizer.vocab_size, (batch, seq_len), device=device)
    targets = torch.randint(0, tokenizer.vocab_size, (batch, word_len), device=device)
    src_mask = torch.zeros(batch, seq_len, dtype=torch.bool, device=device)
    tgt_mask = torch.zeros(batch, word_len - 1, dtype=torch.bool, device=device)

    logits = model(traj, nearest, targets, src_mask=src_mask, tgt_mask=tgt_mask)
    assert logits.shape == (batch, word_len - 1, tokenizer.vocab_size), "Unexpected logits shape"


# ---------------------------------------------------------------------------
def train(args: argparse.Namespace) -> None:
    root = Path(args.project_root).resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = CharTokenizer()
    keyboard = KeyboardGeometry()
    extractor = SwipeFeatureExtractor(
        keyboard=keyboard,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        velocity_scale=args.velocity_scale,
        acceleration_scale=args.acceleration_scale,
        blend_uniform_dt_prob=args.blend_uniform_dt_prob,
    )

    train_path, val_path, test_path = resolve_dataset_paths(root)
    print(f"Train dataset: {train_path}")
    print(f"Val dataset:   {val_path}")
    print(f"Test dataset:  {test_path}")

    train_dataset = SwipeDatasetV2(
        train_path,
        extractor,
        max_word_len=args.max_word_len,
        max_samples=args.max_train_samples,
        use_time_stamps=not args.ignore_timestamps,
    )
    val_dataset = SwipeDatasetV2(
        val_path,
        extractor,
        max_word_len=args.max_word_len,
        use_time_stamps=not args.ignore_timestamps,
    )
    test_dataset = SwipeDatasetV2(
        test_path,
        extractor,
        max_word_len=args.max_word_len,
        use_time_stamps=not args.ignore_timestamps,
    )

    def collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        traj = torch.stack([sample["traj_features"] for sample in batch])
        nearest = torch.stack([sample["nearest_keys"] for sample in batch])
        targets = torch.stack([sample["target"] for sample in batch])
        seq_lens = torch.stack([sample["seq_len"] for sample in batch])
        words = [sample["word"] for sample in batch]
        return {
            "traj_features": traj,
            "nearest_keys": nearest,
            "target": targets,
            "seq_len": seq_lens,
            "word": words,
        }

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate,
    )

    model = CharacterLevelSwipeModel(
        traj_dim=6,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.encoder_layers,
        num_decoder_layers=args.decoder_layers,
        dim_feedforward=args.ffn_dim,
        dropout=args.dropout,
        kb_vocab_size=tokenizer.vocab_size,
        char_vocab_size=tokenizer.vocab_size,
        max_seq_len=args.max_seq_len,
    ).to(device)

    sanity_check_shapes(model, tokenizer)

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_idx, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=args.warmup_epochs / max(args.epochs, 1),
        anneal_strategy="cos",
    )

    best_val = 0.0
    patience_counter = 0
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for batch in pbar:
            traj = batch["traj_features"].to(device)
            nearest = batch["nearest_keys"].to(device)
            targets = batch["target"].to(device)
            seq_lens = batch["seq_len"].to(device)

            src_mask = make_src_mask(seq_lens, traj.size(1))
            tgt_pad_mask = targets[:, :-1] == tokenizer.pad_idx

            logits = model(traj, nearest, targets, src_mask=src_mask, tgt_mask=tgt_pad_mask)
            loss = criterion(logits.reshape(-1, tokenizer.vocab_size), targets[:, 1:].reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                mask = targets[:, 1:] != tokenizer.pad_idx
                total_correct += ((preds == targets[:, 1:]) & mask).sum().item()
                total_tokens += mask.sum().item()
                total_loss += loss.item()

            if total_tokens > 0:
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "char_acc": f"{total_correct / total_tokens:.2%}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                })

        train_char_acc = total_correct / max(total_tokens, 1)
        avg_loss = total_loss / max(len(train_loader), 1)

        val_acc = evaluate_word_accuracy(model, val_loader, tokenizer, device, beam_size=args.beam_size)
        print(f"Epoch {epoch+1}: train_loss={avg_loss:.4f}, train_char_acc={train_char_acc:.2%}, val_word_acc={val_acc:.2%}")

        if val_acc > best_val:
            best_val = val_acc
            patience_counter = 0
            ckpt_path = checkpoint_dir / f"swipe_char_v2_epoch{epoch+1:02d}_{val_acc:.3f}.pt"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_word_acc": val_acc,
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

            if val_acc >= args.target_accuracy:
                print(f"Target accuracy {args.target_accuracy:.2%} reached. Evaluating on test set...")
                test_acc = evaluate_word_accuracy(model, test_loader, tokenizer, device, beam_size=args.beam_size)
                print(f"Test word accuracy: {test_acc:.2%}")
                if args.export_dir:
                    export_mobile_ready_onnx(model, Path(args.export_dir))
                break
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping triggered due to no improvement.")
                break

    if best_val < args.target_accuracy:
        print(f"Finished training. Best validation accuracy: {best_val:.2%}")
        if args.export_dir:
            export_mobile_ready_onnx(model, Path(args.export_dir))


# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train improved swipe model with shared preprocessing.")
    parser.add_argument("--project-root", default="..", help="Repository root (default: .. relative to snapshot)")
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--target-accuracy", type=float, default=0.90)
    parser.add_argument("--beam-size", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=150)
    parser.add_argument("--max-word-len", type=int, default=24)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--ffn-dim", type=int, default=1536)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--encoder-layers", type=int, default=8)
    parser.add_argument("--decoder-layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--checkpoint-dir", default="checkpoints/full_character_model_v2")
    parser.add_argument("--export-dir", default="exported_models_v2")
    parser.add_argument("--ignore-timestamps", action="store_true", help="Ignore recorded timestamps to mimic web inference.")
    parser.add_argument("--velocity-scale", type=float, default=1000.0)
    parser.add_argument("--acceleration-scale", type=float, default=500.0)
    parser.add_argument("--blend-uniform-dt-prob", type=float, default=0.2)
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    train(args)
