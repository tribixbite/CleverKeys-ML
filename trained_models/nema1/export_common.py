"""Shared utilities for CleverKeys export scripts."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Sequence, Tuple

import torch
from shutil import copy2

from model_class import GestureRNNTModel, get_default_config


def _register_dequantize_per_channel_out_variant() -> None:
    """Ensure ExecuTorch can convert quantized_decomposed::dequantize_per_channel to an out variant."""
    import torch

    schema = (
        "dequantize_per_channel.out("
        "Tensor input, Tensor scales, Tensor? zero_points, int axis, int quant_min, int quant_max, "
        "ScalarType dtype, *, ScalarType? out_dtype=None, Tensor(a!) out) -> Tensor(a!)"
    )

    try:
        lib_def = torch.library.Library("quantized_decomposed", "FRAGMENT")
        lib_def.define(schema)
    except RuntimeError:
        # schema already defined
        pass

    meta_lib = torch.library.Library("quantized_decomposed", "IMPL", "Meta")

    def _meta(input, scales, zero_points, axis, quant_min, quant_max, dtype, *, out_dtype=None, out=None):
        return out

    meta_lib.impl("dequantize_per_channel.out", _meta)  # type: ignore[misc]

    impl_lib = torch.library.Library("quantized_decomposed", "IMPL", "CPU")

    def _impl(input, scales, zero_points, axis, quant_min, quant_max, dtype, *, out_dtype=None, out=None):
        result = torch.ops.quantized_decomposed.dequantize_per_channel.default(
            input,
            scales,
            zero_points,
            axis,
            quant_min,
            quant_max,
            dtype,
            out_dtype=out_dtype,
        )
        out.copy_(result)
        return out

    impl_lib.impl("dequantize_per_channel.out", _impl)  # type: ignore[misc]


_register_dequantize_per_channel_out_variant()
from swipe_data_utils import KeyboardGrid, SwipeDataset, SwipeFeaturizer, collate_fn

LOG = logging.getLogger("export_common")


def load_trained_model(checkpoint_path: str) -> GestureRNNTModel:
    """Load a trained RNNT model from a .ckpt or .nemo artifact."""
    checkpoint_path = os.path.expanduser(checkpoint_path)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)

    if checkpoint_path.endswith(".ckpt"):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("hyper_parameters", {}).get("cfg", get_default_config())
        model = GestureRNNTModel(cfg).eval()
        state_dict = ckpt["state_dict"]
        cleaned = {}
        for key, value in state_dict.items():
            if key.startswith("encoder._orig_mod."):
                cleaned[key.replace("encoder._orig_mod.", "encoder.")] = value
            else:
                cleaned[key] = value
        model.load_state_dict(cleaned, strict=False)
        return model.eval()

    LOG.info("Loading NeMo archive: %s", checkpoint_path)
    model = GestureRNNTModel.restore_from(checkpoint_path, map_location="cpu").eval()
    return model


def load_vocab(vocab_path: str) -> Dict[str, int]:
    vocab: Dict[str, int] = {}
    with open(vocab_path, "r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            token = line.strip()
            if token:
                vocab[token] = idx
    return vocab


def create_featurizer(chars: str = "abcdefghijklmnopqrstuvwxyz'") -> SwipeFeaturizer:
    grid = KeyboardGrid(chars=chars)
    return SwipeFeaturizer(grid)


def ensure_default_manifest(path: Optional[str], fallback: str) -> str:
    if path:
        return path
    resolved = Path(fallback)
    if not resolved.exists():
        raise FileNotFoundError(f"No manifest provided and default missing: {fallback}")
    return str(resolved)


def create_calibration_loader(
    manifest_path: str,
    vocab_path: str,
    max_trace_len: int,
    batch_size: int,
    *,
    num_workers: int = 0,
    shuffle: bool = False,
) -> torch.utils.data.DataLoader:
    featurizer = create_featurizer()
    vocab = load_vocab(vocab_path)
    dataset = SwipeDataset(manifest_path, featurizer, vocab, max_trace_len)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )


def batch_to_encoder_inputs(batch) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, dict):
        feats = batch["features"]
        lens = batch.get("feat_lens") or batch.get("lengths")
    else:
        feats, lens = batch[0], batch[1]
    feats_bft = feats.transpose(1, 2).contiguous()
    lens = lens.to(dtype=torch.int32, copy=False)
    return feats_bft, lens


def iter_calibration_batches(
    dataloader: torch.utils.data.DataLoader,
    max_batches: Optional[int] = None,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    count = 0
    for batch in dataloader:
        if max_batches is not None and count >= max_batches:
            break
        yield batch_to_encoder_inputs(batch)
        count += 1


class DatasetBackedCalibrationReader:
    """Minimal CalibrationDataReader for ONNX Runtime."""

    def __init__(self, dataloader: torch.utils.data.DataLoader, max_batches: Optional[int] = None):
        self.dataloader = dataloader
        self.max_batches = max_batches
        self.reset()

    def reset(self) -> None:
        self._iterator = iter(self.dataloader)
        self._count = 0

    def rewind(self) -> None:  # ONNX Runtime expects this name
        self.reset()

    def get_next(self):  # ONNX Runtime naming
        if self.max_batches is not None and self._count >= self.max_batches:
            return None
        try:
            batch = next(self._iterator)
        except StopIteration:
            return None
        self._count += 1
        feats_bft, lens = batch_to_encoder_inputs(batch)
        return {
            "features_bft": feats_bft.cpu().numpy(),
            "lengths": lens.cpu().numpy(),
        }


def make_example_inputs(max_length: int, feature_dim: int = 37, *, dtype: torch.dtype = torch.float32):
    B = 1
    feats = torch.randn(B, feature_dim, max_length, dtype=dtype)
    lens = torch.tensor([max_length], dtype=torch.int32)
    return feats, lens


def package_artifacts(
    outputs: Sequence[str],
    package_dir: str,
    *,
    extra_files: Sequence[str] | None = None,
) -> None:
    """Copy exported files and optional assets into a deployment directory."""

    dest = Path(package_dir)
    dest.mkdir(parents=True, exist_ok=True)

    for src_path in list(outputs) + list(extra_files or []):
        if not src_path:
            continue
        src = Path(src_path)
        if not src.exists():
            LOG.warning("Skipping missing asset: %s", src)
            continue
        target = dest / src.name
        if target.resolve() == src.resolve():
            LOG.debug("Asset already in destination: %s", src)
            continue
        copy2(src, target)
        LOG.info("Packaged %s -> %s", src, target)
