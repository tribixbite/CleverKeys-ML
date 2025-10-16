#!/usr/bin/env python3
"""Evaluate an RNNT checkpoint on custom manifest subsets with configurable beam size."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import sys
import torch
import editdistance  # type: ignore
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "trained_models" / "nema1"
sys.path.append(str(MODEL_ROOT))

from train_transducer_personalized import (  # type: ignore  # noqa: E402
    CONFIG as TRAIN_CONFIG,
    PersonalizedRNNTModel,
    PersonalizedSwipeDataset,
    build_model_config,
    load_vocab,
    collate_fn,
)


def load_checkpoint_model(checkpoint: Path, device: torch.device) -> Tuple[PersonalizedRNNTModel, Dict]:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg_dict = ckpt.get("hyper_parameters", {}).get("cfg")
    if cfg_dict is None:
        raise ValueError("Checkpoint missing hyper_parameters.cfg")
    base_cfg = OmegaConf.create(TRAIN_CONFIG)
    cfg = OmegaConf.merge(base_cfg, OmegaConf.create(cfg_dict))
    OmegaConf.set_struct(cfg, False)

    vocab_path = (MODEL_ROOT / Path(cfg.data.vocab_path)).resolve()
    vocab = load_vocab(str(vocab_path))
    model_cfg = build_model_config(cfg, list(vocab.keys()))
    model = PersonalizedRNNTModel(
        cfg=model_cfg,
        kd_lambda=0.0,
        kd_temperature=1.0,
        teacher_checkpoint=None,
    )
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.to(device)
    model.eval()
    return model, cfg


def decode_batch(
    model: PersonalizedRNNTModel,
    batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
) -> Tuple[List[str], List[str]]:
    features, feature_lengths, tokens, token_lengths = batch
    features = features.to(device)
    feature_lengths = feature_lengths.to(device)
    tokens = tokens.to(device)
    token_lengths = token_lengths.to(device)

    with torch.no_grad():
        encoded, encoded_len = model.forward(input_signal=features, input_signal_length=feature_lengths)
        hypotheses = model.decoding.rnnt_decoder_predictions_tensor(encoded, encoded_len)

    refs: List[str] = []
    for seq, length in zip(tokens, token_lengths):
        token_list = seq[: int(length.item())].detach().cpu().numpy().tolist()
        refs.append(model.decoding.decode_tokens_to_str(token_list))

    preds: List[str] = []
    for hyp in hypotheses:
        if isinstance(hyp, list):
            hyp = hyp[0]
        text = hyp.text if hasattr(hyp, "text") else str(hyp)
        preds.append(text)

    return preds, refs


def compute_wer(preds: List[str], refs: List[str]) -> Tuple[int, int, List[Tuple[str, str]]]:
    errors = 0
    words = 0
    mismatches: List[Tuple[str, str]] = []
    for pred, ref in zip(preds, refs):
        ref_tokens = ref.split()
        pred_tokens = pred.split()
        words += len(ref_tokens)
        err = editdistance.eval(pred_tokens, ref_tokens)
        errors += err
        if pred != ref and len(mismatches) < 50:
            mismatches.append((ref, pred))
    return errors, words, mismatches


def set_beam_strategy(model: PersonalizedRNNTModel, beam_size: int) -> None:
    decoding_container = OmegaConf.to_container(model.cfg.decoding, resolve=True)
    decoding_cfg = OmegaConf.create(decoding_container)
    OmegaConf.set_struct(decoding_cfg, False)
    decoding_cfg.strategy = "beam"
    if "beam" not in decoding_cfg or decoding_cfg.beam is None:
        decoding_cfg.beam = {}
    decoding_cfg.beam["beam_size"] = beam_size
    model.change_decoding_strategy(decoding_cfg)


def evaluate_manifest(
    model: PersonalizedRNNTModel,
    manifest: Path,
    cfg,
    beam_size: int,
    batch_size: int,
    device: torch.device,
    max_batches: int | None = None,
) -> Dict[str, object]:
    dataset = PersonalizedSwipeDataset(
        manifest_path=str(manifest),
        vocab=load_vocab(str((MODEL_ROOT / Path(cfg.data.vocab_path)).resolve())),
        max_trace_len=cfg.data.max_trace_len,
        preprocess_cfg=cfg.preprocess,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    set_beam_strategy(model, beam_size)

    total_errors = 0
    total_words = 0
    samples = 0
    example_mismatches: List[Tuple[str, str]] = []

    for batch_idx, batch in enumerate(loader):
        preds, refs = decode_batch(model, batch, device)
        errors, words, mismatches = compute_wer(preds, refs)
        total_errors += errors
        total_words += words
        samples += len(refs)
        if len(example_mismatches) < 20:
            example_mismatches.extend(mismatches)
        if max_batches is not None and batch_idx + 1 >= max_batches:
            break

    return {
        "manifest": str(manifest),
        "beam_size": beam_size,
        "samples": samples,
        "total_words": total_words,
        "errors": total_errors,
        "wer": total_errors / max(total_words, 1),
        "example_mismatches": example_mismatches[:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RNNT checkpoint with different beam sizes")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--beam-sizes", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-batches", type=int)
    args = parser.parse_args()

    device = torch.device(args.device)
    model, cfg = load_checkpoint_model(args.checkpoint, device)

    results = []
    for manifest in args.manifests:
        for beam in args.beam_sizes:
            res = evaluate_manifest(
                model,
                manifest,
                cfg,
                beam_size=beam,
                batch_size=args.batch_size,
                device=device,
                max_batches=args.max_batches,
            )
            print(
                f"{manifest.name}: beam={beam} -> WER={res['wer']:.4f} "
                f"(err={res['errors']}, words={res['total_words']}, samples={res['samples']})"
            )
            results.append(res)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
