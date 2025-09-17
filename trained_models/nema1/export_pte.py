#!/usr/bin/env python3
"""Quantize RNNT encoder to ExecuTorch (.pte) with PT2E + XNNPACK."""

import argparse
import logging

import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)
from torch.ao.quantization import allow_exported_model_train_eval

from export_common import (
    create_calibration_loader,
    ensure_default_manifest,
    iter_calibration_batches,
    load_trained_model,
    make_example_inputs,
    package_artifacts,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOG = logging.getLogger("export_pte_quant")


class EncoderWrapper(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module):
        super().__init__()
        self.encoder = encoder.eval()

    def forward(self, feats_bft: torch.Tensor, lengths: torch.Tensor):
        return self.encoder(audio_signal=feats_bft, length=lengths)


@torch.no_grad()
def calibrate_encoder(
    prepared_encoder: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    max_batches: int,
) -> None:
    LOG.info("Calibrating encoder (max %d batches)", max_batches)
    try:
        prepared_encoder.eval()
    except NotImplementedError:
        from torch.ao.quantization import move_exported_model_to_eval

        prepared_encoder = move_exported_model_to_eval(prepared_encoder)

    for idx, (feats_bft, lens) in enumerate(iter_calibration_batches(dataloader, max_batches)):
        prepared_encoder(feats_bft, lens)
        if (idx + 1) % 10 == 0:
            LOG.info("  Calibrated %d batches", idx + 1)
    LOG.info("Calibration complete")


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize encoder to ExecuTorch")
    parser.add_argument("--checkpoint", required=True, help="Path to trained .ckpt or .nemo artifact")
    parser.add_argument("--vocab", default="vocab/final_vocab.txt", help="Vocabulary file for calibration")
    parser.add_argument(
        "--calib-manifest",
        help="Calibration manifest (.jsonl). Defaults to data/train_final_val.jsonl if present.",
    )
    parser.add_argument("--output", default="encoder_quant_xnnpack.pte", help="Output ExecuTorch file")
    parser.add_argument("--calib-batches", type=int, default=64, help="Calibration batches to run")
    parser.add_argument("--calib-batch-size", type=int, default=16, help="Calibration batch size")
    parser.add_argument("--max-trace-len", type=int, default=200, help="Max gesture length for export inputs")
    parser.add_argument("--lexicon", help="Optional lexicon to bundle with output")
    parser.add_argument("--package-dir", help="Copy exported file (and assets) into this directory")
    parser.add_argument(
        "--bundle-assets",
        nargs="*",
        default=None,
        help="Additional metadata files to include when packaging",
    )
    args = parser.parse_args()

    model = load_trained_model(args.checkpoint)
    wrapper = EncoderWrapper(model.encoder)

    example_feats, example_lens = make_example_inputs(args.max_trace_len)

    LOG.info("Exporting model with torch.export")
    exported = torch.export.export(wrapper, (example_feats, example_lens))

    quantizer = XNNPACKQuantizer()
    quantizer.set_global(get_symmetric_quantization_config(is_per_channel=True))
    LOG.info("Preparing PT2E module")
    prepared = prepare_pt2e(exported.module(), quantizer)

    manifest = ensure_default_manifest(args.calib_manifest, "data/train_final_val.jsonl")
    dataloader = create_calibration_loader(
        manifest,
        args.vocab,
        args.max_trace_len,
        batch_size=args.calib_batch_size,
        shuffle=False,
        num_workers=0,
    )

    calibrate_encoder(prepared, dataloader, args.calib_batches)

    LOG.info("Converting to quantized encoder")
    quant_encoder = convert_pt2e(prepared)
    if quant_encoder is None:
        quant_encoder = prepared
    LOG.info("Converted encoder type: %s", type(quant_encoder))
    allow_exported_model_train_eval(quant_encoder)
    quant_encoder.eval()

    LOG.info("Lowering quantized encoder to ExecuTorch")
    exported_quant = torch.export.export(quant_encoder, (example_feats, example_lens))
    edge = to_edge(exported_quant)
    edge = edge.to_backend(XnnpackPartitioner())
    program = edge.to_executorch()

    buffer = getattr(program, "buffer", None)
    if buffer is None and hasattr(program, "to_buffer"):
        buffer = program.to_buffer()
    if buffer is None:
        raise RuntimeError("Unable to obtain ExecuTorch program buffer")

    with open(args.output, "wb") as handle:
        handle.write(buffer)
    LOG.info("✓ Saved quantized ExecuTorch encoder -> %s", args.output)
    LOG.info("Decoder + joint remain float for best quality; export separately if needed.")

    if args.package_dir:
        extras = list(args.bundle_assets or [])
        if args.lexicon:
            extras.append(args.lexicon)
        package_artifacts([args.output], args.package_dir, extra_files=extras)


if __name__ == "__main__":
    main()
