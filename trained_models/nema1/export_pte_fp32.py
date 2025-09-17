#!/usr/bin/env python3
"""Export non-quantized ExecuTorch (.pte) encoder."""

import argparse
import logging

import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge

from export_common import load_trained_model, make_example_inputs, package_artifacts

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOG = logging.getLogger("export_pte_fp32")


class EncoderWrapper(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module):
        super().__init__()
        self.encoder = encoder.eval()

    def forward(self, feats_bft: torch.Tensor, lengths: torch.Tensor):
        return self.encoder(audio_signal=feats_bft, length=lengths)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Export ExecuTorch FP32 encoder")
    parser.add_argument("--checkpoint", required=True, help="Path to trained .ckpt or .nemo artifact")
    parser.add_argument("--output", default="encoder_fp32.pte", help="Output ExecuTorch file")
    parser.add_argument("--max-trace-len", type=int, default=200, help="Max gesture length for export sample")
    parser.add_argument("--lexicon", help="Optional lexicon to include when packaging outputs")
    parser.add_argument("--package-dir", help="Copy exported file (and assets) into this directory")
    parser.add_argument(
        "--bundle-assets",
        nargs="*",
        default=None,
        help="Additional files (runtime metadata, trie, etc.) to include when packaging",
    )
    args = parser.parse_args()

    model = load_trained_model(args.checkpoint)
    wrapper = EncoderWrapper(model.encoder)

    example_feats, example_lens = make_example_inputs(args.max_trace_len)

    LOG.info("Exporting encoder to ExecuTorch Edge IR")
    exported = torch.export.export(wrapper, (example_feats, example_lens))
    edge = to_edge(exported)
    edge = edge.to_backend(XnnpackPartitioner())
    program = edge.to_executorch()

    buffer = getattr(program, "buffer", None)
    if buffer is None and hasattr(program, "to_buffer"):
        buffer = program.to_buffer()
    if buffer is None:
        raise RuntimeError("Unable to extract ExecuTorch buffer")

    with open(args.output, "wb") as handle:
        handle.write(buffer)
    LOG.info("✓ Saved ExecuTorch encoder -> %s", args.output)

    if args.package_dir:
        extras = list(args.bundle_assets or [])
        if args.lexicon:
            extras.append(args.lexicon)
        package_artifacts([args.output], args.package_dir, extra_files=extras)


if __name__ == "__main__":
    main()
