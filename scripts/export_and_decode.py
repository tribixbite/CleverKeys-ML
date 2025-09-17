#!/usr/bin/env python3
"""Export an RNNT checkpoint and run beam decoding evaluation."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPORT_DIR = REPO_ROOT / "trained_models" / "nema1"


def run(cmd, cwd: Path | None = None) -> None:
    print(f"→ {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def export_models(checkpoint: Path, workdir: Path, vocab: Path) -> tuple[Path, Path]:
    encoder_path = workdir / "encoder_fp32.onnx"
    decoder_path = workdir / "decoder_fp32.onnx"

    run([
        "python",
        str(EXPORT_DIR / "export_onnx.py"),
        "--checkpoint",
        str(checkpoint),
        "--vocab",
        str(vocab),
        "--fp32-output",
        str(encoder_path),
    ])

    run([
        "python",
        str(EXPORT_DIR / "export_rnnt_step.py"),
        "--checkpoint",
        str(checkpoint),
        "--vocab",
        str(vocab),
        "--onnx_out",
        str(decoder_path),
        "--pte_out",
        str(workdir / "decoder_tmp.pte"),
    ])

    return encoder_path, decoder_path


def run_beam_decode(encoder: Path, decoder: Path, manifest: Path, beam_size: int) -> None:
    run([
        "python",
        str(EXPORT_DIR / "beam_decode_onnx_cli.py"),
        "--encoder",
        str(encoder),
        "--decoder",
        str(decoder),
        "--manifest",
        str(manifest),
        "--beam-size",
        str(beam_size),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Export checkpoint and evaluate via beam decoding")
    parser.add_argument("checkpoint", type=Path, help="Path to RNNT checkpoint (.ckpt or .nemo)")
    parser.add_argument("manifest", type=Path, help="Manifest of samples to decode")
    parser.add_argument("--vocab", type=Path, default=Path("vocab/vocab.txt"), help="Vocabulary file")
    parser.add_argument("--work-dir", type=Path, help="Directory to place exports (defaults to temp dir)")
    parser.add_argument("--beam-size", type=int, default=8)
    parser.add_argument("--keep", action="store_true", help="Keep export artifacts (requires --work-dir)")
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    manifest = args.manifest.resolve()
    vocab = args.vocab.resolve()

    if args.work_dir:
        workdir = args.work_dir.resolve()
        workdir.mkdir(parents=True, exist_ok=True)
        encoder_path, decoder_path = export_models(checkpoint, workdir, vocab)
        run_beam_decode(encoder_path, decoder_path, manifest, args.beam_size)
        if not args.keep:
            (workdir / "decoder_tmp.pte").unlink(missing_ok=True)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            encoder_path, decoder_path = export_models(checkpoint, workdir, vocab)
            run_beam_decode(encoder_path, decoder_path, manifest, args.beam_size)


if __name__ == "__main__":
    main()
