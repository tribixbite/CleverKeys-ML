#!/usr/bin/env bash
# Phase P6 — promote a script's exports into ctc/artifacts/ and freeze its
# golden fixture at the adopted preset (CKDT, lambda 2.0 — unchanged; P6 moves
# the donor footing and nothing else).
#
# Naming: `<code>_synth_v2full_ch80*`.  The P4 `<code>_synth_v2_ch80*` bytes are
# NOT deleted — they are the bytes every P4 number was measured on and stay in
# the registry as the superseded generation.  ru is the exception in name only:
# `ru_synth_v2_ch80` is ALREADY the full-pool arm (PHASE_P §4.1), so it keeps
# its name and its hash, and is not re-exported here.
# usage: phaseP6_artifacts.sh <code>
set -eu
code=$1
CTC=/home/will/git/CleverKeys-ML/ctc
cd "$CTC"
case "$code" in
  el) layout=layouts/el_qwerty.json ;;
  uk) layout=layouts/uk_jcuken.json ;;
  bg) layout=layouts/bg_bds.json ;;
  mk) layout=layouts/mk_lynyertdz.json ;;
  he) layout=layouts/he_1.json ;;
  *)  echo "unknown script $code" >&2; exit 2 ;;
esac
src="$HOME/ctc-train/ckpt/phaseP6-${code}-v2full"
cp "$src/ctc_swipe_encoder.onnx"       "artifacts/${code}_synth_v2full_ch80.onnx"
cp "$src/ctc_swipe_encoder_fp16w.onnx" "artifacts/${code}_synth_v2full_ch80_fp16w.onnx"
# --onnx and --out are resolved against --workdir, so both must be absolute.
python3 make_golden.py --layout "$CTC/$layout" --vocab "$code" \
  --onnx "$CTC/artifacts/${code}_synth_v2full_ch80_fp16w.onnx" \
  --preset "1.05,2.0,0.2,0.3734,0.9882" \
  --out "$CTC/artifacts/${code}_synth_v2full_ch80_fp16w_golden.json"
cd "$CTC/artifacts"
sha256sum "${code}_synth_v2full_ch80.onnx" \
          "${code}_synth_v2full_ch80_fp16w.onnx" \
          "${code}_synth_v2full_ch80_fp16w_golden.json"
stat -c '%n %s' "${code}_synth_v2full_ch80.onnx" \
                "${code}_synth_v2full_ch80_fp16w.onnx" \
                "${code}_synth_v2full_ch80_fp16w_golden.json"
