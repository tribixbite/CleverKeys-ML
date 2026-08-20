#!/usr/bin/env bash
# Phase Q — promote a script's v3 exports into ctc/artifacts/ and freeze its
# golden fixture at the adopted preset (CKDT, lambda 2.0 — unchanged; Q moves
# the generator and nothing else).
#
# Naming: `<code>_synth_v3_ch80*` — generation 4 for the five non-ru scripts
# (after v1, v2, v2full) and generation 3 for ru (after v1==full-pool, v2).
# Every superseded generation stays in the registry: those bytes are what
# their phase's numbers were measured on.
#
# SHIPPING TRACK ONLY.  The sealed research twin never passes through here:
# its weights and samples live under ~/ctc-train/research_only/ and are
# permanently unshippable (PHASE_Q.md §0/§5.2).
# usage: phaseQ_artifacts.sh <code>
set -eu
code=$1
CTC=/home/will/git/CleverKeys-ML/ctc
cd "$CTC"
case "$code" in
  ru) layout=layouts/ru_jcuken_default.json ;;
  el) layout=layouts/el_qwerty.json ;;
  uk) layout=layouts/uk_jcuken.json ;;
  bg) layout=layouts/bg_bds.json ;;
  mk) layout=layouts/mk_lynyertdz.json ;;
  he) layout=layouts/he_1.json ;;
  *)  echo "unknown script $code" >&2; exit 2 ;;
esac
src="$HOME/ctc-train/ckpt/phaseQ-${code}-v3"
cp "$src/ctc_swipe_encoder.onnx"       "artifacts/${code}_synth_v3_ch80.onnx"
cp "$src/ctc_swipe_encoder_fp16w.onnx" "artifacts/${code}_synth_v3_ch80_fp16w.onnx"
# --onnx and --out are resolved against --workdir, so both must be absolute.
python3 make_golden.py --layout "$CTC/$layout" --vocab "$code" \
  --onnx "$CTC/artifacts/${code}_synth_v3_ch80_fp16w.onnx" \
  --preset "1.05,2.0,0.2,0.3734,0.9882" \
  --out "$CTC/artifacts/${code}_synth_v3_ch80_fp16w_golden.json"
cd "$CTC/artifacts"
sha256sum "${code}_synth_v3_ch80.onnx" \
          "${code}_synth_v3_ch80_fp16w.onnx" \
          "${code}_synth_v3_ch80_fp16w_golden.json"
stat -c '%n %s' "${code}_synth_v3_ch80.onnx" \
                "${code}_synth_v3_ch80_fp16w.onnx" \
                "${code}_synth_v3_ch80_fp16w_golden.json"
