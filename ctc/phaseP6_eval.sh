#!/usr/bin/env bash
# Phase P6 — per-script battery, run AFTER a script's training ends.
#
# The probe is deliberately the P4 cache (`cache_<code>_v2/holdout.npz`), which
# phaseP6_assert_holdout.py proves is byte-for-byte the same rows the P6
# regeneration produces.  Using the P4 file by name makes the pairing explicit:
# every P4 number and every P6 number below is read off the identical 10,000
# rows, so the McNemar in phaseP6_paired.py is exact.
#
#  1. export fp32 with the campaign parity gates, then the fp16w ship bytes;
#  2. full-holdout read at the ADOPTED preset (CKDT, lambda 2.0 — no sweep,
#     P6 changes the donor footing and nothing else), fp32 and fp16w, with a
#     per-row dump for the paired test;
#  3. falsification control: permuted key centres, seed 4242;
#  4. the two zero-shot controls, ch192 EN (shipped) and ch80 EN (capacity
#     matched), re-run rather than carried over from P4 even though the probe
#     is unchanged — a free reproduction check on the harness.
set -u
CTC=/home/will/git/CleverKeys-ML/ctc
O=/home/will/ctc-train/evalP6
EN192=artifacts/phaseM_kd_fresh_w1_s1234.onnx
EN80=$HOME/ctc-train/ckpt/phaseH-p50/ctc_swipe_encoder.onnx
cd "$CTC" || exit 1
mkdir -p "$O"

for code in "$@"; do
  case "$code" in
    el) layout=layouts/el_qwerty.json ;;
    uk) layout=layouts/uk_jcuken.json ;;
    bg) layout=layouts/bg_bds.json ;;
    mk) layout=layouts/mk_lynyertdz.json ;;
    he) layout=layouts/he_1.json ;;
    *)  echo "unknown script $code" >&2; exit 2 ;;
  esac
  run="phaseP6-${code}-v2full"
  ck="ckpt/${run}/best.pt"
  fp32="ckpt/${run}/ctc_swipe_encoder.onnx"
  fp16="ckpt/${run}/ctc_swipe_encoder_fp16w.onnx"
  probe="$HOME/ctc-train/cache_${code}_v2/holdout.npz"
  # he's fp32 export needed the parity tolerance at 2e-3 in P4 (sliced residue
  # 1.16e-03, argmax 100/100 on both probes).  Watched again here, not assumed:
  # the default is tried first and the relaxation is only applied if it fails.
  echo "### [$code] export fp32"
  python3 export_onnx.py --layout "$CTC/$layout" --ckpt "$ck" --out "$fp32" \
      --parity-features "cache_${code}_v2full/val.npz" \
      > "$O/${code}_export.log" 2>&1
  rc=$?
  tail -n 20 "$O/${code}_export.log"
  if [ $rc -ne 0 ]; then
    echo "### [$code] fp32 export FAILED at the default 1e-3 — retry at 2e-3"
    python3 export_onnx.py --layout "$CTC/$layout" --ckpt "$ck" --out "$fp32" \
      --parity-features "cache_${code}_v2full/val.npz" --parity-tol 2e-3 \
      > "$O/${code}_export_tol2e-3.log" 2>&1
    tail -n 20 "$O/${code}_export_tol2e-3.log"
  fi
  echo "### [$code] export fp16w"
  python3 quantize_onnx.py --layout "$CTC/$layout" --onnx "$fp32" --out "$fp16" \
      --mode fp16w --calib-npz "cache_${code}_v2full/train_synth.npz" \
      2>&1 | tee "$O/${code}_fp16w.log" | tail -n 12

  ONNX="$HOME/ctc-train/$fp32"
  echo "### [$code] holdout, fp32 then fp16w, CKDT preset"
  python3 eval_script.py --code "$code" --onnx "$ONNX" --probe "$probe" \
      --preset ckdt --progress 0 --dump "$O/dump_${code}_p6.jsonl" \
      --out-json "$O/${code}_v2full.json" > /dev/null 2>&1
  python3 eval_script.py --code "$code" --onnx "$HOME/ctc-train/$fp16" --probe "$probe" \
      --preset ckdt --progress 0 --out-json "$O/${code}_v2full_fp16w.json" \
      > /dev/null 2>&1
  echo "### [$code] falsification control: permuted key centres"
  python3 eval_script.py --code "$code" --onnx "$ONNX" --probe "$probe" \
      --preset ckdt --progress 0 --permute-layout 4242 \
      --out-json "$O/${code}_permuted.json" > /dev/null 2>&1
  echo "### [$code] zero-shot controls on the same rows"
  python3 eval_script.py --code "$code" --onnx "$EN192" --probe "$probe" \
      --preset ckdt --progress 0 --out-json "$O/${code}_en192.json" > /dev/null 2>&1
  python3 eval_script.py --code "$code" --onnx "$EN80" --probe "$probe" \
      --preset ckdt --progress 0 --out-json "$O/${code}_en80.json" > /dev/null 2>&1
  python3 - "$code" <<'PY'
import json, sys
o, c = "/home/will/ctc-train/evalP6", sys.argv[1]
for tag in ("v2full", "v2full_fp16w", "permuted", "en192", "en80"):
    d = json.load(open(f"{o}/{c}_{tag}.json"))
    print(f"  {tag:14s} t1 {d['indict_t1']:6.2f}  t3 {d['indict_t3']:6.2f}  "
          f"t5 {d['indict_t5']:6.2f}  greedy {d['greedy_t1']:6.2f}  "
          f"<=3 {d['le3_t1']:6.2f}  4+ {d['ge4_t1']:6.2f}  n {d['decoded']}")
PY
done
echo PHASEP6-EVAL-DONE
