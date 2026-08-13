#!/bin/bash
# Phase-L battery for a coupled-pair run (train_v2.py).
#
# Order is load-bearing and matches the PHASE_K §8.5 blind protocol:
#   1. export both members,
#   2. measure the LABEL-FREE per-frame agreement gate and write it to json
#      (this is the number that must be committed BEFORE any mix decode),
#   3. only then decode — members solo and the prob-averaged pair — on
#      val-9918 (AOSP/E1) and the six layout bars (az26/E1 + dvorak-vs-app-98k).
#
# Members are the per-member bests (a_best/b_best = the single-model bar); the
# mix is the jointly selected, agreement-gated pair (pair_a_best/pair_b_best =
# the pair bar). test-2400 is never touched.
#
# usage: phaseL_eval.sh <run-name> [what]
#   what = all | gate | members | pair        (default: all)
set -euo pipefail
RUN=$1
WHAT=${2:-all}
CTC=/home/will/git/CleverKeys-ML/ctc
WD=/home/will/ctc-train
D=$WD/ckpt/$RUN
mkdir -p "$WD/altlayout" "$WD/phaseL"
cd "$CTC"

export_one() {   # export_one <ckpt> <out>
  [ -f "$D/$2" ] && return 0
  echo "== export $1 -> $2"
  python3 export_onnx.py --ckpt "ckpt/$RUN/$1" --out "ckpt/$RUN/$2" \
    2>&1 | tee -a "$D/export.log" | tail -2
}

val9918() {      # val9918 <tag> <onnx-list> [ens-avg]
  echo "== val-9918 AOSP E1: $1"
  nice -n 10 python3 eval_beam.py --onnx "$2" ${3:+--ens-avg $3} \
    --test data/val_hwsfuto.jsonl --preset 1.05,1.1,0.2,0.3734,0.9882 \
    --beam-width 100 --top-k 8 --out "$WD/phaseL/${RUN}_$1_dump.jsonl" \
    2>&1 | tee "$WD/phaseL/${RUN}_$1_val9918.log" | tail -6
}

layouts() {      # layouts <tag> <onnx-list> [ens-avg]
  echo "== alt layouts az26 E1: $1"
  nice -n 10 python3 eval_altlayout.py --onnx "$2" ${3:+--ens-avg $3} --arm az26 \
    --layouts dvorak,azerty,qwertz,german,spanish \
    --json-out "$WD/altlayout/${RUN}_$1_az26_e1.json" \
    2>&1 | tee "$WD/altlayout/${RUN}_$1_az26_e1.log" | tail -8
  echo "== dvorak vs app 98k trie: $1"
  nice -n 10 python3 eval_altlayout.py --onnx "$2" ${3:+--ens-avg $3} \
    --layouts dvorak --lexicon dvorak=en \
    --json-out "$WD/altlayout/${RUN}_$1_dvorak_en98k.json" \
    2>&1 | tee "$WD/altlayout/${RUN}_$1_dvorak_en98k.log" | tail -4
}

export_one a_best.pt a.onnx
export_one b_best.pt b.onnx
export_one pair_a_best.pt pair_a.onnx
export_one pair_b_best.pt pair_b.onnx

# ── 2. the label-free gate, BEFORE any decode of the mix ──────────────────────
if [ "$WHAT" = all ] || [ "$WHAT" = gate ]; then
  echo "== label-free pair gate (no labels, no beam)"
  python3 pair_agreement.py --onnx "$D/pair_a.onnx,$D/pair_b.onnx" --rows 2000 \
    --json-out "$WD/phaseL/${RUN}_gate.json" | tail -2
fi

if [ "$WHAT" = all ] || [ "$WHAT" = members ]; then
  val9918 a "$D/a.onnx";  layouts a "$D/a.onnx"
  val9918 b "$D/b.onnx";  layouts b "$D/b.onnx"
fi

if [ "$WHAT" = all ] || [ "$WHAT" = pair ]; then
  val9918 pair "$D/pair_a.onnx,$D/pair_b.onnx" prob
  layouts pair "$D/pair_a.onnx,$D/pair_b.onnx" prob
fi
echo "== done $RUN ($WHAT)"
