#!/usr/bin/env python3
"""Phase P6 — assemble ctc/phase_p6_scripts.json from the run's own outputs.

Reads the P4 record (`ctc/phase_p_scripts.json`), the P6 decodes
(`~/ctc-train/evalP6/*.json`), the paired McNemar (`phaseP6_paired.json`) and
the export/quantize logs, and writes one committed record with both footings
side by side.  Nothing here recomputes anything: it is a transcription with the
parsing done once instead of by eye.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "/home/will/git/CleverKeys-ML/ctc")

W = Path.home() / "ctc-train"
O = W / "evalP6"
CTC = Path("/home/will/git/CleverKeys-ML/ctc")
CODES = ["el", "uk", "bg", "mk", "he"]

FOLD_RE = re.compile(r"max \|Δlog_emissions\| = ([0-9.e+-]+)")
SLICED_RE = re.compile(r"sliced \[\d+,\d+\] max \|onnx - torch\| = ([0-9.e+-]+)"
                       r"\s+argmax agreement (\d+)/(\d+)")
Q_RE = re.compile(r"parity vs source: sliced max\|Δ\| ([0-9.e+-]+)\s+"
                  r"argmax agreement (\d+)/(\d+)")


def jload(p: Path) -> dict:
    return json.loads(p.read_text())


def export_gates(code: str) -> dict:
    tol = 1e-3
    log = O / f"{code}_export.log"
    relaxed = O / f"{code}_export_tol2e-3.log"
    if relaxed.exists():
        log, tol = relaxed, 2e-3
    txt = log.read_text()
    fold = FOLD_RE.search(txt)
    sl = SLICED_RE.search(txt)
    q = Q_RE.search((O / f"{code}_fp16w.log").read_text())
    return {
        "parity_tol": tol,
        "bn_fold_sliced": float(fold.group(1)) if fold else None,
        "fp32_vs_torch_sliced": float(sl.group(1)) if sl else None,
        "fp32_argmax": f"{sl.group(2)}/{sl.group(3)}" if sl else None,
        "fp16w_vs_fp32_noise": float(q.group(1)) if q else None,
        "fp16w_argmax": f"{q.group(2)}/{q.group(3)}" if q else None,
    }


def main() -> int:
    p4 = jload(CTC / "phase_p_scripts.json")
    paired = jload(W / "phaseP6_paired.json")
    out = {
        "probe": p4["probe"] + " — UNCHANGED from P4 (asserted bit-identical, "
                 "so every P4/P6 pair below is the same 10,000 rows)",
        "caveat": p4["caveat"],
        "donor_footing_caveat":
            "P6 trains on --train-donor-side all, which puts the holdout's "
            "reserved donor half INSIDE the training pool. The P6 holdout is "
            "therefore no longer donor-disjoint and its delta over P4 is an "
            "upper bound on the honest effect. ru, read on a REAL probe, "
            "prices the same change at +0.86 (p 0.0023).",
        "scripts": {},
    }
    for code in CODES:
        v2f = jload(O / f"{code}_v2full.json")
        f16 = jload(O / f"{code}_v2full_fp16w.json")
        perm = jload(O / f"{code}_permuted.json")
        en192 = jload(O / f"{code}_en192.json")
        en80 = jload(O / f"{code}_en80.json")
        repro = jload(O / f"{code}_p4_repro.json")
        old = p4["scripts"][code]
        out["scripts"][code] = {
            "p4_v2_90_10": old["v2"],
            "p4_repro_indict_t1": repro["indict_t1"],
            "p6_v2_full_pool": {k: v2f[k] for k in
                                ("greedy_t1", "indict_t1", "indict_t3",
                                 "indict_t5", "le3_n", "le3_t1", "ge4_n",
                                 "ge4_t1")},
            "p6_fp16w_indict_t1": f16["indict_t1"],
            "en_ch192_zeroshot_indict_t1": en192["indict_t1"],
            "en_ch80_zeroshot_indict_t1": en80["indict_t1"],
            "delta_vs_ch192": round(v2f["indict_t1"] - en192["indict_t1"], 2),
            "delta_vs_ch80": round(v2f["indict_t1"] - en80["indict_t1"], 2),
            "p4_delta_vs_ch192": old["delta_vs_ch192"],
            "permuted_geometry_indict_t1": perm["indict_t1"],
            "permuted_geometry_greedy_t1": perm["greedy_t1"],
            "paired_p4_vs_p6": paired[code],
            "export": export_gates(code),
        }
    # Pooled across the five scripts.  Not a single-model test — five models on
    # five probes — but it is the resolution statement the per-script p-values
    # invite: 50,000 paired rows is the largest paired sample this campaign has
    # ever put on the donor-footing question.
    from phase_n_decomp import mcnemar_p as _mp
    out["pooled_p4_vs_p6"] = {}
    for key in ("indict_t1", "greedy", "le3_t1", "ge4_t1"):
        b = sum(paired[c][key]["p4_only"] for c in CODES)
        cc = sum(paired[c][key]["p6_only"] for c in CODES)
        n = sum(paired[c][key]["n"] for c in CODES)
        out["pooled_p4_vs_p6"][key] = {
            "n": n, "p4_only": b, "p6_only": cc,
            "delta": round((cc - b) / n * 100, 3),
            "p": float(f"{_mp(b, cc):.3g}")}
    dst = CTC / "phase_p6_scripts.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("wrote", dst)
    for code in CODES:
        s = out["scripts"][code]
        d = s["paired_p4_vs_p6"]["indict_t1"]
        print(f"{code}: {s['p4_v2_90_10']['indict_t1']:6.2f} -> "
              f"{s['p6_v2_full_pool']['indict_t1']:6.2f} "
              f"({d['delta']:+.2f}, p {d['p']})  greedy "
              f"{s['p4_v2_90_10']['greedy_t1']:6.2f} -> "
              f"{s['p6_v2_full_pool']['greedy_t1']:6.2f}  "
              f"vs192 {s['p4_delta_vs_ch192']:+.2f} -> {s['delta_vs_ch192']:+.2f}  "
              f"perm {s['permuted_geometry_indict_t1']:.2f}  "
              f"tol {s['export']['parity_tol']:g} "
              f"argmax {s['export']['fp32_argmax']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
