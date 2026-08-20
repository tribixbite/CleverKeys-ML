#!/usr/bin/env python3
"""Phase Q — collect the per-script battery into ``ctc/phase_q_scripts.json``.

Every cell is parsed from the run's own eval JSONs under ``~/ctc-train/evalQ``
rather than transcribed by eye (the P6 discipline).  ru's row carries the REAL
probe (the only accuracy claim); the five others carry their own v3 synthesis
holdout plus the EN zero-shot margins, which are the only cross-generation
comparator (PHASE_Q.md §7.5).
"""
from __future__ import annotations

import json
from pathlib import Path

O = Path.home() / "ctc-train" / "evalQ"
OUT = Path(__file__).resolve().parent / "phase_q_scripts.json"


def load(name: str) -> dict:
    return json.loads((O / f"{name}.json").read_text())


def main() -> int:
    rep: dict = {"probe": "own v3 holdout (generator-relative), CKDT preset",
                 "ru_probe": "REAL yandex valid-10k, eval-only", "scripts": {}}
    for code in ("el", "uk", "bg", "mk", "he"):
        v3 = load(f"{code}_v3")
        row = {
            "holdout": {k: v3[k] for k in ("indict_t1", "indict_t3", "indict_t5",
                                           "greedy_t1", "le3_t1", "ge4_t1",
                                           "decoded")},
            "fp16w_t1": load(f"{code}_v3_fp16w")["indict_t1"],
            "permuted_t1": load(f"{code}_permuted")["indict_t1"],
            "en192_t1": load(f"{code}_en192")["indict_t1"],
            "en80_t1": load(f"{code}_en80")["indict_t1"],
        }
        row["margin_ch192"] = round(row["holdout"]["indict_t1"] - row["en192_t1"], 2)
        row["margin_ch80"] = round(row["holdout"]["indict_t1"] - row["en80_t1"], 2)
        row["fp16w_cost"] = round(row["fp16w_t1"] - row["holdout"]["indict_t1"], 2)
        rep["scripts"][code] = row
    ru = load("ru_v3_ship")
    rep["scripts"]["ru"] = {
        "real_probe": {k: ru[k] for k in ("indict_t1", "indict_t3", "indict_t5",
                                          "greedy_t1", "le3_t1", "ge4_t1",
                                          "decoded")},
        "fp16w_t1": load("ru_v3_fp16w")["indict_t1"],
        "holdout_diag_GQT": load("ru_v3_holdout")["indict_t1"],
        "paired_vs_v2full": json.loads(
            (O / "paired_v3_vs_v2.json").read_text())["indict_t1"],
    }
    OUT.write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps(rep, indent=1)[:800])
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
