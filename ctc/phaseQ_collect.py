#!/usr/bin/env python3
"""Phase Q — collect the per-script battery into ``ctc/phase_q_scripts.json``.

Every cell is parsed from the run's own eval JSONs under ``~/ctc-train/evalQ``
rather than transcribed by eye (the P6 discipline).  ru's row carries the REAL
probe (the only accuracy claim); the five others carry their own v3 synthesis
holdout plus the EN zero-shot margins, which are the only cross-generation
comparator (PHASE_Q.md §7.5).

``--seeds`` collects the closing round's replication triple instead
(PHASE_Q.md §8) into ``ctc/phase_q_seeds.json``: per script the three holdout
reads at seeds 1234/4321/7777 with their mean and sample sd, the EN-control
margins recomputed at the seed mean, ru's real-probe triple paired per row
against s1234, and the §8.4 anomaly rules evaluated **in code** so "which bytes
ship" is decided by the pre-registered rule rather than by reading a table.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phaseQ_paired import cells, load as load_dump  # noqa: E402

O = Path.home() / "ctc-train" / "evalQ"
CKPT = Path.home() / "ctc-train" / "ckpt"
HERE = Path(__file__).resolve().parent
OUT = HERE / "phase_q_scripts.json"
OUT_SEEDS = HERE / "phase_q_seeds.json"

SEEDS = (1234, 4321, 7777)
CODES = ("ru", "el", "uk", "bg", "mk", "he")
HOLDOUT_KEYS = ("indict_t1", "indict_t3", "indict_t5", "greedy_t1",
                "le3_t1", "ge4_t1", "decoded")


def load(name: str) -> dict:
    return json.loads((O / f"{name}.json").read_text())


def mean_sd(xs: List[float]) -> Dict[str, float]:
    """Mean and **sample** sd (Bessel), the campaign's seed-table convention."""
    n = len(xs)
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
    return {"mean": round(m, 2), "sd": round(sd, 3), "n": n}


# ── the closing round: the seed triple ──────────────────────────────────────────

def run_name(code: str, seed: int) -> str:
    return f"phaseQ-{code}-v3" + ("" if seed == 1234 else f"-s{seed}")


def holdout_json(code: str, seed: int) -> str:
    """s1234's ru holdout was written by an ad-hoc invocation under a different
    name than the driver's; every other cell follows the driver's convention."""
    if seed == 1234:
        return "ru_v3_holdout" if code == "ru" else f"{code}_v3"
    return f"{code}_v3_s{seed}"


def best_val_greedy(code: str, seed: int) -> Optional[float]:
    """Max val-greedy over the run — the quantity checkpoint selection used."""
    p = CKPT / run_name(code, seed) / "metrics.jsonl"
    if not p.exists():
        return None
    vals = [json.loads(l)["val_greedy"] for l in p.read_text().splitlines() if l]
    return round(max(vals) * 100, 2) if vals else None


_ARGMAX = re.compile(r"argmax agreement (\d+)/(\d+)")
_SLICED = re.compile(r"sliced \[\d+,\d+\] max \|onnx - torch\| = ([0-9.e+-]+)")


def export_parity(code: str, seed: int) -> dict:
    """`{argmax, sliced, tol_relaxed}` from the run's own export log, or `{}`
    when the export predates the driver (ru s1234 was exported by hand)."""
    tag = "" if seed == 1234 else f"_s{seed}"
    log = O / f"{code}{tag}_export.log"
    if not log.exists():
        return {}
    txt = log.read_text()
    a, s = _ARGMAX.search(txt), _SLICED.search(txt)
    return {
        "argmax": f"{a.group(1)}/{a.group(2)}" if a else None,
        "sliced": float(s.group(1)) if s else None,
        "tol_relaxed": (O / f"{code}{tag}_export_tol2e-3.log").exists(),
    }


def anomalies(code: str, rows: Dict[int, dict], t1: Dict[str, float],
              vg: Dict[str, float]) -> List[str]:
    """The §8.4 rules, evaluated rather than eyeballed."""
    fired: List[str] = []
    for seed in SEEDS:
        r = rows[seed]
        par = r.get("export_parity") or {}
        argmax = par.get("argmax")
        if argmax is not None:
            hit, total = (int(v) for v in argmax.split("/"))
            if hit < total:
                fired.append(f"A1 s{seed}: export argmax {argmax}, "
                             f"not full agreement")
        if par.get("tol_relaxed"):
            fired.append(f"A1 s{seed}: export needed a relaxed parity tolerance")
        g = r.get("val_greedy_best")
        if g is not None and vg["mean"] - g > 1.0:
            fired.append(f"A2 s{seed}: val-greedy {g} is {vg['mean'] - g:.2f} "
                         f"below the triple mean {vg['mean']}")
        h = r["holdout"]["indict_t1"]
        if t1["sd"] > 0 and abs(h - t1["mean"]) > 3 * t1["sd"]:
            fired.append(f"A3 s{seed}: holdout t1 {h} outside mean +/- 3sd "
                         f"({t1['mean']} +/- {3 * t1['sd']:.3f})")
        if code == "ru":
            paired = r.get("paired_vs_s1234")
            if paired and paired["delta_b_minus_a"] >= 1.0 \
                    and paired["p_mcnemar"] < 0.05:
                fired.append(f"A4 s{seed}: real-probe t1 beats s1234 by "
                             f"{paired['delta_b_minus_a']:+.2f} at p "
                             f"{paired['p_mcnemar']:.3g}")
    return fired


def collect_seeds() -> int:
    rep: dict = {
        "what": "PHASE_Q.md §8 — the gen-4 decoders at seeds 1234/4321/7777",
        "probe": "each script's own v3 holdout (generator-relative), CKDT preset",
        "ru_probe": "additionally the REAL yandex valid-10k, eval-only",
        "shared_controls": "EN zero-shot + permuted are seed-independent and "
                           "were read once (§8.2)",
        "scripts": {},
    }
    for code in CODES:
        rows: Dict[int, dict] = {}
        for seed in SEEDS:
            j = load(holdout_json(code, seed))
            row = {
                "run": run_name(code, seed),
                "holdout": {k: j[k] for k in HOLDOUT_KEYS},
                "val_greedy_best": best_val_greedy(code, seed),
                "export_parity": export_parity(code, seed),
            }
            if code == "ru":
                row["real_probe"] = {
                    k: load("ru_v3_ship" if seed == 1234 else f"ru_v3_ship_s{seed}")[k]
                    for k in HOLDOUT_KEYS}
            rows[seed] = row

        if code == "ru":
            base = load_dump(O / "dump_ru_v3.jsonl")
            for seed in SEEDS[1:]:
                b = load_dump(O / f"dump_ru_v3_s{seed}.jsonl")
                shared = sorted(set(base) & set(b))
                rows[seed]["paired_vs_s1234"] = cells(
                    base, b, lambda o: o["rank"] == 0, shared)
                rows[seed]["paired_vs_s1234_greedy"] = cells(
                    base, b, lambda o: bool(o["greedy_hit"]), shared)

        t1 = mean_sd([rows[s]["holdout"]["indict_t1"] for s in SEEDS])
        gr = mean_sd([rows[s]["holdout"]["greedy_t1"] for s in SEEDS])
        vg = mean_sd([rows[s]["val_greedy_best"] for s in SEEDS
                      if rows[s]["val_greedy_best"] is not None])
        entry = {
            "seeds": {str(s): rows[s] for s in SEEDS},
            "holdout_t1": t1, "holdout_greedy": gr, "val_greedy_best": vg,
            "en192_t1": load(f"{code}_en192")["indict_t1"],
            "en80_t1": load(f"{code}_en80")["indict_t1"],
            "permuted_t1": load(f"{code}_permuted")["indict_t1"],
        }
        entry["margin_ch192_at_mean"] = round(t1["mean"] - entry["en192_t1"], 2)
        entry["margin_ch80_at_mean"] = round(t1["mean"] - entry["en80_t1"], 2)
        if code == "ru":
            entry["real_t1"] = mean_sd([rows[s]["real_probe"]["indict_t1"]
                                        for s in SEEDS])
            entry["real_greedy"] = mean_sd([rows[s]["real_probe"]["greedy_t1"]
                                            for s in SEEDS])
            entry["real_le3"] = mean_sd([rows[s]["real_probe"]["le3_t1"]
                                         for s in SEEDS])
            entry["real_ge4"] = mean_sd([rows[s]["real_probe"]["ge4_t1"]
                                         for s in SEEDS])
        entry["anomalies"] = anomalies(code, rows, t1, vg)
        rep["scripts"][code] = entry

    rep["any_anomaly"] = any(rep["scripts"][c]["anomalies"] for c in CODES)
    rep["shipped_artifact"] = (
        "s1234 stays the shipped artifact (§8.4: no anomaly fired)"
        if not rep["any_anomaly"] else
        "ANOMALY FIRED — see per-script `anomalies`; §8.4 requires the "
        "supersede to be argued explicitly")
    OUT_SEEDS.write_text(json.dumps(rep, indent=1, ensure_ascii=False))

    print(f"{'script':>7} {'s1234':>7} {'s4321':>7} {'s7777':>7} "
          f"{'mean':>7} {'sd':>6} {'EN192':>7} {'margin':>7}")
    for code in CODES:
        e = rep["scripts"][code]
        v = [e["seeds"][str(s)]["holdout"]["indict_t1"] for s in SEEDS]
        print(f"{code:>7} {v[0]:>7.2f} {v[1]:>7.2f} {v[2]:>7.2f} "
              f"{e['holdout_t1']['mean']:>7.2f} {e['holdout_t1']['sd']:>6.3f} "
              f"{e['en192_t1']:>7.2f} {e['margin_ch192_at_mean']:>+7.2f}")
    ru = rep["scripts"]["ru"]
    v = [ru["seeds"][str(s)]["real_probe"]["indict_t1"] for s in SEEDS]
    print(f"\nru REAL probe: {v[0]:.2f} / {v[1]:.2f} / {v[2]:.2f}  "
          f"mean {ru['real_t1']['mean']:.2f} sd {ru['real_t1']['sd']:.3f}")
    print(f"anomalies: {rep['shipped_artifact']}")
    print(f"-> {OUT_SEEDS}")
    return 0


# ── the original single-seed battery ────────────────────────────────────────────

def collect_scripts() -> int:
    rep: dict = {"probe": "own v3 holdout (generator-relative), CKDT preset",
                 "ru_probe": "REAL yandex valid-10k, eval-only", "scripts": {}}
    for code in ("el", "uk", "bg", "mk", "he"):
        v3 = load(f"{code}_v3")
        row = {
            "holdout": {k: v3[k] for k in HOLDOUT_KEYS},
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
        "real_probe": {k: ru[k] for k in HOLDOUT_KEYS},
        "fp16w_t1": load("ru_v3_fp16w")["indict_t1"],
        "holdout_diag_GQT": load("ru_v3_holdout")["indict_t1"],
        "paired_vs_v2full": json.loads(
            (O / "paired_v3_vs_v2.json").read_text())["indict_t1"],
    }
    OUT.write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps(rep, indent=1)[:800])
    print(f"-> {OUT}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", action="store_true",
                    help="collect the §8 replication triple instead")
    args = ap.parse_args()
    return collect_seeds() if args.seeds else collect_scripts()


if __name__ == "__main__":
    raise SystemExit(main())
