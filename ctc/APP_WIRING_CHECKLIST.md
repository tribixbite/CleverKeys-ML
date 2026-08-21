# APP_WIRING_CHECKLIST — the ordered, current, actionable list for the app agent

**Written**: 2026-08-20. **App HEAD it is written against**: `d717bda7`
(`/home/will/git/swype/CleverKeys`, read-only from this side — nothing here was committed there).
**ML HEAD**: this repo, after **Phase Q and its closing round** (generation 4 = `*_synth_v3_ch80*`
for all six scripts; the λ sweep left `tunedRuCkdt` unchanged — §2.2).

**What this file is.** The three CTC documents in this repo answer different questions and it is
worth not confusing them:

| document | question it answers |
|---|---|
| `ctc-architecture-and-multiscript-guide.md` | *How does it work, and what is true?* — reference |
| `APP_INTEGRATION_AUDIT.md` | *What is wrong, with evidence?* — §2 original, §5 at `9a6ffdd2`, §6 at `d717bda7` |
| **this file** | *What do I do next, in what order, with which bytes?* |

Everything below is either **open** or a **prerequisite**. Nothing that is done appears here
except where its being done unblocks something. Every sha256 in §2 was produced by running
`sha256sum` against the file on disk in this repo on 2026-08-20, not copied from another document.

---

## 1. Remaining audit fixes, in order

Six items. The first four are small and all four remove a way for the next reader to be misled or
for a future model swap to go wrong quietly.

### 1.1 — MEDIUM-3: banner the execution brief · **OPEN, unchanged across three audit passes**

`docs/audit/remediation-plans/ctc-integration-execution-brief.md`. No banner; `:86` still reads
**"Q1 model choice: SUPERSEDED-PENDING — a new model is training and an ML-side agent is running
comparisons"**; `:74` and `:43` still say **"Default engine stays `neural`"**.

Both statements are false and have been since 2026-08-18. This is the last surviving member of the
anti-confusion set — MEDIUM-8, MEDIUM-9, NEW-1, HIGH-3 all closed around it — and it is the
document the original audit named as *the single likeliest source of the "which ONNX?" question*.
One paragraph at the top: no model swap is pending (Phase N is terminal, Phase P shipped nothing
into the APK), `ctc` is the default, neural is deleted.

### 1.2 — HIGH-4's residue: the emission check still runs nowhere automatic · **OPEN**

`d717bda7` did real work here and two of the three parts are genuinely closed:

- `CtcParityTest.kt:43-48` — `MODEL_ASSET_PATH` is now derived from `CtcEngineAdapter.MODEL_ASSET`.
- `CtcParityTest.kt:170-181` — the preset pin now asserts `beamWidth == Defaults.CTC_BEAM_WIDTH`.
- Instrumented tests run in CI for the first time, green on API 21/29/34, `OK (23 tests)`.

**What is not closed.** `.github/scripts/emulator-ci.sh`'s gate runs exactly:

```
tribixbite.cleverkeys.swipe.CtcMultiLanguageInstrumentedTest,
tribixbite.cleverkeys.GeometricSwipeOracleTest,
tribixbite.cleverkeys.CrashGuardInstrumentedTest
```

`CtcEmissionModelParityTest` — the only thing that runs the shipped ONNX and compares its
emissions against the fixture's stored matrices (`EMISSION_TOL = 2e-3`) — is not in it.
`CtcMultiLanguageInstrumentedTest` does open a real ORT session and decode real gestures, which
is valuable and new, but it never reads `ctc_golden.json`. So HIGH-4's precise mechanism is
untouched: **a model swap that updates the fixture's `source_onnx_sha256` header but leaves the
emission matrices stale still passes CI green.**

*Fix*: add `tribixbite.cleverkeys.swipe.CtcEmissionModelParityTest` to the `CLASSES` string. The
infrastructure it needs is built and proven. `CtcLatencyGateTest` is a separate call — a latency
gate on a shared GitHub emulator is a flake source, and with MEDIUM-2 closed the honest home for
it is an ew-cli run on real hardware.

Two facts worth recording alongside, neither a defect but both easy to over-read:

- **`ui-testing.yml` does not trigger on push.** `on:` is `pull_request` to `main`, `cron '0 6 * * *'`,
  and `workflow_dispatch`. Direct commits to `main` — the ordinary path here — are gated only by
  the nightly. `ci.yml`, which *does* run on push, is `assembleDebug` + `runPureTests` + `lint`.
- It is not a required check yet, deliberately, and `d717bda7` gives the right reason.

### 1.3 — HIGH-2's residue: two unmarked `sw2345` citations in `docs/` · **OPEN**

`f172bb8e` widened `CoreImeHygieneDriftTest`'s scan from `src/main/kotlin` to include
`src/test/kotlin` and `src/androidTest/kotlin`, which immediately caught `SwipeEngineRouterTest.kt:20`.
`docs/` is still unscanned, and both survivors live there:

- `docs/audit/2026-08-17-neural-vs-ctc-parity.md:619-623` — finding 13, still unstruck, still
  ending "Resolve before quoting azerty 83.81 / qwertz 83.01 / german 80.64 / spanish 88.45 as
  app-relevant". Its own §2.1.3 already refuted it.
- `docs/eval/2026-08-15-ctc-per-language-lambda.md:101` — "german 80.64 vs spanish 88.45"
  presented as the spread across *"the languages we DID validate"*, and `:112` — "the campaign's
  88.98 dvorak-app figure".

The ship model's numbers are dvorak 91.82 / dvorak-app 91.10 / azerty 84.53 / qwertz 83.97 /
german 81.30 / spanish 89.53 (`MODELS_TABLE.md:113`). Decide separately whether the guard should
walk `docs/` — it would need an allowlist for the three sites that quote the figures *in order to
condemn them*.

### 1.4 — NEW-6: the app's own CTC references are a model generation behind · **OPEN**

`memory/HANDOFF.md` says *"**Russian is delivered**: `CleverKeys-ML/ctc/artifacts/ru_synth_ch80_fp16w.onnx`,
589,406 B, sha `84ac284d…`"* and *"decodes real Russian at 77.41 in-dict top-1"*. Both were true
on 2026-08-18 and are now **two generations** superseded: the ru ship bytes are
`ru_synth_v3_ch80_fp16w.onnx`, sha `8fffa75c…`, at **85.07** (`PHASE_Q.md` §7.3). The guide
mirror is the larger half of this — see §3.

### 1.5 — MEDIUM-4: 11 MB of superseded ONNX in `androidTest` · **OPEN**

Four files in `src/androidTest/assets/ctc_bench/`; `CtcOnnxLatencyBenchmarkTest.kt:45-46` still
calls `ch128` "the ship candidate", which it never was; `:351` is still
`fullDecodePath_ch128_beam100_tunedV2` with E1 constants; `CtcBenchFixture.kt:9` still cites a
rival golden fixture identity. Delete them or add a README saying what they are. Already item 2 of
`memory/HANDOFF.md`.

### 1.6 — Cleanup, ride-along

**LOW-9's remaining half** is the one worth doing early: nothing anywhere asserts
`CtcEngineAdapter.supportsLayout(...) == false` for a Cyrillic or Greek `KeyboardData`. The
router-level negative exists (`SwipeEngineRouterTest`, "the Greek QWERTY trap"); gate 3 has no
negative test. That is cosmetic today and **load-bearing the moment §2 removes gate 1**.

Then: MEDIUM-5 (`CtcSettingsActivity.kt:89-91` still omits the language list — largely superseded
by MEDIUM-7's card), MEDIUM-6's weak `swipe_engine_mode` predicate (documented as deliberate),
LOW-1 (`MappedLayout.padded` unread), LOW-2 (phantom `weight` in `CtcScoringParams.kt:13`), LOW-3
(no `@Deprecated` on `CtcFeaturizer.normalizeRawX/Y`), LOW-4, LOW-5, LOW-6 (the dev absolute path
`/home/will/ctc-train/ckpt/v2kd-fresh-w1/kd_fp16w.onnx` in both fixture copies, line 2), LOW-7,
LOW-8 (`SettingsActivity.kt:579`'s `"futo"` search keyword), LOW-10. Plus the stale line cite in
`docs/specs/ctc-swipe-engine.md:3-4`, which points at `Config.kt:300`; the constant is at `:311`.

---

## 2. Per-script wiring

### 2.0 — Read this first: stage the wiring, not the model

**The shipped English model, zero-shot on real Russian with nothing but the correct layout and
the correct trie, reads 76.32 in-dict top-1** (`PHASE_O.md` §2.1, measured on the Yandex
valid-10k, the only real non-Latin probe in existence). The purpose-built ru model takes that to
**79.73** — a gain of **+3.41**.

So of the ~79.7 points on the table for Russian, the app work delivers **76.3 of them before any
new model is bundled at all**, and the model delivers the last three. 76.3 is at or above the
geometric engine's cross-layout anchors (71–77), which is what a Cyrillic user gets today.

The practical consequence: **do §2.1 (the shared work) and §2.2 (layout + trie + projection) as
one milestone and ship it with no new model asset.** The APK grows by nothing, the risk is one
routing change, and a Cyrillic swipe goes from geometric to CTC-at-76. Then add model assets
per script as a second, independent milestone — each is 589,406 bytes and worth ~+3.

This is also the conservative ordering for a different reason: §2.1 items 1–6 are where every
silent-failure mode lives (slot order, projection, preset scale). Landing them without a new model
means a mistake shows up as *worse decoding*, not as a model that appears not to have trained.

### 2.1 — Shared work, done once, before any script

None of it is ML work. Eight changes; two are already done.

| # | change | file / today's state |
|---|---|---|
| 1 | per-script `ALPHABET` instead of `CharArray(26) { 'a' + it }` | `swipe/CtcEngineAdapter.kt:113` |
| 2 | per-script `buildMappedLayout` — `FloatArray(26)`/`BooleanArray(26)` and `letterOf`'s `'a'..'z'` filter | `swipe/CtcEngineAdapter.kt:266-301` |
| 3 | per-language model asset — `MODEL_ASSET` is a single constant | `swipe/CtcEngineAdapter.kt:99` |
| 4 | per-script routing — only `isLatinScript(script)` reaches `Engine.CTC` | `swipe/SwipeEngineRouter.kt:118-119` |
| 5 | make `tunedRuCkdt` reachable — `presetFor` branches on `LexiconSource` and can never return it | `swipe/ctc/CtcScoringParams.kt:155-165, 205-210` |
| 6 | a second fixture↔model↔preset row in `CtcParityTest` | one row today |
| 7 | trie width — the `MAX_CHILDREN = 26` clamp | **DONE** (`d671d19e`): the bound is a constructor check against the emission-head width |
| 8 | `CtcLayout` generic over `alphabet: CharArray` | **DONE** |

**The sharpest footgun in the whole plan: the model's slot order IS the app's alphabet array.**
Every layout JSON lists its letters in **codepoint-sorted** order and emission column `c` is
`letters[c]`. A mismatch does not throw — it **silently permutes every decode**. The strings are
in §2.2, character-for-character.

**The geometry needs no app-side change.** `app_layout.py` replicates
`KeyboardGeometry.computeKeyRects` + `buildMappedLayout` exactly and reproduces `en_qwerty.json`
from the app's own QWERTY XML to 4.7e-4, so the `layout_keys` the app computes at runtime for
these layouts *is* the geometry the models were trained on. Locale extra keys and the
bottom/number/numpad rows do not perturb it.

**Also do, once**: `LayoutScriptDeclarationTest`'s bidirectional assertion currently encodes
"latin ⟺ a–z-complete". Per-script routing changes what that test is protecting; extend it rather
than weaken it.

### 2.2 — Per script: bytes, alphabet, lexicon, preset

**Preset is the same for all six and does NOT change**: γ 1.05 / **λ 2.0** / β 0.2 /
γ-prune 0.3734 / β-prune 0.9882 = `CtcScoringParams.tunedRuCkdt` verbatim. λ = 2.0 is a
**frequency-scale** constant (`LAMBDA_CKDT_SCALE`), not a Russian one — every lexicon here is on
the CKDT `255 − rank` scale, the same scale `fr/de/es/it/pt/sv` already run at in production.
Item 5 of §2.1 is therefore "make one existing preset reachable", not "add six presets".

> **λ was re-swept in the Phase-Q closing round and the answer is "no change — but the constant
> is off-peak."** `PHASE_Q.md` §9.7: on the ru real probe's tune half, in-dict t1 is **monotone
> decreasing** across λ ∈ {1.1, 1.5, 2.0, 2.5, 3.0, 4.0} — 85.65 → 80.69 — so the optimum lies
> *below* the grid and the pre-registered interior-optimum rule refused adoption. **Do not change
> `tunedRuCkdt`.** But do not read the sweep as "2.0 is right" either: λ = 2.0 was fitted against
> a greedy-37 model and the generation-4 decoder reads greedy 66, so it carries a **measured,
> unconfirmed −0.63 t1 shortfall**. Deciding it needs one more ML-side phase, not an app change,
> and if it ever lands it changes ru's fixture too (fixture-and-preset rule).

**Model + fixture bytes.** All six ONNX are 589,406 B; the fp32 sources are 1,142,727 B each and
are not shipped. Six scripts is ~3.5 MB, which is the argument for gating them behind the langpack
import rather than bundling all of them.

| script | ship ONNX | sha256 | golden fixture | sha256 |
|---|---|---|---|---|
| **ru** | `ru_synth_v3_ch80_fp16w.onnx` | `8fffa75c722eb61e9e8c80d919fbca3e73eb698ebe3e3909cb766b3b8489962c` | `ru_synth_v3_ch80_fp16w_golden.json` (160,384 B) | `2e8de3c5a15e5874366f44f725aeec2eb72befd89b503d4b24b8b4a8d82fdde5` |
| **el** | `el_synth_v3_ch80_fp16w.onnx` | `7083794c501566f411b1f81495ba1f7f3df273c3eb58f6ee635caf168a4f8c3d` | `el_synth_v3_ch80_fp16w_golden.json` (144,427 B) | `d08d5501961e971db2ca120f6ee868b7b67ed37e34b6412dddbc7f7116de5753` |
| **uk** | `uk_synth_v3_ch80_fp16w.onnx` | `af9959a8954961eec117808371937cb26152c82a82cad0fc6a0ac06fd695db76` | `uk_synth_v3_ch80_fp16w_golden.json` (155,068 B) | `93602db1200a3b37ef11570d4f4ee3afdad2a45b0ca4f857a784728cdbb5cc98` |
| **bg** | `bg_synth_v3_ch80_fp16w.onnx` | `119d42f70cc763336f9a86efdc5ae4f562ba4a28179c2d386026bef674c039a7` | `bg_synth_v3_ch80_fp16w_golden.json` (154,835 B) | `f776ea03ab675ff6b741a3297c4f88b11f7af2cb183ce7b2604f082ed8420b9d` |
| **mk** | `mk_synth_v3_ch80_fp16w.onnx` | `4e371d967bf24f260eb539848ead7860f56dc904f6bfc74235879b76e81ae022` | `mk_synth_v3_ch80_fp16w_golden.json` (160,674 B) | `015c9bae7e25a97b0ac8bd6062bb58376caaa3aca99c138d0d531ff1887e0ccf` |
| **he** | `he_synth_v3_ch80_fp16w.onnx` | `a382371363653fbe7c806482035aa9e27968b9c098591910d24f9f1ba43212c7` | `he_synth_v3_ch80_fp16w_golden.json` (140,129 B) | `b29a99f4ac2c4f82547d040131ea48771f2791817287de6e3f9ec52fc9758ad9` |

All twelve live in `CleverKeys-ML/ctc/artifacts/`.

**The suffix is now uniform, and that is the change most likely to be missed.** Generation 4
(`PHASE_Q.md` §7.7) gave all six scripts the **same** `_v3_` suffix, retiring the old ru `_v2_` /
five-script `_v2full_` split that this file previously warned about. Every `*_synth_ch80*`,
`*_synth_v2_ch80*` and `*_synth_v2full_ch80*` file still in `ctc/artifacts/` is a **superseded**
generation kept only because published numbers were measured on it. The twelve hashes above were
produced by `sha256sum` against the files on disk on 2026-08-20.

**What generation 4 is, in one paragraph.** The synthesis generator was replaced by a learned
one — a conditional rectified-flow model over the trace's residual field from the ideal polyline,
trained on MIT data only (FUTO t3 + HWS), conditioned on pure geometry so one generator serves
every script. Nothing about the app-side contract moves: same alphabet strings, same slot order,
same preset, same 589,406-byte graphs, same fixture shape. On the real Yandex probe ru went
**79.73 → 85.07** (+5.34, exact McNemar p = 5.4e-53; greedy 56.12 → 65.66), which also clears the
≤3-stratum corollary generation 2 missed. A sealed, permanently unshippable twin trained on real
Russian swipes puts the upper bound at 85.95 — i.e. **the English-trained generator is within
0.89 of what in-domain data would buy**, and within 3.6 of a model trained on a million real
Russian swipes.

**Alphabet, K, layout, lexicon:**

| script | layout XML (`src/main/layouts/`) | K | alphabet / slot order — codepoint-sorted, copy verbatim | lexicon |
|---|---|---|---|---|
| ru | `cyrl_jcuken_ru.xml` | 31 | `абвгдежзийклмнопрстуфхцчшщыьэюя` | `scripts/dictionaries/langpack-ru.zip` — **exists**, importable today. 533,916 B; `dictionary.bin` 2,088,865 B, magic `CKDT` v2, 50 k words. `eval_cyrillic.build_trie` reads this exact zip, so every ru number is on the app's own lexicon. |
| el | `grek_qwerty.xml` | 25 | `αβγδεζηθικλμνξοπρςστυφχψω` | `scripts/dictionaries/langpack-el.zip` — **exists**, needs the full projection (§2.3) |
| uk | `cyrl_jcuken_uk.xml` | 31 | `абвгдежзийклмнопрстуфхцчшщьюяєі` | **must be built** — `build_wordlist.py --lang uk`; the `cyrillic` script gate already exists |
| bg | `cyrl_ueishsht.xml` | 30 | `абвгдежзийклмнопрстуфхцчшщъьюя` | **must be built** |
| mk | `cyrl_lynyertdz_mk.xml` | 31 | `абвгдежзиклмнопрстуфхцчшѓѕјљњќџ` | **must be built** |
| he | `hebr_1_il.xml` | 27 | `אבגדהוזחטיךכלםמןנסעףפץצקרשת` | **must be built**, and `build_wordlist._is_script_word` needs a new `hebrew` branch (0x0590–0x05FF) — it currently `raise`s on any script but latin/greek/cyrillic |

**el's layout prerequisite is already done.** `src/main/layouts/grek_qwerty.xml` declared
`script="latin"` for months (the `6af11da7` fix landed in `srcs/layouts/`, which no build task
reads). `6f30d60f` corrected the shipped file and added `LayoutScriptDeclarationTest` to keep it
corrected.

### 2.3 — Projection: mirror it exactly, and note el needs both halves

Applied to the lexicon **and** to anything compared against a decode (`PHASE_O.md` §3.4, from
`script_registry.py`):

* **all scripts** — lowercase; strip `- ' ’ ʼ ‘ \``.
* **el, he** — NFD, drop combining marks (`Mn`), NFC. Safe here: Greek accents/diaeresis and
  Hebrew niqqud are not keys.
* **ru, bg, mk** — **no NFD.** It decomposes й into и + breve and destroys the alphabet. Character
  folds instead: ru ё→е, ъ→ь; bg ѝ→и; mk ѐ→е, ѝ→и.
* **el only** — *after* mark stripping, word-final `σ` → `ς`.
* **uk** — no folds; words containing ї or ґ are **rejected as untypeable** (4.03 % of the
  vocabulary). Serving them needs the corner-alias path, which is a different input mode.

> **el is half-implemented, and the implemented half is the second half.**
> `swipe/ctc/CtcGreekOrthography.kt` (shipped `6f30d60f`, currently zero production callers —
> correct code awaiting a consumer, **not** dead code, do not sweep it) implements
> `repairFinalSigma` / `repairLexicon` / `affectedCount`: word-final only, idempotent, medial `σσ`
> preserved, higher frequency wins on collision. That is exactly the last of the four steps.
>
> The **mark-stripping step has no app-side implementation for Greek** — `CtcAzProjection` is
> Latin-specific. It is the step that makes the alphabet 25 letters: the el model's slot order
> contains no accented vowels, so an unprojected `λόγος` carries `ό`, a character with no emission
> slot. Wiring sigma alone upgrades "one Greek word in four is scored against the wrong key in the
> wrong row" (25.7 % of the pack) to "most of the pack cannot be represented at all". **Both
> halves or neither.**

### 2.4 — Per-script gates before you trust a wiring

1. **Slot-order equality.** Assert the app's `ALPHABET` string for the script equals the layout
   JSON's `letters` field, character for character. A permutation is silent.
2. **The 32-frame budget.** The encoder emits a fixed `log_emissions [1,32,65]`; a CTC path
   spends one frame per character **plus a separating blank between adjacent duplicates**, so a
   word is decodable iff `length + adjacent-duplicate-pairs ≤ 32`. `CtcDecodableLength`
   (`2d080c7d`) already computes this, and a test asserts every 20+ character word in
   `en_enhanced.json` clears it. **No script lexicon has been checked** — the v2 word draw is
   wordfreq token mass with no length ceiling, and Greek and Ukrainian carry long inflected
   forms. One loop over the trie's word list. A word over budget is unemittable with no error.
3. **The fixture row.** Each script fixture is 10 cases (5 pure-featurizer branch probes, 1
   word-path featurizer case, 4 model-backed beam cases) at the preset above — the same shape as
   the shipped en fixture, so `CtcParityTest` grows a row, not a mechanism.
4. **Contraction injection is scale-safe already.** `98307dc2` derives the injected frequency as
   `minReal − 1` per lexicon rather than using a constant, so a new CKDT lexicon gets the
   "reachable, never preferred" invariant by construction. Nothing to do per script; just do not
   reintroduce a constant.
5. **On-device latency and memory.** Never measured for any script model. Now *measurable*: the
   settle probes that inflated the first CTC decode by ~720 ms in `LOCAL_BUILD=true` builds were
   removed in `716f7be9`. The script graphs are half the shipped model's bytes, so the
   expectation is favourable — expectation is not measurement.

---

## 3. The app-repo copy of the guide — the v2 **and** v3 edits it still needs

`docs/specs/ctc-architecture-and-multiscript-guide.md` was byte-identical to this repo's copy when
both landed at `d76be9a6`. This repo's copy has since taken the Phase P/P6 edits **and the Phase-Q
generation-4 edit**; the app copy has taken none of them and is **two model generations stale**.
Nothing shipped depends on it, but it is the document `CLAUDE.md`'s spec-driven workflow points a
maintainer at, which is exactly the MEDIUM-3 failure shape.

**The app repo was not touched by the ML side.** This is the whole edit, described; someone with
write access to `/home/will/git/swype/CleverKeys` has to make it. The cheapest correct action is
to **copy this repo's file over the app's** — they were meant to be a mirror and the divergence is
entirely one-directional — and then re-check the app-state paragraphs against the app's real HEAD.
The section-by-section list below is for anyone who would rather merge than replace.

Bring across, from `CleverKeys-ML/ctc/ctc-architecture-and-multiscript-guide.md`:

| section | what changes |
|---|---|
| header | app state `d717bda7`; `PHASE_P.md` added to the source-of-truth list; drop the "byte-identical mirror" claim or restate it as a mirror-warning |
| §2 | the fourth gate — `isModelPermanentlyUnavailable()`, HIGH-1's fix |
| §2.1 | census now **46 / 2 / 36 / 2**; `grek_qwerty` fixed; `LayoutScriptDeclarationTest`; the two-schema (`c=` vs `key0=`) finding; LOW-9's remaining half |
| §3.1 | v2 headline 79.73 / greedy 56.12, and the zero-shot staging argument (76.32 from wiring alone, +3.41 from the model) |
| §3.2 step 3 | `script_synth.py --code <script>` at generator v2 and its five stages (S0 wordfreq token mass, S1/S2 geometry-matched donor draw, S4 vertex-aligned re-timing, S5 acquisition-bandwidth matching, and the never-fit-on-the-validator rule) — the app copy still describes `cyrillic_synth.py` and `weight = 255 − rank` |
| §3.2 step 7 | new — the 32-frame lexicon budget gate |
| §3.3 | **rewritten for v3**: what the learned generator is (conditional rectified flow over the residual field, MIT-only training, the acquisition-imprint repair round, the throughput regression), the four-arm table (ceiling 88.69 / **v3 85.07** / v2 79.73 / v1 77.42), the upper-bound one-liner (of the 8.96-point gap: the English-trained generator closes 5.34, in-domain data adds 0.89, generation itself costs 2.74 — so **real data is worth ~3.6 points now, not ~10 and not ~13**), the licence seal stated as a mechanism, the evidence tiers, and the λ result. The old v1→v2 material survives as §3.3a because its numbers are still cited |
| §4.1, §4.2, §4.3 | the **v3** ship row (85.07) plus the sealed-twin and ceiling rows, the generation-4 hash table for all six scripts with v1/v2 marked superseded, the G5-Q gate table with its paired McNemar block, and the v3 export gates (fp32 7.63e-05 argmax 100/100 at the default tolerance; fp16w +0.01 t1) |
| §4.5 | the λ line, now a **measurement** rather than a suspicion: swept, monotone decreasing, optimum off-grid low, **−0.63 t1 unconfirmed shortfall**, constant unchanged |
| §4.6 | the ru wiring items with their `d717bda7` status |
| **§4.7 (new)** | the refreshed per-script table, the projection rules, the evidence-tier wording, the he flag history |
| §5 | **four** generations in the inventory and the uniform `_v3_` suffix that retires the ru/el-uk-bg-mk-he split; "Phase Q is open and produces nothing deployable" replaced by the closed-phase text and the seal's enforcement rule; the fixture table's note that CI checks the header sha and not the emission matrices |
| §6 | replace the `9a6ffdd2` findings table with the `d717bda7` summary |
| §7 | items 9–12 — el's two projection halves, "wire generation 4, not 1/2/3", never quoting a synthesis-holdout level as accuracy (el 92.12 is not "Greek at 92.12"), and never copying a `RESEARCH_ONLY` byte |

Also update `memory/HANDOFF.md`'s "Russian is delivered" paragraph (§1.4).

---

## 4. What NOT to wire

1. **No Yandex-derived anything.** Not a training row, not a distilled teacher, not a fine-tune,
   not an artifact whose pipeline touched one. `YANDEX_LICENSE_RESEARCH.md` (941 lines): no
   licence grant exists anywhere, and the corpus is a protected database under ГК РФ ст. 1334
   whose ст. 1335.1 carve-outs cover research and education but **not a shipped product**.
   Eval-only, held out. `phaseIB-ru-real` reads **89.64** — ten points better than anything
   shippable — and is permanently unusable. If a proposal's accuracy sounds too good, check
   whether it is that model.

   **This now has a second live instance, and it is permanent.** Phase Q (`PHASE_Q.md`, closed
   2026-08-20) ran two twin generators on deliberately separated licence tracks: a **shipping
   track** fitted only to MIT data (FUTO t3 + HWS), which produced every generation-4 artifact in
   §2.2, and a **sealed research track** fitted to Yandex residuals whose generator weights,
   samples, decoder, onnx and dumps are permanently unshippable. Sealed artifacts carry a
   `RESEARCH_ONLY` suffix, live untracked under `~/ctc-train/research_only/`, and never enter
   `ctc/artifacts/`, the registry, or `exports/`; `synth_v3.py` enforces the path prefix
   mechanically. The sealed track produced **one number** — the upper bound U = 85.95 — and no
   bytes. **If a file is not in `ctc/artifacts/`, it is not wirable** — that is the operative
   test, and it is why the registry is the only place §2.2's hashes come from.
2. **No FUTO weights and no FUTO model outputs** in anything trained or shipped. The corpus and
   the decode-algorithm lineage are the permitted inheritance; `NOTICE:46-64` states it correctly.
   Do not "improve" that wording; do not add a FUTO teacher.
3. **Not the `*_synth_ch80*` (v1), `*_synth_v2_ch80*` (v2) or `*_synth_v2full_ch80*` (v3-of-name,
   generation 3) bytes.** All three generations remain in the registry because published numbers
   were measured on them, not because they are deployable. **Generation 4 — `*_synth_v3_ch80*`,
   uniform across all six scripts — is the deployable one**, and §2.2 has the twelve files with
   hashes taken off disk. Phase Q is closed and its gate passed; the previous edition of this
   line said "v2 / v2full remain the deployable generation until `PHASE_Q.md` says otherwise and
   a hash lands in `ctc/artifacts/`" — both conditions are now met.
4. **he's parity flag is history, not a property of the bytes you would wire.** The generation-2
   `he_synth_v2_ch80` fp32 export needed `--parity-tol 2e-3` (sliced residue 1.16e-03 against a
   0.8e-4…7.6e-4 historical envelope; argmax 100/100 on both probes). It is flagged in the
   registry and stays flagged, because that exceedance was real. The generation-3
   `he_synth_v2full_ch80` export **needed no relaxation**: 4.04e-04 at the default 1e-3, argmax
   100/100, and 100/100 on the fp16w probe too. If you read "he is flagged" somewhere, check which
   generation is meant before treating it as a reason to hold he back. **Generation 4
   (`he_synth_v3_ch80`) is likewise clean**: 3.57e-04 at the default 1e-3 with argmax 100/100.
5. **Not a per-node cap back in the trie.** `MAX_CHILDREN = 26` was removed on purpose; the real
   bound is the constructor's alphabet-vs-head-width check.
6. **Not `CtcFeaturizer.normalizeRawX/Y`** in `CtcEngineAdapter`. The shipped encoder was trained
   on letter-box normalization, not FUTO's 4/3 device frame.
7. **Not a synthesis-holdout number as an accuracy claim.** el 90.78, uk 87.67, bg 82.52, mk
   88.68, he 76.86 measure fit to the v2 generator's own distribution. This campaign has now shown
   three separate times — capacity, λ, donor footing — that this probe does not rank what real
   swipes rank, and on the one script where both probes exist the real one was right. What the
   holdouts *do* establish is a margin against a fixed control (+5.1 … +7.9 over the 3×-capacity
   English zero-shot, where on the v1 holdouts every script **lost**). Quote margins, never
   levels, and never next to ru's 79.73 without saying they are different kinds of number.
