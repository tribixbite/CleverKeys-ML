# APP_INTEGRATION_AUDIT — adversarial review of the CTC engine as wired into CleverKeys

**Audited**: `/home/will/git/swype/CleverKeys` @ `a474ddf9` (pulled 2026-08-18; the app tree was
read-only throughout, nothing was committed there).
**Audited against**: this repo @ `85c0c58` — `RESULTS.md`, `MODELS_TABLE.md` §2–§3,
`UNSEALING_4.md`, `PHASE_M.md`, `PHASE_N.md`, `ALT_LAYOUT_EVAL.md`.
**Scope**: every CTC-touching file in the app — `swipe/ctc/*` (12 files), `CtcEngineAdapter`,
`OnnxCtcEmissionModel`, `SwipeEngineRouter`, `InputCoordinator`'s dispatch + prewarm, `Config`,
the settings surface, manifest, assets, test resources, specs, NOTICE, todo.

The commit range under review is wider than the brief assumed. The Termux agent did not stop at
`743b58fa`/`ba2861fa`; it went on to land `2d6dccc7` (per-language λ sweep), **`90b1efe2`
(widen the layout gate to any Latin layout)** and **`524b4448` (enable French, German and
Spanish)**, plus `c613fc1b`, `bdba49d1`, `4e444d7a`, `a4e7e2c1`, `8f415383`, `5fb44d46`,
`eb80a515`, and finally `a474ddf9` (a self-recheck against this repo's 22 newest commits).
The audit therefore covers a **four-language, any-Latin-layout** integration, not the en-only
QWERTY one described in the brief.

---

## 0. Verdict

**The integration is shippable.** The two things the user was worried about are, in the shipped
code, *correct*:

- exactly one ONNX ships, it is the right one, and the fixture↔model↔preset triple is
  mechanically enforced by a pure-JVM test that actually hashes the asset;
- Russian/Cyrillic cannot reach the CTC path in any state — three independent gates, each of
  which is sufficient on its own, and the `ru` preset has zero production callers.

What must change first is **one availability bug** (HIGH-1: a latched ONNX-load failure
silently kills swipe typing for the session with no fallback to geometric — the only finding
where a user loses working functionality) plus three documentation defects severe enough to
mislead the next reader the same way this one was misled (HIGH-2, HIGH-3, MEDIUM-3).
Everything else is cleanup.

**The reported confusion is real, but it is residual — an artifact, not a live defect.** It
survives in four forms, none of which changes what the shipped code does:

1. Pre-integration KDoc still in `src/main` saying the engine is *"deliberately dead code"*,
   has *"NO production implementation"*, and is *"blocked on retrain"* (HIGH-3). These are the
   files a "does the CTC engine actually run?" search lands on.
2. An unbannered planning doc still declaring *"Q1 model choice: SUPERSEDED-PENDING — a new
   model is training"* and naming four candidate models (MEDIUM-3). This is the likeliest
   direct source of "which ONNX?".
3. Four superseded ONNX in the androidTest tree, read by a benchmark that calls one of them
   *"the ship candidate"* at a preset it mislabels `tunedV2` (MEDIUM-4).
4. `sw2345`'s alt-layout numbers still attributed to the ship model in three code sites and
   three doc sites — including a self-contradiction inside `SwipeEngineRouter.kt` and a parity
   audit whose §5 re-injects the numbers its own §2.1.3 refuted (HIGH-2).

On the two questions the user actually asked, the *code* is right and was right throughout.
Every confusion artifact is in prose.

---

## 1. The definitive answer table

### 1a. "Which ONNX?"

| Question | Answer | Evidence |
|---|---|---|
| How many CTC ONNX files ship in the APK? | **One.** `src/main/assets/models/ctc_swipe_encoder.onnx` | `find . -name '*.onnx' -not -path './build/*'` — the only other `src/main/assets` models are the pre-existing neural `swipe_encoder_android.onnx` / `swipe_decoder_android.onnx`, unrelated to CTC |
| Is it the right one? | **Yes.** sha256 `84718e6ebc8020176f27b9668e50922a765c96838307b640a8db9ab0549e88e5`, 3,052,318 B — byte-identical to `ctc/artifacts/phaseM_kd_fresh_w1_s1234_fp16w.onnx` | `sha256sum src/main/assets/models/ctc_swipe_encoder.onnx`; `MODELS_TABLE.md:113` |
| Does the golden fixture pair with it? | **Yes.** `ctc_golden.json` sha256 `2a449c4f2de19505131b396655ae01d3e3c325e40249446ff6e7a40c2b27559c`, its `source_onnx_sha256` field is exactly `["84718e6e…e88e5"]`, its `preset` is `[0.9, 4.0, 0.25, 0.25, 0.9882]` = `tunedV2`. Both copies (`src/test/resources/ctc/`, `src/androidTest/assets/ctc/`) are byte-identical | `python3 -c "json.load(…)"`; `CtcParityTest.kt:176-178` |
| Is the pairing *enforced*, or just documented? | **The identity half is enforced in CI; the behavioural half is not.** `CtcParityTest.fixture_model_and_shipPreset_travelTogether` reads `sha256(models/ctc_swipe_encoder.onnx)`, asserts equality with the fixture's `source_onnx_sha256`, compares the preset term-by-term against `tunedV2`, and hash-compares both fixture copies. It runs on every push. But it never *executes* the ONNX — see HIGH-4 | `CtcParityTest.kt:141-191`; `build.gradle:499`; `.github/workflows/ci.yml:39` |
| Does any code reference a second / Russian / nonexistent model? | **No.** The only model path constant in production is `CtcEngineAdapter.MODEL_ASSET = "models/ctc_swipe_encoder.onnx"` (`CtcEngineAdapter.kt:99`). No `ru`/Cyrillic ONNX path exists anywhere in `src/main` | grep |
| Does anything imply a model the repo doesn't carry? | **Yes, in three places — and this is the likeliest source of the reported confusion.** A stale plan doc still says `**Q1 model choice: SUPERSEDED-PENDING** — a new model is training` (MEDIUM-3); an androidTest benchmark calls `ch128` "the ship candidate" (MEDIUM-4); and `CtcBenchFixture` cites a rival golden fixture identity (MEDIUM-4). All are historical, none affects shipped behaviour |
| Is a model change pending? | **No.** Phase N is terminal (`85c0c58`); its 91.25 headline is a different corpus/trie/preset and is explicitly not comparable. `a474ddf9` records this correctly in the app | `PHASE_N.md`; app commit `a474ddf9` message |

**Bottom line: there was never a second ONNX to be confused about.** One model, correct hash,
correctly paired, mechanically enforced.

### 1b. "Russian and untested languages/layouts"

| Question | Answer | Evidence |
|---|---|---|
| Which languages can reach CTC? | **Exactly en, fr, de, es.** A hardcoded `linkedMapOf` — not extensible at runtime, not affected by langpacks | `CtcLanguageSupport.kt:63-68` |
| Which layouts can reach CTC? | Any layout whose XML declares `script="latin"` **and** which exposes all 26 a–z letters as *centre* key values | `SwipeEngineRouter.kt:136`; `CtcEngineAdapter.kt:266-301` |
| Can a Russian/Cyrillic layout+language combination reach CTC — in ANY state? | **No.** Three independent gates, each individually sufficient | below |
| Can Greek / Hebrew / Arabic / Devanagari? | **No**, same three gates (their XML declares a non-`latin` script) | `grep -ho 'script="[a-z]*"' res/xml/*` → 100 `latin`, 22 `cyrillic`, 1 `greek`, and 14 other non-Latin scripts |
| Is the `tunedRuCkdt` preset reachable? | **No production caller.** Only `CtcLanguagePresetTest` constructs it. `presetFor` can never return it — it branches on `LexiconSource`, and `ru` is not in `SUPPORTED` | grep; `CtcScoringParams.kt:155-165, 205-210` |
| Can a langpack / custom layout / secondary language get around it? | **No.** `SUPPORTED` is compiled-in; a custom XML with no `script` attribute yields `script == null`, and both `isSwipeTypingSupportedForLayout` and `isLatinScript` return false on null → geometric | `Config.kt:1251-1257`; `SwipeEngineRouter.kt:141-142` |
| What happens if it somehow did? | Nothing bad: `buildMappedLayout` returns null on the first missing a–z letter, `supportsLayout` returns false, and the dispatcher hands the swipe to the geometric engine *before any CTC work starts*. No crash, no empty bar, no garbage decode | `CtcEngineAdapter.kt:288`; `InputCoordinator.kt:756-762` |

**The three gates, in dispatch order:**

1. **Layout script** — `SwipeEngineRouter.route` (`SwipeEngineRouter.kt:127-137`) only returns
   `Engine.CTC` when `Config.isSwipeTypingSupportedForLayout` (QWERTY-Latin) is true *or*
   `isLatinScript(script)` is true. A `script="cyrillic"` layout takes neither branch and falls
   to `Engine.GEOMETRIC`.
2. **Active language** — `InputCoordinator.performCtcSwipeTyping` (`InputCoordinator.kt:723-744`)
   reads the live language *before* dispatch and, on `!supportsLanguage`, routes to neural
   (QWERTY-Latin) or geometric (elsewhere). `CtcEngineAdapter.decodeAsync`/`warmUpAsync`
   re-check the same predicate as defense-in-depth (`CtcEngineAdapter.kt:645, 735`).
3. **Alphabet completeness** — `buildMappedLayout` requires all 26 a–z centre labels
   (`CtcEngineAdapter.kt:288 if (seen.any { !it }) return null`). A ЙЦУКЕН layout has zero.

Gate 2 also settles the inverse case the brief asks about — Russian *language* on a *Latin*
layout (e.g. `ru` primary with QWERTY installed). That swipe never reaches CTC; it goes to
neural, which is what it did before the CTC work existed. `InputCoordinator.kt:730-731`
documents exactly why the fallthrough is layout-aware rather than unconditionally neural.

**Untested Latin layouts (Colemak, custom XML): they DO reach CTC, and that is defensible —
narrowly.** Routing them in is justified by three facts and one caveat:

- the ship model takes key geometry as a model input (`layout_keys`), so it is a–z-arrangement-
  agnostic **by design**, not by luck;
- its layout augmentation covers arbitrary a–z arrangements, so Colemak is in-distribution even
  though it was never benchmarked;
- the six layouts that *were* measured span a wide displacement range and the model holds:
  dvorak 91.82 / dvorak-app 91.10, azerty 84.53, qwertz 83.97, german 81.30, spanish 89.53
  (`MODELS_TABLE.md:113`). `ALT_LAYOUT_EVAL.md` is the reason this is credible rather than
  hopeful — it is where the displacement sensitivity was characterised, and where the earlier,
  weaker models visibly failed it (ch128 dvorak **63.04**, `fast_resbn80` **67.28**), which is
  what makes the ship model's 91.82 a property of the training recipe and not of the eval.
- **Caveat**: the floor is not measured. The worst *measured* layout is german at 81.30, and
  the alternative for these layouts is geometric at ~77% top-1, so the expected-value case for
  routing them in is strong. But "Colemak ≥ geometric" is an inference from the aug design, not
  a measurement, and the app states it as though it were measured — see HIGH-2.

---

## 2. Findings

### BLOCKER

None.

### HIGH-1 — a latched ONNX-load failure silently disables swipe typing with no fallback

`CtcEngineAdapter.modelOrNull()` retries a failed session load three times, then latches
(`CtcEngineAdapter.kt:148-177`). The log line says the consequence out loud:

```kotlin
// CtcEngineAdapter.kt:171-173
"CTC encoder load failed (attempt $modelLoadAttempts/$MAX_MODEL_LOAD_ATTEMPTS)" +
    if (latched) " — ctc mode disabled this session" else " — will retry",
```

**"ctc mode disabled this session" is not what happens.** Nothing disables the mode. The
dispatcher keeps routing every swipe to CTC, and the decode path degrades to a silent empty
slate:

```kotlin
// CtcEngineAdapter.kt:669-671
val model = if (lexicon != null) modelOrNull() else null
val result = if (mapped == null || lexicon == null || model == null) {
    PredictionResult(emptyList(), emptyList())
```

which `InputCoordinator` feeds straight into the shared pipeline, clearing the bar. The user in
`ctc` mode loses swipe typing entirely, for the rest of the process lifetime, with no visible
error — while a `hybrid` or `geometric` user on the same device keeps working.

The same hole swallows a lexicon failure: `lexiconFor` returns null on an asset read error
(`CtcEngineAdapter.kt:406-410`) and produces the identical dead bar.

This is the one gap in an otherwise carefully-built fallthrough design. Note the contrast: the
*layout* failure mode was thought through and handed to geometric
(`InputCoordinator.kt:752-762`), and the *language* failure mode was thought through and handed
to neural/geometric (`InputCoordinator.kt:725-744`). The *model/lexicon* failure mode was not.

Triggers are real if uncommon: a low-memory kill mid-extract, a corrupt asset extraction, an
ORT provider init failure on an unusual device, or `MAX_MODEL_LOAD_ATTEMPTS` being consumed by
three transient failures during a memory-pressure episode — on a device family that this repo's
own `MemoryProbe` work confirms hits the growth limit.

**Proposed diff** — make the latch observable and let the dispatcher fall through exactly like
the layout gate already does.

```diff
--- a/src/main/kotlin/tribixbite/cleverkeys/swipe/CtcEngineAdapter.kt
+++ b/src/main/kotlin/tribixbite/cleverkeys/swipe/CtcEngineAdapter.kt
@@
     /** Failed load attempts so far (audit L5: bounded retry, then latch). */
     private var modelLoadAttempts = 0
 
+    /**
+     * True once the ONNX session has permanently failed to load for this adapter
+     * ([MAX_MODEL_LOAD_ATTEMPTS] exhausted). Read from the MAIN thread by the dispatcher,
+     * written from the decode thread by [modelOrNull] — hence `@Volatile`.
+     *
+     * The dispatcher MUST consult this before routing a swipe here: without a session the
+     * decode can only ever produce an empty slate, and an empty slate is indistinguishable
+     * from "no candidates" once it reaches the shared pipeline. Falling through to the
+     * geometric engine keeps swipe typing alive on a device where the session cannot load,
+     * which is the same coverage promise the layout and language gates already make.
+     */
+    @Volatile
+    private var modelPermanentlyUnavailable = false
+
+    /** See [modelPermanentlyUnavailable]. Safe to call from the main thread. */
+    fun isModelPermanentlyUnavailable(): Boolean = modelPermanentlyUnavailable
+
     private fun modelOrNull(): OnnxCtcEmissionModel? {
         emissionModel?.let { return it }
@@
         } catch (e: Exception) {
             modelLoadAttempts++
             val latched = modelLoadAttempts >= MAX_MODEL_LOAD_ATTEMPTS
+            if (latched) modelPermanentlyUnavailable = true
             Log.e(
                 TAG,
-                "CTC encoder load failed (attempt $modelLoadAttempts/$MAX_MODEL_LOAD_ATTEMPTS)" +
-                    if (latched) " — ctc mode disabled this session" else " — will retry",
+                "CTC encoder load failed (attempt $modelLoadAttempts/$MAX_MODEL_LOAD_ATTEMPTS)" +
+                    if (latched) " — falling through to the geometric engine for this session"
+                    else " — will retry",
                 e
             )
             null
         }
     }
```

```diff
--- a/src/main/kotlin/tribixbite/cleverkeys/InputCoordinator.kt
+++ b/src/main/kotlin/tribixbite/cleverkeys/InputCoordinator.kt
@@
         // The router gates on layout METADATA (Latin script); this layout may still lack an
         // a–z key, which yields no CtcLayout and would leave the bar empty. Geometric can
         // decode it, so hand the swipe over rather than degrade coverage. Memoized — the
         // decode below reuses this same geometry build.
-        if (!ctcAdapterOrCreate().supportsLayout(keyboard, params, frameW, frameH)) {
+        //
+        // Same reasoning for a permanently-failed ONNX session: after the bounded retry
+        // budget is spent the adapter can only ever return an empty slate, which the shared
+        // pipeline renders as a cleared bar — i.e. swipe typing would silently stop working
+        // for the rest of the session. Geometric still works, so use it.
+        val ctc = ctcAdapterOrCreate()
+        if (ctc.isModelPermanentlyUnavailable() ||
+            !ctc.supportsLayout(keyboard, params, frameW, frameH)
+        ) {
             performGeometricSwipeTyping(
                 swipedKeys, swipePath, timestamps, ic, editorInfo, resources,
                 wasShiftActive, wasShiftLocked
             )
             return
         }
```

and the matching prewarm guard at `InputCoordinator.kt:821`:

```diff
-                    val ctcServes = CtcEngineAdapter.supportsLanguage(language) &&
+                    val ctcServes = !ctc.isModelPermanentlyUnavailable() &&
+                        CtcEngineAdapter.supportsLanguage(language) &&
                         ctc.supportsLayout(keyboard, params, frameW, frameH)
```

A follow-up worth considering but *not* proposed here (it is a larger change): give `lexiconFor`
the same treatment, either by hoisting a `canServe(language)` probe onto the prewarm path or by
widening `decodeAsync`'s callback to distinguish "engine failed" from "no candidates".

### HIGH-2 — `sw2345`'s alt-layout numbers are still attributed to the ship model in three of four in-code sites, and `SwipeEngineRouter.kt` now contradicts itself

`a474ddf9` set out to fix exactly this ("our alt-layout numbers named the wrong model") and
**fixed one occurrence of four**. Current state of the app tree:

| Site | Text | Status |
|---|---|---|
| `SwipeEngineRouter.kt:22-23` | "the ship model was validated on alt-layouts during training: dvorak 89.87 / dvorak-app-geometry 88.98 top-1 (3 seeds, en lexicon…)" | **WRONG** — `sw2345` (`MODELS_TABLE.md:139`) |
| `SwipeEngineRouter.kt:82-83` | "dvorak 91.82 / dvorak-app-geometry 91.10 top-1, 3 seeds" | correct (`MODELS_TABLE.md:113`) — this is the one `a474ddf9` fixed |
| `SwipeEngineRouter.kt:131-132` | "validated on alt-layouts (dvorak 89.87 top-1 — **see the class KDoc**)" | **WRONG**, and the cross-reference now points at a contradicting number |
| `SwipeEngineRouterTest.kt:91` | "validated on dvorak 89.87 / dvorak-app-geometry 88.98 top-1, so" | **WRONG** |

Two more doc sites, plus one that actively re-injects the error:

- `docs/eval/2026-08-15-ctc-per-language-lambda.md:101` — "german 80.64 vs spanish 88.45 top-1"
  → ship model is german **81.30** / spanish **89.53**.
- `docs/eval/2026-08-15-ctc-per-language-lambda.md:112` — "the campaign's 88.98 dvorak-app
  figure" → **91.10**. (The sentence's argument survives — 92.72 is still above 91.10 — but the
  number is `sw2345`'s.)
- `docs/audit/2026-08-17-neural-vs-ctc-parity.md:619-623` — **finding 13 contradicts §2.1.3 of
  its own document.** §2.1.3 (`:263-284`) resolves the slot-count question (az26 = what the app
  builds) and corrects the numbers; finding 13 still says *"Resolve before quoting azerty 83.81
  / qwertz 83.01 / german 80.64 / spanish 88.45 as app-relevant"* — i.e. it names the `sw2345`
  set as the figures to validate and quote. Strike it or mark it resolved.

The same audit at `:282` claims `SwipeEngineRouter.kt` was "Corrected". It was not — only
`CtcLanguageSupport.kt` and the spec were.

Why this matters more than a stale comment usually would: `sw2345` was **never decoded on
test**, and the number is being used *in the file that implements the layout gate* as the
justification for widening that gate. The gate is justifiable (see §1b) — but on the ship
model's 91.82, which is a test-validated model's alt-layout figure, not on a val-only
finalist's. The error is conservative in magnitude (every corrected value is higher) and
directionally harmless, but a reviewer who checks the citation finds it names a model the app
does not ship, and finds the same file asserting both numbers.

**Proposed diff:**

```diff
--- a/src/main/kotlin/tribixbite/cleverkeys/swipe/SwipeEngineRouter.kt
+++ b/src/main/kotlin/tribixbite/cleverkeys/swipe/SwipeEngineRouter.kt
@@ -19,10 +19,13 @@
  *    LAYOUT dimension (gate widened 2026-08-15): unlike the QWERTY-trained transformer,
  *    the CTC encoder is layout-agnostic — key geometry is a model input (`layout_keys`) —
  *    and the ship model was validated on alt-layouts during training: dvorak 89.87 /
- *    dvorak-app-geometry 88.98 top-1 (3 seeds, en lexicon — the SAME `en_enhanced` trie +
- *    tunedV2 λ the app ships). So Latin non-QWERTY layouts (Dvorak, Colemak, AZERTY, …)
+ *    and the ship model was validated on alt-layouts during training: dvorak **91.82** /
+ *    dvorak-app-geometry **91.10** top-1 (3 seeds, en lexicon — the SAME `en_enhanced` trie
+ *    + tunedV2 λ the app ships; CleverKeys-ML `MODELS_TABLE.md:113`, the `az26` arm).
+ *    Do NOT quote 89.87 / 88.98 here: those are `sw2345`'s (`MODELS_TABLE.md:139`), a
+ *    superseded Phase-J model never decoded on test. So Latin non-QWERTY layouts (Dvorak, AZERTY, …)
  *    route CTC instead of geometric (~77% top-1), a ~13 pt gain for English users there.
+ *    Colemak specifically was never benchmarked — it is covered by the encoder's arbitrary
+ *    a–z layout augmentation, which is a design property, not a measurement.
@@ -128,9 +131,9 @@
         // Gate widening 2026-08-15 (ctc mode only): the CTC encoder is layout-agnostic
-        // (key geometry is a model input) and was validated on alt-layouts (dvorak 89.87
+        // (key geometry is a model input) and was validated on alt-layouts (dvorak 91.82
         // top-1 — see the class KDoc), so ANY known-Latin layout routes CTC. The
```

and the same 89.87 → 91.82 / 88.98 → 91.10 substitution at `SwipeEngineRouterTest.kt:91` and
`docs/eval/2026-08-15-ctc-per-language-lambda.md:112`.

### HIGH-3 — three production files still document the engine as unimplemented, dead, and blocked on a retrain

This is the clearest surviving artifact of the reported confusion. Three KDoc blocks in
`src/main` describe the pre-integration world as though it were current, and they are the first
thing a reader of the CTC package encounters:

`swipe/ctc/CtcSwipeDecoder.kt:10-15` (on `CtcEmissionModel`):
> "No such model exists in this repo, and producing one is a hard fork … Consequently there is
> intentionally **NO production implementation of this interface here**: the featurizer, trie,
> and beam are all complete and tested, but **the module cannot decode a real swipe end-to-end
> until a CTC model lands.**"

`swipe/ctc/CtcSwipeDecoder.kt:36-39` (on `CtcSwipeDecoder` itself):
> "It is **deliberately dead code today**: without a `CtcEmissionModel` implementation (the
> retrain fork) it cannot be constructed against a real model, so **nothing in the IME
> references it**."

`swipe/ctc/CtcEmissions.kt:12-16`:
> "The model that *produces* these emissions — FUTO's non-autoregressive CTC
> `honorable_sturgeon` encoder … is **OUT OF SCOPE for this module** (it requires a CTC
> retrain/re-export; see `docs/specs/ctc-swipe-engine.md`, **'Blocked on retrain'**)."

Every sentence is false. `OnnxCtcEmissionModel` **is** the production implementation;
`CtcSwipeDecoder` **is** constructed on every decode (`CtcEngineAdapter.kt:517`); the emissions
come from the app's own trained encoder, not FUTO's. A maintainer reading only these files would
conclude the CTC engine does not work — which is very plausibly a chunk of the reported
confusion, since these are the files a "does the CTC engine actually run?" search lands on.

`swipe/ctc/CtcLayout.kt:12` has the milder version of the same error — "so a **future**
layout-parameterized encoder can serve any layout without re-baking QWERTY" — when the shipped
encoder *is* layout-parameterized, and that is the entire basis of the widened gate.

**Proposed diff** (abbreviated — the same treatment for all four):

```diff
--- a/src/main/kotlin/tribixbite/cleverkeys/swipe/ctc/CtcSwipeDecoder.kt
+++ b/src/main/kotlin/tribixbite/cleverkeys/swipe/ctc/CtcSwipeDecoder.kt
@@
-/**
- * Source of per-frame CTC log-emissions for a featurized swipe path.
- *
- * ## This is the retrain-fork boundary
- * FUTO's emissions come from a non-autoregressive CTC encoder (`honorable_sturgeon`),
- * optionally sharpened by a per-layout refinement head (`magic_macaw`) — a DIFFERENT
- * model family from CleverKeys' shipped autoregressive transformer. No such model exists
- * in this repo, and producing one is a hard fork (CTC retrain + ONNX/ExecuTorch export;
- * see `docs/specs/ctc-swipe-engine.md`, Phase B). Consequently there is intentionally NO
- * production implementation of this interface here: the featurizer, trie, and beam are all
- * complete and tested, but the module cannot decode a real swipe end-to-end until a CTC
- * model lands. Tests supply emissions directly (golden matrices frozen from the Python
- * port), which is exactly how the decode path is validated today.
+/**
+ * Source of per-frame CTC log-emissions for a featurized swipe path.
+ *
+ * ## The seam, and what closes it
+ * The production implementation is
+ * [tribixbite.cleverkeys.swipe.OnnxCtcEmissionModel], which runs the CleverKeys-trained
+ * encoder bundled as `models/ctc_swipe_encoder.onnx` (CleverKeys-ML Phase M finalist
+ * `phaseM_kd_fresh_w1`, fp16w). This is a from-scratch CTC model in the same *family* as
+ * FUTO's `honorable_sturgeon`, trained by this project — no FUTO weights are used (see
+ * `NOTICE`). The interface stays abstract so the decode path can also be driven from
+ * golden emission matrices frozen from the Python port, which is how `CtcParityTest`
+ * validates the beam without a device.
  */
@@
- * This wires the three DONE pieces (featurizer, emission model seam, trie beam) into the
- * one call shape a future engine-selector `ctc` mode would invoke. It is deliberately
- * dead code today: without a [CtcEmissionModel] implementation (the retrain fork) it
- * cannot be constructed against a real model, so nothing in the IME references it. See
- * `docs/specs/ctc-swipe-engine.md` for how this slots behind `swipe_engine_mode`.
+ * This wires the three pieces (featurizer, emission model, trie beam) into the one call
+ * shape the `ctc` value of `swipe_engine_mode` invokes. It is LIVE: `CtcEngineAdapter`
+ * memoizes one instance per (layout, trie, beam width, language) and calls [decode] on
+ * every swipe. See `docs/specs/ctc-swipe-engine.md`.
@@
- * @property model the CTC emission source (the missing piece — see [CtcEmissionModel]).
+ * @property model the CTC emission source — [OnnxCtcEmissionModel] in production.
```

```diff
--- a/src/main/kotlin/tribixbite/cleverkeys/swipe/ctc/CtcEmissions.kt
+++ b/src/main/kotlin/tribixbite/cleverkeys/swipe/ctc/CtcEmissions.kt
@@
- * ## Provenance boundary (why this is an INPUT, not something we compute)
- * The model that *produces* these emissions — FUTO's non-autoregressive CTC
- * `honorable_sturgeon` encoder, optionally sharpened by the `magic_macaw` refinement
- * head — is a different model family from CleverKeys' shipped autoregressive transformer
- * and is OUT OF SCOPE for this module (it requires a CTC retrain/re-export; see
- * `docs/specs/ctc-swipe-engine.md`, "Blocked on retrain"). This module decodes a *given*
- * emission matrix, so the whole decode path is testable today against golden emissions
- * frozen from the Python port without any model at all.
+ * ## Why this is an INPUT and not something we compute
+ * Emissions are produced by the bundled CTC encoder via
+ * [tribixbite.cleverkeys.swipe.OnnxCtcEmissionModel] and handed to the beam. Keeping them
+ * a plain value type is what makes the whole decode path testable without a model at all,
+ * against golden emissions frozen from the Python port (`CtcParityTest`).
```

```diff
--- a/src/main/kotlin/tribixbite/cleverkeys/swipe/ctc/CtcLayout.kt
+++ b/src/main/kotlin/tribixbite/cleverkeys/swipe/ctc/CtcLayout.kt
@@
- * For FUTO's
- * `en_qwerty` this ordering is alphabetical a..z, but the type stays layout-agnostic so a
- * future layout-parameterized encoder can serve any layout without re-baking QWERTY.
+ * For the shipped a–z alphabet this ordering is alphabetical a..z. The shipped encoder IS
+ * layout-parameterized — key centers are a model input (`layout_keys`) — which is what lets
+ * one model serve Dvorak/AZERTY/QWERTZ without re-baking QWERTY, and is the basis of the
+ * router's Latin-layout gate.
```

### HIGH-4 — the fixture rule's *behavioural* half never runs automatically

The identity check is strong: `CtcParityTest` hashes the shipped asset against the fixture's
`source_onnx_sha256` (`CtcParityTest.kt:173-180`), hash-compares both fixture copies
(`:184-190`), and pins the preset term-by-term against `tunedV2` (`:148-170`) — with the same
five constants independently pinned again at `CtcModuleTest.kt:128-137` and
`CtcLanguagePresetTest.kt:69-73`. It runs on every push (`build.gradle:499`;
`.github/workflows/ci.yml:39` runs `runPureTests`).

But **that test never executes the ONNX.** It decodes Kotlin against emission matrices *stored
in* the fixture. The only thing that checks the artifact actually *produces* those emissions is
`CtcEmissionModelParityTest` (`src/androidTest/.../CtcEmissionModelParityTest.kt:80-107`,
`EMISSION_TOL = 2e-3`), which is instrumented — and **no workflow runs instrumented tests.**
None of the 7 GitHub workflows invokes `connectedAndroidTest` or `ew-cli`.

The consequence is precise: a model swap that updates the fixture's *header* sha but leaves the
*emission matrices* stale would pass CI green. The rule is enforced against renaming and against
preset drift, but not against the thing it was written to prevent — a model whose outputs no
longer match the fixture. The same gap silences `CtcLatencyGateTest`.

This does not affect the current ship (the shipped triple is verified consistent above), but it
means the guarantee is weaker than the spec claims, and it is what would let a *future* model
swap go wrong quietly.

*Proposed fix* — cheapest credible option, no new infrastructure: add a pure-JVM assertion that
the model's ONNX graph inputs/outputs and the fixture's declared shapes agree, plus a CI note in
`docs/specs/ctc-swipe-engine.md`'s fixture-rule section stating plainly that the emission check
is device-only. The real fix is running the two instrumented tests on ew-cli in the release
checklist; `.claude/skills/ew-cli-testing.md` already documents how.

Two smaller cracks in the same rule:
- `CtcParityTest.kt:38` hardcodes `MODEL_ASSET_PATH = "src/main/assets/models/ctc_swipe_encoder.onnx"`
  rather than deriving it from `CtcEngineAdapter.MODEL_ASSET` (`CtcEngineAdapter.kt:99`). A
  production asset rename leaves the test hashing an orphan. One-line fix.
- The preset comparison pins 5 terms but **not `beamWidth`** — the fixture decodes at 32, the
  ship at 100 — and the beam cases run against a **7-word lexicon**. That proves algorithmic
  parity, which is what it is for; it does not prove shipping-scale ranking. Worth stating in
  the spec so nobody reads the gate as broader than it is.

### MEDIUM

**MEDIUM-1 — the ORT session is never closed, and `close()` has zero callers.**
`CtcEngineAdapter.shutdown()` (`CtcEngineAdapter.kt:761-763`) calls only `tasks.shutdown()`,
which is `executor.shutdownNow()` — it interrupts but does not join. The KDoc's rationale is
sound as far as it goes ("closing a session mid-run is UB in ORT"), but the result is that
`OnnxCtcEmissionModel.close()` (`OnnxCtcEmissionModel.kt:100-106`) is dead code and the ~3 MB
native session leaks once per `InputCoordinator` lifecycle. `CleanupHandler.kt:58` does wire
`inputCoordinator?.shutdown()`, so the path runs; it just doesn't free the session. Bounded
(one leak per IME service destroy/recreate within a process, and the process usually dies), but
this is the same repo that just spent a campaign on an OOM.

*Proposed diff*: graceful-then-forceful shutdown so the close can run on the owning thread.

```diff
--- a/src/main/kotlin/tribixbite/cleverkeys/swipe/CtcEngineAdapter.kt
+++ b/src/main/kotlin/tribixbite/cleverkeys/swipe/CtcEngineAdapter.kt
     fun shutdown() {
-        tasks.shutdown()
+        // Queue the session close on the OWNING thread before tearing the runner down, so it
+        // runs after any in-flight `session.run` returns (closing mid-run is UB in ORT). The
+        // runner then drains for a bounded time and is force-stopped; if the close did not get
+        // to run the session is reclaimed at process death, which is the pre-existing posture.
+        val model = emissionModel
+        emissionModel = null
+        if (model != null) tasks.submitBackground { model.close() }
+        tasks.shutdownGracefully(timeoutMs = 250L)
     }
```
(needs a small `shutdownGracefully` on `PredictionTaskRunner`: `executor.shutdown()`,
`awaitTermination`, then `shutdownNow()`.)

**MEDIUM-2 — three `settle = true` MemoryProbe marks sit inside the CTC trie build, on the
decode thread.** `CtcEngineAdapter.kt:425, 429, 467` each call `MemoryProbe.mark(..., settle =
true)`, and `settledUsedBytes()` runs `SETTLE_PASSES = 2` explicit GCs with a `SETTLE_PAUSE_MS
= 120` sleep after each (`MemoryProbe.kt:48-51, 83-94`). That is **~720 ms** added to the first
CTC decode. It is correctly gated on `BuildConfig.ENABLE_VERBOSE_LOGGING`
(`MemoryProbe.kt:110`), so F-Droid release users pay nothing — but instrumented runs are built
with `LOCAL_BUILD=true`, which means **the latency gate and the on-device benchmarks measure a
trie build inflated by ~0.7 s**, and any verbose release handed to a tester gets a visibly
sluggish first swipe. Either drop these to `settle = false` now that the OOM work is done, or
have `CtcLatencyGateTest` assert `!MemoryProbe.enabled` (or subtract the settle cost) so the
measurement can't be silently wrong.

**MEDIUM-3 — an unbannered plan doc says a model swap is pending. This is almost certainly
where "which ONNX?" came from.**
`docs/audit/remediation-plans/ctc-integration-execution-brief.md` is a pre-decision planning
document with **no superseded banner**, sitting one directory from the live specs. Read as
current, it says:

- `:86-88` — `**Q1 model choice: SUPERSEDED-PENDING** — a new model is training`
- `:21` — the fixture is `139,728 B, sha ce3b5456ad13…, source_onnx = resbn80g_s1234.onnx
  (sha 330cadfb…)` (shipped: 140,462 B, `2a449c4f…`, `phaseM_kd_fresh_w1`)
- `:22` — "The shipping preset must match the fixture's model: **for resbn80g** it is …"
- `:37` (D1) — names `ch128` / `resbn80g` / `resbn192i_fp16w` as the live candidate set
- `:38` (D2) — "`tunedV2` = the shipped model's fixture preset, **whichever ships**"
- `:42` (D6) — "**VALID for resbn80g** … QWERTY-Latin→CTC, else geometric is right" — blessing
  a routing decision that has since been reversed
- `:74`, `:78` — rollback plan "resbn192i → resbn80g"

Every one of these was true when written and is false now. An agent asked "which ONNX should we
use?" and pointed at the app repo finds this file naming four candidates and declaring the
choice pending. **This is the single highest-value doc fix in the audit.**

*Proposed diff*: prepend
`> **SUPERSEDED 2026-08-14 — historical planning document, do not read as current.** The model
question closed at `phaseM_kd_fresh_w1_s1234_fp16w` (sha `84718e6e…`, 3,052,318 B), landed in
`3b9dd666`. The shipped fixture is sha `2a449c4f…`. D1/D2/D6 and Q1 are historical; the routing
decision in D6 was reversed by `90b1efe2`. For as-built state read
`docs/specs/ctc-swipe-engine.md`.`

**MEDIUM-4 — ~11 MB of superseded ONNX in `src/androidTest/assets/ctc_bench/`, and the
benchmark that reads them still calls one "the ship candidate".**
`ch192_s1234.onnx` (6,144,249 B), `ch128_s1234.onnx` (2,799,865 B), `fast_resbn80_s1234.onnx`
(1,142,727 B), `fast_resbn72_s1234.onnx` (944,487 B). All Campaign-2-era. They are
`androidTest`-only — `build.gradle:228-230` overrides only `kotlin.srcDirs` for that source set,
so `src/androidTest/assets` stays the AGP default androidTest asset dir and never enters the
release APK — but they are four extra CTC ONNX a "which model do we use?" search surfaces.

Worse, their only consumer states the wrong thing twice:

- `CtcOnnxLatencyBenchmarkTest.kt:46-48` — "the end-to-end post-gesture cost for **the ship
  candidate**". `ch128` is not, and never became, the ship candidate.
- `:351-361` — the test is named `fullDecodePath_ch128_beam100_tunedV2` but its inline params
  are the **E1** preset (γ 1.05, λ 1.1, β 0.2, γp 0.3734, βp 0.9882), not today's `tunedV2`
  (0.9 / 4.0 / 0.25 / 0.25 / 0.9882). The comment at `:355-356` explains why they were inlined —
  "`CtcScoringParams.tunedV2` lands with the engine wiring" — which has since happened and
  diverged.
- `CtcBenchFixture.kt:8-9` cites golden fixture `a18ea58c…` / 140,204 B, a **different fixture
  identity** from the shipped `2a449c4f…` / 140,462 B. The cx/cy values do match, so it is a
  stale citation rather than a real second fixture — but two fixture identities now coexist in
  the test tree, which is exactly the ambiguity the fixture rule exists to prevent.

*Proposed*: either delete `ctc_bench/` and the benchmark (the arch comparison is settled and
lives in this repo's `PHASE_F`/`PHASE_I` latency tables), or keep them behind a one-line
`ctc_bench/README` stating they are **superseded arch-comparison artifacts, not candidates**,
rename the misleading test, and fix the two stale citations.

**MEDIUM-5 — the settings screen still claims QWERTY-only, and it is the only user-visible
statement of the engine's scope.** `CtcSettingsActivity.kt:82-89`:
> "Tuning for the CTC swipe engine (**QWERTY layouts** under the CTC prediction engine)."

Wrong since `90b1efe2`, and it contradicts the *translated* picker string, which is correct
(`res/values/strings.xml:122`: "CTC beam on Latin layouts in English, French, German and
Spanish…"). The whole activity is hardcoded English with no `stringResource`, and the "100 is
the validated default" prose (`CtcSettingsActivity.kt:94-96`) is decoupled from
`Defaults.CTC_BEAM_WIDTH`. Same stale "on QWERTY" phrasing in `Config.kt:318-319`,
`Config.kt:679`, `InputCoordinator.kt:501-503`, `backup/SettingsDefaults.kt:261`.

*Proposed diff*:
```diff
--- a/src/main/kotlin/tribixbite/cleverkeys/CtcSettingsActivity.kt
+++ b/src/main/kotlin/tribixbite/cleverkeys/CtcSettingsActivity.kt
-            "Tuning for the CTC swipe engine (QWERTY layouts under the CTC prediction " +
-                "engine). Scoring constants are calibrated offline and not user-tunable."
+            "Tuning for the CTC swipe engine. It serves English, French, German and Spanish " +
+                "on any Latin layout that has all 26 letters; other languages and layouts " +
+                "fall through to the neural or geometric engine automatically. Scoring " +
+                "constants are calibrated offline and not user-tunable."
-            "Hypotheses kept per frame in the trie beam. 100 is the validated default; " +
+            "Hypotheses kept per frame in the trie beam. ${Defaults.CTC_BEAM_WIDTH} is the " +
+                "validated default; " +
                 "higher costs CPU per swipe for marginal accuracy."
```

**MEDIUM-6 — imported profiles can persist out-of-range CTC prefs.**
`backup/SettingsValidation.kt` has no case for `ctc_beam_width` (falls to `else -> true`,
line 245) and none for `swipe_engine_mode` (line 342), so an import can write
`ctc_beam_width = 99999` or `swipe_engine_mode = "banana"`. Runtime blast radius is nil — both
are clamped/defaulted on read (`Config.kt:979, 984`, `CtcEngineAdapter.kt:676`,
`Mode.fromPref`) — but the bad value persists, and the slider will then display a clamped value
that disagrees with the stored pref. Inconsistent with `neural_beam_width`, which *is*
validated (`SettingsValidation.kt:226`).

*Proposed diff*: add `"ctc_beam_width" -> value in 10..300` to `validateInt` (and to
`isIntKey`), and `"swipe_engine_mode" -> value in setOf("neural","hybrid","geometric","ctc")`
to `validateString`.

**MEDIUM-7 — nothing tells a user that CTC isn't serving their language.** The picker offers
"CTC" unconditionally (`ui/settings/sections/NeuralPredictionSection.kt:51-70`); the only
"unsupported" warning card in the section is hard-gated to `swipeEngineMode == "neural"`
(line 72-95). A Russian user can select CTC, get neural silently, and have no way to learn
that. The behaviour is correct — the *feedback* is missing. Cheapest fix: widen that existing
card's condition to also fire when `swipeEngineMode == "ctc"` and the active language is not in
`CtcLanguageSupport.SUPPORTED`, with text naming the engine that will actually run.

**MEDIUM-8 — README and the architecture/wiki docs all still say "English on QWERTY".** The
gate was widened on 2026-08-15 (`90b1efe2`) and fr/de/es enabled on 2026-08-16 (`524b4448`);
these were not updated. They **under**-claim, which is the safe direction, but they are the
docs a user or a new contributor reads first:

| Site | Verbatim | Reality |
|---|---|---|
| `README.md:168` | "a 2.9 MB CleverKeys-trained model **for English on QWERTY** that scores 89.3% top-1" | en/fr/de/es on any a–z-complete Latin layout |
| `README.md:251-252` | "**CTC** engine currently covers **English on QWERTY** (other languages fall back to neural, non-QWERTY layouts to geometric)." | same |
| `docs/ARCHITECTURE_MASTER.md:280-281` | "routing is CTC **on QWERTY for English**, neural for other languages on QWERTY, geometric on non-QWERTY layouts" | same |
| `docs/ARCHITECTURE_MASTER.md:298-299` | lexicon = "`dictionaries/en_enhanced.json` a-z-stripped" only | omits the fr/de/es CKDT `.bin` path entirely |
| `docs/ARCHITECTURE_MASTER.md` λ row | "`lambda` \| 4.0" | λ is per-lexicon-scale: 4.0 en / 2.0 CKDT (`presetFor`) |
| `docs/wiki/specs/typing/swipe-typing-spec.md:36,45,47,52` | "CTC → CtcEngineAdapter (**en only**…)"; routing table's `ctc` non-QWERTY cell = GEOMETRIC | en/fr/de/es; non-QWERTY **Latin** = CTC |
| `docs/wiki/settings/neural-settings.md:57` | "CTC model **on QWERTY for English**" | same |
| `docs/specs/ctc-swipe-engine.md:10` | "(QWERTY→CTC, other layouts→geometric hedge)" | contradicts its own `:44` |
| `docs/specs/ctc-swipe-engine.md:34-36` | router gate described as only `Config.isSwipeTypingSupportedForLayout` | the CTC branch is `isLatinScript(script)` (`SwipeEngineRouter.kt:136`) |
| `docs/specs/ctc-swipe-engine.md:168-169` | missing-letter layouts → empty result, "unexpected behind the router's QWERTY gate" | now **expected** — `latn_qwerty_az`/`latn_qwerty_tly` hit it and fall to geometric |

Correct as written and needing no change: `docs/wiki/typing/swipe-typing.md:73-88` (the one
user-facing doc with the current gate), `CHANGELOG.md:32-36`, `RELEASE_NOTES.md:5`, and all 22
copies of `swipe_engine_mode_desc` — including `res/values-ru/strings.xml:585`, which correctly
does **not** claim Russian.

**MEDIUM-9 — `memory/todo.md` carries three stale CTC entries, one of them the exact opposite
of the truth.**
- `:140` — "The CTC engines are **demo-only** — wiring `ctc` into `swipe_engine_mode` on-device
  is still item B below." False since 2026-08-08; `ctc` is wired and user-selectable.
- `:224-227` — a P2 doc-correction item whose *both* halves are already done (the spec now
  reads german 81.66, and the slot-count question was resolved as az26 by `a474ddf9`).
- `:184-206` — the Russian section is factually accurate on the checkpoints but framed as "an
  export step, not a training run", which reads closer-to-shipping than ground truth. **No ru
  ONNX has ever been packaged, all ru evidence is val-tier permanently, and no sealed Cyrillic
  split has ever existed.** Lead the section with that, then the six blockers.

Related: `docs/eval/2026-08-15-ctc-per-language-lambda.md:78` lists "fr, de, es **(and ru per
the prior sweep)**" in a table of recommendations for `presetFor(language)`. The ru sweep is on
the E1 footing, not tunedV2, so λ alone does not transfer — the caveat that `CtcScoringParams`'
own KDoc makes carefully should be repeated inline here.

### LOW

- **LOW-1 — `MappedLayout.padded` is written and never read.** Built at
  `CtcEngineAdapter.kt:299`, declared at `:184`, zero readers — `CtcSwipeDecoder` recomputes it
  in its own constructor (`CtcSwipeDecoder.kt:52`). Harmless (64×2 floats) but it advertises a
  caching intent that isn't realised. Either drop the field or have the decoder accept it.
- **LOW-2 — `CtcScoringParams`' documented score formula has a term the code doesn't.**
  `CtcScoringParams.kt:12` writes `final_score = ctc/max(len,1)^gamma + weight * beta * len +
  lambda*logFreq`, but there is no `weight` field and `CtcBeamDecoder.kt:162-163` computes
  `params.beta * len`. FUTO's formula has the `weight`; ours folded it in. Drop it from the
  KDoc so the formula reads as implemented.
- **LOW-3 — `CtcFeaturizer.normalizeRawX`/`normalizeRawY`/`VERTICAL_ASPECT` are production-dead
  and are a trap.** Only `CtcModuleTest` calls them. The adapter explicitly warns against them
  (`CtcEngineAdapter.kt:51` "do not use `CtcFeaturizer.normalizeRawY` here") because the shipped
  encoder was trained on letter-box normalization, not FUTO's 4/3 device frame. A future
  contributor "fixing" the adapter to use the module's own normalizer would silently break every
  decode. Add `@Deprecated`, or move them into the test source set.
- **LOW-4 — inconsistent handling of an uninitialised `Config` on the decode path.**
  `CtcEngineAdapter.kt:154-158` guards `Config.globalConfig()` with a try/catch and a
  `Defaults` fallback; `CtcEngineAdapter.kt:676` calls it bare. The bare one is inside the
  outer try so it can't crash, but it turns a config-timing problem into a logged "CTC decode
  failed" plus an empty bar rather than a decode at the default beam width. Use
  `Config.globalConfigOrNull()?.ctc_beam_width ?: Defaults.CTC_BEAM_WIDTH`.
- **LOW-5 — a null keyboard in `ctc` mode drops the swipe instead of falling through, and the
  two `route` overloads disagree about it.** `SwipeEngineRouter.route(null, Mode.CTC)` returns
  `Engine.CTC` (`SwipeEngineRouter.kt:119`), while the string overload
  `route(null, null, Mode.CTC)` returns `GEOMETRIC` — and the test suite pins only the *string*
  behaviour (`SwipeEngineRouterTest.kt:129-130`). The production `KeyboardData?` overload is
  never exercised for `Mode.CTC` at all. Downstream, `performCtcSwipeTyping` then does
  `keyboardView.getKeyboard() ?: return` (`InputCoordinator.kt:746`) — a silent drop, where
  `Mode.NEURAL` would reach `dispatchNeuralSwipeTyping`. Practically unreachable (a swipe implies
  a rendered keyboard), but it is the one asymmetry in an otherwise symmetric design, and the
  divergence is untested.
- **LOW-9 — the second gate has no negative test.** `SwipeEngineRouterTest` proves Cyrillic and
  Greek are blocked at the *router* (`:120-125`), and `CtcMultiLanguageInstrumentedTest:457`
  proves `supportsLayout` accepts `latn_qwerty_us` — but nothing feeds a ЙЦУКЕН `KeyboardData`
  to `CtcEngineAdapter.supportsLayout` to prove the alphabet gate catches a layout that got past
  the script check. The claim "Cyrillic can never reach CTC" therefore rests on the script
  string alone in the test suite, even though it rests on three gates in the code. A
  mis-tagged (`script="latin"`) Cyrillic layout is the case a negative test would cover — cheap
  to add as a pure-JVM test using the existing `GeoLayoutFixtures` ЙЦУКЕН rows.
- **LOW-10 — the language-gate *ordering* is enforced only by a source-text scan.**
  `CoreImeHygieneDriftTest.kt:208-211, 248` greps `InputCoordinator`'s source for
  `CtcEngineAdapter.supportsLanguage(` and `supportsLayout` appearing before dispatch. That is
  drift detection, not behaviour — a refactor that preserved the strings but reordered the calls
  would pass. Acceptable given the adapter's defense-in-depth re-check, but worth naming.
- **LOW-6 — the golden fixture ships a developer's absolute path.**
  `ctc_golden.json` `source_onnx = "/home/will/ctc-train/ckpt/v2kd-fresh-w1/kd_fp16w.onnx"`.
  The sha256 next to it is the load-bearing field; the path just leaks a username into two
  checked-in test resources. Replace with the artifact name.
- **LOW-7 — no proguard keep for the CTC classes.** `proguard-rules.pro` keeps
  `tribixbite.cleverkeys.onnx.**` but nothing for `swipe.ctc.**` / `CtcEngineAdapter` /
  `OnnxCtcEmissionModel`. Inert today (`minifyEnabled false` on both build types,
  `build.gradle:268, 287`), and the CTC classes aren't reflection-loaded, so this is a latent
  asymmetry rather than a bug. Worth a rule if minification is ever turned on.
- **LOW-8 — the settings search entry tags CTC with `"futo"`.** `SettingsActivity.kt:583`
  keywords are `listOf("ctc", "futo", "swipe engine", "beam", "trie")`. The shipped model is
  CleverKeys-trained and uses no FUTO weights (`NOTICE:48-55`); "futo" refers only to the
  training *corpus* and the decode *algorithm* lineage. A user-visible search keyword is a
  slightly awkward place to associate the engine with another project.

### INFO — things that are right and should not be "fixed"

- **`NOTICE:46-64` is careful and correct**: it states that the encoder was trained from scratch
  on data derived from the FUTO corpus, that no FUTO weights or outputs were used, that the
  FUTO Model Weights License therefore doesn't apply to the asset, and that the decode
  algorithms are a clean-room port of the GPL-3.0 `swipe-library`. This is the right shape.
- **`aaptOptions { noCompress 'onnx' }`** (`build.gradle:135-136`) covers the CTC asset, so it
  is stored uncompressed and mmap-able.
- **`CtcLexiconMerge.ordinals`** carries an explicit API-21 hazard note about
  `Map#putIfAbsent` being API 24 — exactly the class of bug `8f415383` fixed elsewhere.
- **`CtcAzProjection`** deliberately does *not* reuse `AccentNormalizer` (which expands ß→ss)
  and documents why: the λ sweep's lexicons were built with the drop-not-expand policy, so
  reusing the "nicer" normalizer would silently decalibrate λ. Good discipline.
- **The `tunedRuCkdt` KDoc** (`CtcScoringParams.kt:167-204`) is the strongest documentation in
  the package: it states the footing mismatch, the val-only evidence tier, the fact that the
  seal can never be spent on Cyrillic, and the user-dictionary caveat that travels with λ. It
  is unreachable code kept for a stated reason, with the reason stated. Keep it.
- **The three-gate design** for language/layout fallthrough never reduces coverage relative to
  `hybrid`, and `InputCoordinator.kt:730-731` explains the non-obvious part (why the language
  fallthrough must be layout-aware). This is the best-reasoned part of the integration.

---

## 3. What must change before shipping

Ordered. Only the first is a correctness must; 2–4 are the anti-confusion set and are all
cheap.

1. **HIGH-1** — fall through to geometric when the ONNX session has permanently failed. The
   only finding where a user loses working functionality. ~15 lines across two files.
2. **MEDIUM-3** — banner the execution brief. One paragraph. This is the doc that says a model
   swap is pending, and it is almost certainly what generated the "which ONNX?" question in the
   first place. Highest ratio of confusion-removed to effort in the whole audit.
3. **HIGH-3** — delete the "dead code / blocked on retrain / no production implementation"
   KDoc from `CtcSwipeDecoder.kt`, `CtcEmissions.kt`, `CtcLayout.kt`. This is the confusion
   preserved in amber, sitting in `src/main`.
4. **HIGH-2** — fix the three remaining `sw2345` code citations, the two doc citations, and
   parity-audit finding 13. Leaving a file that asserts both 89.87 and 91.82 for the same
   measurement is how the next reader repeats this exactly.
5. **MEDIUM-5 + MEDIUM-8** — the settings screen and the README both tell users CTC is
   English-on-QWERTY. They are the only user-visible scope statements and both are wrong (in
   the safe direction, but wrong).
6. **MEDIUM-2** — decide whether the settle probes stay, *before* anyone quotes a latency
   number measured with them on. A verbose-build measurement is currently inflated by ~0.7 s.
7. **HIGH-4** — at minimum, state in the spec that the emission check is device-only; better,
   put the two instrumented CTC tests in the release checklist. Not urgent for this ship (the
   current triple is verified consistent), but it is the gap that would let the *next* model
   swap go wrong quietly.

Everything else — MEDIUM-1, MEDIUM-4, MEDIUM-6, MEDIUM-7, MEDIUM-9 and the LOW set — is
cleanup and can ride along with whatever touches those files next.

---

## 4. Method note

The app tree was read-only for the whole audit (`git pull --ff-only`, no writes, no commits).
Every hash in this document was produced by running `sha256sum` / `hashlib` against the file on
disk, not copied from a doc. Every accuracy figure was checked against `MODELS_TABLE.md` §2–§3
rows rather than against the app's own citation of them — which is how the `sw2345`
misattribution surfaced, since the app's prose was internally plausible and only the row
lookup disambiguated it.

---

## 5. RE-VERIFICATION at app `9a6ffdd2` (2026-08-18)

**Re-audited**: `/home/will/git/swype/CleverKeys` @ **`9a6ffdd2`** (`feat(ctc): serve it/pt/sv
on CTC`), pulled 2026-08-18. Tree read-only for the re-check; the only app write in this
session was the new `docs/specs/ctc-architecture-and-multiscript-guide.md`, committed
separately with the user's explicit authorisation.

Between `a474ddf9` and `9a6ffdd2` the app landed the **neural swipe engine removal**
(`54b3bd59` plan, then `a7d03bc8`, `6f9b56fa`, `64f401d2`, `018d94f7`, `eb430fa0`, `6e982d56`,
`83220634`, `d32b6c25`, `f4c981a4`) and then `9a6ffdd2`, which adds `it`/`pt`/`sv`. Every line
number in §1–§3 above is stale; everything below was re-located by grep.

**Score: of 23 findings, 12 persist unchanged, 5 are partially addressed, 1 REGRESSED, 0 are
fully closed.** Three new findings the original audit could not have seen.

### 5.0 Ship state at the new head — unchanged where it matters

| item | value |
|---|---|
| `src/main/assets/models/ctc_swipe_encoder.onnx` | sha256 `84718e6e…e88e5`, 3,052,318 B — **identical** to `a474ddf9` |
| `ctc_golden.json` (both copies, byte-identical) | sha256 `2a449c4f…7559c`, 140,462 B, preset `tunedV2` |
| `CtcLanguageSupport.SUPPORTED` | `en`→EN_JSON; `fr, de, es, it, pt, sv`→CKDT_BIN. `PROVISIONAL = {it, pt, sv}`, `NEEDS_VALIDATION = ∅` |
| **`Defaults.SWIPE_ENGINE_MODE`** | **`"ctc"`** (`Config.kt:300`) — was `"neural"`. **CTC is now every user's default.** |
| `Mode.fromPref` | `"geometric" → GEOMETRIC; else → CTC` — legacy `neural`/`hybrid` prefs migrate INTO ctc |
| `CtcScoringParams.tunedRuCkdt` | unchanged: γ 1.05, λ `LAMBDA_CKDT_SCALE` = 2.0, β 0.2, γp 0.3734, βp 0.9882; still unreachable (`presetFor` branches on `LexiconSource`, `ru` ∉ `SUPPORTED`) |
| Russian dictionary in the APK | **none.** `assets/dictionaries/` is Latin only (`en_enhanced.json` + `{en,de,es,fr,it,pt,sv}_enhanced.bin`). `ru` exists only as the importable langpack `scripts/dictionaries/langpack-ru.zip` (533,916 B; `dictionary.bin` 2,088,865 B, magic `CKDT` v2, lang `ru`) |

### 5.1 Persistence table

| finding | status | where it is now |
|---|---|---|
| **HIGH-1** latched-load kills swipe typing | **PERSISTS — escalated** | latch `CtcEngineAdapter.kt:145-177` byte-identical, log line still claims a disable that does not happen; decode-null path `:667-671` unchanged; dispatch guard `InputCoordinator.kt:690` still only `supportsLayout`; prewarm `:753-755` unguarded; no `isModelPermanentlyUnavailable` anywhere. See §5.2. |
| **HIGH-2** `sw2345` misattribution | **FIXED in `src/main`**, persists in 4 sites | Router now carries 91.82 / 91.10 (`:24, :68-69, :109`) and the correct euro set. New guard `CoreImeHygieneDriftTest.sourceQuotesTheShippedModelsAccuracyNotItsSupersededPredecessors` (`:601-636`) bans the six figures — **but scans `File("src/main/kotlin")` only** (`:23`). Survivors: `src/test/.../SwipeEngineRouterTest.kt:20`; `docs/eval/2026-08-15-ctc-per-language-lambda.md:101, 112`; `docs/audit/2026-08-17-neural-vs-ctc-parity.md:619-623` (finding 13 still unstruck). |
| **HIGH-3** dead-code / blocked-on-retrain KDoc | **PERSISTS verbatim**, all four blocks | `CtcSwipeDecoder.kt:6-15, 35-39, 41`; `CtcEmissions.kt:12-16`; `CtcLayout.kt:12`. §2's proposed diffs still apply unchanged. |
| **HIGH-4** fixture rule's behavioural half never runs | **PERSISTS in full** | 7 workflows, none runs `connectedAndroidTest` or `ew-cli`; `ci.yml:39` / `release.yml:38` are `runPureTests`; `ui-testing.yml` is `adb install` + `dumpsys` greps. `CtcParityTest.kt:38` still hardcodes `MODEL_ASSET_PATH`; preset pin `:148-158` still omits `beamWidth`. No device-only caveat in the spec. |
| **MEDIUM-1** ORT session leak | PERSISTS | `CtcEngineAdapter.kt:761-763`; `OnnxCtcEmissionModel.close()` (`:100-106`) zero callers; no `shutdownGracefully`. |
| **MEDIUM-2** settle probes on the decode thread | PERSISTS | `CtcEngineAdapter.kt:425, 429, 467` all still `settle = true`. Every `LOCAL_BUILD=true` latency measurement is inflated ~720 ms. |
| **MEDIUM-3** unbannered execution brief | PERSISTS | `docs/audit/remediation-plans/ctc-integration-execution-brief.md` — still no banner, `:86` still `Q1 model choice: SUPERSEDED-PENDING`, `:21` still cites `resbn80g_s1234.onnx`, `:74` still "Default engine stays `neural`". Still the highest-value doc fix. |
| **MEDIUM-4** 11.0 MB superseded bench ONNX | PERSISTS | four files in `src/androidTest/assets/ctc_bench/`; `CtcOnnxLatencyBenchmarkTest.kt:45-48` still "the ship candidate"; `:351` still `fullDecodePath_ch128_beam100_tunedV2` with E1 constants; `CtcBenchFixture.kt:9` still cites `a18ea58c…`. |
| **MEDIUM-5** settings scope text | **HALF FIXED** | `CtcSettingsActivity.kt:89-91` now says "Latin layouts" — QWERTY gone. Still hardcoded English, no language list, `:101`'s "100 is the validated default" still decoupled from `Defaults.CTC_BEAM_WIDTH`. The `Config.kt` / `SettingsDefaults.kt` QWERTY phrasings are gone. |
| **MEDIUM-6** import validation | PERSISTS, **inverted** | `SettingsValidation.kt` has no `ctc_beam_width` (`else -> true`, `:278`) and no `swipe_engine_mode` (`:355`), while `:97` still validates `neural_beam_width` — a pref of the deleted engine. |
| **MEDIUM-7** no unsupported-language feedback | **REGRESSED — from mis-gated to ABSENT** | `ui/settings/sections/NeuralPredictionSection.kt` was deleted (`eb430fa0`). Its replacement `SwipeTypingSection.kt` (103 lines) has the engine dropdown (`:47-62`) and **no warning card of any kind**. Its own KDoc `:41-42` says "en/fr/de/es" — already stale. |
| **MEDIUM-8** doc/UI scope | **QWERTY FIXED, language set now stale everywhere** | `9a6ffdd2` made 7 languages true while `README.md:168, 243`, `docs/ARCHITECTURE_MASTER.md:226`, `docs/wiki/layouts/multi-language.md:46`, `docs/wiki/specs/typing/swipe-typing-spec.md:41, 61`, `docs/wiki/typing/swipe-typing.md:80`, `SwipeTypingSection.kt:41-42`, `memory/todo.md:260-262` and **all 22 `swipe_engine_mode_desc` strings** still say four. Two original sub-rows also survive: `ARCHITECTURE_MASTER.md:245` (lexicon = `en_enhanced.json` only, omits the CKDT `.bin` path) and `:237` (λ as a single 4.0; it is per-scale). |
| **MEDIUM-9** stale `memory/todo.md` | **1 of 3 FIXED** | `:178-179` "the CTC engines are demo-only" PERSISTS. `:263-266` P2 doc item PERSISTS (both halves done). **The Russian section `:223-259` is FIXED** — it now leads with "all Cyrillic numbers are val-tier permanently" and "no sealed Cyrillic split has ever existed", which is exactly the framing §2's MEDIUM-9 asked for. New staleness `:260-262` (it/pt/sv described as dead). `docs/eval/2026-08-15-ctc-per-language-lambda.md:78` still lists "(and ru per the prior sweep)" with no footing caveat. |
| **LOW-1** `MappedLayout.padded` unread | PERSISTS | declared `:184`, built `:299-301`, zero readers. |
| **LOW-2** phantom `weight` term in the KDoc formula | PERSISTS | `CtcScoringParams.kt:12` vs `CtcBeamDecoder.kt:163`. |
| **LOW-3** `normalizeRawX/Y` trap | PERSISTS | `CtcFeaturizer.kt:163, 171-172`, no `@Deprecated`. |
| **LOW-4** bare `globalConfig()` on the decode path | PERSISTS | guarded at `:154-158`, bare at `:676`. |
| **LOW-5** null-keyboard asymmetry | **PARTLY MOOT** | the `Mode.NEURAL` contrast is gone with neural. The overload divergence remains (`SwipeEngineRouter.kt:100` returns `Engine.CTC` for a null layout; the string overload returns `GEOMETRIC`), pinned only for the string form. Downstream both paths now `?: return` symmetrically, so nothing is lost relative to any alternative. |
| **LOW-6** dev absolute path in the fixture | PERSISTS | both copies still carry `/home/will/ctc-train/ckpt/v2kd-fresh-w1/kd_fp16w.onnx`. |
| **LOW-7** no proguard keep for `swipe.ctc.**` | PERSISTS | `minifyEnabled false` both types, so still inert. |
| **LOW-8** `"futo"` settings search keyword | PERSISTS | `SettingsActivity.kt:568`. |
| **LOW-9** no ЙЦУКЕН `supportsLayout` negative test | PERSISTS | nothing in `src/test` or `src/androidTest` feeds a Cyrillic `KeyboardData` to `CtcEngineAdapter.supportsLayout`. See NEW-2 — there is now a *real* mis-tagged layout for it to catch. |
| **LOW-10** gate ordering enforced by source scan | PERSISTS | `CoreImeHygieneDriftTest.kt:208, 251, 286-294, 342`. |

### 5.2 HIGH-1 — why the refactor made it worse, and the re-anchored diff

The mechanism is unchanged. What changed is who it hits and where they land:

1. **`Defaults.SWIPE_ENGINE_MODE` went `"neural"` → `"ctc"`** (`Config.kt:300`). At audit time a
   latched session hurt only opt-in users; now it hurts everyone on defaults.
2. **`Mode.fromPref` maps every non-`"geometric"` value — including the removed `"neural"` and
   `"hybrid"` — to `Mode.CTC`.** Users who had explicitly chosen neural are migrated into the
   affected mode.
3. **Neural is deleted**, so there is no second ML engine to fall back to. Three of the four
   gates now hand off to geometric (language `InputCoordinator.kt:666-678`, layout `:690-696`,
   router `SwipeEngineRouter.kt:115`); the model/lexicon gate is the **only remaining way a
   swipe reaches no engine at all**, and MEDIUM-7's regression means nothing tells the user.

The adapter half of §2's proposed diff applies **verbatim** (the `modelLoadAttempts`
declaration, the `catch` block and the log expression are unchanged text). The two
`InputCoordinator` hunks need re-anchoring only — `:686-696` for the dispatch guard (whose
comment lost its neural sibling) and `:753-755` for the prewarm, where the `val ctc` binding the
audit introduced **already exists**, shrinking that hunk to one line. Both re-anchored diffs are
written out in full in
`ctc-architecture-and-multiscript-guide.md` §6.1 (mirrored into the app at
`docs/specs/ctc-architecture-and-multiscript-guide.md`).

**A constraint §2 could not know about**: `CoreImeHygieneDriftTest` (`:208, 251-263, 286-294`)
source-scans these blocks for the literal substrings `CtcEngineAdapter.supportsLanguage(`,
`supportsLayout(` and `performGeometricSwipeTyping` and asserts their relative index order. The
re-anchored diffs preserve all three and their ordering; any further reshaping must too.

### 5.3 New findings

**NEW-1 — `docs/specs/ctc-swipe-engine.md`, the CTC engine's own spec, was not touched by a
single removal commit.** `git log a7d03bc8~1..HEAD -- docs/specs/ctc-swipe-engine.md` is empty,
while the *wiki* copy got a proper banner. The spec for the now-**default** engine still reads:
title "(`ctc` mode — WIRED, **opt-in**)"; `:2-3` "default stays `neural`"; `:10`
"(QWERTY→CTC, other layouts→geometric hedge)"; `:41-44` a four-row routing table with
`neural (default)` and `hybrid` rows and a `ctc`-row cell reading "NEURAL (M1 fallthrough)";
`:52` "`Engine.NEURAL` takes"; `:34-36` the router gate as `Config.isSwipeTypingSupportedForLayout`
(now referenced only in dead comments); `:168-169` missing-letter layouts as "unexpected behind
the router's QWERTY gate"; `:243` an `it, pt, sv | none | none` row. This is a strictly worse
MEDIUM-8 than the audit described, in the document `CLAUDE.md`'s spec-driven workflow points a
maintainer at first. **Belongs beside MEDIUM-3 in the anti-confusion set.**

**NEW-2 — `src/main/layouts/grek_qwerty.xml` declares `script="latin"`.** Its sibling
`srcs/layouts/grek_qwerty.xml` was corrected to `script="greek"` in `6af11da7` ("closes
neural-swipe allowlist leak") — but `srcs/layouts/` is **not referenced by any build task**;
`copyLayoutDefinitions` copies `src/main/layouts/*.xml`. The fix landed in the tree the build
does not read, so the shipped Greek layout passes the router's script gate and is stopped only
by the alphabet gate. No user-visible harm (geometric either way), but gate 1 of 3 is being
relied on and one of the 86 shipped layouts defeats it. Measured census of
`src/main/layouts/` at this head, which also disposes of the "37 undeclared layouts" premise:

| bucket | count |
|---|---|
| `script="latin"` and a–z-complete → CTC | 46 |
| `script="latin"` but a–z-incomplete (router passes, alphabet gate stops) | 3 — `grek_qwerty` (all 26 missing), `latn_qwerty_az` (`w`), `latn_qwerty_tly` (`w`) |
| non-Latin declared, 15 distinct scripts | 35 |
| **no `script` attribute at all** | **2** — `numeric.xml`, `pin.xml`, neither a letter layout |

**NEW-3 — the new sw2345 drift guard has a blind spot a live offender already occupies.**
`CoreImeHygieneDriftTest.kt:23` walks `File("src/main/kotlin")`; `SwipeEngineRouterTest.kt:20`
quotes 89.87 / 88.98 and sits outside it. The guard was written *because* a KDoc rewrite
reintroduced those numbers, and it cannot catch the copy that is there today.

**NEW-4 (minor)** — `InputCoordinator.kt:525` still documents `beginSwipeCapture`'s `engine`
parameter as `ENGINE_NEURAL`/`ENGINE_GEOMETRIC`; both call sites now pass `ENGINE_CTC` (`:700`)
or `ENGINE_GEOMETRIC`. `SwipeMLData.ENGINE_NEURAL` is correctly retained for reading historical
exports.

### 5.4 Revised ship-order

1. **HIGH-1** — now a default-path bug on a keyboard with no second ML engine. Unchanged as #1,
   more urgent than when written.
2. **MEDIUM-3 + NEW-1** — banner the execution brief *and* rewrite `ctc-swipe-engine.md`. The
   two documents most likely to mislead the next reader, and the spec is worse than the brief
   because the workflow points at it.
3. **HIGH-3** — delete the four dead-code KDoc blocks.
4. **HIGH-2 + NEW-3** — the four surviving `sw2345` citations, and widen the drift guard past
   `src/main/kotlin`.
5. **NEW-2** — the `grek_qwerty` attribute, plus the layout-census test and LOW-9's negative
   test.
6. **MEDIUM-8 + MEDIUM-5 + MEDIUM-7** — the language set is stale in 8 doc sites, 22 strings and
   one settings KDoc, and there is now *no* surface that tells a user their language is unserved.
7. **MEDIUM-2**, then **HIGH-4** — as before.
