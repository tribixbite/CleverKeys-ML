# APP_INTEGRATION_PLAN — wiring the CTC swipe engine into CleverKeys (G3 + G5)

**Date:** 2026-08-08 · **Updated:** 2026-08-13 (Phase L — see **§9**: the new
single-model finalist and the coupled-pair recipe) · 2026-08-12 (Phase K — see **§8**: the
ensemble configuration, contract-v2, the rescorer) · 2026-08-11 (Phase J — see **§7**, which carries
the Cyrillic λ finding (§7.1), the post-Phase-J model menu and fixture state
(§7.2), the user-dictionary pointer (§7.3) and the multi-script verdict (§7.4);
§7 supersedes D1, §1(d), §1(e) and O1 where they disagree).
**Scope:** exact, apply-nearly-verbatim diffs for the app repo
(`/home/will/git/swype/CleverKeys`, READ-ONLY for this session — nothing there was
modified). All facts below were verified against app-repo HEAD `79ddfb0f` and this
repo's `ctc/` artifacts.
**Companion change already landed in THIS repo:** `make_golden.py` now also emits six
`"featurize"`-kind cases and a top-level `layout` block, and
`artifacts/ctc_model_golden.json` was regenerated (same model `ch128_s1234`, same E1
preset; the 4 beam cases are **byte-identical** to the previous fixture). Without the
featurize cases the app's `CtcParityTest.featurizer_matchesPythonPort_bitIdentical`
would fail its own `checked > 0` assertion against the model-only fixture.

Fixture now: 10 cases (6 featurize + 4 beam), 140,204 bytes,
sha256 `a18ea58cd662b0e18b6daadaf417361f93fd0b146ce6478d4d6a62e7e185fa8a`,
`source_onnx_sha256 = 6c1144…c51` (= `ch128_s1234.onnx`), preset
`[1.05, 1.1, 0.2, 0.3734, 0.9882]`.

---

## 0. Decisions taken in this plan (each justified inline, revisitable in §6)

| # | Decision | Choice |
|---|---|---|
| D1 | Model shipped | `ch128_s1234.onnx` (test-validated, 0.455–0.475 ms, 2,799,865 B) as `assets/models/ctc_swipe_encoder.onnx` |
| D2 | Scoring preset | new `CtcScoringParams.tunedV2` = E1 (γ 1.05, λ 1.1, β 0.2, α 0.0, γp 0.3734, βp 0.9882) — **required**, published preset costs ~2.3 pt t1 |
| D3 | Beam width | default **100** (every campaign validation decoded at width 100, not the FUTO-ship 300), user-tunable via new `ctc_beam_width` pref (10–300) |
| D4 | Lexicon | bundled `dictionaries/en_enhanced.json` (98,140 words, values **already on the AOSP-like 134–255 log scale**) via `CtcLexiconTrie.loadStrippingNonAlphabet` + user custom/disabled words — no new dictionary asset |
| D5 | Coordinate frame | letter-key bounding box of the live layout, uniform x/y normalization, **no 4/3 aspect correction** (training rows were plain `x/keyb_width`, `y/keyb_height`; the 4/3 helper in `CtcFeaturizer.normalizeRawY` is a FUTO-runtime contract our model never saw) |
| D6 | Routing | new `Mode.CTC`: QWERTY-Latin → `Engine.CTC`, every other layout → `Engine.GEOMETRIC` (same non-QWERTY coverage as HYBRID, so selecting CTC never silently kills swipe on other layouts) |
| D7 | Default engine | stays `neural`; `ctc` is opt-in via the existing Prediction Engine dropdown (4th option) |
| D8 | Runtime | existing `onnxruntime-android:1.20.0` through the existing `ModelLoader` (XNNPACK-first chain, `onnx_xnnpack_threads` pref respected). No new runtime, no new proguard rules needed |

**D1 is stale as of 2026-08-11 and is left standing only as the record of what was
decided on 2026-08-08.** Phase G added `resbn80g` (test-validated, 1.14 MB) and
Phase J added `sw2345` (best measured accuracy, **not** test-validated). The
current menu, with evidence tiers, is **§7.2**. D2's E1 preset still holds for the
Phase-J finalist on the benchmark footing (§7.2), but the app-trie λ question
(O3, §7.2) is open for it, and `resbn80g` would ship at a *different* preset
(λ 4.0). Every other decision (D3–D8) is unaffected: the finalist is
architecturally identical to what D1 assumed, same frozen I/O contract.

---

## 1. File-by-file proposed diffs

### 1(a) ONNX `CtcEmissionModel` implementation — NEW FILE

The pure `swipe/ctc/` package stays Android/ORT-free (spec NFR-1); the ORT-backed
implementation lives beside the adapters in `swipe/` (same placement rationale as
`GeometricEngineAdapter`).

ONNX contract (verified against `export_onnx.py` and the artifact):
inputs `features [1,2,64] f32`, `layout_keys [1,64,2] f32`, `layout_mask [1,64] bool`;
outputs `log_emissions [1,32,65] f32` (+ `coefficients`, `lambda`, unused). Blank sits
at full-head column 64; `CtcEmissions.sliceFromHead` relocates it to column
`numLetters`.

```diff
--- /dev/null
+++ b/src/main/kotlin/tribixbite/cleverkeys/swipe/OnnxCtcEmissionModel.kt
@@ -0,0 +1,110 @@
+package tribixbite.cleverkeys.swipe
+
+import ai.onnxruntime.OnnxJavaType
+import ai.onnxruntime.OnnxTensor
+import ai.onnxruntime.OrtEnvironment
+import ai.onnxruntime.OrtSession
+import java.nio.ByteBuffer
+import java.nio.FloatBuffer
+import tribixbite.cleverkeys.swipe.ctc.CtcEmissionModel
+import tribixbite.cleverkeys.swipe.ctc.CtcEmissions
+import tribixbite.cleverkeys.swipe.ctc.CtcFeaturizer
+
+/**
+ * Production [CtcEmissionModel] over onnxruntime-android — the G3 closure of the
+ * retrain-fork seam (`docs/specs/ctc-swipe-engine.md` FR-5).
+ *
+ * Runs the CleverKeys-trained CTC swipe encoder (`models/ctc_swipe_encoder.onnx`,
+ * CleverKeys-ML `ctc/` campaign 2, arm `phaseE-E3b-hws3x`). Graph contract (opset 17,
+ * fully static shapes, verified at export):
+ *  - `features`     `[1, 2, 64]`  float32 — [CtcFeaturizer.featurize] output.
+ *  - `layout_keys`  `[1, 64, 2]`  float32 — [CtcFeaturizer.PaddedLayout.keys]
+ *    (interleaved cx,cy is exactly the row-major `[64, 2]` layout).
+ *  - `layout_mask`  `[1, 64]`     bool    — [CtcFeaturizer.PaddedLayout.mask].
+ *  - `log_emissions` `[1, 32, 65]` float32 — full head; blank at column 64
+ *    (`MAX_KEYS`), sliced to the active alphabet via [CtcEmissions.sliceFromHead].
+ *  (The graph's `coefficients`/`lambda` outputs are diagnostics and are not fetched.)
+ *
+ * Threading: [OrtSession.run] is thread-safe, but callers ([CtcEngineAdapter])
+ * serialize all calls on one background thread anyway. The session is owned by this
+ * object; [close] releases it.
+ */
+class OnnxCtcEmissionModel(
+    private val env: OrtEnvironment,
+    private val session: OrtSession,
+) : CtcEmissionModel {
+
+    companion object {
+        const val INPUT_FEATURES = "features"
+        const val INPUT_LAYOUT_KEYS = "layout_keys"
+        const val INPUT_LAYOUT_MASK = "layout_mask"
+        const val OUTPUT_LOG_EMISSIONS = "log_emissions"
+
+        /** Full-head width = MAX_KEYS + 1 (blank column at index MAX_KEYS). */
+        const val HEAD_WIDTH = CtcFeaturizer.MAX_KEYS + 1
+    }
+
+    override fun emit(features: FloatArray, layout: CtcFeaturizer.PaddedLayout): CtcEmissions {
+        require(features.size == 2 * CtcFeaturizer.RESAMPLE_LENGTH) {
+            "features length ${features.size} != ${2 * CtcFeaturizer.RESAMPLE_LENGTH}"
+        }
+        require(layout.keys.size == CtcFeaturizer.MAX_KEYS * 2) {
+            "layout keys length ${layout.keys.size} != ${CtcFeaturizer.MAX_KEYS * 2}"
+        }
+        val numLetters = layout.mask.count { it }
+        require(numLetters in 1..CtcFeaturizer.MAX_KEYS) { "empty layout mask" }
+
+        // ORT bool tensors are 1 byte/element via ByteBuffer + OnnxJavaType.BOOL.
+        val maskBytes = ByteArray(CtcFeaturizer.MAX_KEYS)
+        for (i in maskBytes.indices) maskBytes[i] = if (layout.mask[i]) 1 else 0
+
+        OnnxTensor.createTensor(
+            env, FloatBuffer.wrap(features),
+            longArrayOf(1, 2, CtcFeaturizer.RESAMPLE_LENGTH.toLong())
+        ).use { featTensor ->
+            OnnxTensor.createTensor(
+                env, FloatBuffer.wrap(layout.keys),
+                longArrayOf(1, CtcFeaturizer.MAX_KEYS.toLong(), 2)
+            ).use { keysTensor ->
+                OnnxTensor.createTensor(
+                    env, ByteBuffer.wrap(maskBytes),
+                    longArrayOf(1, CtcFeaturizer.MAX_KEYS.toLong()), OnnxJavaType.BOOL
+                ).use { maskTensor ->
+                    session.run(
+                        mapOf(
+                            INPUT_FEATURES to featTensor,
+                            INPUT_LAYOUT_KEYS to keysTensor,
+                            INPUT_LAYOUT_MASK to maskTensor,
+                        ),
+                        setOf(OUTPUT_LOG_EMISSIONS)
+                    ).use { result ->
+                        val out = result.get(0) as OnnxTensor
+                        val shape = out.info.shape // [1, frames, HEAD_WIDTH]
+                        val frames = shape[1].toInt()
+                        val headWidth = shape[2].toInt()
+                        require(headWidth == HEAD_WIDTH) {
+                            "unexpected head width $headWidth (expected $HEAD_WIDTH)"
+                        }
+                        val full = FloatArray(frames * headWidth)
+                        out.floatBuffer.get(full)
+                        return CtcEmissions.sliceFromHead(
+                            full, frames, CtcFeaturizer.MAX_KEYS, numLetters
+                        )
+                    }
+                }
+            }
+        }
+    }
+
+    /** Releases the native session (call only from the owning decode thread). */
+    fun close() {
+        try {
+            session.close()
+        } catch (e: Exception) {
+            // Session close failures are non-actionable at teardown.
+        }
+    }
+}
```

### 1(b) `CtcScoringParams` — the `tunedV2` preset

```diff
--- a/src/main/kotlin/tribixbite/cleverkeys/swipe/ctc/CtcScoringParams.kt
+++ b/src/main/kotlin/tribixbite/cleverkeys/swipe/ctc/CtcScoringParams.kt
@@ -70,6 +70,27 @@ data class CtcScoringParams(
                 gammaPrune = 0.1902, betaPrune = 1.2727,
                 beamWidth = beamWidth, topK = topK,
             )
 
+        /**
+         * Campaign-2 "E1" preset — val-9918-tuned for the CleverKeys-trained CTC
+         * encoder that ships as `models/ctc_swipe_encoder.onnx` (CleverKeys-ML
+         * `ctc/RESULTS.md`, "Shipping recommendation"). This preset is REQUIRED for
+         * that model: decoded at [encoderOnly] it clears only 3 of 5 FUTO-ceiling
+         * bars (~−2.3 pt top-1); at this preset it clears all 5 on every seed.
+         *
+         * The optimum sits far outside FUTO's `scoring.json` neighborhood
+         * (γ 1.05 vs 0.41, λ 1.1 vs 0.018, β 0.2 vs 0.99) because our emissions
+         * are sharper than FUTO's encoder-only head — see RESULTS.md "Retraction".
+         *
+         * @param beamWidth commit-phase width. Every campaign accuracy number was
+         *   decoded at width **100** (not FUTO's 300), so 100 is the default; the
+         *   `ctc_beam_width` pref feeds this.
+         * @param topK size of the returned slate.
+         */
+        fun tunedV2(beamWidth: Int = 100, topK: Int = 4): CtcScoringParams =
+            CtcScoringParams(
+                gamma = 1.05, lambda = 1.1, beta = 0.2, alpha = 0.0,
+                gammaPrune = 0.3734, betaPrune = 0.9882,
+                beamWidth = beamWidth, topK = topK,
+            )
+
         /** `scoring.json` "fallback" — used when no signature-specific set matches. */
         fun fallback(beamWidth: Int = 300, topK: Int = 4): CtcScoringParams =
             CtcScoringParams(
```

Where it is selected: exclusively in `CtcEngineAdapter` (§1c-iii). Nothing else
constructs scoring params, so the committed FUTO presets remain untouched reference
constants (and `CtcModuleTest.scoringParams_presets_matchScoringJson` keeps passing;
a new assertion block for `tunedV2` is added in §3).

### 1(c) Router wiring — `ctc` end-to-end

#### 1(c)-i `Config.kt` (3 hunks)

```diff
--- a/src/main/kotlin/tribixbite/cleverkeys/Config.kt
+++ b/src/main/kotlin/tribixbite/cleverkeys/Config.kt
@@ -313,11 +313,17 @@
     // WP9 R-1 step 7 (v1.1): swipe prediction engine mode — "neural" (QWERTY-only swipe,
     // the long-standing default), "hybrid" (neural on QWERTY + geometric elsewhere), or
-    // "geometric" (SHARK2 on all layouts). Settings → Swipe Typing → Prediction Engine.
+    // "geometric" (SHARK2 on all layouts), or "ctc" (G5: CTC trie-beam on QWERTY +
+    // geometric elsewhere). Settings → Swipe Typing → Prediction Engine.
     const val SWIPE_ENGINE_MODE = "neural"
     // Full Geometric Settings — the three user-tunable geo knobs (defaults MUST equal
     // GeometricEngineConfig's; the rest of the engine's 28 knobs stay code-only because
     // they are calibrated against the spec's measured accuracy floors).
     const val GEO_MAX_RESULTS = 10           // ranked candidates emitted (bar length)
     const val GEO_FREQUENCY_WEIGHT = 0.12f   // λ_f: common-words vs shape-fidelity bias
     const val GEO_ENDPOINT_INSET_KW = 0.30f  // sloppy start/end tolerance (key-widths)
+    // CTC engine (G5) — commit-phase trie-beam width. 100 is the width every
+    // CleverKeys-ML campaign-2 validation number was decoded at; wider buys little
+    // (the beam is Viterbi-max over a 26-ary trie) and costs linear CPU per swipe.
+    const val CTC_BEAM_WIDTH = 100
     const val AUTO_SPACE_AFTER_SUGGESTION = true  // Add trailing space after selecting suggestion
```

```diff
@@ -634,10 +640,12 @@
     // WP9 R-1 step 7 (v1.1): swipe engine mode — "neural" | "hybrid" | "geometric".
+    // G5 adds "ctc".
     @JvmField var swipe_engine_mode = Defaults.SWIPE_ENGINE_MODE
     // Full Geometric Settings knobs (read by GeometricEngineAdapter per decode).
     @JvmField var geo_max_results = Defaults.GEO_MAX_RESULTS
     @JvmField var geo_frequency_weight = Defaults.GEO_FREQUENCY_WEIGHT
     @JvmField var geo_endpoint_inset_kw = Defaults.GEO_ENDPOINT_INSET_KW
+    // CTC engine knob (read by CtcEngineAdapter per decode).
+    @JvmField var ctc_beam_width = Defaults.CTC_BEAM_WIDTH
     @JvmField var swipe_debug_show_raw_output = false
```

```diff
@@ -928,6 +936,8 @@
         swipe_engine_mode = safeGetString(_prefs, "swipe_engine_mode", Defaults.SWIPE_ENGINE_MODE)
         geo_max_results = safeGetInt(_prefs, "geo_max_results", Defaults.GEO_MAX_RESULTS)
         geo_frequency_weight = safeGetFloat(_prefs, "geo_frequency_weight", Defaults.GEO_FREQUENCY_WEIGHT)
         geo_endpoint_inset_kw = safeGetFloat(_prefs, "geo_endpoint_inset_kw", Defaults.GEO_ENDPOINT_INSET_KW)
+        ctc_beam_width = safeGetInt(_prefs, "ctc_beam_width", Defaults.CTC_BEAM_WIDTH)
+            .coerceIn(10, 300)  // clamp mirrors onnx_xnnpack_threads' defensive pattern
```

#### 1(c)-ii `SwipeEngineRouter.kt`

```diff
--- a/src/main/kotlin/tribixbite/cleverkeys/swipe/SwipeEngineRouter.kt
+++ b/src/main/kotlin/tribixbite/cleverkeys/swipe/SwipeEngineRouter.kt
@@ -30,14 +30,17 @@ object SwipeEngineRouter {
 
     enum class Engine {
         /** QWERTY-trained transformer path (existing behavior). */
         NEURAL,
 
         /** Pure-JVM geometric (SHARK2) engine. */
         GEOMETRIC,
 
+        /** CTC trie-beam engine (ONNX encoder + pure-JVM beam, `swipe/ctc/`). */
+        CTC,
+
         /** No engine — non-QWERTY layout in NEURAL-only mode. */
         NONE,
     }
 
     /** User-selected engine mode (the `swipe_engine_mode` pref). */
     enum class Mode {
         /** Neural on QWERTY, no swipe elsewhere (default — pre-geo behavior). */
         NEURAL,
 
         /** Neural on QWERTY, geometric on every other layout. */
         HYBRID,
 
-        /** Geometric on ALL layouts, including QWERTY. */
-        GEOMETRIC;
+        /** Geometric on ALL layouts, including QWERTY. */
+        GEOMETRIC,
+
+        /**
+         * G5: CTC on QWERTY-Latin (the layouts the shipped encoder was trained
+         * for), geometric on every other layout — the same non-QWERTY coverage
+         * as [HYBRID], so selecting CTC never removes swipe from other layouts.
+         */
+        CTC;
 
         companion object {
             /**
              * Parse the pref string. Unknown/legacy values fall back to [NEURAL] (the
              * default) — never crash the router on a corrupted pref.
              */
             @JvmStatic
             fun fromPref(value: String?): Mode = when (value?.lowercase()) {
                 "hybrid" -> HYBRID
                 "geometric" -> GEOMETRIC
+                "ctc" -> CTC
                 else -> NEURAL
             }
         }
     }
@@ -73,8 +76,11 @@ object SwipeEngineRouter {
     @JvmStatic
     fun route(layout: KeyboardData?, mode: Mode): Engine {
         if (mode == Mode.GEOMETRIC) return Engine.GEOMETRIC
-        if (Config.isSwipeTypingSupportedForLayout(layout)) return Engine.NEURAL
+        if (Config.isSwipeTypingSupportedForLayout(layout)) {
+            return if (mode == Mode.CTC) Engine.CTC else Engine.NEURAL
+        }
-        return if (mode == Mode.HYBRID) Engine.GEOMETRIC else Engine.NONE
+        return if (mode == Mode.HYBRID || mode == Mode.CTC) Engine.GEOMETRIC else Engine.NONE
     }
 
     /** String-based overload for pure-JVM tests (mirrors Config's testing overload). */
     @JvmStatic
     fun route(layoutName: String?, script: String?, mode: Mode): Engine {
         if (mode == Mode.GEOMETRIC) return Engine.GEOMETRIC
-        if (Config.isSwipeTypingSupportedForLayout(layoutName, script)) return Engine.NEURAL
+        if (Config.isSwipeTypingSupportedForLayout(layoutName, script)) {
+            return if (mode == Mode.CTC) Engine.CTC else Engine.NEURAL
+        }
-        return if (mode == Mode.HYBRID) Engine.GEOMETRIC else Engine.NONE
+        return if (mode == Mode.HYBRID || mode == Mode.CTC) Engine.GEOMETRIC else Engine.NONE
     }
 }
```

(Also extend the class KDoc's mode list with one line for `Mode.CTC` — cosmetic,
same wording as the enum doc.)

#### 1(c)-iii `CtcEngineAdapter` — NEW FILE (the impurity boundary, mirrors `GeometricEngineAdapter`)

Key contracts encoded here:

- **Coordinate frame (D5).** Training rows were normalized `x/keyb_width`,
  `y/keyb_height` over the keyboard area (`build_tiers.py`), and the canonical layout
  places the three letter rows at y = 1/6, 3/6, 5/6 of the frame. On CleverKeys the
  keyboard view also contains the number row / bottom row, so the adapter normalizes
  over the **bounding box of the 26 letter keys** and passes the same-frame key
  centers as `layout_keys`. Train-time augmentation (affine scale 0.85–1.15,
  translate ±0.05, path-vs-layout jitter) makes this in-distribution as long as path
  and centers share one frame. **`CtcFeaturizer.normalizeRawY`'s 4/3 aspect factor is
  deliberately NOT used** — that constant ports FUTO's runtime contract, which this
  model was never trained under.
- **Lexicon (D4).** `dictionaries/en_enhanced.json` values are already 134–255 —
  i.e. the AOSP-style log-frequency scale `tunedV2`'s λ=1.1 was fitted against
  (`ln f ∈ [4.9, 5.54]`). Words are inserted via `loadStrippingNonAlphabet` (STRIP
  policy — the same normalizer the 146,964-word tuning trie used, making `don't` →
  `dont` reachable). Custom words are merged (freq clamped to 1..255 → a top-of-scale
  boost at the default 1000), disabled words dropped, custom overrides disabled —
  matching `GeometricEngineAdapter.mergeUserWords` semantics. The content-hash
  `version` invalidates the memo on any user-dictionary mutation.
- **v1 is en-only.** If the active dictionary language isn't `en`, the adapter
  returns an empty slate (pipeline clears the bar) — same degrade shape as a missing
  geometric dictionary. See open decision O5.

```diff
--- /dev/null
+++ b/src/main/kotlin/tribixbite/cleverkeys/swipe/CtcEngineAdapter.kt
@@ -0,0 +1,330 @@
+package tribixbite.cleverkeys.swipe
+
+import ai.onnxruntime.OrtEnvironment
+import android.content.Context
+import android.graphics.PointF
+import android.os.Handler
+import android.os.Looper
+import android.util.Log
+import org.json.JSONObject
+import tribixbite.cleverkeys.BuildConfig
+import tribixbite.cleverkeys.Config
+import tribixbite.cleverkeys.DirectBootAwarePreferences
+import tribixbite.cleverkeys.KeyValue
+import tribixbite.cleverkeys.KeyboardData
+import tribixbite.cleverkeys.LanguagePreferenceKeys
+import tribixbite.cleverkeys.PredictionResult
+import tribixbite.cleverkeys.PredictionTaskRunner
+import tribixbite.cleverkeys.a11y.KeyboardGeometry
+import tribixbite.cleverkeys.onnx.ModelLoader
+import tribixbite.cleverkeys.swipe.ctc.CtcCandidate
+import tribixbite.cleverkeys.swipe.ctc.CtcFeaturizer
+import tribixbite.cleverkeys.swipe.ctc.CtcLayout
+import tribixbite.cleverkeys.swipe.ctc.CtcLexiconTrie
+import tribixbite.cleverkeys.swipe.ctc.CtcScoringParams
+import tribixbite.cleverkeys.swipe.ctc.CtcSwipeDecoder
+import java.lang.ref.WeakReference
+import java.security.MessageDigest
+import java.util.Locale
+import kotlin.math.exp
+import kotlin.math.roundToInt
+
+/**
+ * G5 — the impurity boundary between the Android IME and the pure-JVM CTC swipe
+ * engine (`swipe.ctc`, spec `docs/specs/ctc-swipe-engine.md`). Mirrors
+ * [GeometricEngineAdapter]'s duties for the `ctc` value of `swipe_engine_mode`:
+ *
+ *  1. [KeyboardData] → [CtcLayout] via [KeyboardGeometry.computeKeyRects]: the 26
+ *     a–z letter keys' centers, normalized over the LETTER-KEY BOUNDING BOX (the
+ *     model's [0,1] frame — the shipped encoder was trained on paths normalized
+ *     over the letter area with centers passed as `layout_keys`, NOT on FUTO's
+ *     4/3-aspect device frame; do not use [CtcFeaturizer.normalizeRawY] here).
+ *     Memoized per immutable KeyboardData instance + frame + params.
+ *  2. `PointF` trace → normalized double arrays under the SAME letter-box affine.
+ *  3. Dictionary → [CtcLexiconTrie]: bundled `dictionaries/en_enhanced.json`
+ *     ({word: freq}, freq already on the AOSP-like 134..255 log scale the tuned
+ *     λ expects — spec NFR-4), a–z-STRIPPED (`don't`→`dont`), with user custom
+ *     words merged (freq clamped 1..255; custom overrides disabled) and disabled
+ *     words removed. Content-hash `version` recomputed per ensure, so any user
+ *     dictionary mutation rebuilds the trie without ContentObserver plumbing.
+ *  4. ONNX session via the existing [ModelLoader] (XNNPACK-first,
+ *     `onnx_xnnpack_threads` pref), built lazily on the decode thread;
+ *     [warmUpAsync] front-loads session + trie + layout on layout/language switch.
+ *
+ * Threading: everything heavy runs on the single [PredictionTaskRunner] thread;
+ * results post to main. A new decode cancels the in-flight one (last-swipe-wins).
+ * Scores are engine-relative (softmax over final scores × 1000) — never compared
+ * across engines (router KDoc contract).
+ */
+class CtcEngineAdapter(private val context: Context) {
+
+    companion object {
+        private const val TAG = "CtcEngineAdapter"
+
+        /** Shipped CTC emission encoder (CleverKeys-ML ctc/, arm phaseE-E3b-hws3x). */
+        const val MODEL_ASSET = "models/ctc_swipe_encoder.onnx"
+
+        private const val DICT_ASSET = "dictionaries/en_enhanced.json"
+
+        /** v1 model + lexicon are English; other languages degrade to empty. */
+        private const val LANGUAGE = "en"
+
+        /** Emission-column alphabet — a..z, the shipped model's training order. */
+        private val ALPHABET = CharArray(26) { ('a' + it) }
+
+        /**
+         * Slate size handed to the suggestion pipeline. The bar renders ~5 and the
+         * pipeline augments (possessives, contractions); beyond 8 the tail is noise.
+         * Candidates are free at decode time (topK only truncates the final sort).
+         */
+        private const val TOP_K = 8
+    }
+
+    private val tasks = PredictionTaskRunner()
+    private val mainHandler = Handler(Looper.getMainLooper())
+    private val ortEnvironment = OrtEnvironment.getEnvironment()
+
+    // ── ONNX emission model (decode thread only) ────────────────────────────────
+    private var emissionModel: OnnxCtcEmissionModel? = null
+    private var modelLoadFailed = false
+
+    private fun modelOrNull(): OnnxCtcEmissionModel? {
+        emissionModel?.let { return it }
+        if (modelLoadFailed) return null // don't retry a hard failure per swipe
+        return try {
+            val threads = try {
+                Config.globalConfig().onnx_xnnpack_threads
+            } catch (e: Exception) {
+                Defaults_ONNX_THREADS_FALLBACK
+            }.coerceIn(1, 8)
+            val loaded = ModelLoader(context, ortEnvironment)
+                .loadModel(MODEL_ASSET, "CtcEncoder", true, threads)
+            if (BuildConfig.ENABLE_VERBOSE_LOGGING) {
+                Log.d(TAG, "CTC encoder loaded (${loaded.executionProvider}, " +
+                    "${loaded.modelSizeBytes} B)")
+            }
+            OnnxCtcEmissionModel(ortEnvironment, loaded.session).also { emissionModel = it }
+        } catch (e: Exception) {
+            Log.e(TAG, "CTC encoder load failed — ctc mode disabled this session", e)
+            modelLoadFailed = true
+            null
+        }
+    }
+
+    private val Defaults_ONNX_THREADS_FALLBACK get() =
+        tribixbite.cleverkeys.Defaults.ONNX_XNNPACK_THREADS
+
+    // ── Layout memo (per immutable KeyboardData + frame + params) ───────────────
+
+    /** [CtcLayout] plus the letter-box affine mapping view px → the model frame. */
+    private class MappedLayout(
+        val layout: CtcLayout,
+        val padded: CtcFeaturizer.PaddedLayout,
+        val originX: Float, val originY: Float,   // letter-box top-left, view px
+        val invW: Float, val invH: Float,         // 1 / letter-box extent
+    )
+
+    private class LayoutMemo(
+        val source: WeakReference<KeyboardData>,
+        val params: KeyboardGeometry.Params,
+        val frameWidthPx: Float,
+        val frameHeightPx: Float,
+        val mapped: MappedLayout?,
+    )
+
+    @Volatile
+    private var layoutMemo: LayoutMemo? = null
+
+    private fun layoutFor(
+        keyboard: KeyboardData,
+        params: KeyboardGeometry.Params,
+        frameWidthPx: Float,
+        frameHeightPx: Float,
+    ): MappedLayout? {
+        layoutMemo?.let { memo ->
+            if (memo.source.get() === keyboard && memo.params == params &&
+                memo.frameWidthPx == frameWidthPx && memo.frameHeightPx == frameHeightPx
+            ) {
+                return memo.mapped
+            }
+        }
+        val built = try {
+            buildMappedLayout(keyboard, params)
+        } catch (e: Exception) {
+            Log.e(TAG, "CtcLayout build failed", e)
+            null
+        }
+        layoutMemo = LayoutMemo(WeakReference(keyboard), params, frameWidthPx, frameHeightPx, built)
+        return built
+    }
+
+    /** The lowercase a–z letter of [kv] iff its label is exactly one such char. */
+    private fun letterOf(kv: KeyValue): Char? {
+        val raw = when (kv.getKind()) {
+            KeyValue.Kind.Char -> kv.getChar().toString()
+            KeyValue.Kind.String -> kv.getString()
+            else -> return null
+        }
+        if (raw.length != 1) return null
+        val c = raw.lowercase(Locale.ROOT)
+        if (c.length != 1) return null
+        return c[0].takeIf { it in 'a'..'z' }
+    }
+
+    /**
+     * Builds the a..z [CtcLayout] from the final modified layout, or null when any
+     * letter is missing (the router's QWERTY-Latin gate makes that unexpected).
+     * First occurrence of a letter wins (deterministic row-major order).
+     */
+    private fun buildMappedLayout(
+        keyboard: KeyboardData,
+        params: KeyboardGeometry.Params,
+    ): MappedLayout? {
+        val rects = KeyboardGeometry.computeKeyRects(keyboard, params)
+        if (rects.isEmpty()) return null
+
+        val cx = FloatArray(26); val cy = FloatArray(26); val seen = BooleanArray(26)
+        var left = Float.MAX_VALUE; var top = Float.MAX_VALUE
+        var right = -Float.MAX_VALUE; var bottom = -Float.MAX_VALUE
+        for (rect in rects) {
+            val letter = letterOf(rect.kv) ?: continue
+            val i = letter - 'a'
+            if (seen[i]) continue
+            seen[i] = true
+            cx[i] = (rect.bounds.left + rect.bounds.right) / 2f
+            cy[i] = (rect.bounds.top + rect.bounds.bottom) / 2f
+            if (rect.bounds.left < left) left = rect.bounds.left
+            if (rect.bounds.top < top) top = rect.bounds.top
+            if (rect.bounds.right > right) right = rect.bounds.right
+            if (rect.bounds.bottom > bottom) bottom = rect.bounds.bottom
+        }
+        if (seen.any { !it }) return null // not a full a-z layout
+        val w = right - left
+        val h = bottom - top
+        if (w <= 0f || h <= 0f) return null
+
+        val invW = 1f / w
+        val invH = 1f / h
+        val normX = FloatArray(26) { (cx[it] - left) * invW }
+        val normY = FloatArray(26) { (cy[it] - top) * invH }
+        val layout = CtcLayout(ALPHABET.copyOf(), normX, normY)
+        return MappedLayout(
+            layout, CtcFeaturizer.buildPaddedLayout(layout), left, top, invW, invH
+        )
+    }
+
+    // ── Lexicon trie memo (per user-dictionary content version) ─────────────────
+
+    private class TrieMemo(val trie: CtcLexiconTrie, val version: Long)
+
+    @Volatile
+    private var trieMemo: TrieMemo? = null
+
+    private fun trieFor(): CtcLexiconTrie? {
+        val prefs = DirectBootAwarePreferences.get_shared_preferences(context)
+        val customJson = prefs.getString(LanguagePreferenceKeys.customWordsKey(LANGUAGE), "{}") ?: "{}"
+        val disabled = prefs.getStringSet(LanguagePreferenceKeys.disabledWordsKey(LANGUAGE), emptySet())
+            ?: emptySet()
+        val version = contentVersion("asset:$DICT_ASSET", customJson, disabled)
+        trieMemo?.let { if (it.version == version) return it.trie }
+
+        val start = System.currentTimeMillis()
+        val base = try {
+            context.assets.open(DICT_ASSET).use { JSONObject(it.readBytes().decodeToString()) }
+        } catch (e: Exception) {
+            Log.e(TAG, "No CTC lexicon source ($DICT_ASSET)", e)
+            trieMemo = null
+            return null
+        }
+        val disabledLower = disabled.mapTo(HashSet()) { it.lowercase(Locale.ROOT) }
+
+        // Custom words FIRST (freq clamped onto the 1..255 AOSP-like scale; custom
+        // overrides disabled), then the base dictionary minus disabled words.
+        // Insertion order only affects beam tie-breaks; LinkedHashMap keeps it
+        // deterministic (base order = asset JSON order on Android's org.json).
+        val merged = LinkedHashMap<String, Double>(base.length() + 64)
+        if (customJson != "{}") {
+            try {
+                val obj = JSONObject(customJson)
+                val it = obj.keys()
+                while (it.hasNext()) {
+                    val word = it.next()
+                    if (word.isBlank()) continue
+                    merged[word] = obj.optInt(word, 1000).coerceIn(1, 255).toDouble()
+                }
+            } catch (e: Exception) {
+                Log.w(TAG, "Malformed custom-words JSON — ignoring", e)
+            }
+        }
+        val keys = base.keys()
+        while (keys.hasNext()) {
+            val word = keys.next()
+            if (word.lowercase(Locale.ROOT) in disabledLower) continue
+            if (word in merged) continue
+            merged[word] = base.optInt(word, 1).coerceAtLeast(1).toDouble()
+        }
+        // STRIP loader: same non-alphabet policy as the offline tuning trie
+        // (apostrophe forms reachable as their a-z surface).
+        val trie = CtcLexiconTrie.loadStrippingNonAlphabet(ALPHABET, merged)
+        if (BuildConfig.ENABLE_VERBOSE_LOGGING) {
+            Log.d(TAG, "CTC trie: ${trie.wordCount} words in " +
+                "${System.currentTimeMillis() - start}ms (v=$version)")
+        }
+        val built = TrieMemo(trie, version)
+        trieMemo = built
+        return trie
+    }
+
+    /** Stable 64-bit content version over (source id, custom JSON, disabled set). */
+    private fun contentVersion(sourceId: String, customJson: String, disabled: Set<String>): Long {
+        val md = MessageDigest.getInstance("SHA-256")
+        md.update(sourceId.toByteArray(Charsets.UTF_8)); md.update(0)
+        md.update(customJson.toByteArray(Charsets.UTF_8)); md.update(0)
+        for (w in disabled.sorted()) {
+            md.update(w.toByteArray(Charsets.UTF_8)); md.update(1)
+        }
+        val d = md.digest()
+        var v = 0L
+        for (i in 0 until 8) v = (v shl 8) or (d[i].toLong() and 0xFF)
+        return v
+    }
+
+    // ── Decoder memo (per layout + trie + beam width) ───────────────────────────
+
+    private var decoderMemo: CtcSwipeDecoder? = null
+    private var decoderKey: Triple<MappedLayout, CtcLexiconTrie, Int>? = null
+
+    private fun decoderFor(mapped: MappedLayout, trie: CtcLexiconTrie, beamWidth: Int): CtcSwipeDecoder {
+        val key = Triple(mapped, trie, beamWidth)
+        decoderMemo?.let { if (decoderKey == key) return it }
+        val model = modelOrNull() ?: throw IllegalStateException("CTC model unavailable")
+        val built = CtcSwipeDecoder(
+            model, mapped.layout, trie,
+            CtcScoringParams.tunedV2(beamWidth = beamWidth, topK = TOP_K)
+        )
+        decoderMemo = built
+        decoderKey = key
+        return built
+    }
+
+    // ── Public surface (mirrors GeometricEngineAdapter) ─────────────────────────
+
+    /**
+     * Decode a completed swipe on the background thread and deliver a
+     * [PredictionResult] to [onResult] ON THE MAIN THREAD. Empty result when the
+     * layout/model/lexicon is unavailable or [language] isn't English (v1) — the
+     * caller treats that as a no-prediction swipe.
+     */
+    fun decodeAsync(
+        keyboard: KeyboardData,
+        params: KeyboardGeometry.Params,
+        frameWidthPx: Float,
+        frameHeightPx: Float,
+        swipePath: List<PointF>,
+        timestamps: List<Long>,
+        language: String,
+        onResult: (PredictionResult) -> Unit,
+    ) {
+        if (frameWidthPx <= 0f || frameHeightPx <= 0f || swipePath.isEmpty() ||
+            !language.equals(LANGUAGE, ignoreCase = true)
+        ) {
+            onResult(PredictionResult(emptyList(), emptyList()))
+            return
+        }
+        // Snapshot the mutable PointF trace NOW (raw view px; normalized later
+        // on the decode thread once the letter-box affine is known).
+        val n = swipePath.size
+        val rawX = FloatArray(n); val rawY = FloatArray(n); val rawT = LongArray(n)
+        for (i in 0 until n) {
+            rawX[i] = swipePath[i].x
+            rawY[i] = swipePath[i].y
+            rawT[i] = timestamps.getOrElse(i) { 0L }
+        }
+        tasks.cancelAndSubmit {
+            try {
+                val mapped = layoutFor(keyboard, params, frameWidthPx, frameHeightPx)
+                val trie = if (mapped != null) trieFor() else null
+                val model = if (trie != null) modelOrNull() else null
+                val result = if (mapped == null || trie == null || model == null) {
+                    PredictionResult(emptyList(), emptyList())
+                } else {
+                    val px = DoubleArray(n) { ((rawX[it] - mapped.originX) * mapped.invW).toDouble() }
+                    val py = DoubleArray(n) { ((rawY[it] - mapped.originY) * mapped.invH).toDouble() }
+                    val pt = DoubleArray(n) { rawT[it].toDouble() }
+                    val beamWidth = Config.globalConfig().ctc_beam_width.coerceIn(10, 300)
+                    val candidates = decoderFor(mapped, trie, beamWidth).decode(px, py, pt)
+                    toPredictionResult(candidates)
+                }
+                if (!Thread.currentThread().isInterrupted) {
+                    mainHandler.post { onResult(result) }
+                }
+            } catch (e: InterruptedException) {
+                // Cancelled by a newer swipe — drop silently.
+            } catch (e: Exception) {
+                Log.e(TAG, "CTC decode failed", e)
+                if (!Thread.currentThread().isInterrupted) {
+                    mainHandler.post { onResult(PredictionResult(emptyList(), emptyList())) }
+                }
+            }
+        }
+    }
+
+    /** Engine-relative scores: softmax over final scores × 1000 (geometric parity). */
+    private fun toPredictionResult(candidates: List<CtcCandidate>): PredictionResult {
+        if (candidates.isEmpty()) return PredictionResult(emptyList(), emptyList())
+        val max = candidates.maxOf { it.finalScore }
+        val exps = candidates.map { exp(it.finalScore - max) }
+        val sum = exps.sum()
+        val words = candidates.map { it.word }
+        val scores = exps.map { ((it / sum) * 1000.0).roundToInt().coerceIn(0, 1000) }
+        return PredictionResult(words, scores)
+    }
+
+    /**
+     * Background warm-up: ONNX session + lexicon trie + layout mapping, so the
+     * first real swipe decodes in warm-path time. Idempotent via the memos.
+     */
+    fun warmUpAsync(
+        keyboard: KeyboardData,
+        params: KeyboardGeometry.Params,
+        frameWidthPx: Float,
+        frameHeightPx: Float,
+        language: String,
+    ) {
+        if (frameWidthPx <= 0f || frameHeightPx <= 0f) return
+        if (!language.equals(LANGUAGE, ignoreCase = true)) return
+        tasks.cancelAndSubmit {
+            try {
+                val mapped = layoutFor(keyboard, params, frameWidthPx, frameHeightPx)
+                    ?: return@cancelAndSubmit
+                val trie = trieFor() ?: return@cancelAndSubmit
+                modelOrNull() ?: return@cancelAndSubmit
+                if (BuildConfig.ENABLE_VERBOSE_LOGGING) {
+                    Log.d(TAG, "warmUp: model+trie(${trie.wordCount})+layout ready " +
+                        "(letters=${mapped.layout.alphabet.size})")
+                }
+            } catch (e: InterruptedException) {
+                // Superseded — decode's lazy path covers it.
+            } catch (e: Exception) {
+                Log.e(TAG, "CTC warmUp failed", e)
+            }
+        }
+    }
+
+    /**
+     * Cancels in-flight work and shuts the background thread down (IME teardown).
+     * The ORT session is intentionally NOT closed here: shutdown interrupts a
+     * possibly-running `session.run`, and closing a session mid-run is UB in ORT.
+     * The ~3 MB native session is reclaimed at process death — the same teardown
+     * posture as the neural orchestrator's sessions.
+     */
+    fun shutdown() {
+        tasks.shutdown()
+    }
+}
```

> Note on `Defaults_ONNX_THREADS_FALLBACK`: replace with a direct
> `Defaults.ONNX_XNNPACK_THREADS` reference + import when applying — written as a
> property here only to keep the diff self-describing. (`Defaults` is a top-level
> object in `Config.kt`, same package.)

#### 1(c)-iv `InputCoordinator.kt` (4 hunks)

```diff
--- a/src/main/kotlin/tribixbite/cleverkeys/InputCoordinator.kt
+++ b/src/main/kotlin/tribixbite/cleverkeys/InputCoordinator.kt
@@ -8,6 +8,7 @@ import android.view.inputmethod.EditorInfo
 import android.view.inputmethod.InputConnection
 import tribixbite.cleverkeys.ml.SwipeMLData
+import tribixbite.cleverkeys.swipe.CtcEngineAdapter
 import tribixbite.cleverkeys.swipe.GeometricEngineAdapter
 import tribixbite.cleverkeys.swipe.SwipeEngineRouter
@@ -438,6 +439,13 @@
         when (SwipeEngineRouter.route(
             keyboardView.getKeyboard(), SwipeEngineRouter.Mode.fromPref(config.swipe_engine_mode)
         )) {
             SwipeEngineRouter.Engine.NONE -> return
             SwipeEngineRouter.Engine.GEOMETRIC -> {
                 performGeometricSwipeTyping(
                     swipedKeys, swipePath, timestamps, ic, editorInfo, resources,
                     wasShiftActive, wasShiftLocked
                 )
                 return
             }
+            SwipeEngineRouter.Engine.CTC -> {
+                performCtcSwipeTyping(
+                    swipedKeys, swipePath, timestamps, ic, editorInfo, resources,
+                    wasShiftActive, wasShiftLocked
+                )
+                return
+            }
             SwipeEngineRouter.Engine.NEURAL -> Unit // falls through to the neural flow below
         }
```

```diff
@@ -304,6 +312,7 @@
     fun shutdown() {
         cancelPendingCursorSync()
         geometricAdapter?.shutdown()
+        ctcAdapter?.shutdown()
     }
```

After `performGeometricSwipeTyping` (the `── geometric engine path ──` block), add
the CTC twin:

```diff
@@ -569,6 +578,49 @@
+    // ── G5: CTC engine path (QWERTY-Latin layouts under ctc mode) ───────────────────
+
+    private var ctcAdapter: CtcEngineAdapter? = null
+
+    private fun ctcAdapterOrCreate(): CtcEngineAdapter =
+        ctcAdapter ?: CtcEngineAdapter(context).also { ctcAdapter = it }
+
+    /**
+     * Decodes a swipe with the CTC engine (off the main thread) and feeds the result
+     * into the SAME pipeline as neural/geometric results — [handlePredictionResults]
+     * → [SuggestionHandler.handleSwipePredictionResults] — inheriting the password
+     * guard, possessive augmentation, shift/caps transform, and THE commit engine.
+     * An empty decode (no model, non-English dictionary, degenerate trace) flows
+     * through as an empty prediction list → the pipeline clears the bar.
+     */
+    private fun performCtcSwipeTyping(
+        swipedKeys: List<KeyboardData.Key>,
+        swipePath: List<android.graphics.PointF>?,
+        timestamps: List<Long>?,
+        ic: InputConnection?,
+        editorInfo: EditorInfo?,
+        resources: Resources,
+        wasShiftActive: Boolean,
+        wasShiftLocked: Boolean
+    ) {
+        if (swipePath.isNullOrEmpty() || timestamps == null) return
+        val keyboard = keyboardView.getKeyboard() ?: return
+        val params = keyboardView.geometryParams() ?: return
+        val frameW = keyboardView.width.toFloat()
+        val frameH = keyboardView.height.toFloat()
+        if (frameW <= 0f || frameH <= 0f) return
+
+        // Same swipe-state + ML-trace capture as the neural/geometric paths.
+        beginSwipeCapture(swipedKeys, swipePath, timestamps, resources)
+
+        val language = predictionCoordinator.getDictionaryManager()?.getCurrentLanguage()
+            ?: config.primary_language
+        ctcAdapterOrCreate().decodeAsync(
+            keyboard, params, frameW, frameH, swipePath, timestamps, language
+        ) { result ->
+            handlePredictionResults(
+                result.words, result.scores, ic, editorInfo, resources,
+                wasShiftActive, wasShiftLocked
+            )
+        }
+    }
```

Prewarm — generalize the existing geometric-only hook (keeps the public name so
`CleverKeysService.kt:687` is untouched; only the body changes):

```diff
@@ -577,18 +629,30 @@
     fun prewarmGeometricEngine() {
         if (!config.swipe_typing_enabled) return
         val mode = SwipeEngineRouter.Mode.fromPref(config.swipe_engine_mode)
         if (mode == SwipeEngineRouter.Mode.NEURAL) return
         keyboardView.post {
             val keyboard = keyboardView.getKeyboard() ?: return@post
-            if (SwipeEngineRouter.route(keyboard, mode) != SwipeEngineRouter.Engine.GEOMETRIC) {
-                return@post
-            }
             val params = keyboardView.geometryParams() ?: return@post
             val frameW = keyboardView.width.toFloat()
             val frameH = keyboardView.height.toFloat()
             if (frameW <= 0f || frameH <= 0f) return@post
             val language = predictionCoordinator.getDictionaryManager()?.getCurrentLanguage()
                 ?: config.primary_language
-            geometricAdapterOrCreate().warmUpAsync(keyboard, params, frameW, frameH, language)
+            when (SwipeEngineRouter.route(keyboard, mode)) {
+                SwipeEngineRouter.Engine.GEOMETRIC ->
+                    geometricAdapterOrCreate().warmUpAsync(keyboard, params, frameW, frameH, language)
+                // G5: front-load the ONNX session + 98k-word trie build (~100-300 ms
+                // background) so the first ctc swipe decodes in warm-path time.
+                SwipeEngineRouter.Engine.CTC ->
+                    ctcAdapterOrCreate().warmUpAsync(keyboard, params, frameW, frameH, language)
+                else -> return@post
+            }
         }
     }
```

(Also update the fn's KDoc first line to "…of the geometric/CTC engine…".)

#### 1(c)-v Provenance tagging

```diff
--- a/src/main/kotlin/tribixbite/cleverkeys/SuggestionProvenance.kt
+++ b/src/main/kotlin/tribixbite/cleverkeys/SuggestionProvenance.kt
@@ -26,6 +26,9 @@ enum class SuggestionOrigin {
     /** Neural ONNX beam-search swipe output. */
     NEURAL_BEAM,
 
     /** Geometric swipe-decoder output. */
     GEOMETRIC,
 
+    /** CTC trie-beam swipe-decoder output (G5 `ctc` engine mode). */
+    CTC,
+
     /** Dictionary prefix completion of the typed partial (WordPredictor). */
     DICTIONARY_PREFIX,
@@ -50,10 +53,13 @@
         /**
          * Origin tag for the swipe path from the configured engine mode
-         * ("neural" | "hybrid" | "geometric"). Hybrid routes per-layout at
-         * swipe time; it is tagged NEURAL_BEAM here because the neural engine
-         * serves its QWERTY-family default (documented approximation).
+         * ("neural" | "hybrid" | "geometric" | "ctc"). Hybrid and ctc route
+         * per-layout at swipe time; each is tagged by the engine serving its
+         * QWERTY-family default (documented approximation — a non-QWERTY swipe
+         * under ctc mode is actually geometric but tagged CTC).
          */
         fun forSwipeEngineMode(mode: String?): SuggestionOrigin =
-            if (mode == "geometric") GEOMETRIC else NEURAL_BEAM
+            when (mode) {
+                "geometric" -> GEOMETRIC
+                "ctc" -> CTC
+                else -> NEURAL_BEAM
+            }
     }
 }
@@ -191,6 +197,7 @@ object ProvenanceFormatter {
     fun originLabel(origin: SuggestionOrigin): String = when (origin) {
         SuggestionOrigin.NEURAL_BEAM -> "Neural swipe (beam search)"
         SuggestionOrigin.GEOMETRIC -> "Geometric swipe decoder"
+        SuggestionOrigin.CTC -> "CTC swipe (trie beam)"
         SuggestionOrigin.DICTIONARY_PREFIX -> "Dictionary prefix match"
```

```diff
--- a/src/main/kotlin/tribixbite/cleverkeys/SuggestionBar.kt
+++ b/src/main/kotlin/tribixbite/cleverkeys/SuggestionBar.kt
@@ -405,6 +405,7 @@
     private fun originMarkerColor(origin: SuggestionOrigin): Int = when (origin) {
         SuggestionOrigin.NEURAL_BEAM -> 0xFFB39DDB.toInt()       // purple
         SuggestionOrigin.GEOMETRIC -> 0xFF80CBC4.toInt()         // teal
+        SuggestionOrigin.CTC -> 0xFF9FA8DA.toInt()               // indigo
         SuggestionOrigin.DICTIONARY_PREFIX -> 0xFF90CAF9.toInt() // blue
```

#### 1(c)-vi Settings UI — dropdown, buttons, strings, new activity

```diff
--- a/src/main/kotlin/tribixbite/cleverkeys/ui/settings/sections/NeuralPredictionSection.kt
+++ b/src/main/kotlin/tribixbite/cleverkeys/ui/settings/sections/NeuralPredictionSection.kt
@@ -18,6 +18,7 @@ import tribixbite.cleverkeys.SettingsActivity
 import tribixbite.cleverkeys.ui.settings.CollapsibleSettingsSection
 import tribixbite.cleverkeys.ui.settings.SettingsDropdown
 import tribixbite.cleverkeys.ui.settings.SettingsSwitch
+import tribixbite.cleverkeys.ui.settings.openCtcSettings
 import tribixbite.cleverkeys.ui.settings.openGeometricSettings
 import tribixbite.cleverkeys.ui.settings.openNeuralSettings
 import tribixbite.cleverkeys.ui.settings.saveSetting
@@ -43,22 +44,25 @@
                 if (swipeTypingEnabled) {
                     // WP9 R-1 step 7 (v1.1): engine mode selector. Hybrid = neural on QWERTY +
                     // geometric elsewhere; Neural = QWERTY-only swipe (pre-geo behavior);
-                    // Geometric = SHARK2 on all layouts.
+                    // Geometric = SHARK2 on all layouts; CTC (G5) = CTC trie-beam on
+                    // QWERTY + geometric elsewhere.
                     SettingsDropdown(
                         title = stringResource(R.string.swipe_engine_mode_title),
                         description = stringResource(R.string.swipe_engine_mode_desc),
-                        options = listOf("Hybrid", "Neural", "Geometric"),
+                        options = listOf("Hybrid", "Neural", "Geometric", "CTC"),
                         selectedIndex = when (swipeEngineMode) {
                             "hybrid" -> 0
                             "geometric" -> 2
+                            "ctc" -> 3
                             else -> 1 // "neural" (default)
                         },
                         onSelectionChange = { index ->
                             swipeEngineMode = when (index) {
                                 0 -> "hybrid"
                                 2 -> "geometric"
+                                3 -> "ctc"
                                 else -> "neural"
                             }
                             saveSetting("swipe_engine_mode", swipeEngineMode)
                         }
                     )
                 }
@@ -113,16 +117,27 @@
                         Text("Full Neural Settings")
                     }
 
-                    // Geometric engine tuning — only meaningful when a mode that uses it is on.
-                    if (swipeEngineMode != "neural") {
+                    // Geometric engine tuning — only meaningful when a mode that uses it
+                    // is on (hybrid/geometric always; ctc uses it for non-QWERTY layouts).
+                    if (swipeEngineMode == "hybrid" || swipeEngineMode == "geometric" ||
+                        swipeEngineMode == "ctc"
+                    ) {
                         Button(
                             onClick = { openGeometricSettings() },
                             modifier = Modifier
                                 .fillMaxWidth()
                                 .padding(top = 8.dp)
                         ) {
                             Text("Full Geometric Settings")
                         }
                     }
+
+                    // CTC engine tuning (G5) — only under the ctc mode.
+                    if (swipeEngineMode == "ctc") {
+                        Button(
+                            onClick = { openCtcSettings() },
+                            modifier = Modifier
+                                .fillMaxWidth()
+                                .padding(top = 8.dp)
+                        ) {
+                            Text("Full CTC Settings")
+                        }
+                    }
                 }
             }
 }
```

> The `swipeEngineMode != "neural"` → explicit-set change is semantically identical
> for the existing modes and prevents the Geometric button from silently vanishing
> for `ctc` (where geometric still serves non-QWERTY layouts). If you prefer the
> minimal diff, keep `!= "neural"` — behavior is the same for `ctc` too.

```diff
--- a/res/values/strings.xml
+++ b/res/values/strings.xml
@@ -122 +122 @@
-    <string name="swipe_engine_mode_desc">Hybrid: neural on QWERTY, geometric on other layouts. Neural: QWERTY layouts only. Geometric: all layouts.</string>
+    <string name="swipe_engine_mode_desc">Hybrid: neural on QWERTY, geometric on other layouts. Neural: QWERTY layouts only. Geometric: all layouts. CTC: CTC beam on QWERTY, geometric on other layouts.</string>
```

```diff
--- a/src/main/kotlin/tribixbite/cleverkeys/ui/settings/SettingsNavigation.kt
+++ b/src/main/kotlin/tribixbite/cleverkeys/ui/settings/SettingsNavigation.kt
@@ -31,6 +31,10 @@ internal fun SettingsActivity.openGeometricSettings() {
         startActivity(Intent(this, GeometricSettingsActivity::class.java))
 }
 
+internal fun SettingsActivity.openCtcSettings() {
+        startActivity(Intent(this, CtcSettingsActivity::class.java))
+}
+
 internal fun SettingsActivity.openCalibration() {
```

(Add `import tribixbite.cleverkeys.CtcSettingsActivity` beside the existing
`GeometricSettingsActivity` import in that file.)

```diff
--- a/AndroidManifest.xml
+++ b/AndroidManifest.xml
@@ -87,6 +87,7 @@
     <activity android:name="tribixbite.cleverkeys.GeometricSettingsActivity" android:label="Geometric Settings" android:theme="@style/settingsTheme" android:exported="false" android:directBootAware="true" tools:targetApi="24"/>
+    <activity android:name="tribixbite.cleverkeys.CtcSettingsActivity" android:label="CTC Settings" android:theme="@style/settingsTheme" android:exported="false" android:directBootAware="true" tools:targetApi="24"/>
```

New activity — a one-slider clone of `GeometricSettingsActivity` (same private
`ParameterSection`/`ParameterSlider` composables; copy them verbatim from that file
— they are file-private there by design):

```diff
--- /dev/null
+++ b/src/main/kotlin/tribixbite/cleverkeys/CtcSettingsActivity.kt
@@ -0,0 +1,150 @@
+package tribixbite.cleverkeys
+
+// imports: identical block to GeometricSettingsActivity.kt (ComponentActivity,
+// setContent, Compose foundation/material3, KeyboardTheme) minus mutableFloatStateOf.
+
+/**
+ * Full CTC Settings — the user-tunable knob of the CTC (trie-beam) swipe engine
+ * (G5; spec `docs/specs/ctc-swipe-engine.md`). Companion to
+ * [GeometricSettingsActivity], reached from Settings → Swipe Typing when the
+ * Prediction Engine is CTC.
+ *
+ * Deliberately exposes ONE knob — commit-phase beam width. The scoring constants
+ * (gamma/lambda/beta/prune) are `CtcScoringParams.tunedV2`, fitted offline on
+ * val-9918; exposing them would let users silently fall off the validated
+ * operating point (the published-preset control measured −2.3 pt top-1).
+ * Default MUST equal `Defaults.CTC_BEAM_WIDTH` (= the width every campaign
+ * validation number was decoded at). CtcEngineAdapter re-reads Config per decode,
+ * so changes apply on the next swipe with no engine rebuild beyond the memoized
+ * decoder swap.
+ */
+class CtcSettingsActivity : ComponentActivity() {
+
+    private var beamWidth by mutableIntStateOf(Defaults.CTC_BEAM_WIDTH)
+
+    override fun onCreate(savedInstanceState: Bundle?) {
+        super.onCreate(savedInstanceState)
+        loadSavedParameters()
+        setContent { KeyboardTheme { CtcSettingsScreen() } }
+    }
+
+    @Composable
+    private fun CtcSettingsScreen() {
+        val scrollState = rememberScrollState()
+        Column(
+            modifier = Modifier
+                .fillMaxSize()
+                .background(MaterialTheme.colorScheme.background)
+                .padding(16.dp)
+                .verticalScroll(scrollState),
+            verticalArrangement = Arrangement.spacedBy(16.dp)
+        ) {
+            Text(
+                text = "CTC Engine Settings",
+                fontSize = 24.sp,
+                fontWeight = FontWeight.Bold,
+                color = MaterialTheme.colorScheme.onBackground
+            )
+            Text(
+                text = "Tuning for the CTC swipe engine (QWERTY layouts under the " +
+                    "CTC prediction engine). Scoring constants are calibrated " +
+                    "offline and not user-tunable.",
+                fontSize = 14.sp,
+                color = MaterialTheme.colorScheme.onSurfaceVariant
+            )
+
+            ParameterSection(title = "Beam Search") {
+                ParameterSlider(
+                    title = "Beam Width",
+                    description = "Hypotheses kept per frame in the trie beam. " +
+                        "100 is the validated default; higher costs CPU per swipe " +
+                        "for marginal accuracy.",
+                    value = beamWidth.toFloat(),
+                    valueRange = 10f..300f,
+                    steps = 28,
+                    onValueChange = {
+                        beamWidth = it.toInt()
+                        updateParameters()
+                    },
+                    displayValue = beamWidth.toString()
+                )
+            }
+
+            Button(
+                onClick = {
+                    beamWidth = Defaults.CTC_BEAM_WIDTH
+                    updateParameters()
+                },
+                modifier = Modifier.fillMaxWidth()
+            ) {
+                Text("Reset to Validated Default")
+            }
+        }
+    }
+
+    // ParameterSection / ParameterSlider: copy the private composables verbatim
+    // from GeometricSettingsActivity.kt (lines 150-222 at app HEAD 79ddfb0f).
+
+    /** Push to the live Config and persist — the adapter reads Config per decode. */
+    private fun updateParameters() {
+        try {
+            Config.globalConfig().ctc_beam_width = beamWidth
+        } catch (e: Exception) {
+            android.util.Log.e("CtcSettings", "Error updating configuration", e)
+        }
+        DirectBootAwarePreferences.get_shared_preferences(this)
+            .edit()
+            .putInt("ctc_beam_width", beamWidth)
+            .apply()
+    }
+
+    private fun loadSavedParameters() {
+        val prefs = DirectBootAwarePreferences.get_shared_preferences(this)
+        beamWidth = Config.safeGetInt(prefs, "ctc_beam_width", Defaults.CTC_BEAM_WIDTH)
+    }
+}
```

#### 1(c)-vii Settings plumbing — defaults map, search entry, reset presets

```diff
--- a/src/main/kotlin/tribixbite/cleverkeys/backup/SettingsDefaults.kt
+++ b/src/main/kotlin/tribixbite/cleverkeys/backup/SettingsDefaults.kt
@@ -263,6 +263,9 @@
     "swipe_engine_mode" to PrefValue.Str(Defaults.SWIPE_ENGINE_MODE),
     // Full Geometric Settings (GeometricSettingsActivity) — user-tunable geo engine knobs.
     "geo_max_results" to PrefValue.IntV(Defaults.GEO_MAX_RESULTS),
     "geo_frequency_weight" to PrefValue.FloatV(Defaults.GEO_FREQUENCY_WEIGHT),
     "geo_endpoint_inset_kw" to PrefValue.FloatV(Defaults.GEO_ENDPOINT_INSET_KW),
+    // Full CTC Settings (CtcSettingsActivity) — G5 ctc engine knob. The "ctc"
+    // engine-mode VALUE needs no migration: fromPref falls back to NEURAL on
+    // versions that predate it.
+    "ctc_beam_width" to PrefValue.IntV(Defaults.CTC_BEAM_WIDTH),
```

```diff
--- a/src/main/kotlin/tribixbite/cleverkeys/SettingsActivity.kt
+++ b/src/main/kotlin/tribixbite/cleverkeys/SettingsActivity.kt
@@ -579,6 +579,7 @@
             SearchableSetting("Geometric Settings", listOf("geometric", "shape", "shark", "swipe engine", "tolerance"), "Neural Prediction", GeometricSettingsActivity::class.java),
+            SearchableSetting("CTC Settings", listOf("ctc", "futo", "swipe engine", "beam", "trie"), "Neural Prediction", CtcSettingsActivity::class.java, gatedBy = "swipe_typing", settingId = "ctc_settings"),
             SearchableSetting("ONNX Threads", listOf("threads", "cpu", "xnnpack", "performance", "onnx"), "Neural Prediction", NeuralSettingsActivity::class.java, gatedBy = "swipe_typing", settingId = "onnx_threads"),
```

```diff
--- a/src/main/kotlin/tribixbite/cleverkeys/ui/settings/SettingsResetPresets.kt
+++ b/src/main/kotlin/tribixbite/cleverkeys/ui/settings/SettingsResetPresets.kt
@@ (end of the "// Neural prediction - BALANCED profile" block, ~line 101) @@
             editor.putInt("onnx_xnnpack_threads", Defaults.ONNX_XNNPACK_THREADS)
+            // CTC engine (G5). NOTE: swipe_engine_mode and geo_* are deliberately
+            // NOT reset here (pre-existing behavior — engine choice survives a
+            // preset reset); ctc_beam_width follows the geo_* precedent and is
+            // also left alone. Remove this comment if that decision changes.
```

(i.e. **no functional change** to `SettingsResetPresets.kt` — `ctc_beam_width`
deliberately follows the existing `geo_*` precedent of not being preset-reset. If
the maintainer prefers resetting it, add the `putInt` line instead.)

No `SettingsPersistence.kt` change is needed: `swipeEngineMode` is already generic
string state, and `ctc_beam_width` is self-persisted by its activity exactly like
the `geo_*` knobs (which also have no `handlePreferenceChanged` branch).

### 1(d) Asset placement + size impact

```bash
# from the app repo root (G3 step 1)
cp /home/will/git/CleverKeys-ML/ctc/artifacts/ch128_s1234.onnx \
   src/main/assets/models/ctc_swipe_encoder.onnx
sha256sum src/main/assets/models/ctc_swipe_encoder.onnx
# must print 6c1144949e545f626419e1fa7b29e80f9ecf3e303886f30411fc37ae72c45c51
```

| Item | Bytes | Note |
|---|---|---|
| `ch128_s1234.onnx` (ship, D1) | 2,799,865 | test-validated: clears all 5 FUTO-ceiling bars on every seed; 0.455–0.475 ms desktop single-thread |
| current `models/` assets | 10,293,047 | swipe_encoder 5,317,537 + swipe_decoder 4,975,510 |
| APK delta | ≈ +2.6–2.8 MB per ABI APK | fp32 conv weights are high-entropy; zip gains are small. Applies to each of the 3 ABI APKs (asset is ABI-independent but each APK carries all assets) |

**Superseded in part by §7.2** — the shelf has changed twice since this was
written (`resbn80g` in Phase G, `sw2345` in Phase J). The copy command above is
correct in *form* for any of them: same asset path, same contract, only the source
artifact and the expected sha change.

Alternatives on the shelf (do NOT ship two models — see O2):
`fast_resbn80_s1234.onnx` (1,142,727 B, 0.215 ms, val-only evidence, wider t5
margin) and `fast_resbn72_s1234.onnx` (944,487 B, 0.186 ms, val-only). If one of
these is chosen instead, the golden fixture MUST be regenerated from that artifact
(`make_golden.py --onnx artifacts/fast_resbn80_s1234.onnx --preset
1.05,1.1,0.2,0.3734,0.9882`) — the fixture records `source_onnx_sha256` and the app
parity test is meaningless against a different graph. The stale pre-campaign
`artifacts/ctc_swipe_encoder.onnx` (r2, 394 k params) must NOT be shipped.

### 1(e) Golden fixture + parity wiring

> **Fixture bytes below are stale (2026-08-11).** The wiring, the two copy
> destinations and the "fixture and preset move together" rule are unchanged, but
> `artifacts/ctc_model_golden.json` is no longer the ch 128 fixture quoted here —
> it was regenerated from `resbn80g_s1234` at the `resbn80g` preset. Current
> fixture state, and what adopting the Phase-J finalist would require, is **§7.2**.

**Missing-fixture fact (verified):** `src/test/resources/ctc/` does not exist in
the app repo; `CtcParityTest` has been failing its `assertWithMessage("golden
fixture must exist…")` since the module landed (RESULTS.md audit finding #4), and
its generator (`scratchpad/gen_ctc_golden.py`) is gone. The regenerated
`artifacts/ctc_model_golden.json` in THIS repo replaces both: it satisfies the
existing `CtcParityTest` schema (6 `featurize` + 4 `beam` cases) AND carries the
`points → features → emissions` pairs plus the canonical `layout` block for the new
model-backed test.

```bash
# G3 step 2 — pure-JVM fixture (CtcParityTest)
mkdir -p src/test/resources/ctc
cp /home/will/git/CleverKeys-ML/ctc/artifacts/ctc_model_golden.json \
   src/test/resources/ctc/ctc_golden.json

# G3 step 3 — instrumented fixture (same bytes; androidTest APK assets)
mkdir -p src/androidTest/assets/ctc
cp /home/will/git/CleverKeys-ML/ctc/artifacts/ctc_model_golden.json \
   src/androidTest/assets/ctc/ctc_golden.json
```

Both copies come from one source artifact
(sha256 `a18ea58cd662b0e18b6daadaf417361f93fd0b146ce6478d4d6a62e7e185fa8a`); the
instrumented parity test asserts the copy it reads matches the model it runs, so
drift between the two copies surfaces as a test failure, not silence.

`CtcParityTest` itself needs **no code change** — it already reads
`src/test/resources/ctc/ctc_golden.json`, iterates by `kind`, and both of its
tests now find cases. Two doc-comment touch-ups are optional (the fixture is now
regenerated by `CleverKeys-ML/ctc/make_golden.py`, not `scratchpad/gen_ctc_golden.py`).

New instrumented test (full file) — validates the ONNX-backed seam against the
frozen features→emissions pairs and the end-to-end decode:

```diff
--- /dev/null
+++ b/src/androidTest/kotlin/tribixbite/cleverkeys/swipe/CtcEmissionModelParityTest.kt
@@ -0,0 +1,150 @@
+package tribixbite.cleverkeys.swipe
+
+import ai.onnxruntime.OrtEnvironment
+import androidx.test.ext.junit.runners.AndroidJUnit4
+import androidx.test.platform.app.InstrumentationRegistry
+import com.google.common.truth.Truth.assertThat
+import com.google.common.truth.Truth.assertWithMessage
+import org.json.JSONObject
+import org.junit.AfterClass
+import org.junit.BeforeClass
+import org.junit.Test
+import org.junit.runner.RunWith
+import tribixbite.cleverkeys.onnx.ModelLoader
+import tribixbite.cleverkeys.swipe.ctc.CtcBeamDecoder
+import tribixbite.cleverkeys.swipe.ctc.CtcFeaturizer
+import tribixbite.cleverkeys.swipe.ctc.CtcLayout
+import tribixbite.cleverkeys.swipe.ctc.CtcLexiconTrie
+import tribixbite.cleverkeys.swipe.ctc.CtcScoringParams
+
+/**
+ * G3 model-backed parity: the bundled `models/ctc_swipe_encoder.onnx`, run through
+ * [OnnxCtcEmissionModel], must reproduce the frozen features→emissions→top-k chain
+ * in `ctc/ctc_golden.json` (generated by CleverKeys-ML `ctc/make_golden.py` from
+ * the SAME artifact — the fixture records its `source_onnx_sha256`).
+ *
+ * The session is loaded with hardware acceleration DISABLED (plain ORT CPU EP) so
+ * numerics are comparable to the desktop CPUExecutionProvider the fixture was
+ * generated with; export-time parity bounded |onnx−torch| at 3.8e-5 on the sliced
+ * head, so EMISSION_TOL leaves ~50× headroom for cross-platform libm drift while
+ * still catching any real wiring bug (wrong tensor order, missing mask, bad slice).
+ */
+@RunWith(AndroidJUnit4::class)
+class CtcEmissionModelParityTest {
+
+    companion object {
+        private const val FIXTURE = "ctc/ctc_golden.json"
+        private const val EMISSION_TOL = 2e-3
+        private const val SCORE_TOL = 1e-3
+
+        private lateinit var model: OnnxCtcEmissionModel
+        private lateinit var golden: JSONObject
+
+        @JvmStatic
+        @BeforeClass
+        fun setUp() {
+            val target = InstrumentationRegistry.getInstrumentation().targetContext
+            val testCtx = InstrumentationRegistry.getInstrumentation().context
+            golden = JSONObject(
+                testCtx.assets.open(FIXTURE).readBytes().decodeToString()
+            )
+            val env = OrtEnvironment.getEnvironment()
+            val loaded = ModelLoader(target, env).loadModel(
+                CtcEngineAdapter.MODEL_ASSET, "CtcEncoderParity",
+                enableHardwareAcceleration = false, xnnpackThreads = 1
+            )
+            model = OnnxCtcEmissionModel(env, loaded.session)
+        }
+
+        @JvmStatic
+        @AfterClass
+        fun tearDown() {
+            if (::model.isInitialized) model.close()
+        }
+
+        private fun fixtureLayout(): CtcLayout {
+            val lay = golden.getJSONObject("layout")
+            val letters = lay.getString("letters").toList()
+            val cx = lay.getJSONArray("cx")
+            val cy = lay.getJSONArray("cy")
+            return CtcLayout.of(
+                letters,
+                List(cx.length()) { cx.getDouble(it).toFloat() },
+                List(cy.length()) { cy.getDouble(it).toFloat() },
+            )
+        }
+    }
+
+    @Test
+    fun emissions_matchGoldenWithinTolerance() {
+        val padded = CtcFeaturizer.buildPaddedLayout(fixtureLayout())
+        val cases = golden.getJSONArray("cases")
+        var checked = 0
+        for (i in 0 until cases.length()) {
+            val c = cases.getJSONObject(i)
+            if (c.getString("kind") != "beam") continue
+            val name = c.getString("name")
+            val featJson = c.getJSONArray("features")
+            val features = FloatArray(featJson.length()) { featJson.getDouble(it).toFloat() }
+
+            val emissions = model.emit(features, padded)
+            assertWithMessage("$name: frames").that(emissions.frames)
+                .isEqualTo(c.getInt("frames"))
+            assertWithMessage("$name: numClasses").that(emissions.numClasses)
+                .isEqualTo(c.getInt("numClasses"))
+            val rows = c.getJSONArray("emissions")
+            var worst = 0.0
+            for (t in 0 until emissions.frames) {
+                val row = rows.getJSONArray(t)
+                for (k in 0 until emissions.numClasses) {
+                    val d = Math.abs(emissions.at(t, k) - row.getDouble(k))
+                    if (d > worst) worst = d
+                }
+            }
+            assertWithMessage("$name: max |emissions - golden| = $worst")
+                .that(worst).isLessThan(EMISSION_TOL)
+            checked++
+        }
+        assertWithMessage("must exercise model-backed cases").that(checked).isGreaterThan(0)
+    }
+
+    @Test
+    fun endToEnd_featurizeEmitDecode_matchesGoldenTopK() {
+        val layout = fixtureLayout()
+        val padded = CtcFeaturizer.buildPaddedLayout(layout)
+        val cases = golden.getJSONArray("cases")
+        var checked = 0
+        for (i in 0 until cases.length()) {
+            val c = cases.getJSONObject(i)
+            if (c.getString("kind") != "beam") continue
+            val name = c.getString("name")
+            val pts = c.getJSONObject("points")
+            val px = pts.getJSONArray("x"); val py = pts.getJSONArray("y")
+            val pt = pts.getJSONArray("t")
+            val features = CtcFeaturizer.featurize(
+                DoubleArray(px.length()) { px.getDouble(it) },
+                DoubleArray(py.length()) { py.getDouble(it) },
+                DoubleArray(pt.length()) { pt.getDouble(it) },
+            )
+            // Featurizer must be bit-identical on-device too.
+            val featJson = c.getJSONArray("features")
+            for (k in features.indices) {
+                assertWithMessage("$name: feature[$k]").that(features[k])
+                    .isEqualTo(featJson.getDouble(k).toFloat())
+            }
+
+            val trie = CtcLexiconTrie(layout.alphabet.copyOf())
+            val lex = c.getJSONArray("lexicon")
+            for (j in 0 until lex.length()) {
+                val e = lex.getJSONArray(j)
+                trie.insert(e.getString(0), e.getDouble(1))
+            }
+            val p = c.getJSONObject("params")
+            val params = CtcScoringParams(
+                gamma = p.getDouble("gamma"), lambda = p.getDouble("lambda"),
+                beta = p.getDouble("beta"), alpha = p.getDouble("alpha"),
+                gammaPrune = p.getDouble("gammaPrune"), betaPrune = p.getDouble("betaPrune"),
+                beamWidth = p.getInt("beamWidth"), topK = p.getInt("topK"),
+            )
+            val emissions = model.emit(features, padded)
+            assertThat(CtcBeamDecoder.greedy(emissions, layout.alphabet))
+                .isEqualTo(c.getString("greedy"))
+            val result = CtcBeamDecoder.decode(emissions, trie, params)
+            val expected = c.getJSONArray("topk")
+            val expectedWords = (0 until expected.length())
+                .map { expected.getJSONArray(it).getString(0) }
+            assertWithMessage("$name: top-k words").that(result.map { it.word })
+                .isEqualTo(expectedWords)
+            for (k in result.indices) {
+                assertWithMessage("$name: score[$k]").that(result[k].finalScore)
+                    .isWithin(SCORE_TOL).of(expected.getJSONArray(k).getDouble(1))
+            }
+            checked++
+        }
+        assertWithMessage("must exercise model-backed cases").that(checked).isGreaterThan(0)
+    }
+}
```

> `CtcEngineAdapter.MODEL_ASSET` is `const` and public for exactly this reuse. The
> fixture's `params.topK` is 4 and `beamWidth` 32 — the decode assertions run at
> fixture params, independent of the app's `ctc_beam_width` default.

### 1(f) NOTICE additions

```diff
--- a/NOTICE
+++ b/NOTICE
@@ -41,3 +41,32 @@
 Apache-2.0 is one-way compatible with GPLv3; the derived word selections
 distributed with CleverKeys are redistributed under GPL-3.0, with attribution
 preserved here.
+
+================================================================================
+FUTO swipe gesture corpus (swipe.futo.org)
+================================================================================
+The bundled CTC swipe-emission encoder
+(src/main/assets/models/ctc_swipe_encoder.onnx) was trained from scratch by the
+CleverKeys project on gesture data derived from the FUTO swipe corpus. No FUTO
+model weights and no FUTO model outputs were used at any point in training; the
+bundled weights are original CleverKeys work distributed under GPL-3.0 with the
+app. (The FUTO Model Weights License therefore does not apply to this asset;
+the decode algorithms in src/main/kotlin/.../swipe/ctc/ are a clean-room port of
+FUTO's GPL-3.0 swipe-library, license-compatible with this project.)
+
+  swipe.futo.org dataset — FUTO
+  https://huggingface.co/datasets/futo-org/swipe.futo.org
+  License: MIT
+
+================================================================================
+How-We-Swipe shape-writing dataset
+================================================================================
+Training of the CTC swipe-emission encoder additionally used the How-We-Swipe
+shape-writing dataset.
+
+  How-We-Swipe — Luis A. Leiva, Sunjun Kim, Wenzhe Cui, Xiaojun Bi,
+  Antti Oulasvirta (OSF node sj67f)
+  https://osf.io/sj67f/
+  License: MIT (Copyright (c) 2021 the authors above, as declared on the OSF node)
+
+MIT is GPLv3-compatible; the trained weights distributed with CleverKeys are
+redistributed under GPL-3.0, with attribution preserved here.
```

---

## 2. SETTINGS AUDIT — every existing tuning control vs the CTC engine

Source of truth: `NeuralSettingsActivity.kt` (all controls are file-private
`ParameterSlider`/`ParameterDropdown`/`ParameterToggle`, invisible to the generated
search index) + `GeometricSettingsActivity.kt` + `Config.kt` `Defaults`.

| Control (pref key) | Current min/max/default | Applies to CTC? | Action for `ctc` mode |
|---|---|---|---|
| Beam Width (`neural_beam_width`) | 1–16 / 6 | **No.** Different beam entirely: neural is an autoregressive transformer beam (each width step = a decoder NN call); CTC is a CPU trie-Viterbi beam where width 100 is cheap. Reusing the key would clamp CTC to ≤16 and corrupt neural tuning. | Leave untouched; CTC gets its own `ctc_beam_width` (below). No range/default change. |
| Max Sequence Length (`neural_max_length`) | 10–50 / 20 | **No.** CTC word length is bounded by trie depth; there is no generation loop to cap. | N/A — no change, ignored by CTC path. |
| Confidence Threshold (`neural_confidence_threshold`) | 0.0–1.0 / 0.01 | **No.** CTC final scores are length-normalized log-domain values, not probabilities; a 0–1 threshold is meaningless against them. | N/A. (A future confidence-gated cascade — decision doc §3 — would need its own calibrated threshold pref; out of scope for G5.) |
| Length Penalty Alpha (`neural_beam_alpha`) | 0.0–3.0 / 1.4 | **No.** CTC's length normalization is `gamma` inside `tunedV2` — fitted offline, deliberately not user-tunable (published-preset control measured −2.3 pt). | N/A. |
| Pruning Confidence (`neural_beam_prune_confidence`) | 0.0–1.0 / 0.8 | **No** (autoregressive-beam pruning). CTC pruning is `gammaPrune`/`betaPrune` in the preset. | N/A. |
| Score Gap Threshold (`neural_beam_score_gap`) | 0–150 / 80 | **No** (early-stop heuristic of the neural beam). | N/A. |
| Width Pruning Step (`neural_adaptive_width_step`) | 3–20 / 12 | **No.** | N/A. |
| Early Stop Step (`neural_score_gap_step`) | 3–20 / 12 | **No.** | N/A. |
| Temperature (`neural_temperature`) | 0.1–3.0 / 1.0 | **No.** CTC emissions are consumed as raw log-probs; the beam is max-merge (temperature would only rescale monotonically per frame — no ranking effect within a frame, untested interactions across frames). | N/A. |
| Vocabulary Frequency Weight (`neural_frequency_weight`) | 0.0–2.0 / 0.57 | **No.** CTC's frequency lever is `lambda` (1.1) inside the preset, calibrated against the AOSP log scale (spec NFR-4). Reusing this pref would double-apply frequency. | N/A. |
| Touch Smoothing (`swipe_smoothing_window`) | 1–7 / 3 | **No — deliberately.** The CTC featurizer performs its own two-stage resample, and the model was trained on raw (unsmoothed) traces. Pre-smoothing would shift the input distribution off-training. `CtcEngineAdapter` reads the raw `swipePath`. | N/A — document in the CTC settings screen header if users ask. |
| Trajectory Resampling (`neural_resampling_mode`) | discard/interpolate/average | **No.** CTC resampling is fixed by `CtcFeaturizer` (60 Hz → 64, part of the parity contract — changing it breaks bit-parity with training). | N/A, must never apply. |
| Batch Processing (`neural_batch_beams`) | bool / false | **No** (neural decoder batching). | N/A. |
| Greedy Search (`neural_greedy_search`) | bool / false | **No.** (CTC has `CtcBeamDecoder.greedy` but it ignores the lexicon — a debug tool, not a user mode.) | N/A. |
| ONNX Threads (`onnx_xnnpack_threads`) | 1–8 / 2 | **YES.** `CtcEngineAdapter` loads its session through the same `ModelLoader`, which applies this thread count to XNNPACK. The CTC encoder is so small (0.5 ms class) that the setting is latency-neutral for it, but honoring it keeps one mental model. | Reused as-is; no range/default change. Description already says "restart required" — accurate here too (session built once per process). |
| `neural_max_cumulative_boost`, `neural_strict_start_char`, `neural_prefix_boost_*`, `neural_user_max_seq_length` | persisted, no UI | **No** (neural vocab/boost pipeline). | N/A. |
| Geo: Max Suggestions (`geo_max_results`) | 3–15 / 10 | **Not for the CTC path** (CTC slate size is the adapter's `TOP_K = 8` constant), but **still live under `ctc` mode** for non-QWERTY layouts routed to geometric. | Leave untouched; keep the Full Geometric Settings button visible under ctc mode (§1c-vi). |
| Geo: Frequency Weight (`geo_frequency_weight`) | 0.0–0.4 / 0.12 | Same as above. | Leave untouched. |
| Geo: Endpoint Tolerance (`geo_endpoint_inset_kw`) | 0.0–0.8 / 0.30 | Same as above. | Leave untouched. |
| Prediction Engine (`swipe_engine_mode`) | hybrid/neural/geometric, default neural | **YES — the entry point.** | Add `"ctc"` as the 4th dropdown option (§1c-vi). Default stays `"neural"` (D7). Unknown-value fallback in `fromPref` already covers downgrade safety. |

**Hide-for-ctc audit:** nothing needs hiding. The "Full Neural Settings" button
stays (neural remains reachable by switching modes; the screen is engine-scoped, not
mode-scoped — same reasoning that keeps it visible under `geometric` today). The
only conditional-UI fix is the Geometric-button gate (§1c-vi) so it doesn't
disappear under `ctc`, plus the new CTC button.

### NEW ctc-specific settings

| Pref key | Type / default | Range | UI | Registration |
|---|---|---|---|---|
| `ctc_beam_width` | Int / **100** | slider 10–300, steps 28 (10-unit detents) | `CtcSettingsActivity` (new, §1c-vi), reached via "Full CTC Settings" button shown when mode == ctc | `Defaults.CTC_BEAM_WIDTH` + `Config.ctc_beam_width` (+ `coerceIn(10,300)` in `refresh()`) + `SETTINGS_DEFAULTS["ctc_beam_width"] = PrefValue.IntV(...)` |

Answer to "the CTC beam is width-100 by spec — expose or fix?": **both.** Fix the
default at 100 (the committed presets' 300 is FUTO's ship value; every CleverKeys-ML
validation number — val and the sealed test — was decoded at width 100, so 100 is
the *validated* operating point), and expose it as the one CTC slider. Scoring
constants (γ/λ/β/α/γp/βp) are **not** exposed: the published-preset control shows a
mis-set preset costs ~2.3 pt t1, and there is no in-app feedback loop that would let
a user tune them productively.

Deliberately NOT added (see §6): `ctc_top_k` (constant 8 in the adapter — topK only
truncates the final sorted slate, no accuracy interplay), an engine-variant
fast/accurate selector (would double the asset payload — O2), scoring-constant
sliders (above).

### Search-index / drift-test impact — exact list

- **Generated index (`generateSettingsSearchIndex` gradle task):** regenerates
  automatically; the Prediction Engine dropdown keeps its title, so its entry is
  unchanged (options lists are not indexed). No manual step.
- **`SettingsSearchCoverageTest` — NO changes required**, verified against each of
  its five tests: (1) no new `SettingsSwitch/Slider/Dropdown` literal is added to
  scanned files (the new sliders are file-private `ParameterSlider`s in
  `CtcSettingsActivity`, which is not scanned — same as `NeuralSettingsActivity`);
  (2)/(3) no new generated entries/keywords; (4) the new `gatedBy =
  "swipe_typing"` on the CTC Settings hand entry already has its
  `isGateEnabled` branch; (5) no new top-level section.
- **`SettingsDefaultsDriftTest`:** satisfied by the `SETTINGS_DEFAULTS` hunk in
  §1c-vii (`ctc_beam_width` is read via `safeGetInt` in `Config.refresh` and
  written via `putInt` in `CtcSettingsActivity` — both regex-matched by the drift
  scanner, both now classified).
- **Hand-maintained search list:** one new `SearchableSetting("CTC Settings", …)`
  (§1c-vii).
- **Docs (convention, not test-enforced):** add the `ctc_beam_width` row and the
  `swipe_engine_mode` value list to `docs/SETTINGS_MAPPING.md` and
  `docs/wiki/specs/settings/neural-settings-spec.md`; flip
  `docs/specs/ctc-swipe-engine.md` status from "DESIGN ONLY, NOT WIRED" and check
  off FR-5/FR-6; update `memory/todo.md`.

---

## 3. Test plan

### Existing tests that MUST change

| Test | Change |
|---|---|
| `src/test/kotlin/tribixbite/cleverkeys/swipe/SwipeEngineRouterTest.kt` | Add the `Mode.CTC` routing rows: QWERTY→`Engine.CTC`; Dvorak/AZERTY (latin non-QWERTY)→`Engine.GEOMETRIC`; cyrillic/greek→`Engine.GEOMETRIC`; unknown-metadata→`Engine.GEOMETRIC` (never NEURAL, mirroring the conservative section); plus `Mode.fromPref("ctc") == Mode.CTC` and `fromPref("CTC")` case-insensitivity if a fromPref test exists. |
| `src/test/kotlin/tribixbite/cleverkeys/SuggestionProvenanceTest.kt` | In `swipe engine mode maps to the right origin`: add `assertEquals(SuggestionOrigin.CTC, SuggestionOrigin.forSwipeEngineMode("ctc"))`. `every origin has a distinct human label` passes automatically once the new label exists (it iterates `values()`). |
| `src/test/kotlin/tribixbite/cleverkeys/swipe/ctc/CtcModuleTest.kt` | Extend `scoringParams_presets_matchScoringJson` with the `tunedV2` block: γ 1.05, λ 1.1, β 0.2, α 0.0, γp 0.3734, βp 0.9882, beamWidth 100, topK 4. |
| `CtcParityTest` | **No code change** — starts passing once the fixture lands (§1e). It is already registered in `runPureTests` (drift-checked by `TestRunnerListDriftTest`, which therefore also needs no change). |
| `ConfigDefaultsTest` / backup exporter-import tests | Run them; they enumerate `SETTINGS_DEFAULTS`-backed keys generically, so the new entry should be picked up without edits. If `ConfigDefaultsTest` hardcodes a key list, add `ctc_beam_width`. |

### New tests

1. **`CtcEmissionModelParityTest`** (instrumented, §1e — full source above): tensor
   wiring + sliced-emission parity (tol 2e-3 vs desktop CPU EP) + bit-identical
   on-device featurizer + exact greedy/top-k words + scores within 1e-3.
2. **`CtcLatencyGateTest`** (instrumented — the G3 gate). Shape: load the model via
   `ModelLoader` with production settings (hardware accel ON), build the trie from
   the real `dictionaries/en_enhanced.json` asset via the same merge code path
   (factor the trie build into an `internal` function of `CtcEngineAdapter` or
   replicate), decode the fixture's `model_keyboard` points 30× at
   `tunedV2(beamWidth=100, topK=8)` after 5 warmups, assert
   `median < 150 ms && p90 < 250 ms`. Rationale: G3's bar is "≤ our current neural
   ~100–300 ms"; expected actuals are ~1 ms encoder + a beam whose cost on an
   emulator core should sit in the tens of ms against a 98k-word trie, so the gate
   has wide margin yet still catches a pathological regression (e.g. accidental
   beam-300 default or trie rebuild per swipe). Also assert a SECOND decode of the
   same trace reuses memos (elapsed < first-decode elapsed including trie build) to
   pin the warm path.
3. **Router coverage** is pure-JVM (item above) — no instrumented router test
   needed (`GeometricSwipeOracleTest` precedent stays geometric-only).

### Invocations

```bash
# pure JVM (Termux-safe)
./gradlew runPureTests -PtestClass=swipe.ctc.CtcParityTest
./gradlew runPureTests -PtestClass=swipe.ctc.CtcModuleTest
./gradlew runPureTests -PtestClass=swipe.SwipeEngineRouterTest
./gradlew test --tests "tribixbite.cleverkeys.SuggestionProvenanceTest" \
               --tests "tribixbite.cleverkeys.backup.SettingsDefaultsDriftTest" \
               --tests "tribixbite.cleverkeys.SettingsSearchCoverageTest"

# instrumented (ew-cli; per .claude/skills/ew-cli-testing.md — EW_VERSION pin required)
./gradlew assembleDebug assembleDebugAndroidTest
mkdir -p ~/ew-output
EW_VERSION=1.3.4 ew-cli \
  --app build/outputs/apk/debug/CleverKeys-*-x86_64.apk \
  --test build/outputs/apk/androidTest/debug/CleverKeys-debug-androidTest.apk \
  --outputs-dir ~/ew-output --timeout 40m \
  --device model=Pixel7,version=34 --use-orchestrator \
  --test-targets "class tribixbite.cleverkeys.swipe.CtcEmissionModelParityTest,class tribixbite.cleverkeys.swipe.CtcLatencyGateTest"
# then the FULL suite once, before any release tag (regression sweep):
EW_VERSION=1.3.4 ew-cli --app ... --test ... --outputs-dir ~/ew-output \
  --timeout 40m --device model=Pixel7,version=34 --use-orchestrator
```

Note the latency gate is measured on an x86_64 cloud emulator — a proxy, not a
phone little core. RESULTS.md "Next" item 5 (re-measure on a phone) stays open;
the user-facing check is the manual QA pass in §4.

---

## 4. Rollout

1. **Default engine stays `neural`** (`Defaults.SWIPE_ENGINE_MODE` untouched).
   `ctc` is reachable only via Settings → Swipe Typing → Prediction Engine → CTC.
   This is the decision doc's "ship dark → power-user visible" posture; flipping the
   default is a separate future decision gated on beta feedback (O6).
2. **Migration: none required.**
   - New pref *value* `"ctc"`: on any build that predates the router change,
     `Mode.fromPref` falls back to NEURAL by design ("never crash the router on a
     corrupted pref") — downgrades and stale backups are safe.
   - New pref *key* `ctc_beam_width`: carries a compile-time default in
     `SETTINGS_DEFAULTS`; backup export/import handles it generically; import into
     an older build ignores the unknown key.
   - No DB, no schema, no `SettingsMigration` step.
3. **Version:** ship as **v1.6.0** (versionCode 10600; per-ABI codes 106001/2/3
   under the existing `versionCode * 10 + abiCode` scheme) — a minor bump for a new
   engine + ~2.8 MB asset. Follow `.claude/skills/release-process.md` for the
   fastlane changelog files (`fastlane/metadata/android/en-US/changelogs/10600{abi}.txt`)
   — and NO tag/push without explicit user permission (repo rule).
4. **R8/proguard: no new rules needed** (verified against `proguard-rules.pro`):
   `-keep class ai.onnxruntime.** { *; }` (line 193) already covers the ORT Java/JNI
   surface the new code calls; `OnnxCtcEmissionModel`/`CtcEngineAdapter`/`swipe.ctc.*`
   are referenced directly (no reflection, no JNI of their own), so default R8
   retention applies. The existing `tribixbite.cleverkeys.onnx.**` keeps cover
   `ModelLoader`. Optional hardening if a release-build smoke test ever hits
   stripping: `-keep class tribixbite.cleverkeys.swipe.ctc.** { *; }` — not included
   by default per minimal-rules practice.
5. **Release-build QA (manual, per repo testing policy — ADB install-only):**
   install the release APK, switch engine to CTC, verify (a) first swipe after IME
   open decodes without visible jank (warmup worked), (b) long-word accuracy feels
   ≥ neural ("keyboard", "particular"), (c) short words are acceptable (the known
   ≤3-char stratum trade), (d) non-QWERTY layout under CTC mode still swipes
   (geometric hedge), (e) suggestion long-press provenance shows "CTC swipe (trie
   beam)", (f) battery/thermals unremarkable across a typing session.
6. **Docs to update in the same PR:** `docs/specs/ctc-swipe-engine.md` (status:
   wired behind opt-in; FR-5/FR-6 → DONE), `memory/todo.md`, `README.md` feature
   list (one line), `docs/SETTINGS_MAPPING.md`, wiki settings spec, and a
   `docs/audit/` note that G3+G5 executed per this plan with the G4-refinement and
   cascade phases still open.

---

## 5. Sequencing (suggested commit series in the app repo)

1. `test(ctc): land golden fixture + model asset` — §1d + §1e copies (fixture,
   androidTest fixture, onnx asset). `CtcParityTest` goes green here.
2. `feat(ctc): onnx emission model + tunedV2 preset` — §1a + §1b +
   `CtcModuleTest` preset assertions + `CtcEmissionModelParityTest`.
3. `feat(ctc): ctc engine adapter + router mode` — §1c-i…v + router/provenance
   test updates.
4. `feat(ctc): settings surface for ctc mode` — §1c-vi/vii (+ search entry).
5. `test(ctc): latency gate` — G3 instrumented gate + ew-cli run evidence.
6. `docs(ctc): spec status, notice, settings mapping` — §1f + §4.6.

Each commit compiles and tests green on its own; the engine only becomes
user-reachable at commit 4.

---

## 6. Open decisions for the user (each with a recommendation)

- **O1 — Which model ships (D1). Superseded 2026-08-11 → see §7.2** for the
  post-Phase-G/J menu (`resbn80g` test-validated at 1.14 MB; `sw2345` best
  measured accuracy but **not** test-validated). The 2026-08-08 text is kept
  below unedited as the record of the state it was written in.
- **O1 (2026-08-08 text) — Which model ships (D1).** Recommend `ch128_s1234`
  (2.8 MB) for maximum accuracy — 87.29 → 87.92 test t1 over `fast_resbn80` at the
  AOSP trie. But the evidence-tier argument for it is **gone**: `fast_resbn80`
  (1.1 MB, 0.215 ms) is now *also* test-validated, clearing all five test bars on
  every seed at both the AOSP and the shipped app lexicon (`PHASE_F.md` §16.5),
  so the choice is a straight −0.63 t1 for −1.7 MB and −0.24 ms. Watch its config-B
  top-5: worst-seed margin +0.08 pt. If `fast_resbn80` is chosen, regenerate and
  re-land the golden fixture from that artifact (§1d) and update RESULTS/NOTICE
  wording accordingly. `fast_resbn72` (0.186 ms) remains **val-only**.
- **O2 — Fast/accurate variant selector.** Recommend **no**: shipping both models
  costs +1.1–3.9 MB and doubles the parity-test matrix for a latency difference
  (0.26 ms) that is invisible next to the beam + pipeline cost. Revisit only if a
  low-end-device complaint materializes.
- **O3 — Lexicon (D4). RESOLVED 2026-08-08 — recommendation confirmed, no preset
  change.** The validation this bullet asked for was run: `PHASE_F.md` §15.
  `eval_beam.py --vocab-kind json-strip` (the app's own
  `loadStrippingNonAlphabet` policy; `--vocab-kind json` is bit-identical here)
  over full val-9918 at the E1 preset, both ship candidates, three seeds each,
  against a bar **re-measured on the same 98,081-word trie** from FUTO's real
  weights. **All five bars clear on the seed mean and on every seed at
  λ = 1.1 unchanged.** The frequency-scale risk is real — `log_freq` spread
  collapses 5.40 → 0.64 — but it is offset by coverage: the app trie has *fewer*
  OOV targets (2.52 % of val vs 3.39 %) because the 63,851 words it drops are rare
  forms nobody typed, and only 12 val rows are in AOSP and not in it. `resbn:80`
  was additionally test-validated at this lexicon (§16.5). A λ-only re-sweep puts
  the optimum at 2.0–2.5 (+0.6 to +1.1 t1 on untouched val rows) and is documented
  in §15.4, but is **not** recommended: it would diverge the shipped preset from
  the golden fixture and from every published number for a gain no gate needs.
- **O4 — Non-QWERTY under ctc mode (D6).** Recommend the geometric hedge as
  diffed (never remove swipe coverage a mode switch away). Alternative
  (`Engine.NONE`) is simpler conceptually but strictly worse UX.
- **O5 — Non-English dictionary language under ctc on QWERTY.** v1 adapter returns
  an empty slate (bar clears; user sees no swipe output). Alternative: route those
  swipes to neural at the InputCoordinator level. Recommend the empty slate for v1
  (mode is opt-in, en-first) + a follow-up once the multi-language story (langpack
  tries + per-language validation) exists. **Updated 2026-08-11:** part of that
  story now exists — a Cyrillic model with a measured number and a per-language λ
  that must travel with it (§7.1), and a measured verdict against folding en+ru
  into one model (§7.4). The en-only gate stays for v1; the per-language preset
  axis is the piece to build first, because every non-en script the app can serve
  reads its frequencies off the CKDT `255 − rank` scale, not the AOSP scale E1 was
  fitted on.
- **O6 — When does ctc become default / replace hybrid's internals?** Not in this
  change. Per the decision doc: only after a beta cycle, and ideally after the G4
  refinement-head question is settled. Revisit with field feedback.
- **O7 — Two-phase preview decode (spec integration step 5).** Deferred: the
  geometric engine also decodes only at gesture-end today, and the preview path
  (beam 32/top 1 during gesture) needs its own UI plumbing. The adapter's memo
  structure already supports it (a second `decoderFor(..., beamWidth=32)` call);
  propose as a fast-follow.
- **O8 — `SettingsResetPresets` behavior for `ctc_beam_width`.** Recommendation in
  §1c-vii: follow the `geo_*` precedent (not reset). Flag because the current
  non-reset of `swipe_engine_mode`/`geo_*` looks like an accident of history rather
  than a decision.
- **O9 — Adopt the Phase-J finalist `sw2345` (new, 2026-08-11)?** Recommendation:
  **decide it explicitly, do not drift into it.** It is the best-measured model in
  the campaign (all six alt-layout bars, four of five val bars, 2.91 MiB, 0.842 ms)
  and a pure file swap, but it is **val + alt-layout validated only** and would
  displace a **test-validated** incumbent (`resbn80g`). Adoption also pulls two
  chores: regenerating the golden fixture at the ship preset, and answering the
  app-trie λ question. All of it is laid out in §7.2.
- **O10 — Per-language decode preset (new, 2026-08-11).** Recommendation:
  **build the axis, independent of O9.** λ = 2.0 on the ru CKDT lexicon is worth
  ≈ +1.2 in-dict t1 to whichever Cyrillic model ships, needs no retrain, and is
  blocked only by the fact that `CtcScoringParams` has no language axis today
  (§7.1). It is also the prerequisite for ever relaxing O5's en-only gate.

---

## 7. Phase J update (2026-08-11) — new candidate, and one free win that needs no model change

Full record: `PHASE_J.md` (verdict §9, artifacts §10); cross-repo summary in
`MODEL_COMPARISON.md` §2.8 and §5.

**The headline an integrator must not miss: Phase J's terminal condition was NOT
met.** Ten of eleven bars fell; the `≤3` val stratum missed by 0.07 and the
Cyrillic bar was not beaten, so **test-2400 was not unsealed and nothing in Phase
J is test-validated** — `resbn80g` keeps that tier. What follows is therefore one
free decode constant that needs no decision (§7.1), one model swap that needs an
owner decision made with its evidence tier in full view (§7.2), one unshipped
prerequisite that interacts with both (§7.3), and one closed question (§7.4).

### 7.1 The free win: the Cyrillic decode λ is mistuned — worth ≈ +1.2 t1, no model change

**This requires no new model.** Every Cyrillic number this campaign ever
published, including the 76.21 bar, was decoded at the English benchmark
preset's `lambda = 1.1`. But the app's **langpack-ru CKDT v2** lexicon stores
`freq = 255 − rank` — the compressed CKDT scale that `PHASE_I.md` §7.4 already
showed wants a *larger* λ. Nobody had ever swept λ per language
(`PHASE_I_DATA.md` disclosed the gap).

Lexicon provenance, verified in the app repo: `scripts/dictionaries/langpack-ru.zip`
(manifest `{"code":"ru","version":2,"wordCount":50000}`), imported by
`LanguagePackManager` into `files/langpacks/ru/dictionary.bin` and read by
`CkdtDictionaryReader` (magic `CKDT`, version 2, per word `uint8` frequency rank,
0 = most frequent). `eval_cyrillic.py` builds its trie from **that exact zip** with
`freq = max(1, 255 − rank)`, so the eval scale and the app scale match by
construction — this is the app's own ru lexicon, not a proxy. Note it is an
*importable* pack, not an APK asset (nothing under `src/main/assets/dictionaries/`
is ru).

Swept symmetrically over both ru models, tuned on ru val rows 0:4708 and
confirmed on the untouched 4708:9416 (`PHASE_J.md` §6.9):

| λ | shippable synth-only ru model (`phaseIB-ru-synth`), tune / confirm | joint en+ru challenger, tune / confirm |
|---|---|---|
| 1.1 (the published footing) | 75.73 / 76.70 | 76.77 / 76.34 |
| **2.0** | **76.91 / 77.92** | **77.83 / 78.23** |
| 3.0 | 75.82 / — | 76.39 / — |
| 4.0 | 73.88 / — | 74.50 / — |

**+1.2 in-dict t1 on both halves. The correct expected Cyrillic accuracy is
≈ 77.4, not the 76.21 currently documented.** The lever is **model-independent**
— it lifted the joint challenger by the same order (+1.1 tune / +1.9 confirm),
which is why it changes no verdict in `PHASE_J.md` §6.9 but does change the
number the app should expect. It is a pure decode constant: no retrain, no new
artifact, no change to the Cyrillic model itself.

Scope of "today", stated exactly: the Cyrillic *model* needs no change, but the
app cannot exercise any of this yet — `src/main/assets/models/` carries only the
legacy transformer (`swipe_encoder_android.onnx` / `swipe_decoder_android.onnx`),
there is no CTC artifact of any script in the app, and this repo's
`ctc/artifacts/` has no ru ONNX either. So λ = 2.0 is a constant that must land
*with* the Cyrillic path whenever it is wired — free, but not yet collectable.

**Where it has to live — this is a real code change, not a config edit.**
Verified against app HEAD `62c9419f` (clean tree; nothing in the app repo was
modified):

* **The scoring preset is GLOBAL, not per-language.** `CtcScoringParams`
  (`src/main/kotlin/tribixbite/cleverkeys/swipe/ctc/CtcScoringParams.kt`) is a
  data class with three companion factories — `encoderOnly()`, `encoderDecoder()`,
  `fallback()` — each a fixed constant set ported from FUTO's `scoring.json` and
  keyed, per its own KDoc, by the **active model-combination signature**. There is
  no language parameter anywhere in the file, and no caller passes one.
* **Nothing in `src/main/` constructs one today** — the only references are
  `CtcModuleTest` and `CtcOnnxLatencyBenchmarkTest`. The preset is global by
  omission (the engine is unwired), not by a considered decision, so adding the
  axis breaks no existing contract.
* **This plan keeps it global.** §1(b) adds one more constant factory,
  `tunedV2(beamWidth, topK)`, and §1(c)-iii constructs it in exactly one place —
  `CtcEngineAdapter.decoderFor(mapped, trie, beamWidth)`, whose memo key is
  `Triple(mapped, trie, beamWidth)`, with **no language component**. The adapter
  does receive the active language (`decodeAsync(..., language: String)`, sourced
  in `InputCoordinator` from
  `predictionCoordinator.getDictionaryManager()?.getCurrentLanguage()` falling back
  to `config.primary_language`) but uses it only as a gate against the constant
  `LANGUAGE = "en"`, returning an empty slate otherwise.
* **Minimal shape of the change**, therefore: (1) select the preset by language
  next to the signature — e.g. `CtcScoringParams.tunedV2(language, beamWidth, topK)`
  or a `presetFor(language)` table — returning λ 2.0 for CKDT-scale scripts and
  λ 1.1 for the AOSP-scale en trie; (2) add `language` to the `decoderFor` memo key
  so a language switch rebuilds the decoder instead of silently reusing the
  previous λ; (3) relax the `LANGUAGE = "en"` gate for the scripts that have a
  model *and* a validated preset (O5). Nothing else in the preset moves.
* **No precedent to copy for a per-language *scoring* constant.** The app's
  per-language state is done with key suffixes
  (`LanguagePreferenceKeys.customWordsKey("ru") == "custom_words_ru"`), but the
  nearest analogue to λ on the geometric side — `Config.geo_frequency_weight`
  (`Defaults.GEO_FREQUENCY_WEIGHT`, read once in `GeometricEngineAdapter` into
  `GeometricEngineConfig.frequencyWeight`) — is a single global float used for
  every language. The λ axis would be the first language-keyed scorer constant in
  the app.
* **Do not generalise this to the geometric engine.** The sweep measured the CTC
  beam's λ against CTC emissions only. `geo_frequency_weight` is a different
  scorer with a different range (0.0–0.4, default 0.12) and no equivalent sweep
  exists; nothing here licenses touching it.

**Caution, stated because it interacts:** no evaluation in this campaign
included a **user dictionary**, and λ multiplies the frequency term, so a larger
λ amplifies top-of-scale injected competitors (`PHASE_G.md` §6). The
user-dictionary v1 fix (see §7.3) is unshipped; if it lands, λ = 2.0 should be
re-confirmed with user-dictionary entries present rather than assumed to carry.

### 7.2 The new model candidate: `sw2345` — better, and NOT test-validated

**The menu, with evidence tiers.** `MODEL_COMPARISON.md` §5 is the authoritative
matrix; this is the app-facing subset, and it replaces D1/§1(d)/O1. Read the tier
column before the accuracy column — they no longer point the same way.

| option | ship bytes / params / encoder | evidence tier | preset + fixture that must travel with it |
|---|---|---|---|
| **`sw2345`** (Phase-J finalist, NEW) | 3,052,318 fp16w (2.91 MiB) / 1,512,802 / 0.842 ms mean, 0.859 p90 | **val + alt-layout validated ONLY — NOT test-validated.** 3 seeds, val-9918 + 7 alt-layout corpora, no sealed split at all | **E1** (`1.05 / 1.1 / 0.2 / 0.3734 / 0.9882`); fixture regenerated from it at the ship preset — **does not exist as a replacement yet** (below) |
| `resbn80g` (Phase-G recommendation) | 1,142,727 fp32 (1.09 MiB) / 279,346 / 0.215 ms class | **Test-validated on both footings, every seed** — the only model that still holds that tier | `0.9 / 4.0 / 0.25 / 0.25 / 0.9882` **and** fixture `ctc_model_golden.json` sha256 `ce3b5456ad13543ac09ac8c2610374bd8847b15f740f9004a98efea59d74f134`, which is what `artifacts/` carries today |
| `ch128_s1234` (the original D1 pick) | 2,799,865 fp32 / 689,282 / 0.455 ms | Test-validated, all five bars every seed, both tries | E1; its fixture (sha `a18ea58c…`) is **no longer** the one in `artifacts/` |

The accuracy ranking and the evidence ranking are inverted, deliberately: the seal
was not spent because Phase J's terminal condition was not met. **Choosing
`sw2345` means choosing the best-measured model and the weaker evidence tier at
the same time.**

| | value |
|---|---|
| ship artifact | `artifacts/sw2345_s1234_fp16w.onnx`, **3,052,318 B (2.91 MiB)** |
| sha256 | `2e820c121fc69ae95a9b2e22444fe14c47f5c5253df4696a0d0a432e364fc7b8` |
| fp32 reference | `artifacts/sw2345_s1234.onnx`, 6,068,519 B, sha256 `96dd27ece698fa981530639700e66e0689acd2d3f024ad214e8a79b3fa083a30` |
| params / latency | 1,512,802 / 0.842 ms mean, 0.859 p90 (encoder only) |
| architecture | **identical** to the `resbn192i` candidate — `resbn:192:1,2,4,8`, embed_hid 96, T′ = 32, the same frozen `[1,32,·]` I/O contract |
| decode preset | **E1 unchanged** (`gamma 1.05, lambda 1.1, beta 0.2, gammaPrune 0.3734, betaPrune 0.9882`) |
| 3-seed val-9918 | 88.51 / 92.67 / 93.37 / **91.20** / 87.11 |
| alt-layouts (3-seed) | dvorak 89.87, dvorak-app 88.98, azerty 83.81, qwertz 83.01, german 80.64, spanish 88.45 — **all six bars beaten** |

**Integration is a file swap: no code change.** The whole Phase-J gain is
training data (two new FUTO pools, 126,549 extra rows); the graph, the contract
and the preset are unchanged. Re-verified on the artifact itself: inputs
`features [1,2,64] f32`, `layout_keys [1,64,2] f32`, `layout_mask [1,64] bool`;
outputs `log_emissions [1,32,65] f32` (+ the unfetched `coefficients`/`lambda`),
opset 17, fully static — i.e. exactly the contract §1(a)'s `OnnxCtcEmissionModel`
is written against. §1(d)'s copy command applies verbatim with the source path and
expected sha changed. Note the ship bytes are **fp16w**, so the APK delta is
≈ +2.9 MB per ABI rather than D1's +2.6–2.8 MB.

**fp16w is free on accuracy here, measured rather than inherited**
(`PHASE_J.md` §10): full val-9918 (E1/AOSP) decoded *through the fp16w graph*
gives 88.51 / 92.58 / 93.35 / 90.91 / 87.26 against the fp32 artifact's
88.51 / 92.59 / 93.35 / 90.91 / 87.26 — Δ ≤ 0.01 on every metric, seed 1234. Two
asterisks travel with it: fp16w is **3 % slower** on this instrument (0.842 vs
0.816 ms fp32, not the "identical" Phase I published), and its weight-rounding
residue is 2.30e-02 on the sliced head (argmax 100/100) — real in the emissions,
invisible after the lexicon beam.

**The preset needs no per-model retuning on the benchmark footing, and this is
now well evidenced.** Phase J swept the E1 region symmetrically over both the
finalist and the incumbent with a stratum-aware `minmargin` objective (tuned on
val`[0:4959]`, confirmed on val`[4959:9918]`), and **both models landed back on
their own E1 numbers to within ±0.07 on every metric** — the fifth model family
for which E1 transfers unchanged (`PHASE_J.md` §6.8b). So D2's `tunedV2` constants
carry over to `sw2345` as written.

**Open, and material to shipping: the app-trie λ has never been swept for this
model.** Everything above is the AOSP 146,964-word footing. The app ships
`en_enhanced.json` (98,140 entries → the 98,081-word STRIP trie of O3, on the
compressed 134–255 scale), and on that trie the
precedent runs the other way — O3/`PHASE_F.md` §15.4 measured the λ optimum at
2.0–2.5 for the campaign-2 models, and `resbn80g` ships at λ 4.0
(`MODEL_COMPARISON.md` §5). `MODEL_COMPARISON.md` §5's own entry for `sw2345` says
"no app-trie sweep has been run for it". Two honest options: ship E1 unchanged
(consistent with every published `sw2345` number and with the fixture rule), or
sweep λ on the app trie first and move the fixture with it. Do **not** mix — the
fixture records the preset it was generated at (`MODEL_COMPARISON.md` §5.1, and
the fixture paragraph below).

**Evidence tier — read this before adopting.** `sw2345` is **val + alt-layout
validated only. It is NOT test-validated.** Phase J's terminal condition was
*not* met: the `≤3` val stratum missed its bar by **0.07** (91.20 seed-mean
against 91.27 — roughly two rows of 3,389, and closer than the two-seed estimate,
but a miss and not rounded away), so under the campaign's own pre-registered rule
**test-2400 was not unsealed** — no pre-registration was filed and no seal-ledger
entry appended, and nothing in Phase J may be quoted against any test bar.
**`resbn80g` remains the only test-validated model.** Adopting `sw2345` therefore
means promoting a val-only model over a test-validated incumbent; that is the
owner's call, not a default. (The Cyrillic bar is the second axis Phase J did not
beat — §7.1, §7.4 — but it is orthogonal to this en-side swap.)

**Golden fixture — REQUIRED before adoption, and not landed.** The
fixture-and-preset rule (`MODEL_COMPARISON.md` §5.1) is that the shipped model,
the runtime preset and the fixture move together: the fixture records its own
`source_onnx_sha256` and `preset`, and `CtcParityTest` asserts Kotlin reproduces
Python bit-for-bit *at that preset*. So adopting `sw2345` requires regenerating
the fixture from `sw2345_s1234.onnx` at whatever preset ships and re-landing both
app-repo copies (§1(e)).

State today, verified in `artifacts/`:

| file | source model / preset | bytes | sha256 |
|---|---|---|---|
| `ctc_model_golden.json` (the real fixture) | `resbn80g_s1234` at `0.9 / 4.0 / 0.25 / 0.25 / 0.9882` | 140,204 | `ce3b5456ad13543ac09ac8c2610374bd8847b15f740f9004a98efea59d74f134` |
| `sw2345_s1234_golden_CANDIDATE.json` (candidate only) | `sw2345_s1234` at E1 | 140,098 | `b397715091b0ccb26be802842a6b3048efbeba7fbc3fd19572face62f12b47b7` |

The candidate carries the same 10 cases (6 `featurize` + 4 `beam`) and the same
`layout` block the app's two parity tests consume, so it is drop-in *if* adoption
happens — but it has deliberately **not** replaced `ctc_model_golden.json`,
because adoption is undecided. Nothing should be copied into the app repo from
the `_CANDIDATE` file until that decision is made; and if the app-trie λ question
above resolves to anything other than E1, the candidate is stale and must be
regenerated at the chosen preset.

### 7.3 User-dictionary v1 fix — still unshipped, and it interacts with λ

**Where the design lives (the pointer this plan should have carried):**
`RESEARCH_SCAN.md` §2.5 item 1 — *"wire the personal lexicon into the CTC path"*:
merge the system user dictionary + custom words + `UserVocabulary` into
`CtcLexiconTrie` (clamped frequency, modelled on
`GeometricEngineAdapter.mergeUserWords`) and add a **capped** personalization term
via the reserved `alpha` slot, validation-gated per user. Supporting sections:
§2.2 (the survey finding — *"there is no user-dictionary merge in the CTC trie
yet"*), §2.3 (the value bound, +0.5–2 t1 for an active user, and the λ
amplification caveat), and the table row **(c1)**, ranked 1 for value per
engineering unit.

**Status: not implemented.** The app has no CTC engine wired at all, so it has no
CTC user-dictionary merge either. This plan's own §1(c)-iii/D4 covers *part* of
v1 on paper — custom words merged at clamped 1..255, disabled words dropped,
memo invalidated by a content hash — but it is unshipped like the rest of the
plan, and it does not cover the system user dictionary, `UserVocabulary` boosts,
or the `alpha` rerank term.

**Why it belongs in this section — the caution.** **No evaluation anywhere in
this campaign included a user dictionary** (`PHASE_F.md` §15.5, `PHASE_G.md` §6,
`MODEL_COMPARISON.md` §5.2), so every number in this document is a
no-user-dictionary number. User entries are injected at the **top of the
frequency scale** (freq 255 after clamping; the geometric precedent inserts custom
words at 1000 before clamping), and λ multiplies exactly that term — so a larger
λ amplifies them. This is a live risk for two presets already on the table:
`resbn80g`'s app preset at λ 4.0, and §7.1's ru λ 2.0 on a CKDT `255 − rank`
lexicon where a user word outranks every real word in the pack. Whichever lands
first, the personal-lexicon merge must ship with a boost cap and a validation
gate rather than a bare insert, and λ should be re-confirmed with user entries
present rather than assumed to carry.

### 7.4 Multi-script status — separate per-script models remain the plan

A joint en+ru single model was built and measured (`PHASE_J.md` §6.8, §6.9): the
base recipe plus 1,000,000 synthetic ru rows on the `ru_jcuken` geometry, one
65-wide per-key-slot head serving both scripts (no Yandex training rows anywhere;
Yandex val is eval-only). **Two scripts in one head demonstrably works** — and
the measured price is why it is not adopted:

| axis | joint | reference | Δ |
|---|---|---|---|
| ru in-dict t1, λ 1.1 (app-ru 50 k, E1, n = 8,471) | 76.56 | 76.21 bar | +0.35, inside one binomial SE (±0.46) |
| ru in-dict t1, λ 2.0, confirm half | 78.23 | 77.92 | +0.31, inside one SE (±0.64 at n = 4,240) |
| ru in-dict t3 / t5, λ 1.1 | 88.16 / 91.12 | 88.53 / 91.42 | **−0.37 / −0.30** |
| ru in-dict t3 / t5, λ 2.0 confirm half | 88.94 / 91.49 | 89.50 / 92.00 | **−0.56 / −0.51** |
| ru greedy | 23.68 | 37.07 | **−13.39** |
| en val t1 | 87.90 | 88.32 | **−0.42** against a stated 0.3 tolerance |

**Verdict: feasible, not adopted — separate per-script models remain the plan.**
The ru side is a tie at best (ahead on t1 by less than one SE on both sweep
halves, *behind* on t3/t5), the en side breaks its tolerance, and the greedy
collapse shows what the shared head costs: per-slot emissions get much blurrier
and only the lexicon beam hides it.

Consequence for this plan: O5's en-only gate is unchanged, and the multi-language
route is a per-script model plus a per-language preset (§7.1), not one model for
everything.

## 8. Phase K update (2026-08-12) — the ensemble configuration, contract-v2, and the rescorer

Full measurement record: `PHASE_K.md`. Three integration-relevant outcomes.

### 8.1 The `mix2-i8f16` configuration (all 11 en bars, val+alt-layout footing)

`phaseK_sw2345_s1234_int8w.onnx` (1,554,355 B) + `phaseK_resbn192i_s1234_fp16w.onnx`
(3,052,318 B), total **4.45 MB**, encoder cost 0.930 + 0.858 ms sequential.
App-side requirements, all inside `swipe/ctc/`:

* `CtcEmissionModel` gains a **dual-session mode**: run both graphs on the same
  `features`/`layout_keys`/`layout_mask` feed, then average **probabilities**
  per frame over the 65 columns: `avg = logaddexp(a, b) − log 2` on the raw
  log-emission heads, BEFORE `CtcEmissions.sliceFromHead`. (Do NOT average
  log-probs without renormalizing — `len^γ` breaks the invariance; do NOT use
  geometric mean at all: refuted, `PHASE_K.md` §4.1.)
* Everything downstream (slice, beam, presets, tries) is unchanged — contract
  v1 `[1,32,65]` holds for both members.
* Fixture: `artifacts/phaseK_mix2i8f16_golden.json` (E1; the `emissions`
  arrays are the AVERAGED head — the dual-session parity target;
  `source_onnx_sha256` lists both members).
* Pair validity is NOT generic: member pairs must pass the per-frame-agreement
  gate (≥95 % argmax agreement on unlabeled traces, `PHASE_K.md` §4.3). The
  shipped pair is fixed, so this is a build-time check, not a runtime one.
* Evidence tier: val + alt-layout only, NOT test-validated; deterministic-
  configuration footing with disclosures (`PHASE_K.md` §8.2).

### 8.2 Contract-v2 (`[1,64,65]`, T′ = 64) — documented, NOT promoted

`phaseK_t64_s1234_contractv2.onnx` + `phaseK_t64_golden_contractv2.json`
(`frames: 64`). If ever adopted: `CtcEmissions.sliceFromHead` frame loop and
any `[·,32,·]` assumption must read `frames` from the fixture/model; the
refine-head `[T′,92]` input breaks (the refine head is dead code — Phase E
kept it out); decode cost ≈ **2.1×** measured (29.0 vs 60.7 tr/s same box,
same beam) — on-device that scales the 1.5–7 ms beam to ~3–14 ms, inside the
50 ms bar but past the 10 ms preference at the high end. Val: −0.19 t1/−0.39
4+ against its T′=32 twin (misses two val bars) while clearing all six layout
bars — a transfer-biased option, on the shelf.

### 8.3 The rescorer (optional, small, symmetric)

`phaseK_ranker_sw2345_2seed.onnx` (21,782 B): 14 features per top-k candidate
(`ranker_features.py` — forced-alignment replay of the beam's own Viterbi,
trie log-freq, slate stats), blended `final += 0.05 · ranker`. Seed-mean
+0.08 t1 / +0.11 4+, sign-consistent; NOT a ≤3 lever; **flat when stacked on
the ensemble** — so it is an option for a SINGLE-model ship only. Kotlin cost:
a ~200-line feature port + a 5 k-param MLP session; the reserved `alpha` slot
in `CtcScoringParams` is the natural blend hook. Recommendation: skip for v1
(the ensemble subsumes it); revisit only if the app ships a single model and
wants +0.1 t1 for 22 KB.

> **Records the 2026-08-08 state.** The `make_golden.py` changes below still hold
> (they are why the fixture carries `featurize` cases and a `layout` block at all),
> but the fixture *contents* described here are two regenerations out of date —
> current fixture state is the table in §7.2.

`make_golden.py`:
- new `FEATURIZE_CASES` (5 fixed branch probes: single-point, zero-duration,
  two-point-long, non-uniform timestamps, out-of-range clamp) + a 6th realistic
  `feat_word_path_cat` case built from the layout;
- fixture gains a top-level `layout` block (`letters`/`cx`/`cy` — the exact
  en_qwerty geometry the emissions were generated against), consumed by
  `CtcEmissionModelParityTest`;
- module docstring updated.

`artifacts/ctc_model_golden.json`: regenerated from `ch128_s1234.onnx` at the E1
preset — 10 cases, the 4 pre-existing beam cases verified **byte-identical** to the
previous fixture; 140,204 bytes; sha256
`a18ea58cd662b0e18b6daadaf417361f93fd0b146ce6478d4d6a62e7e185fa8a`.

`RESULTS.md`: artifacts-table fixture row updated (bytes + sha) + one regeneration
note. No accuracy number, seal statement, or claim wording was touched.

---

## 9. Phase L update (2026-08-13) — new single-model finalist, and a pair that is a recipe

Full record `PHASE_L.md`. **Nothing in the app repo changed**; this section
states what the app should carry when it next syncs models.

### 9.1 The model menu changes at the single-model slot

| slot | was (Phase J/K) | **now** | why |
|---|---|---|---|
| single-model finalist | `sw2345_s1234` (10/11 seed-mean, ≤3 −0.07) | **`phaseM_kd_fresh_w1_s1234_fp16w.onnx`, 2.91 MB** — 11/11 campaign bars on ALL 3 seeds and the seed-mean (PHASE_M §9); distilled from the coupled pair; **recommended ship model** (one session, no app code change) | ⚠ the Phase-L promotion of `phaseL_memberA_s1234_fp16w` is **RETRACTED** (PHASE_M.md §7.1): at five seeds it is 9/11, not 11/11 (t3 −0.024, qwertz −0.158). It still clears ≤3 on a five-seed mean (91.358) and must ship fp16w if used |
| two-model configuration | `mix2-i8f16` (4.45 MB, 11/11 single-config) | `v2pair-s1234` i8f16, **4.39 MB**, 11/11 campaign bars **every seed** (5/5) | reproducible by construction (6/6 gate passes) rather than a draw; pre-registered bar 1 vs the mix2 card was **not** met, so the recorded ship configuration is unchanged pending an orchestrator decision |

**Contract is unchanged**: `[1,32,65]` log-emission head, E1 preset, AOSP/az26
tries, same `CtcEmissionModel` seam, same dual-session averaging path §8
already describes. No app-side code change is required to adopt either — only
the asset swap and the fixture swap.

### 9.2 Packaging rule, corrected by measurement

**Ship the single model `fp16w`, not `int8w`.** §8 recorded int8w as val-free
*for the pair*; that does **not** generalize. Decoded, not inferred:

| packaging | size | ≤3 | dvorak | verdict |
|---|---|---|---|---|
| fp16w | 2.91 MB | **91.32** | 91.17 | free vs fp32 — **ship this** |
| int8w | 1.55 MB | **91.24 (below the 91.27 bar)** | 90.39 (−0.78) | costs the ≤3 bar |

For the *pair*, int8w + fp16w remains free and is the 4.39 MB packaging.

### 9.3 Fixtures to swap

| configuration | fixture | sha256 |
|---|---|---|
| single model fp16w | `artifacts/phaseL_memberA_fp16w_golden.json` | `7c3948c6…e7a184c2` |
| pair int8w+fp16w (averaged emissions) | `artifacts/phaseL_v2pair_i8f16_golden.json` | `7440873a…dc8dc749` |

Both are E1-preset, 10 cases, same schema `make_golden.py` has emitted since
Phase K — the averaged-emission fixture is the parity target for the
dual-session path, exactly as §8.4 specifies.

### 9.4 What did NOT change, and one thing to stop planning for

* Preset, beam, trie, λ-per-lexicon, rescorer (still the optional 21.8 KB
  add-on), contract-v2/T′=64 (still documented, still not promoted).
* **Stop planning around targeted English synthesis.** Phase L built it,
  gated it, and **refuted it** at three paired seeds (−0.21 t1). It is not a
  data source the app or any future collection should assume.

### 9.5 Phase M final (2026-08-14) — ship menu

| option | asset | size | footing |
|---|---|---|---|
| **recommended** | `artifacts/phaseM_kd_fresh_w1_s1234_fp16w.onnx` + `phaseM_kd_fresh_w1_fp16w_golden.json` | **2.91 MB** | 11/11 campaign bars, every seed (3/3) — single session, zero app code change |
| accuracy-first | `phaseL_v2pair_s1234_{a_int8w,b_fp16w}.onnx` + `phaseL_v2pair_i8f16_golden.json` | 4.39 MB | 11/11 campaign bars, every seed (5/5); needs the dual-session seam §8 describes |

The Phase-L single-model promotion (`phaseL_memberA_*`) is **retracted**
(PHASE_M §7.1) — do not ship it. test-2400 remains sealed; no number in §9 is
test-validated.
