# APP_INTEGRATION_PLAN — wiring the CTC swipe engine into CleverKeys (G3 + G5)

**Date:** 2026-08-08
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

Alternatives on the shelf (do NOT ship two models — see O2):
`fast_resbn80_s1234.onnx` (1,142,727 B, 0.215 ms, val-only evidence, wider t5
margin) and `fast_resbn72_s1234.onnx` (944,487 B, 0.186 ms, val-only). If one of
these is chosen instead, the golden fixture MUST be regenerated from that artifact
(`make_golden.py --onnx artifacts/fast_resbn80_s1234.onnx --preset
1.05,1.1,0.2,0.3734,0.9882`) — the fixture records `source_onnx_sha256` and the app
parity test is meaningless against a different graph. The stale pre-campaign
`artifacts/ctc_swipe_encoder.onnx` (r2, 394 k params) must NOT be shipped.

### 1(e) Golden fixture + parity wiring

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

- **O1 — Which model ships (D1).** Recommend `ch128_s1234` (2.8 MB): it is the only
  configuration with *test-validated* all-five-bars evidence. `fast_resbn80`
  (1.1 MB, 0.215 ms) is a defensible alternative if APK size matters more than the
  evidence tier — it clears all five **val** bars on every seed but "must never be
  quoted as test-validated" (PHASE_F). If chosen, regenerate + re-land the fixture
  from that artifact (§1d) and update RESULTS/NOTICE wording accordingly.
- **O2 — Fast/accurate variant selector.** Recommend **no**: shipping both models
  costs +1.1–3.9 MB and doubles the parity-test matrix for a latency difference
  (0.26 ms) that is invisible next to the beam + pipeline cost. Revisit only if a
  low-end-device complaint materializes.
- **O3 — Lexicon (D4).** Recommend `en_enhanced.json` + user words (zero new
  assets, user dictionary respected, spec-prescribed). Residual risk: the tuned
  preset was fitted against the 146,964-word AOSP-frequency STRIP trie; the app trie
  is 98k words with a compressed frequency floor (134–255). Cheap pre-ship
  validation in THIS repo (val is not sealed): run `futo_decoder_eval.py`
  `--vocab-json` with the app's `en_enhanced.json` over val-9918 at the E1 preset
  and confirm the five val bars still clear; re-sweep λ only if they don't. I
  recommend actually running this before commit 3 lands.
- **O4 — Non-QWERTY under ctc mode (D6).** Recommend the geometric hedge as
  diffed (never remove swipe coverage a mode switch away). Alternative
  (`Engine.NONE`) is simpler conceptually but strictly worse UX.
- **O5 — Non-English dictionary language under ctc on QWERTY.** v1 adapter returns
  an empty slate (bar clears; user sees no swipe output). Alternative: route those
  swipes to neural at the InputCoordinator level. Recommend the empty slate for v1
  (mode is opt-in, en-first) + a follow-up once the multi-language story (langpack
  tries + per-language validation) exists.
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

---

## Appendix A — this repo's supporting change (already committed alongside this plan)

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
