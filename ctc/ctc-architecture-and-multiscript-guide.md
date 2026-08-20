# CTC architecture and the multi-script question — the definitive guide

**Status**: reference. **Written**: 2026-08-18. **Revised**: 2026-08-20 (generator v2 / P6, and
the app at `d717bda7`).
**App state described**: `d717bda7` — `ctc` is the DEFAULT `swipe_engine_mode`, seven served
languages, the neural engine deleted, and the 2026-08-20 remediation wave landed.
**Training-side source of truth**: `CleverKeys-ML` @ `ctc/` — `MODELS_TABLE.md` §4.16,
`PHASE_O.md`, **`PHASE_P.md` (v2 generator, §6.1 and §8.4 artifact registries)**,
`PHASE_I_DATA.md` §4–§6, `PHASE_J.md` §6.9, `ALT_LAYOUT_EVAL.md`,
`YANDEX_LICENSE_RESEARCH.md`, `APP_INTEGRATION_AUDIT.md`.

> **Mirror warning.** This file has an app-repo copy at
> `docs/specs/ctc-architecture-and-multiscript-guide.md`. The two were byte-identical when written
> and **have since diverged**: the app copy is still at the pre-Phase-P text — `cyrillic_synth.py`,
> `weight = 255 − rank`, "real data is worth ~13 points", greedy 37, and the v1 `ru_synth_ch80*`
> hashes. The sections that must be brought across are enumerated in `APP_WIRING_CHECKLIST.md` §3.
> Until that lands, **this copy is the authority** and the app copy is one model generation stale.

This document exists to kill four recurring confusions permanently:

1. "The CTC model's alphabet is hardcoded a–z." — **The model has no alphabet at all.**
2. "Non-Latin needs a per-script model; is that tractable? No." — **Per-script models are
   needed and they cost ~30 minutes of GPU per seed.** Russian is done; the artifacts are
   named in §4.
3. "37 layouts don't declare a script." — **Two do not, and neither is a letter layout.**
   The real gap was three *mis-declared* or letter-incomplete layouts; one of the three was a
   genuine bug and is now fixed and guarded (§2.1).
4. "Which ONNX?" — §5 is the inventory table.

---

## 1. How the architecture actually works

### 1.1 The head has 64 geometry-conditioned slots, not 26 letters

The exported graph's contract (`CleverKeys-ML/ctc/model.py`, and the Kotlin twin in
`swipe/ctc/`):

```
in : features    [1, 2, 64]  float32   x row, y row, resampled to 64 frames, both in [0,1]
     layout_keys [1, 64, 2]  float32   key centers in the model's [0,1] frame, pad = (0,0)
     layout_mask [1, 64]     bool      true for the first K real key slots
out: log_emissions [1, 32, 65] float32 log-softmaxed; blank is column 64
     coefficients  [1, 32, 64] float32 (phase-2 refinement input, unused in the app)
     lambda        [1, 32, 1]  float32 (per-frame positive gate)
```

`MAX_KEYS = 64`. The emission head is 65 wide: **64 key slots plus one blank**. There is no
letter anywhere in the graph. Emission column `c` means *"the key that the caller put in slot
`c` of `layout_keys`"* — nothing more. Scoring is
`key_logits = coefficients @ keyEmbed(layout_keys)ᵀ * lambda`, and `keyEmbed` is a function of
`(cx, cy)` plus a 64-dim cosine positional encoding of that same `(cx, cy)` — **never of the
slot index**. That is precisely why the slot-permutation augmentation works during training,
and it is why a 31-letter ЙЦУКЕН layout simply uses 31 of the 64 slots with no model change
(`PHASE_I_DATA.md` §4, measured, not assumed).

Two independent confirmations that the alphabet is *data*:

- the Russian encoder's ONNX is **1,142,727 bytes — byte-size-identical** to every other
  `resbn:80` artifact in the campaign, because the alphabet never enters the parameter count;
- the app's own decode-side slice, `CtcEmissions.sliceFromHead(fullHead, frames, maxKeys,
  numLetters)`, takes `numLetters` as a *parameter* and copies columns `0 until numLetters`
  plus column `maxKeys` (the blank). It has no 26 in it.

### 1.2 Where the a–z actually lives

Three places, all app-side, none of them the model:

| Site | The a–z assumption | File |
|---|---|---|
| `CtcEngineAdapter.ALPHABET` | `CharArray(26) { 'a' + it }` — the only alphabet the adapter ever builds | `swipe/CtcEngineAdapter.kt:113` |
| `buildMappedLayout` | `FloatArray(26)` / `BooleanArray(26)`, `letterOf` accepts only `'a'..'z'`, and returns **null** unless all 26 are present | `swipe/CtcEngineAdapter.kt:266-301` |
| `SwipeEngineRouter.isLatinScript` | `script.equals("latin", ignoreCase = true)` | `swipe/SwipeEngineRouter.kt:118-119` |

Everything downstream is already script-generic and was written that way on purpose:

- `CtcLayout(alphabet: CharArray, keyCentersX, keyCentersY)` — any K, any chars, deduplicated;
- `CtcLexiconTrie(alphabet: CharArray)` — constructor bound is `alphabet.size <=
  CtcFeaturizer.MAX_KEYS` (64), i.e. the *emission head width*, not 26;
- `CtcTrieNode.addChild` — the child-array growth is **deliberately unclamped**. It used to be
  `minOf(chars.size * 2, MAX_CHILDREN=26)`, which saturates at 26 and throws
  `ArrayIndexOutOfBounds` on the 27th distinct child. That clamp is **already removed**
  (`swipe/ctc/CtcLexiconTrie.kt:86-104`, and the constant is gone rather than merely unused,
  `:123-130`). **A 31- or 33-letter Cyrillic trie is safe in the shipped code today.** This is
  the single most important piece of pre-work that is already done.

So: "the ALPHABET is hardcoded a–z" is true of *two constants in one adapter file* and false of
the model, the emissions type, the layout type and the trie.

### 1.3 The layout frame — and why it lines up across scripts

`buildMappedLayout` normalizes each letter's center against the **bounding box of the letter
keys only** (`left/top/right/bottom` accumulated over a–z rects), not against the keyboard
frame. Every trained geometry in the campaign lives in that same canonical letter-area frame —
including the Cyrillic conversion, where using a naive `y / height` instead would have squashed
the letter block to 46 % of the unit square and landed outside the measured affine-tolerance
envelope (`PHASE_I_DATA.md` §4). The frames already agree; a ru wiring does not have to invent
one.

### 1.4 Why ONE model serves ALL Latin a–z layouts, including custom user XML

Because key geometry is a model *input*, the encoder is arrangement-agnostic **by design**.
Training then makes that design real: the layout-resampling augmentation (`layout_aug.warp_path`)
re-anchors real human traces onto arbitrary a–z arrangements, so an arbitrary Latin layout is
in-distribution rather than merely "hopefully close".

The measured evidence for the shipped model (`phaseM_kd_fresh_w1_s1234_fp16w`,
`MODELS_TABLE.md:113`, in-dict top-1, en lexicon, 3 seeds):

| layout | top-1 | note |
|---|---|---|
| dvorak | **91.82** | never trained on as a named layout — held out |
| dvorak, app-geometry trie | **91.10** | the app's own 98 k trie footing |
| spanish | 89.53 | |
| azerty | 84.53 | |
| qwertz | 83.97 | |
| german | 81.30 | worst measured layout |

Two things make this credible rather than hopeful. First, `ALT_LAYOUT_EVAL.md` characterised
the displacement sensitivity and showed **earlier, weaker models visibly failing it** — ch128
scored dvorak **63.04**, `fast_resbn80` **67.28**. The ship model's 91.82 is therefore a
property of the training recipe, not of a forgiving benchmark. Second, six layouts spanning a
wide displacement range all hold.

**Honest tier note.** Colemak specifically, and arbitrary user XML generally, were **never
benchmarked**. They are covered *by design* (geometry input + arbitrary-arrangement
augmentation) and the floor is not measured — the worst *measured* layout is german at 81.30
against geometric's ~77 % top-1, so the expected-value case for routing them to CTC is strong,
but "Colemak ≥ geometric" is an inference, not a measurement. Say it that way.

**Do NOT quote 89.87 / 88.98 / 80.64 / 88.45 / 83.81 / 83.01.** Those are `sw2345`'s
(`MODELS_TABLE.md:139`) — a superseded Phase-J finalist that was **never decoded on test**. The
app's `src/main` was cleaned of them, and `CoreImeHygieneDriftTest` now bans them — but the
guard walks `src/main/kotlin` only, and `src/test/.../SwipeEngineRouterTest.kt:20` still quotes
them (see §6, checklist item 3).

---

## 2. The routing rule, stated once

```
swipe_engine_mode == "geometric"          -> GEOMETRIC
layout.script resolves to "latin"
    AND the layout exposes all 26 a-z as CENTRE key values   -> CTC
    AND the active language is in CtcLanguageSupport.SUPPORTED
anything else                                                -> GEOMETRIC
```

As implemented, this is three gates in dispatch order, each individually sufficient:

1. **Script** — `SwipeEngineRouter.route` returns `Engine.CTC` iff `isLatinScript(script)`.
   `script == null` is false, so an undeclared layout goes geometric.
2. **Language** — `InputCoordinator.performCtcSwipeTyping` reads the live language before
   dispatch; `CtcEngineAdapter.decodeAsync`/`warmUpAsync` re-check as defense in depth.
   `SUPPORTED` at `d717bda7` is `en` (EN_JSON scale) and `fr, de, es, it, pt, sv` (CKDT scale);
   `it, pt, sv` are marked `PROVISIONAL`.
3. **Alphabet completeness** — `buildMappedLayout` returns null on the first missing a–z
   letter, `supportsLayout` returns false, and the dispatcher hands the swipe to geometric
   *before any CTC work starts*. No crash, no empty bar, no garbage decode.

A **fourth** condition joined these on 2026-08-20 (`ad18e0e3`, closing audit HIGH-1): if the
ONNX session has permanently failed to load, `CtcEngineAdapter.isModelPermanentlyUnavailable()`
is true and both the dispatcher (`InputCoordinator.kt:725-733`) and the prewarm (`:793-796`)
route to geometric. Before that fix a dead session produced an empty slate, which the shared
pipeline rendered as a cleared bar — indistinguishable from a bad gesture. It matters for
multi-script work because **per-script routing removes gate 1**, so the remaining gates carry
more weight than they did.

**Russian cannot reach CTC in any state**, and that is correct, because no Russian model,
trie, or fixture is wired into the app. Gate 2 also covers the inverse case (`ru` language on a
QWERTY layout): geometric, not CTC.

### 2.1 The "undeclared script" question — measured, not assumed

Measured on `src/main/layouts/` at `d717bda7` (86 XML files; this is the tree
`copyLayoutDefinitions` ships — `srcs/layouts/` is **not** referenced by any build task and has
been divergent before):

| bucket | count | consequence |
|---|---|---|
| `script="latin"` **and** a–z-complete | **46** | route CTC — the intended set |
| `script="latin"` but a–z-incomplete | **2** | router lets them past, the alphabet gate stops them → geometric |
| non-Latin declared (15 distinct scripts) | **36** | geometric at gate 1 |
| no `script` attribute at all | **2** | `numeric.xml`, `pin.xml` — not letter layouts, correctly geometric |

There is **no population of 37 undeclared layouts.** The two latin-declared-but-incomplete files
are `latn_qwerty_az.xml` and `latn_qwerty_tly.xml`, both genuinely lacking `w`, both correctly
declared — geometric is the right answer for them.

**The third one was a real bug and is now fixed.** `src/main/layouts/grek_qwerty.xml` declared
`script="latin"` while its sibling `srcs/layouts/grek_qwerty.xml` had been corrected to `greek`
in `6af11da7` ("closes neural-swipe allowlist leak") — a tree no build task reads, so the fix
looked landed and was not. `6f30d60f` (2026-08-20) corrected the shipped file; both copies now
read `script="greek"` and are byte-identical.

It is also **guarded**, which is the part worth keeping. `LayoutScriptDeclarationTest` walks the
real shipped tree (`File("src/main/layouts")`) and asserts `script="latin"` ⟺ a–z-complete
**bidirectionally** — it catches a Greek layout tagged Latin *and* an a–z-complete layout denied
CTC for no reason — with `latn_qwerty_az` / `latn_qwerty_tly` as named exceptions, plus a by-name
pin on Greek and a pin on the no-script set. Writing it turned up something worth recording:
**the layout tree uses two schemas for the centre key value**, `<key c="q" ne="1"/>` and
`<key key0="p" key1="…"/>`, and both are live. Matching only one makes ~40 layouts look like they
contain no letters, and the header comment in every layout file documents `key0` while
`latn_qwerty_us` uses `c` — so the comment is not a reliable guide.

**Still owed** (audit LOW-9, half done): the router-level negative exists
(`SwipeEngineRouterTest` pins "the Greek QWERTY trap: script wins over the QWERTY-shaped name"),
but nothing asserts `CtcEngineAdapter.supportsLayout(...) == false` for a Cyrillic or Greek
`KeyboardData`. Gate 3 is untested in the negative direction — which becomes load-bearing the
moment per-script routing removes gate 1.

---

## 3. Non-Latin: per-script models are required, and they are cheap

### 3.1 Both halves of the claim

**Required.** A model trained on Latin-arrangement geometry with English lexicon statistics does
not zero-shot another script. The geometry is a model input, but the *motor statistics* and the
*implicit character-transition prior the encoder learns* are trained. Do not route a non-Latin
layout at CTC and hope.

**Cheap.** The residual-transplant synthesis pipeline generalizes to **any** script given only
(i) a word list and (ii) the layout geometry. No data collection, no corpus licensing, no human
subjects. Measured cost for Russian: 94 k steps of `resbn:80` — well under an hour of a single
RTX 5080 — plus a few minutes to export and evaluate.

**Proven, not projected.** `phaseP-ru-v2full` was trained on 1,000,000 rows in which **no real
Cyrillic sample appears anywhere** (checkpoint selection ran on a synthetic val split too), and
it decodes **real** Russian swipes at in-dict top-1 **79.73** with greedy **56.12** (§4.3). Its
v1 predecessor read 77.41 / 37.07 on the same rows; the +2.31 is paired at p = 2.6e-09. Both are
above the shipped geometric engine's cross-layout anchors (71–77).

**And the cheaper option first, because this is where the accuracy actually is.** The shipped
**English** model, zero-shot on real Russian with nothing but the right layout and the right
trie, reads **76.32** in-dict top-1 (`PHASE_O.md` §2.1). The purpose-built ru model adds
**+3.41** on top of that. So of the ~79.7 points available, the app wiring — per-script alphabet,
routing, trie, projection — delivers 76.3 of them **before any model ships**, and the model
delivers the last three. If the work has to be staged, stage the wiring first; a script with the
wiring and no model is already at the geometric engine's level or above.

### 3.2 The recipe, step by step

All tools live in `CleverKeys-ML/ctc/`.

1. **Vendor the geometry.** One JSON per layout: `{name, letters, letter_block_px, keys:[{letter, cx, cy, rx, ry}]}`
   with centers in the **canonical letter-area frame** (x over the grid width, y over the
   *letter-key block* — rows at cy 0.167 / 0.5 / 0.833). See `layouts/ru_jcuken_default.json`.
   Loads through the campaign's `load_layout`.
2. **Validate the frame before training anything.** Endpoint-proximity on a couple of thousand
   traces against the claimed geometry, **with a deliberately wrong geometry as a falsification
   control**. Russian measured start-hit 0.917 / end-hit 0.647 against a wrong-geometry control
   that collapsed to 0.008 / 0.004 (`PHASE_I_DATA.md` §4). Skipping this step is how a whole
   training run gets wasted on a squashed frame. Endpoint statistics are necessary and **badly
   insufficient**: they are the *least* discriminative view a real-vs-synthetic classifier has
   (0.663 against the speed profile's 0.904), which is exactly why v1 passed its only gate while
   carrying a KS-0.60 speed defect. Gate on the kinematics too — `synth_gap_audit.py --stage
   gates` is the standing battery, and its cheapest high-value metric is **lag-1 speed
   autocorrelation**, the largest single-statistic real-vs-synth gap found anywhere.
3. **Synthesize.** `script_synth.py --code <script>` — the generic residual transplant, at
   **generator v2** (`cyrillic_synth.py` remains as the historical record of the ru v1 run).
   English human traces are the donor pool; the correspondence is geometric
   (`layout_aug.warp_path` verbatim through per-vertex virtual indices) and **letter identity
   never enters**. Five stages, all measured into place by `SYNTH_V2_DESIGN.md` /
   `SYNTH_V2_RESEARCH_AUDIT.md` and gated in `PHASE_P.md`:
   * **S0 word draw = wordfreq token mass**, not `255 − rank`. The rank scale is a 255:1
     compression of frequencies that span 10⁵:1, and it produced a corpus with **3.3 % ≤3-letter
     words against real usage's 35.6 %**. Accumulate each wordfreq *token's* mass into **its
     projection** — querying `word_frequency` with the projected form returns zero for 90 % of
     the Greek pack, because the projection strips accents and restores final ς.
   * **S1/S2 geometry-matched donor draw**: index by (vertex count, log polyline length), take
     k = 16 reservoir candidates and pick the one minimising `Σ_seg |log(L_dst/L_src)|`. Takes
     the per-segment stretch p95 from 3.63 to 1.72.
   * **S4 vertex-aligned per-segment re-timing** at α = 0.5. v1 let the arc remap scale the
     donor's sample spacing by the segment length ratio, which multiplies the implicit speed
     profile by that ratio and never re-times it — the single largest measured defect
     (step_cv KS 0.60, peak per-step speed 3.2× real). Copy the donor's *within-segment*
     arc-progress so its dwells land on the **target's** vertices, and reallocate the sample
     budget across segments by `m_k ∝ n_k·ρ_k^α`. α is not tuned: the within-trace regression of
     per-segment time share on ideal-length share reads 0.460 (HWS) / 0.493 (FUTO) / 0.447 (real
     ru), an isochrony invariant that transfers across scripts, so it is fit on MIT English.
   * **S5 acquisition-bandwidth matching**: predict a duration from the donor's own
     (`T_target = T_donor·(L_dst/L_src)^0.262`, the exponent fit on MIT English only) and
     re-featurize through the real 60 Hz chain. Half the residual "synthetic traces are jagged"
     gap is a sampling artefact — fast target-script traces are *upsampled* to 64 points while
     slower English donors are *downsampled* and keep their jitter — not motor behaviour.
   * **Never fit any generator parameter against the validator's statistics.** Tempo, duration,
     start-dwell and the length mix are all fit on MIT English and only *checked* on Russian.
   `--generator v1` reproduces the old mechanism bit-exactly for paired ablations.
4. **Train.** `train.py --layout <geometry.json> --cache <cache_dir> --beam-val-rows 0`. The
   `--beam-val-rows 0` matters: the in-training beam validator's vocab loader is a–z-hardcoded,
   so selection runs on greedy and the lexicon beam runs offline in `eval_cyrillic.py`. The rest
   of the recipe is untouched — `resbn:80:1,2,4,8`, embed_hid 96, 94 k steps, batch 256, lr 3e-3,
   coupled affine sampler, no layout-alt (single geometry).
5. **Export.** `export_onnx.py --ckpt ... --layout <geometry.json> --parity-features <cache>/val.npz`.
   The parity assertion runs on the **sliced contract view** and on **real traces on the real
   layout**; argmax agreement must be 100/100. Then `quantize_onnx.py --mode fp16w` for the ship
   bytes.
6. **Freeze a golden fixture.** `make_golden.py --vocab <script> --layout <geometry.json>
   --preset <the app preset>`. The fixture must be generated at the preset the app will actually
   ship, on the lexicon's real frequency scale, or it validates a ranking that never runs.
7. **Check the lexicon against the 32-frame budget.** The exported encoder emits a fixed
   `log_emissions [1,32,65]` and a CTC path spends one frame per character **plus a separating
   blank between two identical adjacent characters**, so a word is decodable iff
   `length + adjacent-duplicate-pairs ≤ 32`. The app added `CtcDecodableLength` and a
   custom-word warning for this on 2026-08-20 (`2d080c7d`), with a test asserting every 20+
   character word in `en_enhanced.json` clears it. **No script lexicon has been checked** — the
   v2 word draw is wordfreq token mass with no length ceiling, and Greek and Ukrainian both
   carry long inflected forms. It is one loop over the trie's word list; run it before training,
   because a word over budget is silently unemittable rather than an error.
8. **Measure on real data if any exists** — as a held-out *eval-only* probe. Synthesis is the
   training story; real data is how you find out whether the synthesis worked.

### 3.3 What synthesis buys and what it costs

The paired arms answer this exactly. Same recipe, same eval, same 9,416 real rows
(8,471 in-dict; the v2 row is at the app CKDT preset λ 2.0, the two older rows at E1 λ 1.1):

| arm | training data | in-dict t1 | greedy |
|---|---|---|---|
| `phaseIB-ru-real` | 1 M **real** Yandex rows | 89.64 | 75.23 |
| `phaseIB-ru-synth` — generator **v1** | 1 M synthetic rows, zero real | 76.21 (77.42 at λ 2.0) | 37.07 |
| **`phaseP-ru-v2full`** — generator **v2** | 1 M synthetic rows, zero real | **79.73** | **56.12** |

Two readings, and the second is the one that changed.

**Real data is still worth ~10 top-1 points**, down from ~13. That gap is the honest price of
having no corpus, and `DATASET_SCOUT.md` §4.4 argues causing collection is the only clean route
to closing it.

**But half of what looked like "the price of synthesis" was generator error.** v2 is +2.31 real
top-1 over v1 (paired McNemar p = 2.6e-09) and **+19.05 real greedy** — the encoder's own
emissions, with no real Cyrillic row in training. v1's greedy of 37 was read as "English-magnitude
start noise on a denser board"; it was mostly a speed profile that made synthetic traces 90 %
separable from real ones. Consequences for anyone launching a script:

* a v2-trained model still leans on its trie, but far less — budget the lexicon accordingly,
  and note that **λ = 2.0 was tuned against a weak-emission model** and probably wants revisiting
  once a second real corpus exists to tune it on;
* the gain lands on **long words** (≥4 letters: +4.22, p = 3.6e-17). Short words were already
  carried by the lexicon prior and moved −0.70 (n.s.);
* an English-donor transplant **cannot** be made statistically indistinguishable from real
  target-script swipes. The English→English control — same generator, same script, disjoint
  halves of one corpus — closes 84 % of the gap to a measured 0.50 floor, while the Russian arm
  closes 36 %. The difference is the donor bank's population, and the only lever on it is
  target-script motor data.

---

## 4. Russian — DELIVERED, as the worked example

### 4.1 What is shippable and what is not

| model | in-dict t1 | status |
|---|---|---|
| `phaseIB-ru-real` (1 M real Yandex rows) | 89.64 | **LICENSE-BLOCKED FOREVER.** Not shippable in any form, at any time. |
| `phaseJ-joint` (single en+ru model) | 78.23 confirm-half @ λ 2.0 | **REJECTED.** Its data is license-clean, but it cost **−0.42 en top-1** against a 0.3 tolerance and was not adopted. Its ru lead over the bar-holder is +0.31 on the confirm half — inside one binomial SE (±0.64 at n = 4,240) — and it *loses* t3 and t5 on that same half. |
| `phaseIB-ru-synth` = `ru_synth_ch80` (generator v1) | 77.41 full-set / 77.92 confirm-half @ λ 2.0 | **SUPERSEDED.** Shippable, but beaten by its own successor at matched everything. |
| **`phaseP-ru-v2full`** = **`ru_synth_v2_ch80`** (generator v2, full donor pool) | **79.73** full-set @ λ 2.0, greedy 56.12 | **THE SHIPPABLE ONE.** Trained purely on residual-transplant synthesis; zero Yandex rows anywhere in its pipeline. G5 PASS against a pre-registered ≥ 79.41 floor. |
| the shipped **English** model, zero-shot on ru | 76.32 | not a Russian model at all — the baseline the wiring alone reaches (§3.1). |

**The licensing line, stated exactly** (`YANDEX_LICENSE_RESEARCH.md`, 941 lines of it): the
Yandex Cup 2023 corpus has **no license grant anywhere** — not in the contest rules, the Cup
regulations, the Disk link, the solution repos, or the Kaggle mirror. Background terms are
restrictive: Yandex's services agreement authorises only personal non-commercial use, and the
corpus is a protected database under ГК РФ ст. 1334 (6 M rows ≫ the 10 k presumption; term to
~2039) whose ст. 1335.1 carve-outs cover research and education but **not a shipped product**.
Verdict: **research and held-out evaluation only; synthesis-only for anything that ships.**
`ru-synth` is trained synth-only and *evaluated* on Yandex rows — which is exactly the permitted
footing. `ru-real` is a research artifact and stays one.

### 4.2 The artifacts

Committed in `CleverKeys-ML/ctc/artifacts/`. **Generation 2 supersedes generation 1 for
deployment** (`PHASE_P.md` §6.1); the v1 bytes stay because every pre-Phase-P number was
measured on them.

| file | bytes | sha256 |
|---|---|---|
| **`ru_synth_v2_ch80.onnx`** (fp32, **generator v2**) | 1,142,727 | `763190f9bc9854a3183f10d7dba7d8e1de1c101812b5958ee9bdbb403b93089b` |
| **`ru_synth_v2_ch80_fp16w.onnx`** (**ship bytes**) | 589,406 | `9004befb6ff07b744c65d3c13481539e758ebe10d4f47cbeffe68d39d12b0e52` |
| **`ru_synth_v2_ch80_fp16w_golden.json`** (fixture) | 160,282 | `a5ed2b9f62843d085779f5ab7457e6608f5c47e8994c224146ebdaf32fcdb82d` |
| `ru_synth_ch80.onnx` (fp32, generator v1 — superseded) | 1,142,727 | `d78a9fb9f8e170595a7714220cf5fd9dfc2324935900aec6cb6d7a2ec1a36666` |
| `ru_synth_ch80_fp16w.onnx` (v1 — superseded) | 589,406 | `84ac284d4f0d0cb86061df9c557507e1489ab93a75b40885a4431976cee32469` |
| `ru_synth_ch80_fp16w_golden.json` (v1 fixture — superseded) | 160,876 | `041c20722a957d1341108eb969dc677a123363011094ad05b36fdc1baa1050b0` |

The v2 source checkpoint is `~/ctc-train/ckpt/phaseP-ru-v2full/best.pt` — same architecture,
same 94 k schedule, same seed, **different training distribution**. The app-side contract is
unchanged: same alphabet string, same slot order, same preset, same fixture shape, so swapping
generations is a model-and-fixture swap and nothing else. The five other scripts have the same
pair of generations (`{el,uk,bg,mk,he}_synth_v2_ch80*`); hashes in `PHASE_P.md` §6.1.
Those five then gained a **third** generation in P6 — `*_synth_v2full_ch80*`, the same
generator on the full donor pool, which is what should be wired if any of them is —
hashes in `PHASE_P.md` §8.4. ru has no third generation: `ru_synth_v2_ch80` is already
the full-pool arm.


Architecture, identical across both generations: `resbn:80`, dil `1,2,4,8`, embed_hid 96,
feat_v1, `t_out` 32, 279,346 params, 94,000-step schedule, batch 256, lr 3e-3, wd 0.01,
warmup 1,000, coupled affine sampler, no layout-alt, greedy checkpoint selection, seed 1234.
The v1 fp32 re-export was **byte-identical** to the artifact its 2026-08-09 training run
produced — a free determinism check on the whole export path.

Export gates for the **v2 ship bytes**, all passed (`PHASE_P.md` §5):

- BN fold: max |Δlog_emissions| **1.20e-04** on the sliced contract view (tolerance 5e-3);
- fp32 export parity vs torch, real traces on `ru_jcuken_default`: sliced **9.92e-05**, argmax
  **100/100** (tolerance 1e-3, and argmax is the binding gate);
- fp16w vs fp32, real traces: sliced **1.06e-01**, argmax **93/100**. This residue is large —
  larger than the ch192 ship model's 2.30e-02 — and it is **disclosed, not hidden**: the binding
  check is the decode, and the decode costs **+0.02 t1** (79.73 → 79.75).

(The v1 numbers, for anyone reading a pre-Phase-P document: BN fold 1.60e-04, fp32 sliced
1.14e-04 argmax 100/100, fp16w sliced 1.16e-01 argmax 98/100.)

### 4.3 Validation of the exported artifact

`eval_cyrillic.py` / `eval_script.py`, layout `ru_jcuken_default`, lexicon `app` (the langpack-ru
CKDT v2 50 k trie), preset `1.05, 2.0, 0.2, 0.3734, 0.9882` = the app's
`CtcScoringParams.tunedRuCkdt` verbatim, beam 100. Probe = the untouched Yandex valid-10k, all
9,416 default-grid rows, 8,471 in-dict, **eval-only footing**.

The generation-2 gate (`PHASE_P.md` §4, G5 — the only gate that decided shipping):

| arm | training cache | in-dict t1 | ≤3 | ≥4 | greedy | t3 | t5 |
|---|---|---|---|---|---|---|---|
| `ru_synth_ch80` — the v1 baseline | v1, full donor pool | 77.42 | 86.47 | 71.70 | 37.07 | 89.06 | 91.76 |
| `phaseP-ru-v1ctl` — paired v1 control | v1, 90/10 train side | 75.73 | 83.66 | 70.71 | 31.34 | 88.44 | 90.93 |
| `phaseP-ru-v2` | v2, 90/10 train side | 78.87 | 83.60 | 75.88 | 55.67 | 90.73 | 93.13 |
| **`phaseP-ru-v2full` = `ru_synth_v2_ch80` — SHIP** | **v2, full donor pool** | **79.73** | 85.77 | **75.92** | **56.12** | **90.77** | **93.26** |

| paired comparison (exact McNemar, n = 8,471) | Δ t1 | p |
|---|---|---|
| **v2 full pool vs the v1 baseline** | **+2.31** | **2.6e-09** |
| v2 vs v1 at matched donor footing | +3.14 | 6.4e-14 |
| the cost of the 90/10 donor split, v1 arm | −1.69 | 5.2e-07 |
| v2 full pool vs v2 train side | +0.86 | 0.0023 |

**G5 PASS** — 79.73 against a pre-registered floor of 79.41. Two honest qualifications carried
from `PHASE_P.md` §4.2: the ≥4-letter stratum gained **+4.22** (p = 3.6e-17, and 61 % of real
usage lives there), while the ≤3 stratum **missed its registered corollary** at 85.77 against
86.4 — indistinguishable from zero (p = 0.27), entirely carried by the donor-side term rather
than by v2, and in the stratum where the lexicon prior rather than the encoder does the work.
Re-tuning λ to recover it was **refused**: λ is already one validator-fit parameter and the
Yandex probe is the only real one that exists.

**Export gates** for the v2 bytes: BN fold 1.20e-04 (sliced), fp32-vs-torch 9.92e-05 with argmax
**100/100** on real traces, fp16w-vs-fp32 1.06e-01 with argmax 93/100 — and the decode cost of
fp16w is **+0.02 t1** (79.73 → 79.75). The large fp16w emission residue is disclosed rather than
hidden, exactly as in generation 1; the binding gate is the decode and the decode is free.

The v1 numbers, for anyone reading a pre-Phase-P document: fp16w full set **77.41** / 89.07 /
91.76, confirm half 77.92, tune half 76.88. Those are the bytes every number published before
2026-08-19 was measured on.

### 4.4 The preset, the trie and the layout

**Preset.** `tunedRuCkdt` = γ 1.05, **λ 2.0**, β 0.2, γ-prune 0.3734, β-prune 0.9882
(`swipe/ctc/CtcScoringParams.kt:205-210`). λ = 2.0 is not a Russian constant, it is a
**frequency-scale** constant: `LAMBDA_CKDT_SCALE`. The app's CKDT dictionaries store
`freq = 255 − rank`; the `en_enhanced.json` scale wants λ = 4.0. The ru sweep
(`PHASE_J.md` §6.9) and the Latin per-language sweep independently landed on 2.0 for the CKDT
scale from different bases, which is what makes it a property of the scale rather than of
either footing. **λ = 1.1 (E1) is under-tuned for ru and costs ~1.2 top-1** — every ru number
published before 2026-08-11 is at that footing.

**Trie / dictionary.** Russian is **not** a bundled asset. `src/main/assets/dictionaries/`
ships `en_enhanced.json` plus `en/de/es/fr/it/pt/sv_enhanced.bin`, all Latin. Russian exists
only as an **importable langpack**: `scripts/dictionaries/langpack-ru.zip` (533,916 B),
`manifest.json` = `{"code":"ru","name":"Russian","version":2,"wordCount":50000,"hasPrefixBoost":false}`,
`dictionary.bin` = 2,088,865 B, magic `CKDT` v2, lang `ru` — the same container as the bundled
`*_enhanced.bin`. `eval_cyrillic.build_trie` reads exactly this zip, so the campaign's ru
numbers are on the app's own lexicon, not a research one. Projection policy for both targets
and lexicon: lowercase, strip `-` and `'`, ё→е, ъ→ь, and **no Unicode NFD** — NFD decomposes
й into и + breve and silently destroys the alphabet.

**Layout.** `CleverKeys-ML/ctc/layouts/ru_jcuken_default.json` — 31 letters
(`абвгдежзийклмнопрстуфхцчшщыьэюя`, no ё, no ъ), generated from the corpus's own embedded grid,
frame-validated in §3.2 step 2. A 33-letter `ru_jcuken_extra.json` is vendored but **untrained**
(it covers 6.2 % of the corpus).

**MAX_CHILDREN.** Acknowledged and already handled: the trie's per-node child-array clamp at 26
would have thrown on the 27th distinct child, i.e. on the first real Cyrillic trie. It was
removed deliberately, with the constant deleted rather than left unused, and the real bound
moved to a constructor check against the emission head width
(`swipe/ctc/CtcLexiconTrie.kt:86-104, 123-130, 196-206`). **No trie work is blocking Russian.**

### 4.5 Evidence tier — say it exactly this way

> Russian CTC is **val-tier, single-seed, license-clean-synthesis-trained, Yandex-eval-only**.

Unpacked:

- **val-only, permanently.** No sealed Cyrillic test split has ever existed and none can be
  created from a corpus that cannot be licensed. The campaign's sealing discipline
  (test-2400, ledger reads) has **no Cyrillic counterpart** and never will. Russian numbers can
  never be called "test-validated".
- **single seed** (1234). Every other campaign bar is a seed-mean bar; this is not one.
- **no per-language preset sweep beyond λ**, and λ itself is now suspect in a new way: it was
  tuned in `PHASE_J.md` §6.9 against a *weak-emission* model (greedy 37). The v2 model reads
  greedy 56 and leans on the prior far less, so λ = 2.0 is probably no longer the right balance.
  Registered open, deliberately not re-tuned (`PHASE_P.md` §4.2). γ, β and the prune constants
  are E1's, carried.
- **the eval corpus is Yandex.** Permitted (research/held-out eval), but it means the *only*
  real-Russian evidence for this model comes from a source whose data can never enter training
  or the APK.
- **no on-device measurement.** No latency number, no memory number, no instrumented run.

### 4.6 What still has to happen before ru could ship

Not blockers on the ML side — app-side wiring, listed so nobody thinks the export was the last
step. Status is against the app at `d717bda7`; the ordered, actionable form of this list lives in
`APP_WIRING_CHECKLIST.md`.

| # | change | status at `d717bda7` |
|---|---|---|
| 1 | bundle or gate on the ru langpack (an import today, not an asset) | **open** — `scripts/dictionaries/langpack-ru.zip`, not in `assets/dictionaries/` |
| 2 | per-script `ALPHABET` + `buildMappedLayout` in `CtcEngineAdapter` — today `letterOf` filters `'a'..'z'` and the arrays are `FloatArray(26)`/`BooleanArray(26)` | **open** |
| 3 | per-language model asset — `CtcEngineAdapter.MODEL_ASSET` is one constant | **open** |
| 4 | `ru` in `CtcLanguageSupport.SUPPORTED`, and a script gate that admits `script="cyrillic"` | **open** |
| 5 | make `tunedRuCkdt` reachable — `presetFor` branches on `LexiconSource` and can never return it | **open** |
| 6 | ship the golden fixture and extend `CtcParityTest`'s fixture↔model↔preset triple to a second row | **open** |
| 7 | trie width — the `MAX_CHILDREN = 26` clamp | **done** (`d671d19e`); the bound is now a constructor check against the emission-head width |
| 8 | `CtcLayout` generic over `alphabet: CharArray` | **done** |
| 9 | measure latency and memory on device | **open**, but now *possible*: the settle-probe inflation (audit MEDIUM-2) was removed in `716f7be9`, so a `LOCAL_BUILD=true` measurement is no longer ~720 ms high |

The ru model is *half* the shipped English model's bytes, so the latency expectation is
favourable — expectation is not measurement.

### 4.7 The other five scripts — the per-script wiring table, refreshed

Supersedes `PHASE_O.md` §3.2/§3.5, which named the generation-1 artifacts. **Wire
`*_synth_v2full_ch80_fp16w*` for the five, and `ru_synth_v2_ch80_fp16w*` for Russian** — see §5
for the hashes and §4.2 for why ru has no `v2full`.

| script | layout XML (`src/main/layouts/`) | K | alphabet / slot order (codepoint-sorted — **this IS the app's array**) | ship bytes | lexicon | preset |
|---|---|---|---|---|---|---|
| **ru** | `cyrl_jcuken_ru.xml` | 31 | `абвгдежзийклмнопрстуфхцчшщыьэюя` | `ru_synth_v2_ch80_fp16w.onnx` | `langpack-ru.zip` — exists, importable today | `tunedRuCkdt` |
| **el** | `grek_qwerty.xml` (now correctly `script="greek"`) | 25 | `αβγδεζηθικλμνξοπρςστυφχψω` | `el_synth_v2full_ch80_fp16w.onnx` | `langpack-el.zip` — exists, **needs the full el projection**, not just the ς repair (§7 item 9) | same numbers as `tunedRuCkdt` |
| **uk** | `cyrl_jcuken_uk.xml` | 31 | `абвгдежзийклмнопрстуфхцчшщьюяєі` | `uk_synth_v2full_ch80_fp16w.onnx` | **must be built** (`build_wordlist.py --lang uk`) | same |
| **bg** | `cyrl_ueishsht.xml` | 30 | `абвгдежзийклмнопрстуфхцчшщъьюя` | `bg_synth_v2full_ch80_fp16w.onnx` | **must be built** | same |
| **mk** | `cyrl_lynyertdz_mk.xml` | 31 | `абвгдежзиклмнопрстуфхцчшѓѕјљњќџ` | `mk_synth_v2full_ch80_fp16w.onnx` | **must be built** | same |
| **he** | `hebr_1_il.xml` | 27 | `אבגדהוזחטיךכלםמןנסעףפץצקרשת` | `he_synth_v2full_ch80_fp16w.onnx` | **must be built**, and `build_wordlist._is_script_word` needs a new `hebrew` branch (0x0590–0x05FF) | same |

**Preset: there is no per-script preset.** All six use γ 1.05 / **λ 2.0** / β 0.2 / γp 0.3734 /
βp 0.9882. λ = 2.0 is a **frequency-scale** constant (`LAMBDA_CKDT_SCALE`), not a Russian one —
every one of these lexicons is on the CKDT `255 − rank` scale, the same scale `fr/de/es/it/pt/sv`
already run at in production. The only preset that differs is `en`, whose `en_enhanced.json`
scale wants λ = 4.0. So `presetFor` needs to become reachable for `tunedRuCkdt`, not to grow six
new presets.

**Projection rules the app must mirror** (`PHASE_O.md` §3.4 — applied to the lexicon **and** to
anything compared against a decode):

* **all scripts** — lowercase; strip `- ' ’ ʼ ‘ \``.
* **el, he** — NFD, drop combining marks (`Mn`), NFC. Safe because no letter's identity depends
  on a mark here: Greek accents/diaeresis and Hebrew niqqud are not keys.
* **ru, bg, mk** — **no NFD** (it decomposes й into и + breve and destroys the alphabet).
  Character folds instead: ru ё→е, ъ→ь; bg ѝ→и; mk ѐ→е, ѝ→и.
* **el only** — *after* mark stripping, word-final `σ` → `ς`.
* **uk** — no folds; words containing ї or ґ are **rejected as untypeable** (4.03 % of the
  vocabulary). Serving them needs the corner-alias path, which is a different input mode.

**Evidence tier for these five — say it exactly this way** (`PHASE_O.md` §2.7, carried forward
through `PHASE_P.md` §5.1 and §8.6):

> Greek, Ukrainian, Bulgarian, Macedonian and Hebrew CTC are **synthesis-trained,
> synthesis-holdout-only, single-seed, and calibrated against Russian rather than measured on
> their own script.**

Their v2full holdout numbers — el 90.78, uk 87.67, bg 82.52, mk 88.68, he 76.86 — are **not
accuracy figures for those languages**. A v2 holdout is generated by v2, and Phase O proved this
class of probe inverts model comparisons on capacity and on λ; Phase P §8 makes it three, adding
donor footing. What the holdouts establish is a *margin against a fixed control* (every script
beats the 3×-capacity English zero-shot by +5.1 … +7.9, where on the v1 holdouts every script
**lost**), and that a change measured as +0.86 real points on ru costs nothing on their own
distribution. Never quote the levels as if they were ru's 79.73.

**he carries a history flag that does not apply to its ship bytes.** The generation-2
`he_synth_v2_ch80` fp32 export needed `--parity-tol 2e-3` (sliced residue 1.16e-03 against a
historical 0.8e-4…7.6e-4 envelope, argmax 100/100 on both probes) and is **flagged in the
registry**. The generation-3 `he_synth_v2full_ch80` export needed **no relaxation** — 4.04e-04
at the default 1e-3, argmax 100/100, and 100/100 on the fp16w probe too. The flag stays on the
P4 bytes because that exceedance was real; it does **not** carry to the bytes you would wire.

---

## 5. The model inventory — which ONNX is which

**Exactly one CTC ONNX ships in the APK.**

| artifact | ships? | bytes | sha256 | serves | tier |
|---|---|---|---|---|---|
| `src/main/assets/models/ctc_swipe_encoder.onnx` = `ctc/artifacts/phaseM_kd_fresh_w1_s1234_fp16w.onnx` | **YES — the only one** | 3,052,318 | `84718e6ebc8020176f27b9668e50922a765c96838307b640a8db9ab0549e88e5` | en + fr/de/es/it/pt/sv on any a–z-complete Latin layout | **test-validated**, both footings, every seed |
| `ctc/artifacts/ru_synth_v2_ch80_fp16w.onnx` | no — **the ru ship candidate**, not wired | 589,406 | `9004befb6ff07b744c65d3c13481539e758ebe10d4f47cbeffe68d39d12b0e52` | Russian ЙЦУКЕН (31-letter default grid) | **val-only**, single seed, generator-v2 synth-trained, Yandex-eval-only |
| `ctc/artifacts/{el,uk,bg,mk,he}_synth_v2full_ch80_fp16w.onnx` | no — **the five ship candidates**, not wired | 589,406 each | §4.7 / `PHASE_P.md` §8.4 | Greek, Ukrainian, Bulgarian, Macedonian, Hebrew | **synthesis-holdout-only**, single seed, calibrated against ru rather than measured |
| `ctc/artifacts/{ru,el,uk,bg,mk,he}_synth_ch80*` (generation 1) | no — **superseded** | — | `PHASE_O.md` §2.6 | — | kept because every pre-Phase-P number was measured on them |
| `ctc/artifacts/{el,uk,bg,mk,he}_synth_v2_ch80*` (generation 2) | no — **superseded by v2full** | — | `PHASE_P.md` §6.1 | — | kept because `PHASE_P.md` §5 was measured on them; `he_synth_v2_ch80` carries a parity flag its v2full successor does not |
| `src/androidTest/assets/ctc_bench/{ch192,ch128,fast_resbn80,fast_resbn72}_s1234.onnx` | no — androidTest only, 11.0 MB | — | — | **nothing.** Superseded Campaign-2 arch-comparison artifacts, one of them still labelled "the ship candidate" in the benchmark's KDoc (audit MEDIUM-4, open) | historical |
| `phaseIB-ru-real` encoder (`cb8ece6b…`) | **NEVER** | 1,142,727 | — | — | license-blocked research artifact |
| `phaseJ-joint`, `sw2345`, `resbn192i`, `phaseL_*`, `phaseK_*`, `ch128/ch192`, `fast_resbn*` | no | — | — | — | superseded campaign arms |

**Three generations exist and all three stay in the registry**, which is deliberate and is the
commonest source of a wrong hash: `*_synth_ch80*` (v1, Phase O), `*_synth_v2_ch80*` (v2, Phase P
§5), `*_synth_v2full_ch80*` (v2 on the full donor pool, Phase P §8). **Deploy the newest one that
exists for the script**: `ru_synth_v2_ch80*` for Russian (ru has no v2full — it was already
trained on the full pool), `*_synth_v2full_ch80*` for the other five. The older generations are
kept only because published numbers were measured on them.

Golden fixtures:

| fixture | pairs with | preset |
|---|---|---|
| `src/test/resources/ctc/ctc_golden.json` = `src/androidTest/assets/ctc/ctc_golden.json`, sha `2a449c4f2de19505131b396655ae01d3e3c325e40249446ff6e7a40c2b27559c`, 140,462 B | the shipped ONNX (`84718e6e…`) — the **header sha** is asserted in CI, the **emission matrices are not**; see `APP_INTEGRATION_AUDIT.md` §6.2 | `tunedV2` = 0.9 / 4.0 / 0.25 / 0.25 / 0.9882 |
| `ctc/artifacts/ru_synth_v2_ch80_fp16w_golden.json`, 160,282 B, sha `a5ed2b9f6284…` | `ru_synth_v2_ch80_fp16w.onnx` (`9004befb…`) | `tunedRuCkdt` = 1.05 / 2.0 / 0.2 / 0.3734 / 0.9882 |
| `ctc/artifacts/{el,uk,bg,mk,he}_synth_v2full_ch80_fp16w_golden.json` | the matching v2full fp16w bytes | same `tunedRuCkdt` numbers |

Every script fixture is 10 cases (5 pure-featurizer branch probes, 1 word-path featurizer case,
4 model-backed beam cases) at the same preset — the same shape as the shipped en fixture, so
`CtcParityTest` grows a **row**, not a new mechanism.

**No model change is pending.** Phase N is terminal (`CleverKeys-ML` @ `85c0c58`); its 91.25
headline is a different corpus, trie and preset and is explicitly not comparable. Anything you
find that says otherwise — in particular
`docs/audit/remediation-plans/ctc-integration-execution-brief.md`, which still reads
*"Q1 model choice: SUPERSEDED-PENDING — a new model is training"* and names four candidates —
is a pre-decision planning document that nobody bannered. **That file is the single likeliest
source of the "which ONNX?" question.**

---

## 6. Audit findings — status at `d717bda7`

The full record is `APP_INTEGRATION_AUDIT.md`: §2 the original findings, §5 the re-verification at
`9a6ffdd2`, **§6 the second re-verification at `d717bda7`**. The ordered, actionable form for the
app agent is `APP_WIRING_CHECKLIST.md`. This section is the summary only, so it cannot drift into
a third competing task list.

**Of the 23 original findings plus the 4 added in §5: 15 CLOSED, 6 PARTIAL, 6 OPEN, 0 regressed.**

**Closed in the 2026-08-20 wave** — HIGH-1 (a latched ONNX-load failure now falls through to
geometric on both dispatch and prewarm, `ad18e0e3`; the diff this guide proposed applied
essentially verbatim), HIGH-3 (the dead-code KDoc is gone from all three files), MEDIUM-1 (the ORT
session is closed at teardown, after the decode thread is confirmed dead), MEDIUM-2 (**the settle
probes are gone — latency numbers from a `LOCAL_BUILD=true` build are quotable again**), MEDIUM-7
(the "this engine isn't serving your language" card is back, and formats its list from
`CtcLanguageSupport.SUPPORTED.keys` so it cannot drift), MEDIUM-8 (seven languages everywhere,
including all 22 locale strings), MEDIUM-9, NEW-1 (the spec rewritten around `ctc`-as-default),
NEW-2 (the `grek_qwerty` script tag, plus `LayoutScriptDeclarationTest`), NEW-4.

**Still open, and the short list is now genuinely short:**

| # | item | why it is still here |
|---|---|---|
| 1 | **MEDIUM-3** — `docs/audit/remediation-plans/ctc-integration-execution-brief.md` has no superseded banner; `:86` still says *"Q1 model choice: SUPERSEDED-PENDING — a new model is training"*, `:74` still *"Default engine stays `neural`"* | One paragraph. Unchanged across all three audit passes, and now the **only** anti-confusion finding left. It is the single likeliest source of the "which ONNX?" question. |
| 2 | **HIGH-4's residue** — CI now runs instrumented tests (real, green on API 21/29/34), but the curated class list omits `CtcEmissionModelParityTest` | So the fixture's *header* sha is checked on every push and its *emission matrices* are checked nowhere automatic — the exact swap failure HIGH-4 was written about. One string edit to `.github/scripts/emulator-ci.sh`. Also note `ui-testing.yml` runs on PR + nightly, **not on push to `main`**. Detail in `APP_INTEGRATION_AUDIT.md` §6.2. |
| 3 | **HIGH-2's residue** — two unmarked `sw2345` citations in `docs/`, at `docs/audit/2026-08-17-neural-vs-ctc-parity.md:619-623` (finding 13, unstruck) and `docs/eval/2026-08-15-ctc-per-language-lambda.md:101, 112` | The drift guard was widened to `src/test` and `src/androidTest` (`f172bb8e`), which caught the test-side copy. It still does not scan `docs/`, which is where both survivors are. |
| 4 | **NEW-6** — the app's guide mirror and `memory/HANDOFF.md` still cite generation-1 ru artifacts (`ru_synth_ch80_fp16w.onnx`, `84ac284d…`, 77.41) | `APP_WIRING_CHECKLIST.md` §3 enumerates the sections to bring across. |
| 5 | **MEDIUM-4** — 11 MB of superseded ONNX in `src/androidTest/assets/ctc_bench/`, one labelled "the ship candidate" | Item 2 of the app's own `HANDOFF.md`. |
| 6 | **MEDIUM-5, MEDIUM-6, LOW-9 (half), LOW-1..LOW-8, LOW-10** | Cleanup. LOW-9's remaining half — a `supportsLayout` negative for a Cyrillic `KeyboardData` — stops being cosmetic the moment per-script routing removes gate 1. |

---

## 7. What NOT to do

1. **No Yandex data in any training run or any shipped artifact — ever.** Eval-only, held-out,
   research footing. If a model's training pipeline touched a Yandex row, it cannot ship, no
   matter how good it is. `ru-real` at 89.64 is the standing example: better than everything
   else and permanently unusable.
2. **No FUTO model weights and no FUTO model outputs** in anything we train or ship. The
   corpus and the decode-algorithm lineage are the permitted inheritance and `NOTICE:46-64`
   states that carefully and correctly. Do not "improve" that wording; do not add a FUTO
   teacher.
3. **Respect the evidence tiers in prose.** "test-validated" means decoded on a sealed split
   whose read was spent from the ledger. "val-only" means everything else. Russian is val-only
   permanently. A val-only finalist's alt-layout number must never be quoted as the ship
   model's — that is exactly how HIGH-2 happened, and how it happened *twice*.
4. **Never route a non-Latin script to CTC without all three of** a per-script model, a
   per-script trie on the app's own lexicon at the app's own frequency scale, and a golden
   fixture at the preset that will actually ship. Two of three is a silently wrong decode, not
   a partial feature.
5. **Do not quote a latency number measured in a `LOCAL_BUILD=true` build** until MEDIUM-2 is
   resolved — the settle probes add ~720 ms to the first decode.
6. **Do not "fix" `CtcEngineAdapter` to use `CtcFeaturizer.normalizeRawX/Y`.** The shipped
   encoder was trained on letter-box normalization, not FUTO's 4/3 device frame; the adapter
   warns about this at `:51` and the module's own normalizers are production-dead (LOW-3).
7. **Do not add a per-node cap back to the trie.** The `MAX_CHILDREN = 26` clamp was removed on
   purpose (§1.2); the real bound is the constructor's alphabet-vs-head-width check.
8. **Do not assume `srcs/layouts/` ships.** It does not; `src/main/layouts/` does. A fix applied
   to the wrong one looks landed and is not — that is exactly what happened to the Greek script
   tag, for months (§2.1). The two trees agree today; `LayoutScriptDeclarationTest` guards the
   one that ships.
9. **Do not wire the el lexicon with only the final-sigma repair.** `CtcGreekOrthography`
   (`swipe/ctc/CtcGreekOrthography.kt`, shipped `6f30d60f`) is correct and is the *last* of the
   four steps in the el projection. The step before it — NFD, drop combining marks, NFC — has
   **no app-side implementation**, and it is the step that makes the alphabet 25 letters. The
   el model's slot order contains no accented vowels, so an unprojected `λόγος` has a character
   with no emission slot. Repairing sigma alone converts "one word in four is mis-keyed" into
   "most of the pack is unrepresentable". Both halves or neither.
10. **Do not wire a `*_synth_v2_ch80*` model for el/uk/bg/mk/he.** Those are generation 2; the
    deployment bytes are `*_synth_v2full_ch80*` (`PHASE_P.md` §8.4). The v2 rows survive in the
    registry only because §5's numbers were measured on them, and `he_synth_v2_ch80` additionally
    carries a parity flag that its successor does not. For **ru** the opposite holds:
    `ru_synth_v2_ch80` *is* the full-pool arm and there is no `v2full`.
11. **Do not quote a script's synthesis-holdout number as an accuracy figure.** el 90.78 is not
    "Greek at 90.78"; it is fit to the v2 generator's own distribution, on a probe this campaign
    has now shown three separate times to rank things real swipes do not rank (capacity, λ,
    donor footing). Quote margins against a fixed control, never levels. §4.7 gives the wording.
