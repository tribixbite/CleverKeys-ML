// CleverKeys RNNT Architecture Flow Diagram
// Generates ASCII and structured visualization of the complete pipeline

function generateArchitectureDiagram() {
    const diagram = `
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          CLEVERKEYS RNNT ARCHITECTURE                           │
│                              End-to-End Pipeline                                │
└─────────────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    1. TRAINING PHASE                                           │
└───────────────────────────────────────────────────────────────────────────────────────────────┘

Raw Training Data (JSONL)                 Feature Engineering
┌─────────────────────┐                   ┌──────────────────────────────────┐
│ {                   │                   │ SwipeFeaturizer                  │
│   "word": "hello",  │ ───────────────▶  │                                  │
│   "points": [       │                   │ 9 Kinematic Features:            │
│     {x:0.1,y:0.2,   │                   │ • x, y, t (position, time)       │
│      t:0.0},        │                   │ • vx, vy, speed (velocity)       │
│     {x:0.15,y:0.25, │                   │ • ax, ay, acc (acceleration)     │
│      t:0.1},        │                   │                                  │
│     ...             │                   │ 28 RBF Key Features:            │
│   ]                 │                   │ • Gaussian RBF to each QWERTY   │
│ }                   │                   │ • σ = 0.2, centers in [-1,1]    │
└─────────────────────┘                   └──────────────────────────────────┘
                                                           │
                                                           ▼
Feature Tensor [B, T, 37] ─────────────────────────────────────┐
                                                               │
                                                               ▼
                            ┌─────────────────────────────────────────┐
                            │            RNNT MODEL                   │
                            │                                         │
                            │  Encoder (Conformer)                    │
                            │  ┌───────────────────────────────────┐  │
                            │  │ • feat_in: 37                     │  │
                            │  │ • d_model: 256, n_heads: 4        │  │
                            │  │ • num_layers: 8                   │  │
                            │  │ • conv_kernel_size: 31            │  │
                            │  │ • subsampling_factor: 2           │  │
                            │  │                                   │  │
                            │  │ [B, 37, T] → [B, D=256, T/2]     │  │
                            │  └───────────────────────────────────┘  │
                            │                    │                    │
                            │                    ▼                    │
                            │  Joint Network                          │
                            │  ┌───────────────────────────────────┐  │
                            │  │ • joint_hidden: 512               │  │
                            │  │ • Combines encoder + decoder      │  │
                            │  │ • Output: [B, V=29, T']           │  │
                            │  └───────────────────────────────────┘  │
                            │                    ▲                    │
                            │                    │                    │
                            │  Decoder (LSTM Prediction Network)     │
                            │  ┌───────────────────────────────────┐  │
                            │  │ • pred_hidden: 320                │  │
                            │  │ • pred_rnn_layers: 2              │  │
                            │  │ • Takes previous token + state    │  │
                            │  │ • Output: [B, H=320, 1]           │  │
                            │  └───────────────────────────────────┘  │
                            └─────────────────────────────────────────┘
                                                │
                                                ▼
                            ┌─────────────────────────────────────────┐
                            │            RNNT LOSS                   │
                            │                                         │
                            │ • Alignment-free sequence loss          │
                            │ • Handles variable input/output lengths │
                            │ • Learns blank vs character emissions   │
                            │ • Enables streaming inference           │
                            └─────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    2. EXPORT PHASE                                             │
└───────────────────────────────────────────────────────────────────────────────────────────────┘

Trained Model (.nemo/.ckpt)
           │
           ├──────────────────────────────────────────────────────────────────────────────────┐
           │                                                                                  │
           ▼                                                                                  ▼
┌─────────────────────────┐                                              ┌─────────────────────────┐
│   ENCODER EXPORT        │                                              │   STEP MODEL EXPORT     │
│   (export_onnx.py)      │                                              │   (export_rnnt_step.py) │
│                         │                                              │                         │
│ 1. Extract Conformer    │                                              │ 1. Extract LSTM+Joint   │
│ 2. ONNX FP32 Export     │                                              │ 2. Single-step wrapper  │
│ 3. INT8 Quantization    │                                              │ 3. ONNX + PTE Export    │
│                         │                                              │                         │
│ Input:  [1, 37, T]      │                                              │ Input:  y_prev [N]      │
│         lengths [1]     │                                              │         h0 [L, N, H]    │
│ Output: encoded [1,D,T']│                                              │         c0 [L, N, H]    │
│         enc_len [1]     │                                              │         enc_t [N, D]    │
└─────────────────────────┘                                              │ Output: logits [N, V]   │
           │                                                              │         h1 [L, N, H]    │
           ▼                                                              │         c1 [L, N, H]    │
encoder_int8_qdq.onnx                                                    └─────────────────────────┘
                                                                                     │
                                                                                     ▼
                                                                          rnnt_step_fp32.onnx

┌───────────────────────────────────────────────────────────────────────────────────────────────┐
│                                  3. INFERENCE PHASE                                            │
└───────────────────────────────────────────────────────────────────────────────────────────────┘

Browser Swipe Input                    Feature Extraction
┌─────────────────────┐               ┌──────────────────────────────────┐
│ Touch Events:       │               │ Coordinate Normalization:        │
│ [{x: px, y: px,     │ ────────────▶ │ canvas [0,1] → model [-1,1]     │
│   t: timestamp},    │               │                                  │
│   {x: px, y: px,    │               │ Kinematic Features:              │
│    t: timestamp},   │               │ • Position: x, y, t              │
│   ...]              │               │ • Velocity: vx, vy, speed        │
└─────────────────────┘               │ • Acceleration: ax, ay, acc      │
                                      │                                  │
                                      │ RBF Key Features (28):           │
                                      │ • rbf = exp(-d²/(2σ²))           │
                                      │ • For each QWERTY key            │
                                      └──────────────────────────────────┘
                                                       │
                                                       ▼
                                      Features [1, 37, T] ──┐
                                                           │
                                      ┌────────────────────┘
                                      ▼
                      ┌─────────────────────────────────────────┐
                      │         ENCODER (ONNX)                  │
                      │                                         │
                      │ • Load: encoder_int8_qdq.onnx          │
                      │ • Input: features [1, 37, T]           │
                      │ • Input: lengths [1]                   │
                      │ • Output: encoded [1, D, T']           │
                      └─────────────────────────────────────────┘
                                         │
                                         ▼
              ┌─────────────────────────────────────────────────────────────┐
              │                  BEAM SEARCH DECODER                        │
              │                                                             │
              │  Word List & Trie:                                          │
              │  ┌─────────────────────────────────────┐                    │
              │  │ • words.txt (~153k English words)  │                    │
              │  │ • Build prefix trie for lexicon    │                    │
              │  │   constraints                      │                    │
              │  │ • Word frequency priors            │                    │
              │  └─────────────────────────────────────┘                    │
              │                                                             │
              │  For each encoder time step t:                              │
              │  ┌─────────────────────────────────────────────────────┐    │
              │  │ Active Beams → Step Model → New Beams              │    │
              │  │                                                     │    │
              │  │ ┌─────────────────────────────────────────────────┐ │    │
              │  │ │           STEP MODEL (ONNX)                     │ │    │
              │  │ │                                                 │ │    │
              │  │ │ • Load: rnnt_step_fp32.onnx                   │ │    │
              │  │ │ • Input: y_prev [N], h0 [L,N,H], c0 [L,N,H]   │ │    │
              │  │ │          enc_t [N, D]                         │ │    │
              │  │ │ • LSTM prediction + joint network             │ │    │
              │  │ │ • Output: logits [N, V=29]                    │ │    │
              │  │ │           h1 [L,N,H], c1 [L,N,H]              │ │    │
              │  │ └─────────────────────────────────────────────────┘ │    │
              │  │                                                     │    │
              │  │ Transitions:                                        │    │
              │  │ • Blank transition (no output)                     │    │
              │  │ • Character transitions (trie-constrained)         │    │
              │  │                                                     │    │
              │  │ Scoring: rnnt_logp + λ * lm_prior                  │    │
              │  │ λ = 0.4 (language model weight)                    │    │
              │  └─────────────────────────────────────────────────────┘    │
              │                                                             │
              │  Beam Management:                                           │
              │  • Beam Size: 16 active hypotheses                         │
              │  • Pruning: Top-8 chars per beam                           │
              │  • Early termination on word completion                    │
              └─────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
                              Final Predictions
                              ┌───────────────────────┐
                              │ [                     │
                              │   {word: "hello",     │
                              │    score: -2.34,      │
                              │    rnnt: -1.98},      │
                              │   {word: "help",      │
                              │    score: -3.12,      │
                              │    rnnt: -2.87},      │
                              │   ...                 │
                              │ ]                     │
                              └───────────────────────┘

┌───────────────────────────────────────────────────────────────────────────────────────────────┐
│                                  VOCABULARY SYSTEM                                             │
└───────────────────────────────────────────────────────────────────────────────────────────────┘

Character Vocabulary (29 tokens):
┌────────────────────────────────────────────────────────────┐
│ ["<blank>", "'", "a", "b", "c", "d", ..., "z", "<unk>"]   │
│                                                            │
│ • blank_id = 0  (RNNT blank transitions)                  │
│ • unk_id = 28   (out-of-vocabulary)                       │
│ • Regular chars: a-z, apostrophe                          │
└────────────────────────────────────────────────────────────┘

Word Vocabulary (~153k words):
┌────────────────────────────────────────────────────────────┐
│ words.txt: ["a", "aa", "aaa", ..., "zzzs"]                │
│                                                            │
│ Trie Structure:                                            │
│ • Prefix tree for lexicon constraints                     │
│ • Terminal nodes marked with word_id                      │
│ • Enables beam search vocabulary pruning                  │
└────────────────────────────────────────────────────────────┘

Word Frequency Priors:
┌────────────────────────────────────────────────────────────┐
│ swipe_vocabulary.json:                                     │
│ {                                                          │
│   "the": 0.0234,    (high frequency)                      │
│   "hello": 0.0012,  (medium frequency)                    │
│   "zymurgy": 1e-8   (low frequency)                       │
│ }                                                          │
│                                                            │
│ • Bias beam search toward common words                    │
│ • Combined with neural model scores                       │
└────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 CRITICAL REQUIREMENTS                                          │
└───────────────────────────────────────────────────────────────────────────────────────────────┘

✅ Coordinate Space: [-1, 1] (NOT [0, 1])
✅ Feature Order: 9 kinematic + 28 RBF keys (EXACT training match)
✅ Tensor I/O: Runtime detection (NO hardcoded names)
✅ QWERTY Layout: Key centers in [-1, 1] space
✅ RBF Sigma: 0.2 (Gaussian kernel width)
✅ Beam Parameters: beamSize=16, lmLambda=0.4, prunePerBeam=8

DOMAIN MISMATCH = FAILED PREDICTIONS!
    `;

    return {
        ascii: diagram,

        // Structured data representation
        pipeline: {
            training: {
                input: "JSONL gesture data",
                featurizer: "SwipeFeaturizer (37D: 9 kinematic + 28 RBF)",
                model: "RNNT (Conformer + LSTM + Joint)",
                loss: "RNNT Loss (alignment-free)",
                output: ".nemo/.ckpt checkpoint"
            },

            export: {
                encoder: {
                    script: "export_onnx.py",
                    input: "Conformer component",
                    quantization: "INT8 QDQ",
                    output: "encoder_int8_qdq.onnx"
                },
                step: {
                    script: "export_rnnt_step.py",
                    input: "LSTM + Joint components",
                    formats: ["ONNX FP32", "ExecuTorch PTE"],
                    output: "rnnt_step_fp32.onnx"
                }
            },

            inference: {
                feature_extraction: {
                    input: "Touch events (x, y, t)",
                    normalization: "[0,1] → [-1,1]",
                    features: "37D vectors",
                    output: "[1, 37, T] tensor"
                },
                encoder: {
                    model: "encoder_int8_qdq.onnx",
                    input: "features [1,37,T], lengths [1]",
                    output: "encoded [1,D,T']"
                },
                beam_search: {
                    lexicon: "words.txt (153k words)",
                    priors: "swipe_vocabulary.json",
                    step_model: "rnnt_step_fp32.onnx",
                    parameters: {beamSize: 16, lmLambda: 0.4},
                    output: "Ranked word predictions"
                }
            }
        },

        // File dependencies with versions
        files: {
            training: {
                "train_transducer.py": "2025-09-16 04:40:10",
                "swipe_data_utils.py": "2025-09-16 05:20:36",
                "model_class.py": "Referenced in training"
            },
            export: {
                "export_onnx.py": "2025-09-16 05:20:36",
                "export_rnnt_step.py": "2025-09-16 05:20:36"
            },
            inference: {
                "swipe-onnx.html": "2025-09-16 09:10:32",
                "runtime_meta.json": "2025-09-16 09:10:32",
                "words.txt": "2025-09-16 05:20:36",
                "swipe_vocabulary.json": "Web demo vocabulary"
            }
        },

        // Critical parameters
        config: {
            model: {
                feat_in: 37,
                encoder: {d_model: 256, n_heads: 4, num_layers: 8},
                decoder: {pred_hidden: 320, pred_rnn_layers: 2},
                joint: {joint_hidden: 512}
            },
            features: {
                kinematic: 9,
                rbf_keys: 28,
                coordinate_range: "[-1, 1]",
                rbf_sigma: 0.2
            },
            vocabulary: {
                char_vocab_size: 29,
                word_vocab_size: "~153k",
                blank_id: 0,
                unk_id: 28
            },
            beam_search: {
                beam_size: 16,
                lm_lambda: 0.4,
                prune_per_beam: 8,
                max_symbols: 15
            }
        }
    };
}

// Export for use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { generateArchitectureDiagram };
} else if (typeof window !== 'undefined') {
    window.CleverKeysArchitecture = { generateArchitectureDiagram };
}

// Print diagram if run directly
if (typeof require !== 'undefined' && require.main === module) {
    console.log(generateArchitectureDiagram().ascii);
}