# CleverKeys Vocabulary System

This document describes the robust vocabulary management system implemented for CleverKeys, ensuring consistent behavior across Web and Android deployments while preventing illegal token expansions during beam search.

## Overview

The system addresses key issues:
- **Sync Problem**: Web and Android using different vocabulary mappings
- **Illegal Expansions**: Beam search expanding to `<unk>` and `<blank>` tokens
- **Hardcoded IDs**: Assumptions about `blank_id=0` that could break
- **Character Inconsistency**: Curly vs straight apostrophes, mixed case

## Architecture

### 1. Runtime Metadata (`runtime_meta.json`)

Central source of truth generated from the character-level vocabulary file:

```json
{
  "tokens": ["<blank>", "'", "a", "b", ..., "z", "<unk>"],
  "blank_id": 0,
  "unk_id": 28,
  "char_to_id": {"'": 1, "a": 2, ..., "z": 27},
  "id_to_char": {"1": "'", "2": "a", ..., "27": "z"},
  "vocab_size": 29,
  "allowed_chars": ["'", "a", "b", ..., "z"]
}
```

### 2. Vocabulary Filtering

Dictionary words are filtered to only include valid characters (`a-z` and `'`):

- ✅ `"hello"` → allowed
- ✅ `"don't"` → allowed
- ❌ `"hello123"` → filtered out (contains digits)
- ❌ `"café"` → filtered out (contains accented character)

### 3. Trie Building

Tries are built **only** from filtered words using character IDs, ensuring beam search never encounters invalid tokens.

## Files and Components

### Core Scripts

#### `scripts/make_runtime_meta.py`
Generates runtime metadata from vocabulary file:
```bash
python scripts/make_runtime_meta.py trained_models/data/vocab.txt --output exports/runtime_meta.json --pretty
```

#### `scripts/vocab_utils.py`
Vocabulary filtering and validation utilities:
```bash
# Normalize a word
python scripts/vocab_utils.py normalize --word "DON'T"

# Filter dictionary
python scripts/vocab_utils.py filter --dictionary vocab/final_vocab.txt --meta exports/runtime_meta.json --output filtered_vocab.txt

# Validate coverage
python scripts/vocab_utils.py validate --dictionary vocab/final_vocab.txt --meta exports/runtime_meta.json
```

### Export Scripts

#### `trained_models/nema1/export_rnnt_step.py`
Updated to derive `blank_id` from vocabulary:
```bash
# Basic usage (default blank_id=0)
python export_rnnt_step.py --nemo_model model.nemo

# With vocabulary (derives blank_id automatically)
python export_rnnt_step.py --nemo_model model.nemo --vocab trained_models/data/vocab.txt
```

### Web Integration

#### `web-demo/vocab-meta-utils.js`
Web utilities for metadata loading and trie building:
```javascript
// Load runtime metadata
const meta = await VocabMetaUtils.loadVocabMeta("runtime_meta.json");

// Create filtered lexicon
const lexicon = VocabMetaUtils.createLexicon(words, meta);

// Use in beam search with derived blank_id
const results = await rnntBeamSearchWord(
    encoder, step, features, F, T, L, H, D, lexicon,
    { blankId: meta.blankId, beamSize: 16, ... }
);
```

### Android Integration

#### `android/VocabMetaUtils.kt`
Kotlin utilities for Android deployment:
```kotlin
// Load runtime metadata
val meta = VocabMetaUtils.loadVocabMetaFromAssets(context, "runtime_meta.json")

// Create filtered lexicon
val lexicon = VocabMetaUtils.createLexicon(words, meta)

// Use in decoder with derived blank_id
val decoder = RNNTBeamDecoder(encProg, stepProg, L, H, D, blankId = meta.blank_id)
```

## Usage Patterns

### 1. Build Process

Generate runtime metadata during export:
```bash
# Generate metadata
python scripts/make_runtime_meta.py trained_models/data/vocab.txt --output exports/runtime_meta.json

# Copy to deployment directories
cp exports/runtime_meta.json web-demo/
cp exports/runtime_meta.json android/src/main/assets/
```

### 2. Web Deployment

```javascript
// Load metadata and dictionary
const meta = await VocabMetaUtils.loadVocabMeta("runtime_meta.json");
const words = await loadDictionary("dictionary.txt");

// Create lexicon with filtering
const lexicon = VocabMetaUtils.createLexicon(words, meta);

// Beam search with derived parameters
const decoded = await rnntBeamSearchWord(encoder, step, features, F, T, L, H, D, lexicon, {
    blankId: meta.blankId,  // Derived from metadata
    beamSize: 16,
    prunePerBeam: 6,
    maxSymbols: 20,
    lmLambda: 0.4
});
```

### 3. Android Deployment

```kotlin
class SwipeDecoder(context: Context) {
    private val meta = VocabMetaUtils.loadVocabMetaFromAssets(context, "runtime_meta.json")
    private val words = loadDictionaryFromAssets(context, "dictionary.txt")
    private val lexicon = VocabMetaUtils.createLexicon(words, meta)

    private val decoder = RNNTBeamDecoder(
        encProg, stepProg, L, H, D,
        blankId = meta.blank_id
    )

    fun decode(features: FloatArray): List<String> {
        return decoder.decode(features, lexicon)
    }
}
```

## Key Benefits

### 1. Prevents Illegal Expansions
- Trie built only from valid characters (`a-z`, `'`)
- Beam search never expands to `<unk>` or `<blank>`
- Higher accuracy through legal-only token sequences

### 2. Platform Consistency
- Web and Android use identical character mappings
- Runtime metadata ensures synchronization
- No hardcoded assumptions about token IDs

### 3. Apostrophe Handling
- Normalizes curly (`'`) to straight (`'`) apostrophes
- Handles words like `"it's"`, `"don't"`, `"won't"`
- Consistent typography across platforms

### 4. High Coverage
- Filters only non-alphabetic characters
- Maintains 95%+ dictionary coverage
- Drops edge cases (numbers, accents, hyphens)

### 5. Maintainability
- Single source of truth for vocabulary mapping
- Programmatic derivation prevents manual errors
- Easy to update vocabulary without code changes

## Validation

Check vocabulary coverage:
```bash
python scripts/vocab_utils.py validate --dictionary vocab/final_vocab.txt --meta exports/runtime_meta.json
```

Expected output:
```json
{
  "total_words": 153000,
  "valid_words": 145000,
  "coverage_percentage": 94.8,
  "invalid_characters": [],
  "sample_valid_words": ["hello", "world", "don't", ...]
}
```

## Migration Guide

### For Existing Web Code
1. Include `vocab-meta-utils.js` in your HTML
2. Load `runtime_meta.json` instead of hardcoding mappings
3. Use `VocabMetaUtils.createLexicon()` for trie building
4. Pass `meta.blankId` to beam search decoder

### For Existing Android Code
1. Add `VocabMetaUtils.kt` to your project
2. Include `runtime_meta.json` in assets
3. Use `VocabMetaUtils.createLexicon()` for vocabulary
4. Update decoder constructor with `meta.blank_id`

### For Export Scripts
1. Add `--vocab` parameter to export commands
2. Remove hardcoded `blank_id=0` assumptions
3. Use derived IDs from vocabulary file

## Troubleshooting

### Low Coverage (< 90%)
Check for unexpected characters in dictionary:
```bash
python scripts/vocab_utils.py validate --dictionary your_dict.txt --meta runtime_meta.json
```

### Illegal Token Errors
Ensure trie uses filtered words:
```javascript
// ❌ Wrong - uses unfiltered words
const trie = buildTrie(rawWords, charToId);

// ✅ Correct - uses filtered words
const lexicon = VocabMetaUtils.createLexicon(rawWords, meta);
const trie = lexicon.trie;
```

### Blank ID Mismatches
Use derived blank_id:
```python
# ❌ Wrong - hardcoded
blank_id = 0

# ✅ Correct - derived
meta = load_runtime_meta("runtime_meta.json")
blank_id = meta["blank_id"]
```

This system ensures robust, consistent vocabulary handling across all CleverKeys deployments while maximizing accuracy through legal-only token expansions.