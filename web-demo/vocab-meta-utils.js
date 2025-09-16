/**
 * vocab-meta-utils.js
 *
 * Web vocabulary metadata utilities for CleverKeys.
 * Provides runtime meta loading and vocabulary filtering to ensure
 * trie building only uses valid characters, preventing <unk> expansions.
 */

/**
 * Runtime metadata structure loaded from runtime_meta.json
 */
class VocabMeta {
    constructor(data) {
        this.tokens = data.tokens || [];
        this.blankId = data.blank_id || 0;
        this.unkId = data.unk_id || 28;
        this.charToId = new Map(Object.entries(data.char_to_id || {}));
        this.idToChar = new Map(
            Object.entries(data.id_to_char || {}).map(([k, v]) => [parseInt(k), v])
        );
        this.vocabSize = data.vocab_size || this.tokens.length;
        this.allowedChars = new Set(data.allowed_chars || []);
    }

    /**
     * Check if a character is valid for trie building
     */
    isValidChar(ch) {
        return this.allowedChars.has(ch);
    }

    /**
     * Get character ID for a valid character
     */
    getCharId(ch) {
        return this.charToId.get(ch);
    }

    /**
     * Get character for an ID
     */
    getChar(id) {
        return this.idToChar.get(id);
    }
}

/**
 * Load runtime metadata from JSON URL
 */
async function loadVocabMeta(url) {
    try {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`Failed to fetch meta: ${response.status} ${response.statusText}`);
        }
        const data = await response.json();
        return new VocabMeta(data);
    } catch (error) {
        console.error('Failed to load vocabulary metadata:', error);
        throw error;
    }
}

/**
 * Normalize word for vocabulary consistency
 * - Convert to lowercase
 * - Replace curly apostrophes with straight apostrophes
 * - Trim whitespace
 */
function normalizeWord(word) {
    if (!word) return word;

    return word
        .toLowerCase()
        .replace(/\u2019/g, "'")  // Replace curly apostrophe (') with straight (')
        .trim();
}

/**
 * Check if a word contains only allowed characters
 */
function isValidWord(word, meta) {
    if (!word) return false;
    return [...word].every(ch => meta.isValidChar(ch));
}

/**
 * Filter dictionary words to only include valid characters
 */
function filterDictionary(words, meta) {
    const validWords = [];

    for (const word of words) {
        const normalized = normalizeWord(word);
        if (normalized && isValidWord(normalized, meta)) {
            validWords.push(normalized);
        }
    }

    return validWords;
}

/**
 * Simple trie node for building vocabulary trie
 */
class TrieNode {
    constructor() {
        this.children = new Map();
        this.isEndOfWord = false;
        this.wordEnding = false;  // Alternative naming for compatibility
    }
}

/**
 * Build trie from filtered words using character-to-ID mapping
 */
function buildTrieFromWords(words, meta) {
    const root = new TrieNode();

    for (const word of words) {
        const normalized = normalizeWord(word);
        if (!isValidWord(normalized, meta)) {
            continue;  // Skip words with invalid characters
        }

        let current = root;
        for (const char of normalized) {
            const charId = meta.getCharId(char);
            if (charId === undefined) {
                console.warn(`Character '${char}' not found in charToId mapping`);
                break;  // Skip this word if any character is invalid
            }

            if (!current.children.has(charId)) {
                current.children.set(charId, new TrieNode());
            }
            current = current.children.get(charId);
        }
        current.isEndOfWord = true;
        current.wordEnding = true;  // Set both for compatibility
    }

    return root;
}

/**
 * Legacy trie builder that uses character keys instead of IDs
 * Kept for backward compatibility
 */
function buildTrieLegacy(words, charToIdMap) {
    const root = new TrieNode();

    for (const word of words) {
        let current = root;
        for (const char of word) {
            if (!current.children.has(char)) {
                current.children.set(char, new TrieNode());
            }
            current = current.children.get(char);
        }
        current.isEndOfWord = true;
        current.wordEnding = true;
    }

    return root;
}

/**
 * Create lexicon object for beam search decoder
 */
function createLexicon(words, meta, wordLogPrior = null) {
    // Filter words to only include valid characters
    const filteredWords = filterDictionary(words, meta);

    console.log(`Vocabulary filtering: ${words.length} → ${filteredWords.length} words (${((filteredWords.length / words.length) * 100).toFixed(1)}% coverage)`);

    // Build trie from filtered words
    const trie = buildTrieFromWords(filteredWords, meta);

    return {
        words: filteredWords,
        charToId: meta.charToId,
        idToChar: Object.fromEntries(meta.idToChar.entries()),
        trie: trie,
        wordLogPrior: wordLogPrior,
        meta: meta  // Include meta for easy access
    };
}

/**
 * Validate vocabulary coverage and provide statistics
 */
function validateVocabularyCoverage(words, meta) {
    const originalCount = words.length;
    const filteredWords = filterDictionary(words, meta);
    const validCount = filteredWords.length;
    const coverage = (validCount / originalCount) * 100;

    // Find invalid characters
    const invalidChars = new Set();
    for (const word of words) {
        const normalized = normalizeWord(word);
        for (const char of normalized) {
            if (!meta.isValidChar(char)) {
                invalidChars.add(char);
            }
        }
    }

    return {
        totalWords: originalCount,
        validWords: validCount,
        coveragePercentage: coverage,
        invalidCharacters: Array.from(invalidChars).sort(),
        allowedCharacters: Array.from(meta.allowedChars).sort(),
        sampleValidWords: filteredWords.slice(0, 10),
        sampleInvalidWords: words.filter(w => !isValidWord(normalizeWord(w), meta)).slice(0, 10)
    };
}

// Export functions for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    // Node.js environment
    module.exports = {
        VocabMeta,
        loadVocabMeta,
        normalizeWord,
        isValidWord,
        filterDictionary,
        buildTrieFromWords,
        buildTrieLegacy,
        createLexicon,
        validateVocabularyCoverage,
        TrieNode
    };
} else {
    // Browser environment - attach to window
    window.VocabMetaUtils = {
        VocabMeta,
        loadVocabMeta,
        normalizeWord,
        isValidWord,
        filterDictionary,
        buildTrieFromWords,
        buildTrieLegacy,
        createLexicon,
        validateVocabularyCoverage,
        TrieNode
    };
}