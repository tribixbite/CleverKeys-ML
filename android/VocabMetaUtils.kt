package com.cleverkeys.vocab

import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlinx.serialization.decodeFromString
import java.io.InputStream

/**
 * VocabMetaUtils.kt
 *
 * Android vocabulary metadata utilities for CleverKeys.
 * Provides runtime meta loading and vocabulary filtering to ensure
 * trie building only uses valid characters, preventing <unk> expansions.
 */

/**
 * Runtime metadata structure loaded from runtime_meta.json
 */
@Serializable
data class VocabMeta(
    val tokens: List<String>,
    val blank_id: Int,
    val unk_id: Int,
    val char_to_id: Map<String, Int>,
    val id_to_char: Map<String, String>,
    val vocab_size: Int,
    val allowed_chars: List<String>
) {
    // Convert to more convenient types for Kotlin
    val charToIdMap: Map<Char, Int> by lazy {
        char_to_id.mapKeys { it.key.single() }
    }

    val idToCharMap: Map<Int, Char> by lazy {
        id_to_char.mapKeys { it.key.toInt() }.mapValues { it.value.single() }
    }

    val allowedCharSet: Set<Char> by lazy {
        allowed_chars.map { it.single() }.toSet()
    }

    /**
     * Check if a character is valid for trie building
     */
    fun isValidChar(ch: Char): Boolean = allowedCharSet.contains(ch)

    /**
     * Get character ID for a valid character
     */
    fun getCharId(ch: Char): Int? = charToIdMap[ch]

    /**
     * Get character for an ID
     */
    fun getChar(id: Int): Char? = idToCharMap[id]
}

/**
 * Simple trie node for building vocabulary trie
 */
class TrieNode {
    val children = mutableMapOf<Int, TrieNode>()  // Use Int keys for character IDs
    var isEndOfWord = false
    var wordEnding = false  // Alternative naming for compatibility

    // Legacy version with Char keys for backward compatibility
    val charChildren = mutableMapOf<Char, TrieNode>()
}

/**
 * Vocabulary utilities object
 */
object VocabMetaUtils {

    /**
     * Load runtime metadata from JSON input stream
     */
    fun loadVocabMeta(inputStream: InputStream): VocabMeta {
        return inputStream.use { stream ->
            val jsonString = stream.bufferedReader().readText()
            Json.decodeFromString<VocabMeta>(jsonString)
        }
    }

    /**
     * Load runtime metadata from assets
     */
    fun loadVocabMetaFromAssets(context: android.content.Context, fileName: String): VocabMeta {
        return context.assets.open(fileName).use { inputStream ->
            loadVocabMeta(inputStream)
        }
    }

    /**
     * Normalize word for vocabulary consistency
     * - Convert to lowercase
     * - Replace curly apostrophes with straight apostrophes
     * - Trim whitespace
     */
    fun normalizeWord(word: String): String {
        return word.lowercase()
            .replace('\u2019', '\'')  // Replace curly apostrophe (') with straight (')
            .trim()
    }

    /**
     * Check if a word contains only allowed characters
     */
    fun isValidWord(word: String, meta: VocabMeta): Boolean {
        if (word.isEmpty()) return false
        return word.all { ch -> meta.isValidChar(ch) }
    }

    /**
     * Filter dictionary words to only include valid characters
     */
    fun filterDictionary(words: List<String>, meta: VocabMeta): List<String> {
        return words.mapNotNull { word ->
            val normalized = normalizeWord(word)
            if (normalized.isNotEmpty() && isValidWord(normalized, meta)) {
                normalized
            } else {
                null
            }
        }
    }

    /**
     * Build trie from filtered words using character-to-ID mapping
     */
    fun buildTrieFromWords(words: List<String>, meta: VocabMeta): TrieNode {
        val root = TrieNode()

        for (word in words) {
            val normalized = normalizeWord(word)
            if (!isValidWord(normalized, meta)) {
                continue  // Skip words with invalid characters
            }

            var current = root
            for (char in normalized) {
                val charId = meta.getCharId(char)
                if (charId == null) {
                    println("Warning: Character '$char' not found in charToId mapping")
                    break  // Skip this word if any character is invalid
                }

                if (!current.children.containsKey(charId)) {
                    current.children[charId] = TrieNode()
                }
                current = current.children[charId]!!
            }
            current.isEndOfWord = true
            current.wordEnding = true  // Set both for compatibility
        }

        return root
    }

    /**
     * Legacy trie builder that uses character keys instead of IDs
     * Kept for backward compatibility
     */
    fun buildTrieLegacy(words: List<String>, charToIdMap: Map<Char, Int>): TrieNode {
        val root = TrieNode()

        for (word in words) {
            var current = root
            for (char in word) {
                if (!current.charChildren.containsKey(char)) {
                    current.charChildren[char] = TrieNode()
                }
                current = current.charChildren[char]!!
            }
            current.isEndOfWord = true
            current.wordEnding = true
        }

        return root
    }

    /**
     * Create lexicon data class for beam search decoder
     */
    data class Lexicon(
        val words: List<String>,
        val charToId: Map<Char, Int>,
        val idToChar: Map<Int, Char>,
        val trie: TrieNode,
        val wordLogPrior: Map<String, Float>? = null,
        val meta: VocabMeta
    )

    /**
     * Create lexicon object for beam search decoder
     */
    fun createLexicon(
        words: List<String>,
        meta: VocabMeta,
        wordLogPrior: Map<String, Float>? = null
    ): Lexicon {
        // Filter words to only include valid characters
        val filteredWords = filterDictionary(words, meta)

        println("Vocabulary filtering: ${words.size} → ${filteredWords.size} words " +
               "(${String.format("%.1f", (filteredWords.size.toFloat() / words.size) * 100)}% coverage)")

        // Build trie from filtered words
        val trie = buildTrieFromWords(filteredWords, meta)

        return Lexicon(
            words = filteredWords,
            charToId = meta.charToIdMap,
            idToChar = meta.idToCharMap,
            trie = trie,
            wordLogPrior = wordLogPrior,
            meta = meta
        )
    }

    /**
     * Validate vocabulary coverage and provide statistics
     */
    data class VocabCoverageStats(
        val totalWords: Int,
        val validWords: Int,
        val coveragePercentage: Float,
        val invalidCharacters: List<Char>,
        val allowedCharacters: List<Char>,
        val sampleValidWords: List<String>,
        val sampleInvalidWords: List<String>
    )

    fun validateVocabularyCoverage(words: List<String>, meta: VocabMeta): VocabCoverageStats {
        val originalCount = words.size
        val filteredWords = filterDictionary(words, meta)
        val validCount = filteredWords.size
        val coverage = (validCount.toFloat() / originalCount) * 100

        // Find invalid characters
        val invalidChars = mutableSetOf<Char>()
        for (word in words) {
            val normalized = normalizeWord(word)
            for (char in normalized) {
                if (!meta.isValidChar(char)) {
                    invalidChars.add(char)
                }
            }
        }

        return VocabCoverageStats(
            totalWords = originalCount,
            validWords = validCount,
            coveragePercentage = coverage,
            invalidCharacters = invalidChars.sorted(),
            allowedCharacters = meta.allowedCharSet.sorted(),
            sampleValidWords = filteredWords.take(10),
            sampleInvalidWords = words.filter { w ->
                !isValidWord(normalizeWord(w), meta)
            }.take(10)
        )
    }
}