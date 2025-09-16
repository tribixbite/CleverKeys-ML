package com.cleverkeys.beam

import android.content.Context
import com.facebook.executorch.BinaryProgram
import com.facebook.executorch.Program
import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.json.JSONObject

/**
 * LexiconLoader.kt
 *
 * Android Kotlin implementation for ExecuTorch-based lexicon beam search.
 * Loads meta/words/priors from assets/, builds trie, runs beam via RNNTBeamDecoder.
 */

data class Meta(
    val blank_id: Int,
    val unk_id: Int,
    val char_to_id: Map<String, Int>
)

data class Lexicon(
    val words: List<String>,
    val charToId: Map<Char, Int>,
    val trieRoot: TrieNode,
    val wordLogPrior: FloatArray?
)

class TrieNode {
    val children = mutableMapOf<Int, TrieNode>()
    var isWord = false
    var wordId = -1
}

/**
 * Load text file from Android assets
 */
fun loadAssetText(ctx: Context, path: String): String =
    ctx.assets.open(path).bufferedReader().use { it.readText() }

/**
 * Load runtime metadata from JSON in assets
 */
fun loadMeta(ctx: Context, path: String): Meta {
    val j = JSONObject(loadAssetText(ctx, path))
    val map = mutableMapOf<String, Int>()
    val ct = j.getJSONObject("char_to_id")
    val it = ct.keys()
    while (it.hasNext()) {
        val k = it.next()
        map[k] = ct.getInt(k)
    }
    return Meta(j.getInt("blank_id"), j.getInt("unk_id"), map)
}

/**
 * Load word list from text file in assets
 */
fun loadWords(ctx: Context, path: String): List<String> =
    loadAssetText(ctx, path).split(Regex("\\r?\\n")).filter { it.isNotBlank() }

/**
 * Normalize word for consistent processing
 * - Convert to lowercase
 * - Replace curly apostrophes with straight apostrophes
 */
fun normalizeWord(s: String): String = s.lowercase().replace(''', '\'')

/**
 * Build trie from word list using character-to-ID mapping
 * Only includes words with valid characters (a-z and apostrophe)
 */
fun buildTrie(words: List<String>, charToId: Map<Char, Int>): TrieNode {
    fun node() = TrieNode()
    val root = node()
    var kept = 0

    for ((wid, w0) in words.withIndex()) {
        val w = normalizeWord(w0)
        if (!w.all { charToId.containsKey(it) }) continue

        var cur = root
        for (ch in w) {
            val cid = charToId[ch]!!
            cur = cur.children.getOrPut(cid) { node() }
        }
        cur.isWord = true
        cur.wordId = wid
        kept++
    }

    println("Trie built: ${words.size} words -> $kept kept (${(kept.toFloat() / words.size * 100).toInt()}% coverage)")
    return root
}

/**
 * Load word log priors from binary float32 file in assets
 * Returns null if path is null
 */
fun loadPriorsF32(ctx: Context, path: String?): FloatArray? {
    if (path == null) return null
    val bytes = ctx.assets.open(path).readBytes()
    val bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
    val arr = FloatArray(bytes.size / 4)
    bb.asFloatBuffer().get(arr)
    return arr
}

/**
 * Complete lexicon loading helper
 */
fun loadLexicon(
    ctx: Context,
    metaPath: String = "models/runtime_meta.json",
    wordsPath: String = "lexicon/words.txt",
    priorsPath: String? = "lexicon/word_priors.f32"
): Triple<Meta, Lexicon, Int> {
    val meta = loadMeta(ctx, metaPath)
    val words = loadWords(ctx, wordsPath)
    val charToId = meta.char_to_id.mapKeys { it.key.single() }
    val trieRoot = buildTrie(words, charToId)
    val priors = loadPriorsF32(ctx, priorsPath)
    val lexicon = Lexicon(words, charToId, trieRoot, priors)

    return Triple(meta, lexicon, meta.blank_id)
}

/**
 * Usage example:
 *
 * ```kotlin
 * // Load lexicon and models
 * val (meta, lexicon, blankId) = loadLexicon(context)
 *
 * // Load ExecuTorch programs
 * val encProg = Program(BinaryProgram.fromPath(context, "models/encoder_quant_xnnpack.pte"))
 * val stepProg = Program(BinaryProgram.fromPath(context, "models/rnnt_step_fp32.pte"))
 *
 * // Create decoder
 * val decoder = RNNTBeamDecoder(
 *     encProg = encProg,
 *     stepProg = stepProg,
 *     L = 2,
 *     H = 320,
 *     D = ENC_OUT_DIM,  // encoder output dimension
 *     blankId = blankId
 * )
 *
 * // Decode features
 * val featuresBFT: FloatArray = ... // shape [1*F*T] flattened
 * val results = decoder.decodeWord(
 *     featuresBFT = featuresBFT,
 *     F = 37,
 *     T = traceLen,
 *     lexicon = lexicon,
 *     beamSize = 16,
 *     prunePerBeam = 6,
 *     maxSymbols = 20,
 *     lmLambda = 0.4f,
 *     returnTopK = 5
 * )
 * // results: List<Triple<String, Float, Float>> = [(word, totalScore, rnntLogp), ...]
 * ```
 */