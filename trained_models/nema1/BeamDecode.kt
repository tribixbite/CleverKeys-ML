import com.facebook.executorch.Program
import com.facebook.executorch.Runner
import com.facebook.executorch.Tensor
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.ln

data class TrieNode(
    val children: HashMap<Int, TrieNode> = HashMap(),
    var isWord: Boolean = false,
    var wordId: Int = -1
)
data class Lexicon(
    val words: List<String>,
    val charToId: Map<Char, Int>,
    val trieRoot: TrieNode,
    val wordLogPrior: FloatArray? = null  // optional
)

fun buildTrie(words: List<String>, charToId: Map<Char, Int>): TrieNode {
    fun node() = TrieNode()
    val root = node()
    for ((wid, w) in words.withIndex()) {
        var cur = root
        var skip = false
        for (ch in w) {
            val cid = charToId[ch] ?: run { skip = true; break }
            cur = cur.children.getOrPut(cid) { node() }
        }
        if (!skip) { cur.isWord = true; cur.wordId = wid }
    }
    return root
}

class RNNTBeamDecoder(
    private val encProg: Program,     // encoder_quant_xnnpack.pte
    private val stepProg: Program,    // rnnt_step_fp32.pte
    private val L: Int,
    private val H: Int,
    private val D: Int,
    private val blankId: Int = 0
) {
    // Tensor builders
    private fun tensor(shape: LongArray, dtype: Tensor.DType): Tensor {
        val elSize = when (dtype) {
            Tensor.DType.FLOAT32, Tensor.DType.INT32 -> 4
            Tensor.DType.INT64 -> 8
            else -> throw IllegalArgumentException("dtype not supported")
        }
        val size = shape.fold(1L){a,b->a*b}.toInt()
        return Tensor.fromBlob(ByteBuffer.allocateDirect(elSize*size).order(ByteOrder.nativeOrder()), shape, dtype)
    }

    data class Beam(
        var yPrev: Long,
        var h: Tensor,   // (L,1,H) float32
        var c: Tensor,   // (L,1,H) float32
        var trie: TrieNode,
        var logp: Float,
        val chars: MutableList<Int> = ArrayList()
    )

    fun decodeWord(
        featuresBFT: FloatArray, F: Int, T: Int,
        lex: Lexicon,
        beamSize: Int = 16, prunePerBeam: Int = 6,
        maxSymbols: Int = 20, lmLambda: Float = 0.4f,
        returnTopK: Int = 5
    ): List<Triple<String, Float, Float>> { // (word, totalScore, rnntLogp)
        // 1) encoder run
        val encRunner = encProg.createRunner()
        val feats = tensor(longArrayOf(1, F.toLong(), T.toLong()), Tensor.DType.FLOAT32)
        feats.buffer.asFloatBuffer().put(featuresBFT)
        val lens = tensor(longArrayOf(1), Tensor.DType.INT32)
        lens.buffer.asIntBuffer().put(intArrayOf(T))
        encRunner.setInput("features_bft", feats).setInput("lengths", lens).run()
        val encBtf = encRunner.getOutput("encoded_btf")!!
        val dims = encBtf.shape
        val Tout = dims[1].toInt()
        val Denc = dims[2].toInt()
        require(Denc == D) { "Encoder D mismatch: expected $D got $Denc" }
        val encData = FloatArray(Tout * D)
        encBtf.buffer.asFloatBuffer().get(encData)

        // 2) init beams
        fun zerosLH(): Tensor = tensor(longArrayOf(L.toLong(), 1, H.toLong()), Tensor.DType.FLOAT32)
        var beams = mutableListOf(
            Beam(blankId.toLong(), zerosLH(), zerosLH(), lex.trieRoot, 0.0f)
        )

        val stepRunner = stepProg.createRunner()
        val enc_t = tensor(longArrayOf(beamSize.toLong(), D.toLong()), Tensor.DType.FLOAT32)

        // 3) time frames
        for (t in 0 until Tout) {
            for (s in 0 until maxSymbols) {
                // Take top N
                beams.sortByDescending { it.logp }
                val N = minOf(beamSize, beams.size)
                val act = beams.subList(0, N)

                // Stack y_prev, h0, c0, enc_t
                val yPrev = tensor(longArrayOf(N.toLong()), Tensor.DType.INT64)
                val ybuf = yPrev.buffer.asLongBuffer()
                val h0 = tensor(longArrayOf(L.toLong(), N.toLong(), H.toLong()), Tensor.DType.FLOAT32)
                val c0 = tensor(longArrayOf(L.toLong(), N.toLong(), H.toLong()), Tensor.DType.FLOAT32)
                val hbuf = h0.buffer.asFloatBuffer()
                val cbuf = c0.buffer.asFloatBuffer()
                // copy states
                for (i in 0 until N) {
                    ybuf.put(act[i].yPrev)
                    hbuf.put(act[i].h.buffer.asFloatBuffer().rewind().let { val tmp = FloatArray(L*H); it.get(tmp); tmp })
                    cbuf.put(act[i].c.buffer.asFloatBuffer().rewind().let { val tmp = FloatArray(L*H); it.get(tmp); tmp })
                }
                // tile enc_t
                val encSlice = FloatArray(D)
                System.arraycopy(encData, t*D, encSlice, 0, D)
                val encBuf = enc_t.buffer.asFloatBuffer()
                repeat(N) { encBuf.put(encSlice) }

                // Step run
                stepRunner.setInput("y_prev", yPrev).setInput("h0", h0).setInput("c0", c0).setInput("enc_t", enc_t).run()
                val logits = stepRunner.getOutput("logits")!!
                val h1 = stepRunner.getOutput("h1")!!
                val c1 = stepRunner.getOutput("c1")!!
                val V = logits.shape[1].toInt()
                val logRow = FloatArray(N * V); logits.buffer.asFloatBuffer().get(logRow)

                val next = ArrayList<Beam>(N * (1 + prunePerBeam))
                for (i in 0 until N) {
                    val base = i * V
                    // blank
                    val lpBlank = logRow[base + blankId]
                    next.add(
                        Beam(
                            blankId.toLong(),
                            sliceLC(h1, i, L, H),
                            sliceLC(c1, i, L, H),
                            act[i].trie,
                            act[i].logp + lpBlank,
                            ArrayList(act[i].chars)
                        )
                    )
                    // expand allowed children
                    val allowed = act[i].trie.children.keys
                    if (!allowed.isEmpty()) {
                        // score & top-k
                        val scored = allowed.map { cid -> cid to logRow[base + cid] }.sortedByDescending { it.second }
                        val toExp = scored.take(minOf(prunePerBeam, scored.size))
                        for ((cid, val) in toExp) {
                            next.add(
                                Beam(
                                    cid.toLong(),
                                    sliceLC(h1, i, L, H),
                                    sliceLC(c1, i, L, H),
                                    act[i].trie.children[cid]!!,
                                    act[i].logp + val,
                                    ArrayList<Int>().apply { addAll(act[i].chars); add(cid) }
                                )
                            )
                        }
                    }
                }
                next.sortByDescending { it.logp }
                beams = next.take(beamSize).toMutableList()
                if (beams[0].yPrev.toInt() == blankId) break
            }
        }

        // 4) collect terminal words
        data class Cand(val wid: Int, val word: String, val logp: Float, val logpLM: Float, val score: Float)
        val cands = ArrayList<Cand>()
        for (b in beams) {
            if (b.trie.isWord && b.trie.wordId >= 0) {
                val wid = b.trie.wordId
                val lm = lex.wordLogPrior?.get(wid) ?: 0.0f
                val score = b.logp + lmLambda * lm
                cands.add(Cand(wid, lex.words[wid], b.logp, lm, score))
            }
        }
        cands.sortByDescending { it.score }
        val seen = HashSet<Int>()
        val out = ArrayList<Triple<String, Float, Float>>()
        for (c in cands) {
            if (seen.contains(c.wid)) continue
            seen.add(c.wid)
            out.add(Triple(c.word, c.score, c.logp))
            if (out.size >= returnTopK) break
        }
        return out
    }

    // slice beam i from (L,N,H) -> (L,1,H)
    private fun sliceLC(all: Tensor, i: Int, L: Int, H: Int): Tensor {
        val out = tensor(longArrayOf(L.toLong(), 1, H.toLong()), Tensor.DType.FLOAT32)
        val src = all.buffer.asFloatBuffer()
        val dst = out.buffer.asFloatBuffer()
        val N = all.shape[1].toInt()
        val temp = FloatArray(H)
        for (l in 0 until L) {
            val srcBase = (l*N + i) * H
            src.position(srcBase); src.get(temp, 0, H)
            dst.put(temp)
        }
        return out
    }
}




/* usage

// Load programs (ExecuTorch)
val encProg = Program(BinaryProgram.fromPath(context, "encoder_quant_xnnpack.pte"))
val stepProg = Program(BinaryProgram.fromPath(context, "rnnt_step_fp32.pte"))

// Build lexicon (once)
val words: List<String> = loadWordsFromAssets(...)
val charToId = mapOf( 'a' to 0, /* ... */, '\'' to 26 )
val trieRoot = buildTrie(words, charToId)
val lex = Lexicon(words, charToId, trieRoot, wordLogPrior = loadLogPriors(...))

// Instantiate decoder with model dims
val L = 2; val H = 320; val D = /* encoder output dim */
val decoder = RNNTBeamDecoder(encProg, stepProg, L, H, D, blankId = 0)

// For each swipe:
val featuresBFT: FloatArray = featurizeSwipeToBFT(swTrace) // (1,F,T) flatten
val top = decoder.decodeWord(featuresBFT, F=37, T=traceLen, lex, beamSize=16, prunePerBeam=6, maxSymbols=20, lmLambda=0.4f, returnTopK=5)
*/ top: List<Triple<word, totalScore, rnntLogp>>

