import ai.onnxruntime.* // only if you also test ONNX; for ExecuTorch:
import com.facebook.executorch.BinaryProgram
import com.facebook.executorch.Program
import com.facebook.executorch.Runner
import com.facebook.executorch.Tensor
import java.nio.ByteBuffer
import java.nio.ByteOrder

// Notes Uses ExecuTorch Program + Runner API for both models. For best perf, reuse Runner and pre-allocate tensors for loop. You can pool buffers to avoid GC churn.

class RNNTRealtime(
    private val encoderProg: Program, // from encoder_quant_xnnpack.pte
    private val stepProg: Program,    // from rnnt_step_fp32.pte
    private val L: Int,
    private val H: Int,
    private val blankId: Long = 0L,
    private val maxSymbols: Int = 15
) {
    private fun tensorFloat(shape: LongArray) =
        Tensor.fromBlob(ByteBuffer.allocateDirect(4 * shape.reduce { a,b -> a*b }.toInt())
            .order(ByteOrder.nativeOrder()), shape, Tensor.DType.FLOAT32)

    private fun tensorInt32(shape: LongArray) =
        Tensor.fromBlob(ByteBuffer.allocateDirect(4 * shape.reduce { a,b -> a*b }.toInt())
            .order(ByteOrder.nativeOrder()), shape, Tensor.DType.INT32)

    private fun tensorInt64(shape: LongArray) =
        Tensor.fromBlob(ByteBuffer.allocateDirect(8 * shape.reduce { a,b -> a*b }.toInt())
            .order(ByteOrder.nativeOrder()), shape, Tensor.DType.INT64)

    fun greedyDecode(featuresBFT: FloatArray, B: Int, F: Int, T: Int, lengths: IntArray): List<IntArray> {
        // 1) Run encoder: inputs = features_bft (B,F,T), lengths (B)
        val encRunner: Runner = encoderProg.createRunner()
        val feats = tensorFloat(longArrayOf(B.toLong(), F.toLong(), T.toLong()))
        feats.buffer.asFloatBuffer().put(featuresBFT)
        val lens = tensorInt32(longArrayOf(B.toLong()))
        lens.buffer.asIntBuffer().put(lengths)

        encRunner
            .setInput("features_bft", feats)
            .setInput("lengths", lens)
            .run()
        val encBtf = encRunner.getOutput("encoded_btf")!! // (B,T_out,D) float32
        val encDims = encBtf.shape // [B, T_out, D]
        val Bout = encDims[0].toInt()
        val Tout = encDims[1].toInt()
        val D = encDims[2].toInt()
        val encData = FloatArray(Bout * Tout * D)
        encBtf.buffer.asFloatBuffer().get(encData)

        // 2) Initialize decoder state
        var h = tensorFloat(longArrayOf(L.toLong(), B.toLong(), H.toLong()))
        var c = tensorFloat(longArrayOf(L.toLong(), B.toLong(), H.toLong()))
        val yPrev = tensorInt64(longArrayOf(B.toLong()))
        val yPrevArr = LongArray(B) { blankId }
        yPrev.buffer.asLongBuffer().put(yPrevArr)

        val outSeq = ArrayList<IntArray>(B)
        repeat(B) { outSeq.add(IntArray(0)) }

        // 3) Loop time frames
        val stepRunner: Runner = stepProg.createRunner()
        val enc_t_tensor = tensorFloat(longArrayOf(B.toLong(), D.toLong()))
        val enc_t_buf = enc_t_tensor.buffer.asFloatBuffer()

        for (t in 0 until Tout) {
            // Slice enc_t (B,D)
            enc_t_buf.clear()
            for (b in 0 until B) {
                val base = (b * Tout + t) * D
                enc_t_buf.put(encData, base, D)
            }

            // Inner RNNT loop
            for (s in 0 until maxSymbols) {
                stepRunner
                    .setInput("y_prev", yPrev)
                    .setInput("h0", h)
                    .setInput("c0", c)
                    .setInput("enc_t", enc_t_tensor)
                    .run()

                val logits = stepRunner.getOutput("logits")!! // (B,V)
                val h1 = stepRunner.getOutput("h1")!!
                val c1 = stepRunner.getOutput("c1")!!

                val V = logits.shape[1].toInt()
                val logData = FloatArray(B * V)
                logits.buffer.asFloatBuffer().get(logData)

                // Argmax + append non-blank
                val nextIds = LongArray(B) { blankId }
                var anyNonBlank = false
                for (b in 0 until B) {
                    var best = 0
                    var bestVal = Float.NEGATIVE_INFINITY
                    val rowBase = b * V
                    for (v in 0 until V) {
                        val vval = logData[rowBase + v]
                        if (vval > bestVal) { bestVal = vval; best = v }
                    }
                    nextIds[b] = best.toLong()
                    if (best != blankId.toInt()) {
                        // append
                        val prev = outSeq[b]
                        outSeq[b] = prev + best
                        anyNonBlank = true
                    }
                }
                // Update state and token
                h = h1; c = c1
                yPrev.buffer.asLongBuffer().rewind()
                yPrev.buffer.asLongBuffer().put(nextIds)

                if (!anyNonBlank) break
            }
        }
        return outSeq
    }
}
