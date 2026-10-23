package ai.ondevice

import kotlin.math.roundToInt

/**
 * Symmetric per-tensor int8 quantization — the trick that makes a model
 * phone-friendly: weights shrink 4x (float32 -> int8) and the hot loop runs
 * on integer arithmetic instead of floats.
 *
 * Symmetric means the zero point is fixed at 0, which keeps the integer
 * math simple: `real ~= quantized * scale`, and a dot product becomes
 * `scaleW * scaleX * sum(qW * qX)` computed in [Int] before one final
 * float multiply. Accuracy loss on the bundled shape model is under 2
 * points (measured by the demo benchmark), which is the usual trade for
 * a 4x smaller model.
 */

/** Scale for one symmetric-quantized tensor: `real ~= q * scale`, `q` in [-127, 127]. */
fun symmetricScale(values: FloatArray): Float {
    val peak = maxAbs(values)
    if (peak == 0f) return 1f // degenerate all-zero tensor: any scale dequantizes back to 0
    return peak / 127f
}

/** Quantize one tensor to signed int8 bytes. */
fun quantizeSymmetric(values: FloatArray): Pair<ByteArray, Float> {
    val scale = symmetricScale(values)
    val out = ByteArray(values.size) { i ->
        (values[i] / scale).roundToInt().coerceIn(-127, 127).toByte()
    }
    return out to scale
}

/** Dequantize one tensor. */
fun dequantizeSymmetric(quantized: ByteArray, scale: Float): FloatArray =
    FloatArray(quantized.size) { quantized[it].toInt() * scale }

/** One quantized fully-connected layer: int8 weights, float bias, int matmul. */
class QuantizedLinearLayer(
    val weightsQ: ByteArray,
    val rows: Int,
    val cols: Int,
    val weightScale: Float,
    val bias: FloatArray,
) {
    init {
        require(weightsQ.size == rows * cols) { "quantized weights must be rows*cols bytes" }
        require(bias.size == rows) { "bias size (${bias.size}) must match rows ($rows)" }
    }

    /**
     * Forward pass with integer accumulation. The input is quantized on the
     * fly (one pass to find its scale, one to quantize), then each output is
     * `weightScale * inputScale * sum(qW * qX) + bias`, all in [Int] until
     * the final multiply — exactly how a phone DSP/NPU kernel would do it.
     */
    fun forward(input: FloatArray): FloatArray {
        require(input.size == cols) { "expected input size $cols, got ${input.size}" }
        val inputScale = symmetricScale(input)
        val inputQ = IntArray(cols) { (input[it] / inputScale).roundToInt().coerceIn(-127, 127) }
        val combinedScale = weightScale * inputScale
        return FloatArray(rows) { row ->
            var acc = 0
            val base = row * cols
            for (i in 0 until cols) acc += weightsQ[base + i].toInt() * inputQ[i]
            acc * combinedScale + bias[row]
        }
    }
}

/** An MLP whose layers all run the int8 path. Hidden activations are ReLU'd in float. */
class QuantizedMlp(val layers: List<QuantizedLinearLayer>) {
    val inputSize: Int get() = layers.first().cols
    val outputSize: Int get() = layers.last().rows

    init {
        require(layers.isNotEmpty()) { "a quantized MLP needs at least one layer" }
        for (i in 1 until layers.size) {
            require(layers[i].cols == layers[i - 1].rows) {
                "layer $i input (${layers[i].cols}) must match layer ${i - 1} output (${layers[i - 1].rows})"
            }
        }
    }

    fun logits(input: FloatArray): FloatArray {
        var current = input
        for (i in layers.indices) {
            current = layers[i].forward(current)
            if (i < layers.size - 1) reluInPlace(current)
        }
        return current
    }

    fun probabilities(input: FloatArray): FloatArray = softmax(logits(input))

    fun predict(input: FloatArray): Int = argmax(probabilities(input))
}

/** Convert a float MLP to its quantized twin, layer by layer. */
fun Mlp.toQuantized(): QuantizedMlp =
    QuantizedMlp(
        layers.map { layer ->
            val flat = FloatArray(layer.outputSize * layer.inputSize) { idx ->
                layer.weights[idx / layer.inputSize][idx % layer.inputSize]
            }
            val (weightsQ, scale) = quantizeSymmetric(flat)
            QuantizedLinearLayer(weightsQ, layer.outputSize, layer.inputSize, scale, layer.bias.copyOf())
        },
    )

/** Model size in bytes for the float vs int8 weights (biases stay float in both). */
fun Mlp.weightBytes(): Long = layers.sumOf { it.outputSize.toLong() * it.inputSize * 4L }

fun QuantizedMlp.weightBytes(): Long = layers.sumOf { it.rows.toLong() * it.cols }
