package ai.ondevice

import kotlin.math.exp

/**
 * Minimal dense-vector math for the on-device inference core.
 *
 * Everything here is allocation-light on purpose: a phone running inference
 * in a camera loop cannot afford a heavyweight tensor library, so the core
 * works on plain [FloatArray]s and small helper functions instead.
 */

/** Matrix-vector product: `out[j] = sum_i matrix[j][i] * vec[i]`. */
fun matVec(matrix: Array<FloatArray>, vec: FloatArray): FloatArray {
    require(matrix.isEmpty() || matrix[0].size == vec.size) {
        "matrix columns (${matrix.firstOrNull()?.size}) must match vector size (${vec.size})"
    }
    return FloatArray(matrix.size) { row ->
        var acc = 0f
        val w = matrix[row]
        for (i in vec.indices) acc += w[i] * vec[i]
        acc
    }
}

/** Element-wise bias add, in place for the caller's array. */
fun addBiasInPlace(values: FloatArray, bias: FloatArray): FloatArray {
    require(values.size == bias.size) { "bias size (${bias.size}) must match values size (${values.size})" }
    for (i in values.indices) values[i] += bias[i]
    return values
}

/** ReLU, in place. */
fun reluInPlace(values: FloatArray): FloatArray {
    for (i in values.indices) if (values[i] < 0f) values[i] = 0f
    return values
}

/** Numerically stable softmax: output sums to 1 and every entry is in (0, 1). */
fun softmax(logits: FloatArray): FloatArray {
    val peak = logits.maxOrNull() ?: 0f
    val exps = FloatArray(logits.size) { exp((logits[it] - peak).toDouble()).toFloat() }
    val total = exps.sum()
    require(total > 0f) { "softmax denominator must be positive" }
    for (i in exps.indices) exps[i] /= total
    return exps
}

/** Index of the largest entry; ties resolve to the lowest index. */
fun argmax(values: FloatArray): Int {
    require(values.isNotEmpty()) { "argmax needs a non-empty array" }
    var best = 0
    for (i in 1 until values.size) if (values[i] > values[best]) best = i
    return best
}

/** A single fully-connected layer: `y = Wx + b`. Weights are row-major, one row per output. */
class LinearLayer(val weights: Array<FloatArray>, val bias: FloatArray) {
    val inputSize: Int get() = if (weights.isEmpty()) 0 else weights[0].size
    val outputSize: Int get() = weights.size

    init {
        require(weights.isNotEmpty()) { "a layer needs at least one output row" }
        require(weights.all { it.size == inputSize }) { "all weight rows must share one input size" }
        require(bias.size == outputSize) { "bias size (${bias.size}) must match output size ($outputSize)" }
    }

    /** Forward pass without activation. */
    fun forward(input: FloatArray): FloatArray {
        require(input.size == inputSize) { "expected input size $inputSize, got ${input.size}" }
        return addBiasInPlace(matVec(weights, input), bias)
    }
}

/**
 * A small multi-layer perceptron: ReLU hidden layers, softmax output.
 * This is the model shape the Android app runs on-device — small enough
 * (tens of thousands of weights) to live in the APK and run in milliseconds.
 */
class Mlp(val layers: List<LinearLayer>) {
    val inputSize: Int get() = layers.first().inputSize
    val outputSize: Int get() = layers.last().outputSize

    init {
        require(layers.isNotEmpty()) { "an MLP needs at least one layer" }
        for (i in 1 until layers.size) {
            require(layers[i].inputSize == layers[i - 1].outputSize) {
                "layer $i input (${layers[i].inputSize}) must match layer ${i - 1} output (${layers[i - 1].outputSize})"
            }
        }
    }

    /** Raw logits (pre-softmax). */
    fun logits(input: FloatArray): FloatArray {
        var current = input
        for (i in layers.indices) {
            current = layers[i].forward(current)
            if (i < layers.size - 1) reluInPlace(current)
        }
        return current
    }

    /** Class probabilities. */
    fun probabilities(input: FloatArray): FloatArray = softmax(logits(input))

    /** Predicted class index. */
    fun predict(input: FloatArray): Int = argmax(probabilities(input))
}

/** Random weight init (Xavier/Glorot uniform) for training. */
fun xavierInit(rows: Int, cols: Int, rng: java.util.Random): Array<FloatArray> {
    val limit = kotlin.math.sqrt(6.0 / (rows + cols)).toFloat()
    return Array(rows) { FloatArray(cols) { (rng.nextFloat() * 2f - 1f) * limit } }
}

/** Maximum absolute value, used by the quantizer. */
fun maxAbs(values: FloatArray): Float {
    var peak = 0f
    for (v in values) {
        val a = if (v < 0f) -v else v
        if (a > peak) peak = a
    }
    return peak
}
