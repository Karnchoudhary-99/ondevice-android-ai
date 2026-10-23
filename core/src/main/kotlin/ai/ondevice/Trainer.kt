package ai.ondevice

/**
 * Plain mini-batch SGD trainer for the core MLP (cross-entropy loss,
 * softmax output, ReLU hidden layers). Training runs on the JVM (demo) or
 * ahead of time; the phone only ever runs inference — that split is the
 * whole point of on-device AI: train once, ship the weights, infer offline.
 *
 * Backprop is derived by hand here, not auto-diff: for softmax + cross
 * entropy the output delta is simply `probs - oneHot`, and ReLU passes
 * the gradient through only where the pre-activation was positive.
 */

/** One epoch's training loss (mean cross-entropy). */
data class TrainResult(val mlp: Mlp, val epochLosses: List<Float>)

/**
 * Train an MLP with [hidden] units per hidden layer ([hiddenLayers] of them).
 * Fully deterministic for a fixed [seed]: same seed, same weights.
 */
fun trainMlp(
    xs: List<FloatArray>,
    ys: IntArray,
    labels: Int,
    hidden: Int = 64,
    hiddenLayers: Int = 1,
    epochs: Int = 60,
    learningRate: Float = 0.1f,
    batchSize: Int = 32,
    seed: Long = 42L,
): TrainResult {
    require(xs.isNotEmpty()) { "training set must not be empty" }
    require(xs.size == ys.size) { "inputs and labels must align" }
    val inputSize = xs[0].size
    val rng = java.util.Random(seed)

    val dims = mutableListOf(inputSize)
    repeat(hiddenLayers) { dims.add(hidden) }
    dims.add(labels)
    val layers = (0 until dims.size - 1).map { i ->
        LinearLayer(xavierInit(dims[i + 1], dims[i], rng), FloatArray(dims[i + 1]))
    }
    val mlp = Mlp(layers)
    val losses = ArrayList<Float>()

    repeat(epochs) {
        var totalLoss = 0f
        val order = xs.indices.shuffled(rng)
        var start = 0
        while (start < order.size) {
            val batch = order.subList(start, minOf(start + batchSize, order.size))
            // Gradient accumulators, zeroed per batch.
            val gradW = layers.map { Array(it.outputSize) { FloatArray(it.inputSize) } }
            val gradB = layers.map { FloatArray(it.outputSize) }
            for (idx in batch) {
                val input = xs[idx]
                // Forward pass, keeping pre-activations for the ReLU backward.
                val preActivations = ArrayList<FloatArray>()
                val activations = ArrayList<FloatArray>()
                var current = input
                activations.add(current)
                for (i in layers.indices) {
                    val pre = layers[i].forward(current)
                    preActivations.add(pre)
                    current = if (i < layers.size - 1) reluInPlace(pre.copyOf()) else pre
                    activations.add(current)
                }
                val logits = current
                val probs = softmax(logits)
                val target = ys[idx]
                totalLoss += -kotlin.math.ln(probs[target].toDouble().coerceAtLeast(1e-12)).toFloat()
                // Output delta: probs - oneHot.
                var delta = FloatArray(labels) { j -> probs[j] - if (j == target) 1f else 0f }
                for (i in layers.indices.reversed()) {
                    val prev = activations[i]
                    for (r in 0 until layers[i].outputSize) {
                        for (c in 0 until layers[i].inputSize) {
                            gradW[i][r][c] += delta[r] * prev[c]
                        }
                        gradB[i][r] += delta[r]
                    }
                    if (i > 0) {
                        val back = FloatArray(layers[i].inputSize)
                        for (c in 0 until layers[i].inputSize) {
                            var sum = 0f
                            for (r in 0 until layers[i].outputSize) sum += layers[i].weights[r][c] * delta[r]
                            // ReLU derivative: pass through only where pre-activation > 0.
                            back[c] = if (preActivations[i - 1][c] > 0f) sum else 0f
                        }
                        delta = back
                    }
                }
            }
            val scale = learningRate / batch.size
            for (i in layers.indices) {
                for (r in 0 until layers[i].outputSize) {
                    for (c in 0 until layers[i].inputSize) {
                        layers[i].weights[r][c] -= scale * gradW[i][r][c]
                    }
                    layers[i].bias[r] -= scale * gradB[i][r]
                }
            }
            start += batchSize
        }
        losses.add(totalLoss / xs.size)
    }
    return TrainResult(mlp, losses)
}

/** Fraction of examples the model classifies correctly. */
fun accuracy(mlp: Mlp, xs: List<FloatArray>, ys: IntArray): Float {
    require(xs.size == ys.size) { "inputs and labels must align" }
    if (xs.isEmpty()) return 0f
    var correct = 0
    for (i in xs.indices) if (mlp.predict(xs[i]) == ys[i]) correct++
    return correct.toFloat() / xs.size
}

fun accuracy(mlp: QuantizedMlp, xs: List<FloatArray>, ys: IntArray): Float {
    require(xs.size == ys.size) { "inputs and labels must align" }
    if (xs.isEmpty()) return 0f
    var correct = 0
    for (i in xs.indices) if (mlp.predict(xs[i]) == ys[i]) correct++
    return correct.toFloat() / xs.size
}

/** Cross-entropy loss of one example, for tests and diagnostics. */
fun crossEntropyLoss(probs: FloatArray, target: Int): Float =
    -kotlin.math.ln(probs[target].toDouble().coerceAtLeast(1e-12)).toFloat()
