package ai.ondevice

/**
 * The single entry point the Android UI talks to.
 *
 * The app never touches layers or tensors directly: it hands the
 * classifier a preprocessed image vector and gets back ranked predictions
 * with confidences, plus an optional latency measurement. Swapping the
 * float model for the quantized one is a one-line change at construction
 * time, which is how the app will A/B the two on real phones.
 */
data class Prediction(val label: String, val confidence: Float)

data class Classification(val predictions: List<Prediction>, val latencyMs: Double) {
    /** The top prediction, or null when [predictions] is empty. */
    val best: Prediction? get() = predictions.firstOrNull()
}

/** Shared top-k ranking over a probability vector. */
fun topK(probs: FloatArray, labels: List<String>, k: Int): List<Prediction> {
    require(labels.size == probs.size) { "label count (${labels.size}) must match probabilities (${probs.size})" }
    require(k > 0) { "k must be positive" }
    return probs.indices
        .sortedByDescending { probs[it] }
        .take(minOf(k, probs.size))
        .map { Prediction(labels[it], probs[it]) }
}

class OnDeviceClassifier(
    val floatMlp: Mlp,
    val quantizedMlp: QuantizedMlp,
    val labels: List<String>,
) {
    /** Set false to run the float model instead of the quantized one. */
    var useQuantized: Boolean = true

    init {
        require(labels.size == floatMlp.outputSize) { "labels must match model outputs" }
        require(quantizedMlp.outputSize == floatMlp.outputSize) { "quantized model must match float outputs" }
    }

    /** Classify one preprocessed image vector, returning the top-[k] predictions. */
    fun classify(input: FloatArray, k: Int = 3): Classification {
        val start = System.nanoTime()
        val probs = if (useQuantized) quantizedMlp.probabilities(input) else floatMlp.probabilities(input)
        val latencyMs = (System.nanoTime() - start) / 1_000_000.0
        return Classification(topK(probs, labels, k), latencyMs)
    }

    /**
     * Benchmark helper: mean and p50 latency over [runs] warm-up-free runs
     * (the first run is discarded — class loading and JIT warm-up are not
     * inference latency). Returns (meanMs, p50Ms).
     */
    fun benchmark(input: FloatArray, runs: Int = 200): Pair<Double, Double> {
        require(runs > 0) { "runs must be positive" }
        classify(input) // warm-up, discarded
        val times = DoubleArray(runs) { classify(input).latencyMs }
        times.sort()
        return times.average() to times[times.size / 2]
    }

    /**
     * Raw per-inference latencies (ms), first run discarded as warm-up.
     * The reporting module in Benchmark.kt turns these into full
     * distributions (p95/p99/throughput); [benchmark] stays as the
     * mean/p50 shortcut the demo console output uses.
     */
    fun benchmarkTimes(input: FloatArray, runs: Int = 200): DoubleArray {
        require(runs > 0) { "runs must be positive" }
        classify(input) // warm-up, discarded
        return DoubleArray(runs) { classify(input).latencyMs }
    }
}
