package ai.ondevice

import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Synthetic 28x28 shape dataset — the "MNIST of this repo".
 *
 * A real on-device classifier would train on photos; for a self-contained
 * demo that trains in seconds with zero downloads, five procedurally drawn
 * shapes (circle, square, triangle, cross, diamond) with random jitter,
 * scale and pixel noise are the honest stand-in: the pipeline (preprocess
 * -> MLP -> quantized inference -> top-k) is identical, only the data is
 * synthetic. The Android app later swaps this dataset for camera frames
 * without touching the core.
 */
object Shapes {
    val LABELS = listOf("circle", "square", "triangle", "cross", "diamond")
    const val SIZE = 28

    /** Draw one shape into a [SIZE] x [SIZE] float image (1 = ink, 0 = paper). */
    fun generate(label: String, rng: java.util.Random): FloatArray {
        require(label in LABELS) { "unknown shape: $label" }
        val img = FloatArray(SIZE * SIZE)
        val cx = SIZE / 2f + (rng.nextFloat() * 4f - 2f) // centre jitter +-2 px
        val cy = SIZE / 2f + (rng.nextFloat() * 4f - 2f)
        val r = 7f + rng.nextFloat() * 3f // radius 7..10
        for (y in 0 until SIZE) {
            for (x in 0 until SIZE) {
                val dx = x - cx
                val dy = y - cy
                val on = when (label) {
                    "circle" -> sqrt(dx * dx + dy * dy) <= r
                    "square" -> abs(dx) <= r && abs(dy) <= r
                    "triangle" -> dy <= r * 0.8f && dy >= -r && abs(dx) <= (r - dy) * 0.9f
                    "cross" -> abs(dx) <= r * 0.35f || abs(dy) <= r * 0.35f
                    "diamond" -> abs(dx) + abs(dy) <= r * 1.25f
                    else -> false
                }
                img[y * SIZE + x] = if (on) 1f else 0f
            }
        }
        // Salt noise: flip ~2% of pixels so the model cannot memorize clean edges.
        for (i in img.indices) {
            if (rng.nextFloat() < 0.02f) img[i] = 1f - img[i]
        }
        return img
    }

    /** Deterministic dataset: `perClass` examples of each label, shuffled by [seed]. */
    fun dataset(perClass: Int, seed: Long): Pair<List<FloatArray>, IntArray> {
        val rng = java.util.Random(seed)
        val xs = ArrayList<FloatArray>()
        val ys = ArrayList<Int>()
        for ((index, label) in LABELS.withIndex()) {
            repeat(perClass) {
                xs.add(generate(label, rng))
                ys.add(index)
            }
        }
        val order = xs.indices.shuffled(java.util.Random(seed + 1))
        return order.map { xs[it] } to order.map { ys[it] }.toIntArray()
    }
}
