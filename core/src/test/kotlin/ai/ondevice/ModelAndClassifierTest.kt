package ai.ondevice

import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows
import org.junit.jupiter.api.io.TempDir
import java.io.File
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class ModelAndClassifierTest {

    @Test
    fun `model save and load round-trips weights and labels`(@TempDir dir: File) {
        val rng = java.util.Random(9L)
        val mlp = Mlp(
            listOf(
                LinearLayer(xavierInit(8, 4, rng), FloatArray(8) { 0.1f }),
                LinearLayer(xavierInit(3, 8, rng), FloatArray(3)),
            ),
        )
        val file = File(dir, "model.bin")
        ModelFile.save(file, mlp, listOf("a", "b", "c"))
        val loaded = ModelFile.load(file)
        assertEquals(listOf("a", "b", "c"), loaded.labels)
        for (l in mlp.layers.indices) {
            for (r in mlp.layers[l].weights.indices) {
                assertTrue(loaded.mlp.layers[l].weights[r].contentEquals(mlp.layers[l].weights[r]))
            }
            assertTrue(loaded.mlp.layers[l].bias.contentEquals(mlp.layers[l].bias))
        }
    }

    @Test
    fun `model load rejects bad magic`(@TempDir dir: File) {
        val file = File(dir, "junk.bin").also { it.writeBytes(byteArrayOf(1, 2, 3, 4, 0, 1)) }
        assertThrows<IllegalArgumentException> { ModelFile.load(file) }
    }

    @Test
    fun `model save rejects a label count mismatch`() {
        val mlp = Mlp(listOf(LinearLayer(Array(2) { FloatArray(3) }, FloatArray(2))))
        assertThrows<IllegalArgumentException> {
            ModelFile.save(File("unused.bin"), mlp, listOf("only-one"))
        }
    }

    @Test
    fun `topK ranks descending and caps at k`() {
        val ranked = topK(floatArrayOf(0.1f, 0.6f, 0.3f), listOf("a", "b", "c"), 2)
        assertEquals(listOf("b", "c"), ranked.map { it.label })
        assertEquals(0.6f, ranked[0].confidence, 1e-6f)
    }

    @Test
    fun `classifier predicts the generated shape and reports positive latency`() {
        val (trainX, trainY) = Shapes.dataset(120, seed = 42L)
        val result = trainMlp(trainX, trainY, Shapes.LABELS.size, epochs = 30)
        val classifier = OnDeviceClassifier(result.mlp, result.mlp.toQuantized(), Shapes.LABELS)
        for (label in Shapes.LABELS) {
            val image = Shapes.generate(label, java.util.Random(11L))
            val classification = classifier.classify(image)
            assertEquals(label, classification.best?.label, "shape $label misclassified")
            assertTrue(classification.latencyMs >= 0.0)
            assertTrue(classification.predictions.zipWithNext().all { (a, b) -> a.confidence >= b.confidence })
        }
    }

    @Test
    fun `benchmark returns positive mean and p50`() {
        val (trainX, trainY) = Shapes.dataset(60, seed = 42L)
        val result = trainMlp(trainX, trainY, Shapes.LABELS.size, epochs = 10)
        val classifier = OnDeviceClassifier(result.mlp, result.mlp.toQuantized(), Shapes.LABELS)
        val (mean, p50) = classifier.benchmark(trainX[0], runs = 10)
        assertTrue(mean > 0.0)
        assertTrue(p50 > 0.0)
    }

    @Test
    fun `training loss decreases over epochs`() {
        val (trainX, trainY) = Shapes.dataset(100, seed = 42L)
        val result = trainMlp(trainX, trainY, Shapes.LABELS.size, epochs = 15)
        assertTrue(result.epochLosses.last() < result.epochLosses.first(), "loss did not decrease")
        assertTrue(accuracy(result.mlp, trainX, trainY) >= 0.9f)
    }
}
