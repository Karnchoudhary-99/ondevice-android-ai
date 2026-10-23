package ai.ondevice

import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class BenchmarkTest {

    @Test
    fun `latency stats use nearest-rank percentiles`() {
        val stats = latencyStats(doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0))
        assertEquals(10, stats.runs)
        assertEquals(5.5, stats.meanMs, 1e-9)
        assertEquals(5.0, stats.p50Ms, 1e-9)
        assertEquals(10.0, stats.p95Ms, 1e-9)
        assertEquals(10.0, stats.p99Ms, 1e-9)
        assertEquals(1.0, stats.minMs, 1e-9)
        assertEquals(10.0, stats.maxMs, 1e-9)
        assertEquals(1000.0 / 5.5, stats.throughputPerSec, 1e-6)
    }

    @Test
    fun `latency stats reject empty input`() {
        assertThrows<IllegalArgumentException> { latencyStats(doubleArrayOf()) }
    }

    @Test
    fun `per-label accuracy counts per class and tolerates unseen labels`() {
        val xs = listOf(FloatArray(3), FloatArray(3), FloatArray(3), FloatArray(3))
        val ys = intArrayOf(0, 0, 1, 2)
        val result = perLabelAccuracy({ 0 }, xs, ys, listOf("a", "b", "c", "d"))
        assertEquals(1.0, result[0].accuracy, 1e-9)
        assertEquals(2, result[0].total)
        assertEquals(0.0, result[1].accuracy, 1e-9)
        assertEquals(0.0, result[2].accuracy, 1e-9)
        assertEquals(0, result[3].total) // unseen label: reported, not crashed
        assertEquals(0.0, result[3].accuracy, 1e-9)
    }

    @Test
    fun `model size breakdown is 4x float vs int8 per layer`() {
        val rng = java.util.Random(5L)
        val mlp = Mlp(
            listOf(
                LinearLayer(xavierInit(4, 3, rng), FloatArray(4)),
                LinearLayer(xavierInit(2, 4, rng), FloatArray(2)),
            ),
        )
        val layers = modelSizeBreakdown(mlp)
        assertEquals(2, layers.size)
        assertEquals(48L, layers[0].floatBytes)
        assertEquals(12L, layers[0].int8Bytes)
        assertEquals(32L, layers[1].floatBytes)
        assertEquals(8L, layers[1].int8Bytes)
    }

    @Test
    fun `benchmark times return one non-negative timing per run`() {
        val rng = java.util.Random(5L)
        val mlp = Mlp(listOf(LinearLayer(xavierInit(2, 3, rng), FloatArray(2))))
        val classifier = OnDeviceClassifier(mlp, mlp.toQuantized(), listOf("a", "b"))
        val times = classifier.benchmarkTimes(FloatArray(3) { 0.5f }, runs = 5)
        assertEquals(5, times.size)
        assertTrue(times.all { it >= 0.0 })
    }

    @Test
    fun `full report on a quick-trained model is consistent`() {
        val (trainX, trainY) = Shapes.dataset(perClass = 60, seed = 42L)
        val (testX, testY) = Shapes.dataset(perClass = 10, seed = 1234L)
        val result = trainMlp(trainX, trainY, Shapes.LABELS.size, epochs = 20)
        val classifier = OnDeviceClassifier(result.mlp, result.mlp.toQuantized(), Shapes.LABELS)
        val report = runBenchmark(classifier, testX.first(), testX, testY, runs = 20)
        assertEquals(20, report.runs)
        assertEquals(testX.size, report.testSize)
        assertEquals(5, report.floatPerLabel.size)
        assertEquals(testX.size, report.floatPerLabel.sumOf { it.total })
        assertTrue(report.floatAccuracy >= 0.8, "float accuracy ${report.floatAccuracy} unexpectedly low")
        assertTrue(report.int8Accuracy >= 0.8, "int8 accuracy ${report.int8Accuracy} unexpectedly low")
        assertEquals(4.0, report.sizeRatio, 1e-9)
        val markdown = report.toMarkdown()
        assertTrue("## Latency" in markdown)
        assertTrue("## Accuracy" in markdown)
        assertTrue("## Model size" in markdown)
        assertTrue("Overall" in markdown)
        assertTrue(report.modelShape in markdown)
    }
}
