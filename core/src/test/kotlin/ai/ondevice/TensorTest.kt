package ai.ondevice

import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class TensorTest {

    @Test
    fun `matVec multiplies rows by the vector`() {
        val matrix = arrayOf(floatArrayOf(1f, 2f), floatArrayOf(0f, -1f))
        val out = matVec(matrix, floatArrayOf(3f, 4f))
        assertEquals(11f, out[0], 1e-6f)
        assertEquals(-4f, out[1], 1e-6f)
    }

    @Test
    fun `matVec rejects a dimension mismatch`() {
        assertThrows<IllegalArgumentException> {
            matVec(arrayOf(floatArrayOf(1f, 2f)), floatArrayOf(1f))
        }
    }

    @Test
    fun `relu zeroes negatives and keeps positives`() {
        val out = reluInPlace(floatArrayOf(-2f, 0f, 3f))
        assertTrue(out.contentEquals(floatArrayOf(0f, 0f, 3f)))
    }

    @Test
    fun `softmax sums to one and is shift invariant`() {
        val a = softmax(floatArrayOf(1f, 2f, 3f))
        val b = softmax(floatArrayOf(11f, 12f, 13f))
        assertEquals(1f, a.sum(), 1e-5f)
        for (i in a.indices) assertEquals(a[i], b[i], 1e-5f)
        assertTrue(a[2] > a[1] && a[1] > a[0])
    }

    @Test
    fun `argmax resolves ties to the lowest index`() {
        assertEquals(0, argmax(floatArrayOf(0.5f, 0.5f, 0.1f)))
        assertEquals(2, argmax(floatArrayOf(0.1f, 0.2f, 0.9f)))
    }

    @Test
    fun `mlp forward produces probabilities that sum to one`() {
        val rng = java.util.Random(1L)
        val mlp = Mlp(
            listOf(
                LinearLayer(xavierInit(8, 4, rng), FloatArray(8)),
                LinearLayer(xavierInit(3, 8, rng), FloatArray(3)),
            ),
        )
        val probs = mlp.probabilities(floatArrayOf(0.1f, -0.2f, 0.3f, 0f))
        assertEquals(3, probs.size)
        assertEquals(1f, probs.sum(), 1e-5f)
        assertTrue(probs.all { it > 0f && it < 1f })
    }

    @Test
    fun `mlp rejects mismatched layer dimensions`() {
        assertThrows<IllegalArgumentException> {
            Mlp(
                listOf(
                    LinearLayer(Array(4) { FloatArray(3) }, FloatArray(4)),
                    LinearLayer(Array(2) { FloatArray(5) }, FloatArray(2)),
                ),
            )
        }
    }
}
