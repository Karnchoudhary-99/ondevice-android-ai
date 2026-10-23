package ai.ondevice

import org.junit.jupiter.api.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class PreprocessTest {

    @Test
    fun `grayscale uses luminance weights`() {
        // Pure red -> 0.299, pure green -> 0.587, pure blue -> 0.114.
        val argb = intArrayOf(0xFFFF0000.toInt(), 0xFF00FF00.toInt(), 0xFF0000FF.toInt())
        val gray = toGrayscale(argb, 3, 1)
        assertEquals(0.299f, gray[0], 1e-3f)
        assertEquals(0.587f, gray[1], 1e-3f)
        assertEquals(0.114f, gray[2], 1e-3f)
    }

    @Test
    fun `resize to the same size is a copy, and output is always size squared`() {
        val gray = FloatArray(4) { it / 4f }
        val same = resizeBilinear(gray, 2, 2, 2, 2)
        assertTrue(same.contentEquals(gray))
        val up = resizeBilinear(gray, 2, 2, 4, 4)
        assertEquals(16, up.size)
        assertTrue(up.all { it in 0f..1f })
    }

    @Test
    fun `preprocess produces a 28x28 vector in range`() {
        val white = 0xFFFFFFFF.toInt()
        val argb = IntArray(64 * 48) { white }
        val vec = preprocessImage(argb, 64, 48)
        assertEquals(28 * 28, vec.size)
        assertTrue(vec.all { it == 1f })
    }

    @Test
    fun `renderAscii emits one row per line`() {
        val vec = FloatArray(28 * 28)
        val art = renderAscii(vec)
        assertEquals(28, art.lines().size)
        assertTrue(art.lines().all { it.length == 28 })
    }
}
