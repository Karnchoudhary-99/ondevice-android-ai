package ai.ondevice

import org.junit.jupiter.api.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class QuantizationTest {

    @Test
    fun `symmetric round-trip error stays within one quantization step`() {
        val values = floatArrayOf(-2.5f, -0.3f, 0f, 0.7f, 3.1f)
        val (quantized, scale) = quantizeSymmetric(values)
        assertTrue(quantized.all { it in -127..127 })
        val back = dequantizeSymmetric(quantized, scale)
        for (i in values.indices) {
            assertTrue(kotlin.math.abs(back[i] - values[i]) <= scale, "index $i: ${back[i]} vs ${values[i]}")
        }
    }

    @Test
    fun `quantized forward tracks the float forward`() {
        val rng = java.util.Random(3L)
        val layer = LinearLayer(xavierInit(6, 5, rng), FloatArray(6) { rng.nextFloat() - 0.5f })
        val flat = FloatArray(30) { layer.weights[it / 5][it % 5] }
        val (weightsQ, scale) = quantizeSymmetric(flat)
        val qLayer = QuantizedLinearLayer(weightsQ, 6, 5, scale, layer.bias.copyOf())
        val input = floatArrayOf(0.2f, -0.4f, 0.9f, 0.1f, -0.7f)
        val floatOut = layer.forward(input)
        val intOut = qLayer.forward(input)
        for (i in floatOut.indices) {
            // Integer path error comes from two quantization steps; bound it loosely.
            assertTrue(kotlin.math.abs(intOut[i] - floatOut[i]) < 0.35f, "row $i: ${intOut[i]} vs ${floatOut[i]}")
        }
    }

    @Test
    fun `quantized model is 4x smaller in weights and stays accurate on shapes`() {
        val (trainX, trainY) = Shapes.dataset(120, seed = 42L)
        val (testX, testY) = Shapes.dataset(50, seed = 1234L)
        val result = trainMlp(trainX, trainY, Shapes.LABELS.size, epochs = 30)
        val quantized = result.mlp.toQuantized()
        assertEquals(result.mlp.weightBytes() / 4, quantized.weightBytes())
        val floatAcc = accuracy(result.mlp, testX, testY)
        val intAcc = accuracy(quantized, testX, testY)
        assertTrue(floatAcc >= 0.95f, "float accuracy $floatAcc")
        assertTrue(intAcc >= floatAcc - 0.05f, "int8 accuracy $intAcc vs float $floatAcc")
    }
}
