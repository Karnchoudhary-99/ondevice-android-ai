package ai.ondevice

import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream

/**
 * Bundled-model file format ("ODAI").
 *
 * Layout (big-endian, via [DataOutputStream]):
 * ```
 * magic      4 bytes  "ODAI"
 * version    u16      1
 * labels     u16 count, then per label: u16 length + UTF-8 bytes
 * layers     u16 count, then per layer:
 *   rows     int, cols int
 *   weights  rows*cols floats (row-major)
 *   bias     rows floats
 * ```
 * The Android app ships one of these files in `assets/`; the demo trains
 * one and writes it to `models/`. Version and magic are validated on load
 * so a truncated or foreign file fails loudly instead of misclassifying.
 */
object ModelFile {
    private val MAGIC = byteArrayOf('O'.code.toByte(), 'D'.code.toByte(), 'A'.code.toByte(), 'I'.code.toByte())
    private const val VERSION = 1

    /** A loaded model: the network plus the label for each output index. */
    data class Loaded(val mlp: Mlp, val labels: List<String>)

    fun save(file: File, mlp: Mlp, labels: List<String>) {
        require(labels.size == mlp.outputSize) {
            "label count (${labels.size}) must match output size (${mlp.outputSize})"
        }
        file.parentFile?.mkdirs()
        DataOutputStream(FileOutputStream(file)).use { out ->
            out.write(MAGIC)
            out.writeShort(VERSION)
            out.writeShort(labels.size)
            for (label in labels) {
                val bytes = label.toByteArray(Charsets.UTF_8)
                require(bytes.size <= 0xFFFF) { "label too long: $label" }
                out.writeShort(bytes.size)
                out.write(bytes)
            }
            out.writeShort(mlp.layers.size)
            for (layer in mlp.layers) {
                out.writeInt(layer.outputSize)
                out.writeInt(layer.inputSize)
                for (row in layer.weights) for (w in row) out.writeFloat(w)
                for (b in layer.bias) out.writeFloat(b)
            }
        }
    }

    fun load(file: File): Loaded {
        DataInputStream(FileInputStream(file)).use { input ->
            val magic = ByteArray(4).also { input.readFully(it) }
            require(magic.contentEquals(MAGIC)) { "not an ODAI model file (bad magic): ${file.path}" }
            val version = input.readUnsignedShort()
            require(version == VERSION) { "unsupported ODAI version $version (expected $VERSION)" }
            val labelCount = input.readUnsignedShort()
            val labels = (0 until labelCount).map {
                val len = input.readUnsignedShort()
                val bytes = ByteArray(len).also { b -> input.readFully(b) }
                String(bytes, Charsets.UTF_8)
            }
            val layerCount = input.readUnsignedShort()
            val layers = (0 until layerCount).map {
                val rows = input.readInt()
                val cols = input.readInt()
                require(rows > 0 && cols > 0) { "layer dimensions must be positive, got ${rows}x$cols" }
                val weights = Array(rows) { FloatArray(cols) { input.readFloat() } }
                val bias = FloatArray(rows) { input.readFloat() }
                LinearLayer(weights, bias)
            }
            val mlp = Mlp(layers)
            require(labels.size == mlp.outputSize) {
                "label count (${labels.size}) must match output size (${mlp.outputSize})"
            }
            return Loaded(mlp, labels)
        }
    }
}
