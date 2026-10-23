package ai.ondevice

import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows
import org.junit.jupiter.api.io.TempDir
import java.io.File
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

private fun tinyModel(dir: File, name: String, seed: Long = 3L): File {
    val rng = java.util.Random(seed)
    val mlp = Mlp(
        listOf(
            LinearLayer(xavierInit(4, 3, rng), FloatArray(4)),
            LinearLayer(xavierInit(2, 4, rng), FloatArray(2)),
        ),
    )
    val file = File(dir, name)
    ModelFile.save(file, mlp, listOf("a", "b"))
    return file
}

class ModelStoreTest {

    @Test
    fun `install then list activate and verify`(@TempDir dir: File) {
        val store = ModelStore(File(dir, "store"))
        val source = tinyModel(dir, "m.bin")
        val result = store.install(source, "tiny", 1)
        assertTrue(result is InstallResult.Installed)
        assertEquals(listOf("tiny"), store.list().map { it.name })
        assertEquals(1, store.list().single().version)
        assertNull(store.active())
        store.activate("tiny")
        assertEquals("tiny", store.active()?.name)
        assertNotNull(store.activeModelFile())
        assertTrue(store.verify("tiny"))
    }

    @Test
    fun `reinstalling identical bytes is a download-once no-op`(@TempDir dir: File) {
        val store = ModelStore(File(dir, "store"))
        val source = tinyModel(dir, "m.bin")
        store.install(source, "tiny", 1)
        val again = store.install(source, "tiny", 1)
        assertTrue(again is InstallResult.AlreadyPresent)
        assertEquals(1, store.list().size)
    }

    @Test
    fun `downgrade and same-version-different-bytes are rejected`(@TempDir dir: File) {
        val store = ModelStore(File(dir, "store"))
        store.install(tinyModel(dir, "m1.bin", seed = 3L), "tiny", 2)
        val other = tinyModel(dir, "m2.bin", seed = 4L) // different random weights, same shape
        assertThrows<IllegalArgumentException> { store.install(other, "tiny", 2) }
        assertThrows<IllegalArgumentException> { store.install(other, "tiny", 1) }
        // The installed v2 model survives both rejected installs.
        assertEquals(2, store.list().single().version)
        assertTrue(store.verify("tiny"))
    }

    @Test
    fun `corrupt source never clobbers an installed model`(@TempDir dir: File) {
        val store = ModelStore(File(dir, "store"))
        store.install(tinyModel(dir, "m.bin"), "tiny", 1)
        val junk = File(dir, "junk.bin").also { it.writeBytes(byteArrayOf(9, 9, 9, 9)) }
        assertThrows<IllegalArgumentException> { store.install(junk, "tiny", 2) }
        assertTrue(store.verify("tiny"))
    }

    @Test
    fun `verify detects tampering and remove clears the active pointer`(@TempDir dir: File) {
        val store = ModelStore(File(dir, "store"))
        store.install(tinyModel(dir, "m.bin"), "tiny", 1)
        store.activate("tiny")
        File(dir, "store/tiny.bin").appendBytes(byteArrayOf(0))
        assertFalse(store.verify("tiny"))
        store.remove("tiny")
        assertTrue(store.list().isEmpty())
        assertNull(store.active())
    }

    @Test
    fun `activate and remove reject unknown models`(@TempDir dir: File) {
        val store = ModelStore(File(dir, "store"))
        assertThrows<IllegalArgumentException> { store.activate("ghost") }
        assertThrows<IllegalArgumentException> { store.remove("ghost") }
        assertFalse(store.verify("ghost"))
    }
}

class FramesTest {

    @Test
    fun `simulated camera is deterministic for a seed and exhausts`(@TempDir dir: File) {
        val labels = listOf("circle", "triangle", "square")
        val first = SimulatedCamera(labels, seed = 5L)
        val second = SimulatedCamera(labels, seed = 5L)
        val a = generateSequence { first.next() }.toList()
        val b = generateSequence { second.next() }.toList()
        assertEquals(3, a.size)
        assertTrue(a.zip(b).all { (x, y) -> x.argb.contentEquals(y.argb) })
        assertEquals(56, a[0].width)
    }

    @Test
    fun `gallery png round-trips pixels through load`(@TempDir dir: File) {
        val gray = Shapes.generate("cross", java.util.Random(2L))
        val frame = ImageFrame(gray.toArgb(), Shapes.SIZE, Shapes.SIZE)
        val png = File(dir, "cross.png")
        GalleryImages.savePng(frame, png)
        val loaded = GalleryImages.load(png)
        assertEquals(Shapes.SIZE, loaded.width)
        assertTrue(loaded.argb.contentEquals(frame.argb))
    }

    @Test
    fun `live classifier smooths to the majority label`(@TempDir dir: File) {
        val (trainX, trainY) = Shapes.dataset(120, seed = 42L)
        val result = trainMlp(trainX, trainY, Shapes.LABELS.size, epochs = 30)
        val classifier = OnDeviceClassifier(result.mlp, result.mlp.toQuantized(), Shapes.LABELS)
        val live = LiveClassifier(classifier, windowSize = 5)
        // Four triangle frames, one circle frame: the vote must stay triangle.
        val frames = listOf("triangle", "triangle", "circle", "triangle", "triangle")
            .map { SimulatedCamera(listOf(it), seed = 8L).next()!! }
        var smoothed: SmoothedResult? = null
        for (frame in frames) smoothed = live.onFrame(frame)
        assertEquals("triangle", smoothed?.label)
        assertEquals(4, smoothed?.votes)
        assertEquals(5, smoothed?.window)
        assertEquals(listOf("triangle", "triangle", "circle", "triangle", "triangle"), live.windowLabels())
    }

    @Test
    fun `preprocess accepts a gallery-sized frame`(@TempDir dir: File) {
        val gray = Shapes.generate("diamond", java.util.Random(4L))
        val frame = ImageFrame(gray.toArgb(), Shapes.SIZE, Shapes.SIZE)
        val vector = preprocessImage(frame.argb, frame.width, frame.height, 28)
        assertEquals(28 * 28, vector.size)
        assertTrue(vector.all { it in 0f..1f })
    }
}
