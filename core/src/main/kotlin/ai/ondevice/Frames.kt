package ai.ondevice

import java.awt.image.BufferedImage
import java.io.File
import javax.imageio.ImageIO

/**
 * Camera + gallery input for the Android app, in platform-neutral form.
 *
 * On Android, CameraX hands the app a stream of frames and the gallery
 * hands it a bitmap; both arrive here as ARGB pixels in an [ImageFrame]
 * and go through the one shared [preprocessImage] pipeline — the JVM
 * demo uses the same types with a simulated camera and PNG/JPEG files,
 * so the live-input logic (smoothing, throttling, latency) is tested
 * without an emulator.
 */

/** One camera/gallery frame: ARGB pixels, row-major, alpha ignored. */
data class ImageFrame(val argb: IntArray, val width: Int, val height: Int) {
    init {
        require(argb.size == width * height) {
            "pixel count (${argb.size}) must equal width*height ($width*$height)"
        }
    }
}

/** A source of frames: a camera stream, a gallery pick, or a test double. */
fun interface FrameSource {
    /** The next frame, or null when the source is exhausted. */
    fun next(): ImageFrame?
}

/** Convert a [Shapes]-style grayscale image (0 = paper, 1 = ink) to ARGB. */
fun FloatArray.toArgb(): IntArray =
    IntArray(size) { i ->
        val gray = (this[i].coerceIn(0f, 1f) * 255f).toInt()
        (0xFF shl 24) or (gray shl 16) or (gray shl 8) or gray
    }

/**
 * A deterministic stand-in for a phone camera: it emits frames of one
 * shape per [label] in sequence, each frame re-drawn with fresh jitter
 * and noise — exactly what a shaky hand-held camera produces. Same seed
 * means same frames, so live-input behaviour is reproducible in tests.
 */
class SimulatedCamera(
    private val labels: List<String>,
    seed: Long = 42L,
    private val frameWidth: Int = 56,
    private val frameHeight: Int = 56,
) : FrameSource {
    private val rng = java.util.Random(seed)
    private var index = 0

    override fun next(): ImageFrame? {
        if (index >= labels.size) return null
        val small = Shapes.generate(labels[index++], rng)
        // Upscale 28x28 -> frame size with the shared bilinear resizer, so
        // the simulated camera exercises the same path as a real frame.
        val gray = resizeBilinear(small, Shapes.SIZE, Shapes.SIZE, frameWidth, frameHeight)
        return ImageFrame(gray.toArgb(), frameWidth, frameHeight)
    }
}

/** Gallery input on the JVM: decode PNG/JPEG files with ImageIO. */
object GalleryImages {
    /** Load an image file into an [ImageFrame] (any size; preprocessing resizes). */
    fun load(file: File): ImageFrame {
        require(file.isFile) { "image not found: ${file.path}" }
        val image: BufferedImage = ImageIO.read(file)
            ?: throw IllegalArgumentException("unsupported or corrupt image: ${file.path}")
        val argb = IntArray(image.width * image.height)
        image.getRGB(0, 0, image.width, image.height, argb, 0, image.width)
        return ImageFrame(argb, image.width, image.height)
    }

    /** Write a frame as a PNG (used by the demo to produce gallery samples). */
    fun savePng(frame: ImageFrame, file: File) {
        file.parentFile?.mkdirs()
        val image = BufferedImage(frame.width, frame.height, BufferedImage.TYPE_INT_ARGB)
        image.setRGB(0, 0, frame.width, frame.height, frame.argb, 0, frame.width)
        require(ImageIO.write(image, "png", file)) { "could not write PNG: ${file.path}" }
    }
}

/**
 * Live classification over a frame stream. Single frames flicker (noise
 * flips the top label between neighbours), so the pipeline keeps a
 * sliding window of recent top labels and reports the majority vote —
 * the same smoothing a camera UI shows, plus the latest raw result and
 * the winning label's confidence on that frame.
 */
data class SmoothedResult(
    val label: String,
    val votes: Int,
    val window: Int,
    val confidence: Float,
    val latest: Classification,
)

class LiveClassifier(
    private val classifier: OnDeviceClassifier,
    private val windowSize: Int = 5,
    private val minIntervalMs: Long = 0L,
) {
    private val recent = ArrayDeque<String>()
    private var lastAtMs: Long = 0L

    init {
        require(windowSize > 0) { "windowSize must be positive" }
        require(minIntervalMs >= 0) { "minIntervalMs must be non-negative" }
    }

    /**
     * Classify one frame. Returns null when throttled (a frame arrived
     * before [minIntervalMs] elapsed) — the caller keeps showing the last
     * result instead of burning battery on duplicate work.
     */
    fun onFrame(frame: ImageFrame, modelInputSize: Int = 28): SmoothedResult? {
        val now = System.currentTimeMillis()
        if (minIntervalMs > 0 && lastAtMs != 0L && now - lastAtMs < minIntervalMs) return null
        lastAtMs = now
        val vector = preprocessImage(frame.argb, frame.width, frame.height, modelInputSize)
        val classification = classifier.classify(vector)
        val best = classification.best ?: return null
        recent.addLast(best.label)
        while (recent.size > windowSize) recent.removeFirst()
        val counts = recent.groupingBy { it }.eachCount()
        val (label, votes) = counts.maxByOrNull { it.value }!!
        // Confidence for the winning label from this frame's ranking; the
        // vote count above is what carries the smoothing signal.
        val confidence = if (label == best.label) best.confidence else classification.predictions
            .firstOrNull { it.label == label }?.confidence ?: 0f
        return SmoothedResult(label, votes, recent.size, confidence, classification)
    }

    /** Frames currently in the smoothing window (for tests / debugging). */
    fun windowLabels(): List<String> = recent.toList()
}
