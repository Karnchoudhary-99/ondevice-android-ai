package ai.ondevice

/**
 * Image preprocessing shared by the JVM demo and (later) the Android app.
 *
 * The Android side feeds camera/gallery pixels in as an ARGB [IntArray] and
 * calls [preprocessImage]; the JVM demo does the same with generated
 * shapes. One function, one contract, no platform drift: grayscale, resize
 * to the model's input size, scale to [0, 1].
 */

/** Luminance grayscale: 0.299 R + 0.587 G + 0.114 B, scaled to [0, 1]. */
fun toGrayscale(argb: IntArray, width: Int, height: Int): FloatArray {
    require(argb.size == width * height) { "pixel count (${argb.size}) must equal width*height ($width*$height)" }
    return FloatArray(argb.size) { i ->
        val pixel = argb[i]
        val r = (pixel shr 16) and 0xFF
        val g = (pixel shr 8) and 0xFF
        val b = pixel and 0xFF
        (0.299f * r + 0.587f * g + 0.114f * b) / 255f
    }
}

/** Bilinear resize of a single-channel image. */
fun resizeBilinear(gray: FloatArray, width: Int, height: Int, outWidth: Int, outHeight: Int): FloatArray {
    require(gray.size == width * height) { "pixel count must equal width*height" }
    require(outWidth > 0 && outHeight > 0) { "output size must be positive" }
    if (width == outWidth && height == outHeight) return gray.copyOf()
    val out = FloatArray(outWidth * outHeight)
    val xScale = if (outWidth == 1) 0f else (width - 1).toFloat() / (outWidth - 1)
    val yScale = if (outHeight == 1) 0f else (height - 1).toFloat() / (outHeight - 1)
    for (oy in 0 until outHeight) {
        val sy = oy * yScale
        val y0 = sy.toInt().coerceIn(0, height - 1)
        val y1 = (y0 + 1).coerceIn(0, height - 1)
        val fy = sy - y0
        for (ox in 0 until outWidth) {
            val sx = ox * xScale
            val x0 = sx.toInt().coerceIn(0, width - 1)
            val x1 = (x0 + 1).coerceIn(0, width - 1)
            val fx = sx - x0
            val top = gray[y0 * width + x0] * (1 - fx) + gray[y0 * width + x1] * fx
            val bottom = gray[y1 * width + x0] * (1 - fx) + gray[y1 * width + x1] * fx
            out[oy * outWidth + ox] = top * (1 - fy) + bottom * fy
        }
    }
    return out
}

/**
 * Full pipeline: ARGB pixels -> model-ready vector of [size] x [size]
 * floats in [0, 1]. The model input size is fixed at training time, so
 * every caller goes through here.
 */
fun preprocessImage(argb: IntArray, width: Int, height: Int, size: Int = 28): FloatArray {
    require(size > 0) { "model input size must be positive" }
    val gray = toGrayscale(argb, width, height)
    return resizeBilinear(gray, width, height, size, size)
}

/** Render a [size] x [size] grayscale vector as ASCII art (demo/debugging aid). */
fun renderAscii(vector: FloatArray, size: Int = 28): String {
    require(vector.size == size * size) { "vector size (${vector.size}) must equal size*size (${size * size})" }
    val shades = " .:-=+*#%@"
    val sb = StringBuilder()
    for (y in 0 until size) {
        for (x in 0 until size) {
            val v = vector[y * size + x].coerceIn(0f, 1f)
            sb.append(shades[(v * (shades.length - 1)).toInt()])
        }
        if (y < size - 1) sb.append('\n')
    }
    return sb.toString()
}
