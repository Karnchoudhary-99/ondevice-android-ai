package ai.ondevice.demo

import ai.ondevice.ModelFile
import ai.ondevice.OnDeviceClassifier
import ai.ondevice.Shapes
import ai.ondevice.accuracy
import ai.ondevice.renderAscii
import ai.ondevice.toQuantized
import ai.ondevice.trainMlp
import java.io.File
import kotlin.system.exitProcess

/**
 * Demo CLI for the on-device inference core.
 *
 * ```
 * demo train --out models/shapes-model.bin --epochs 60
 * demo classify --model models/shapes-model.bin --shape triangle --seed 7
 * demo benchmark --model models/shapes-model.bin --runs 200
 * ```
 * Every command also works without `--model`: it trains a quick in-memory
 * model first (30 epochs), so `gradle :demo:run --args="benchmark --runs 50"`
 * is a valid CI smoke test with no files involved.
 */

private fun usage(): Nothing {
    println(
        """
        Usage: demo <train|classify|benchmark> [options]
          train     --out <path> [--epochs N] [--hidden N] [--per-class N]
          classify  [--model <path>] --shape <label> [--seed N] [--no-quantized]
          benchmark [--model <path>] [--runs N]
        Shapes: ${Shapes.LABELS.joinToString(", ")}
        """.trimIndent(),
    )
    exitProcess(2)
}

/** Parse `--key value` / `--flag` options into a map. */
private fun parseOptions(args: List<String>): Map<String, String> {
    val out = HashMap<String, String>()
    var i = 0
    while (i < args.size) {
        val arg = args[i]
        if (arg.startsWith("--")) {
            val key = arg.removePrefix("--")
            if (i + 1 < args.size && !args[i + 1].startsWith("--")) {
                out[key] = args[i + 1]
                i += 2
            } else {
                out[key] = "true"
                i += 1
            }
        } else {
            i += 1
        }
    }
    return out
}

private fun loadOrTrain(opts: Map<String, String>, quickEpochs: Int = 30): ModelFile.Loaded {
    val modelPath = opts["model"]
    if (modelPath != null) return ModelFile.load(File(modelPath))
    println("No --model given; training a quick in-memory model ($quickEpochs epochs)...")
    val (xs, ys) = Shapes.dataset(perClass = 120, seed = 42L)
    val result = trainMlp(xs, ys, Shapes.LABELS.size, epochs = quickEpochs)
    return ModelFile.Loaded(result.mlp, Shapes.LABELS)
}

private fun cmdTrain(opts: Map<String, String>) {
    val outPath = opts["out"] ?: run {
        println("train needs --out <path>")
        usage()
    }
    val epochs = opts["epochs"]?.toIntOrNull() ?: 60
    val hidden = opts["hidden"]?.toIntOrNull() ?: 64
    val perClass = opts["per-class"]?.toIntOrNull() ?: 200
    val (trainX, trainY) = Shapes.dataset(perClass, seed = 42L)
    val (testX, testY) = Shapes.dataset(50, seed = 1234L)
    println("Training ${hidden}-unit MLP for $epochs epochs on ${trainX.size} shapes...")
    val result = trainMlp(trainX, trainY, Shapes.LABELS.size, hidden = hidden, epochs = epochs)
    val quantized = result.mlp.toQuantized()
    println("Final train loss: %.4f".format(result.epochLosses.last()))
    println("Train accuracy:     %.3f".format(accuracy(result.mlp, trainX, trainY)))
    println("Test accuracy:      %.3f".format(accuracy(result.mlp, testX, testY)))
    println("Test accuracy int8: %.3f".format(accuracy(quantized, testX, testY)))
    val file = File(outPath)
    ModelFile.save(file, result.mlp, Shapes.LABELS)
    println("Saved ${file.length()} bytes to $outPath")
}

private fun cmdClassify(opts: Map<String, String>) {
    val shape = opts["shape"] ?: run {
        println("classify needs --shape <label>")
        usage()
    }
    val seed = opts["seed"]?.toLongOrNull() ?: 7L
    val loaded = loadOrTrain(opts)
    val classifier = OnDeviceClassifier(loaded.mlp, loaded.mlp.toQuantized(), loaded.labels)
    classifier.useQuantized = !"no-quantized" in opts
    val image = Shapes.generate(shape, java.util.Random(seed))
    println(renderAscii(image))
    println("Input shape: $shape (seed $seed), model: ${if (classifier.useQuantized) "int8" else "float32"}")
    val result = classifier.classify(image)
    for ((rank, pred) in result.predictions.withIndex()) {
        println("#${rank + 1} ${pred.label} — %.1f%%".format(pred.confidence * 100))
    }
    println("Latency: %.3f ms".format(result.latencyMs))
    if (result.best?.label != shape) {
        println("NOTE: top prediction differs from the generated shape (noise can blur shapes).")
    }
}

private fun cmdBenchmark(opts: Map<String, String>) {
    val runs = opts["runs"]?.toIntOrNull() ?: 200
    val loaded = loadOrTrain(opts)
    val classifier = OnDeviceClassifier(loaded.mlp, loaded.mlp.toQuantized(), loaded.labels)
    val sample = Shapes.generate("circle", java.util.Random(1L))
    println("Model: ${loaded.mlp.inputSize} inputs -> ${loaded.mlp.outputSize} outputs")
    println("Float weights: ${loaded.mlp.weightBytes()} bytes; int8 weights: ${loaded.mlp.toQuantized().weightBytes()} bytes")
    classifier.useQuantized = false
    val (floatMean, floatP50) = classifier.benchmark(sample, runs)
    println("float32: mean %.3f ms, p50 %.3f ms".format(floatMean, floatP50))
    classifier.useQuantized = true
    val (intMean, intP50) = classifier.benchmark(sample, runs)
    println("int8:    mean %.3f ms, p50 %.3f ms".format(intMean, intP50))
    val (testX, testY) = Shapes.dataset(50, seed = 1234L)
    println("Test accuracy float32: %.3f".format(accuracy(loaded.mlp, testX, testY)))
    println("Test accuracy int8:    %.3f".format(accuracy(loaded.mlp.toQuantized(), testX, testY)))
}

fun main(args: Array<String>) {
    if (args.isEmpty()) usage()
    val opts = parseOptions(args.drop(1).toList())
    when (args[0]) {
        "train" -> cmdTrain(opts)
        "classify" -> cmdClassify(opts)
        "benchmark" -> cmdBenchmark(opts)
        else -> usage()
    }
}
