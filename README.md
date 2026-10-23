# On-device AI Android App

Run AI models fully on-device in an Android app (Kotlin): no server,
no data leaving the phone.

The core is a dependency-free Kotlin inference engine: a small MLP with
a symmetric int8 quantization path, a binary model format, and one
preprocessing pipeline shared by the JVM demo and the Android app. The
bundled model classifies five procedurally drawn shapes; the Android
app (next chunk) swaps the data source for camera/gallery frames
without touching the core.

## Roadmap

- [x] Project scaffold (Gradle, README, CI)
- [x] On-device inference pipeline (custom lightweight runtime — see below)
- [x] Camera + gallery input with live results (simulated camera + gallery PNG/JPEG, smoothed voting)
- [x] Model management (bundled model, download-once, versioning, integrity checks)
- [ ] Benchmarks (latency, model size) + demo
- [ ] Self-review checklist, then public

## Why a custom runtime instead of TensorFlow Lite?

For a tiny MLP, TFLite adds megabytes of native dependency for work a
few hundred lines of Kotlin already do — and the custom runtime keeps
every step (quantization, preprocessing, top-k) visible and testable.
If the model grows to a CNN, swapping in TFLite/ONNX Runtime is a
single-module change behind `OnDeviceClassifier`.

## Architecture

```
core/   ai.ondevice — no dependencies, no Android SDK
  Tensor.kt        matvec, ReLU, softmax, LinearLayer, Mlp
  Quantization.kt  symmetric int8 quantization, integer-accumulation forward
  Preprocess.kt    ARGB -> grayscale -> bilinear resize -> model vector
  Model.kt         ODAI binary model format (magic + version validated)
  Models.kt        ModelStore: download-once install, versioning, SHA-256 verify
  Frames.kt        ImageFrame, simulated camera, gallery loading, smoothed live classifier
  Shapes.kt        synthetic 28x28 shape dataset (deterministic by seed)
  Trainer.kt       mini-batch SGD (softmax + cross-entropy, ReLU backprop)
  Classifier.kt    UI-facing API: top-k predictions + latency benchmark
demo/   CLI: train | classify | benchmark | gallery | frames | models (see examples/run-demo.sh)
models/ shapes-model.bin — bundled trained model (see below)
examples/ sample-triangle.png — gallery sample (triangle, seed 7)
```

## Quick start

```bash
# Classify a shape with the bundled model
gradle :demo:run --args="classify --model models/shapes-model.bin --shape triangle --seed 7"

# Benchmark float32 vs int8
gradle :demo:run --args="benchmark --model models/shapes-model.bin --runs 200"

# Retrain from scratch (deterministic seed 42)
gradle :demo:run --args="train --out models/shapes-model.bin --epochs 60 --per-class 200"

# Run the tests
gradle :core:test
```

Or the full tour: `./examples/run-demo.sh`

## Model management

The app never runs a model file handed to it blindly. `ModelStore`
installs a model only after it parses as ODAI, records its SHA-256,
and refuses downgrades; installing byte-identical bytes again is a
no-op (download-once), and `verify` re-checks hash + parse on startup
so a tampered file falls back to the bundled model instead of
misclassifying.

```bash
gradle :demo:run --args="models install --from models/shapes-model.bin --name shapes-v1 --version 1 --dir model-store"
gradle :demo:run --args="models activate --name shapes-v1 --dir model-store"
gradle :demo:run --args="models verify --name shapes-v1 --dir model-store"
gradle :demo:run --args="models list --dir model-store"
# With no --model, gallery/frames/classify use the store's active model.
```

## Camera + gallery input

On Android, CameraX frames and gallery bitmaps both arrive as ARGB
pixels and go through the shared `preprocessImage` pipeline. The JVM
demo exercises the same path with a deterministic simulated camera
(fresh jitter + noise per frame, like a shaky hand) and real PNG/JPEG
files from the gallery. A single frame can flicker between neighbour
labels, so `LiveClassifier` reports the majority vote over a sliding
window (default 5 frames) — the vote count, not one frame, is what the
UI shows.

```bash
# Gallery: classify a real image file
gradle :demo:run --args="gallery --image examples/sample-triangle.png --model models/shapes-model.bin"

# Live camera simulation: 8 frames, smoothed output per frame
gradle :demo:run --args="frames --frames 8 --labels triangle,triangle,circle,triangle,triangle,triangle,circle,triangle --model models/shapes-model.bin"
```

Or the full tour: `./examples/run-demo.sh`

## Measured results (JVM, 2026-10-23)

Bundled model: 784 inputs -> 64 hidden -> 5 outputs, trained 60 epochs
on 1000 synthetic shapes (seed 42).

| Metric | float32 | int8 |
| --- | --- | --- |
| Test accuracy (250 held-out shapes) | 1.000 | 1.000 |
| Weight size | 201,984 bytes | 50,496 bytes (4x smaller) |
| Inference mean latency (200 runs) | 0.179 ms | 0.269 ms |
| Inference p50 latency | 0.148 ms | 0.160 ms |

Honest note: on the JVM the int8 path is *not* faster — integer math
without SIMD plus per-call input quantization costs more than it saves
at this model size. The int8 win here is the 4x smaller model (APK size,
memory, download) and suitability for phone DSP/NPU kernels, where the
integer path is what actually runs fast. Accuracy is identical either way.

## Tests

31 JUnit tests in `core/src/test`: tensor ops, quantization error
bounds and the 4x size claim, preprocessing contracts, ODAI save/load
round-trip plus bad-magic rejection, top-k ranking, end-to-end shape
classification, benchmark sanity, and decreasing training loss — plus
the model store (download-once, downgrade rejection, corrupt-source
safety, tamper detection, activation/removal) and the live pipeline
(deterministic simulated camera, PNG round-trip, majority smoothing).
CI runs `gradle build` plus a demo smoke test on every push and PR.
Note: on this dev VM the Gradle daemon cannot talk over localhost, so
local verification runs kotlinc + the JUnit console directly; GitHub
Actions CI is the verification of record for the Gradle build.

## License

MIT
