#!/usr/bin/env bash
# End-to-end demo tour for the on-device inference core.
# Requires: JDK 17 + Gradle on PATH (or run the same commands via CI).
# Usage: ./examples/run-demo.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== Classify a triangle with the bundled model =="
gradle :demo:run --no-daemon -q --args="classify --model models/shapes-model.bin --shape triangle --seed 7"

echo
echo "== Benchmark float32 vs int8 (50 runs) =="
gradle :demo:run --no-daemon -q --args="benchmark --model models/shapes-model.bin --runs 50"

echo
echo "== Gallery + live camera simulation =="
gradle :demo:run --no-daemon -q --args="gallery --image examples/sample-triangle.png --model models/shapes-model.bin"
gradle :demo:run --no-daemon -q --args="frames --frames 8 --labels triangle,triangle,circle,triangle,triangle,triangle,circle,triangle --model models/shapes-model.bin"

echo "== Retrain a fresh model from scratch (30 epochs) =="
gradle :demo:run --no-daemon -q --args="train --out /tmp/odai-retrained.bin --epochs 30 --per-class 120"
