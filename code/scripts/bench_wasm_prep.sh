#!/usr/bin/env bash
# Prepare a serving directory for bench_wasm.html and start a local HTTP server.
# Copies fold-0 student ONNX files into ./bench_wasm_serve/ with the flat names
# the HTML page expects, then starts an HTTP server on port 8080.
#
# Usage: bash scripts/bench_wasm_prep.sh
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SERVE_DIR="$REPO_ROOT/bench_wasm_serve"

mkdir -p "$SERVE_DIR"
cd "$SERVE_DIR"

# Copy fold-0 student ONNX files with flat names
for arch in mobilenetv3_small efficientnet_lite0 mobilevit_xxs; do
  for kind in fp32 int8; do
    src="$REPO_ROOT/exports/${arch}_fold0_${kind}.onnx"
    dst="$SERVE_DIR/${arch}_${kind}.onnx"
    if [ -f "$src" ]; then
      cp -v "$src" "$dst"
      # Copy .data sidecar if present (PyTorch 2.8+ external weights)
      [ -f "${src}.data" ] && cp -v "${src}.data" "${dst}.data"
    else
      echo "WARNING: $src not found"
    fi
  done
done

# Copy the bench HTML
cp "$REPO_ROOT/scripts/bench_wasm.html" "$SERVE_DIR/index.html"

echo ""
echo "Serving from: $SERVE_DIR"
echo "Open in Chrome (NOT Safari — Safari's WASM SIMD is slower):"
echo "  http://localhost:8080/"
echo ""
echo "After the run finishes:"
echo "  1. Click the 'Copy' equivalent — select the JSON block on the page"
echo "  2. Save it to: $REPO_ROOT/runs/3fold/lat_wasm.json"
echo "  3. Ctrl+C to stop this server."
echo ""

python3 -m http.server 8080
