#!/usr/bin/env bash
set -euo pipefail

# Where your code expects the engine:
TARGET_DIR="stockfish"
TARGET_BIN="$TARGET_DIR/stockfish-macos-m1-apple-silicon"

mkdir -p "$TARGET_DIR"

# If the target file already exists and is executable, stop early.
if [ -x "$TARGET_BIN" ]; then
  echo "✔ Found $TARGET_BIN"
  exit 0
fi

# Try to use a system-installed stockfish first (fast path)
if command -v stockfish >/dev/null 2>&1; then
  echo "✔ Using system stockfish: $(command -v stockfish)"
  # Copy to the expected path (not a symlink, for cross-zip portability)
  cp "$(command -v stockfish)" "$TARGET_BIN"
  chmod +x "$TARGET_BIN"
  echo "✔ Placed engine at $TARGET_BIN"
  exit 0
fi

# Otherwise, clone and build from source
echo "→ Building Stockfish from source..."
if [ ! -d "Stockfish" ]; then
  git clone --depth 1 https://github.com/official-stockfish/Stockfish.git
fi

pushd Stockfish/src >/dev/null
# Let make auto-detect the best ARCH; parallel if possible
if command -v nproc >/dev/null 2>&1; then
  MAKE_J="-j$(nproc)"
else
  MAKE_J=""
fi
make build $MAKE_J
popd >/dev/null

# Copy the built binary to the expected path
cp Stockfish/src/stockfish "$TARGET_BIN"
chmod +x "$TARGET_BIN"
echo "✔ Built and placed engine at $TARGET_BIN"
