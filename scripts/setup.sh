#!/usr/bin/env bash
# MacGaze — one-time setup script.
#
# Run this from the macgaze/ root directory after cloning the repo.
#
#   ./scripts/setup.sh
#
# Downloads pre-built binary assets from GitHub releases, then builds
# the Swift package and generates the Xcode project. No Python or
# model conversion needed on the target machine.
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RELEASE_URL="https://github.com/AACTools/MacGaze/releases/download/v0.1.0-assets"

echo "=== MacGaze Setup ==="
echo ""

# 1. Check prerequisites.
echo "[1/5] Checking prerequisites..."
command -v swift >/dev/null || { echo "ERROR: Swift not found. Install Xcode 16+."; exit 1; }
command -v xcodegen >/dev/null || {
    echo "  xcodegen not found. Installing via Homebrew..."
    brew install xcodegen 2>/dev/null || { echo "ERROR: Install xcodegen: brew install xcodegen"; exit 1; }
}
echo "  Swift: $(swift --version 2>&1 | head -1)"
echo "  xcodegen: $(xcodegen --version 2>&1 | head -1)"

# Check architecture.
ARCH=$(uname -m)
if [ "$ARCH" != "arm64" ]; then
    echo "  WARNING: MacGaze is designed for Apple Silicon (arm64)."
    echo "  Current arch: $ARCH. Things may not work correctly."
fi

# 2. Download pre-built assets from GitHub releases.
echo ""
echo "[2/5] Downloading pre-built assets..."

mkdir -p Frameworks
mkdir -p Sources/MacGaze/Gaze

# BlazeGaze CoreML model.
if [ -d "Sources/MacGaze/Gaze/blazegaze.mlmodelc" ]; then
    echo "  BlazeGaze model: already present"
else
    echo "  Downloading BlazeGaze CoreML model..."
    curl -sL "$RELEASE_URL/blazegaze.mlmodelc.zip" -o /tmp/blazegaze.mlmodelc.zip
    unzip -qo /tmp/blazegaze.mlmodelc.zip -d Sources/MacGaze/Gaze/
    rm /tmp/blazegaze.mlmodelc.zip
    echo "  BlazeGaze model: $(du -sh Sources/MacGaze/Gaze/blazegaze.mlmodelc | cut -f1)"
fi

# MediaPipe native library.
if [ -f "Frameworks/libmediapipe.dylib" ]; then
    echo "  libmediapipe.dylib: already present"
else
    echo "  Downloading libmediapipe.dylib (48 MB)..."
    curl -sL "$RELEASE_URL/libmediapipe.dylib" -o Frameworks/libmediapipe.dylib
    echo "  libmediapipe.dylib: $(du -sh Frameworks/libmediapipe.dylib | cut -f1)"
fi

# MediaPipe face landmarker model.
if [ -f "Frameworks/face_landmarker_v2_with_blendshapes.task" ]; then
    echo "  face_landmarker.task: already present"
else
    echo "  Downloading face_landmarker.task (3.6 MB)..."
    curl -sL "$RELEASE_URL/face_landmarker_v2_with_blendshapes.task" -o Frameworks/face_landmarker_v2_with_blendshapes.task
    echo "  face_landmarker.task: $(du -sh Frameworks/face_landmarker_v2_with_blendshapes.task | cut -f1)"
fi

# 3. Build the Swift package.
echo ""
echo "[3/5] Building Swift package..."
swift build 2>&1 | tail -1
echo "  Build: OK"

# 4. Run tests.
echo ""
echo "[4/5] Running unit tests..."
swift test 2>&1 | grep "Executed.*tests" | tail -1
echo "  Tests: OK"

# 5. Generate Xcode project.
echo ""
echo "[5/5] Generating Xcode project..."
xcodegen generate 2>&1 | tail -1
echo "  Xcode project: OK"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo ""
echo "  # Camera + Vision perf test (triggers camera permission):"
echo "  swift run macgaze-smoke --seconds 10"
echo ""
echo "  # Headless pipeline test on a recorded video:"
echo "  swift run macgaze-replay ~/path/to/video.mov --native --verbose"
echo ""
echo "  # With RBF calibration:"
echo "  swift run macgaze-replay ~/path/to/video.mov --native \\"
echo "    --calibrate \"0:2:0.5:0.5 2:4:0.8:0.5 4:6:0.2:0.5 6:8:0.5:0.8\""
echo ""
echo "  # Debug app (camera + BlazeGaze live):"
echo "  xcodegen generate"
echo "  APP=\$(find ~/Library/Developer/Xcode/DerivedData -maxdepth 6 \\"
echo "    -name MacGazeDebug.app -type d -not -path \"*Index.noindex*\" | head -1)"
echo "  open \"\$APP\""
echo ""
echo "See TESTING.md for the full test checklist."
