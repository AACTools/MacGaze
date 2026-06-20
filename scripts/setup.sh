#!/usr/bin/env bash
# MacGaze — one-time setup script.
#
# Run this from the macgaze/ root directory after cloning the repo.
#
#   ./scripts/setup.sh
#
# Installs all dependencies, converts the BlazeGaze model, and copies
# the MediaPipe native library into place. After this, you can:
#
#   swift test                    # run unit tests
#   swift run macgaze-smoke       # camera + Vision perf test
#   swift run macgaze-replay      # headless pipeline test
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "=== MacGaze Setup ==="
echo ""

# 1. Check prerequisites.
echo "[1/6] Checking prerequisites..."
command -v swift >/dev/null || { echo "ERROR: Swift not found. Install Xcode."; exit 1; }
command -v xcodegen >/dev/null || { echo "ERROR: xcodegen not found. Install: brew install xcodegen"; exit 1; }
echo "  Swift: $(swift --version 2>&1 | head -1)"
echo "  xcodegen: $(xcodegen --version 2>&1 | head -1)"

# 2. Build the Swift package (verifies GazeBridgeCore dependency resolves).
echo ""
echo "[2/6] Building Swift package..."
swift build 2>&1 | tail -1
echo "  Build: OK"

# 3. Convert BlazeGaze model (if not already done).
BLAZEGAZE_MLMODELC="Sources/MacGaze/Gaze/blazegaze.mlmodelc"
if [ -d "$BLAZEGAZE_MLMODELC" ]; then
    echo ""
    echo "[3/6] BlazeGaze model already exists, skipping conversion."
else
    echo ""
    echo "[3/6] Converting BlazeGaze model..."
    CONV_DIR="Tools/Conversion"

    # Clone WebEyeTrack if needed.
    if [ ! -d "$CONV_DIR/WebEyeTrack-upstream" ]; then
        echo "  Cloning WebEyeTrack..."
        git clone --depth=1 https://github.com/RedForestAI/WebEyeTrack.git "$CONV_DIR/WebEyeTrack-upstream"
    fi

    # Set up Python venv if needed.
    if [ ! -d "$CONV_DIR/.venv" ]; then
        echo "  Creating Python venv..."
        python3 -m venv "$CONV_DIR/.venv"
        "$CONV_DIR/.venv/bin/pip" install --upgrade pip -q
        "$CONV_DIR/.venv/bin/pip" install tensorflow coremltools -q
    fi

    # Convert the model.
    echo "  Converting Keras → CoreML..."
    "$CONV_DIR/.venv/bin/python" "$CONV_DIR/convert_blazegaze.py"

    # Compile to .mlmodelc.
    if [ -f "Sources/MacGaze/Gaze/blazegaze.mlpackage" ]; then
        echo "  Compiling .mlpackage → .mlmodelc..."
        xcrun coremlc compile "Sources/MacGaze/Gaze/blazegaze.mlpackage" "Sources/MacGaze/Gaze"
    fi

    if [ -d "$BLAZEGAZE_MLMODELC" ]; then
        echo "  BlazeGaze model: OK ($(du -sh $BLAZEGAZE_MLMODELC | cut -f1))"
    else
        echo "  ERROR: BlazeGaze model conversion failed."
        exit 1
    fi
fi

# 4. Copy MediaPipe native library.
echo ""
echo "[4/6] Setting up MediaPipe native library..."
DYLIB_INSTALLED=""
TASK_INSTALLED=""

# Check if already in place.
if [ -f "Frameworks/libmediapipe.dylib" ]; then
    DYLIB_INSTALLED="yes"
fi
if [ -f "Frameworks/face_landmarker_v2_with_blendshapes.task" ]; then
    TASK_INSTALLED="yes"
fi

# Try to get from the Python venv (if it exists).
if [ -z "$DYLIB_INSTALLED" ] && [ -d "Tools/Conversion/.venv" ]; then
    DYLIB_SRC=$(find "Tools/Conversion/.venv" -name "libmediapipe.dylib" -path "*/mediapipe/*" 2>/dev/null | head -1)
    if [ -n "$DYLIB_SRC" ]; then
        mkdir -p Frameworks
        cp "$DYLIB_SRC" Frameworks/libmediapipe.dylib
        DYLIB_INSTALLED="yes"
        echo "  Copied libmediapipe.dylib from Python venv."
    fi
fi

if [ -z "$DYLIB_INSTALLED" ]; then
    echo "  libmediapipe.dylib not found."
    echo "  Install via: pip install mediapipe"
    echo "  Then copy from: <venv>/lib/python*/site-packages/mediapipe/tasks/c/libmediapipe.dylib"
    echo "  OR run: Tools/Conversion/.venv/bin/pip install mediapipe"
    if [ -d "Tools/Conversion/.venv" ]; then
        echo "  Installing mediapipe into existing venv..."
        Tools/Conversion/.venv/bin/pip install mediapipe -q
        DYLIB_SRC=$(find "Tools/Conversion/.venv" -name "libmediapipe.dylib" -path "*/mediapipe/*" 2>/dev/null | head -1)
        if [ -n "$DYLIB_SRC" ]; then
            mkdir -p Frameworks
            cp "$DYLIB_SRC" Frameworks/libmediapipe.dylib
            DYLIB_INSTALLED="yes"
            echo "  Copied libmediapipe.dylib."
        fi
    fi
fi

# Copy the .task model.
if [ -z "$TASK_INSTALLED" ]; then
    TASK_SRC=""
    if [ -d "Tools/Conversion/WebEyeTrack-upstream" ]; then
        TASK_SRC="Tools/Conversion/WebEyeTrack-upstream/python/webeyetrack/model_weights/face_landmarker_v2_with_blendshapes.task"
    fi
    if [ -n "$TASK_SRC" ] && [ -f "$TASK_SRC" ]; then
        mkdir -p Frameworks
        cp "$TASK_SRC" Frameworks/face_landmarker_v2_with_blendshapes.task
        TASK_INSTALLED="yes"
        echo "  Copied face_landmarker.task from WebEyeTrack."
    else
        echo "  Downloading face_landmarker.task from Google..."
        mkdir -p Frameworks
        curl -sL "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task" -o Frameworks/face_landmarker_v2_with_blendshapes.task
        if [ -f Frameworks/face_landmarker_v2_with_blendshapes.task ]; then
            TASK_INSTALLED="yes"
            echo "  Downloaded face_landmarker.task."
        fi
    fi
fi

if [ -n "$DYLIB_INSTALLED" ] && [ -n "$TASK_INSTALLED" ]; then
    echo "  MediaPipe: OK"
else
    echo "  WARNING: MediaPipe not fully set up. Native landmark detection won't work."
    echo "  The Vision-based fallback pipeline still works."
fi

# 5. Run tests.
echo ""
echo "[5/6] Running unit tests..."
swift test 2>&1 | grep "Executed"
echo "  Tests: OK"

# 6. Generate Xcode project.
echo ""
echo "[6/6] Generating Xcode project..."
xcodegen generate 2>&1 | tail -1
echo "  Xcode project: OK"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Quick test (no camera needed):"
echo "  swift run macgaze-smoke --seconds 5"
echo ""
echo "With a recorded video:"
echo "  swift run macgaze-replay ~/path/to/video.mov --native --verbose"
echo ""
echo "Open in Xcode:"
echo "  open ~/GitHub/gaze/Gaze.xcworkspace"
echo ""
echo "See TESTING.md for the full test checklist."
