# MacGaze

Software eye tracking for macOS, built natively for Apple Silicon.

MacGaze turns the built-in FaceTime HD camera into a gaze input device.
It runs entirely on-device and conforms to GazeBridge's `TrackerDriver`
protocol so the same menu-bar app can drive an eyetuitive **or** the
built-in camera.

**Status:** Pipeline verified — 8.1% mean gaze error with 4-point
calibration on recorded video. Native MediaPipe + BlazeGaze + RBF, all
in Swift.

## Architecture

```
CVPixelBuffer
  → MediaPipeFaceLandmarker (libmediapipe.dylib, 478-pt landmarks)
  → HomographyEyePatchExtractor (perspective warp, exact training format)
  → BlazeGazeRunner (CoreML, 725 KB, 17ms inference)
  → RBFGazeCorrector (Gaussian RBF, LAPACK dgesv)
  → corrected (x, y) screen position
```

**Model:** BlazeGaze from [WebEyeTrack](https://github.com/RedForestAi/WebEyeTrack)
(MIT, Vanderbilt 2025). 725 KB CoreML.

**Face landmarks:** Google MediaPipe Tasks via `libmediapipe.dylib`.
478 3D landmarks + 4×4 facial transformation matrix.

## Quick start

```sh
git clone https://github.com/AACTools/MacGaze.git
cd MacGaze
./scripts/setup.sh
```

That's it. The script downloads pre-built binary assets (BlazeGaze CoreML
model, MediaPipe native library) from the [GitHub
release](https://github.com/AACTools/MacGaze/releases/tag/v0.1.0-assets),
builds the package, runs tests, and generates the Xcode project.

No Python, no model conversion, no manual file copying needed.

Requires macOS 15+, Xcode 16+, Apple Silicon.

## Running

**Unit tests (no camera):**
```sh
swift test
```

**Camera + Vision perf test:**
```sh
swift run macgaze-smoke --seconds 10
```

**Headless pipeline test on recorded video:**
```sh
swift run macgaze-replay ~/path/to/video.mov --native --verbose
```

**With RBF calibration:**
```sh
swift run macgaze-replay ~/path/to/video.mov --native \
  --calibrate "0:2:0.5:0.5 2:4:0.8:0.5 4:6:0.2:0.5 6:8:0.5:0.8"
```

**Debug app (camera + BlazeGaze live):**
```sh
xcodegen generate
# Then either open via Xcode or launch directly:
APP=$(find ~/Library/Developer/Xcode/DerivedData -maxdepth 6 \
  -name MacGazeDebug.app -type d -not -path "*Index.noindex*" | head -1)
open "$APP"
```

## Verified results (2026-06-20)

On a recorded video looking at center/right/left/down with 4-point RBF
calibration, native MediaPipe + BlazeGaze pipeline:

| Direction | Corrected (x,y) | Target (x,y) | Error |
|---|---|---|---|
| Center | (0.52, 0.51) | (0.5, 0.5) | 0.024 |
| Right | (0.98, 0.47) | (0.8, 0.5) | 0.18 |
| Left | (0.00, 0.62) | (0.2, 0.5) | 0.20 |
| Down | (0.42, 0.78) | (0.5, 0.8) | 0.08 |
| **Mean** | | | **0.081** |

## Project structure

```
MacGaze/
├── Sources/
│   ├── CMediaPipe/               C bridge for libmediapipe.dylib
│   ├── MacGaze/
│   │   ├── Capture/              CameraCapture (AVCaptureSession)
│   │   ├── Vision/               FaceLandmarkDetector (Apple Vision)
│   │   │                         MediaPipeFaceLandmarker (native)
│   │   ├── Eyes/                 EyePatchExtractor (Vision-based)
│   │   │                         HomographyEyePatchExtractor (MediaPipe)
│   │   ├── Gaze/                 BlazeGazeRunner, HeadPoseEstimator,
│   │   │                         RBFGazeCorrector, CalibrationCollector
│   │   └── Tracker.swift         MacGazeTracker: TrackerDriver
│   ├── MacGazeSmoke/             CLI perf tool
│   ├── MacGazeReplay/            CLI pipeline test + calibration
│   └── MacGazeEval/              CLI evaluation tool
├── Tests/MacGazeTests/           21 unit tests
├── Apps/MacGazeDebug/            SwiftUI debug app
├── Tools/Conversion/             BlazeGaze Keras→CoreML + MediaPipe tools
├── Frameworks/                   libmediapipe.dylib + .task model (gitignored)
├── scripts/setup.sh              One-time setup script
└── TESTING.md                    Full test checklist
```

## Setup details

The `./scripts/setup.sh` script downloads pre-built assets from the
[GitHub release](https://github.com/AACTools/MacGaze/releases/tag/v0.1.0-assets):

| Asset | Size | Source | License |
|---|---|---|---|
| `blazegaze.mlmodelc.zip` | 577 KB | Converted from WebEyeTrack (MIT) | MIT |
| `libmediapipe.dylib` | 48 MB | From Python `mediapipe` package | Apache 2.0 |
| `face_landmarker_v2_with_blendshapes.task` | 3.6 MB | Google MediaPipe models | Apache 2.0 |

The BlazeGaze model can be regenerated from source via the
`.github/workflows/build-assets.yml` GitHub Actions workflow (triggers on
tag push). The conversion script (`Tools/Conversion/convert_blazegaze.py`)
is kept for reproducibility but is not needed for normal use.

## Known issues

1. **MacGazeDebug.app crashes on some M1 Macs** — Metal telemetry bug in
   GPUToolsCapture/RenderBox. CLI tools (`macgaze-smoke`, `macgaze-replay`)
   are unaffected. Likely works on M2+.
2. **MediaPipe latency ~170ms/frame** — BGRA→RGB pixel conversion is the
   bottleneck. Optimizable with vImage.
3. **4-point calibration only** — accuracy will improve with 9-point.

## Relationship to GazeBridge

Both projects live as siblings:
```
gaze/
├── gazebridge/    ← host menu-bar app (AACTools/GazeBridge)
├── macgaze/       ← this project (AACTools/MacGaze)
└── Gaze.xcworkspace   ← open this in Xcode (contains both projects)
```

MacGaze depends on `GazeBridgeCore` (the shared library). Clone both
repos under the same parent directory.

## License

MIT — see [LICENSE](LICENSE).
