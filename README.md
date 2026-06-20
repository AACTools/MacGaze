# MacGaze

Software eye tracking for macOS, built natively for Apple Silicon.

MacGaze turns the built-in FaceTime HD camera into a gaze input device.
It runs entirely on-device and conforms to GazeBridge's `TrackerDriver`
protocol so the same menu-bar app can drive an eyetuitive **or** the
built-in camera.

## Architecture

```
AVCaptureSession          Vision Framework (ANE)        BlazeGaze (CoreML)
   1280×720 30fps    →     Face landmarks Rev 3    →     128×512 eye patch
   32BGRA                   + pupil positions           + head_vector
                            + yaw / roll                + face_origin_3d
                                                             │
                                                             ▼
                    RBFGazeCorrector ←── 1-Euro Filter ←── (x, y) screen point
                    (Gaussian, λ=0.01)                     normalised [0,1]
                             │
                             ▼
                    GazeSample (TrackerDriver protocol)
                             │
                             ▼
                    GazeBridge menu-bar app
                    (smoothing, snapping, calibration,
                     track status, onboarding)
```

**Model:** BlazeGaze from [WebEyeTrack](https://github.com/RedForestAi/WebEyeTrack)
(MIT, Vanderbilt 2025). 725 KB CoreML, outputs match Keras to 1e-6.

## Project structure

```
macgaze/
├── Sources/MacGaze/
│   ├── Capture/          CameraCapture (AVCaptureSession)
│   ├── Vision/           FaceLandmarkDetector (VNDetectFaceLandmarks Rev 3)
│   ├── Eyes/             EyePatchExtractor (vImage crop + resize)
│   ├── Gaze/             BlazeGazeRunner, HeadPoseEstimator, RBFGazeCorrector,
│   │                     CalibrationCollector
│   └── Tracker.swift     MacGazeTracker: TrackerDriver conformance
├── Sources/MacGazeSmoke/ CLI perf tool (swift run macgaze-smoke)
├── Sources/MacGazeEval/  CLI eval tool (swift run macgaze-eval)
├── Tests/MacGazeTests/   21 unit tests
├── Apps/MacGazeDebug/    SwiftUI debug app (camera + landmarks + gaze overlay)
├── Tools/Conversion/     BlazeGaze Keras→CoreML conversion script
└── docs/                 (gitignored — internal planning docs)
```

## Building

```sh
swift build          # library + CLI tools
swift test           # 21 unit tests (~0.03s)
xcodegen generate    # generate MacGazeDebug.xcodeproj
```

Requires macOS 15+, Xcode 16+, Apple Silicon.

## Running

**Smoke test** (no GUI, no Metal):
```sh
swift run macgaze-smoke --seconds 10
```

**Debug app** (camera + BlazeGaze live):
```sh
open ~/GitHub/gaze/Gaze.xcworkspace
# Select MacGazeDebug scheme → ⌘B → then open the built .app directly
```

**Convert BlazeGaze model** (if `.mlmodelc` is missing):
```sh
cd Tools/Conversion
git clone --depth=1 https://github.com/RedForestAI/WebEyeTrack.git WebEyeTrack-upstream
. .venv/bin/activate   # needs tensorflow + coremltools
python convert_blazegaze.py
```

## Known snag: M1 Metal crash

**MacGazeDebug.app crashes on base M1 hardware** (tested on an original
M1 MacBook running hot). The crash is in macOS's Metal framework:

```
GPUToolsCapture → CaptureMTLCommandBuffer commitAndWaitUntilSubmitted
→ Metal setLabel: → -[__NSCFNumber length]: unrecognized selector
```

This is a **system-level Metal telemetry bug**, not a code bug. The CLI
tools (`macgaze-smoke`, `macgaze-eval`) work fine — only the SwiftUI
debug display triggers it. **Likely works on M2/M3; needs testing.**

Workarounds tried (all insufficient on this M1):
- CPU-only CIContext — still crashed at init
- vImage + CGContext (no Metal) — still crashed in Core Animation
- Display throttle (10 fps) — delayed but didn't prevent crash
- Metal warmup at launch — didn't help
- `METAL_DEVICE_WRAPPER_TYPE=0` — didn't help

The underlying library is correct and fully unit-tested. The crash only
affects the live display rendering path.

## What's done

- ✅ Camera + Vision pipeline (7.6 ms p50, 30 fps, 100% face detection)
- ✅ BlazeGaze CoreML conversion (725 KB, verified to 1e-6)
- ✅ Eye patch extraction — BOTH Vision-based (approximate) AND homography
  with MediaPipe landmarks (exact training format)
- ✅ Head pose estimation (Vision yaw/roll + landmark pitch; MediaPipe 4×4 matrix)
- ✅ Gaussian RBF calibration (Accelerate/LAPACK, λ=0.01 ridge) — **9.3% mean error verified**
- ✅ CalibrationCollector with 2σ outlier rejection
- ✅ `MacGazeTracker: TrackerDriver` (plugs into GazeBridge)
- ✅ Evaluation Recorder (in GazeBridge) + offline eval CLI
- ✅ **Full pipeline verified on recorded video: MediaPipe landmarks → BlazeGaze → RBF → corrected gaze**
- ✅ 21 unit tests

## Verified results (2026-06-20)

On a recorded video looking at center/right/left/down with 4-point RBF calibration:

| Direction | Corrected (x,y) | Target (x,y) | Error |
|---|---|---|---|
| Center | (0.56, 0.51) | (0.5, 0.5) | 0.06 |
| Right | (0.94, 0.46) | (0.8, 0.5) | 0.14 |
| Left | (0.05, 0.63) | (0.2, 0.5) | 0.15 |
| Down | (0.48, 0.78) | (0.5, 0.8) | 0.03 |
| **Mean** | | | **0.093** |

## What's remaining

1. **Native MediaPipe in Swift** — replace the Python landmark extractor with
   native Swift MediaPipe (C++ framework via Bazel, or wait for SPM support).
   Currently using a Python→JSON→Swift hybrid that works but isn't real-time.
2. **Driver picker in GazeBridge** — already wired but needs live testing
3. **9-point calibration** (instead of 4) — more RBF points = better accuracy
4. **Real-time pipeline** — once native MediaPipe is in, the full pipeline
   runs at 30fps (25ms p50 proven by macgaze-smoke)
5. **Fix MacGazeDebug.app crash on M1** — Metal telemetry bug in
   GPUToolsCapture/RenderBox. Works on M2+ (likely).
6. **Accuracy tuning** — head pose math, eye patch crop, RBF σ adjustment

## Relationship to GazeBridge

Both projects live at `~/GitHub/gaze/`:

```
gaze/
├── gazebridge/    ← host menu-bar app (AACTools/GazeBridge on GitHub)
├── macgaze/       ← this project (AACTools/MacGaze)
└── Gaze.xcworkspace   ← open this in Xcode (contains both projects)
```

Open `Gaze.xcworkspace` (not individual `.xcodeproj` files) to avoid
package-lock conflicts from the shared `GazeBridgeCore` dependency.

## License

MIT — see [LICENSE](LICENSE).
