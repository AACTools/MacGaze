# MacGaze

Software eye tracking for macOS, built natively for Apple Silicon.

MacGaze turns the built-in FaceTime HD camera into a gaze input device,
no external hardware required. It runs entirely on-device, integrates
with [GazeBridge](https://github.com/AACTools/GazeBridge) for
click/dwell/switch handling via macOS Accessibility, and is designed to
feel like a first-class citizen of the Apple accessibility ecosystem.

## Status

Pre-alpha. Architecture is settled and Phase 0 (camera + Vision pipeline
spike) is in progress. Not yet usable end-to-end. See the
[change log](CHANGELOG.md) once there is one.

## Goals

- **On-device only.** Camera frames never leave the machine. No cloud,
  no telemetry, no analytics — ever.
- **Good-citizen accessibility.** Drives the system pointer; clicks,
  dwells, and switch activation are handled by macOS Accessibility
  (Pointer Control / Dwell Control / Alternate Pointer Actions).
- **Apple Silicon native.** AVCapture + Vision (ANE) + Metal + CoreML.
  Targets M1 and newer.
- **Pluggable.** Conforms to GazeBridge's `TrackerDriver` protocol so
  the same host app can drive an eyetuitive / Hiru or the built-in
  camera.

## Approach

Hybrid gaze estimation:

1. **AVCaptureSession** pulls 1280×720 frames from the FaceTime HD.
2. **Vision framework** (revision 3) on the ANE returns face landmarks
   and pupil positions — robust to the FaceTime HD's aggressive ISP
   denoising.
3. **Metal compute shader** head-pose-normalises the eye crop using
   intrinsics estimated from the camera's documented ~60° field of view
   and a 6-point EPnP solve against a canonical face model.
4. **Pretrained gaze CNN** (MIT-licensed BlazeGaze from
   [WebEyeTrack](https://github.com/RedForestAi/WebEyeTrack), converted
   to CoreML) predicts a base `(pitch, yaw)` gaze vector.
5. **Gaussian RBF personalisation** layers on top, fitted at calibration
   time (9 points), correcting user-specific offsets like angle kappa.
6. **1-Euro filter** smooths the final coordinate stream.

## Repository layout

```
macgaze/
├── Package.swift              SwiftPM package
├── Sources/MacGaze/           Library target
│   ├── Capture/               AVCapture pipeline
│   ├── Vision/                Face + landmark detection
│   ├── Eyes/                  Eye-region extraction
│   ├── Gaze/                  Base CNN + RBF mapping
│   ├── Metal/                 Warp kernel + wrappers
│   └── Smoothing/             1-Euro (reuses GazeBridgeCore)
├── Tests/MacGazeTests/        Unit tests
├── Apps/MacGazeDebug/         Live preview debug app
├── Tools/Eval/                Offline evaluation scripts
└── Data/Evaluation/           Local-only eval data (gitignored)
```

## Building

```sh
swift build
swift test
```

Requires Xcode 16+, macOS 15+, Apple Silicon.

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgements

- [WebEyeTrack](https://github.com/RedForestAi/WebEyeTrack) (Davalos et al.,
  Vanderbilt, 2025) — base CNN architecture and few-shot personalisation
  framework, MIT-licensed.
- [L2CS-Net](https://github.com/AhmedSHAkl/AVEC-2019) (Abdelrahman et al.,
  IEEE TBIOM 2023) — alternative base CNN reference.
- [ETH-XGaze](https://ait.ethz.ch/projects/2020/ETH-XGaze/) (Zhang et al.,
  ECCV 2020) — head-pose normalisation theory and evaluation dataset.
- [MPIIGaze](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild/)
  (Zhang et al., CVPR 2017) — primary evaluation benchmark.
- [GazeBridge](https://github.com/AACTools/GazeBridge) — host application
  and `TrackerDriver` abstraction.
