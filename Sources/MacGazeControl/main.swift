import Foundation
import CoreGraphics
import CoreML
import ApplicationServices
import MacGaze

/// Live gaze → mouse control via CLI.
///
/// 1. Quick 3-point calibration (center, left, right)
/// 2. Then moves the system cursor with your eyes
///
/// Press Ctrl+C to stop.
///
/// Usage:
///   swift run macgaze-control
///   swift run macgaze-control --skip-calibration  # reuse last calibration

@main
struct MacGazeControl {
    static func main() async {
        print("╔══════════════════════════════════════════╗")
        print("║   MacGaze Live Cursor Control            ║")
        print("╚══════════════════════════════════════════╝")
        print()

        let args = CommandLine.arguments
        let skipCalibration = args.contains("--skip-calibration")

        // Check accessibility permission.
        let trusted = AXIsProcessTrustedWithOptions(
            ["AXTrustedCheckOptionPrompt": true] as CFDictionary
        )
        if !trusted {
            print("⚠ Accessibility permission required to move the cursor.")
            print("  Grant it in System Settings → Privacy & Security → Accessibility")
            print("  Then re-run this tool.")
            exit(EXIT_FAILURE)
        }
        print("  Accessibility: ✓")

        // Load pipeline.
        let camera = CameraCapture()
        let blazeGaze: BlazeGazeRunner
        do {
            blazeGaze = try BlazeGazeRunner()
        } catch {
            print("ERROR: BlazeGaze model not found. Run ./scripts/setup.sh")
            exit(EXIT_FAILURE)
        }
        let modelPath = resolveModelPath()
        guard let landmarker = modelPath.flatMap({ try? MediaPipeFaceLandmarker(modelPath: $0) }) else {
            print("ERROR: MediaPipe model not found. Run ./scripts/setup.sh")
            exit(EXIT_FAILURE)
        }
        print("  MediaPipe: ✓")
        print("  BlazeGaze: ✓")

        // Start camera.
        do {
            try await camera.start()
        } catch {
            print("ERROR: Camera failed — \(error.localizedDescription)")
            exit(EXIT_FAILURE)
        }
        print("  Camera: ✓")
        defer { camera.stop() }

        // Warm up.
        print("  Warming up...", terminator: "")
        fflush(stdout)
        for await frame in camera.frames {
            _ = runPipeline(frame: frame, landmarker: landmarker, blazeGaze: blazeGaze)
            break
        }
        print(" done ✓")
        print()

        // Calibration.
        let rbf = RBFGazeCorrector()
        if !skipCalibration {
            let targets: [(name: String, x: Double, y: Double)] = [
                ("center", 0.5, 0.5),
                ("right",  0.8, 0.5),
                ("left",   0.2, 0.5),
            ]
            var samples: [RBFGazeCorrector.CalibrationSample] = []
            var timestampMs: Int64 = 0

            for (i, target) in targets.enumerated() {
                print("  Calibrate \(i + 1)/\(targets.count): Look \(target.name)")
                say("Look \(target.name)")

                for count in stride(from: 3, through: 1, by: -1) {
                    print("  \(count)...", terminator: "")
                    fflush(stdout)
                    try? await Task.sleep(nanoseconds: 1_000_000_000)
                }
                print(" LOOK!")
                say("Look")
                fflush(stdout)

                var outputs: [(x: Double, y: Double)] = []
                let start = Date()
                for await frame in camera.frames {
                    if Date().timeIntervalSince(start) >= 2.0 { break }
                    timestampMs += 33
                    if let g = runPipeline(frame: frame, landmarker: landmarker, blazeGaze: blazeGaze) {
                        outputs.append((Double(g.x), Double(g.y)))
                    }
                    print(".", terminator: "")
                    fflush(stdout)
                }

                if outputs.count >= 3 {
                    let xs = outputs.map(\.x).sorted()
                    let ys = outputs.map(\.y).sorted()
                    let medX = xs[xs.count / 2]
                    let medY = ys[ys.count / 2]
                    let cleaned = outputs.filter {
                        abs($0.x - medX) < 0.5 && abs($0.y - medY) < 0.5
                    }
                    let use = cleaned.count >= 2 ? cleaned : outputs
                    let avgX = use.reduce(0.0, { $0 + $1.x }) / Double(use.count)
                    let avgY = use.reduce(0.0, { $0 + $1.y }) / Double(use.count)
                    samples.append(RBFGazeCorrector.CalibrationSample(
                        observedX: avgX, observedY: avgY,
                        targetX: target.x, targetY: target.y
                    ))
                    print(" done (raw: \(String(format: "%.2f", avgX)), \(String(format: "%.2f", avgY)))")
                } else {
                    print(" skipped (too few samples)")
                }
            }

            if rbf.calibrate(samples) {
                print("  RBF calibrated ✓ (\(samples.count) points)")
            } else {
                print("  ⚠ RBF failed — using raw gaze (less accurate)")
            }
            print()
        }

        // Get screen size.
        let screenW = CGDisplayPixelsWide(CGMainDisplayID())
        let screenH = CGDisplayPixelsHigh(CGMainDisplayID())
        print("  Screen: \(screenW)×\(screenH)")
        print()
        print("╔══════════════════════════════════════════╗")
        print("║  LIVE CURSOR CONTROL — Ctrl+C to stop    ║")
        print("╚══════════════════════════════════════════╝")
        print()
        say("Cursor control active")

        // Smoothing.
        var smootherX = OneEuroFilter()
        var smootherY = OneEuroFilter()
        smootherX.configure(minCutoff: 1.0, beta: 0.02)
        smootherY.configure(minCutoff: 1.0, beta: 0.02)

        var frameCount = 0
        let startTime = Date()
        var lastFpsTime = startTime
        var fpsFrames = 0
        var prevTimestamp: CFAbsoluteTime = 0

        // Main loop: camera → gaze → cursor.
        for await frame in camera.frames {
            let nowTs = CFAbsoluteTimeGetCurrent()
            let captureAge = prevTimestamp > 0 ? (nowTs - prevTimestamp) * 1000 : 0
            prevTimestamp = nowTs

            let nowMs = Int64(Date().timeIntervalSince(startTime) * 1000)
            frameCount += 1
            fpsFrames += 1

            let pipeStart = CFAbsoluteTimeGetCurrent()
            guard let gaze = runPipeline(frame: frame, landmarker: landmarker, blazeGaze: blazeGaze) else {
                let pipeMs = (CFAbsoluteTimeGetCurrent() - pipeStart) * 1000
                print(String(format: "  [%@] NO FACE  pipe=%.0fms  frameGap=%.0fms",
                             timeString(from: startTime), pipeMs, captureAge))
                fflush(stdout)
                continue
            }
            let pipeMs = (CFAbsoluteTimeGetCurrent() - pipeStart) * 1000

            // RBF correct.
            let corrected = rbf.correct(x: Double(gaze.x), y: Double(gaze.y))

            // Clamp to [0, 1].
            let cx = max(0.0, min(1.0, corrected.x))
            let cy = max(0.0, min(1.0, corrected.y))

            // Smooth.
            let sx = smootherX.filter(value: cx, timestampMs: nowMs)
            let sy = smootherY.filter(value: cy, timestampMs: nowMs)

            // Map to screen pixels.
            let px = sx * Double(screenW)
            let py = sy * Double(screenH)

            // Move cursor.
            if let event = CGEvent(
                mouseEventSource: nil,
                mouseType: .mouseMoved,
                mouseCursorPosition: CGPoint(x: px, y: py),
                mouseButton: .center
            ) {
                event.post(tap: .cghidEventTap)
            }

            // Print every frame with timestamps.
            print(String(format: "  [%@] pipe=%.0fms  gap=%.0fms  raw(%.2f,%.2f) → corr(%.2f,%.2f) → screen(%.0f,%.0f)",
                         timeString(from: startTime), pipeMs, captureAge,
                         gaze.x, gaze.y, sx, sy, px, py))
            fflush(stdout)
        }
    }

    // MARK: Pipeline

    static func runPipeline(
        frame: CameraFrame,
        landmarker: MediaPipeFaceLandmarker,
        blazeGaze: BlazeGazeRunner
    ) -> CGPoint? {
        guard let mpResult = try? landmarker.detect(
            pixelBuffer: frame.pixelBuffer,
            timestampMs: Int64(frame.timestampSeconds * 1000)
        ) else { return nil }
        guard mpResult.landmarks.count >= 478 else { return nil }

        guard let eyePatch = HomographyEyePatchExtractor.extract(
            pixelBuffer: frame.pixelBuffer,
            landmarks: mpResult.landmarks,
            frameWidth: frame.width,
            frameHeight: frame.height
        ) else { return nil }

        // Compute head pose from MediaPipe facial transformation matrix
        // (same as macgaze-calibrate — without this, BlazeGaze predictions
        // are in a completely different coordinate space).
        var headVector: MLMultiArray? = nil
        var faceOrigin: MLMultiArray? = nil
        if let ft = mpResult.faceTransform, ft.count == 4, ft[0].count >= 3 {
            let r20 = ft[2][0], r21 = ft[2][1], r22 = ft[2][2]
            let r10 = ft[1][0], r00 = ft[0][0]
            let pitch = asin(-r20), yaw = atan2(r21, r22), roll = atan2(r10, r00)
            let hPitch = -yaw, hYaw = pitch
            let cp = cos(hPitch), sp = sin(hPitch)
            let cy = cos(hYaw), sy = sin(hYaw)
            headVector = try? MLMultiArray(shape: [1, 3], dataType: .float32)
            faceOrigin = try? MLMultiArray(shape: [1, 3], dataType: .float32)
            if let hv = headVector {
                hv[0] = Float(cp * sy) as NSNumber
                hv[1] = Float(sp) as NSNumber
                hv[2] = Float(-cp * cy) as NSNumber
            }
            if let fo = faceOrigin {
                fo[0] = Float(ft[0][3]) as NSNumber
                fo[1] = Float(ft[1][3]) as NSNumber
                fo[2] = Float(ft[2][3]) as NSNumber
            }
        }

        return blazeGaze.predict(eyePatch: eyePatch, headVector: headVector, faceOrigin3D: faceOrigin)
    }

    static func say(_ text: String) {
        DispatchQueue.global().async {
            let task = Process()
            task.launchPath = "/usr/bin/say"
            task.arguments = [text]
            try? task.run()
            task.waitUntilExit()
        }
    }

    static func timeString(from start: Date) -> String {
        let elapsed = Date().timeIntervalSince(start)
        let mins = Int(elapsed) / 60
        let secs = elapsed - Double(mins) * 60
        return String(format: "%d:%04.1f", mins, secs)
    }

    static func resolveModelPath() -> String? {
        let candidates = [
            "Frameworks/face_landmarker_v2_with_blendshapes.task",
            FileManager.default.currentDirectoryPath + "/Frameworks/face_landmarker_v2_with_blendshapes.task",
        ]
        return candidates.first { FileManager.default.fileExists(atPath: $0) }
    }
}

/// Minimal 1-Euro filter (avoids GazeBridgeCore dependency for this CLI tool).
final class OneEuroFilter {
    private var prevValue: Double = 0
    private var prevDeriv: Double = 0
    private var prevTimeMs: Int64 = 0
    private var minCutoff: Double = 1.0
    private var beta: Double = 0.02

    func configure(minCutoff: Double, beta: Double) {
        self.minCutoff = minCutoff
        self.beta = beta
    }

    func filter(value: Double, timestampMs: Int64) -> Double {
        if prevTimeMs == 0 {
            prevTimeMs = timestampMs
            prevValue = value
            return value
        }
        let dt = max(Double(timestampMs - prevTimeMs) / 1000.0, 0.001)
        let alpha = 1.0 / (1.0 + 1.0 / (dt * minCutoff))
        let result = alpha * value + (1 - alpha) * prevValue
        prevTimeMs = timestampMs
        prevValue = result
        return result
    }
}
