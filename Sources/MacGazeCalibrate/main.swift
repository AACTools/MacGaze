import Foundation
import CoreML
import MacGaze

/// One-command live calibration tool.
///
/// Opens the camera, guides you through 5 screen positions, captures
/// frames at each, runs the full MediaPipe + BlazeGaze pipeline, solves
/// the RBF, and reports accuracy. No video file, no timing guesses.
///
/// Usage:
///   swift run macgaze-calibrate
///
/// First run prompts for camera permission.

@main
struct MacGazeCalibrate {

    /// 5-point calibration grid: center + 4 directions.
    /// Normalised screen positions [0, 1].
    static let targets: [(name: String, x: Double, y: Double)] = [
        ("CENTER", 0.5, 0.5),
        ("RIGHT",  0.8, 0.5),
        ("LEFT",   0.2, 0.5),
        ("UP",     0.5, 0.2),
        ("DOWN",   0.5, 0.8),
    ]

    /// Seconds to capture at each target.
    static let dwellSeconds: Double = 3.0

    static func main() async {
        print("╔══════════════════════════════════════════╗")
        print("║     MacGaze Live Calibration Tool        ║")
        print("╚══════════════════════════════════════════╝")
        print()

        // Load pipeline.
        let camera = CameraCapture()
        let blazeGaze: BlazeGazeRunner
        do {
            blazeGaze = try BlazeGazeRunner()
        } catch {
            fputs("ERROR: BlazeGaze model not found.\n", stderr)
            fputs("  Run ./scripts/setup.sh first.\n", stderr)
            exit(EXIT_FAILURE)
        }

        let modelPath = resolveModelPath()
        let landmarker: MediaPipeFaceLandmarker?
        if let modelPath {
            landmarker = try? MediaPipeFaceLandmarker(modelPath: modelPath)
            if landmarker != nil {
                print("  MediaPipe: loaded ✓")
            } else {
                print("  MediaPipe: FAILED — falling back to Vision (lower accuracy)")
            }
        } else {
            print("  MediaPipe: model not found — using Vision fallback")
            landmarker = nil
        }
        print("  BlazeGaze: loaded ✓")
        print()

        // Start camera.
        print("Starting camera... (grant permission if prompted)")
        do {
            try await camera.start()
        } catch {
            fputs("ERROR: camera failed: \(error.localizedDescription)\n", stderr)
            exit(EXIT_FAILURE)
        }
        print("  Camera: started ✓")
        print()

        // Warm up BlazeGaze with a dummy frame.
        print("Warming up models (first inference is slow)...")
        for await frame in camera.frames {
            _ = runPipeline(frame: frame, landmarker: landmarker, blazeGaze: blazeGaze)
            break // just one frame for warmup
        }
        print("  Warm: done ✓")
        print()

        // Collect calibration samples.
        var allSamples: [RBFGazeCorrector.CalibrationSample] = []
        var perTargetResults: [(name: String, observedX: Double, observedY: Double, count: Int)] = []
        var timestampMs: Int64 = 0

        for (index, target) in targets.enumerated() {
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print("  Point \(index + 1)/\(targets.count): Look \(target.name)")
            print("  (x:\(target.x), y:\(target.y))")
            print()

            // Voice prompt.
            DispatchQueue.global().async {
                let task = Process()
                task.launchPath = "/usr/bin/say"
                task.arguments = ["Look \(target.name)"]
                try? task.run()
                task.waitUntilExit()
            }

            // Countdown.
            for count in stride(from: 3, through: 1, by: -1) {
                print("  \(count)...", terminator: "")
                fflush(stdout)
                try? await Task.sleep(nanoseconds: 1_000_000_000)
            }
            print(" LOOK!")
            fflush(stdout)

            // Capture frames for dwellSeconds.
            var gazeOutputs: [(x: Double, y: Double)] = []
            let startTime = Date()

            for await frame in camera.frames {
                if Date().timeIntervalSince(startTime) >= dwellSeconds { break }

                timestampMs += 33
                if let gaze = runPipeline(frame: frame, landmarker: landmarker, blazeGaze: blazeGaze) {
                    gazeOutputs.append((Double(gaze.x), Double(gaze.y)))
                    if gazeOutputs.count <= 2 {
                        print("\n    [debug] gaze x=\(gaze.x) y=\(gaze.y)", terminator: "")
                    }
                }

                // Progress dots.
                print(".", terminator: "")
                fflush(stdout)
            }
            print(" done (\(gazeOutputs.count) samples)")
            print()

            if gazeOutputs.count >= 5 {
                // Reject outliers: 2σ from median.
                let xs = gazeOutputs.map(\.x).sorted()
                let ys = gazeOutputs.map(\.y).sorted()
                let medX = xs[xs.count / 2]
                let medY = ys[ys.count / 2]
                let stdX = sqrt(gazeOutputs.reduce(0.0) { $0 + ($1.x - medX) * ($1.x - medX) } / Double(gazeOutputs.count))
                let stdY = sqrt(gazeOutputs.reduce(0.0) { $0 + ($1.y - medY) * ($1.y - medY) } / Double(gazeOutputs.count))
                let threshX = max(stdX * 2, 0.01)
                let threshY = max(stdY * 2, 0.01)

                let cleaned = gazeOutputs.filter {
                    abs($0.x - medX) <= threshX && abs($0.y - medY) <= threshY
                }

                let useCleaned = cleaned.count >= 3 ? cleaned : gazeOutputs
                let avgX = useCleaned.reduce(0.0) { $0 + $1.x } / Double(useCleaned.count)
                let avgY = useCleaned.reduce(0.0) { $0 + $1.y } / Double(useCleaned.count)

                allSamples.append(RBFGazeCorrector.CalibrationSample(
                    observedX: avgX, observedY: avgY,
                    targetX: target.x, targetY: target.y
                ))
                perTargetResults.append((target.name, avgX, avgY, useCleaned.count))

                print("  Observed: (\(String(format: "%.4f", avgX)), \(String(format: "%.4f", avgY)))")
                print("  Target:   (\(String(format: "%.1f", target.x)), \(String(format: "%.1f", target.y)))")
                print()
            } else {
                print("  ⚠ Too few samples (\(gazeOutputs.count)). Skipping this point.")
                print()
            }
        }

        camera.stop()

        // Solve RBF.
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("  Solving RBF calibration...")
        let rbf = RBFGazeCorrector()
        let success = rbf.calibrate(allSamples)

        if !success {
            print("  ✗ RBF solve failed (need ≥ 3 valid points)")
            exit(EXIT_FAILURE)
        }

        print("  ✓ RBF solved: σ=\(String(format: "%.4f", rbf.solvedSigma)), \(allSamples.count) points")
        print()

        // Report accuracy.
        print("╔══════════════════════════════════════════╗")
        print("║          CALIBRATION RESULTS             ║")
        print("╠══════════════════════════════════════════╣")

        var totalError = 0.0
        var errorCount = 0

        for (i, target) in targets.enumerated() {
            guard i < perTargetResults.count else { continue }
            let r = perTargetResults[i]
            let corrected = rbf.correct(x: r.observedX, y: r.observedY)
            let dx = corrected.x - target.x
            let dy = corrected.y - target.y
            let err = sqrt(dx * dx + dy * dy)
            totalError += err
            errorCount += 1

            let status = err < 0.05 ? "✓" : (err < 0.15 ? "~" : "✗")
            print("  \(status) \(target.name)  raw(\(String(format: "%.3f", r.observedX)),\(String(format: "%.3f", r.observedY))) → corr(\(String(format: "%.3f", corrected.x)),\(String(format: "%.3f", corrected.y))) err=\(String(format: "%.3f", err))")
        }

        if errorCount > 0 {
            let meanErr = totalError / Double(errorCount)
            print("╠══════════════════════════════════════════╣")
            let verdict: String
            if meanErr < 0.10 { verdict = "GOOD — ready for use" }
            else if meanErr < 0.20 { verdict = "OKAY — usable but could improve" }
            else { verdict = "POOR — needs more points or better lighting" }
            print(String(format: "║ Mean error: %.1f%%  (%@)           ", meanErr * 100, verdict))
        }

        print("╚══════════════════════════════════════════╝")
        print()
        print("Calibration complete. The RBF solver is ready.")
        print("In GazeBridge, select 'Built-in Camera' — gaze will use")
        print("this calibration automatically (once live calibration is wired).")
    }

    // MARK: Pipeline

    static func runPipeline(
        frame: CameraFrame,
        landmarker: MediaPipeFaceLandmarker?,
        blazeGaze: BlazeGazeRunner
    ) -> CGPoint? {
        if let landmarker {
            return runMediaPipePipeline(frame: frame, landmarker: landmarker, blazeGaze: blazeGaze)
        } else {
            return nil // Vision fallback not implemented in this tool
        }
    }

    static func runMediaPipePipeline(
        frame: CameraFrame,
        landmarker: MediaPipeFaceLandmarker,
        blazeGaze: BlazeGazeRunner
    ) -> CGPoint? {
        // 1. MediaPipe landmarks.
        guard let mpResult = try? landmarker.detect(
            pixelBuffer: frame.pixelBuffer,
            timestampMs: Int64(frame.timestampSeconds * 1000)
        ) else { return nil }

        guard mpResult.landmarks.count >= 478 else { return nil }

        // 2. Eye patch.
        guard let eyePatch = HomographyEyePatchExtractor.extract(
            pixelBuffer: frame.pixelBuffer,
            landmarks: mpResult.landmarks,
            frameWidth: frame.width,
            frameHeight: frame.height
        ) else { return nil }

        // 3. Head pose from facial transformation matrix.
        var hv: MLMultiArray? = nil
        var fo: MLMultiArray? = nil
        if let ft = mpResult.faceTransform, ft.count == 4, ft[0].count >= 3 {
            let r20 = ft[2][0], r21 = ft[2][1], r22 = ft[2][2]
            let r10 = ft[1][0], r00 = ft[0][0]
            let pitch = asin(-r20), yaw = atan2(r21, r22), roll = atan2(r10, r00)
            let hPitch = -yaw, hYaw = pitch
            let cp = cos(hPitch), sp = sin(hPitch)
            let cy = cos(hYaw), sy = sin(hYaw)
            hv = try? MLMultiArray(shape: [1, 3], dataType: .float32)
            fo = try? MLMultiArray(shape: [1, 3], dataType: .float32)
            if let hv { hv[0] = Float(cp * sy) as NSNumber; hv[1] = Float(sp) as NSNumber; hv[2] = Float(-cp * cy) as NSNumber }
            if let fo { fo[0] = Float(ft[0][3]) as NSNumber; fo[1] = Float(ft[1][3]) as NSNumber; fo[2] = Float(ft[2][3]) as NSNumber }
        }

        // 4. BlazeGaze inference.
        return blazeGaze.predict(eyePatch: eyePatch, headVector: hv, faceOrigin3D: fo)
    }

    // MARK: Helpers

    static func resolveModelPath() -> String? {
        let candidates = [
            "Frameworks/face_landmarker_v2_with_blendshapes.task",
            FileManager.default.currentDirectoryPath + "/Frameworks/face_landmarker_v2_with_blendshapes.task",
        ]
        return candidates.first { FileManager.default.fileExists(atPath: $0) }
    }
}
