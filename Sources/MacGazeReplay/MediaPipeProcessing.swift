import Foundation
import AVFoundation
import CoreVideo
import CoreML
import MacGaze

// MARK: MediaPipe-based video processing (extension of MacGazeReplay)

extension MacGazeReplay {

    static func processVideoMediaPipe(
        url: URL,
        blazeGaze: BlazeGazeRunner,
        verbose: Bool,
        landmarksData: [LandmarkFrame],
        calibWindows: [(start: Double, end: Double, targetX: Double, targetY: Double)] = []
    ) async {
        let asset = AVURLAsset(url: url)
        guard let track = asset.tracks(withMediaType: .video).first else {
            fputs("error: no video track\n", stderr); exit(EXIT_FAILURE)
        }
        let reader: AVAssetReader
        do { reader = try AVAssetReader(asset: asset) } catch {
            fputs("error: \(error.localizedDescription)\n", stderr); exit(EXIT_FAILURE)
        }
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ])
        output.alwaysCopiesSampleData = false
        reader.add(output)
        reader.startReading()

        var landmarkByFrame: [Int: LandmarkFrame] = [:]
        for lm in landmarksData { landmarkByFrame[lm.frame] = lm }

        var frameIndex = 0
        var gazeFrames = 0
        var allGaze: [(t: Double, x: Double, y: Double)] = []
        var perSecondGaze: [Int: (sumX: Double, sumY: Double, count: Int)] = [:]

        // Warmup BlazeGaze.
        let warmBuf = try? MLMultiArray(shape: [1, 128, 512, 3], dataType: .float32)
        if let warmBuf { _ = blazeGaze.predict(eyePatch: Self.makeDummyPixelBuffer(), headVector: nil, faceOrigin3D: nil) }

        while reader.status == .reading {
            guard let sampleBuffer = output.copyNextSampleBuffer() else { break }
            guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { continue }
            let pts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
            let timestamp = Double(pts.value) / Double(pts.timescale)
            frameIndex += 1

            guard let lm = landmarkByFrame[frameIndex - 1] else { continue }
            let width = lm.width, height = lm.height

            // 1. Homography eye patch extraction (exact training format).
            guard let eyePatch = HomographyEyePatchExtractor.extract(
                pixelBuffer: pixelBuffer, landmarks: lm.landmarks,
                frameWidth: width, frameHeight: height
            ) else { continue }

            // 2. Head pose from the 4×4 facial transformation matrix.
            var hv: MLMultiArray? = nil
            var fo: MLMultiArray? = nil
            if let ft = lm.faceTransform, ft.count == 4, ft[0].count >= 3 {
                let r20 = ft[2][0], r21 = ft[2][1], r22 = ft[2][2]
                let r10 = ft[1][0], r00 = ft[0][0]
                let pitch = asin(-r20), yaw = atan2(r21, r22), roll = atan2(r10, r00)
                let hPitch = -yaw, hYaw = pitch
                let cp = cos(hPitch), sp = sin(hPitch), cy = cos(hYaw), sy = sin(hYaw)
                hv = try? MLMultiArray(shape: [1, 3], dataType: .float32)
                fo = try? MLMultiArray(shape: [1, 3], dataType: .float32)
                if let hv { hv[0] = Float(cp * sy) as NSNumber; hv[1] = Float(sp) as NSNumber; hv[2] = Float(-cp * cy) as NSNumber }
                if let fo { fo[0] = Float(ft[0][3]) as NSNumber; fo[1] = Float(ft[1][3]) as NSNumber; fo[2] = Float(ft[2][3]) as NSNumber }
            }

            // 3. BlazeGaze inference.
            guard let gaze = blazeGaze.predict(eyePatch: eyePatch, headVector: hv, faceOrigin3D: fo) else { continue }
            gazeFrames += 1
            let gx = Double(gaze.x), gy = Double(gaze.y)
            allGaze.append((timestamp, gx, gy))

            let sec = Int(timestamp)
            var entry = perSecondGaze[sec] ?? (0, 0, 0)
            entry = (entry.sumX + gx, entry.sumY + gy, entry.count + 1)
            perSecondGaze[sec] = entry

            if verbose || frameIndex % 30 == 0 {
                print(String(format: "  frame %4d  t=%5.1fs  gaze=(%.3f, %.3f)", frameIndex, timestamp, gx, gy))
            }
        }

        // Summary.
        print("")
        print("=== MediaPipe + BlazeGaze summary ===")
        print("  Total frames       : \(frameIndex)")
        print("  Gaze predicted     : \(gazeFrames)")

        // Per-second averages.
        if !perSecondGaze.isEmpty {
            print("")
            print("=== Per-second gaze ===")
            print("  second |  avg X   |  avg Y   | samples")
            for sec in perSecondGaze.keys.sorted() {
                let g = perSecondGaze[sec]!
                print(String(format: "  %5ds  |  %.4f  |  %.4f  |  %d", sec, g.sumX / Double(g.count), g.sumY / Double(g.count), g.count))
            }
        }

        // RBF calibration.
        if !calibWindows.isEmpty && !allGaze.isEmpty {
            print("")
            print("=== RBF calibration ===")
            var calibSamples: [MacGaze.RBFGazeCorrector.CalibrationSample] = []
            for w in calibWindows {
                let wg = allGaze.filter { $0.t >= w.start && $0.t < w.end }
                guard !wg.isEmpty else { continue }
                let ax = wg.reduce(0.0) { $0 + $1.x } / Double(wg.count)
                let ay = wg.reduce(0.0) { $0 + $1.y } / Double(wg.count)
                calibSamples.append(MacGaze.RBFGazeCorrector.CalibrationSample(
                    observedX: ax, observedY: ay, targetX: w.targetX, targetY: w.targetY))
                print(String(format: "  window %.1f-%.1fs: observed (%.4f, %.4f) → target (%.1f, %.1f) [%d samples]",
                            w.start, w.end, ax, ay, w.targetX, w.targetY, wg.count))
            }
            if calibSamples.count >= 3 {
                let rbf = MacGaze.RBFGazeCorrector()
                let ok = rbf.calibrate(calibSamples)
                print("\n  RBF solve: \(ok ? "✓ success" : "✗ failed")  σ=\(String(format: "%.4f", rbf.solvedSigma))")
                if ok {
                    print("\n=== Corrected gaze (RBF applied) ===")
                    print("  second |  raw X   |  raw Y   |  corr X  |  corr Y  | target")
                    for sec in perSecondGaze.keys.sorted() {
                        let raw = perSecondGaze[sec]!
                        let rax = raw.sumX / Double(raw.count), ray = raw.sumY / Double(raw.count)
                        let c = rbf.correct(x: rax, y: ray)
                        var dir = ""
                        for w in calibWindows {
                            if Double(sec) >= w.start && Double(sec) < w.end {
                                let dx = abs(c.x - w.targetX), dy = abs(c.y - w.targetY)
                                dir = String(format: "(%.1f,%.1f) err=%.3f", w.targetX, w.targetY, max(dx, dy))
                                break
                            }
                        }
                        print(String(format: "  %5ds  |  %.4f  |  %.4f  |  %.4f  |  %.4f  | %@",
                                    sec, rax, ray, c.x, c.y, dir))
                    }
                    // Mean error.
                    var totalErr = 0.0; var errCount = 0
                    for w in calibWindows {
                        let wg = allGaze.filter { $0.t >= w.start && $0.t < w.end }
                        let corrected = wg.map { rbf.correct(x: $0.x, y: $0.y) }
                        if !corrected.isEmpty {
                            let ax = corrected.reduce(0.0) { $0 + $1.x } / Double(corrected.count)
                            let ay = corrected.reduce(0.0) { $0 + $1.y } / Double(corrected.count)
                            let dx = ax - w.targetX, dy = ay - w.targetY
                            totalErr += (dx * dx + dy * dy).squareRoot()
                            errCount += 1
                        }
                    }
                    if errCount > 0 {
                        let meanErr = totalErr / Double(errCount)
                        print(String(format: "\n  Mean calibration error: %.4f", meanErr))
                        if meanErr < 0.1 { print("  ✓ GOOD — corrected gaze within 10%% of targets") }
                        else if meanErr < 0.2 { print("  ~ OKAY — in the right ballpark") }
                        else { print("  ⚠ POOR — needs more calibration points") }
                    }
                }
            }
        }
    }

    /// Create a dummy 128×512 pixel buffer for BlazeGaze warmup.
    private static func makeDummyPixelBuffer() -> CVPixelBuffer {
        var buf: CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault, 512, 128, kCVPixelFormatType_32BGRA, nil, &buf)
        return buf!
    }
}

