import Foundation
import AVFoundation
import CoreVideo
import CoreML
import ImageIO
import MacGaze

/// Headless pipeline test — replays a recorded video through the full
/// MacGaze pipeline and prints gaze predictions to stdout.
///
/// No SwiftUI, no Core Animation, no Metal display = no crash on M1.
///
/// Usage:
///   swift run macgaze-replay <video-file> [--verbose] [--dump-patches]
///   swift run macgaze-replay <video-file> --landmarks <landmarks.json>
///   swift run macgaze-replay <video-file> --landmarks <landmarks.json> --calibrate "..."
///
/// When --landmarks is provided, uses MediaPipe 478-point landmarks
/// (extracted by extract_landmarks.py) instead of Apple Vision. This
/// produces the exact input format BlazeGaze was trained on.

@main
struct MacGazeReplay {
    static func main() async {
        let args = CommandLine.arguments

        // Handle help first.
        if args.contains("-h") || args.contains("--help") {
            print(usage)
            exit(EXIT_SUCCESS)
        }

        guard args.count >= 2 else {
            print(usage)
            exit(EXIT_FAILURE)
        }
        let filePath = args[1]
        let verbose = args.contains("--verbose")
        let dumpPatches = args.contains("--dump-patches")
        let dumpDir = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent("patch-dump")

        // Parse calibration windows: --calibrate "start:end:tx:ty start:end:tx:ty ..."
        var calibWindows: [(start: Double, end: Double, targetX: Double, targetY: Double)] = []
        if let calibIdx = args.firstIndex(of: "--calibrate"), calibIdx + 1 < args.count {
            let spec = args[calibIdx + 1]
            for segment in spec.split(separator: " ") {
                let parts = segment.split(separator: ":")
                if parts.count == 4,
                   let s = Double(parts[0]), let e = Double(parts[1]),
                   let tx = Double(parts[2]), let ty = Double(parts[3]) {
                    calibWindows.append((s, e, tx, ty))
                }
            }
            if !calibWindows.isEmpty {
                print("Calibration mode: \(calibWindows.count) windows")
                for w in calibWindows {
                    print(String(format: "  %.1f-%.1fs → target (%.1f, %.1f)", w.start, w.end, w.targetX, w.targetY))
                }
                print("")
            }
        }

        let url = URL(fileURLWithPath: filePath)
        guard FileManager.default.fileExists(atPath: url.path) else {
            fputs("error: file not found: \(url.path)\n", stderr)
            exit(EXIT_FAILURE)
        }

        // Detect file type.
        let ext = url.pathExtension.lowercased()
        let isImage = ["jpg", "jpeg", "png", "heic"].contains(ext)

        // Parse --landmarks <json-file>
        var landmarksData: [LandmarkFrame]? = nil
        if let lmIdx = args.firstIndex(of: "--landmarks"), lmIdx + 1 < args.count {
            let lmPath = args[lmIdx + 1]
            if FileManager.default.fileExists(atPath: lmPath) {
                if let data = try? Data(contentsOf: URL(fileURLWithPath: lmPath)),
                   let decoded = try? JSONDecoder().decode([LandmarkFrame].self, from: data) {
                    landmarksData = decoded
                    print("Loaded \(decoded.count) frames of MediaPipe landmarks from \(URL(fileURLWithPath: lmPath).lastPathComponent)")
                } else {
                    fputs("warning: could not parse landmarks JSON, falling back to Vision\n", stderr)
                }
            } else {
                fputs("warning: landmarks file not found: \(lmPath), falling back to Vision\n", stderr)
            }
        }

        if landmarksData != nil {
            print("  Using MediaPipe 478-pt landmarks + homography eye patch (exact training format)")
        } else {
            print("  Using Apple Vision landmarks (approximate)")
        }
        print("")

        // Load pipeline components.
        let detector = FaceLandmarkDetector()
        let eyePatchExtractor = EyePatchExtractor()
        let headPose = HeadPoseEstimator()

        guard let blazeGaze = try? BlazeGazeRunner() else {
            fputs("error: BlazeGaze model not found. Run Tools/Conversion/convert_blazegaze.py first.\n", stderr)
            exit(EXIT_FAILURE)
        }

        print("MacGaze Replay — processing \(url.lastPathComponent)")
        print("  BlazeGaze model: loaded")
        print("")

        // Parse --native flag (use native MediaPipe dylib instead of JSON)
        let useNative = args.contains("--native")
        let nativeModelPath = "Frameworks/face_landmarker_v2_with_blendshapes.task"

        if useNative {
            print("  Using NATIVE MediaPipe (libmediapipe.dylib) — real-time path")
        }

        if isImage {
            await processImage(url: url, detector: detector, extractor: eyePatchExtractor,
                               headPose: headPose, blazeGaze: blazeGaze, verbose: verbose)
        } else if useNative {
            await processVideoNativeMediaPipe(
                url: url, blazeGaze: blazeGaze, verbose: verbose,
                modelPath: nativeModelPath, calibWindows: calibWindows
            )
        } else if let lmData = landmarksData {
            await processVideoMediaPipe(url: url, blazeGaze: blazeGaze,
                                       verbose: verbose, landmarksData: lmData,
                                       calibWindows: calibWindows)
        } else {
            await processVideo(url: url, detector: detector, extractor: eyePatchExtractor,
                               headPose: headPose, blazeGaze: blazeGaze, verbose: verbose,
                               dumpPatches: dumpPatches, dumpDir: dumpDir,
                               calibWindows: calibWindows)
        }
    }

    static let usage = """
    Usage: macgaze-replay <video-or-image-file> [options]

    Options:
      --verbose        Print per-frame gaze + latency.
      --dump-patches   Save first 10 eye patches as PNGs to ./patch-dump/.
      -h, --help       Show this help.

    Record a test video with QuickTime looking at 5 screen positions
    (center, left, right, up, down) for ~2s each. Save as .mov or .mp4.
    """

    // MARK: Video processing

    static func processVideo(
        url: URL,
        detector: FaceLandmarkDetector,
        extractor: EyePatchExtractor,
        headPose: HeadPoseEstimator,
        blazeGaze: BlazeGazeRunner,
        verbose: Bool,
        dumpPatches: Bool = false,
        dumpDir: URL = URL(fileURLWithPath: "."),
        calibWindows: [(start: Double, end: Double, targetX: Double, targetY: Double)] = [],
        landmarksData: [LandmarkFrame]? = nil
    ) async {
        let useMediaPipe = landmarksData != nil
        var landmarkByFrame: [Int: LandmarkFrame] = [:]
        if let landmarksData {
            for lm in landmarksData { landmarkByFrame[lm.frame] = lm }
        }
        let asset = AVURLAsset(url: url)
        guard let track = asset.tracks(withMediaType: .video).first else {
            fputs("error: no video track found\n", stderr)
            exit(EXIT_FAILURE)
        }

        let reader: AVAssetReader
        do {
            reader = try AVAssetReader(asset: asset)
        } catch {
            fputs("error: can't open video: \(error.localizedDescription)\n", stderr)
            exit(EXIT_FAILURE)
        }

        let output = AVAssetReaderTrackOutput(
            track: track,
            outputSettings: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
            ]
        )
        output.alwaysCopiesSampleData = false
        reader.add(output)
        reader.startReading()

        var frameIndex = 0
        var faceFrames = 0
        var gazeFrames = 0
        var latencies: [Double] = []
        // Per-second gaze accumulation for calibration timing.
        var perSecondGaze: [Int: (sumX: Double, sumY: Double, count: Int)] = [:]
        // Store all gaze results for RBF post-processing.
        var allGaze: [(t: Double, x: Double, y: Double)] = []

        while reader.status == .reading {
            guard let sampleBuffer = output.copyNextSampleBuffer() else { break }
            guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { continue }

            let width = CVPixelBufferGetWidth(pixelBuffer)
            let height = CVPixelBufferGetHeight(pixelBuffer)
            let pts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
            let timestamp = Double(pts.value) / Double(pts.timescale)

            frameIndex += 1
            let frame = CameraFrame(pixelBuffer: pixelBuffer, timestampSeconds: timestamp)

            // Run pipeline.
            let t0 = Date()
            let detection = detector.detect(frame)

            if detection.stats.faceFound {
                faceFrames += 1

                if let vnFace = detector.lastRawObservation {
                    let frameW = CVPixelBufferGetWidth(pixelBuffer)
                    let frameH = CVPixelBufferGetHeight(pixelBuffer)

                    // Debug: frame dimensions + bounding box.
                    if verbose && frameIndex <= 3 {
                        let bbox = vnFace.boundingBox
                        print("  [debug] frame \(frameIndex): \(frameW)×\(frameH)  bbox=({\(String(format: "%.2f", bbox.origin.x)),\(String(format: "%.2f", bbox.origin.y))} \(String(format: "%.2f", bbox.width))×\(String(format: "%.2f", bbox.height)))")
                        print("         landmarks: leftEye=\(vnFace.landmarks?.leftEye?.pointCount ?? 0)pts rightEye=\(vnFace.landmarks?.rightEye?.pointCount ?? 0)pts")
                    }

                    // Eye patch.
                    guard let eyePatch = extractor.extract(
                        frame: frame.pixelBuffer,
                        faceObservation: vnFace
                    ) else {
                        if verbose && frameIndex <= 5 {
                            print("  [debug] frame \(frameIndex): eye patch extraction FAILED")
                        }
                        continue
                    }

                    // Dump eye patch as PNG for visual inspection.
                    if dumpPatches && frameIndex <= 10 {
                        if !FileManager.default.fileExists(atPath: dumpDir.path) {
                            try? FileManager.default.createDirectory(at: dumpDir, withIntermediateDirectories: true)
                        }
                        let pngURL = dumpDir.appendingPathComponent("patch_\(String(format: "%04d", frameIndex)).png")
                        Self.savePixelBufferAsPNG(eyePatch, to: pngURL)
                        print("  [dump] saved \(pngURL.lastPathComponent)")
                    }

                    if verbose && frameIndex <= 3 {
                        let ew = CVPixelBufferGetWidth(eyePatch)
                        let eh = CVPixelBufferGetHeight(eyePatch)
                        print("  [debug] frame \(frameIndex): eye patch \(ew)×\(eh)")
                    }

                    // Head pose.
                    var hv: MLMultiArray? = nil
                    var fo: MLMultiArray? = nil
                    if let pose = headPose.estimate(face: vnFace, frameWidth: frameW, frameHeight: frameH) {
                        hv = try? MLMultiArray(shape: [1, 3], dataType: .float32)
                        fo = try? MLMultiArray(shape: [1, 3], dataType: .float32)
                        if let hv { hv[0] = pose.headVector[0] as NSNumber; hv[1] = pose.headVector[1] as NSNumber; hv[2] = pose.headVector[2] as NSNumber }
                        if let fo { fo[0] = pose.faceOrigin3D[0] as NSNumber; fo[1] = pose.faceOrigin3D[1] as NSNumber; fo[2] = pose.faceOrigin3D[2] as NSNumber }
                        if verbose && frameIndex <= 3 {
                            print("  [debug] frame \(frameIndex): head_vec=(\(String(format: "%.3f", pose.headVector[0])), \(String(format: "%.3f", pose.headVector[1])), \(String(format: "%.3f", pose.headVector[2])))  origin=(\(String(format: "%.1f", pose.faceOrigin3D[0])), \(String(format: "%.1f", pose.faceOrigin3D[1])), \(String(format: "%.1f", pose.faceOrigin3D[2])))")
                        }
                    }

                    // BlazeGaze inference.
                    guard let gaze = blazeGaze.predict(eyePatch: eyePatch, headVector: hv, faceOrigin3D: fo) else {
                        if verbose && frameIndex <= 5 {
                            print("  [debug] frame \(frameIndex): BlazeGaze prediction FAILED")
                        }
                        continue
                    }
                    gazeFrames += 1
                    let latency = Date().timeIntervalSince(t0) * 1000
                    latencies.append(latency)
                    allGaze.append((timestamp, Double(gaze.x), Double(gaze.y)))

                    // Accumulate per-second averages.
                    let sec = Int(timestamp)
                    var entry = perSecondGaze[sec] ?? (0, 0, 0)
                    entry = (entry.sumX + Double(gaze.x),
                             entry.sumY + Double(gaze.y),
                             entry.count + 1)
                    perSecondGaze[sec] = entry

                    if verbose || frameIndex % 30 == 0 {
                        print(String(format: "  frame %4d  t=%5.1fs  gaze=(%.3f, %.3f)  latency=%.1fms",
                                     frameIndex, timestamp, gaze.x, gaze.y, latency))
                    }
                }
            }
        }

        // Summary.
        print("")
        print("=== Replay summary ===")
        print("  Total frames       : \(frameIndex)")
        print("  Face detected      : \(faceFrames) (\(faceFrames > 0 ? String(format: "%.0f%%", Double(faceFrames) / Double(frameIndex) * 100) : "0%"))")
        print("  Gaze predicted     : \(gazeFrames)")
        if !latencies.isEmpty {
            let sorted = latencies.sorted()
            let mean = latencies.reduce(0, +) / Double(latencies.count)
            print(String(format: "  Pipeline latency    : mean %.1fms  p50 %.1fms  p95 %.1fms",
                         mean, sorted[sorted.count / 2], sorted[Int(Double(sorted.count) * 0.95)]))
        }

        // Per-second gaze averages for calibration timing.
        if !perSecondGaze.isEmpty {
            print("")
            print("=== Per-second gaze (for calibration) ===")
            print("  second |  avg X   |  avg Y   | samples")
            for sec in perSecondGaze.keys.sorted() {
                let g = perSecondGaze[sec]!
                let avgX = g.sumX / Double(g.count)
                let avgY = g.sumY / Double(g.count)
                print(String(format: "  %5ds  |  %.4f  |  %.4f  |  %d", sec, avgX, avgY, g.count))
            }
            print("")
            print("  Tell me which seconds = which direction you were looking,")
            print("  e.g.: '0-2=center 2-4=left 4-6=right 6-8=up 8-10=down'")
        }

        // RBF calibration post-processing.
        if !calibWindows.isEmpty && !allGaze.isEmpty {
            print("")
            print("=== RBF calibration ===")

            // Collect calibration samples from the windows.
            var calibSamples: [MacGaze.RBFGazeCorrector.CalibrationSample] = []
            for window in calibWindows {
                let windowGaze = allGaze.filter { $0.t >= window.start && $0.t < window.end }
                guard !windowGaze.isEmpty else { continue }
                let avgX = windowGaze.reduce(0.0) { $0 + $1.x } / Double(windowGaze.count)
                let avgY = windowGaze.reduce(0.0) { $0 + $1.y } / Double(windowGaze.count)
                calibSamples.append(MacGaze.RBFGazeCorrector.CalibrationSample(
                    observedX: avgX, observedY: avgY,
                    targetX: window.targetX, targetY: window.targetY
                ))
                print(String(format: "  window %.1f-%.1fs: observed (%.4f, %.4f) → target (%.1f, %.1f)  [%d samples]",
                             window.start, window.end, avgX, avgY,
                             window.targetX, window.targetY, windowGaze.count))
            }

            if calibSamples.count >= 3 {
                let rbf = MacGaze.RBFGazeCorrector()
                let success = rbf.calibrate(calibSamples)
                print("")
                print("  RBF solve: \(success ? "✓ success" : "✗ failed"), σ=\(String(format: "%.4f", rbf.solvedSigma)), \(calibSamples.count) points")

                if success {
                    print("")
                    print("=== Corrected gaze (RBF applied) ===")
                    print("  second |  raw X   |  raw Y   |  corr X  |  corr Y  |  direction")
                    print("  -------+----------+----------+----------+----------+----------")

                    // Group corrected results by second.
                    var perSecondCorrected: [Int: (sumX: Double, sumY: Double, count: Int)] = [:]
                    for g in allGaze {
                        let corrected = rbf.correct(x: g.x, y: g.y)
                        let sec = Int(g.t)
                        var entry = perSecondCorrected[sec] ?? (0, 0, 0)
                        entry = (entry.sumX + corrected.x, entry.sumY + corrected.y, entry.count + 1)
                        perSecondCorrected[sec] = entry
                    }

                    for sec in perSecondCorrected.keys.sorted() {
                        let raw = perSecondGaze[sec]!
                        let corr = perSecondCorrected[sec]!
                        let rawAvgX = raw.sumX / Double(raw.count)
                        let rawAvgY = raw.sumY / Double(raw.count)
                        let corrAvgX = corr.sumX / Double(corr.count)
                        let corrAvgY = corr.sumY / Double(corr.count)

                        // Determine which direction this second falls in.
                        var direction = ""
                        for w in calibWindows {
                            if Double(sec) >= w.start && Double(sec) < w.end {
                                let dx = abs(corrAvgX - w.targetX)
                                let dy = abs(corrAvgY - w.targetY)
                                direction = String(format: "target (%.1f,%.1f) err=%.3f", w.targetX, w.targetY, max(dx, dy))
                                break
                            }
                        }

                        print(String(format: "  %5ds  |  %.4f  |  %.4f  |  %.4f  |  %.4f  |  %@",
                                     sec, rawAvgX, rawAvgY, corrAvgX, corrAvgY, direction.isEmpty ? "" : direction))
                    }

                    // Verdict.
                    print("")
                    var totalError: Double = 0
                    var errorCount = 0
                    for w in calibWindows {
                        let windowGaze = allGaze.filter { $0.t >= w.start && $0.t < w.end }
                        let corrected = windowGaze.compactMap { rbf.correct(x: $0.x, y: $0.y) }
                        if !corrected.isEmpty {
                            let avgX = corrected.reduce(0.0) { $0 + $1.x } / Double(corrected.count)
                            let avgY = corrected.reduce(0.0) { $0 + $1.y } / Double(corrected.count)
                            let dx = avgX - w.targetX
                            let dy = avgY - w.targetY
                            let err = (dx * dx + dy * dy).squareRoot()
                            totalError += err
                            errorCount += 1
                        }
                    }
                    if errorCount > 0 {
                        let meanErr = totalError / Double(errorCount)
                        print(String(format: "  Mean calibration error: %.4f (normalised distance)", meanErr))
                        if meanErr < 0.1 {
                            print("  ✓ GOOD — corrected gaze lands within 10%% of targets")
                        } else if meanErr < 0.2 {
                            print("  ~ OKAY — corrected gaze is in the right ballpark")
                        } else {
                            print("  ⚠ POOR — needs more calibration points or better eye patch")
                        }
                    }
                }
            } else {
                print("  Not enough calibration windows with data (need ≥3)")
            }
        }

        if gazeFrames > 0 {
            print("")
            print("  ✓ Pipeline produced gaze predictions from recorded video.")
            print("    If the (x,y) values tracked your eye movements, BlazeGaze works.")
        } else {
            print("")
            print("  ⚠ No gaze predictions — check that faces are visible in the video.")
        }
    }

    // MARK: Image processing

    static func processImage(
        url: URL,
        detector: FaceLandmarkDetector,
        extractor: EyePatchExtractor,
        headPose: HeadPoseEstimator,
        blazeGaze: BlazeGazeRunner,
        verbose: Bool
    ) async {
        guard let cgImage = loadImage(url: url) else {
            fputs("error: can't load image: \(url.path)\n", stderr)
            exit(EXIT_FAILURE)
        }

        // Convert CGImage → CVPixelBuffer.
        let width = cgImage.width
        let height = cgImage.height
        var pixelBuffer: CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault, width, height,
                           kCVPixelFormatType_32BGRA, nil, &pixelBuffer)
        guard let pb = pixelBuffer else {
            fputs("error: can't create pixel buffer\n", stderr)
            exit(EXIT_FAILURE)
        }
        CVPixelBufferLockBaseAddress(pb, [])
        let ctx = CGContext(
            data: CVPixelBufferGetBaseAddress(pb),
            width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(pb),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        )
        ctx?.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        CVPixelBufferUnlockBaseAddress(pb, [])

        let frame = CameraFrame(pixelBuffer: pb, timestampSeconds: 0)
        let detection = detector.detect(frame)

        print("  Image size  : \(width)×\(height)")
        print("  Face found  : \(detection.stats.faceFound)")
        print("  Latency     : \(String(format: "%.1f", detection.stats.latencyMs)) ms")

        if detection.stats.faceFound, let vnFace = detector.lastRawObservation {
            if let eyePatch = extractor.extract(frame: pb, faceObservation: vnFace) {
                var hv: MLMultiArray? = nil
                var fo: MLMultiArray? = nil
                if let pose = headPose.estimate(face: vnFace, frameWidth: width, frameHeight: height) {
                    hv = try? MLMultiArray(shape: [1, 3], dataType: .float32)
                    fo = try? MLMultiArray(shape: [1, 3], dataType: .float32)
                    if let hv { hv[0] = pose.headVector[0] as NSNumber; hv[1] = pose.headVector[1] as NSNumber; hv[2] = pose.headVector[2] as NSNumber }
                    if let fo { fo[0] = pose.faceOrigin3D[0] as NSNumber; fo[1] = pose.faceOrigin3D[1] as NSNumber; fo[2] = pose.faceOrigin3D[2] as NSNumber }
                    print("  Head vector : (\(String(format: "%.3f", pose.headVector[0])), \(String(format: "%.3f", pose.headVector[1])), \(String(format: "%.3f", pose.headVector[2])))")
                    print("  Face origin : (\(String(format: "%.1f", pose.faceOrigin3D[0])), \(String(format: "%.1f", pose.faceOrigin3D[1])), \(String(format: "%.1f", pose.faceOrigin3D[2])))")
                }
                if let gaze = blazeGaze.predict(eyePatch: eyePatch, headVector: hv, faceOrigin3D: fo) {
                    print("")
                    print("  ✓ BlazeGaze gaze: (\(String(format: "%.3f", gaze.x)), \(String(format: "%.3f", gaze.y)))")
                    print("    (0.5, 0.5) = screen center. Values should be in [0, 1].")
                } else {
                    print("  ⚠ BlazeGaze inference failed.")
                }
            } else {
                print("  ⚠ Eye patch extraction failed.")
            }
        } else {
            print("  ⚠ No face detected in image.")
        }
    }

    /// Load image via NSImage (fallback if CGImage.create fails).
    private static func loadImage(url: URL) -> CGImage? {
        // Use CoreGraphics directly to avoid AppKit/Metal.
        if let source = CGImageSourceCreateWithURL(url as CFURL, nil),
           let cg = CGImageSourceCreateImageAtIndex(source, 0, nil) {
            return cg
        }
        return nil
    }

    /// Save a CVPixelBuffer as a PNG file (pure CoreGraphics, no Metal).
    private static func savePixelBufferAsPNG(_ buffer: CVPixelBuffer, to url: URL) {
        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)

        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }

        guard let baseAddress = CVPixelBufferGetBaseAddress(buffer) else { return }
        let colorSpace = CGColorSpaceCreateDeviceRGB()

        guard let provider = CGDataProvider(
            dataInfo: nil,
            data: baseAddress,
            size: height * bytesPerRow,
            releaseData: { _, _, _ in }
        ) else { return }

        guard let cgImage = CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGBitmapInfo(
                rawValue: CGImageAlphaInfo.noneSkipFirst.rawValue
                  | CGBitmapInfo.byteOrder32Little.rawValue
            ),
            provider: provider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
        ) else { return }

        guard let dest = CGImageDestinationCreateWithURL(
            url as CFURL, "public.png" as CFString, 1, nil
        ) else { return }
        CGImageDestinationAddImage(dest, cgImage, nil)
        CGImageDestinationFinalize(dest)
    }
}
