import Foundation
import AVFoundation
import CoreVideo
import CoreML
import MacGaze

/// Headless pipeline test — replays a recorded video through the full
/// MacGaze pipeline and prints gaze predictions to stdout.
///
/// No SwiftUI, no Core Animation, no Metal display = no crash on M1.
///
/// Usage:
///   swift run macgaze-replay <video-file>     [--verbose]
///   swift run macgaze-replay <image-file>     [--verbose]
///
/// Record a video with QuickTime (File → New Movie Recording) looking
/// at: center → left → right → up → down → center, ~2s each direction.
/// Save as .mov or .mp4. Then:
///
///   swift run macgaze-replay ~/Desktop/my-gaze-test.mov
///
/// If the printed gaze (x,y) values track your eye movements, the
/// pipeline works correctly.

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

        let url = URL(fileURLWithPath: filePath)
        guard FileManager.default.fileExists(atPath: url.path) else {
            fputs("error: file not found: \(url.path)\n", stderr)
            exit(EXIT_FAILURE)
        }

        // Detect file type.
        let ext = url.pathExtension.lowercased()
        let isImage = ["jpg", "jpeg", "png", "heic"].contains(ext)

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

        if isImage {
            await processImage(url: url, detector: detector, extractor: eyePatchExtractor,
                               headPose: headPose, blazeGaze: blazeGaze, verbose: verbose)
        } else {
            await processVideo(url: url, detector: detector, extractor: eyePatchExtractor,
                               headPose: headPose, blazeGaze: blazeGaze, verbose: verbose)
        }
    }

    static let usage = """
    Usage: macgaze-replay <video-or-image-file> [options]

    Options:
      --verbose    Print per-frame gaze + latency.
      -h, --help   Show this help.

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
        verbose: Bool
    ) async {
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
}
