import Foundation
import AVFoundation
import AppKit
import QuartzCore
import Combine
import Vision
import MacGaze

/// Drives the CameraCapture + FaceLandmarkDetector + BlazeGaze pipeline
/// for the debug window.  All Vision + CoreML work happens off the main
/// actor; only the @Published final results land on it.
@MainActor
final class DebugPipeline: ObservableObject {

    @Published private(set) var latestImage: NSImage?
    @Published private(set) var latestDetection: DetectionResult?
    @Published private(set) var gazePrediction: CGPoint?      // BlazeGaze output
    @Published private(set) var gazeLatencyMs: Double = 0     // BlazeGaze inference time
    @Published private(set) var sessionState: SessionState = .idle
    @Published private(set) var framesPerSecond: Double = 0
    @Published private(set) var medianLatencyMs: Double = 0
    @Published private(set) var errorMessage: String?

    enum SessionState: Equatable {
        case idle, starting, running, stopped, failed
    }

    let camera = CameraCapture()
    private let detector = FaceLandmarkDetector()
    private let eyePatchExtractor = EyePatchExtractor()
    private let headPose = HeadPoseEstimator()
    private var blazeGaze: BlazeGazeRunner?

    private var captureTask: Task<Void, Never>?
    private var frameCount: Int = 0
    private var displayFrameCount: Int = 0
    private var lastFpsTick = Date()
    private var lastDisplayTime: Date = .distantPast
    private var recentLatencies: [Double] = []
    private let maxLatencySamples = 30

    func start() {
        guard captureTask == nil else { return }
        sessionState = .starting
        errorMessage = nil
        captureTask = Task { await runLoop() }
    }

    func stop() {
        captureTask?.cancel()
        captureTask = nil
        camera.stop()
        sessionState = .stopped
    }

    private func runLoop() async {
        do {
            try await camera.start()
            await MainActor.run { self.sessionState = .running }
        } catch {
            await MainActor.run {
                self.sessionState = .failed
                self.errorMessage = error.localizedDescription
            }
            return
        }

        // Frame + detection + BlazeGaze happen on the AVCapture video queue.
        // We marshal results back to MainActor for @Published state.
        let stream = camera.frames
        for await frame in stream {
            // 1. Vision landmark detection.
            let detection = detector.detect(frame)

        // 2. BlazeGaze gaze prediction (if face found + model loaded).
        var gaze: CGPoint? = nil
        var gazeMs: Double = 0
        if detection.stats.faceFound,
           let vnFace = detector.lastRawObservation {
            let t0 = Date()
            gaze = runBlazeGaze(frame: frame, vnFace: vnFace)
            gazeMs = Date().timeIntervalSince(t0) * 1000
        }

            // Render display image at most 10 fps to reduce Metal load
            // on Core Animation's display cycle.  The camera stream +
            // BlazeGaze still runs at full 30 fps; only the NSImage
            // display is throttled.
            let now = Date()
            let shouldDisplay = now.timeIntervalSince(lastDisplayTime) >= 0.1  // 10 fps
            let image: NSImage? = shouldDisplay ? Self.renderToNSImage(frame: frame) : nil
            if shouldDisplay { lastDisplayTime = now }

            await MainActor.run {
                self.latestImage = image
                self.latestDetection = detection
                self.gazePrediction = gaze
                self.gazeLatencyMs = gazeMs
                self.recordLatency(detection.stats.latencyMs)
                self.tickFps()
            }
        }
    }

    /// Run BlazeGaze inference on the current frame.  Returns nil if the
    /// model isn't loaded or prediction fails.
    private func runBlazeGaze(frame: CameraFrame, vnFace: VNFaceObservation) -> CGPoint? {
        // Ensure the model is loaded (lazy init on first call).
        if blazeGaze == nil {
            do {
                blazeGaze = try BlazeGazeRunner()
            } catch {
                // Model not found — BlazeGaze stays nil; pipeline falls back
                // to landmark-only display.  Silently fail rather than spam
                // the user every frame.
                return nil
            }
        }

        // Extract the eye patch from the camera frame.
        guard let eyePatch = eyePatchExtractor.extract(
            frame: frame.pixelBuffer,
            faceObservation: vnFace
        ) else { return nil }

        // Solve head pose from Vision landmarks for BlazeGaze's
        // head_vector + face_origin_3d inputs.
        var headVector: MLMultiArray? = nil
        var faceOrigin: MLMultiArray? = nil
        if let pose = headPose.estimate(
            face: vnFace,
            frameWidth: frame.width,
            frameHeight: frame.height
        ) {
            headVector = try? MLMultiArray(
                shape: [1, 3],
                dataType: .float32
            )
            faceOrigin = try? MLMultiArray(
                shape: [1, 3],
                dataType: .float32
            )
            if let hv = headVector {
                hv[0] = NSNumber(value: pose.headVector[0])
                hv[1] = NSNumber(value: pose.headVector[1])
                hv[2] = NSNumber(value: pose.headVector[2])
            }
            if let fo = faceOrigin {
                fo[0] = NSNumber(value: pose.faceOrigin3D[0])
                fo[1] = NSNumber(value: pose.faceOrigin3D[1])
                fo[2] = NSNumber(value: pose.faceOrigin3D[2])
            }
        }

        // Run inference with solved (or neutral) head pose.
        return blazeGaze?.predict(
            eyePatch: eyePatch,
            headVector: headVector,
            faceOrigin3D: faceOrigin
        )
    }

    // MARK: Stats

    private func recordLatency(_ ms: Double) {
        recentLatencies.append(ms)
        if recentLatencies.count > maxLatencySamples {
            recentLatencies.removeFirst()
        }
        let sorted = recentLatencies.sorted()
        medianLatencyMs = sorted[sorted.count / 2]
    }

    private func tickFps() {
        frameCount += 1
        let now = Date()
        let elapsed = now.timeIntervalSince(lastFpsTick)
        if elapsed >= 1.0 {
            framesPerSecond = Double(frameCount) / elapsed
            frameCount = 0
            lastFpsTick = now
        }
    }

    // MARK: Rendering

    /// Convert a 32BGRA CVPixelBuffer to an NSImage for display.
    /// Uses pure CoreGraphics with a **pixel data copy** to prevent a
    /// use-after-free: the camera recycles CVPixelBuffers for the next
    /// frame, so we must copy before handing the pointer to CGDataProvider.
    /// Mirrored horizontally for selfie view.
    private static func renderToNSImage(frame: CameraFrame) -> NSImage? {
        let buffer = frame.pixelBuffer
        let width = frame.width
        let height = frame.height

        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }

        guard let baseAddress = CVPixelBufferGetBaseAddress(buffer) else { return nil }
        let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
        let dataSize = height * bytesPerRow
        let colorSpace = CGColorSpaceCreateDeviceRGB()

        // CRITICAL: copy the pixel data into our own buffer.  The camera's
        // CVPixelBuffer will be recycled for the next frame, and Core
        // Animation may draw the CGImage asynchronously on a later display
        // cycle.  Without a copy, we get a use-after-free that corrupts
        // Metal reads → "-[__NSCFNumber length]" crash.
        let dataCopy = UnsafeMutableRawPointer.allocate(byteCount: dataSize, alignment: 16)
        memcpy(dataCopy, baseAddress, dataSize)

        // Provider owns dataCopy; releases it when the CGImage is freed.
        // We pass dataCopy as both dataInfo and data so the release callback
        // can deallocate it.
        let provider = CGDataProvider(
            dataInfo: dataCopy,
            data: dataCopy,
            size: dataSize
        ) { info, _, _ in
            info?.deallocate()
        }

        guard let provider else {
            dataCopy.deallocate()
            return nil
        }

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
        ) else { return nil }

        // Mirror X (selfie view) via CGContext.
        guard let ctx = CGContext(
            data: nil, width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: 0,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
              | CGBitmapInfo.byteOrder32Little.rawValue
        ) else { return nil }

        ctx.translateBy(x: CGFloat(width), y: 0)
        ctx.scaleBy(x: -1, y: 1)
        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

        guard let mirrored = ctx.makeImage() else { return nil }
        return NSImage(cgImage: mirrored, size: NSSize(width: width, height: height))
    }
}
