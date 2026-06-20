import Foundation
import AVFoundation
import AppKit
import QuartzCore
import Combine
import MacGaze

/// Drives the CameraCapture + FaceLandmarkDetector pipeline for the
/// debug window.  All Vision work happens off the main actor; only the
/// @Published final results land on it.
@MainActor
final class DebugPipeline: ObservableObject {

    @Published private(set) var latestImage: NSImage?
    @Published private(set) var latestDetection: DetectionResult?
    @Published private(set) var sessionState: SessionState = .idle
    @Published private(set) var framesPerSecond: Double = 0
    @Published private(set) var medianLatencyMs: Double = 0
    @Published private(set) var errorMessage: String?

    enum SessionState: Equatable {
        case idle, starting, running, stopped, failed
    }

    let camera = CameraCapture()
    private let detector = FaceLandmarkDetector()

    private var captureTask: Task<Void, Never>?
    private var frameCount: Int = 0
    private var lastFpsTick = Date()
    private var recentLatencies: [Double] = []  // ring buffer, last 30 samples
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

        // Frame + detection happen on the AVCapture video queue.  We
        // marshal results back to MainActor for @Published state.
        let stream = camera.frames
        for await frame in stream {
            // Run Vision synchronously — it locks the pixelBuffer for
            // the duration, so we must finish before awaiting the next
            // frame.  Off-main because we're on the AVCapture video queue.
            let detection = detector.detect(frame)
            let image = Self.renderToNSImage(frame: frame)

            await MainActor.run {
                self.latestImage = image
                self.latestDetection = detection
                self.recordLatency(detection.stats.latencyMs)
                self.tickFps()
            }
        }
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
    /// Mirrored horizontally to match the user's expectation (like
    /// Photo Booth) — the raw AVCapture front-camera feed is *not*
    /// automatically mirrored at the buffer level.
    private static func renderToNSImage(frame: CameraFrame) -> NSImage? {
        let buffer = frame.pixelBuffer
        let ciImage = CIImage(cvPixelBuffer: buffer)
            .transformed(by: CGAffineTransform(scaleX: 1, y: -1))   // Vision uses bottom-left origin; flip for screen
            .transformed(by: CGAffineTransform(translationX: 0, y: CGFloat(frame.height)))
            .transformed(by: CGAffineTransform(scaleX: -1, y: 1))   // mirror like a selfie
            .transformed(by: CGAffineTransform(translationX: CGFloat(frame.width), y: 0))
        let context = CIContext()
        guard let cg = context.createCGImage(ciImage, from: CGRect(x: 0, y: 0, width: frame.width, height: frame.height)) else {
            return nil
        }
        return NSImage(cgImage: cg, size: NSSize(width: frame.width, height: frame.height))
    }
}
