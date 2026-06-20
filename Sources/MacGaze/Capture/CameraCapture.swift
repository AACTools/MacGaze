import Foundation
import AVFoundation
import CoreVideo

/// Errors surfaced by `CameraCapture`.
public enum CameraCaptureError: Error, LocalizedError {
    case noCameraAvailable
    case authorizationDenied
    case authorizationRestricted
    case sessionConfigurationFailed(String)
    case notRunning

    public var errorDescription: String? {
        switch self {
        case .noCameraAvailable:           return "No camera device is available."
        case .authorizationDenied:         return "Camera access was denied. Grant permission in System Settings → Privacy & Security → Camera."
        case .authorizationRestricted:     return "Camera access is restricted by device policy."
        case .sessionConfigurationFailed(let msg): return "AVCapture session configuration failed: \(msg)"
        case .notRunning:                  return "Capture session isn't running; call start() first."
        }
    }
}

/// Wraps an `AVCaptureSession` streaming 1280×720 32BGRA frames from the
/// FaceTime HD camera.
///
/// Frames are delivered on a dedicated serial dispatch queue (never the
/// main thread) via an `AsyncStream`. The session itself runs on its own
/// queue too — all `AVCaptureSession` mutations are dispatched there to
/// avoid the well-known main-thread-deadlock pitfall.
///
/// Usage:
/// ```swift
/// let camera = CameraCapture()
/// try await camera.start()
/// for await frame in camera.frames {
///     // consume frame.pixelBuffer synchronously before next iteration
/// }
/// ```
public final class CameraCapture: @unchecked Sendable {

    /// Configuration knobs exposed for testing + later tuning.
    public struct Configuration: Sendable {
        public var preferredWidth: Int = 1280
        public var preferredHeight: Int = 720
        public var preferredFrameRate: Double = 30
        public var pixelFormat: OSType = kCVPixelFormatType_32BGRA

        public init() {}

        /// The dedicated session queue label (visible in CrashLogs /
        /// Instruments; keep it stable).
        public static let sessionQueueLabel = "macgaze.capture.session"
        /// The dedicated video-output callback queue label.
        public static let videoQueueLabel = "macgaze.capture.video"
    }

    public let configuration: Configuration

    private let session = AVCaptureSession()
    private let sessionQueue = DispatchQueue(label: Configuration.sessionQueueLabel)
    private let videoQueue = DispatchQueue(label: Configuration.videoQueueLabel)
    private let videoOutput = AVCaptureVideoDataOutput()

    /// Last error reported synchronously from configuration. Read on the
    /// main actor; written under `sessionQueue`.
    @MainActor private static var lastConfigurationError: String?
    private let errorBox = ErrorBox()

    private final class ErrorBox: @unchecked Sendable {
        private let lock = NSLock()
        private var value: String?
        func set(_ s: String?) { lock.lock(); defer { lock.unlock() }; value = s }
        func get() -> String? { lock.lock(); defer { lock.unlock() }; return value }
    }

    /// Continuation backing `frames`. Held in a box so the delegate
    /// (which is a separate NSObject) can yield to it.
    private let continuationBox = ContinuationBox()

    private final class ContinuationBox: @unchecked Sendable {
        private let lock = NSLock()
        private var continuation: AsyncStream<CameraFrame>.Continuation?
        func set(_ c: AsyncStream<CameraFrame>.Continuation) { lock.lock(); defer { lock.unlock() }; continuation = c }
        func clear() { lock.lock(); defer { lock.unlock() }; continuation = nil }
        func yield(_ f: CameraFrame) { lock.lock(); let c = continuation; lock.unlock(); c?.yield(f) }
    }

    /// Delegate object that owns the sample-buffer callback. Lives for the
    /// lifetime of the session; never accessed cross-actor outside the
    /// video queue.
    private lazy var sampleBufferDelegate = SampleBufferDelegate(
        onSample: { [weak continuationBox] frame in
            continuationBox?.yield(frame)
        }
    )

    private final class SampleBufferDelegate: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
        let onSample: (CameraFrame) -> Void
        private var firstSampleTime: TimeInterval?

        init(onSample: @escaping (CameraFrame) -> Void) {
            self.onSample = onSample
        }

        func captureOutput(
            _ output: AVCaptureOutput,
            didOutput sampleBuffer: CMSampleBuffer,
            from connection: AVCaptureConnection
        ) {
            guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
            let pts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
            // Convert to seconds; treat first sample as t=0 for stability.
            let raw = Double(pts.value) / Double(pts.timescale)
            if firstSampleTime == nil { firstSampleTime = raw }
            let secs = raw - (firstSampleTime ?? raw)
            onSample(CameraFrame(pixelBuffer: pixelBuffer, timestampSeconds: max(0, secs)))
        }
    }

    public init(configuration: Configuration = Configuration()) {
        self.configuration = configuration
    }

    deinit {
        // Synchronous teardown — safe because sessionQueue is exclusive.
        sessionQueue.sync {
            if session.isRunning { session.stopRunning() }
        }
        continuationBox.clear()
    }

    // MARK: Lifecycle

    /// Request camera permission if not yet determined. Throws on denial.
    public static func ensureAuthorized() async throws {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            return
        case .notDetermined:
            let granted = await AVCaptureDevice.requestAccess(for: .video)
            if !granted { throw CameraCaptureError.authorizationDenied }
        case .denied:
            throw CameraCaptureError.authorizationDenied
        case .restricted:
            throw CameraCaptureError.authorizationRestricted
        @unknown default:
            throw CameraCaptureError.authorizationRestricted
        }
    }

    /// Configure the session and start streaming. Safe to call once.
    public func start() async throws {
        try await Self.ensureAuthorized()

        // Set up the session on its dedicated queue. Capture any thrown
        // error into `errorBox` so we can rethrow from this async API.
        errorBox.set(nil)
        sessionQueue.sync {
            do {
                try configureSessionLocked()
                if !session.isRunning { session.startRunning() }
            } catch {
                errorBox.set(error.localizedDescription)
            }
        }
        if let msg = errorBox.get() {
            throw CameraCaptureError.sessionConfigurationFailed(msg)
        }
    }

    public func stop() {
        sessionQueue.sync {
            if session.isRunning { session.stopRunning() }
        }
        continuationBox.clear()
    }

    // MARK: Frame stream

    /// Async stream of camera frames. Single-consumer for Phase 0; if you
    /// need fan-out, wrap with `for await ... in stream.multicast(...)`.
    public var frames: AsyncStream<CameraFrame> {
        AsyncStream { continuation in
            continuationBox.set(continuation)
            continuation.onTermination = { [weak self] _ in
                self?.continuationBox.clear()
            }
        }
    }

    // MARK: Session configuration (sessionQueue only)

    private func configureSessionLocked() throws {
        session.beginConfiguration()
        defer { session.commitConfiguration() }

        session.sessionPreset = .hd1280x720

        guard let device = AVCaptureDevice.default(
            .builtInWideAngleCamera,
            for: .video,
            position: .front
        ) else {
            throw CameraCaptureError.noCameraAvailable
        }

        let input = try AVCaptureDeviceInput(device: device)
        guard session.canAddInput(input) else {
            throw CameraCaptureError.sessionConfigurationFailed("Cannot add device input")
        }
        session.addInput(input)

        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: Int(configuration.pixelFormat)
        ]
        guard session.canAddOutput(videoOutput) else {
            throw CameraCaptureError.sessionConfigurationFailed("Cannot add video output")
        }
        session.addOutput(videoOutput)
        videoOutput.setSampleBufferDelegate(sampleBufferDelegate, queue: videoQueue)

        // Try to lock the device to our preferred frame rate.
        do {
            try device.lockForConfiguration()
            device.activeVideoMinFrameDuration = CMTime(value: 1, timescale: Int32(configuration.preferredFrameRate))
            device.activeVideoMaxFrameDuration = CMTime(value: 1, timescale: Int32(configuration.preferredFrameRate))
            device.unlockForConfiguration()
        } catch {
            // Frame-rate lock is best-effort; carry on at default fps.
        }
    }
}
