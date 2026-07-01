import Foundation
import CoreGraphics
import CoreML
import CoreVideo
import Vision
import GazeBridgeCore

/// MacGaze's implementation of GazeBridge's `TrackerDriver` protocol.
///
/// Uses the built-in FaceTime HD camera + BlazeGaze CoreML model +
/// Gaussian RBF personalisation to produce gaze data — no external
/// hardware required.
///
/// Two landmark backends are available:
/// - `.mediaPipe` (default): Google MediaPipe 478-pt landmarks + homography
///   eye patch. Exact training format for BlazeGaze. Requires libmediapipe.dylib.
/// - `.vision`: Apple Vision 76-pt landmarks + approximate eye patch.
///   Works without MediaPipe but BlazeGaze can't distinguish gaze directions well.
public final class MacGazeTracker: TrackerDriver, @unchecked Sendable {

    public let displayName: String
    public let driverIdentifier: String

    /// Which landmark detection backend to use.
    public enum LandmarkBackend: String, CaseIterable, Sendable {
        case mediaPipe = "MediaPipe (478-pt, default)"
        case vision = "Apple Vision (76-pt, fallback)"
    }

    public var backend: LandmarkBackend

    // Pipeline components (shared across backends).
    private let camera: CameraCapture
    private let blazeGazeRunner: BlazeGazeRunner?
    private let rbfCorrector: RBFGazeCorrector
    private let calibrationCollector: CalibrationCollector

    // Vision backend components.
    private let visionDetector = FaceLandmarkDetector()
    private let visionEyePatch = EyePatchExtractor()
    private let visionHeadPose = HeadPoseEstimator()

    // MediaPipe backend components.
    private var mediaPipe: MediaPipeFaceLandmarker?
    private var mediaPipeTimestampMs: Int64 = 0

    // State.
    private let stateLock = NSLock()
    private var _connectionState: TrackerConnectionState = .disconnected
    private var stateContinuations: [UUID: AsyncStream<TrackerConnectionState>.Continuation] = [:]

    // Active streams.
    private var gazeStreamTask: Task<Void, Never>?

    // 1-Euro filter for smoothing.
    private var smootherX = OneEuroFilter()
    private var smootherY = OneEuroFilter()

    // Calibration callback (set externally by the calibration UI).
    private var calibrationObservationHandler: ((Double, Double) -> Void)?

    public init(backend: LandmarkBackend = .mediaPipe) {
        self.displayName = "Built-in Camera (MacGaze)"
        self.driverIdentifier = "macgaze.builtin"
        self.backend = backend
        self.camera = CameraCapture()
        self.blazeGazeRunner = try? BlazeGazeRunner()
        self.rbfCorrector = RBFGazeCorrector()
        self.calibrationCollector = CalibrationCollector()

        // Try to load MediaPipe if requested.
        if backend == .mediaPipe {
            let modelPath = Self.resolveModelPath()
            if let modelPath {
                self.mediaPipe = try? MediaPipeFaceLandmarker(modelPath: modelPath)
                if mediaPipe == nil {
                    // Fallback to Vision if MediaPipe can't load.
                    self.backend = .vision
                }
            } else {
                self.backend = .vision
            }
        }
    }

    // MARK: TrackerDriver — Connection

    public var connectionState: TrackerConnectionState {
        stateLock.lock(); defer { stateLock.unlock() }
        return _connectionState
    }

    public func connectionStateStream() -> AsyncStream<TrackerConnectionState> {
        AsyncStream { continuation in
            let id = UUID()
            stateLock.lock()
            stateContinuations[id] = continuation
            let current = _connectionState
            stateLock.unlock()
            continuation.yield(current)
            continuation.onTermination = { [weak self] _ in
                guard let self else { return }
                self.stateLock.lock()
                self.stateContinuations.removeValue(forKey: id)
                self.stateLock.unlock()
            }
        }
    }

    public func connect() async throws {
        setState(.connecting)
        do {
            try await camera.start()
            setState(.connected)
        } catch {
            setState(.connectionFailed(error.localizedDescription))
            throw error
        }
    }

    public func disconnect() async {
        gazeStreamTask?.cancel()
        gazeStreamTask = nil
        camera.stop()
        setState(.disconnected)
    }

    // MARK: TrackerDriver — Gaze stream

    public func gazeStream(unfiltered: Bool) -> AsyncStream<GazeSample> {
        AsyncStream { continuation in
            gazeStreamTask?.cancel()
            gazeStreamTask = Task { [weak self] in
                guard let self else { return }
                for await frame in self.camera.frames {
                    if Task.isCancelled { break }
                    let sample = self.processFrame(frame)
                    continuation.yield(sample)
                }
                continuation.finish()
            }
            continuation.onTermination = { [weak self] _ in
                self?.gazeStreamTask?.cancel()
            }
        }
    }

    // MARK: TrackerDriver — Positioning

    public func positioningStream() -> AsyncStream<PositioningInfo> {
        AsyncStream { continuation in
            continuation.finish()
        }
    }

    // MARK: TrackerDriver — Device info

    public func deviceInformation() async throws -> TrackerDeviceInformation {
        TrackerDeviceInformation(
            serial: 0,
            firmwareVersion: blazeGazeRunner != nil ? "BlazeGaze 1.0 (\(backend == .mediaPipe ? "MediaPipe" : "Vision"))" : "unloaded",
            hardwareConfig: 0,
            machine: "FaceTime HD",
            cpuTempCelsius: 0
        )
    }

    public func currentUserSettings() async throws -> TrackerUserSettings {
        TrackerUserSettings(smoothing: 5, leftEyeOnly: false, rightEyeOnly: false)
    }

    public func applyUserSettings(_ settings: TrackerUserSettings) async throws {
        let cfg = OneEuroFilter.Configuration.from(smoothingSlider: settings.smoothing)
        smootherX = OneEuroFilter(cfg)
        smootherY = OneEuroFilter(cfg)
    }

    // MARK: TrackerDriver — Calibration

    public func calibrate(_ options: CalibrationOptions) -> AsyncStream<CalibrationEvent> {
        AsyncStream { continuation in
            continuation.yield(.started)
        }
    }

    public func lastCalibrationResult() async throws -> CalibrationResultSummary? {
        guard rbfCorrector.isCalibrated else { return nil }
        return CalibrationResultSummary(
            overallRating: 0, points: [], canImprove: false,
            timestampMs: Int64(Date().timeIntervalSince1970 * 1000)
        )
    }

    public func confirmCurrentCalibrationPoint() async throws {
        _ = calibrationCollector.finishTarget()
    }

    public func improveCalibrationPoints(_ sequences: [Int]) async throws {}

    public func stopCalibration() async {
        if calibrationCollector.hasEnoughPoints {
            rbfCorrector.calibrate(calibrationCollector.samples)
        }
        calibrationCollector.reset()
    }

    // MARK: Pipeline processing

    private func processFrame(_ frame: CameraFrame) -> GazeSample {
        switch backend {
        case .mediaPipe:
            return processFrameMediaPipe(frame)
        case .vision:
            return processFrameVision(frame)
        }
    }

    // MARK: MediaPipe pipeline

    private func processFrameMediaPipe(_ frame: CameraFrame) -> GazeSample {
        guard let mediaPipe, let blazeGazeRunner else {
            return GazeSample.invalid()
        }

        mediaPipeTimestampMs += 33 // ~30fps

        // 1. MediaPipe landmark detection.
        guard let mpResult = try? mediaPipe.detect(
            pixelBuffer: frame.pixelBuffer, timestampMs: mediaPipeTimestampMs
        ) else {
            return GazeSample.invalid()
        }

        guard mpResult.landmarks.count >= 478 else {
            return GazeSample.invalid()
        }

        // 2. Homography eye patch (exact training format).
        guard let eyePatch = HomographyEyePatchExtractor.extract(
            pixelBuffer: frame.pixelBuffer,
            landmarks: mpResult.landmarks,
            frameWidth: frame.width,
            frameHeight: frame.height
        ) else {
            return GazeSample.invalid()
        }

        // 3. Head pose from facial transformation matrix.
        var headVector: MLMultiArray? = nil
        var faceOrigin: MLMultiArray? = nil
        if let ft = mpResult.faceTransform, ft.count == 4, ft[0].count >= 3 {
            let r20 = ft[2][0], r21 = ft[2][1], r22 = ft[2][2]
            let pitch = asin(-r20), yaw = atan2(r21, r22)
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

        return runBlazeGazeAndSmooth(
            eyePatch: eyePatch,
            headVector: headVector,
            faceOrigin: faceOrigin,
            blazeGazeRunner: blazeGazeRunner
        )
    }

    // MARK: Vision pipeline (fallback)

    private func processFrameVision(_ frame: CameraFrame) -> GazeSample {
        guard let blazeGazeRunner else { return GazeSample.invalid() }

        let detection = visionDetector.detect(frame)

        guard detection.stats.faceFound,
              let vnFace = visionDetector.lastRawObservation else {
            return GazeSample.invalid()
        }

        guard let eyePatch = visionEyePatch.extract(
            frame: frame.pixelBuffer, faceObservation: vnFace
        ) else {
            return GazeSample.invalid()
        }

        var headVector: MLMultiArray? = nil
        var faceOrigin: MLMultiArray? = nil
        if let pose = visionHeadPose.estimate(
            face: vnFace, frameWidth: frame.width, frameHeight: frame.height
        ) {
            headVector = try? MLMultiArray(shape: [1, 3], dataType: .float32)
            faceOrigin = try? MLMultiArray(shape: [1, 3], dataType: .float32)
            if let hv = headVector {
                hv[0] = pose.headVector[0] as NSNumber
                hv[1] = pose.headVector[1] as NSNumber
                hv[2] = pose.headVector[2] as NSNumber
            }
            if let fo = faceOrigin {
                fo[0] = pose.faceOrigin3D[0] as NSNumber
                fo[1] = pose.faceOrigin3D[1] as NSNumber
                fo[2] = pose.faceOrigin3D[2] as NSNumber
            }
        }

        return runBlazeGazeAndSmooth(
            eyePatch: eyePatch,
            headVector: headVector,
            faceOrigin: faceOrigin,
            blazeGazeRunner: blazeGazeRunner
        )
    }

    // MARK: Shared BlazeGaze + smoothing + RBF

    private func runBlazeGazeAndSmooth(
        eyePatch: CVPixelBuffer,
        headVector: MLMultiArray?,
        faceOrigin: MLMultiArray?,
        blazeGazeRunner: BlazeGazeRunner
    ) -> GazeSample {
        // BlazeGaze inference.
        guard let rawGaze = blazeGazeRunner.predict(
            eyePatch: eyePatch, headVector: headVector, faceOrigin3D: faceOrigin
        ) else {
            return GazeSample.invalid()
        }

        // Feed calibration collector if active.
        calibrationObservationHandler?(Double(rawGaze.x), Double(rawGaze.y))

        // RBF correction.
        let corrected: (x: Double, y: Double)
        if rbfCorrector.isCalibrated {
            corrected = rbfCorrector.correct(x: Double(rawGaze.x), y: Double(rawGaze.y))
        } else {
            corrected = (Double(rawGaze.x), Double(rawGaze.y))
        }

        // 1-Euro smoothing.
        let nowMs = GazeSample.nowMs()
        let smoothedX = smootherX.filter(value: corrected.x, timeMs: nowMs)
        let smoothedY = smootherY.filter(value: corrected.y, timeMs: nowMs)

        // Map to screen coordinates.
        let mapper = GazeCoordinateMapper()
        let screenPoint = mapper.map(normalizedX: smoothedX, normalizedY: smoothedY)

        return GazeSample(
            point: screenPoint,
            leftEyeNormalized: nil,
            rightEyeNormalized: nil,
            fixation: false,
            userPresent: true,
            leftEyeOpen: true,
            rightEyeOpen: true,
            confidence: nil,
            deviceTimestampMs: nowMs,
            receivedAtMs: nowMs
        )
    }

    // MARK: Public calibration API

    /// Called by the calibration UI when a new target is shown.
    public func beginCalibrationTarget(x: Double, y: Double) {
        calibrationCollector.startTarget(x: x, y: y)
        calibrationObservationHandler = { [weak self] gx, gy in
            self?.calibrationCollector.addObservation(x: gx, y: gy, timestampMs: GazeSample.nowMs())
        }
    }

    /// Called by the calibration UI when a target's dwell period ends.
    @discardableResult
    public func endCalibrationTarget() -> RBFGazeCorrector.CalibrationSample? {
        calibrationObservationHandler = nil
        return calibrationCollector.finishTarget()
    }

    // MARK: Internals

    private func setState(_ state: TrackerConnectionState) {
        stateLock.lock()
        _connectionState = state
        let conts = Array(stateContinuations.values)
        stateLock.unlock()
        for c in conts { c.yield(state) }
    }

    private static func resolveModelPath() -> String? {
        let candidates = [
            "Frameworks/face_landmarker_v2_with_blendshapes.task",
            FileManager.default.currentDirectoryPath + "/Frameworks/face_landmarker_v2_with_blendshapes.task",
            FileManager.default.currentDirectoryPath + "/macgaze/Frameworks/face_landmarker_v2_with_blendshapes.task",
        ]
        return candidates.first { FileManager.default.fileExists(atPath: $0) }
    }
}
