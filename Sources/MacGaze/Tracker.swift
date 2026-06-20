import Foundation
import CoreGraphics
import CoreML
import Vision
import GazeBridgeCore

/// MacGaze's implementation of GazeBridge's `TrackerDriver` protocol.
///
/// Uses the built-in FaceTime HD camera + Vision landmarks + BlazeGaze
/// CoreML model + Gaussian RBF personalisation to produce gaze data
/// — no external hardware required.
///
/// This is what lets GazeBridge's menu-bar app drive either an eyetuitive
/// (via GazeFirstTracker) OR the built-in camera (via MacGazeTracker).
public final class MacGazeTracker: TrackerDriver, @unchecked Sendable {

    public let displayName: String = "Built-in Camera (MacGaze)"
    public let driverIdentifier: String = "macgaze.builtin"

    // Pipeline components.
    private let camera: CameraCapture
    private let detector: FaceLandmarkDetector
    private let eyePatchExtractor: EyePatchExtractor
    private let headPose: HeadPoseEstimator
    private var blazeGaze: BlazeGazeRunner?
    private let rbfCorrector: RBFGazeCorrector
    private let calibrationCollector: CalibrationCollector

    // State.
    private let stateLock = NSLock()
    private var _connectionState: TrackerConnectionState = .disconnected
    private var stateContinuations: [UUID: AsyncStream<TrackerConnectionState>.Continuation] = [:]

    // Active streams.
    private var gazeStreamTask: Task<Void, Never>?
    private var cameraStream: AsyncStream<CameraFrame>?
    private var cameraContinuation: AsyncStream<CameraFrame>.Continuation?

    // 1-Euro filter for smoothing (reuses GazeBridgeCore).
    private var smootherX = OneEuroFilter()
    private var smootherY = OneEuroFilter()

    public init() {
        self.camera = CameraCapture()
        self.detector = FaceLandmarkDetector()
        self.eyePatchExtractor = EyePatchExtractor()
        self.headPose = HeadPoseEstimator()
        self.rbfCorrector = RBFGazeCorrector()
        self.calibrationCollector = CalibrationCollector()

        // Set up camera stream that MacGazeTracker owns.
        let stream = AsyncStream<CameraFrame> { continuation in
            self.cameraContinuation = continuation
        }
        self.cameraStream = stream
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
                self?.stateLock.lock()
                self?.stateContinuations.removeValue(forKey: id)
                self?.stateLock.unlock()
            }
        }
    }

    public func connect() async throws {
        setState(.connecting)
        do {
            try await camera.start()
            // Try to load BlazeGaze model (non-fatal if missing).
            blazeGaze = try? BlazeGazeRunner()
            setState(.connected)
        } catch {
            setState(.connectionFailed(error.localizedDescription))
            throw error
        }
    }

    public func disconnect() async {
        gazeStreamTask?.cancel()
        gazeStreamTask = nil
        cameraContinuation?.finish()
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
            continuation.onTermination = { _ in
                self.gazeStreamTask?.cancel()
            }
        }
    }

    // MARK: TrackerDriver — Positioning

    public func positioningStream() -> AsyncStream<PositioningInfo> {
        // MacGaze doesn't have hardware depth sensing; return a synthetic
        // stream with estimated depth from face bounding box size.
        AsyncStream { continuation in
            // Phase 5: wire real positioning from HeadPoseEstimator.
            // For now, emit nothing — GazeBridge handles nil gracefully.
            continuation.finish()
        }
    }

    // MARK: TrackerDriver — Device info

    public func deviceInformation() async throws -> TrackerDeviceInformation {
        TrackerDeviceInformation(
            serial: 0,
            firmwareVersion: blazeGaze != nil ? "BlazeGaze 1.0" : "unloaded",
            hardwareConfig: 0,
            machine: "FaceTime HD",
            cpuTempCelsius: 0
        )
    }

    public func currentUserSettings() async throws -> TrackerUserSettings {
        TrackerUserSettings(
            smoothing: 5,
            leftEyeOnly: false,
            rightEyeOnly: false
        )
    }

    public func applyUserSettings(_ settings: TrackerUserSettings) async throws {
        // Update 1-Euro filter configuration from smoothing slider.
        let cfg = OneEuroFilter.Configuration.from(smoothingSlider: settings.smoothing)
        smootherX = OneEuroFilter(cfg)
        smootherY = OneEuroFilter(cfg)
    }

    // MARK: TrackerDriver — Calibration

    public func calibrate(_ options: CalibrationOptions) -> AsyncStream<CalibrationEvent> {
        AsyncStream { continuation in
            // For Phase 3: we reuse GazeBridge's calibration UI to show
            // targets, then collect BlazeGaze outputs via this stream.
            // The actual target loop is driven by GazeBridge's
            // CalibrationWindowController, which calls
            // calibrationCollector.startTarget / addObservation /
            // finishTarget as each point is shown.
            //
            // For now, emit .started and let the UI drive the flow.
            // When all points are collected, GazeBridge calls
            // confirmCurrentCalibrationPoint() which triggers the RBF solve.
            continuation.yield(.started)
        }
    }

    public func lastCalibrationResult() async throws -> CalibrationResultSummary? {
        guard rbfCorrector.isCalibrated else { return nil }
        return CalibrationResultSummary(
            overallRating: 0,  // not measured yet
            points: [],
            canImprove: false,
            timestampMs: Int64(Date().timeIntervalSince1970 * 1000)
        )
    }

    public func confirmCurrentCalibrationPoint() async throws {
        // Finish the current target and collect the sample.
        _ = calibrationCollector.finishTarget()
    }

    public func improveCalibrationPoints(_ sequences: [Int]) async throws {
        // Phase 5: re-collect specific calibration points.
    }

    public func stopCalibration() async {
        if calibrationCollector.hasEnoughPoints {
            rbfCorrector.calibrate(calibrationCollector.samples)
        }
        calibrationCollector.reset()
    }

    // MARK: Pipeline processing

    /// Process one camera frame through the full pipeline.
    private func processFrame(_ frame: CameraFrame) -> GazeSample {
        // 1. Vision landmark detection.
        let detection = detector.detect(frame)

        guard detection.stats.faceFound,
              let vnFace = detector.lastRawObservation,
              let blazeGaze else {
            return GazeSample.invalid()
        }

        // 2. Extract eye patch.
        guard let eyePatch = eyePatchExtractor.extract(
            frame: frame.pixelBuffer,
            faceObservation: vnFace
        ) else {
            return GazeSample.invalid()
        }

        // 3. Head pose for BlazeGaze's auxiliary inputs.
        var headVector: MLMultiArray? = nil
        var faceOrigin: MLMultiArray? = nil
        if let pose = headPose.estimate(
            face: vnFace,
            frameWidth: frame.width,
            frameHeight: frame.height
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

        // 4. BlazeGaze inference.
        guard let rawGaze = blazeGaze.predict(
            eyePatch: eyePatch,
            headVector: headVector,
            faceOrigin3D: faceOrigin
        ) else {
            return GazeSample.invalid()
        }

        // 5. RBF correction (if calibrated).
        let corrected: (x: Double, y: Double)
        if rbfCorrector.isCalibrated {
            corrected = rbfCorrector.correct(x: Double(rawGaze.x), y: Double(rawGaze.y))
        } else {
            corrected = (Double(rawGaze.x), Double(rawGaze.y))
        }

        // 6. 1-Euro smoothing.
        let nowMs = GazeSample.nowMs()
        let smoothedX = smootherX.filter(value: corrected.x, timeMs: nowMs)
        let smoothedY = smootherY.filter(value: corrected.y, timeMs: nowMs)

        // 7. Map to screen coordinates via GazeCoordinateMapper.
        let mapper = GazeCoordinateMapper()
        let screenPoint = mapper.map(normalizedX: smoothedX, normalizedY: smoothedY)

        // 8. Feed calibration collector if a target is active.
        if calibrationCollector.samples.count < 9 {  // still calibrating
            // CalibrationCollector.addObservation is called externally by
            // the calibration UI, not here.
        }

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

    // MARK: Public API for calibration UI

    /// Called by the calibration UI when a new target is shown.
    public func beginCalibrationTarget(x: Double, y: Double) {
        calibrationCollector.startTarget(x: x, y: y)
    }

    /// Called by the calibration UI to feed a raw gaze observation.
    public func feedCalibrationObservation(x: Double, y: Double) {
        calibrationCollector.addObservation(
            x: x, y: y,
            timestampMs: GazeSample.nowMs()
        )
    }

    /// Called by the calibration UI when a target's dwell period ends.
    @discardableResult
    public func endCalibrationTarget() -> RBFGazeCorrector.CalibrationSample? {
        calibrationCollector.finishTarget()
    }

    // MARK: Internals

    private func setState(_ state: TrackerConnectionState) {
        stateLock.lock()
        _connectionState = state
        let conts = Array(stateContinuations.values)
        stateLock.unlock()
        for c in conts { c.yield(state) }
    }
}
