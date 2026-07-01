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
        case coreMLFaceMesh = "CoreML FaceMesh (468-pt, ANE)"
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

    // CoreML FaceMesh (ANE) backend.
    private var coreMLMesh: CoreMLFaceMeshLandmarker?

    // State.
    private let stateLock = NSLock()
    private var _connectionState: TrackerConnectionState = .disconnected
    private var stateContinuations: [UUID: AsyncStream<TrackerConnectionState>.Continuation] = [:]

    // Active frame-processing loop + fan-out buses.  A single loop consumes
    // `camera.frames` (keeping camera access single-consumer) and distributes
    // gaze samples, positioning info, and raw video to whichever streams are
    // subscribed.  This lets Track Status show live face/depth/video while the
    // cursor is paused — parity with the eyetuitive device's always-on feed.
    private var frameLoopTask: Task<Void, Never>?
    private let gazeBus = Bus<GazeSample>()
    private let positioningBus = Bus<PositioningInfo>()
    private let videoBus = Bus<TrackerVideoFrame>()

    // 1-Euro filter for smoothing.
    private var smootherX = OneEuroFilter()
    private var smootherY = OneEuroFilter()

    // Calibration callback (set externally by the calibration UI).
    private var calibrationObservationHandler: ((Double, Double) -> Void)?

    public init(backend: LandmarkBackend = .mediaPipe) {
        self.displayName = "Built-in Camera (MacGaze)"
        self.driverIdentifier = "macgaze.builtin"
        self.backend = backend
        // A/B override: MACGAZE_BACKEND=coreml | mediapipe | vision
        if let env = ProcessInfo.processInfo.environment["MACGAZE_BACKEND"]?.lowercased() {
            switch env {
            case "coreml", "coremlfacemesh", "facemesh": self.backend = .coreMLFaceMesh
            case "mediapipe", "mp": self.backend = .mediaPipe
            case "vision": self.backend = .vision
            default: break
            }
        }
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

        // Load the CoreML FaceMesh model if requested.
        if backend == .coreMLFaceMesh {
            if let url = Self.resolveFaceMeshModelURL() {
                self.coreMLMesh = try? CoreMLFaceMeshLandmarker(modelURL: url)
            }
            if coreMLMesh == nil {
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
            startFrameLoop()
            setState(.connected)
        } catch {
            setState(.connectionFailed(error.localizedDescription))
            throw error
        }
    }

    public func disconnect() async {
        frameLoopTask?.cancel()
        frameLoopTask = nil
        gazeBus.clear()
        positioningBus.clear()
        videoBus.clear()
        camera.stop()
        setState(.disconnected)
    }

    // MARK: TrackerDriver — Gaze stream

    public func gazeStream(unfiltered: Bool) -> AsyncStream<GazeSample> {
        gazeBus.subscribe(replayLatest: false)
    }

    // MARK: TrackerDriver — Positioning

    public func positioningStream() -> AsyncStream<PositioningInfo> {
        positioningBus.subscribe(replayLatest: true)
    }

    // MARK: TrackerDriver — Video

    public func videoStream() -> AsyncStream<TrackerVideoFrame>? {
        videoBus.subscribe(replayLatest: true)
    }

    // MARK: Demand-driven frame loop (gaze + positioning + video fan-out)

    /// Sole consumer of `camera.frames`. Work is scaled to demand so we never
    /// saturate the CPU and get jetsam-killed:
    ///
    /// • gaze subscriber attached (cursor injection / calibration) → run the
    ///   full pipeline every frame: landmarks → homography eye patch → head
    ///   pose → BlazeGaze → RBF. This is the only path that needs the expensive
    ///   homography warp + CNN inference.
    /// • only positioning/video wanted (Track Status open, cursor paused) →
    ///   run the cheap face-detect path, throttled to ~10 fps.
    /// • nobody listening → consume the frame and do nothing, so the camera
    ///   queue drains instead of backing up.
    private func startFrameLoop() {
        frameLoopTask?.cancel()
        frameLoopTask = Task { [weak self] in
            guard let self else { return }
            let auxInterval: TimeInterval = 1.0 / 10.0      // positioning/video idle cap
            let videoInterval: TimeInterval = 1.0 / 12.0
            var lastAux = Date.distantPast
            var lastVideo = Date.distantPast
            for await frame in self.camera.frames {
                if Task.isCancelled { break }
                let now = Date()

                // Full gaze pipeline only while a gaze consumer is attached.
                if self.gazeBus.hasSubscribers {
                    let (sample, faceBox) = self.analyzeFrame(frame)
                    self.gazeBus.yield(sample)
                    if let box = faceBox {
                        self.positioningBus.yield(Self.makePositioning(faceBox: box))
                    }
                    if now.timeIntervalSince(lastVideo) >= videoInterval,
                       let vf = Self.makeVideoFrame(from: frame) {
                        self.videoBus.yield(vf)
                        lastVideo = now
                    }
                    continue
                }

                // Idle / Track-Status-only: nothing to do unless someone wants
                // positioning or video.
                guard self.positioningBus.hasSubscribers || self.videoBus.hasSubscribers else {
                    continue
                }
                if now.timeIntervalSince(lastAux) < auxInterval { continue }
                lastAux = now

                if let box = self.faceBoxOnly(frame) {
                    self.positioningBus.yield(Self.makePositioning(faceBox: box))
                }
                if self.videoBus.hasSubscribers,
                   now.timeIntervalSince(lastVideo) >= videoInterval,
                   let vf = Self.makeVideoFrame(from: frame) {
                    self.videoBus.yield(vf)
                    lastVideo = now
                }
            }
            self.gazeBus.clear()
        }
    }

    /// Build a `PositioningInfo` from the detected face box (normalized,
    /// top-left origin).  Depth is a rough estimate from face-width fraction;
    /// eye "open" states default to open (blink detection not yet wired).
    private static func makePositioning(faceBox box: CGRect) -> PositioningInfo {
        let widthFrac = max(0.08, min(0.6, Double(box.width)))
        // Empirical: a face ~0.26 of frame width ≈ 600 mm on a typical
        // FaceTime HD.  Clamp to the depth-zone range the UI understands.
        let depthMM = max(350.0, min(950.0, 160.0 / widthFrac))
        let cy = Double(box.midY)
        let leftX = min(1.0, max(0.0, Double(box.minX) + widthFrac * 0.30))
        let rightX = min(1.0, max(0.0, Double(box.maxX) - widthFrac * 0.30))
        return PositioningInfo(
            depthInMM: depthMM,
            leftEyePos: CGPoint(x: leftX, y: cy),
            rightEyePos: CGPoint(x: rightX, y: cy),
            leftEyeClosed: false,
            rightEyeClosed: false,
            gazeIsPaused: false
        )
    }

    /// Normalized bounding rect (top-left origin) of an array of [x, y(, z)]
    /// landmark points in image space.
    private static func faceBox(fromLandmarks landmarks: [[Double]]) -> CGRect? {
        guard !landmarks.isEmpty else { return nil }
        var minX = Double.infinity, minY = Double.infinity
        var maxX = -Double.infinity, maxY = -Double.infinity
        for lm in landmarks {
            guard lm.count >= 2 else { continue }
            let x = lm[0], y = lm[1]
            if x < minX { minX = x }
            if y < minY { minY = y }
            if x > maxX { maxX = x }
            if y > maxY { maxY = y }
        }
        guard minX.isFinite, maxX > minX, maxY > minY else { return nil }
        return CGRect(x: minX, y: minY, width: maxX - minX, height: maxY - minY)
    }

    /// Reconstruct BlazeGaze's `head_vector` + `face_origin_3d` from FaceMesh
    /// landmarks via `HeadPoseSolver`. Returns nil on failure — BlazeGaze then
    /// falls back to neutral head pose. Mirrors the macgaze-control CLI path.
    private static func headPose(
        fromLandmarks landmarks: [[Double]], width: Int, height: Int
    ) -> (headVector: MLMultiArray, faceOrigin: MLMultiArray)? {
        guard let pose = HeadPoseSolver.solve(landmarks: landmarks, width: width, height: height),
              let hv = makeVec(pose.headVector),
              let fo = makeVec(pose.faceOrigin3D) else { return nil }
        return (hv, fo)
    }

    private static func makeVec(_ v: [Float]) -> MLMultiArray? {
        guard v.count == 3,
              let a = try? MLMultiArray(shape: [1, 3], dataType: .float32) else { return nil }
        a[0] = v[0] as NSNumber; a[1] = v[1] as NSNumber; a[2] = v[2] as NSNumber
        return a
    }

    /// Copy a BGRA `CVPixelBuffer` into a tightly-packed `TrackerVideoFrame`,
    /// de-striding if the capture surface has row padding.
    private static func makeVideoFrame(from frame: CameraFrame) -> TrackerVideoFrame? {
        let pb = frame.pixelBuffer
        let width = CVPixelBufferGetWidth(pb)
        let height = CVPixelBufferGetHeight(pb)
        let stride = CVPixelBufferGetBytesPerRow(pb)
        guard width > 0, height > 0, stride > 0 else { return nil }
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pb) else { return nil }
        let tight = width * 4
        let data: Data
        if stride == tight {
            data = Data(bytes: base, count: stride * height)
        } else {
            var packed = Data(capacity: tight * height)
            for row in 0..<height {
                let src = base.advanced(by: row * stride)
                    .assumingMemoryBound(to: UInt8.self)
                packed.append(UnsafeBufferPointer(start: src, count: tight))
            }
            data = packed
        }
        return TrackerVideoFrame(
            width: width, height: height, channels: 4, data: data,
            timestamp: Int64(frame.timestampSeconds * 1000)
        )
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

    /// Processes one camera frame, returning the gaze sample plus the detected
    /// face bounding box (normalized, top-left origin) used for positioning.
    /// `nil` box means no face was found this frame.
    private func analyzeFrame(_ frame: CameraFrame) -> (GazeSample, CGRect?) {
        switch backend {
        case .mediaPipe:
            return processFrameMediaPipe(frame)
        case .vision:
            return processFrameVision(frame)
        case .coreMLFaceMesh:
            return processFrameCoreMLMesh(frame)
        }
    }

    /// Cheap path for when only positioning/video is wanted (no gaze consumer):
    /// run the landmarker purely for a face box, skipping the expensive
    /// homography eye-patch + head pose + BlazeGaze inference.
    private func faceBoxOnly(_ frame: CameraFrame) -> CGRect? {
        switch backend {
        case .coreMLFaceMesh:
            guard let coreMLMesh,
                  let r = coreMLMesh.detect(pixelBuffer: frame.pixelBuffer),
                  r.landmarks.count >= 468 else { return nil }
            return Self.faceBox(fromLandmarks: r.landmarks)
        case .mediaPipe:
            guard let mediaPipe else { return nil }
            mediaPipeTimestampMs += 33
            guard let r = try? mediaPipe.detect(
                pixelBuffer: frame.pixelBuffer, timestampMs: mediaPipeTimestampMs
            ), r.landmarks.count >= 468 else { return nil }
            return Self.faceBox(fromLandmarks: r.landmarks)
        case .vision:
            let d = visionDetector.detect(frame)
            guard d.stats.faceFound, let b = d.observation?.boundingBox else { return nil }
            // Vision box is bottom-left origin; flip to top-left.
            return CGRect(x: b.minX, y: 1 - b.maxY, width: b.width, height: b.height)
        }
    }

    // MARK: CoreML FaceMesh pipeline (ANE landmarks → same homography path)

    private func processFrameCoreMLMesh(_ frame: CameraFrame) -> (GazeSample, CGRect?) {
        guard let coreMLMesh, let blazeGazeRunner else { return (GazeSample.invalid(), nil) }

        guard let result = coreMLMesh.detect(pixelBuffer: frame.pixelBuffer),
              result.landmarks.count >= 468 else {
            return (GazeSample.invalid(), nil)
        }

        let faceBox = Self.faceBox(fromLandmarks: result.landmarks)

        guard let eyePatch = HomographyEyePatchExtractor.extract(
            pixelBuffer: frame.pixelBuffer,
            landmarks: result.landmarks,
            frameWidth: frame.width,
            frameHeight: frame.height
        ) else {
            return (GazeSample.invalid(), faceBox)
        }

        // Head pose reconstructed from the 468-pt mesh (Kabsch alignment to the
        // canonical face model), matching the validated macgaze-control CLI.
        let headPose = Self.headPose(fromLandmarks: result.landmarks,
                                     width: frame.width, height: frame.height)
        // Metric face origin (cm) — BlazeGaze was trained with WebEyeTrack's
        // cm-scale face_origin_3d; pixel-scale values put it out of distribution.
        let faceOrigin = Self.makeVec(MetricFaceOrigin.compute(
            landmarks: result.landmarks, width: frame.width, height: frame.height))
        let sample = runBlazeGazeAndSmooth(
            eyePatch: eyePatch,
            headVector: headPose?.headVector,
            faceOrigin: faceOrigin,
            blazeGazeRunner: blazeGazeRunner
        )
        return (sample, faceBox)
    }

    // MARK: MediaPipe pipeline

    private func processFrameMediaPipe(_ frame: CameraFrame) -> (GazeSample, CGRect?) {
        guard let mediaPipe, let blazeGazeRunner else {
            return (GazeSample.invalid(), nil)
        }

        mediaPipeTimestampMs += 33 // ~30fps

        // 1. MediaPipe landmark detection.
        guard let mpResult = try? mediaPipe.detect(
            pixelBuffer: frame.pixelBuffer, timestampMs: mediaPipeTimestampMs
        ) else {
            return (GazeSample.invalid(), nil)
        }

        guard mpResult.landmarks.count >= 478 else {
            return (GazeSample.invalid(), nil)
        }

        let faceBox = Self.faceBox(fromLandmarks: mpResult.landmarks)

        // 2. Homography eye patch (exact training format).
        guard let eyePatch = HomographyEyePatchExtractor.extract(
            pixelBuffer: frame.pixelBuffer,
            landmarks: mpResult.landmarks,
            frameWidth: frame.width,
            frameHeight: frame.height
        ) else {
            return (GazeSample.invalid(), faceBox)
        }

        // 3. Head pose from facial transformation matrix.
        var headVector: MLMultiArray? = nil
        var rotationR: [[Double]]? = nil
        if let ft = mpResult.faceTransform, ft.count == 4, ft[0].count >= 3 {
            let r20 = ft[2][0], r21 = ft[2][1], r22 = ft[2][2]
            let pitch = asin(-r20), yaw = atan2(r21, r22)
            let hPitch = -yaw, hYaw = pitch
            let cp = cos(hPitch), sp = sin(hPitch)
            let cy = cos(hYaw), sy = sin(hYaw)
            headVector = try? MLMultiArray(shape: [1, 3], dataType: .float32)
            if let hv = headVector {
                hv[0] = Float(cp * sy) as NSNumber
                hv[1] = Float(sp) as NSNumber
                hv[2] = Float(-cp * cy) as NSNumber
            }
            rotationR = [
                [ft[0][0], ft[0][1], ft[0][2]],
                [ft[1][0], ft[1][1], ft[1][2]],
                [ft[2][0], ft[2][1], ft[2][2]],
            ]
        }
        // Metric face origin (cm) — see processFrameCoreMLMesh.
        let faceOrigin = Self.makeVec(MetricFaceOrigin.compute(
            landmarks: mpResult.landmarks, width: frame.width, height: frame.height,
            rotationR: rotationR))

        let sample = runBlazeGazeAndSmooth(
            eyePatch: eyePatch,
            headVector: headVector,
            faceOrigin: faceOrigin,
            blazeGazeRunner: blazeGazeRunner
        )
        return (sample, faceBox)
    }

    // MARK: Vision pipeline (fallback)

    private func processFrameVision(_ frame: CameraFrame) -> (GazeSample, CGRect?) {
        guard let blazeGazeRunner else { return (GazeSample.invalid(), nil) }

        let detection = visionDetector.detect(frame)

        // Vision bounding box is normalized with origin bottom-left; flip to
        // top-left so it matches the landmark-based boxes and the rendering.
        let faceBox: CGRect? = detection.observation.map {
            CGRect(x: $0.boundingBox.minX,
                   y: 1 - $0.boundingBox.maxY,
                   width: $0.boundingBox.width,
                   height: $0.boundingBox.height)
        }

        guard detection.stats.faceFound,
              let vnFace = visionDetector.lastRawObservation else {
            return (GazeSample.invalid(), faceBox)
        }

        guard let eyePatch = visionEyePatch.extract(
            frame: frame.pixelBuffer, faceObservation: vnFace
        ) else {
            return (GazeSample.invalid(), faceBox)
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

        let sample = runBlazeGazeAndSmooth(
            eyePatch: eyePatch,
            headVector: headVector,
            faceOrigin: faceOrigin,
            blazeGazeRunner: blazeGazeRunner
        )
        return (sample, faceBox)
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

    private static func resolveFaceMeshModelURL() -> URL? {
        // Bundled resource (GUI app) first, then dev/cwd-relative paths (CLI).
        if let url = Bundle.main.url(forResource: "face_mesh", withExtension: "mlmodelc") {
            return url
        }
        let cwd = FileManager.default.currentDirectoryPath
        let candidates = [
            "Models/face_mesh.mlmodelc",
            cwd + "/Models/face_mesh.mlmodelc",
            cwd + "/macgaze/Models/face_mesh.mlmodelc",
        ]
        return candidates.first { FileManager.default.fileExists(atPath: $0) }
            .map { URL(fileURLWithPath: $0) }
    }
}

/// Thread-safe multi-subscriber broadcast for a value type.  Used to fan a
/// single camera-frame analysis out to the gaze / positioning / video streams.
private final class Bus<T: Sendable>: @unchecked Sendable {
    private let lock = NSLock()
    private var subscribers: [UUID: AsyncStream<T>.Continuation] = [:]
    private var latest: T?

    /// Subscribe to future values.  When `replayLatest` is true, the most
    /// recently yielded value (if any) is emitted immediately so a late
    /// subscriber sees current state without waiting for the next frame.
    func subscribe(replayLatest: Bool) -> AsyncStream<T> {
        AsyncStream { continuation in
            let id = UUID()
            lock.lock()
            subscribers[id] = continuation
            let snapshot = latest
            lock.unlock()
            if replayLatest, let snapshot {
                continuation.yield(snapshot)
            }
            continuation.onTermination = { [weak self] _ in
                guard let self else { return }
                self.lock.lock()
                self.subscribers.removeValue(forKey: id)
                self.lock.unlock()
            }
        }
    }

    /// Whether anyone is currently subscribed. Used by the frame loop to decide
    /// whether the heavy gaze pipeline needs to run.
    var hasSubscribers: Bool {
        lock.lock(); defer { lock.unlock() }
        return !subscribers.isEmpty
    }

    func yield(_ value: T) {
        lock.lock()
        latest = value
        let conts = Array(subscribers.values)
        lock.unlock()
        for c in conts { c.yield(value) }
    }

    func clear() {
        lock.lock()
        subscribers.removeAll()
        latest = nil
        lock.unlock()
    }
}
