import Foundation
import CoreGraphics

/// A single observation of one face, holding the bits MacGaze actually
/// uses.  All coordinates are in normalized image space [0,1] with origin
/// bottom-left — same convention Vision reports.
public struct FaceObservation: Sendable, Equatable {
    /// Bounding box of the face in normalized image coordinates.
    public let boundingBox: CGRect
    /// Left-pupil centroid in normalized image coordinates, if Vision
    /// reported one.  ANE-estimated; robust to FaceTime HD ISP noise.
    public let leftPupil: CGPoint?
    /// Right-pupil centroid, ditto.
    public let rightPupil: CGPoint?
    /// Outer/inner eye corners used by downstream geometry (head pose,
    /// gaze-feature extraction).  All four may be nil if Vision couldn't
    /// resolve the eye contour.
    public let leftEyeCorners: EyeCorners?
    public let rightEyeCorners: EyeCorners?

    public struct EyeCorners: Sendable, Equatable {
        public let outer: CGPoint
        public let inner: CGPoint
        public init(outer: CGPoint, inner: CGPoint) {
            self.outer = outer; self.inner = inner
        }
    }

    public init(
        boundingBox: CGRect,
        leftPupil: CGPoint?,
        rightPupil: CGPoint?,
        leftEyeCorners: EyeCorners?,
        rightEyeCorners: EyeCorners?
    ) {
        self.boundingBox = boundingBox
        self.leftPupil = leftPupil
        self.rightPupil = rightPupil
        self.leftEyeCorners = leftEyeCorners
        self.rightEyeCorners = rightEyeCorners
    }
}

/// Performance metrics for a single detection pass; surfaced for the
/// debug overlay + later telemetry.
public struct DetectionStats: Sendable, Equatable {
    public let latencyMs: Double
    public let faceFound: Bool
    public init(latencyMs: Double, faceFound: Bool) {
        self.latencyMs = latencyMs; self.faceFound = faceFound
    }
}

/// Output of a `FaceLandmarkDetector.detect(_:)` call.
public struct DetectionResult: Sendable, Equatable {
    public let observation: FaceObservation?
    public let stats: DetectionStats
    public init(observation: FaceObservation?, stats: DetectionStats) {
        self.observation = observation; self.stats = stats
    }
}
