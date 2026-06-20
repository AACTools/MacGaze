import Foundation
import CoreVideo
import CoreGraphics
import Vision

/// Runs `VNDetectFaceLandmarksRequest` (revision 3) on a `CVPixelBuffer`
/// and yields the bits MacGaze needs: bounding box, pupil centroids,
/// and eye-corner points for downstream geometry.
///
/// **Revision 3** is the right call — it's the most recent and uses
/// Apple's ANE-optimised CNN that's robust to the FaceTime HD's aggressive
/// ISP denoising.  The `leftPupil` / `rightPupil` landmark regions it
/// produces are the foundation of our gaze pipeline (no custom edge
/// detection required).
///
/// One instance can be reused across frames; Vision's request handler is
/// per-frame but the request object is cheap to keep around.
public final class FaceLandmarkDetector: @unchecked Sendable {

    /// The raw `VNFaceObservation` from the most recent successful detect()
    /// call.  Consumers that need the full Vision landmark regions (e.g.
    /// EyePatchExtractor) can read this instead of carrying their own copy.
    public private(set) var lastRawObservation: VNFaceObservation?

    public struct Configuration: Sendable {
        /// Vision frame-region-of-interest in normalized image coords.
        /// `nil` = whole frame.  Useful for focusing on a sub-region in
        /// the future; default is whole frame for Phase 0.
        public var regionOfInterest: CGRect? = nil
        public init() {}
    }

    public let configuration: Configuration

    public init(configuration: Configuration = Configuration()) {
        self.configuration = configuration
    }

    /// Run landmark detection on a single camera frame.  Must be called
    /// off the main thread (Vision blocks).  `frame.pixelBuffer` is read
    /// synchronously inside this call, so the buffer's lifetime is bounded.
    public func detect(_ frame: CameraFrame) -> DetectionResult {
        let t0 = Date()
        let request = VNDetectFaceLandmarksRequest()
        request.revision = VNDetectFaceLandmarksRequestRevision3
        // Vision wants orientation.  The AVCapture front-camera output is
        // mirrored by us at the display layer; the raw pixel buffer is
        // right-way-up for Vision when the connection's isVideoMirrored
        // is false.  We set that explicitly in CameraCapture, so .up is
        // correct here.
        let handler = VNImageRequestHandler(
            cvPixelBuffer: frame.pixelBuffer,
            orientation: .up,
            options: [:]
        )
        do {
            try handler.perform([request])
        } catch {
            return DetectionResult(
                observation: nil,
                stats: DetectionStats(
                    latencyMs: Date().timeIntervalSince(t0) * 1000,
                    faceFound: false
                )
            )
        }

        guard let face = request.results?.first else {
            self.lastRawObservation = nil
            return DetectionResult(
                observation: nil,
                stats: DetectionStats(
                    latencyMs: Date().timeIntervalSince(t0) * 1000,
                    faceFound: false
                )
            )
        }

        self.lastRawObservation = face

        let observation = FaceObservation(
            boundingBox: face.boundingBox,
            leftPupil: centroid(face.landmarks?.leftPupil),
            rightPupil: centroid(face.landmarks?.rightPupil),
            leftEyeCorners: eyeCorners(face.landmarks?.leftEye),
            rightEyeCorners: eyeCorners(face.landmarks?.rightEye)
        )

        return DetectionResult(
            observation: observation,
            stats: DetectionStats(
                latencyMs: Date().timeIntervalSince(t0) * 1000,
                faceFound: true
            )
        )
    }

    // MARK: Helpers

    /// Mean of a landmark region's normalized points, or nil if absent.
    private func centroid(_ region: VNFaceLandmarkRegion2D?) -> CGPoint? {
        guard let region, region.pointCount > 0 else { return nil }
        let count = CGFloat(region.pointCount)
        var sumX: CGFloat = 0
        var sumY: CGFloat = 0
        for i in 0..<region.pointCount {
            let p = region.normalizedPoints[i]
            sumX += p.x
            sumY += p.y
        }
        return CGPoint(x: sumX / count, y: sumY / count)
    }

    /// Outer + inner canthus (eye corner) for one eye.
    /// Vision doesn't expose "canthus" as a named region; we approximate
    /// from the eye contour by picking the points with the most extreme
    /// x values.  For the left eye (subject's left, image's right) the
    /// outer canthus is the higher x; inner is lower x.  For the right
    /// eye it's the mirror.
    /// This is good enough for Procrustes alignment + head-pose
    /// initialisation; can refine later if needed.
    private func eyeCorners(_ region: VNFaceLandmarkRegion2D?) -> FaceObservation.EyeCorners? {
        guard let region, region.pointCount >= 2 else { return nil }
        // Pick the two points with the most extreme x values.  For both
        // eyes (subject's left and right), the lower-x point is the
        // inner corner (closer to the nose) and the higher-x point is
        // the outer corner — this is consistent because Vision reports
        // the face upright and the nose is centred in x.
        var minPoint = CGPoint.zero
        var maxPoint = CGPoint.zero
        var minX: CGFloat = .infinity
        var maxX: CGFloat = -.infinity
        for i in 0..<region.pointCount {
            let p = region.normalizedPoints[i]
            if p.x < minX { minX = p.x; minPoint = p }
            if p.x > maxX { maxX = p.x; maxPoint = p }
        }
        let inner = minPoint.x < maxPoint.x ? minPoint : maxPoint
        let outer = minPoint.x < maxPoint.x ? maxPoint : minPoint
        return FaceObservation.EyeCorners(outer: outer, inner: inner)
    }
}
