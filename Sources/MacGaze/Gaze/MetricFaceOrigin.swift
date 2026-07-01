import Foundation

/// Computes the metric `face_origin_3d` (in centimetres) that BlazeGaze expects.
///
/// Background: BlazeGaze was trained with WebEyeTrack's `compute_face_origin_3d`,
/// which runs a full 3D face reconstruction and returns the face origin in
/// real-world **centimetres** (face depth ≈ 60 cm).  We used to feed pixel-space
/// or raw-translation values — orders of magnitude off — which drove the model
/// wildly out of distribution (raw outputs of -1.5 / +2.3 instead of [0,1]).
///
/// This reproduces the *scale* of WebEyeTrack's result using the same
/// similar-triangles depth relationship (`depth ≈ focal·IPD / apparent_IPD`)
/// plus a perspective mapping for the lateral X/Y position, without needing
/// the iris landmarks or iterative solver the full reconstruction requires.
/// The dominant, model-critical component is Z (depth); X/Y are small offsets
/// that the RBF calibration layer corrects for anyway.
public enum MetricFaceOrigin {

    /// Real-world inter-pupillary distance (cm). Matches WebEyeTrack constant.
    static let realWorldIPDCm: Double = 6.3
    /// Assumed camera vertical field of view (degrees). Matches WebEyeTrack.
    static let verticalFOVDegrees: Double = 60
    /// Horizontal eye-corner landmark indices (same as HeadPoseSolver).
    static let leftEyeHorizontal = [362, 263]
    static let rightEyeHorizontal = [33, 133]

    /// Compute the face origin in cm.
    ///
    /// - Parameters:
    ///   - landmarks: N×3 normalized [0,1] (top-left origin), with relative z.
    ///   - width/height: frame size in pixels.
    ///   - rotationR: optional 3×3 head rotation (used only for a tilt
    ///     correction on depth); pass nil if unavailable.
    /// - Returns: `[x_cm, y_cm, depth_cm]` — roughly `[±few, ±few, ~60]`.
    public static func compute(
        landmarks: [[Double]],
        width: Int, height: Int,
        rotationR: [[Double]]? = nil
    ) -> [Float] {
        let w = Double(width), h = Double(height)

        // 2D eye centers in pixels.
        let left2d = mean2D(landmarks, indices: leftEyeHorizontal, width: w, height: h)
        let right2d = mean2D(landmarks, indices: rightEyeHorizontal, width: w, height: h)
        let imageIPDPx = (left2d.x - right2d.x) * (left2d.x - right2d.x)
                       + (left2d.y - right2d.y) * (left2d.y - right2d.y)
        let imageIPD = imageIPDPx.squareRoot()
        guard imageIPD > 1 else { return [0, 0, 60] }

        // Focal length in pixels for the assumed vertical FOV.
        let focalPx = h / (2.0 * tan(verticalFOVDegrees * .pi / 180.0 / 2.0))

        // Head-tilt theta = atan(R[0][2] / R[2][2]) (WebEyeTrack depth formula).
        let theta: Double
        if let R = rotationR, R.count == 3, R[0].count >= 3, R[2].count >= 3 {
            theta = atan2(R[0][2], R[2][2])
        } else {
            theta = 0
        }

        // Face depth (cm) via similar triangles, tilt-corrected.
        let depthCm = (focalPx * realWorldIPDCm * cos(theta)) / imageIPD

        // At this depth, 1 pixel ≈ depth/focal cm. Map the face centre offset
        // from the frame centre to cm (Y flipped to "up").
        let cmPerPx = depthCm / focalPx
        let eyeMidXpx = (left2d.x + right2d.x) / 2.0
        let eyeMidYpx = (left2d.y + right2d.y) / 2.0
        let originXCm = (eyeMidXpx - w / 2.0) * cmPerPx
        let originYCm = (h / 2.0 - eyeMidYpx) * cmPerPx

        return [Float(originXCm), Float(originYCm), Float(depthCm)]
    }

    private static func mean2D(
        _ landmarks: [[Double]], indices: [Int], width: Double, height: Double
    ) -> (x: Double, y: Double) {
        var sx = 0.0, sy = 0.0
        for i in indices {
            guard i < landmarks.count, landmarks[i].count >= 2 else { continue }
            sx += landmarks[i][0] * width
            sy += landmarks[i][1] * height
        }
        let n = Double(indices.count)
        return (sx / n, sy / n)
    }
}
