import Foundation
import Vision
import CoreGraphics
import CoreImage
import GazeBridgeCore

/// Estimates head pose from Vision landmarks, producing the
/// `head_vector[3]` + `face_origin_3d[3]` inputs that BlazeGaze needs.
///
/// **Pragmatic Phase 2.3 approach:**
/// - `yaw` and `roll` come directly from `VNFaceObservation` (free, accurate).
/// - `pitch` is estimated from the nose landmark's vertical position relative
///   to the eye midpoint (approximate but sufficient for BlazeGaze).
/// - `face_origin_3d` is estimated from the face bounding box size and
///   the assumed ~60° camera FOV.
///
/// This replaces the need for a full EPnP solver against a 3D face model.
/// If accuracy is poor, we upgrade to EPnP later (Phase 5).
public final class HeadPoseEstimator {

    /// Average human face width (cm) for depth estimation.
    private static let averageFaceWidthCM: Double = 14.0

    public init() {}

    /// Compute head_vector and face_origin_3d from a Vision face observation.
    ///
    /// - Parameters:
    ///   - face: The `VNFaceObservation` from `VNDetectFaceLandmarksRequest`.
    ///   - frameWidth: Camera frame width in pixels.
    ///   - frameHeight: Camera frame height in pixels.
    /// - Returns: A tuple `(headVector: [Float], faceOrigin3D: [Float])`
    ///            ready for BlazeGaze, or nil if the face isn't usable.
    public func estimate(
        face: VNFaceObservation,
        frameWidth: Int,
        frameHeight: Int
    ) -> (headVector: [Float], faceOrigin3D: [Float])? {

        // --- Yaw + Roll from Vision (radians) ---
        let yaw = face.yaw?.floatValue ?? 0
        let roll = face.roll?.floatValue ?? 0

        // --- Pitch from landmark geometry ---
        // If the nose tip is higher than the eye midpoint, head is tilted up
        // (positive pitch). If lower, tilted down (negative pitch).
        let pitch = estimatePitch(face: face)

        // --- Convert to head_vector via WebEyeTrack's formula ---
        // get_head_vector():
        //   h_pitch = -yaw
        //   h_yaw   = pitch
        //   h_roll  = roll
        //   → pitch_yaw_roll_to_gaze_vector(h_pitch, h_yaw, h_roll)
        let hPitch = Double(-yaw)
        let hYaw = Double(pitch)
        let hRoll = Double(roll)
        let headVec = pitchYawRollToGazeVector(
            pitch: hPitch, yaw: hYaw, roll: hRoll
        )

        // --- face_origin_3d from bounding box + FOV ---
        let faceOrigin = estimateFaceOrigin3D(
            boundingBox: face.boundingBox,
            frameWidth: frameWidth,
            frameHeight: frameHeight
        )

        return (headVector: headVec, faceOrigin3D: faceOrigin)
    }

    // MARK: Pitch estimation from landmarks

    /// Estimate pitch (radians) from the nose tip's position relative to
    /// the eye midpoint.
    ///
    /// When looking straight ahead, the nose tip is roughly at the same
    /// Y as the eye midpoint. Looking up moves the nose tip higher;
    /// looking down moves it lower. We map this to a pitch angle.
    private func estimatePitch(face: VNFaceObservation) -> Float {
        guard let landmarks = face.landmarks,
              let nose = landmarks.nose, nose.pointCount > 0,
              let leftEye = landmarks.leftEye, leftEye.pointCount > 0,
              let rightEye = landmarks.rightEye, rightEye.pointCount > 0 else {
            return 0  // neutral
        }

        // Nose tip = last point of the nose region (closest to tip).
        let noseTipY = Float(nose.normalizedPoints[nose.pointCount - 1].y)

        // Eye midpoint Y.
        var eyeSumY: Float = 0
        var eyeCount: Float = 0
        for region in [leftEye, rightEye] {
            for i in 0..<region.pointCount {
                eyeSumY += Float(region.normalizedPoints[i].y)
                eyeCount += 1
            }
        }
        let eyeMidY = eyeSumY / max(1, eyeCount)

        // Ratio: positive when nose is above eyes (looking up).
        let ratio = noseTipY - eyeMidY

        // Map to radians empirically. A ratio of 0.1 ≈ ~15° of pitch.
        let pitchRad = ratio * 2.5  // empirical scaling
        return max(-0.8, min(0.8, pitchRad))
    }

    // MARK: Gaze vector conversion

    /// Port of WebEyeTrack's `pitch_yaw_roll_to_gaze_vector()`.
    /// Converts angles (radians) to a 3D direction unit vector.
    private func pitchYawRollToGazeVector(
        pitch: Double, yaw: Double, roll: Double
    ) -> [Float] {
        let cp = cos(pitch), sp = sin(pitch)
        let cy = cos(yaw), sy = sin(yaw)
        // Forward is -Z. Roll doesn't affect the direction in this model.
        let z = -cp * cy
        let x = cp * sy
        let y = sp
        return [Float(x), Float(y), Float(z)]
    }

    // MARK: Face origin 3D estimation

    /// Estimate the face's 3D position relative to the camera.
    /// Uses the bounding box size + assumed ~60° FOV to derive depth,
    /// then maps the box center to X/Y offsets.
    private func estimateFaceOrigin3D(
        boundingBox: CGRect,
        frameWidth: Int,
        frameHeight: Int
    ) -> [Float] {
        // Face width in pixels.
        let faceWidthPx = Double(boundingBox.width) * Double(frameWidth)

        // Depth via similar triangles: faceWidthPx / focalLength = faceWidthMM / depthMM
        // focalLength ≈ (frameWidth / 2) / tan(30°) for 60° hFOV.
        let focalLength = Double(frameWidth) / 2.0 / tan(60.0 * .pi / 180.0 / 2.0)
        let faceWidthMM = Self.averageFaceWidthCM * 10.0  // → mm
        let depthMM = faceWidthMM * focalLength / max(1, faceWidthPx)

        // X, Y offsets from frame center (camera principal point ≈ center).
        let centerX = Double(boundingBox.midX - 0.5) * Double(frameWidth)
        let centerY = Double(boundingBox.midY - 0.5) * Double(frameHeight)

        // Scale to mm using the same depth/focalLength ratio.
        let xMM = centerX * depthMM / focalLength
        let yMM = centerY * depthMM / focalLength

        return [Float(xMM), Float(yMM), Float(depthMM)]
    }
}
