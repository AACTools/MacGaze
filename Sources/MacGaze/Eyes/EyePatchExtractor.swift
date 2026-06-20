import Foundation
import CoreVideo
import CoreImage
import CoreImage.CIFilterBuiltins
import Vision
import GazeBridgeCore

/// Extracts a 128×512×3 eye-region crop from a camera frame, suitable as
/// input to the BlazeGaze CoreML model.
///
/// The original WebEyeTrack implementation (`obtain_eyepatch()` in
/// `model_based.py`) uses MediaPipe's 468-point landmarks to compute a
/// perspective homography that maps the face to a canonical square, then
/// crops the eye band. We approximate this with Apple Vision landmarks:
///
/// 1. Use the face bounding box from Vision to define the overall face region.
/// 2. Use the left/right eye landmark regions to find the vertical extent
///    of the eye band.
/// 3. Crop the eye band from the camera frame.
/// 4. Resize to 512×128 (width × height) via Core Image.
/// 5. Normalise to float32 [0, 1].
///
/// This is **not** perspective-corrected like the Python version. We rely
/// on BlazeGaze's internal head_vector + face_origin_3d inputs to compensate
/// for head pose. If accuracy suffers, Phase 2.3 will add the full
/// homography-based crop.
public final class EyePatchExtractor {

    /// Target eye-patch dimensions expected by BlazeGaze.
    /// Order is (width, height) to match Core Image convention.
    public static let patchWidth = 512
    public static let patchHeight = 128

    /// Shared CIContext — creating one per call is expensive + can crash
    /// Metal's command queue labeller under load.  One instance for the
    /// lifetime of the extractor.
    private let ciContext = CIContext()

    public init() {}

    /// Extract a 512×128 eye patch from a camera frame.
    ///
    /// - Parameters:
    ///   - frame: The raw camera pixel buffer (1280×720 32BGRA).
    ///   - faceObservation: Vision's face landmark observation.
    /// - Returns: A `CVPixelBuffer` containing the normalised eye patch,
    ///            or `nil` if the face or eye landmarks can't be resolved.
    public func extract(
        frame: CVPixelBuffer,
        faceObservation: VNFaceObservation
    ) -> CVPixelBuffer? {
        let frameWidth = CVPixelBufferGetWidth(frame)
        let frameHeight = CVPixelBufferGetHeight(frame)

        // 1. Convert the face bounding box from Vision's normalised
        //    (bottom-left origin) to pixel coordinates (top-left origin).
        let faceBox = faceObservation.boundingBox  // normalised, bottom-left
        let faceRect = CGRect(
            x: faceBox.minX * CGFloat(frameWidth),
            y: (1 - faceBox.maxY) * CGFloat(frameHeight),  // flip Y
            width: faceBox.width * CGFloat(frameWidth),
            height: faceBox.height * CGFloat(frameHeight)
        )

        // 2. Determine the eye-band vertical extent from eye landmark
        //    regions. If absent, fall back to the top 35% of the face box.
        var eyeBandTopY: CGFloat
        var eyeBandBottomY: CGFloat

        if let leftEye = faceObservation.landmarks?.leftEye,
           let rightEye = faceObservation.landmarks?.rightEye,
           leftEye.pointCount > 0,
           rightEye.pointCount > 0 {

            // Find the vertical extent of both eye regions combined.
            // Vision reports points in normalised image coordinates with
            // origin bottom-left.
            var minY: CGFloat = .infinity
            var maxY: CGFloat = -.infinity
            for region in [leftEye, rightEye] {
                for i in 0..<region.pointCount {
                    let p = region.normalizedPoints[i]
                    minY = min(minY, CGFloat(p.y))
                    maxY = max(maxY, CGFloat(p.y))
                }
            }
            // Expand vertically: include eyebrows above and cheek below.
            let eyeHeight = maxY - minY
            let expandUp: CGFloat = eyeHeight * 1.5     // eyebrows + forehead
            let expandDown: CGFloat = eyeHeight * 2.0   // nose bridge + cheek

            // Convert to pixel coordinates (flip Y).
            eyeBandTopY = (1 - min(CGFloat(1), maxY + expandUp)) * CGFloat(frameHeight)
            eyeBandBottomY = (1 - max(CGFloat(0), minY - expandDown)) * CGFloat(frameHeight)
        } else {
            // Fallback: top 35% of the face bounding box.
            eyeBandTopY = faceRect.minY
            eyeBandBottomY = faceRect.minY + faceRect.height * 0.35
        }

        // Clamp to frame bounds.
        eyeBandTopY = max(0, min(CGFloat(frameHeight), eyeBandTopY))
        eyeBandBottomY = max(0, min(CGFloat(frameHeight), eyeBandBottomY))

        // 3. Build the crop rectangle: full face width + eye-band height.
        //    Expand horizontally by 10% on each side to capture the full
        //    eye region including temple area.
        let hExpand = faceRect.width * 0.1
        let cropRect = CGRect(
            x: max(0, faceRect.minX - hExpand),
            y: eyeBandTopY,
            width: min(CGFloat(frameWidth), faceRect.width + 2 * hExpand),
            height: eyeBandBottomY - eyeBandTopY
        ).intersection(CGRect(x: 0, y: 0, width: frameWidth, height: frameHeight))

        guard cropRect.width > 10, cropRect.height > 10 else { return nil }

        // 4. Crop + resize via Core Image.
        let ciImage = CIImage(cvPixelBuffer: frame, options: nil)
        let cropped = ciImage.cropped(to: cropRect)

        // Create a scale filter to resize to the target dimensions.
        let scaleX = CGFloat(Self.patchWidth) / cropRect.width
        let scaleY = CGFloat(Self.patchHeight) / cropRect.height
        let resized = cropped.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))

        // 5. Render to a new pixel buffer.
        var outputBuffer: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
        ]
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            Self.patchWidth,
            Self.patchHeight,
            kCVPixelFormatType_32BGRA,
            attrs as CFDictionary,
            &outputBuffer
        )
        guard status == kCVReturnSuccess, let outputBuffer else { return nil }

        ciContext.render(resized, to: outputBuffer)
        return outputBuffer
    }
}
