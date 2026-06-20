import Foundation
import CoreVideo
import Accelerate
import Vision
import GazeBridgeCore

/// Extracts a 128×512×3 eye-region crop from a camera frame using vImage
/// (Accelerate framework, pure CPU — no Metal, no CIContext).
///
/// Replaces the CIImage-based version which crashed due to a Metal
/// telemetry race with macOS's Portrait/VFX camera effects system.
public final class EyePatchExtractor {

    public static let patchWidth = 512
    public static let patchHeight = 128

    public init() {}

    /// Extract a 512×128 eye patch from a camera frame using vImage.
    ///
    /// - Parameters:
    ///   - frame: The raw camera pixel buffer (1280×720 32BGRA).
    ///   - faceObservation: Vision's face landmark observation.
    /// - Returns: A `CVPixelBuffer` containing the eye patch in 32BGRA,
    ///            or `nil` if the face or eye landmarks can't be resolved.
    public func extract(
        frame: CVPixelBuffer,
        faceObservation: VNFaceObservation
    ) -> CVPixelBuffer? {
        let frameWidth = CVPixelBufferGetWidth(frame)
        let frameHeight = CVPixelBufferGetHeight(frame)

        // 1. Determine the eye-band region from Vision landmarks.
        let faceBox = faceObservation.boundingBox
        let faceRect = CGRect(
            x: faceBox.minX * CGFloat(frameWidth),
            y: (1 - faceBox.maxY) * CGFloat(frameHeight),
            width: faceBox.width * CGFloat(frameWidth),
            height: faceBox.height * CGFloat(frameHeight)
        )

        var eyeBandTopY: CGFloat
        var eyeBandBottomY: CGFloat

        if let leftEye = faceObservation.landmarks?.leftEye,
           let rightEye = faceObservation.landmarks?.rightEye,
           leftEye.pointCount > 0,
           rightEye.pointCount > 0 {

            var minY: CGFloat = .infinity
            var maxY: CGFloat = -.infinity
            for region in [leftEye, rightEye] {
                for i in 0..<region.pointCount {
                    let p = region.normalizedPoints[i]
                    minY = min(minY, CGFloat(p.y))
                    maxY = max(maxY, CGFloat(p.y))
                }
            }
            let eyeHeight = maxY - minY
            let expandUp: CGFloat = eyeHeight * 1.5
            let expandDown: CGFloat = eyeHeight * 2.0
            eyeBandTopY = (1 - min(CGFloat(1), maxY + expandUp)) * CGFloat(frameHeight)
            eyeBandBottomY = (1 - max(CGFloat(0), minY - expandDown)) * CGFloat(frameHeight)
        } else {
            eyeBandTopY = faceRect.minY
            eyeBandBottomY = faceRect.minY + faceRect.height * 0.35
        }

        eyeBandTopY = max(0, min(CGFloat(frameHeight), eyeBandTopY))
        eyeBandBottomY = max(0, min(CGFloat(frameHeight), eyeBandBottomY))

        let hExpand = faceRect.width * 0.1
        let cropRect = CGRect(
            x: max(0, faceRect.minX - hExpand),
            y: eyeBandTopY,
            width: min(CGFloat(frameWidth), faceRect.width + 2 * hExpand),
            height: eyeBandBottomY - eyeBandTopY
        ).intersection(CGRect(x: 0, y: 0, width: frameWidth, height: frameHeight))

        guard cropRect.width > 10, cropRect.height > 10 else { return nil }

        // 2. Use vImage to crop + resize.  Pure CPU, no Metal.
        return vImageCropAndResize(
            frame: frame,
            cropRect: cropRect,
            destWidth: Self.patchWidth,
            destHeight: Self.patchHeight
        )
    }

    /// Crop + resize a CVPixelBuffer using vImage (Accelerate).
    /// Returns a new 32BGRA pixel buffer.
    private func vImageCropAndResize(
        frame: CVPixelBuffer,
        cropRect: CGRect,
        destWidth: Int,
        destHeight: Int
    ) -> CVPixelBuffer? {
        CVPixelBufferLockBaseAddress(frame, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(frame, .readOnly) }

        guard let baseAddress = CVPixelBufferGetBaseAddress(frame) else { return nil }
        let srcBytesPerRow = CVPixelBufferGetBytesPerRow(frame)

        let cropX = Int(cropRect.origin.x)
        let cropY = Int(cropRect.origin.y)
        let cropW = Int(cropRect.width)
        let cropH = Int(cropRect.height)

        guard cropW > 0, cropH > 0,
              cropX + cropW <= CVPixelBufferGetWidth(frame),
              cropY + cropH <= CVPixelBufferGetHeight(frame) else { return nil }

        // Source vImage buffer: pointer into the crop region of the frame.
        // BGRA format = ARGB8888 in vImage (channel order depends on endianness).
        let cropBase = baseAddress.advanced(by: cropY * srcBytesPerRow + cropX * 4)
        var srcBuffer = vImage_Buffer(
            data: cropBase,
            height: vImagePixelCount(cropH),
            width: vImagePixelCount(cropW),
            rowBytes: srcBytesPerRow
        )

        // Destination buffer.
        var outputBuffer: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
        ]
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            destWidth, destHeight,
            kCVPixelFormatType_32BGRA,
            attrs as CFDictionary,
            &outputBuffer
        )
        guard status == kCVReturnSuccess, let outputBuffer else { return nil }

        CVPixelBufferLockBaseAddress(outputBuffer, [])
        defer { CVPixelBufferUnlockBaseAddress(outputBuffer, []) }

        guard let destBase = CVPixelBufferGetBaseAddress(outputBuffer) else { return nil }
        let destBytesPerRow = CVPixelBufferGetBytesPerRow(outputBuffer)
        var destBuffer = vImage_Buffer(
            data: destBase,
            height: vImagePixelCount(destHeight),
            width: vImagePixelCount(destWidth),
            rowBytes: destBytesPerRow
        )

        // Scale BGRA8888.  vImage uses ARGB channel order on little-endian
        // which matches BGRA byte order.
        let scaleError = vImageScale_ARGB8888(&srcBuffer, &destBuffer, nil, vImage_Flags(kvImageEdgeExtend))
        guard scaleError == kvImageNoError else { return nil }

        return outputBuffer
    }
}
