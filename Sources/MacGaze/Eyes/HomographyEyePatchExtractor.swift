import Foundation
import CoreGraphics
import CoreVideo
import Accelerate

/// Codable model for the JSON output of `extract_landmarks.py`.
public struct LandmarkFrame: Codable {
    public let frame: Int
    public let timestampMs: Int
    public let width: Int
    public let height: Int
    public let landmarks: [[Double]]
    public let faceTransform: [[Double]]?

    public enum CodingKeys: String, CodingKey {
        case frame
        case timestampMs = "timestamp_ms"
        case width, height
        case landmarks
        case faceTransform = "face_transform"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        frame = try c.decode(Int.self, forKey: .frame)
        timestampMs = try c.decode(Int.self, forKey: .timestampMs)
        width = try c.decode(Int.self, forKey: .width)
        height = try c.decode(Int.self, forKey: .height)
        landmarks = try c.decode([[Double]].self, forKey: .landmarks)
        faceTransform = try c.decodeIfPresent([[Double]].self, forKey: .faceTransform)
    }
}

/// Extracts a 128×512 eye patch using the EXACT algorithm from
/// WebEyeTrack's `obtain_eyepatch()` — 4-point perspective homography
/// with MediaPipe landmark indices 103, 150, 332, 379, 4, 151, 195.
///
/// This produces the input format BlazeGaze was trained on.
public struct HomographyEyePatchExtractor {

    static let patchWidth = 512
    static let patchHeight = 128
    static let faceCropSize = 512

    /// Extract eye patch from a camera frame using MediaPipe landmarks.
    ///
    /// - Parameters:
    ///   - pixelBuffer: Camera frame (1280×720 32BGRA).
    ///   - landmarks: 478 normalized landmarks from MediaPipe.
    ///   - frameWidth: Frame width in pixels.
    ///   - frameHeight: Frame height in pixels.
    /// - Returns: 512×128 eye patch as a CVPixelBuffer, or nil.
    public static func extract(
        pixelBuffer: CVPixelBuffer,
        landmarks: [[Double]],
        frameWidth: Int,
        frameHeight: Int
    ) -> CVPixelBuffer? {
        // 468 (base FaceMesh / CoreML) or 478 (attention mesh). Every index
        // used below is < 468, so the base mesh is sufficient.
        guard landmarks.count >= 468 else { return nil }

        // 1. Get the 4 corner landmarks + center (MediaPipe indices).
        let lefttop = pixelPoint(landmarks[103], frameWidth, frameHeight)
        let leftbottom = pixelPoint(landmarks[150], frameWidth, frameHeight)
        let righttop = pixelPoint(landmarks[332], frameWidth, frameHeight)
        let rightbottom = pixelPoint(landmarks[379], frameWidth, frameHeight)
        let center = pixelPoint(landmarks[4], frameWidth, frameHeight)

        // 2. Expand radially from center (padding coefficients from WebEyeTrack).
        let paddingCoefs: (x: CGFloat, y: CGFloat) = (x: 0.4, y: 0.2)
        let src = [
            expand(lefttop, from: center, by: paddingCoefs),
            expand(leftbottom, from: center, by: paddingCoefs),
            expand(rightbottom, from: center, by: paddingCoefs),
            expand(righttop, from: center, by: paddingCoefs),
        ]
        let dst = [
            CGPoint(x: 0, y: 0),
            CGPoint(x: 0, y: CGFloat(faceCropSize)),
            CGPoint(x: CGFloat(faceCropSize), y: CGFloat(faceCropSize)),
            CGPoint(x: CGFloat(faceCropSize), y: 0),
        ]

        // 3. Compute homography (src → dst) and its inverse (dst → src).
        guard let invH = computeInverseHomography(src: src, dst: dst) else { return nil }

        // 4. Warp the frame into a 512×512 square using the inverse homography.
        guard let warped = perspectiveWarp(
            pixelBuffer: pixelBuffer,
            inverseHomography: invH,
            outputSize: faceCropSize
        ) else { return nil }

        // 5. Find the eye band rows (MediaPipe indices 151 and 195 in warped space).
        // Transform landmarks 151 and 195 through the forward homography.
        guard let fwdH = computeHomography(src: src, dst: dst) else { return nil }

        let lm151 = pixelPoint(landmarks[151], frameWidth, frameHeight)
        let lm195 = pixelPoint(landmarks[195], frameWidth, frameHeight)
        let warped151 = applyHomography(fwdH, to: lm151)
        let warped195 = applyHomography(fwdH, to: lm195)

        let topRow = max(0, min(Int(warped151.y), faceCropSize - 1))
        let bottomRow = max(topRow + 1, min(Int(warped195.y), faceCropSize))

        // 6. Crop the eye band from the warped image.
        guard let eyeBand = cropRows(
            from: warped,
            faceCropSize: faceCropSize,
            topRow: topRow,
            bottomRow: bottomRow
        ) else { return nil }

        // 7. Resize to 512×128.
        let result = vImageResize(
            source: eyeBand,
            srcWidth: faceCropSize,
            srcHeight: bottomRow - topRow,
            destWidth: patchWidth,
            destHeight: patchHeight
        )
        return result
    }

    // MARK: Helpers

    private static func pixelPoint(_ lm: [Double], _ w: Int, _ h: Int) -> CGPoint {
        CGPoint(x: CGFloat(lm[0]) * CGFloat(w), y: CGFloat(lm[1]) * CGFloat(h))
    }

    private static func expand(_ pt: CGPoint, from center: CGPoint,
                               by coefs: (x: CGFloat, y: CGFloat)) -> CGPoint {
        let dx = pt.x - center.x
        let dy = pt.y - center.y
        return CGPoint(x: pt.x + coefs.x * dx, y: pt.y + coefs.y * dy)
    }

    // MARK: Homography

    /// Solve for a 3×3 homography matrix mapping `src` → `dst`.
    /// Returns 9 Doubles in row-major order (h11..h33, with h33=1).
    static func computeHomography(src: [CGPoint], dst: [CGPoint]) -> [Double]? {
        guard src.count == 4, dst.count == 4 else { return nil }

        // Build 8×8 system A·h = b.
        var A = [[Double]](repeating: [Double](repeating: 0, count: 8), count: 8)
        var b = [Double](repeating: 0, count: 8)

        for i in 0..<4 {
            let x = Double(src[i].x), y = Double(src[i].y)
            let X = Double(dst[i].x), Y = Double(dst[i].y)
            A[i*2] = [x, y, 1, 0, 0, 0, -X*x, -X*y]
            A[i*2+1] = [0, 0, 0, x, y, 1, -Y*x, -Y*y]
            b[i*2] = X
            b[i*2+1] = Y
        }

        // Gaussian elimination with partial pivoting.
        for col in 0..<8 {
            var maxRow = col
            for row in (col+1)..<8 {
                if abs(A[row][col]) > abs(A[maxRow][col]) { maxRow = row }
            }
            if maxRow != col {
                A.swapAt(col, maxRow)
                b.swapAt(col, maxRow)
            }
            guard abs(A[col][col]) > 1e-10 else { return nil }
            for row in (col+1)..<8 {
                let factor = A[row][col] / A[col][col]
                for j in col..<8 { A[row][j] -= factor * A[col][j] }
                b[row] -= factor * b[col]
            }
        }

        // Back substitution.
        var h = [Double](repeating: 0, count: 8)
        for i in (0..<8).reversed() {
            var sum = b[i]
            for j in (i+1)..<8 { sum -= A[i][j] * h[j] }
            h[i] = sum / A[i][i]
        }

        return [h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1.0]
    }

    /// Compute the inverse homography (dst → src) directly.
    static func computeInverseHomography(src: [CGPoint], dst: [CGPoint]) -> [Double]? {
        computeHomography(src: dst, dst: src)
    }

    /// Apply a 3×3 homography to a 2D point.
    static func applyHomography(_ H: [Double], to pt: CGPoint) -> CGPoint {
        let x = Double(pt.x), y = Double(pt.y)
        let w = H[6]*x + H[7]*y + H[8]
        guard abs(w) > 1e-10 else { return .zero }
        let xp = (H[0]*x + H[1]*y + H[2]) / w
        let yp = (H[3]*x + H[4]*y + H[5]) / w
        return CGPoint(x: xp, y: yp)
    }

    // MARK: Perspective warp

    /// Warp a CVPixelBuffer using a perspective transform (inverse homography).
    /// For each output pixel, find the source pixel and bilinear-sample.
    static func perspectiveWarp(
        pixelBuffer: CVPixelBuffer,
        inverseHomography H: [Double],
        outputSize: Int
    ) -> CVPixelBuffer? {
        let srcW = CVPixelBufferGetWidth(pixelBuffer)
        let srcH = CVPixelBufferGetHeight(pixelBuffer)

        var outputBuffer: CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault, outputSize, outputSize,
                           kCVPixelFormatType_32BGRA, nil, &outputBuffer)
        guard let outputBuffer else { return nil }

        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        CVPixelBufferLockBaseAddress(outputBuffer, [])
        defer {
            CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly)
            CVPixelBufferUnlockBaseAddress(outputBuffer, [])
        }

        guard let srcBase = CVPixelBufferGetBaseAddress(pixelBuffer),
              let dstBase = CVPixelBufferGetBaseAddress(outputBuffer) else { return nil }

        let srcRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let dstRow = CVPixelBufferGetBytesPerRow(outputBuffer)
        let srcPtr = srcBase.assumingMemoryBound(to: UInt8.self)
        let dstPtr = dstBase.assumingMemoryBound(to: UInt8.self)

        for dy in 0..<outputSize {
            for dx in 0..<outputSize {
                let x = Double(dx), y = Double(dy)
                let w = H[6]*x + H[7]*y + H[8]
                guard abs(w) > 1e-10 else { continue }
                let sx = (H[0]*x + H[1]*y + H[2]) / w
                let sy = (H[3]*x + H[4]*y + H[5]) / w

                // Clamp + bilinear sample.
                let sx0 = max(0, min(srcW - 1, Int(sx)))
                let sy0 = max(0, min(srcH - 1, Int(sy)))
                let sx1 = min(srcW - 1, sx0 + 1)
                let sy1 = min(srcH - 1, sy0 + 1)
                let fx = max(0, min(1, sx - Double(sx0)))
                let fy = max(0, min(1, sy - Double(sy0)))

                for c in 0..<4 {
                    let p00 = Double(srcPtr[sy0 * srcRow + sx0 * 4 + c])
                    let p01 = Double(srcPtr[sy0 * srcRow + sx1 * 4 + c])
                    let p10 = Double(srcPtr[sy1 * srcRow + sx0 * 4 + c])
                    let p11 = Double(srcPtr[sy1 * srcRow + sx1 * 4 + c])
                    let val = p00 * (1-fx) * (1-fy) + p01 * fx * (1-fy)
                            + p10 * (1-fx) * fy + p11 * fx * fy
                    dstPtr[dy * dstRow + dx * 4 + c] = UInt8(max(0, min(255, val)))
                }
            }
        }

        return outputBuffer
    }

    // MARK: Row crop + resize

    /// Extract rows [topRow, bottomRow) from a square CVPixelBuffer.
    static func cropRows(from buffer: CVPixelBuffer, faceCropSize: Int,
                         topRow: Int, bottomRow: Int) -> CVPixelBuffer? {
        let height = bottomRow - topRow
        guard height > 0 else { return nil }

        var output: CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault, faceCropSize, height,
                           kCVPixelFormatType_32BGRA, nil, &output)
        guard let output else { return nil }

        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        CVPixelBufferLockBaseAddress(output, [])
        defer {
            CVPixelBufferUnlockBaseAddress(buffer, .readOnly)
            CVPixelBufferUnlockBaseAddress(output, [])
        }

        guard let srcBase = CVPixelBufferGetBaseAddress(buffer),
              let dstBase = CVPixelBufferGetBaseAddress(output) else { return nil }

        let srcRow = CVPixelBufferGetBytesPerRow(buffer)
        let dstRow = CVPixelBufferGetBytesPerRow(output)
        memcpy(dstBase, srcBase.advanced(by: topRow * srcRow), height * srcRow)
        _ = dstRow  // dstRow should equal srcRow for same width

        return output
    }

    /// Resize using vImage.
    static func vImageResize(source: CVPixelBuffer, srcWidth: Int, srcHeight: Int,
                            destWidth: Int, destHeight: Int) -> CVPixelBuffer? {
        var output: CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault, destWidth, destHeight,
                           kCVPixelFormatType_32BGRA, nil, &output)
        guard let output else { return nil }

        CVPixelBufferLockBaseAddress(source, .readOnly)
        CVPixelBufferLockBaseAddress(output, [])
        defer {
            CVPixelBufferUnlockBaseAddress(source, .readOnly)
            CVPixelBufferUnlockBaseAddress(output, [])
        }

        guard let srcBase = CVPixelBufferGetBaseAddress(source),
              let dstBase = CVPixelBufferGetBaseAddress(output) else { return nil }

        var srcBuf = vImage_Buffer(
            data: srcBase, height: vImagePixelCount(srcHeight),
            width: vImagePixelCount(srcWidth),
            rowBytes: CVPixelBufferGetBytesPerRow(source)
        )
        var dstBuf = vImage_Buffer(
            data: dstBase, height: vImagePixelCount(destHeight),
            width: vImagePixelCount(destWidth),
            rowBytes: CVPixelBufferGetBytesPerRow(output)
        )
        vImageScale_ARGB8888(&srcBuf, &dstBuf, nil, vImage_Flags(kvImageEdgeExtend))
        return output
    }
}
