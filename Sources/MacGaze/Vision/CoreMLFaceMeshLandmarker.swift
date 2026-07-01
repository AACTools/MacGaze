import Foundation
import CoreML
import CoreGraphics

/// Runs the base-468 MediaPipe FaceMesh via CoreML (ANE) and returns landmarks
/// in the same full-image normalized space MediaPipe produced, so the existing
/// `HomographyEyePatchExtractor` works unchanged.
///
/// Convention validated in `Tools/Conversion/facemesh_parity.py` (2026-07-01):
///   • input  `input_1`  : 1×192×192×3 Float32, RGB, NHWC, ÷255 → [0,1]
///   • output `conv2d_20`: 468×(x,y,z) interleaved, x,y in crop-pixel space 0–192
///   • output `conv2d_30`: face-presence logit (apply sigmoid)
///   • crop: square centred on the landmark bbox, side = max(w,h) × 1.5
///
/// All image/landmark coordinates here use a **top-left origin (y-down)** to
/// match MediaPipe's normalized output.
public final class CoreMLFaceMeshLandmarker {

    public struct Result {
        /// N×3, full-image normalized [0,1], top-left origin. z is relative depth.
        public let landmarks: [[Double]]
        /// sigmoid(conv2d_30) — face presence / tracking confidence.
        public let score: Double
    }

    private let model: MLModel

    public init(modelURL: URL) throws {
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .all
        model = try MLModel(contentsOf: modelURL, configuration: cfg)
    }

    /// Run FaceMesh on a square pixel crop (top-left origin) of `image`.
    public func run(image: CGImage, cropRect: CGRect) -> Result? {
        guard let input = Self.makeInput(image: image, cropRect: cropRect),
              let out = try? model.prediction(from: input),
              let lm = out.featureValue(for: "conv2d_20")?.multiArrayValue,
              let sc = out.featureValue(for: "conv2d_30")?.multiArrayValue
        else { return nil }

        let width = Double(image.width)
        let height = Double(image.height)
        let originX = Double(cropRect.origin.x)
        let originY = Double(cropRect.origin.y)
        let side = Double(cropRect.width)   // square crop

        let count = lm.count / 3
        var landmarks = [[Double]]()
        landmarks.reserveCapacity(count)
        for i in 0..<count {
            let cropX = lm[i * 3 + 0].doubleValue / 192.0   // crop-normalized
            let cropY = lm[i * 3 + 1].doubleValue / 192.0
            let z = lm[i * 3 + 2].doubleValue
            let fullX = (originX + cropX * side) / width
            let fullY = (originY + cropY * side) / height
            landmarks.append([fullX, fullY, z])
        }
        let score = 1.0 / (1.0 + exp(-sc[0].doubleValue))
        return Result(landmarks: landmarks, score: score)
    }

    /// Render the square crop into a 192×192 RGB [0,1] MultiArray for `input_1`.
    private static func makeInput(image: CGImage, cropRect: CGRect) -> MLFeatureProvider? {
        guard let sub = image.cropping(to: cropRect),
              let arr = try? MLMultiArray(shape: [1, 192, 192, 3], dataType: .float32),
              let space = CGColorSpace(name: CGColorSpace.sRGB)
        else { return nil }

        let side = 192
        let bytesPerRow = side * 4
        var buffer = [UInt8](repeating: 0, count: side * bytesPerRow)
        let bitmap = CGImageAlphaInfo.premultipliedLast.rawValue
        let drawn: Bool = buffer.withUnsafeMutableBytes { raw in
            guard let ctx = CGContext(data: raw.baseAddress, width: side, height: side,
                                      bitsPerComponent: 8, bytesPerRow: bytesPerRow,
                                      space: space, bitmapInfo: bitmap) else { return false }
            ctx.interpolationQuality = .low
            ctx.draw(sub, in: CGRect(x: 0, y: 0, width: side, height: side))
            return true
        }
        guard drawn else { return nil }

        let ptr = arr.dataPointer.assumingMemoryBound(to: Float.self)
        for y in 0..<side {
            for x in 0..<side {
                let src = y * bytesPerRow + x * 4
                let dst = (y * side + x) * 3
                ptr[dst + 0] = Float(buffer[src + 0]) / 255.0   // R
                ptr[dst + 1] = Float(buffer[src + 1]) / 255.0   // G
                ptr[dst + 2] = Float(buffer[src + 2]) / 255.0   // B
            }
        }
        return try? MLDictionaryFeatureProvider(
            dictionary: ["input_1": MLFeatureValue(multiArray: arr)])
    }
}
