import Foundation
import CoreML
import CoreGraphics
import CoreImage
import CoreVideo
import Vision

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
    private var lastLandmarks: [[Double]]?
    private static let ciContext = CIContext(options: [.useSoftwareRenderer: false])

    public init(modelURL: URL) throws {
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .all
        model = try MLModel(contentsOf: modelURL, configuration: cfg)
    }

    // MARK: High-level detect (Vision bootstrap + landmark tracking)

    /// Detect landmarks for a camera frame. Same convention as
    /// `MediaPipeFaceLandmarker.detect` so it drops into the pipeline.
    public func detect(pixelBuffer: CVPixelBuffer) -> Result? {
        guard let image = Self.cgImage(from: pixelBuffer) else { return nil }
        return detect(image: image)
    }

    /// Track from the previous frame's landmarks when possible; otherwise
    /// bootstrap from a Vision face rect then re-crop to the landmark bbox
    /// (2-pass) — MediaPipe's detect-then-track design.
    public func detect(image: CGImage) -> Result? {
        let w = image.width, h = image.height

        if let last = lastLandmarks,
           let crop = Self.trackCrop(from: last, width: w, height: h),
           let tracked = run(image: image, cropRect: crop), tracked.score >= 0.5 {
            lastLandmarks = tracked.landmarks
            return tracked
        }

        guard let visionRect = Self.visionFaceRect(image) else {
            lastLandmarks = nil
            return nil
        }
        let crop1 = Self.squareCrop(cx: visionRect.midX, cy: visionRect.midY,
                                    side: max(visionRect.width, visionRect.height) * 1.4,
                                    width: w, height: h)
        guard let pass1 = run(image: image, cropRect: crop1),
              let crop2 = Self.trackCrop(from: pass1.landmarks, width: w, height: h),
              let pass2 = run(image: image, cropRect: crop2) else {
            lastLandmarks = nil
            return nil
        }
        lastLandmarks = pass2.landmarks
        return pass2
    }

    // MARK: Crop / image helpers

    /// Square crop = landmark bbox × 1.5 (the validated framing).
    private static func trackCrop(from landmarks: [[Double]], width: Int, height: Int) -> CGRect? {
        guard !landmarks.isEmpty else { return nil }
        var minX = 1.0, minY = 1.0, maxX = 0.0, maxY = 0.0
        for p in landmarks {
            minX = min(minX, p[0]); maxX = max(maxX, p[0])
            minY = min(minY, p[1]); maxY = max(maxY, p[1])
        }
        let x0 = minX * Double(width), x1 = maxX * Double(width)
        let y0 = minY * Double(height), y1 = maxY * Double(height)
        let side = max(x1 - x0, y1 - y0) * 1.5
        return squareCrop(cx: (x0 + x1) / 2, cy: (y0 + y1) / 2, side: side,
                          width: width, height: height)
    }

    private static func squareCrop(cx: Double, cy: Double, side: Double,
                                   width: Int, height: Int) -> CGRect {
        var x0 = max(0, cx - side / 2)
        var y0 = max(0, cy - side / 2)
        let s = min(side, Double(width) - x0, Double(height) - y0)
        x0 = min(x0, Double(width) - s)
        y0 = min(y0, Double(height) - s)
        return CGRect(x: x0, y: y0, width: s, height: s)
    }

    private static func visionFaceRect(_ image: CGImage) -> CGRect? {
        let request = VNDetectFaceRectanglesRequest()
        try? VNImageRequestHandler(cgImage: image, options: [:]).perform([request])
        guard let face = request.results?.first else { return nil }
        let bb = face.boundingBox   // normalized, bottom-left origin
        let x = bb.origin.x * Double(image.width)
        let y = (1.0 - bb.origin.y - bb.height) * Double(image.height)  // → top-left
        return CGRect(x: x, y: y,
                      width: bb.width * Double(image.width),
                      height: bb.height * Double(image.height))
    }

    private static func cgImage(from pixelBuffer: CVPixelBuffer) -> CGImage? {
        let ci = CIImage(cvPixelBuffer: pixelBuffer)
        return ciContext.createCGImage(ci, from: ci.extent)
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
            let cropZ = lm[i * 3 + 2].doubleValue / 192.0
            let fullX = (originX + cropX * side) / width
            let fullY = (originY + cropY * side) / height
            // z is relative depth (no origin); keep it on the same normalized
            // scale as x (÷ image width), matching MediaPipe's convention.
            let fullZ = cropZ * side / width
            landmarks.append([fullX, fullY, fullZ])
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
