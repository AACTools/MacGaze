import Foundation
import CoreML
import CoreVideo
import Vision
import GazeBridgeCore

/// Wraps the compiled BlazeGaze CoreML model and runs gaze prediction.
///
/// Takes three inputs:
/// 1. **image**: 128×512×3 eye-region crop (produced by `EyePatchExtractor`).
/// 2. **head_vector**: 3D head-direction unit vector (from head pose).
/// 3. **face_origin_3d**: 3D face origin position (from head pose).
///
/// Produces one output:
/// - **gaze_output**: normalised (x, y) screen point, [0, 1].
///
/// **Phase 2.4 placeholder:** `head_vector` and `face_origin_3d` default
/// to neutral values (looking straight ahead, centred).  Phase 2.3 will
/// replace these with real EPnP-solved head pose.  Zero-shot accuracy
/// with neutral head pose will be poor but sufficient to prove the
/// pipeline end-to-end.
public final class BlazeGazeRunner {

    /// Errors surfaced by the runner.
    public enum Error: Swift.Error, LocalizedError {
        case modelNotFound(String)
        case modelLoadFailed(String)
        case predictionFailed(String)

        public var errorDescription: String? {
            switch self {
            case .modelNotFound(let p):   return "BlazeGaze model not found at \(p)"
            case .modelLoadFailed(let m):  return "Failed to load BlazeGaze: \(m)"
            case .predictionFailed(let m): return "BlazeGaze prediction failed: \(m)"
            }
        }
    }

    private let model: MLModel

    /// Create the neutral head_vector MLMultiArray. Looking straight ahead.
    private static func makeNeutralHeadVector() -> MLMultiArray {
        let arr = try! MLMultiArray(shape: [1, 3], dataType: .float32)
        arr[0] = 0.0
        arr[1] = 0.0
        arr[2] = -1.0
        return arr
    }

    /// Create the neutral face_origin_3d MLMultiArray. Face at ~50 cm.
    private static func makeNeutralFaceOrigin() -> MLMultiArray {
        let arr = try! MLMultiArray(shape: [1, 3], dataType: .float32)
        arr[0] = 0.0
        arr[1] = 0.0
        arr[2] = 500.0
        return arr
    }

    /// Load the compiled model from a `.mlmodelc` URL.
    public init(modelURL: URL) throws {
        let config = MLModelConfiguration()
        config.computeUnits = .all  // let CoreML pick: ANE > GPU > CPU
        do {
            self.model = try MLModel(contentsOf: modelURL, configuration: config)
        } catch {
            throw Error.modelLoadFailed(error.localizedDescription)
        }
    }

    /// Convenience: try to locate the model. The model is NOT embedded in
    /// the Swift package (that causes Metal shader auto-registration → crash
    /// on some M1 machines). It's loaded at runtime from development paths.
    public convenience init() throws {
        // Development paths — model lives in Sources/MacGaze/Gaze/ but is
        // excluded from the package via `exclude` in Package.swift.
        let devPaths = [
            "Sources/MacGaze/Gaze/blazegaze.mlmodelc",
            FileManager.default.currentDirectoryPath + "/Sources/MacGaze/Gaze/blazegaze.mlmodelc",
            FileManager.default.currentDirectoryPath + "/macgaze/Sources/MacGaze/Gaze/blazegaze.mlmodelc",
        ]
        for path in devPaths {
            if FileManager.default.fileExists(atPath: path) {
                try self.init(modelURL: URL(fileURLWithPath: path))
                return
            }
        }
        // Main bundle (if shipped as a resource by the host app).
        if let url = Bundle.main.url(forResource: "blazegaze", withExtension: "mlmodelc") {
            try self.init(modelURL: url)
            return
        }
        throw Error.modelNotFound(
            "blazegaze.mlmodelc — run scripts/setup.sh first"
        )
    }

    /// Run gaze prediction.
    ///
    /// - Parameters:
    ///   - eyePatch: 128×512 CVPixelBuffer (32BGRA) from EyePatchExtractor.
    ///   - headVector: 3D head direction unit vector (use `.neutralHeadVector`
    ///     until Phase 2.3 is done).
    ///   - faceOrigin3D: 3D face origin position (use `.neutralFaceOrigin3D`
    ///     until Phase 2.3 is done).
    /// - Returns: predicted normalised gaze (x, y) in [0, 1], or nil if
    ///   prediction failed.
    public func predict(
        eyePatch: CVPixelBuffer,
        headVector: MLMultiArray? = nil,
        faceOrigin3D: MLMultiArray? = nil
    ) -> CGPoint? {
        let hv = headVector ?? Self.makeNeutralHeadVector()
        let fo = faceOrigin3D ?? Self.makeNeutralFaceOrigin()

        // Convert the eye-patch CVPixelBuffer to the MLMultiArray the model
        // expects: shape (1, 128, 512, 3), float32, RGB, normalised [0,1].
        // The model was trained on images normalised to [0,1] via /255.
        guard let imageArray = Self.pixelBufferToMLMultiArray(eyePatch) else {
            return nil
        }

        let inputDict: [String: MLFeatureValue] = [
            "image": MLFeatureValue(multiArray: imageArray),
            "head_vector": MLFeatureValue(multiArray: hv),
            "face_origin_3d": MLFeatureValue(multiArray: fo),
        ]

        let inputProvider: MLFeatureProvider
        do {
            inputProvider = try MLDictionaryFeatureProvider(dictionary: inputDict)
        } catch {
            return nil
        }

        let outputProvider: MLFeatureProvider
        do {
            outputProvider = try model.prediction(from: inputProvider)
        } catch {
            return nil
        }

        // Find the gaze output.
        let outputName = outputProvider.featureNames.first ?? "gaze_output"
        guard let outputArray = outputProvider.featureValue(for: outputName)?.multiArrayValue else {
            return nil
        }

        let x = outputArray[0].doubleValue
        let y = outputArray[1].doubleValue
        return CGPoint(x: x, y: y)
    }

    /// Convert a 32BGRA CVPixelBuffer to an MLMultiArray with shape
    /// (1, H, W, 3), float32, RGB channel order, normalised to [0, 1].
    ///
    /// Uses raw pointer access to the MLMultiArray's contiguous data
    /// buffer instead of NSNumber subscripts — ~10× faster (was 90ms,
    /// now <10ms for 128×512).
    static func pixelBufferToMLMultiArray(_ buffer: CVPixelBuffer) -> MLMultiArray? {
        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)

        guard let array = try? MLMultiArray(
            shape: [1, NSNumber(value: height), NSNumber(value: width), 3],
            dataType: .float32
        ) else { return nil }

        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }

        guard let baseAddress = CVPixelBufferGetBaseAddress(buffer) else { return nil }
        let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)

        // Get raw pointer to the MLMultiArray's contiguous float32 data.
        // This avoids per-pixel NSNumber allocation (the old bottleneck).
        let dataPtr = array.dataPointer.assumingMemoryBound(to: Float.self)
        let srcPtr = baseAddress.assumingMemoryBound(to: UInt8.self)

        // BGRA byte order on little-endian macOS → convert to RGB float32.
        let w = width
        for y in 0..<height {
            let rowStart = y * bytesPerRow
            let dstRow = y * w * 3
            for x in 0..<width {
                let src = rowStart + x * 4
                let dst = dstRow + x * 3
                // BGRA → RGB, normalize to [0, 1]
                dataPtr[dst + 0] = Float(srcPtr[src + 2]) / 255.0  // R
                dataPtr[dst + 1] = Float(srcPtr[src + 1]) / 255.0  // G
                dataPtr[dst + 2] = Float(srcPtr[src + 0]) / 255.0  // B
            }
        }

        return array
    }
}
