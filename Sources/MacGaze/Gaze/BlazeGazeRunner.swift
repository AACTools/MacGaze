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

    /// Convenience: try to locate the model in the main bundle, then
    /// fall back to common development paths.
    public convenience init() throws {
        // 1. Try the main bundle.
        if let url = Bundle.main.url(forResource: "blazegaze", withExtension: "mlmodelc") {
            try self.init(modelURL: url)
            return
        }
        // 2. Try the MacGaze package itself (for library consumers).
        if let url = Bundle(for: BlazeGazeRunner.self)
            .url(forResource: "blazegaze", withExtension: "mlmodelc") {
            try self.init(modelURL: url)
            return
        }
        // 3. Development fallback: look for the model relative to the
        //    workspace root.
        let devPaths = [
            // SwiftPM build output
            FileManager.default.currentDirectoryPath +
                "/Sources/MacGaze/Gaze/blazegaze.mlmodelc",
            // DerivedData
            FileManager.default.currentDirectoryPath +
                "/macgaze/Sources/MacGaze/Gaze/blazegaze.mlmodelc",
        ]
        for path in devPaths {
            if FileManager.default.fileExists(atPath: path) {
                try self.init(modelURL: URL(fileURLWithPath: path))
                return
            }
        }
        throw Error.modelNotFound(
            "blazegaze.mlmodelc — run Tools/Conversion/convert_blazegaze.py first"
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

        let imageFeature = MLFeatureValue(pixelBuffer: eyePatch)

        let inputDict: [String: MLFeatureValue] = [
            "image": imageFeature,
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

        // Find the gaze output. The exact name depends on the conversion;
        // try common names then fall back to the first output feature.
        let outputName = outputProvider.featureNames.first ?? "gaze_output"
        guard let outputArray = outputProvider.featureValue(for: outputName)?.multiArrayValue else {
            return nil
        }

        // Output shape is [1, 2] → (x, y) normalised.
        let x = outputArray[0].doubleValue
        let y = outputArray[1].doubleValue
        return CGPoint(x: x, y: y)
    }
}
