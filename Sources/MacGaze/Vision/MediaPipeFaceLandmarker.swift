import Foundation
import CoreVideo
import CMediaPipe

/// Native MediaPipe Face Landmarker for macOS.
///
/// Wraps Google's `libmediapipe.dylib` via a C bridging layer (CMediaPipe)
/// that handles struct alignment correctly. Provides real-time 478-point
/// face landmark detection + facial transformation matrix.
public final class MediaPipeFaceLandmarker {

    public enum MPError: Error, LocalizedError {
        case dylibNotFound(String)
        case createFailed(String)
        case detectFailed(String)

        public var errorDescription: String? {
            switch self {
            case .dylibNotFound(let p):  return "libmediapipe.dylib not found: \(p)"
            case .createFailed(let m):    return "FaceLandmarker create failed: \(m)"
            case .detectFailed(let m):    return "Detect failed: \(m)"
            }
        }
    }

    public struct Result: Sendable {
        public let landmarks: [[Double]]
        public let faceTransform: [[Double]]?
    }

    private var handle: MPFaceLandmarkerHandle?

    /// Initialize with a model asset path (the `.task` file).
    public init(modelPath: String) throws {
        let absPath: String
        if modelPath.hasPrefix("/") {
            absPath = modelPath
        } else {
            absPath = FileManager.default.currentDirectoryPath + "/" + modelPath
        }

        guard FileManager.default.fileExists(atPath: absPath) else {
            throw MPError.createFailed("model not found: \(absPath)")
        }

        var status: Int32 = 0
        let h = absPath.withCString { cPath in
            cmp_face_landmarker_create(cPath, &status)
        }

        guard status == 0, let h else {
            throw MPError.createFailed("status=\(status) (model: \(absPath))")
        }
        self.handle = h
    }

    deinit {
        if let handle { cmp_face_landmarker_close(handle) }
    }

    /// Detect face landmarks in a CVPixelBuffer (video mode).
    public func detect(pixelBuffer: CVPixelBuffer, timestampMs: Int64) throws -> Result {
        guard let handle else { throw MPError.detectFailed("no handle") }

        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)

        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            throw MPError.detectFailed("can't lock pixel buffer")
        }
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let srcPtr = baseAddress.assumingMemoryBound(to: UInt8.self)

        // Convert BGRA → contiguous RGB (what MediaPipe expects).
        let rgbSize = width * height * 3
        let rgbBuffer = UnsafeMutablePointer<UInt8>.allocate(capacity: rgbSize)
        defer { rgbBuffer.deallocate() }

        for y in 0..<height {
            for x in 0..<width {
                let srcOffset = y * bytesPerRow + x * 4
                let dstOffset = (y * width + x) * 3
                rgbBuffer[dstOffset + 0] = srcPtr[srcOffset + 2]  // R
                rgbBuffer[dstOffset + 1] = srcPtr[srcOffset + 1]  // G
                rgbBuffer[dstOffset + 2] = srcPtr[srcOffset + 0]  // B
            }
        }

        // Call the C bridge.
        var cResult = MPFaceLandmarkerResult()
        let status = rgbBuffer.withMemoryRebound(to: UInt8.self, capacity: rgbSize) { rgbPtr in
            cmp_face_landmarker_detect_video(
                handle,
                UnsafePointer(rgbPtr),
                Int32(width), Int32(height),
                timestampMs,
                &cResult
            )
        }

        guard status == 0, cResult.success != 0 else {
            throw MPError.detectFailed("status=\(status)")
        }

        defer { cmp_face_landmarker_free_result(&cResult) }

        // Extract landmarks.
        var landmarks: [[Double]] = []
        let count = Int(cResult.landmark_count)
        if count > 0, let lmPtr = cResult.landmarks {
            for i in 0..<count {
                let x = Double(lmPtr[i * 3 + 0])
                let y = Double(lmPtr[i * 3 + 1])
                let z = Double(lmPtr[i * 3 + 2])
                landmarks.append([x, y, z])
            }
        }

        // Extract transform matrix (4×4 column-major → row-major).
        var faceTransform: [[Double]]? = nil
        if cResult.has_transform != 0 {
            var matrix: [[Double]] = Array(repeating: Array(repeating: 0.0, count: 4), count: 4)
            let t = cResult.transform
            for col in 0..<4 {
                for row in 0..<4 {
                    // Column-major: transform is a fixed array [16]
                    // Access via withUnsafeBufferPointer since Swift imports
                    // C fixed arrays as tuples.
                    let idx = col * 4 + row
                    let val: Float = withUnsafePointer(to: t) { ptr in
                        ptr.withMemoryRebound(to: Float.self, capacity: 16) { floatPtr in
                            floatPtr[idx]
                        }
                    }
                    matrix[row][col] = Double(val)
                }
            }
            faceTransform = matrix
        }

        return Result(landmarks: landmarks, faceTransform: faceTransform)
    }
}
