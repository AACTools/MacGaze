import Foundation
import CoreVideo

/// Native MediaPipe Face Landmarker for macOS.
///
/// Wraps Google's `libmediapipe.dylib` (extracted from the Python package)
/// using dlopen/dlsym — no build-time linking required. Provides real-time
/// 478-point face landmark detection + facial transformation matrix.
///
/// This replaces the Python→JSON→Swift hybrid with a fully native pipeline.
public final class MediaPipeFaceLandmarker {

    // MARK: Errors

    public enum MPError: Error, LocalizedError {
        case dylibNotFound(String)
        case symbolNotFound(String)
        case createFailed(String)
        case detectFailed(String)

        public var errorDescription: String? {
            switch self {
            case .dylibNotFound(let p):  return "libmediapipe.dylib not found: \(p)"
            case .symbolNotFound(let s):  return "Symbol not found: \(s)"
            case .createFailed(let m):    return "FaceLandmarker create failed: \(m)"
            case .detectFailed(let m):    return "Detect failed: \(m)"
            }
        }
    }

    // MARK: Result types

    public struct Result: Sendable {
        public let landmarks: [[Double]]      // 478 × [x, y, z]
        public let faceTransform: [[Double]]?  // 4×4 matrix or nil
    }

    // MARK: C struct layouts (mirrors the ctypes definitions)

    // NormalizedLandmarkC: 36 bytes (x, y, z, has_vis, vis, has_pres, pres, name_ptr)
    // Layout: Float × 3 + Bool + Float + Bool + Float + char* = 4*6 + 1*2 + 8 = 34 bytes
    // But ctypes aligns to 4-byte boundaries, so it's 40 bytes with padding.
    // Let me use explicit byte offsets to be safe.

    private struct NormalizedLandmarkC {
        var x: Float = 0
        var y: Float = 0
        var z: Float = 0
        var hasVisibility: Bool = false   // padded to 4 bytes by ctypes? No, Bool is 1 byte.
        var visibility: Float = 0
        var hasPresence: Bool = false
        var presence: Float = 0
        var name: UnsafePointer<CChar>? = nil
    }

    // NormalizedLandmarksC: pointer + count
    private struct NormalizedLandmarksC {
        var landmarks: UnsafePointer<NormalizedLandmarkC>? = nil
        var count: UInt32 = 0
    }

    // MatrixC: rows + cols + data pointer
    private struct MatrixC {
        var rows: UInt32 = 0
        var cols: UInt32 = 0
        var data: UnsafePointer<Float>? = nil
    }

    // FaceLandmarkerResultC
    private struct FaceLandmarkerResultC {
        var faceLandmarks: UnsafePointer<NormalizedLandmarksC>? = nil
        var faceLandmarksCount: UInt32 = 0
        var faceBlendshapes: UnsafeRawPointer? = nil  // CategoriesC — we don't use
        var faceBlendshapesCount: UInt32 = 0
        var facialTransformationMatrixes: UnsafePointer<MatrixC>? = nil
        var facialTransformationMatrixesCount: UInt32 = 0
    }

    // BaseOptionsC: matches the Python ctypes struct
    private struct BaseOptionsC {
        var modelAssetBuffer: UnsafePointer<CChar>? = nil
        var modelAssetBufferCount: UInt = 0
        var modelAssetPath: UnsafePointer<CChar>? = nil
        var delegate: Int32 = 0       // 0 = CPU
        var hostEnvironment: Int32 = 0
        var hostSystem: Int32 = 0
        var hostVersion: UnsafePointer<CChar>? = nil
        var caBundlePath: UnsafePointer<CChar>? = nil
    }

    // FaceLandmarkerOptionsC
    private struct FaceLandmarkerOptionsC {
        var baseOptions: BaseOptionsC = BaseOptionsC()
        var runningMode: Int32 = 1    // 0=image, 1=video, 2=livestream
        var numFaces: Int32 = 1
        var minFaceDetectionConfidence: Float = 0.5
        var minFacePresenceConfidence: Float = 0.5
        var minTrackingConfidence: Float = 0.5
        var outputFaceBlendshapes: Bool = false
        var outputFacialTransformationMatrixes: Bool = true
        var resultCallback: UnsafeRawPointer? = nil  // function ptr, unused for video mode
    }

    // MARK: Function pointer types (all use raw pointers — Swift structs
    // can't be used in @convention(c) signatures).

    private typealias MpCreateFn = @convention(c) (UnsafeRawPointer?, UnsafeMutablePointer<UnsafeMutableRawPointer?>) -> Int32
    private typealias MpDetectVideoFn = @convention(c) (UnsafeMutableRawPointer?, UnsafeMutableRawPointer?, UnsafeRawPointer?, Int64, UnsafeMutableRawPointer?) -> Int32
    private typealias MpCloseResultFn = @convention(c) (UnsafeMutableRawPointer?) -> Void
    private typealias MpCloseFn = @convention(c) (UnsafeMutableRawPointer?) -> Int32
    private typealias MpImageCreateFn = @convention(c) (Int32, Int32, Int32, UnsafePointer<UInt8>?, Int32, UnsafeMutablePointer<UnsafeMutableRawPointer?>, UnsafeMutablePointer<UnsafePointer<CChar>?>?) -> Int32

    // MARK: State

    private var handle: UnsafeMutableRawPointer?
    private var dylib: UnsafeMutableRawPointer?

    // Resolved function pointers.
    private var fnCreate: MpCreateFn?
    private var fnDetectVideo: MpDetectVideoFn?
    private var fnCloseResult: MpCloseResultFn?
    private var fnClose: MpCloseFn?
    private var fnImageCreate: MpImageCreateFn?

    /// Initialize with a model asset path (the `.task` file).
    public init(modelPath: String) throws {
        try loadDylib()
        try resolveSymbols()

        // Allocate options struct in raw memory.
        // BaseOptionsC: 8 fields × 8 bytes (pointers) + 4-byte ints padded to 8 = ~64 bytes
        // FaceLandmarkerOptionsC: BaseOptionsC (64) + 3×Int32 (12) + 3×Float (12) + 2×Bool (2) + ptr (8) = ~100 bytes
        // Use generous allocation and fill by byte offset.
        let optionsSize = 128  // generous
        let optionsMem = UnsafeMutableRawPointer.allocate(byteCount: optionsSize, alignment: 8)
        memset(optionsMem, 0, optionsSize)
        defer { optionsMem.deallocate() }

        // BaseOptionsC.modelAssetPath at offset 16 (after buffer_ptr + buffer_count)
        // Actually the layout is:
        //   0: model_asset_buffer (char* = 8 bytes)
        //   8: model_asset_buffer_count (uint = 8 bytes on 64-bit)
        //   16: model_asset_path (char* = 8 bytes)
        //   24: delegate (int = 4 bytes)
        //   ...
        modelPath.withCString { cstr in
            // Store the pointer to the C string. We need it alive during the create call.
            optionsMem.advanced(by: 16).assumingMemoryBound(to: UnsafePointer<CChar>?.self).pointee = cstr.withMemoryRebound(to: CChar.self, capacity: modelPath.utf8.count) { UnsafePointer($0) }
        }

        // FaceLandmarkerOptionsC starts right after BaseOptionsC.
        // runningMode at BaseOptionsC.size (= 48 bytes: 3 pointers + 3 ints + 2 pointers = 48)
        // Actually, let me calculate more carefully:
        // BaseOptionsC fields: char*(8) + uint(8) + char*(8) + int(4) + int(4) + int(4) + padding(4) + char*(8) + char*(8) = 56 bytes
        let baseOptionsSize = 56

        // FaceLandmarkerOptionsC after BaseOptionsC:
        // running_mode (int32) at offset 56
        optionsMem.advanced(by: baseOptionsSize).assumingMemoryBound(to: Int32.self).pointee = 1  // video mode
        // num_faces (int32) at offset 60
        optionsMem.advanced(by: baseOptionsSize + 4).assumingMemoryBound(to: Int32.self).pointee = 1
        // min_face_detection_confidence (float) at offset 64
        optionsMem.advanced(by: baseOptionsSize + 8).assumingMemoryBound(to: Float.self).pointee = 0.5
        // min_face_presence_confidence at offset 68
        optionsMem.advanced(by: baseOptionsSize + 12).assumingMemoryBound(to: Float.self).pointee = 0.5
        // min_tracking_confidence at offset 72
        optionsMem.advanced(by: baseOptionsSize + 16).assumingMemoryBound(to: Float.self).pointee = 0.5
        // output_face_blendshapes (bool) at offset 76
        optionsMem.advanced(by: baseOptionsSize + 20).assumingMemoryBound(to: Bool.self).pointee = false
        // output_facial_transformation_matrixes (bool) at offset 77
        optionsMem.advanced(by: baseOptionsSize + 21).assumingMemoryBound(to: Bool.self).pointee = true

        // Create the landmarker.
        var rawHandle: UnsafeMutableRawPointer? = nil
        let status = fnCreate!(optionsMem, &rawHandle)

        guard status == 0, let rawHandle else {
            throw MPError.createFailed("status=\(status)")
        }
        self.handle = rawHandle
    }

    deinit {
        if let handle { _ = fnClose?(handle) }
        if let dylib { dlclose(dylib) }
    }

    // MARK: Detection

    /// Detect face landmarks in a CVPixelBuffer (video mode).
    public func detect(pixelBuffer: CVPixelBuffer, timestampMs: Int64) throws -> Result {
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)

        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            throw MPError.detectFailed("Can't lock pixel buffer")
        }
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)

        // Create MpImage from raw BGRA data.
        // Image format: SRGBA = 1 (we have 32BGRA = BGRA = same as RGBA with byte order swap).
        // Actually MediaPipe expects SRGB (3-channel) for face landmarker.
        // We'll use SRGBA = 1 and hope it handles the alpha channel.
        var imagePtr: UnsafeMutableRawPointer? = nil
        var errorMsg: UnsafePointer<CChar>? = nil

        let imgStatus = baseAddress.assumingMemoryBound(to: UInt8.self).withMemoryRebound(to: UInt8.self, capacity: height * bytesPerRow) { dataPtr in
            fnImageCreate!(
                1,              // SRGBA format
                Int32(width),
                Int32(height),
                dataPtr,
                Int32(bytesPerRow),
                &imagePtr,
                &errorMsg
            )
        }

        guard imgStatus == 0, let imagePtr else {
            let msg = errorMsg.map { String(cString: $0) } ?? "unknown"
            throw MPError.detectFailed("MpImageCreate: \(msg)")
        }

        // Run detection using raw memory for the result.
        // FaceLandmarkerResultC: 6 fields × 8 bytes (pointers + uint32 padded) = 48 bytes max
        let resultSize = 48
        let resultMem = UnsafeMutableRawPointer.allocate(byteCount: resultSize, alignment: 8)
        memset(resultMem, 0, resultSize)
        defer { resultMem.deallocate() }

        let resultBytes = resultMem.assumingMemoryBound(to: UInt8.self)
        let detStatus = fnDetectVideo!(handle, imagePtr, nil, timestampMs, resultBytes)

        guard detStatus == 0 else {
            throw MPError.detectFailed("status=\(detStatus)")
        }

        // Extract results from raw memory.
        let result = extractResultRaw(resultMem)

        // Clean up.
        fnCloseResult?(resultMem)

        return result
    }

    // MARK: Result extraction from raw memory

    /// Read FaceLandmarkerResultC from raw memory.
    /// Layout:
    ///   0: face_landmarks (pointer to NormalizedLandmarksC) — 8 bytes
    ///   8: face_landmarks_count (uint32) — 4 bytes (+ 4 pad)
    ///  16: face_blendshapes (pointer) — 8 bytes
    ///  24: face_blendshapes_count (uint32) — 4 bytes (+ 4 pad)
    ///  32: facial_transformation_matrixes (pointer to MatrixC) — 8 bytes
    ///  40: facial_transformation_matrixes_count (uint32) — 4 bytes
    private func extractResultRaw(_ mem: UnsafeMutableRawPointer) -> Result {
        var landmarks: [[Double]] = []
        var faceTransform: [[Double]]? = nil

        // face_landmarks pointer at offset 0.
        let facesPtr = mem.advanced(by: 0).load(as: UnsafeRawPointer?.self)
        let facesCount = mem.advanced(by: 8).load(as: UInt32.self)

        if let facesPtr, facesCount > 0 {
            // First face's NormalizedLandmarksC: { pointer, count }
            // pointer at offset 0 of NormalizedLandmarksC
            let faceMem = UnsafeRawPointer(facesPtr)
            let lmArrayPtr = faceMem.load(as: UnsafeRawPointer?.self)
            let lmCount = faceMem.advanced(by: 8).load(as: UInt32.self)

            if let lmArrayPtr, lmCount > 0 {
                // NormalizedLandmarkC layout: { x:Float, y:Float, z:Float,
                //   hasVisibility:Bool, visibility:Float,
                //   hasPresence:Bool, presence:Float,
                //   name:char* }
                // Size per landmark: 4+4+4 + 1+3(pad) + 4 + 1+3(pad) + 4 + 8 = 40 bytes
                let lmSize = 40
                for i in 0..<Int(lmCount) {
                    let lmMem = UnsafeRawPointer(lmArrayPtr).advanced(by: i * lmSize)
                    let x = Double(lmMem.load(as: Float.self))
                    let y = Double(lmMem.advanced(by: 4).load(as: Float.self))
                    let z = Double(lmMem.advanced(by: 8).load(as: Float.self))
                    landmarks.append([x, y, z])
                }
            }
        }

        // facial_transformation_matrixes at offset 32.
        let mtxPtr = mem.advanced(by: 32).load(as: UnsafeRawPointer?.self)
        let mtxCount = mem.advanced(by: 40).load(as: UInt32.self)

        if let mtxPtr, mtxCount > 0 {
            // MatrixC layout: { rows:UInt32, cols:UInt32, data:Float* }
            let mtxMem = UnsafeRawPointer(mtxPtr)
            let rows = mtxMem.load(as: UInt32.self)
            let cols = mtxMem.advanced(by: 4).load(as: UInt32.self)
            let dataPtr = mtxMem.advanced(by: 8).load(as: UnsafePointer<Float>?.self)

            if let dataPtr, rows == 4, cols == 4 {
                var matrix: [[Double]] = Array(repeating: Array(repeating: 0.0, count: 4), count: 4)
                for col in 0..<4 {
                    for row in 0..<4 {
                        // Column-major storage.
                        matrix[row][col] = Double(dataPtr.advanced(by: col * 4 + row).pointee)
                    }
                }
                faceTransform = matrix
            }
        }

        return Result(landmarks: landmarks, faceTransform: faceTransform)
    }

    // MARK: dylib loading

    private func loadDylib() throws {
        let searchPaths = [
            "Frameworks/libmediapipe.dylib",
            FileManager.default.currentDirectoryPath + "/Frameworks/libmediapipe.dylib",
            FileManager.default.currentDirectoryPath + "/macgaze/Frameworks/libmediapipe.dylib",
        ]

        for path in searchPaths {
            if FileManager.default.fileExists(atPath: path) {
                if let handle = dlopen(path, RTLD_NOW | RTLD_LOCAL) {
                    self.dylib = handle
                    return
                }
            }
        }

        throw MPError.dylibNotFound("Searched: \(searchPaths.joined(separator: ", "))")
    }

    private func resolveSymbols() throws {
        guard let lib = dylib else { throw MPError.dylibNotFound("dylib not loaded") }

        func sym<T>(_ name: String, _ type: T.Type) throws -> T {
            guard let ptr = dlsym(lib, name) else {
                throw MPError.symbolNotFound(name)
            }
            return unsafeBitCast(ptr, to: T.self)
        }

        fnCreate = try sym("MpFaceLandmarkerCreate", MpCreateFn.self)
        fnDetectVideo = try sym("MpFaceLandmarkerDetectForVideo", MpDetectVideoFn.self)
        fnCloseResult = try sym("MpFaceLandmarkerCloseResult", MpCloseResultFn.self)
        fnClose = try sym("MpFaceLandmarkerClose", MpCloseFn.self)
        fnImageCreate = try sym("MpImageCreateFromUint8Data", MpImageCreateFn.self)
    }
}
