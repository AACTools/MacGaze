import Foundation
import CoreVideo

/// A single camera frame plus its capture timestamp.
///
/// `pixelBuffer` is a `CVBuffer`-backed CoreVideo object — internally
/// thread-safe but not `Sendable` in Swift's type system.  We mark the
/// struct `@unchecked Sendable` because the contract is: **the consumer
/// must finish reading `pixelBuffer` before awaiting the next frame**.
/// Vision and CIImage rendering both lock the buffer for the duration of
/// their synchronous work, so this contract holds in practice.
public struct CameraFrame: @unchecked Sendable, Equatable {
    public let pixelBuffer: CVPixelBuffer
    /// Capture time in seconds since first frame (monotonic).
    public let timestampSeconds: TimeInterval
    public let width: Int
    public let height: Int

    public init(pixelBuffer: CVPixelBuffer, timestampSeconds: TimeInterval) {
        self.pixelBuffer = pixelBuffer
        self.timestampSeconds = timestampSeconds
        self.width = CVPixelBufferGetWidth(pixelBuffer)
        self.height = CVPixelBufferGetHeight(pixelBuffer)
    }

    public static func == (lhs: CameraFrame, rhs: CameraFrame) -> Bool {
        lhs.timestampSeconds == rhs.timestampSeconds
            && lhs.width == rhs.width
            && lhs.height == rhs.height
    }
}
