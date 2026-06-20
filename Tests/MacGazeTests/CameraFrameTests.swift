import XCTest
import CoreVideo
@testable import MacGaze

final class CameraFrameTests: XCTestCase {

    func test_frameExposesWidthAndHeight() throws {
        let buffer = try makeTestPixelBuffer(width: 1280, height: 720)
        let frame = CameraFrame(pixelBuffer: buffer, timestampSeconds: 1.5)
        XCTAssertEqual(frame.width, 1280)
        XCTAssertEqual(frame.height, 720)
        XCTAssertEqual(frame.timestampSeconds, 1.5)
    }

    func test_equalityIgnoresBufferIdentity() throws {
        // Two frames with same dims + timestamp should be equal even if
        // backed by different buffers.  (We compare metadata, not pixels.)
        let b1 = try makeTestPixelBuffer(width: 640, height: 480)
        let b2 = try makeTestPixelBuffer(width: 640, height: 480)
        let f1 = CameraFrame(pixelBuffer: b1, timestampSeconds: 2.0)
        let f2 = CameraFrame(pixelBuffer: b2, timestampSeconds: 2.0)
        XCTAssertEqual(f1, f2)
    }

    func test_unequalTimestampsAreNotEqual() throws {
        let b = try makeTestPixelBuffer(width: 640, height: 480)
        let f1 = CameraFrame(pixelBuffer: b, timestampSeconds: 1.0)
        let f2 = CameraFrame(pixelBuffer: b, timestampSeconds: 1.5)
        XCTAssertNotEqual(f1, f2)
    }

    // MARK: Helpers

    private func makeTestPixelBuffer(width: Int, height: Int) throws -> CVPixelBuffer {
        var buffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width, height,
            kCVPixelFormatType_32BGRA,
            nil,
            &buffer
        )
        XCTAssertEqual(status, kCVReturnSuccess)
        return try XCTUnwrap(buffer)
    }
}
