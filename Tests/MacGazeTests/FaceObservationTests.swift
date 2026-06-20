import XCTest
import CoreGraphics
@testable import MacGaze

final class FaceObservationTests: XCTestCase {

    func test_equalityIsValueBased() {
        // Two observations with the same payload should be equal.
        let a = FaceObservation(
            boundingBox: CGRect(x: 0.1, y: 0.2, width: 0.5, height: 0.6),
            leftPupil: CGPoint(x: 0.3, y: 0.4),
            rightPupil: CGPoint(x: 0.7, y: 0.4),
            leftEyeCorners: FaceObservation.EyeCorners(
                outer: CGPoint(x: 0.2, y: 0.4),
                inner: CGPoint(x: 0.4, y: 0.4)
            ),
            rightEyeCorners: nil
        )
        let b = a
        XCTAssertEqual(a, b)
    }

    func test_eyeCornersEquality() {
        let c1 = FaceObservation.EyeCorners(outer: .init(x: 1, y: 2), inner: .init(x: 3, y: 4))
        let c2 = FaceObservation.EyeCorners(outer: .init(x: 1, y: 2), inner: .init(x: 3, y: 4))
        let c3 = FaceObservation.EyeCorners(outer: .init(x: 9, y: 9), inner: .init(x: 3, y: 4))
        XCTAssertEqual(c1, c2)
        XCTAssertNotEqual(c1, c3)
    }

    func test_observationWithMissingPupilsIsStillValid() {
        // The realistic Phase 0 case: face found but pupil not resolved.
        let obs = FaceObservation(
            boundingBox: CGRect(x: 0, y: 0, width: 1, height: 1),
            leftPupil: nil,
            rightPupil: nil,
            leftEyeCorners: nil,
            rightEyeCorners: nil
        )
        XCTAssertNil(obs.leftPupil)
        XCTAssertNil(obs.rightPupil)
    }
}
