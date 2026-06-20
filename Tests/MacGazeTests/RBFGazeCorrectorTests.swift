import XCTest
@testable import MacGaze

final class RBFGazeCorrectorTests: XCTestCase {

    // MARK: Perfect recovery — RBF should recover the identity mapping
    // when observed == target at every calibration point.

    func test_identityMappingRecoversPerfectly() {
        let points: [(Double, Double)] = [
            (0.2, 0.2), (0.5, 0.2), (0.8, 0.2),
            (0.2, 0.5), (0.5, 0.5), (0.8, 0.5),
            (0.2, 0.8), (0.5, 0.8), (0.8, 0.8),
        ]
        let samples = points.map {
            RBFGazeCorrector.CalibrationSample(
                observedX: $0.0, observedY: $0.1,
                targetX: $0.0, targetY: $0.1
            )
        }
        let rbf = RBFGazeCorrector()
        XCTAssertTrue(rbf.calibrate(samples))

        // At each calibration point, corrected output should match target.
        // Tolerance: ridge (λ=0.01) + edge effect at grid corners → ~0.015
        // max deviation on a 9-point grid. Use 0.02 to cover corners.
        for p in points {
            let result = rbf.correct(x: p.0, y: p.1)
            XCTAssertEqual(result.x, p.0, accuracy: 0.02,
                           "Identity mapping should recover target x at (\(p.0),\(p.1))")
            XCTAssertEqual(result.y, p.1, accuracy: 0.02,
                           "Identity mapping should recover target y at (\(p.0),\(p.1))")
        }
    }

    // MARK: Constant offset — if BlazeGaze consistently predicts 0.1 too
    // far right, RBF should correct it.

    func test_constantOffsetCorrection() {
        let targets: [(Double, Double)] = [
            (0.2, 0.2), (0.5, 0.5), (0.8, 0.8),
        ]
        // BlazeGaze output is always 0.1 too high in X.
        let samples = targets.map {
            RBFGazeCorrector.CalibrationSample(
                observedX: $0.0 + 0.1, observedY: $0.1,
                targetX: $0.0, targetY: $0.1
            )
        }
        let rbf = RBFGazeCorrector()
        XCTAssertTrue(rbf.calibrate(samples))

        // Correcting a point that was observed at (0.6, 0.5) should give ~0.5 in X.
        let result = rbf.correct(x: 0.6, y: 0.5)
        XCTAssertEqual(result.x, 0.5, accuracy: 0.05,
                       "Should correct the 0.1 X offset")
    }

    // MARK: Extrapolation safety — far from calibration points, the
    // correction should decay toward 0 (fall back gracefully).

    func test_extrapolationDecaysGracefully() {
        let targets: [(Double, Double)] = [
            (0.4, 0.4), (0.5, 0.5), (0.6, 0.6),
        ]
        let samples = targets.map {
            RBFGazeCorrector.CalibrationSample(
                observedX: $0.0, observedY: $0.1,
                targetX: $0.0 + 0.2, targetY: $0.1  // +0.2 correction
            )
        }
        let rbf = RBFGazeCorrector()
        XCTAssertTrue(rbf.calibrate(samples))

        // At a calibration point: correction should be close to +0.2.
        let atCalib = rbf.correct(x: 0.5, y: 0.5)
        XCTAssertGreaterThan(atCalib.x, 0.6,
                             "At calibration point, correction should be strong")

        // Far from calibration points (x=10, y=10): correction should
        // decay toward 0 because Gaussian kernel → 0 at large distances.
        let farAway = rbf.correct(x: 10.0, y: 10.0)
        XCTAssertLessThan(abs(farAway.x), 0.1,
                          "Far from calibration, correction should decay toward 0")
    }

    // MARK: Minimum points check

    func test_tooFewPointsFailsCalibration() {
        let samples = [
            RBFGazeCorrector.CalibrationSample(
                observedX: 0.5, observedY: 0.5, targetX: 0.5, targetY: 0.5
            ),
        ]
        let rbf = RBFGazeCorrector()
        XCTAssertFalse(rbf.calibrate(samples), "Should reject < minPoints")
        XCTAssertFalse(rbf.isCalibrated)
    }

    // MARK: Uncalibrated passthrough

    func test_uncalibratedPassthrough() {
        let rbf = RBFGazeCorrector()
        let result = rbf.correct(x: 0.42, y: 0.7)
        XCTAssertEqual(result.x, 0.42, accuracy: 1e-9)
        XCTAssertEqual(result.y, 0.7, accuracy: 1e-9,
                       "Uncalibrated corrector should pass input through unchanged")
    }

    // MARK: Reset

    func test_resetClearsCalibration() {
        let samples = (0..<5).map { i in
            RBFGazeCorrector.CalibrationSample(
                observedX: Double(i) * 0.2, observedY: 0.5,
                targetX: Double(i) * 0.2, targetY: 0.5
            )
        }
        let rbf = RBFGazeCorrector()
        XCTAssertTrue(rbf.calibrate(samples))
        XCTAssertTrue(rbf.isCalibrated)
        rbf.reset()
        XCTAssertFalse(rbf.isCalibrated)
        let result = rbf.correct(x: 0.5, y: 0.5)
        XCTAssertEqual(result.x, 0.5, accuracy: 1e-9,
                       "After reset, should passthrough")
    }

    // MARK: Mean inter-point distance

    func test_meanInterPointDistance_9pointGrid() {
        let inputs: [Double] = [
            0.2, 0.2,  0.5, 0.2,  0.8, 0.2,
            0.2, 0.5,  0.5, 0.5,  0.8, 0.5,
            0.2, 0.8,  0.5, 0.8,  0.8, 0.8,
        ]
        let mean = RBFGazeCorrector.meanInterPointDistance(inputs: inputs, count: 9)
        // In a 9-point grid with 0.3 spacing, mean distance should be
        // somewhere between 0.3 (adjacent) and 0.85 (diagonal).
        XCTAssertGreaterThan(mean, 0.3)
        XCTAssertLessThan(mean, 0.85)
    }

    func test_meanInterPointDistance_singlePoint() {
        let mean = RBFGazeCorrector.meanInterPointDistance(inputs: [0.5, 0.5], count: 1)
        XCTAssertEqual(mean, 0.1, accuracy: 1e-9,
                       "Single point should return default σ = 0.1")
    }

    // MARK: Custom sigma

    func test_customSigmaIsRespected() {
        var config = RBFGazeCorrector.Configuration()
        config.sigma = 0.01  // very tight — almost identity at calib points
        let rbf = RBFGazeCorrector(configuration: config)

        let samples: [RBFGazeCorrector.CalibrationSample] = [
            .init(observedX: 0.3, observedY: 0.5, targetX: 0.3, targetY: 0.5),
            .init(observedX: 0.5, observedY: 0.5, targetX: 0.5, targetY: 0.5),
            .init(observedX: 0.7, observedY: 0.5, targetX: 0.7, targetY: 0.5),
        ]
        XCTAssertTrue(rbf.calibrate(samples))
        XCTAssertEqual(rbf.solvedSigma, 0.01, accuracy: 1e-9)
    }
}
