import XCTest
@testable import MacGaze

final class CalibrationCollectorTests: XCTestCase {

    func test_collectSamplesAndAverage() {
        let collector = CalibrationCollector()
        collector.minSamplesPerPoint = 3

        collector.startTarget(x: 0.5, y: 0.5)
        // 5 observations clustered around (0.52, 0.48)
        for i in 0..<5 {
            collector.addObservation(x: 0.52 + Double(i) * 0.001,
                                     y: 0.48 - Double(i) * 0.001,
                                     timestampMs: Int64(i * 33))
        }
        let sample = collector.finishTarget()
        XCTAssertNotNil(sample)
        XCTAssertEqual(sample!.targetX, 0.5, accuracy: 1e-9)
        XCTAssertEqual(sample!.targetY, 0.5, accuracy: 1e-9)
        XCTAssertEqual(sample!.observedX, 0.522, accuracy: 0.001,
                       "Observed should be mean of inputs (0.52..0.524)")
    }

    func test_outlierRejection() {
        let collector = CalibrationCollector()
        collector.minSamplesPerPoint = 3

        collector.startTarget(x: 0.5, y: 0.5)
        // 8 clean samples + 2 wild outliers (extreme enough to survive
        // the σ-inflation effect where outliers themselves inflate the std)
        for i in 0..<8 {
            collector.addObservation(x: 0.5, y: 0.5, timestampMs: Int64(i))
        }
        collector.addObservation(x: 10.0, y: 10.0, timestampMs: 998)   // extreme outlier
        collector.addObservation(x: -10.0, y: -10.0, timestampMs: 999) // extreme outlier

        let sample = collector.finishTarget()
        XCTAssertNotNil(sample)
        // Outliers should be rejected → observed ≈ 0.5 (the clean mean)
        XCTAssertEqual(sample!.observedX, 0.5, accuracy: 0.01,
                       "Outliers should not skew the average")
        XCTAssertEqual(sample!.observedY, 0.5, accuracy: 0.01)
    }

    func test_tooFewSamplesReturnsNil() {
        let collector = CalibrationCollector()
        collector.minSamplesPerPoint = 10

        collector.startTarget(x: 0.5, y: 0.5)
        collector.addObservation(x: 0.5, y: 0.5, timestampMs: 0)
        collector.addObservation(x: 0.5, y: 0.5, timestampMs: 33)

        let sample = collector.finishTarget()
        XCTAssertNil(sample, "Should reject point with < minSamplesPerPoint")
    }

    func test_multipleTargetsAccumulate() {
        let collector = CalibrationCollector()
        collector.minSamplesPerPoint = 3

        for (tx, ty) in [(0.2, 0.2), (0.5, 0.5), (0.8, 0.8)] {
            collector.startTarget(x: tx, y: ty)
            for i in 0..<5 {
                collector.addObservation(x: tx + 0.01, y: ty + 0.01,
                                         timestampMs: Int64(i))
            }
            collector.finishTarget()
        }

        XCTAssertEqual(collector.samples.count, 3)
        XCTAssertTrue(collector.hasEnoughPoints)
    }

    func test_resetClearsAll() {
        let collector = CalibrationCollector()
        collector.minSamplesPerPoint = 1

        collector.startTarget(x: 0.5, y: 0.5)
        collector.addObservation(x: 0.5, y: 0.5, timestampMs: 0)
        collector.finishTarget()

        XCTAssertEqual(collector.samples.count, 1)
        collector.reset()
        XCTAssertEqual(collector.samples.count, 0)
        XCTAssertFalse(collector.hasEnoughPoints)
    }

    func test_endToEndCollectorFeedsRBF() {
        // Collect 5 calibration points, feed to RBF, verify it calibrates.
        let collector = CalibrationCollector()
        collector.minSamplesPerPoint = 3

        let gridPoints: [(Double, Double)] = [
            (0.2, 0.2), (0.5, 0.2), (0.8, 0.2),
            (0.2, 0.8), (0.8, 0.8),
        ]
        for (tx, ty) in gridPoints {
            collector.startTarget(x: tx, y: ty)
            // Simulated BlazeGaze output with small noise around target.
            for i in 0..<10 {
                let noise = Double(i % 3) * 0.002 - 0.002
                collector.addObservation(x: tx + noise, y: ty + noise,
                                         timestampMs: Int64(i * 33))
            }
            collector.finishTarget()
        }

        let rbf = RBFGazeCorrector()
        let success = rbf.calibrate(collector.samples)
        XCTAssertTrue(success)
        XCTAssertTrue(rbf.isCalibrated)

        // At a known input, RBF should give reasonable output.
        let result = rbf.correct(x: 0.5, y: 0.2)
        XCTAssertGreaterThan(result.x, 0.4, "RBF should produce reasonable correction")
        XCTAssertLessThan(result.x, 0.6)
    }
}
