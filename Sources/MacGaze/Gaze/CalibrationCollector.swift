import Foundation
import CoreGraphics

/// Accumulates (BlazeGaze output, true target) pairs during a calibration
/// session, with outlier rejection.
///
/// During calibration, for each of the 9 (or 5, or 13) target points:
/// 1. The user looks at the target.
/// 2. BlazeGaze produces a stream of (x, y) predictions.
/// 3. This collector captures them for ~1 second.
/// 4. Outliers (samples > 2σ from the running median) are rejected.
/// 5. The surviving samples are averaged → one CalibrationSample per point.
/// 6. All CalibrationSamples are fed to RBFGazeCorrector.calibrate().
public final class CalibrationCollector {

    /// One raw observation: BlazeGaze's output at a single moment.
    public struct RawObservation: Sendable, Equatable {
        public let x: Double
        public let y: Double
        public let timestampMs: Int64
        public init(x: Double, y: Double, timestampMs: Int64) {
            self.x = x; self.y = y; self.timestampMs = timestampMs
        }
    }

    /// Collected calibration samples (one per target point, after averaging).
    public private(set) var samples: [RBFGazeCorrector.CalibrationSample] = []

    /// Raw observations for the current target, before averaging.
    private var currentObservations: [RawObservation] = []
    /// The target the user is currently looking at.
    private var currentTarget: (x: Double, y: Double)?

    /// Minimum samples required per point before averaging.
    public var minSamplesPerPoint: Int = 10
    /// Outlier rejection threshold in standard deviations.
    public var outlierThresholdSigma: Double = 2.0

    public init() {}

    /// Begin collecting observations for a new target point.
    public func startTarget(x: Double, y: Double) {
        currentTarget = (x, y)
        currentObservations.removeAll()
    }

    /// Add a raw BlazeGaze observation for the current target.
    public func addObservation(x: Double, y: Double, timestampMs: Int64) {
        guard currentTarget != nil else { return }
        currentObservations.append(RawObservation(x: x, y: y, timestampMs: timestampMs))
    }

    /// Finish the current target: reject outliers, average the survivors,
    /// append a CalibrationSample. Returns the sample or nil if too few
    /// observations survived.
    @discardableResult
    public func finishTarget() -> RBFGazeCorrector.CalibrationSample? {
        guard let target = currentTarget else { return nil }
        defer {
            currentTarget = nil
            currentObservations.removeAll()
        }

        let cleaned = rejectOutliers(currentObservations)
        guard cleaned.count >= minSamplesPerPoint else { return nil }

        let avgX = cleaned.reduce(0.0) { $0 + $1.x } / Double(cleaned.count)
        let avgY = cleaned.reduce(0.0) { $0 + $1.y } / Double(cleaned.count)

        let sample = RBFGazeCorrector.CalibrationSample(
            observedX: avgX, observedY: avgY,
            targetX: target.x, targetY: target.y
        )
        samples.append(sample)
        return sample
    }

    /// Discard all collected samples (e.g. user wants to restart calibration).
    public func reset() {
        samples.removeAll()
        currentObservations.removeAll()
        currentTarget = nil
    }

    /// Whether enough points have been collected for a full calibration.
    public var hasEnoughPoints: Bool {
        samples.count >= 3
    }

    // MARK: Outlier rejection

    /// Reject samples that are more than `outlierThresholdSigma` standard
    /// deviations from the median, in both X and Y.
    private func rejectOutliers(_ observations: [RawObservation]) -> [RawObservation] {
        guard observations.count >= 4 else { return observations }

        let xs = observations.map(\.x).sorted()
        let ys = observations.map(\.y).sorted()
        let medX = xs[xs.count / 2]
        let medY = ys[ys.count / 2]

        let varX = observations.reduce(0.0) { $0 + ($1.x - medX) * ($1.x - medX) } / Double(observations.count)
        let varY = observations.reduce(0.0) { $0 + ($1.y - medY) * ($1.y - medY) } / Double(observations.count)
        let stdX = varX.squareRoot()
        let stdY = varY.squareRoot()

        // If variance is ~0 (all points nearly identical), keep everything.
        guard stdX > 1e-6 || stdY > 1e-6 else { return observations }

        let threshX = outlierThresholdSigma * max(stdX, 1e-6)
        let threshY = outlierThresholdSigma * max(stdY, 1e-6)

        return observations.filter { obs in
            abs(obs.x - medX) <= threshX && abs(obs.y - medY) <= threshY
        }
    }
}
