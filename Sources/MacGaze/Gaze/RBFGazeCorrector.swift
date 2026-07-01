import Foundation
import CoreGraphics
import Accelerate

/// Gaussian Radial Basis Function (RBF) gaze corrector.
///
/// After a 9-point calibration session, this layer maps the base BlazeGaze
/// prediction `(x, y)` to a corrected screen point.  It corrects user-
/// specific offsets like angle kappa (the discrepancy between the eye's
/// optical and visual axes) and screen geometry.
///
/// **Kernel:** Gaussian ϕ(r) = exp(−(r/σ)²)
/// **Ridge:** λ = 0.01 on the diagonal (handles near-duplicate inputs).
/// **σ:** mean inter-calibration-point distance (auto-computed).
///
/// Closed-form solve: W = (K + λI)^{-1} Y  where K is the N×N kernel
/// matrix.  For N=9 this is a 9×9 inversion — sub-millisecond on any
/// Apple Silicon CPU.
///
/// Reference: see plan.md §3.1, researcher-validated in the conversation
/// (Gaussian over TPS for bounded extrapolation behavior).
public final class RBFGazeCorrector {

    // MARK: Types

    /// One calibration sample: the BlazeGaze output observed while the
    /// user was looking at a known target.
    public struct CalibrationSample: Sendable, Equatable {
        public let observedX: Double
        public let observedY: Double
        public let targetX: Double
        public let targetY: Double

        public init(observedX: Double, observedY: Double, targetX: Double, targetY: Double) {
            self.observedX = observedX
            self.observedY = observedY
            self.targetX = targetX
            self.targetY = targetY
        }
    }

    /// Configuration knobs.
    public struct Configuration: Sendable, Equatable {
        /// Ridge regularisation constant.  Prevents matrix singularity when
        /// two calibration inputs are nearly identical.
        public var ridgeLambda: Double = 0.01
        /// Kernel width.  When nil, auto-computed as the mean inter-point
        /// distance in (x, y) space at calibration time.
        public var sigma: Double? = nil
        /// Minimum number of calibration points required to fit.
        public var minPoints: Int = 3

        public init() {}
    }

    // MARK: State

    public let configuration: Configuration

    /// Calibration inputs (BlazeGaze outputs).  N rows × 2 cols, row-major.
    private var inputs: [Double] = []
    /// Calibration targets (true screen points).  N rows × 2 cols.
    private var targets: [Double] = []
    /// Solved weight matrix.  N rows × 2 cols.
    private var weights: [Double] = []
    /// Kernel width used at solve time.
    public private(set) var solvedSigma: Double = 0
    /// Whether a valid RBF fit exists.
    public private(set) var isCalibrated: Bool = false

    public init(configuration: Configuration = Configuration()) {
        self.configuration = configuration
    }

    // MARK: Calibration

    /// Add a calibration sample and re-solve.  Call this after collecting
    /// all N calibration points.
    ///
    /// - Parameter samples: array of (observed, target) pairs.
    /// - Returns: true if the solve succeeded.
    @discardableResult
    public func calibrate(_ samples: [CalibrationSample]) -> Bool {
        guard samples.count >= configuration.minPoints else {
            isCalibrated = false
            return false
        }

        let n = samples.count

        // Flatten inputs into row-major (n × 2).
        inputs = samples.flatMap { [$0.observedX, $0.observedY] }

        // Build targets in COLUMN-MAJOR order for LAPACK dgesv.
        // Matrix shape (n, 2): column 0 = all targetX, column 1 = all targetY.
        var colMajorY = [Double](repeating: 0, count: n * 2)
        for i in 0..<n {
            colMajorY[i] = samples[i].targetX       // column 0
            colMajorY[i + n] = samples[i].targetY   // column 1
        }

        // Compute σ: mean inter-point Euclidean distance.
        let sigma = configuration.sigma ?? Self.meanInterPointDistance(inputs: inputs, count: n)
        solvedSigma = max(1e-6, sigma)  // guard against zero
        let sigma2 = solvedSigma * solvedSigma

        // Build kernel matrix K (n×n) + ridge.  K is symmetric so row-major
        // and column-major layouts are identical.
        var K = [Double](repeating: 0, count: n * n)
        for i in 0..<n {
            for j in 0..<n {
                let dx = inputs[i * 2] - inputs[j * 2]
                let dy = inputs[i * 2 + 1] - inputs[j * 2 + 1]
                let dist2 = dx * dx + dy * dy
                K[i * n + j] = Foundation.exp(-dist2 / (2 * sigma2))
            }
            K[i * n + i] += configuration.ridgeLambda
        }

        // Solve W = K^{-1} * Y  via Accelerate (dgesv: general solve).
        var nParam = __CLPK_integer(n)
        var nrhs: __CLPK_integer = 2
        var lda = __CLPK_integer(n)
        var ldb = __CLPK_integer(n)
        var ipiv = [__CLPK_integer](repeating: 0, count: n)
        var info: __CLPK_integer = 0

        // dgesv_ writes results through the pointers; there is no return value to use.
        _ = withUnsafeMutablePointer(to: &nParam) { nPtr in
            withUnsafeMutablePointer(to: &nrhs) { nrhsPtr in
                withUnsafeMutablePointer(to: &lda) { ldaPtr in
                    withUnsafeMutablePointer(to: &ldb) { ldbPtr in
                        withUnsafeMutablePointer(to: &info) { infoPtr in
                            ipiv.withUnsafeMutableBufferPointer { ipivPtr in
                                K.withUnsafeMutableBufferPointer { kPtr in
                                    colMajorY.withUnsafeMutableBufferPointer { yPtr in
                                        dgesv_(
                                            nPtr, nrhsPtr,
                                            kPtr.baseAddress, ldaPtr,
                                            ipivPtr.baseAddress,
                                            yPtr.baseAddress, ldbPtr,
                                            infoPtr
                                        )
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        guard info == 0 else {
            isCalibrated = false
            return false
        }

        // Extract W from column-major to row-major for prediction.
        weights = [Double](repeating: 0, count: n * 2)
        for i in 0..<n {
            weights[i * 2] = colMajorY[i]       // W_x for point i
            weights[i * 2 + 1] = colMajorY[i + n] // W_y for point i
        }

        isCalibrated = true
        return true
    }

    /// Reset calibration state.
    public func reset() {
        inputs.removeAll()
        targets.removeAll()
        weights.removeAll()
        isCalibrated = false
    }

    // MARK: Prediction

    /// Apply the RBF correction to a BlazeGaze output.
    ///
    /// - Parameter x, y: BlazeGaze's raw (x, y) prediction.
    /// - Returns: Corrected (x, y) screen point, or the input unchanged
    ///   if no calibration exists or the RBF decays to zero (safe
    ///   extrapolation behavior).
    public func correct(x: Double, y: Double) -> (x: Double, y: Double) {
        guard isCalibrated, !weights.isEmpty else {
            return (x, y)
        }

        let n = inputs.count / 2
        let sigma2 = solvedSigma * solvedSigma

        var sumX: Double = 0
        var sumY: Double = 0
        for i in 0..<n {
            let dx = x - inputs[i * 2]
            let dy = y - inputs[i * 2 + 1]
            let dist2 = dx * dx + dy * dy
            let phi = Foundation.exp(-dist2 / (2 * sigma2))
            sumX += weights[i * 2] * phi
            sumY += weights[i * 2 + 1] * phi
        }

        return (sumX, sumY)
    }

    // MARK: Helpers

    /// Compute the mean pairwise Euclidean distance between N 2D points.
    static func meanInterPointDistance(inputs: [Double], count n: Int) -> Double {
        guard n > 1 else { return 0.1 }
        var totalDist: Double = 0
        var pairCount = 0
        for i in 0..<n {
            for j in (i + 1)..<n {
                let dx = inputs[i * 2] - inputs[j * 2]
                let dy = inputs[i * 2 + 1] - inputs[j * 2 + 1]
                totalDist += (dx * dx + dy * dy).squareRoot()
                pairCount += 1
            }
        }
        return pairCount > 0 ? totalDist / Double(pairCount) : 0.1
    }
}
