import Foundation
import Accelerate

/// Reconstructs head pose from FaceMesh landmarks (for the CoreML path, which —
/// unlike MediaPipe — has no facial-transformation matrix), producing the
/// `head_vector` + `face_origin_3d` BlazeGaze expects.
///
/// Rotation is found by Kabsch/orthogonal-Procrustes alignment of the canonical
/// face model to the observed landmarks over a rigid, expression-stable subset;
/// `head_vector` then uses WebEyeTrack's exact euler-swap + spherical formula
/// (model_based.py get_head_vector), and `face_origin_3d` is the eye-landmark
/// midpoint of the aligned (metric) face.
///
/// Conventions are matched empirically against the MediaPipe ground-truth
/// head_vector via `macgaze-headpose-check`.
public enum HeadPoseSolver {

    /// Rigid, expression-stable landmark indices used for alignment.
    static let rigidIndices: [Int] = [
        1, 4, 6, 168, 197, 195, 5,      // nose bridge / tip
        33, 133, 362, 263,              // eye corners
        127, 356, 234, 454,             // temples / face sides
        10, 151, 9,                     // forehead center
    ]

    // Eye-horizontal landmarks for face_origin_3d (WebEyeTrack).
    static let leftEyeIdx = [362, 263]
    static let rightEyeIdx = [33, 133]

    public struct Pose {
        public let headVector: [Float]     // [3]
        public let faceOrigin3D: [Float]   // [3]
    }

    /// `landmarks`: 468×3 (x,y normalized [0,1] top-left; z FaceMesh depth).
    public static func solve(landmarks: [[Double]], width: Int, height: Int) -> Pose? {
        guard landmarks.count >= 468 else { return nil }
        let w = Double(width), h = Double(height)

        // Observed landmarks → isotropic camera-ish 3D (Y up, matching canonical).
        func observed(_ i: Int) -> (Double, Double, Double) {
            let p = landmarks[i]
            return ((p[0] - 0.5) * w, (0.5 - p[1]) * h, -p[2] * w)
        }
        // Canonical model with WebEyeTrack's [-1, 1, -1] flip.
        func canonical(_ i: Int) -> (Double, Double, Double) {
            let v = CanonicalFaceModel.vertices[i]
            return (-Double(v[0]), Double(v[1]), -Double(v[2]))
        }

        var P = [(Double, Double, Double)]()   // canonical
        var Q = [(Double, Double, Double)]()   // observed
        for idx in rigidIndices {
            P.append(canonical(idx)); Q.append(observed(idx))
        }
        guard let R = kabschRotation(from: P, to: Q) else { return nil }

        let head = headVector(from: R)

        // face_origin_3d: midpoint of eye-landmark means, in the observed frame.
        func mean(_ idxs: [Int]) -> [Float] {
            var s = (0.0, 0.0, 0.0)
            for i in idxs { let o = observed(i); s = (s.0 + o.0, s.1 + o.1, s.2 + o.2) }
            let n = Double(idxs.count)
            return [Float(s.0 / n), Float(s.1 / n), Float(s.2 / n)]
        }
        let l = mean(leftEyeIdx), r = mean(rightEyeIdx)
        let origin = [(l[0] + r[0]) / 2, (l[1] + r[1]) / 2, (l[2] + r[2]) / 2]

        return Pose(headVector: head, faceOrigin3D: origin)
    }

    /// WebEyeTrack get_head_vector: euler from R, swap pitch/yaw, spherical, roll.
    /// Public so the ground-truth (MediaPipe transform) path can reuse it.
    public static func headVector(from R: [[Double]]) -> [Float] {
        let pitch = asin(-clamp(R[2][0]))
        let yaw = atan2(R[2][1], R[2][2])
        let roll = atan2(R[1][0], R[0][0])
        let hPitch = -yaw, hYaw = pitch, hRoll = roll
        let x = cos(hPitch) * sin(hYaw)
        let y = sin(hPitch)
        let z = -cos(hPitch) * cos(hYaw)
        // Apply roll about Z.
        let cr = cos(hRoll), sr = sin(hRoll)
        let rx = cr * x - sr * y
        let ry = sr * x + cr * y
        return [Float(rx), Float(ry), Float(z)]
    }

    private static func clamp(_ v: Double) -> Double { max(-1.0, min(1.0, v)) }

    /// Kabsch: rotation aligning centred P onto centred Q (minimises ‖R·P − Q‖).
    static func kabschRotation(from P: [(Double, Double, Double)],
                               to Q: [(Double, Double, Double)]) -> [[Double]]? {
        guard P.count == Q.count, P.count >= 3 else { return nil }
        let n = Double(P.count)
        func centroid(_ a: [(Double, Double, Double)]) -> (Double, Double, Double) {
            var s = (0.0, 0.0, 0.0)
            for p in a { s = (s.0 + p.0, s.1 + p.1, s.2 + p.2) }
            return (s.0 / n, s.1 / n, s.2 / n)
        }
        let cP = centroid(P), cQ = centroid(Q)

        // H = Σ (P_i - cP)(Q_i - cQ)^T   (3×3)
        var H = [Double](repeating: 0, count: 9)
        for i in 0..<P.count {
            let p = (P[i].0 - cP.0, P[i].1 - cP.1, P[i].2 - cP.2)
            let q = (Q[i].0 - cQ.0, Q[i].1 - cQ.1, Q[i].2 - cQ.2)
            let pv = [p.0, p.1, p.2], qv = [q.0, q.1, q.2]
            for r in 0..<3 { for c in 0..<3 { H[r * 3 + c] += pv[r] * qv[c] } }
        }

        guard let (U, V) = svd3x3(H) else { return nil }
        // R = V · U^T, with a sign fix so det(R) = +1 (no reflection).
        var R = matMul3(V, transpose3(U))
        if det3(R) < 0 {
            // Flip sign of V's last column and recompute.
            var Vf = V
            Vf[2] = -Vf[2]; Vf[5] = -Vf[5]; Vf[8] = -Vf[8]
            R = matMul3(Vf, transpose3(U))
        }
        return [[R[0], R[1], R[2]], [R[3], R[4], R[5]], [R[6], R[7], R[8]]]
    }

    // MARK: 3×3 linear algebra (row-major flat arrays)

    /// SVD of a 3×3 (row-major) via LAPACK dgesvd. Returns (U, Vt-transposed→V).
    private static func svd3x3(_ a: [Double]) -> (U: [Double], V: [Double])? {
        // LAPACK is column-major; transpose in/out.
        var A = transpose3(a)
        var m = __CLPK_integer(3), n = __CLPK_integer(3), lda = __CLPK_integer(3)
        var s = [Double](repeating: 0, count: 3)
        var u = [Double](repeating: 0, count: 9)
        var vt = [Double](repeating: 0, count: 9)
        var ldu = __CLPK_integer(3), ldvt = __CLPK_integer(3)
        var work = [Double](repeating: 0, count: 200)
        var lwork = __CLPK_integer(200)
        var info = __CLPK_integer(0)
        var jobu = Int8(UInt8(ascii: "A")), jobvt = Int8(UInt8(ascii: "A"))
        dgesvd_(&jobu, &jobvt, &m, &n, &A, &lda, &s, &u, &ldu, &vt, &ldvt,
                &work, &lwork, &info)
        guard info == 0 else { return nil }
        // A was column-major: u is U (col-major) → transpose to row-major.
        // vt is V^T (col-major) → its transpose (row-major) is V^T; we want V.
        let U = transpose3(u)        // row-major U
        let V = vt                    // col-major V^T == row-major V
        return (U, V)
    }

    private static func transpose3(_ a: [Double]) -> [Double] {
        [a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]]
    }
    private static func matMul3(_ a: [Double], _ b: [Double]) -> [Double] {
        var r = [Double](repeating: 0, count: 9)
        for i in 0..<3 { for j in 0..<3 { var s = 0.0
            for k in 0..<3 { s += a[i * 3 + k] * b[k * 3 + j] }
            r[i * 3 + j] = s } }
        return r
    }
    private static func det3(_ a: [Double]) -> Double {
        a[0] * (a[4] * a[8] - a[5] * a[7])
        - a[1] * (a[3] * a[8] - a[5] * a[6])
        + a[2] * (a[3] * a[7] - a[4] * a[6])
    }
}
