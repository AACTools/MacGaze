import Foundation

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
            return ((p[0] - 0.5) * w, (0.5 - p[1]) * h, p[2] * w)
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

    /// Kabsch via Horn's quaternion method (no LAPACK). Rotation R such that
    /// q_i ≈ R·p_i (aligns centred canonical P onto observed Q). Always a proper
    /// rotation, so no reflection fix needed.
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

        // Cross-covariance S = Σ (P_i-cP)(Q_i-cQ)^T
        var Sxx = 0.0, Sxy = 0.0, Sxz = 0.0
        var Syx = 0.0, Syy = 0.0, Syz = 0.0
        var Szx = 0.0, Szy = 0.0, Szz = 0.0
        for i in 0..<P.count {
            let px = P[i].0 - cP.0, py = P[i].1 - cP.1, pz = P[i].2 - cP.2
            let qx = Q[i].0 - cQ.0, qy = Q[i].1 - cQ.1, qz = Q[i].2 - cQ.2
            Sxx += px*qx; Sxy += px*qy; Sxz += px*qz
            Syx += py*qx; Syy += py*qy; Syz += py*qz
            Szx += pz*qx; Szy += pz*qy; Szz += pz*qz
        }

        // Symmetric 4×4 N (Horn 1987).
        var N = [
            [Sxx + Syy + Szz, Syz - Szy,        Szx - Sxz,        Sxy - Syx],
            [Syz - Szy,       Sxx - Syy - Szz,  Sxy + Syx,        Szx + Sxz],
            [Szx - Sxz,       Sxy + Syx,       -Sxx + Syy - Szz,  Syz + Szy],
            [Sxy - Syx,       Szx + Sxz,        Syz + Szy,       -Sxx - Syy + Szz],
        ]
        // Shift to positive-definite so power iteration finds the most-positive
        // eigenvalue (the optimal rotation quaternion).
        var shift = 1.0
        for r in 0..<4 { for c in 0..<4 { shift += abs(N[r][c]) } }
        for i in 0..<4 { N[i][i] += shift }

        // Power iteration → dominant eigenvector = quaternion [w, x, y, z].
        var v = [1.0, 0.0, 0.0, 0.0]
        for _ in 0..<128 {
            var nv = [0.0, 0.0, 0.0, 0.0]
            for r in 0..<4 { var s = 0.0; for c in 0..<4 { s += N[r][c] * v[c] }; nv[r] = s }
            let mag = (nv[0]*nv[0] + nv[1]*nv[1] + nv[2]*nv[2] + nv[3]*nv[3]).squareRoot()
            guard mag > 1e-12 else { return nil }
            for i in 0..<4 { v[i] = nv[i] / mag }
        }

        let w = v[0], x = v[1], y = v[2], z = v[3]
        return [
            [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
            [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
        ]
    }
}
