import Foundation
import AppKit
import CoreVideo
import CoreGraphics
import MacGaze

// Validate HeadPoseSolver (landmark Kabsch) against MediaPipe's ground-truth
// head_vector (from its facial-transformation matrix) on a still image.
//
//   swift run -c release macgaze-headpose-check demoPic.jpg
let args = CommandLine.arguments
guard args.count >= 2 else { print("usage: macgaze-headpose-check <image>"); exit(1) }

guard let nsImage = NSImage(contentsOf: URL(fileURLWithPath: args[1])),
      let cg = nsImage.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
    print("cannot load image"); exit(1)
}
let W = cg.width, H = cg.height

func makePixelBuffer(from image: CGImage) -> CVPixelBuffer? {
    let attrs = [kCVPixelBufferCGImageCompatibilityKey: true,
                 kCVPixelBufferCGBitmapContextCompatibilityKey: true] as CFDictionary
    var pb: CVPixelBuffer?
    CVPixelBufferCreate(kCFAllocatorDefault, image.width, image.height,
                        kCVPixelFormatType_32BGRA, attrs, &pb)
    guard let buffer = pb else { return nil }
    CVPixelBufferLockBaseAddress(buffer, [])
    defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
    guard let base = CVPixelBufferGetBaseAddress(buffer),
          let space = CGColorSpace(name: CGColorSpace.sRGB),
          let ctx = CGContext(data: base, width: image.width, height: image.height,
                              bitsPerComponent: 8,
                              bytesPerRow: CVPixelBufferGetBytesPerRow(buffer), space: space,
                              bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
                                        | CGBitmapInfo.byteOrder32Little.rawValue) else { return nil }
    ctx.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
    return buffer
}

func resolveModelPath() -> String? {
    let cwd = FileManager.default.currentDirectoryPath
    return [
        "Frameworks/face_landmarker_v2_with_blendshapes.task",
        cwd + "/Frameworks/face_landmarker_v2_with_blendshapes.task",
    ].first { FileManager.default.fileExists(atPath: $0) }
}

func angleDeg(_ a: [Float], _ b: [Float]) -> Double {
    let dot = Double(a[0]*b[0] + a[1]*b[1] + a[2]*b[2])
    let na = (Double(a[0]*a[0] + a[1]*a[1] + a[2]*a[2])).squareRoot()
    let nb = (Double(b[0]*b[0] + b[1]*b[1] + b[2]*b[2])).squareRoot()
    guard na > 0, nb > 0 else { return .nan }
    return acos(max(-1, min(1, dot / (na * nb)))) * 180 / .pi
}

guard let path = resolveModelPath(),
      let mp = try? MediaPipeFaceLandmarker(modelPath: path) else {
    print("MediaPipe model not found — run ./scripts/setup.sh"); exit(1)
}
guard let pb = makePixelBuffer(from: cg) else { print("pixel buffer failed"); exit(1) }
// VIDEO mode wants increasing timestamps + a warm-up frame or two.
var landmarks: [[Double]] = []
var faceTransform: [[Double]]?
for t in 1...5 {
    do {
        let r = try mp.detect(pixelBuffer: pb, timestampMs: Int64(t) * 33)
        landmarks = r.landmarks
        faceTransform = r.faceTransform
    } catch {
        if t == 5 { print("detect threw: \(error)"); exit(1) }
    }
}
guard landmarks.count >= 468 else { print("MediaPipe landmarks=\(landmarks.count)"); exit(1) }
guard let ft = faceTransform, ft.count == 4, ft[0].count >= 3 else {
    print("no facial transform"); exit(1)
}

let R: [[Double]] = [[Double(ft[0][0]), Double(ft[0][1]), Double(ft[0][2])],
                     [Double(ft[1][0]), Double(ft[1][1]), Double(ft[1][2])],
                     [Double(ft[2][0]), Double(ft[2][1]), Double(ft[2][2])]]
let res = (landmarks: landmarks, faceTransform: ft)
let gt = HeadPoseSolver.headVector(from: R)

guard let est = HeadPoseSolver.solve(landmarks: res.landmarks, width: W, height: H) else {
    print("HeadPoseSolver failed"); exit(1)
}

func fmt(_ v: [Float]) -> String { String(format: "[% .3f, % .3f, % .3f]", v[0], v[1], v[2]) }
print("ground-truth head_vector (MediaPipe): \(fmt(gt))")
print("estimated    head_vector (Kabsch):    \(fmt(est.headVector))")
print(String(format: "angular error: %.1f°", angleDeg(gt, est.headVector)))
print("face_origin_3d (Kabsch): \(fmt(est.faceOrigin3D))")
