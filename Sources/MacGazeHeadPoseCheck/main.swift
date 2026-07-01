import Foundation
import AppKit
import CoreVideo
import CoreGraphics
import MacGaze

// Headless sanity-check for HeadPoseSolver: feed CoreML FaceMesh landmarks
// (which run headless, unlike the MediaPipe VIDEO bridge) into the solver and
// print the head_vector + face_origin. For a roughly forward-facing photo the
// head_vector should be ≈ [~0, ~0, -1] with unit magnitude (mostly -Z).
//
//   swift run -c release macgaze-headpose-check Models/face_mesh.mlmodelc demoPic.jpg
let args = CommandLine.arguments
guard args.count >= 3 else {
    print("usage: macgaze-headpose-check <face_mesh.mlmodelc> <image>"); exit(1)
}
guard let nsImage = NSImage(contentsOf: URL(fileURLWithPath: args[2])),
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

let landmarker: CoreMLFaceMeshLandmarker
do { landmarker = try CoreMLFaceMeshLandmarker(modelURL: URL(fileURLWithPath: args[1])) }
catch { print("model load failed: \(error)"); exit(1) }

guard let pb = makePixelBuffer(from: cg),
      let r = landmarker.detect(pixelBuffer: pb), r.landmarks.count >= 468 else {
    print("CoreML detect failed"); exit(1)
}
guard let pose = HeadPoseSolver.solve(landmarks: r.landmarks, width: W, height: H) else {
    print("HeadPoseSolver returned nil"); exit(1)
}
let hv = pose.headVector
let mag = (hv[0]*hv[0] + hv[1]*hv[1] + hv[2]*hv[2]).squareRoot()
print(String(format: "head_vector   = [% .3f, % .3f, % .3f]  |v|=%.3f", hv[0], hv[1], hv[2], mag))
print(String(format: "face_origin_3d = [% .3f, % .3f, % .3f]",
             pose.faceOrigin3D[0], pose.faceOrigin3D[1], pose.faceOrigin3D[2]))
print("(forward-facing photo ⇒ head_vector should be mostly -Z, |v|≈1, no crash)")
