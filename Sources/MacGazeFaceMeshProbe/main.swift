import Foundation
import AppKit
import CoreVideo
import CoreGraphics
import MacGaze

// Usage: macgaze-facemesh-probe <face_mesh.mlmodelc> <image> [out.png]
// Exercises the exact tracker path: image → CVPixelBuffer → detect(pixelBuffer:).
let args = CommandLine.arguments
guard args.count >= 3 else {
    print("usage: macgaze-facemesh-probe <model.mlmodelc> <image> [out.png]")
    exit(1)
}
let modelURL = URL(fileURLWithPath: args[1])
let imageURL = URL(fileURLWithPath: args[2])
let outURL = URL(fileURLWithPath: args.count > 3 ? args[3] : "facemesh_probe_out.png")

guard let nsImage = NSImage(contentsOf: imageURL),
      let cg = nsImage.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
    print("cannot load image \(imageURL.path)"); exit(1)
}
let W = cg.width, H = cg.height

// CGImage → 32BGRA CVPixelBuffer (matches the camera's frame format).
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
                                        | CGBitmapInfo.byteOrder32Little.rawValue)
    else { return nil }
    ctx.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
    return buffer
}

let landmarker: CoreMLFaceMeshLandmarker
do { landmarker = try CoreMLFaceMeshLandmarker(modelURL: modelURL) }
catch { print("model load failed: \(error)"); exit(1) }

guard let pixelBuffer = makePixelBuffer(from: cg) else { print("pixel buffer failed"); exit(1) }
guard let result = landmarker.detect(pixelBuffer: pixelBuffer) else {
    print("detect failed (no face?)"); exit(1)
}

let keyIndices = [4, 103, 150, 151, 195, 332, 379]
print(String(format: "score=%.3f  landmarks=%d", result.score, result.landmarks.count))
for k in keyIndices {
    let p = result.landmarks[k]
    print(String(format: "  [%3d] x=%.4f y=%.4f", k, p[0], p[1]))
}

// Draw all landmarks (green) + the 7 key indices (red) onto the image.
guard let space = CGColorSpace(name: CGColorSpace.sRGB),
      let ctx = CGContext(data: nil, width: W, height: H, bitsPerComponent: 8,
                          bytesPerRow: W * 4, space: space,
                          bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
    print("overlay context failed"); exit(1)
}
ctx.draw(cg, in: CGRect(x: 0, y: 0, width: W, height: H))
for (i, p) in result.landmarks.enumerated() {
    let px = p[0] * Double(W)
    let py = (1.0 - p[1]) * Double(H)   // top-left normalized → bottom-left context
    let key = keyIndices.contains(i)
    ctx.setFillColor(key ? CGColor(red: 1, green: 0, blue: 0, alpha: 1)
                         : CGColor(red: 0, green: 1, blue: 0, alpha: 0.7))
    let r = key ? 5.0 : 1.5
    ctx.fillEllipse(in: CGRect(x: px - r, y: py - r, width: 2 * r, height: 2 * r))
}
if let img = ctx.makeImage() {
    let rep = NSBitmapImageRep(cgImage: img)
    if let data = rep.representation(using: .png, properties: [:]) {
        try? data.write(to: outURL)
        print("wrote overlay: \(outURL.path)")
    }
}
