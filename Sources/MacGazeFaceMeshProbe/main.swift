import Foundation
import AppKit
import Vision
import CoreGraphics
import MacGaze

// Usage: macgaze-facemesh-probe <face_mesh.mlmodelc> <image> [out.png]
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

// Vision face rectangle → top-left pixel rect (Vision is bottom-left normalized).
func visionFaceRect(_ image: CGImage) -> CGRect? {
    let request = VNDetectFaceRectanglesRequest()
    try? VNImageRequestHandler(cgImage: image, options: [:]).perform([request])
    guard let face = request.results?.first else { return nil }
    let bb = face.boundingBox
    let x = bb.origin.x * Double(image.width)
    let y = (1.0 - bb.origin.y - bb.height) * Double(image.height)
    return CGRect(x: x, y: y, width: bb.width * Double(image.width),
                  height: bb.height * Double(image.height))
}

func squareCrop(cx: Double, cy: Double, side: Double) -> CGRect {
    var x0 = max(0, cx - side / 2)
    var y0 = max(0, cy - side / 2)
    let s = min(side, Double(W) - x0, Double(H) - y0)
    x0 = min(x0, Double(W) - s)
    y0 = min(y0, Double(H) - s)
    return CGRect(x: x0, y: y0, width: s, height: s)
}

func bboxPixels(_ lms: [[Double]]) -> (Double, Double, Double, Double) {
    var minX = 1.0, minY = 1.0, maxX = 0.0, maxY = 0.0
    for p in lms {
        minX = min(minX, p[0]); maxX = max(maxX, p[0])
        minY = min(minY, p[1]); maxY = max(maxY, p[1])
    }
    return (minX * Double(W), minY * Double(H), maxX * Double(W), maxY * Double(H))
}

let landmarker: CoreMLFaceMeshLandmarker
do { landmarker = try CoreMLFaceMeshLandmarker(modelURL: modelURL) }
catch { print("model load failed: \(error)"); exit(1) }

guard let vr = visionFaceRect(cg) else { print("Vision found no face"); exit(1) }
print("vision rect: \(vr)")

// Pass 1: crop from the Vision rect (generous), get rough landmarks.
let crop1 = squareCrop(cx: vr.midX, cy: vr.midY, side: max(vr.width, vr.height) * 1.4)
guard let r1 = landmarker.run(image: cg, cropRect: crop1) else { print("pass1 failed"); exit(1) }

// Pass 2: re-crop from the landmark bbox × 1.5 (the validated framing).
let (bx0, by0, bx1, by1) = bboxPixels(r1.landmarks)
let crop2 = squareCrop(cx: (bx0 + bx1) / 2, cy: (by0 + by1) / 2,
                       side: max(bx1 - bx0, by1 - by0) * 1.5)
guard let r2 = landmarker.run(image: cg, cropRect: crop2) else { print("pass2 failed"); exit(1) }

let keyIndices = [4, 103, 150, 151, 195, 332, 379]
print(String(format: "pass1 score=%.3f  pass2 score=%.3f  (crop2=%@)",
             r1.score, r2.score, "\(crop2)"))
for k in keyIndices {
    let p = r2.landmarks[k]
    print(String(format: "  [%3d] x=%.4f y=%.4f", k, p[0], p[1]))
}

// Draw all 468 landmarks (green) + the 7 key indices (red) onto the image.
guard let space = CGColorSpace(name: CGColorSpace.sRGB),
      let ctx = CGContext(data: nil, width: W, height: H, bitsPerComponent: 8,
                          bytesPerRow: W * 4, space: space,
                          bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
    print("overlay context failed"); exit(1)
}
ctx.draw(cg, in: CGRect(x: 0, y: 0, width: W, height: H))
for (i, p) in r2.landmarks.enumerated() {
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
