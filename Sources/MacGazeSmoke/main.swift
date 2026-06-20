import Foundation
import MacGaze

/// Phase 0 exit-criterion smoke test.
///
/// Runs the camera + Vision pipeline for N seconds and prints a one-line
/// summary plus a longer report.  Use this to verify the < 33 ms median
/// latency target before proceeding to Phase 1.
///
/// Usage:
///   swift run macgaze-smoke --seconds 10
///
/// First run will hit the TCC camera permission prompt — grant it in
/// System Settings → Privacy & Security → Camera for Terminal / your
/// IDE, then re-run.

@main
struct MacGazeSmoke {
    static func main() async {
        let args = CommandLine.arguments
        var seconds: Double = 10
        if let i = args.firstIndex(of: "--seconds"), i + 1 < args.count, let s = Double(args[i + 1]) {
            seconds = s
        }
        let verbose = args.contains("--verbose")

        print("MacGaze Phase 0 smoke — running for \(seconds) s")

        let camera = CameraCapture()
        let detector = FaceLandmarkDetector()

        do {
            try await camera.start()
        } catch {
            fputs("error: \(error.localizedDescription)\n", stderr)
            exit(EXIT_FAILURE)
        }
        defer { camera.stop() }

        var latencies: [Double] = []
        var framesWithFace = 0
        var totalFrames = 0
        let deadline = Date().addingTimeInterval(seconds)

        for await frame in camera.frames {
            if Date() >= deadline { break }
            let result = detector.detect(frame)
            latencies.append(result.stats.latencyMs)
            totalFrames += 1
            if result.stats.faceFound { framesWithFace += 1 }
            if verbose {
                print(String(format: "frame %4d  latency=%5.1fms  face=%@",
                             totalFrames, result.stats.latencyMs,
                             result.stats.faceFound ? "yes" : "no"))
            }
        }

        guard !latencies.isEmpty else {
            print("no frames captured — is the camera in use by another app?")
            exit(EXIT_FAILURE)
        }

        let sorted = latencies.sorted()
        let p50 = sorted[sorted.count / 2]
        let p95 = sorted[Int(Double(sorted.count) * 0.95)]
        let mean = latencies.reduce(0, +) / Double(latencies.count)
        let fps = Double(totalFrames) / seconds
        let faceDetectRate = Double(framesWithFace) / Double(totalFrames)

        print("")
        print("=== Phase 0 smoke report ===")
        print(String(format: "Frames captured     : %d", totalFrames))
        print(String(format: "Frame rate          : %.1f fps", fps))
        print(String(format: "Latency  mean       : %.2f ms", mean))
        print(String(format: "Latency  p50        : %.2f ms", p50))
        print(String(format: "Latency  p95        : %.2f ms", p95))
        print(String(format: "Face detection rate : %.1f%%", faceDetectRate * 100))
        print("")
        if p50 < 33 {
            print("✓ PASS — median latency < 33 ms target (Phase 0 exit criterion met)")
        } else {
            print("✗ FAIL — median latency \(String(format: "%.1f", p50)) ms >= 33 ms target")
            print("  Investigate before proceeding to Phase 1.")
        }
    }
}
