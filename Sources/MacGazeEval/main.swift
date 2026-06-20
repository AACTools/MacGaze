import Foundation
import CoreGraphics
import GazeBridgeCore

/// Offline evaluation tool for GazeBridge Evaluation Recorder sessions.
///
/// Usage:
///   swift run macgaze-eval <session-dir> [--skip-first-ms N] [--validate-only]
///
/// Loads metadata.json, targets.json, reference.jsonl from the given
/// session directory, computes per-target + aggregate metrics, prints
/// a report.  Exit code is non-zero if files are missing or malformed.
///
/// Webcam.mp4 is NOT processed by this tool — that comes later when
/// MacGaze is integrated (Phase 2+).  For now the tool only evaluates
/// the reference tracker vs targets.

@main
struct MacGazeEval {
    static func main() async {
        let args = CommandLine.arguments
        guard args.count >= 2 else {
            print(usage)
            exit(EXIT_FAILURE)
        }
        let sessionDir = URL(fileURLWithPath: args[1])
        var skipFirstMs: Int64 = 500
        var validateOnly = false
        var expectedRate: Double = 30

        var i = 2
        while i < args.count {
            switch args[i] {
            case "--skip-first-ms":
                if i + 1 < args.count, let v = Int64(args[i + 1]) {
                    skipFirstMs = v; i += 2; continue
                }
            case "--validate-only":
                validateOnly = true
            case "--expected-rate":
                if i + 1 < args.count, let v = Double(args[i + 1]) {
                    expectedRate = v; i += 2; continue
                }
            case "-h", "--help":
                print(usage); exit(EXIT_SUCCESS)
            default:
                break
            }
            i += 1
        }

        // 1. Validate directory + files exist.
        do {
            try validateSessionDirectory(sessionDir)
        } catch {
            fputs("error: \(error.localizedDescription)\n", stderr)
            exit(EXIT_FAILURE)
        }
        print("✓ Session directory OK: \(sessionDir.path)")

        if validateOnly {
            print("Validation complete (--validate-only).")
            exit(EXIT_SUCCESS)
        }

        // 2. Load metadata + targets.
        guard let metadata = loadJSON(EvaluationSession.self, at: sessionDir.appendingPathComponent("metadata.json")) else {
            fputs("error: metadata.json is missing or malformed\n", stderr)
            exit(EXIT_FAILURE)
        }
        guard let targets = loadJSON([EvaluationTargetEvent].self, at: sessionDir.appendingPathComponent("targets.json")) else {
            fputs("error: targets.json is missing or malformed\n", stderr)
            exit(EXIT_FAILURE)
        }
        let samples = loadReferenceSamples(at: sessionDir.appendingPathComponent("reference.jsonl"))
        print("  Targets    : \(targets.count)")
        print("  Samples    : \(samples.count)")
        print("  Subject    : \(metadata.subjectIdentifier)")
        print("  Lighting   : \(metadata.lightingNotes.isEmpty ? "(none)" : metadata.lightingNotes)")
        print("")

        // 3. Compute metrics.
        let metrics = evaluateSession(
            targets: targets,
            samples: samples,
            screenBounds: metadata.screenBounds,
            screenWidthMM: metadata.screenWidthMM,
            screenHeightMM: metadata.screenHeightMM,
            skipFirstMs: skipFirstMs,
            expectedSampleRateHz: expectedRate
        )

        // 4. Print report.
        printReport(metrics: metrics, metadata: metadata, skipFirstMs: skipFirstMs)

        // 5. Exit code: 0 if we got usable metrics, 1 if no targets evaluated.
        if metrics.meanErrorNormalized == nil {
            exit(EXIT_FAILURE)
        }
    }

    static let usage = """
    Usage: macgaze-eval <session-dir> [options]

    Options:
      --skip-first-ms N    Exclude the first N ms of each target's dwell
                            window from the mean (default 500).
      --expected-rate HZ   Reference tracker's expected sample rate, used
                            for detection-rate calculation (default 30).
      --validate-only      Only check the directory is well-formed; don't
                            compute metrics.
      -h, --help           Show this help.
    """
}

// MARK: - Validation

private func validateSessionDirectory(_ url: URL) throws {
    var isDir: ObjCBool = false
    guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir),
          isDir.boolValue else {
        throw NSError(domain: "MacGazeEval", code: 1,
                      userInfo: [NSLocalizedDescriptionKey: "Not a directory: \(url.path)"])
    }
    let required: [(String, Bool)] = [
        ("metadata.json", true),
        ("targets.json", true),
        ("reference.jsonl", false),  // may legitimately be empty
        ("webcam.mp4", false),       // optional if recordWebcam was off
    ]
    for (filename, requiredPresent) in required {
        let path = url.appendingPathComponent(filename).path
        if FileManager.default.fileExists(atPath: path) { continue }
        if requiredPresent {
            throw NSError(domain: "MacGazeEval", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "Missing required file: \(filename)"])
        }
    }
}

// MARK: - JSON loading

private func loadJSON<T: Decodable>(_ type: T.Type, at url: URL) -> T? {
    guard let data = try? Data(contentsOf: url) else {
        fputs("loadJSON: could not read \(url.path)\n", stderr)
        return nil
    }
    let decoder = JSONDecoder()
    decoder.dateDecodingStrategy = .iso8601
    do {
        return try decoder.decode(T.self, from: data)
    } catch {
        fputs("loadJSON: decode failed for \(url.path): \(error)\n", stderr)
        return nil
    }
}

private func loadReferenceSamples(at url: URL) -> [EvaluationReferenceSample] {
    guard let data = try? Data(contentsOf: url),
          let text = String(data: data, encoding: .utf8) else { return [] }
    var samples: [EvaluationReferenceSample] = []
    for line in text.split(separator: "\n") {
        guard let lineData = line.data(using: .utf8),
              let sample = try? JSONDecoder().decode(EvaluationReferenceSample.self, from: lineData) else {
            continue
        }
        samples.append(sample)
    }
    return samples
}

// MARK: - Report printing

private func printReport(metrics: SessionMetrics, metadata: EvaluationSession, skipFirstMs: Int64) {
    print("=== Session report ===")
    print(String(format: "  Duration              : %.1f s", Double(metrics.sessionDurationMs) / 1000.0))
    print(String(format: "  Total samples         : %d", metrics.totalSamples))
    if metrics.sessionDurationMs > 0 {
        let actualRate = Double(metrics.totalSamples) / (Double(metrics.sessionDurationMs) / 1000.0)
        print(String(format: "  Effective sample rate : %.1f Hz (expected %.0f Hz)",
                     actualRate, metrics.expectedSampleRateHz))
    }
    print(String(format: "  Overall detection     : %.0f%%", metrics.overallDetectionRate * 100))
    print("")

    if let mean = metrics.meanErrorNormalized {
        print("=== Reference-tracker accuracy ===")
        print(String(format: "  Mean error (normalised): %.4f", mean))
        if let p50 = metrics.p50ErrorNormalized {
            print(String(format: "  p50 error (normalised) : %.4f", p50))
        }
        if let p95 = metrics.p95ErrorNormalized {
            print(String(format: "  p95 error (normalised) : %.4f", p95))
        }
        if let px = metrics.meanErrorPixels {
            print(String(format: "  Mean error (pixels)    : %.1f px", px))
        }
        if let mm = metrics.meanErrorMM {
            print(String(format: "  Mean error (mm)        : %.1f mm", mm))
        }
        print("")
    } else {
        print("⚠ No reference samples fell inside any target's evaluation window.")
        print("  Check that recording was started with the reference tracker enabled.")
        print("")
    }

    print("=== Per-target detail ===")
    print("  #    target       mean gaze    meanErr   samples  detect")
    for t in metrics.targetMetrics {
        let err = t.errorNormalized.map { String(format: "%.4f", $0) } ?? "—"
        let mean = t.meanGaze.map { String(format: "(%.3f,%.3f)", $0.x, $0.y) } ?? "—"
        let expected = Int(Double(max(0, t.dwellMs - skipFirstMs)) / 1000.0 * metrics.expectedSampleRateHz)
        let target = String(format: "(%.3f,%.3f)", t.targetPosition.x, t.targetPosition.y)
        let detectPct = Int(t.detectionRate * 100)
        // Pad each field to the header column width to keep the table aligned.
        let row = "  \(pad("\(t.targetSequence + 1)", to: 4)) " +
                  "\(pad(target, to: 12)) " +
                  "\(pad(mean, to: 12)) " +
                  "\(pad(err, to: 10)) " +
                  "\(pad("\(t.sampleCount)/\(expected)", to: 8)) " +
                  "\(detectPct)%"
        print(row)
    }
    print("")
    print("(Evaluation window excludes the first \(skipFirstMs) ms of each target dwell.)")
}

/// Right-pad a string to at least `length` characters (no-op if already longer).
private func pad(_ s: String, to length: Int) -> String {
    if s.count >= length { return s }
    return s + String(repeating: " ", count: length - s.count)
}
