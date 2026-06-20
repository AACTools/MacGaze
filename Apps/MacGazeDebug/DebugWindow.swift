import SwiftUI
import AppKit
import MacGaze

struct DebugWindow: View {
    @EnvironmentObject private var pipeline: DebugPipeline

    var body: some View {
        HStack(spacing: 0) {
            cameraPane
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            sidebar
                .frame(width: 260)
                .background(.quaternary.opacity(0.3))
        }
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Toggle(isOn: bindingForRunning) {
                    Label(pipeline.sessionState == .running ? "Stop" : "Start",
                          systemImage: pipeline.sessionState == .running ? "stop.fill" : "play.fill")
                }
                .toggleStyle(.button)
                .help("Start / stop the camera + Vision pipeline")
            }
        }
        .task {
            // Auto-start so the developer sees something immediately.
            pipeline.start()
        }
    }

    private var bindingForRunning: Binding<Bool> {
        Binding(
            get: { pipeline.sessionState == .running },
            set: { newValue in
                if newValue { pipeline.start() } else { pipeline.stop() }
            }
        )
    }

    // MARK: Camera pane

    @ViewBuilder
    private var cameraPane: some View {
        ZStack {
            Color.black
            if let image = pipeline.latestImage {
                Image(nsImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .overlay(alignment: .topLeading) {
                        if let detection = pipeline.latestDetection {
                            LandmarkOverlay(detection: detection)
                                .allowsHitTesting(false)
                        }
                    }
            } else {
                ContentUnavailableView(
                    pipeline.sessionState == .failed ? "Camera failed" : "No camera",
                    systemImage: pipeline.sessionState == .failed ? "exclamationmark.triangle" : "camera",
                    description: Text(pipeline.errorMessage ?? "Press Start to begin capturing.")
                )
                .foregroundStyle(.white)
            }
        }
    }

    // MARK: Sidebar

    private var sidebar: some View {
        VStack(alignment: .leading, spacing: 14) {
            statRow("Pipeline") {
                Text(pipeline.sessionState.label)
                    .foregroundStyle(pipeline.sessionState.tint)
            }

            Divider()

            statRow("Frame rate") {
                Text(String(format: "%.1f fps", pipeline.framesPerSecond))
                    .monospacedDigit()
            }
            statRow("Detection latency") {
                Text(String(format: "%.1f ms (median)", pipeline.medianLatencyMs))
                    .monospacedDigit()
                    .foregroundStyle(pipeline.medianLatencyMs < 33 ? .green : .orange)
            }

            Divider()

            if let detection = pipeline.latestDetection {
                statRow("Face detected") {
                    Text(detection.stats.faceFound ? "yes" : "no")
                        .foregroundStyle(detection.stats.faceFound ? .green : .secondary)
                }
                if let obs = detection.observation {
                    if let p = obs.leftPupil {
                        statRow("Left pupil") {
                            Text(String(format: "(%.3f, %.3f)", p.x, p.y)).monospacedDigit()
                        }
                    }
                    if let p = obs.rightPupil {
                        statRow("Right pupil") {
                            Text(String(format: "(%.3f, %.3f)", p.x, p.y)).monospacedDigit()
                        }
                    }
                    statRow("Bounding box") {
                        Text(String(format: "(%.2f, %.2f) %.2f×%.2f",
                                    obs.boundingBox.origin.x, obs.boundingBox.origin.y,
                                    obs.boundingBox.width, obs.boundingBox.height))
                            .monospacedDigit()
                            .font(.caption)
                    }
                }
            } else {
                Text("Waiting for first frame…")
                    .foregroundStyle(.secondary)
                    .font(.callout)
            }

            Divider()

            // BlazeGaze gaze prediction section.
            Text("BlazeGaze").font(.headline)
            if let gaze = pipeline.gazePrediction {
                statRow("Gaze (x, y)") {
                    Text(String(format: "(%.3f, %.3f)", gaze.x, gaze.y))
                        .monospacedDigit()
                }
                statRow("Inference") {
                    Text(String(format: "%.1f ms", pipeline.gazeLatencyMs))
                        .monospacedDigit()
                        .foregroundStyle(pipeline.gazeLatencyMs < 10 ? .green : .orange)
                }
            } else {
                statRow("Gaze") {
                    Text(pipeline.latestDetection?.stats.faceFound == true
                         ? "Model loading…" : "No face")
                        .foregroundStyle(.secondary)
                }
            }

            Spacer()
            Text("MacGaze Debug · Phase 2")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .padding(16)
    }

    @ViewBuilder
    private func statRow<Content: View>(_ title: String, @ViewBuilder content: () -> Content) -> some View {
        HStack(alignment: .firstTextBaseline) {
            Text(title)
                .font(.caption)
                .foregroundStyle(.secondary)
            Spacer()
            content().font(.caption.weight(.medium))
        }
    }
}

// MARK: Landmark overlay

/// Draws face bounding box + pupil dots on top of the camera image.
///
/// Vision reports coordinates in *normalized* image space with origin
/// bottom-left.  SwiftUI views use origin top-left.  We flip Y inside
/// the GeometryReader so the overlay aligns with the camera image
/// (which itself is already flipped to look like a selfie).
struct LandmarkOverlay: View {
    let detection: DetectionResult

    var body: some View {
        GeometryReader { geo in
            if let obs = detection.observation {
                Canvas { ctx, size in
                    let w = size.width
                    let h = size.height
                    // Bounding box
                    let bx = obs.boundingBox.origin.x * w
                    let by = (1 - obs.boundingBox.origin.y - obs.boundingBox.height) * h
                    let bw = obs.boundingBox.width * w
                    let bh = obs.boundingBox.height * h
                    let rect = CGRect(x: bx, y: by, width: bw, height: bh)
                    ctx.stroke(Path(rect), with: .color(.green.opacity(0.8)), lineWidth: 2)

                    // Pupil dots
                    if let p = obs.leftPupil {
                        drawDot(ctx, at: flipY(p, w: w, h: h), color: .red, label: "L")
                    }
                    if let p = obs.rightPupil {
                        drawDot(ctx, at: flipY(p, w: w, h: h), color: .blue, label: "R")
                    }
                }
            }
        }
    }

    private func flipY(_ p: CGPoint, w: CGFloat, h: CGFloat) -> CGPoint {
        CGPoint(x: p.x * w, y: (1 - p.y) * h)
    }

    private func drawDot(_ ctx: GraphicsContext, at point: CGPoint, color: Color, label: String) {
        let rect = CGRect(x: point.x - 6, y: point.y - 6, width: 12, height: 12)
        ctx.fill(Path(ellipseIn: rect), with: .color(color))
        ctx.stroke(Path(ellipseIn: rect), with: .color(.white), lineWidth: 1)
    }
}

extension DebugPipeline.SessionState {
    var label: String {
        switch self {
        case .idle: return "Idle"
        case .starting: return "Starting…"
        case .running: return "Running"
        case .stopped: return "Stopped"
        case .failed: return "Failed"
        }
    }
    var tint: Color {
        switch self {
        case .idle, .stopped: return .secondary
        case .starting: return .orange
        case .running: return .green
        case .failed: return .red
        }
    }
}
