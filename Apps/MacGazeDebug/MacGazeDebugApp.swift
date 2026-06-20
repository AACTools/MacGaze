import SwiftUI
import AppKit
import Metal

@main
struct MacGazeDebugApp: App {
    @StateObject private var pipeline = DebugPipeline()

    init() {
        // Warm up Metal synchronously on the main thread BEFORE the
        // display cycle starts.  This prevents a telemetry labeller race
        // ( -[NSNumber length]: unrecognized selector in setLabel: )
        // that crashes when Core Animation creates its first Metal
        // command buffer under load.
        if let device = MTLCreateSystemDefaultDevice() {
            let queue = device.makeCommandQueue()
            queue?.label = "MacGazeDebug.preinit"
            _ = queue?.makeCommandBuffer()?.makeBlitCommandEncoder()
        }
    }

    var body: some Scene {
        WindowGroup {
            DebugWindow()
                .environmentObject(pipeline)
                .frame(minWidth: 1100, minHeight: 720)
        }
        .windowResizability(.contentSize)
    }
}
