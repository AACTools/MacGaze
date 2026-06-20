import SwiftUI
import AppKit

@main
struct MacGazeDebugApp: App {
    @StateObject private var pipeline = DebugPipeline()

    var body: some Scene {
        WindowGroup {
            DebugWindow()
                .environmentObject(pipeline)
                .frame(minWidth: 1100, minHeight: 720)
        }
        .windowResizability(.contentSize)
    }
}
