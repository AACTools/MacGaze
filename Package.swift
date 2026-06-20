// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "MacGaze",
    platforms: [
        .macOS(.v15)
    ],
    products: [
        .library(name: "MacGaze", targets: ["MacGaze"]),
        .executable(name: "macgaze-smoke", targets: ["MacGazeSmoke"]),
    ],
    dependencies: [
        // Reuse GazeBridgeCore (OneEuroFilter, TrackerDriver protocol, etc.)
        .package(path: "../gazebridge"),
    ],
    targets: [
        .target(
            name: "MacGaze",
            dependencies: [
                .product(name: "GazeBridgeCore", package: "gazebridge"),
            ],
            path: "Sources/MacGaze"
        ),
        .executableTarget(
            name: "MacGazeSmoke",
            dependencies: ["MacGaze"],
            path: "Sources/MacGazeSmoke"
        ),
        .testTarget(
            name: "MacGazeTests",
            dependencies: ["MacGaze"],
            path: "Tests/MacGazeTests"
        ),
    ]
)
