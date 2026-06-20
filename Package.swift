// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "MacGaze",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .library(name: "MacGaze", targets: ["MacGaze"]),
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
        .testTarget(
            name: "MacGazeTests",
            dependencies: ["MacGaze"],
            path: "Tests/MacGazeTests"
        ),
    ]
)
