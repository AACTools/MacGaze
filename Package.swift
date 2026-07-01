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
        .executable(name: "macgaze-eval", targets: ["MacGazeEval"]),
        .executable(name: "macgaze-replay", targets: ["MacGazeReplay"]),
        .executable(name: "macgaze-calibrate", targets: ["MacGazeCalibrate"]),
        .executable(name: "macgaze-control", targets: ["MacGazeControl"]),
    ],
    dependencies: [
        // Reuse GazeBridgeCore (OneEuroFilter, TrackerDriver protocol, etc.)
        .package(path: "../gazebridge"),
    ],
    targets: [
        .target(
            name: "CMediaPipe",
            path: "Sources/CMediaPipe",
            publicHeadersPath: "include"
        ),
        .target(
            name: "MacGaze",
            dependencies: [
                "CMediaPipe",
                .product(name: "GazeBridgeCore", package: "gazebridge"),
            ],
            path: "Sources/MacGaze",
            exclude: [
                "Gaze/blazegaze.mlmodelc",
                "Gaze/blazegaze.mlpackage",
            ]
        ),
        .executableTarget(
            name: "MacGazeSmoke",
            dependencies: ["MacGaze"],
            path: "Sources/MacGazeSmoke"
        ),
        .executableTarget(
            name: "MacGazeEval",
            dependencies: [
                "MacGaze",
                .product(name: "GazeBridgeCore", package: "gazebridge"),
            ],
            path: "Sources/MacGazeEval"
        ),
        .executableTarget(
            name: "MacGazeReplay",
            dependencies: ["MacGaze"],
            path: "Sources/MacGazeReplay"
        ),
        .executableTarget(
            name: "MacGazeCalibrate",
            dependencies: ["MacGaze"],
            path: "Sources/MacGazeCalibrate"
        ),
        .executableTarget(
            name: "MacGazeControl",
            dependencies: ["MacGaze"],
            path: "Sources/MacGazeControl"
        ),
        .testTarget(
            name: "MacGazeTests",
            dependencies: ["MacGaze"],
            path: "Tests/MacGazeTests"
        ),
    ]
)
