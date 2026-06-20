#!/usr/bin/env python3
"""
Extract MediaPipe face landmarks from a video and save as JSON.

This is the "MediaPipe half" of the hybrid pipeline. Swift handles the
BlazeGaze inference + RBF calibration; Python handles MediaPipe landmark
detection (which we can't yet do natively in Swift).

Usage:
    python extract_landmarks.py <video-file> [--output landmarks.json]

Output JSON format (one entry per frame with a face):
    [
      {
        "frame": 0,
        "timestamp_ms": 0,
        "width": 1280,
        "height": 720,
        "landmarks": [[x,y,z], ...],     // 478 × 3, normalized [0,1]
        "face_transform": [[...4x4...]]  // 4×4 facial transformation matrix
      },
      ...
    ]
"""

import sys
import os
import json
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_landmarks.py <video-file> [--output landmarks.json]")
        sys.exit(1)

    video_path = sys.argv[1]
    output_path = "landmarks.json"
    if "--output" in sys.argv:
        idx = sys.argv.index("--output")
        if idx + 1 < len(sys.argv):
            output_path = sys.argv[idx + 1]

    if not os.path.exists(video_path):
        print(f"error: file not found: {video_path}")
        sys.exit(1)

    # Find the .task model (ships with WebEyeTrack or download from Google).
    script_dir = os.path.dirname(os.path.abspath(__file__))
    task_model = os.path.join(script_dir, "WebEyeTrack-upstream",
                              "python", "webeyetrack", "model_weights",
                              "face_landmarker_v2_with_blendshapes.task")
    if not os.path.exists(task_model):
        # Download from Google's storage.
        print("Downloading face_landmarker.task model...")
        import urllib.request
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
        urllib.request.urlretrieve(url, task_model)

    # Load MediaPipe Face Landmarker.
    print("Loading MediaPipe Face Landmarker...")
    base_options = mp_python.BaseOptions(model_asset_path=task_model)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
        output_facial_transformation_matrixes=True,
    )
    landmarker = mp_vision.FaceLandmarker.create_from_options(options)
    print("  MediaPipe loaded.")

    # Process video.
    print(f"Processing: {os.path.basename(video_path)}")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  {width}x{height} @ {fps:.1f}fps, {total} frames")

    results = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_ms = int(frame_idx * 1000 / fps)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.face_landmarks:
            landmarks = [[lm.x, lm.y, lm.z] for lm in result.face_landmarks[0]]
            face_transform = None
            if result.facial_transformation_matrixes:
                face_transform = np.array(
                    result.facial_transformation_matrixes[0]
                ).tolist()

            results.append({
                "frame": frame_idx,
                "timestamp_ms": timestamp_ms,
                "width": width,
                "height": height,
                "landmarks": landmarks,
                "face_transform": face_transform,
            })

        if frame_idx % 60 == 0:
            print(f"  frame {frame_idx}/{total} ({len(results)} with face)")

        frame_idx += 1

    cap.release()
    landmarker.close()

    # Save JSON.
    with open(output_path, "w") as f:
        json.dump(results, f)

    print(f"\nSaved {len(results)} frames with landmarks to {output_path}")
    print(f"  ({len(results)}/{frame_idx} frames had detectable faces)")
    print(f"\nNow run:")
    print(f"  swift run macgaze-replay {video_path} --landmarks {output_path}")


if __name__ == "__main__":
    main()
