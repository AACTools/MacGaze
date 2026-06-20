#!/usr/bin/env python3
"""
Validate that MediaPipe landmarks + BlazeGaze produce better gaze
predictions than our Vision-based approximation.

Runs the EXACT WebEyeTrack pipeline (MediaPipe 468-pt landmarks +
obtain_eyepatch + BlazeGaze) on a recorded video and prints per-second
gaze averages — same format as macgaze-replay so we can compare.

Usage (from Tools/Conversion/ with .venv activated):
    python validate_mediapipe.py ../../../me/me-calibrated.mov
"""

import sys
import os
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import tensorflow as tf

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
UPSTREAM = os.path.join(SCRIPT_DIR, "WebEyeTrack-upstream")
MODEL_WEIGHTS = os.path.join(UPSTREAM, "python", "webeyetrack", "model_weights")
FACE_LANDMARKER = os.path.join(MODEL_WEIGHTS, "face_landmarker_v2_with_blendshapes.task")
BLAZEGAZE = os.path.join(MODEL_WEIGHTS, "blazegaze_mpiifacegaze.keras")

# Load the Keras BlazeGaze model directly (no WebEyeTrack package imports).
print("Loading BlazeGaze model...")
blaze_model = tf.keras.models.load_model(BLAZEGAZE, compile=False)
print(f"  Model: {blaze_model.name}")
print(f"  Inputs: {[(inp.name, inp.shape) for inp in blaze_model.inputs]}")
print(f"  Outputs: {[(out.name, out.shape) for out in blaze_model.outputs]}")


def obtain_eyepatch(frame, face_landmarks, face_padding_coefs=(0.4, 0.2),
                    face_crop_size=512, dst_img_size=(512, 128)):
    """Ported from webeyetrack/model_based.py — the EXACT eye patch crop
    BlazeGaze was trained on. Uses MediaPipe landmark indices."""
    lefttop = face_landmarks[103]
    leftbottom = face_landmarks[150]
    righttop = face_landmarks[332]
    rightbottom = face_landmarks[379]
    center = face_landmarks[4]

    src_pts = np.array([lefttop, leftbottom, rightbottom, righttop], dtype=np.float32)
    src_direction = src_pts - center
    src_pts = src_pts + np.array(face_padding_coefs) * src_direction

    dst_pts = np.array([
        [0, 0], [0, face_crop_size],
        [face_crop_size, face_crop_size], [face_crop_size, 0]
    ], dtype=np.float32)

    M, _ = cv2.findHomography(src_pts, dst_pts)
    warped = cv2.warpPerspective(frame, M, (face_crop_size, face_crop_size))

    warped_lm = np.dot(M, np.vstack((face_landmarks.T, np.ones((1, face_landmarks.shape[0])))))
    warped_lm = (warped_lm[:2, :] / warped_lm[2, :]).T.astype(np.int32)

    top = warped_lm[151]
    bottom = warped_lm[195]
    eyes_patch = warped[top[1]:bottom[1], :]
    eyes_patch = cv2.resize(eyes_patch, dst_img_size)
    return eyes_patch


def get_head_vector(rt):
    """Ported from webeyetrack/model_based.py."""
    R = rt[:3, :3]
    pitch = np.arcsin(-R[2, 0])
    yaw = np.arctan2(R[2, 1], R[2, 2])
    roll = np.arctan2(R[1, 0], R[0, 0])

    h_pitch, h_yaw, h_roll = -yaw, pitch, roll
    cp, sp = np.cos(h_pitch), np.sin(h_pitch)
    cy, sy = np.cos(h_yaw), np.sin(h_yaw)
    z = -cp * cy
    x = cp * sy
    y = sp
    return np.array([x, y, z])


def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_mediapipe.py <video-file>")
        sys.exit(1)

    video_path = sys.argv[1]
    if not os.path.exists(video_path):
        print(f"error: file not found: {video_path}")
        sys.exit(1)

    # 1. BlazeGaze model already loaded above via tf.keras.models.load_model

    @tf.function
    def infer_fn(image, head_vector, face_origin_3d):
        return blaze_model([image, head_vector, face_origin_3d], training=False)

    print("  BlazeGaze loaded.")

    # 2. Load MediaPipe Face Landmarker
    print("Loading MediaPipe Face Landmarker...")
    base_options = mp_python.BaseOptions(model_asset_path=FACE_LANDMARKER)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1
    )
    landmarker = mp_vision.FaceLandmarker.create_from_options(options)
    print("  MediaPipe loaded.")

    # 3. Process video
    print(f"\nProcessing: {os.path.basename(video_path)}")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_idx = 0
    per_second = {}  # sec → (sum_x, sum_y, count)

    # Warmup BlazeGaze
    dummy_img = tf.random.uniform((1, 128, 512, 3))
    dummy_hv = tf.zeros((1, 3))
    dummy_fo = tf.zeros((1, 3))
    _ = infer_fn(dummy_img, dummy_hv, dummy_fo)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_ms = int(frame_idx * 1000 / fps)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run MediaPipe face landmark detection
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.face_landmarks:
            frame_idx += 1
            continue

        # Extract landmarks (478 points, normalized [0,1])
        landmarks = np.array([[lm.x, lm.y, lm.z, lm.visibility, lm.presence]
                             for lm in result.face_landmarks[0]], dtype=np.float32)

        # Get the facial transformation matrix (4x4)
        if result.facial_transformation_matrixes:
            face_rt = np.array(result.facial_transformation_matrixes[0], dtype=np.float32)
        else:
            face_rt = np.eye(4, dtype=np.float32)

        # Prepare BlazeGaze inputs using the exact WebEyeTrack eye patch crop
        landmarks_2d = (landmarks[:, :2] * np.array([frame.shape[1], frame.shape[0]],
                         dtype=np.float32)).astype(np.float32)

        # obtain_eyepatch: the EXACT crop the model was trained on
        eye_patch = obtain_eyepatch(frame_rgb, landmarks_2d)

        # head_vector from the facial transformation matrix
        head_vector = get_head_vector(face_rt)

        # face_origin_3d (simplified — use translation from the matrix)
        face_origin_3d = face_rt[:3, 3]

        # Run BlazeGaze inference
        eye_patch_batch = tf.cast(tf.expand_dims(eye_patch, 0), tf.float32) / 255.0
        hv_batch = tf.expand_dims(tf.constant(head_vector, dtype=tf.float32), 0)
        fo_batch = tf.expand_dims(tf.constant(face_origin_3d, dtype=tf.float32), 0)

        output = infer_fn(eye_patch_batch, hv_batch, fo_batch)
        if isinstance(output, list):
            gaze = output[-1].numpy()[0]
        else:
            gaze = output.numpy()[0]

        gx, gy = float(gaze[0]), float(gaze[1])

        # Accumulate per-second averages
        sec = int(timestamp_ms / 1000)
        if sec not in per_second:
            per_second[sec] = [0.0, 0.0, 0]
        per_second[sec][0] += gx
        per_second[sec][1] += gy
        per_second[sec][2] += 1

        if frame_idx % 30 == 0:
            print(f"  frame {frame_idx:4d}  t={timestamp_ms/1000:.1f}s  gaze=({gx:.3f}, {gy:.3f})")

        frame_idx += 1

    cap.release()
    landmarker.close()

    # Summary
    print(f"\n=== MediaPipe + BlazeGaze summary ===")
    print(f"  Frames processed : {frame_idx}")
    print(f"  Per-second averages:")
    print(f"  {'second':>6s} | {'avg X':>8s} | {'avg Y':>8s} | samples")
    print(f"  {'------':>6s}-+-{'--------':>8s}-+-{'--------':>8s}-+--------")
    for sec in sorted(per_second.keys()):
        sx, sy, n = per_second[sec]
        print(f"  {sec:5d}s  |  {sx/n:8.4f}  |  {sy/n:8.4f}  |  {n}")

    print(f"\n  Compare these values to the macgaze-replay (Vision-based) output.")
    print(f"  If X distinguishes left vs right here, MediaPipe is the fix.")


if __name__ == "__main__":
    main()
