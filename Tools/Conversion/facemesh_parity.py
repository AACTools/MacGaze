#!/usr/bin/env python3
"""Phase 1 parity: does PINTO #032 face_mesh.mlmodel reproduce MediaPipe's
landmarks on the indices HomographyEyePatchExtractor uses?

Oracle  : MediaPipe FaceLandmarker (.task, 478 attention mesh, full-image
          normalized [0,1]) — same producer the verified pipeline used.
Under test: face_mesh.mlmodel (base 468, crop-pixel space) + our own crop.

Because our CoreML model is only the landmark net (no built-in face crop), we
derive a square face crop from the oracle's landmark bbox, sweep an expansion
factor, run the CoreML model, map its landmarks back to full-image normalized,
and report per-index pixel error. The expansion that minimises error on the
KEY_INDICES is the crop parameter we hardcode in Swift.

Note: oracle is 478-attention, model-under-test is 468-base; they diverge around
eyes/lips (attention refines those) but NOT on KEY_INDICES — judge success on
KEY_INDICES only.

Run (both deps on one Python that supports them):
    uv run --python 3.11 --with mediapipe --with coremltools \
        --with opencv-python --with numpy \
        python Tools/Conversion/facemesh_parity.py \
        /tmp/facemesh/032_FaceMesh/07_coreml/face_mesh.mlmodel ./demoPic.jpg
"""
import os
import sys
import numpy as np
import cv2
import coremltools as ct
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# Indices HomographyEyePatchExtractor consumes (see Phase 0.2).
KEY_INDICES = [4, 103, 150, 151, 195, 332, 379]
EXPANSIONS = [1.25, 1.5, 1.75, 2.0, 2.25]
NORMALIZERS = {"[0,1]": lambda a: a / 255.0, "[-1,1]": lambda a: a / 127.5 - 1.0}


def mediapipe_landmarks(image_rgb, task_model):
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=task_model),
        running_mode=mp_vision.RunningMode.IMAGE, num_faces=1,
    )
    with mp_vision.FaceLandmarker.create_from_options(opts) as lm:
        res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb))
    if not res.face_landmarks:
        raise SystemExit("Oracle found no face — use a clearer, centred face photo.")
    return np.array([[p.x, p.y] for p in res.face_landmarks[0]])  # (478,2) normalized


def coreml_landmarks(model, image_rgb, oracle_xy, expansion, normalize):
    H, W = image_rgb.shape[:2]
    xs, ys = oracle_xy[:, 0] * W, oracle_xy[:, 1] * H
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    side = max(xs.max() - xs.min(), ys.max() - ys.min()) * expansion
    x0, y0 = cx - side / 2, cy - side / 2

    # Square crop (clamped) → 192×192.
    xi, yi, s = int(round(x0)), int(round(y0)), int(round(side))
    xi, yi = max(0, xi), max(0, yi)
    s = min(s, W - xi, H - yi)
    crop = image_rgb[yi:yi + s, xi:xi + s]
    if crop.size == 0:
        return None
    crop192 = cv2.resize(crop, (192, 192)).astype(np.float32)

    inp = normalize(crop192).reshape(1, 192, 192, 3)
    out = model.predict({"input_1": inp})
    lm = np.asarray(out["conv2d_20"]).ravel().reshape(468, 3)[:, :2]  # crop-pixel 0..192

    # crop-pixel → full-image normalized
    full = np.empty_like(lm)
    full[:, 0] = (xi + (lm[:, 0] / 192.0) * s) / W
    full[:, 1] = (yi + (lm[:, 1] / 192.0) * s) / H
    return full


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    model_path, image_path = sys.argv[1], sys.argv[2]

    here = os.path.dirname(os.path.abspath(__file__))
    task_model = os.path.join(here, "WebEyeTrack-upstream", "python", "webeyetrack",
                              "model_weights", "face_landmarker_v2_with_blendshapes.task")
    if not os.path.exists(task_model):
        os.makedirs(os.path.dirname(task_model), exist_ok=True)
        print("Downloading MediaPipe face_landmarker.task ...")
        import urllib.request
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/latest/face_landmarker.task", task_model)

    bgr = cv2.imread(image_path)
    if bgr is None:
        raise SystemExit(f"Could not read image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]
    diag = (W ** 2 + H ** 2) ** 0.5

    oracle = mediapipe_landmarks(rgb, task_model)
    model = ct.models.MLModel(model_path)

    print(f"image {W}x{H}; oracle 478 landmarks; testing base-468 model")
    print(f"error = mean pixel distance on KEY_INDICES {KEY_INDICES}\n")
    best = None
    for norm_name, norm in NORMALIZERS.items():
        for exp in EXPANSIONS:
            pred = coreml_landmarks(model, rgb, oracle, exp, norm)
            if pred is None:
                continue
            d = np.linalg.norm((pred[KEY_INDICES] - oracle[KEY_INDICES]) *
                               [W, H], axis=1)
            mean_px, max_px = d.mean(), d.max()
            pct = 100 * mean_px / diag
            print(f"  norm={norm_name:7s} expand={exp:4.2f}  "
                  f"mean={mean_px:6.1f}px ({pct:4.1f}% diag)  max={max_px:6.1f}px")
            if best is None or mean_px < best[0]:
                best = (mean_px, norm_name, exp)

    print(f"\nBEST: norm={best[1]} expand={best[2]:.2f}  mean={best[0]:.1f}px")
    print("→ use these as the Swift crop params. <~1-2% of diagonal = PASS.")


if __name__ == "__main__":
    main()
