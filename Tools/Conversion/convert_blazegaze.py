#!/usr/bin/env python3
"""
Convert BlazeGaze Keras model to CoreML format.

Usage (from Tools/Conversion/ with .venv activated):
    python convert_blazegaze.py

Input:  WebEyeTrack-upstream/python/webeyetrack/model_weights/blazegaze_mpiifacegaze.keras
Output: ../../Sources/MacGaze/Gaze/blazegaze.mlpackage

The model has 3 inputs (image, head_vector, face_origin_3d) and 1 output
(gaze_output: normalized screen x,y). See docs/blazegaze-model-signature.md.
"""

import os
import sys
import numpy as np

def main():
    # Paths relative to this script.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    upstream = os.path.join(script_dir, "WebEyeTrack-upstream")
    keras_path = os.path.join(
        upstream, "python", "webeyetrack", "model_weights",
        "blazegaze_mpiifacegaze.keras"
    )
    output_path = os.path.join(
        script_dir, "..", "..", "Sources", "MacGaze", "Gaze",
        "blazegaze.mlpackage"
    )

    if not os.path.exists(keras_path):
        print(f"ERROR: Keras model not found at {keras_path}")
        print("Clone WebEyeTrack first:")
        print("  git clone --depth=1 https://github.com/RedForestAI/WebEyeTrack.git WebEyeTrack-upstream")
        sys.exit(1)

    # 1. Load the Keras model.
    print(f"Loading Keras model from {keras_path}...")
    import tensorflow as tf
    model = tf.keras.models.load_model(keras_path, compile=False)
    print(f"  Model: {model.name}")
    print(f"  Inputs: {[(inp.name, inp.shape) for inp in model.inputs]}")
    print(f"  Outputs: {[(out.name, out.shape) for out in model.outputs]}")

    # 2. Convert to CoreML.
    print("\nConverting to CoreML...")
    import coremltools as ct

    # Define inputs explicitly so CoreML gets the right names + shapes.
    # The Keras model's input layers are named: image, head_vector, face_origin_3d.
    coreml_inputs = [
        ct.TensorType(name="image", shape=(1, 128, 512, 3)),
        ct.TensorType(name="head_vector", shape=(1, 3)),
        ct.TensorType(name="face_origin_3d", shape=(1, 3)),
    ]

    mlmodel = ct.convert(
        model,
        source="tensorflow",
        inputs=coreml_inputs,
        # Don't specify outputs — let coremltools auto-detect the final
        # tensor.  The Keras 3.x tensor names don't match layer names,
        # which breaks explicit output specification.
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.macOS15,
    )

    # Add metadata.
    mlmodel.author = "MacGaze project (converted from WebEyeTrack MIT-licensed model)"
    mlmodel.short_description = "BlazeGaze gaze estimation: eye crop + head pose → normalized (x, y) screen point"
    mlmodel.version = "1.0"

    # 3. Save.
    print(f"\nSaving to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mlmodel.save(output_path)
    print(f"  Saved. Size: {get_dir_size(output_path) / 1024:.0f} KB")

    # 4. Sanity-check: compare Keras vs CoreML on synthetic input.
    print("\nVerifying: Keras vs CoreML on synthetic input...")
    verify(model, mlmodel)

    print("\n✓ Conversion complete.")
    print(f"  CoreML model: {output_path}")
    print(f"  To compile to .mlmodelc for Swift:")
    print(f"    xcrun coremlc compile {output_path} {os.path.dirname(output_path)}")


def verify(keras_model, mlmodel):
    """Run the same synthetic input through both models, compare outputs."""
    np.random.seed(42)
    image = np.random.rand(1, 128, 512, 3).astype(np.float32)
    head_vector = np.array([[0.1, -0.2, 0.9]], dtype=np.float32)
    face_origin_3d = np.array([[-50.0, 30.0, 400.0]], dtype=np.float32)

    # Keras prediction.
    keras_out = keras_model.predict([image, head_vector, face_origin_3d], verbose=0)
    if isinstance(keras_out, list):
        keras_gaze = keras_out[-1]  # last output is gaze
    else:
        keras_gaze = keras_out

    # CoreML prediction.
    coreml_out = mlmodel.predict({
        "image": image,
        "head_vector": head_vector,
        "face_origin_3d": face_origin_3d,
    })
    # CoreML returns a dict keyed by output name.
    coreml_gaze = coreml_out.get("gaze_output",
                                  list(coreml_out.values())[-1])

    print(f"  Keras output:   {keras_gaze}")
    print(f"  CoreML output:  {coreml_gaze}")
    diff = np.abs(keras_gaze.flatten() - np.asarray(coreml_gaze).flatten())
    max_diff = float(np.max(diff))
    print(f"  Max abs diff:   {max_diff:.6f}")
    if max_diff < 1e-3:
        print("  ✓ Outputs match within tolerance.")
    else:
        print("  ⚠ Outputs differ — check conversion.")


def get_dir_size(path):
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total += os.path.getsize(fp)
    return total


if __name__ == "__main__":
    main()
