#!/usr/bin/env python3
"""Probe PINTO #032 face_mesh.mlmodel to nail the pre/post-processing convention
before we write the Swift landmarker (Phase 1 of coreml-facemesh-plan.md).

Prints exact input/output shapes, and — given a face photo — runs the model
under three input normalizations and prints each output's value range, so we
can read off:
  • which input normalization is correct  (which one yields sane landmarks)
  • the output coordinate space            (are x,y in 0..192 or 0..1?)
  • confirmation that conv2d_20 == 1404 (468*3) and conv2d_30 == face score

Usage:
    pip install coremltools pillow numpy
    python3 facemesh_probe.py /path/to/face_mesh.mlmodel [face.jpg]

Tips: use a real, roughly-centred, upright face crop for face.jpg so the
landmark ranges are meaningful.
"""
import sys
import numpy as np
import coremltools as ct


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    model_path = sys.argv[1]
    model = ct.models.MLModel(model_path)
    spec = model.get_spec()

    def shape_of(feature):
        kind = feature.type.WhichOneof("Type")
        if kind == "multiArrayType":
            return list(feature.type.multiArrayType.shape)
        return kind

    print("== INPUTS ==")
    for f in spec.description.input:
        print(f"  {f.name}: {shape_of(f)}")
    print("== OUTPUTS ==")
    for f in spec.description.output:
        print(f"  {f.name}: {shape_of(f)}")

    if len(sys.argv) < 3:
        print("\n(no image given — pass a face photo to probe the coordinate/"
              "normalization convention)")
        return

    from PIL import Image
    input_name = spec.description.input[0].name
    img = Image.open(sys.argv[2]).convert("RGB").resize((192, 192))
    hwc = np.asarray(img).astype(np.float32)  # 192×192×3, 0..255

    candidates = [
        ("[0,1]  (÷255)", hwc / 255.0),
        ("[-1,1] (÷127.5-1)", hwc / 127.5 - 1.0),
        ("[0,255] raw", hwc),
    ]
    for name, x in candidates:
        inp = x.reshape(1, 192, 192, 3)
        try:
            out = model.predict({input_name: inp})
        except Exception as exc:  # noqa: BLE001
            print(f"\n[{name}] predict failed: {exc}")
            continue
        print(f"\n== normalization {name} ==")
        for k, v in out.items():
            a = np.asarray(v).ravel()
            print(f"  {k}: size={a.size}  min={a.min():.3f}  "
                  f"max={a.max():.3f}  first6={np.round(a[:6], 3)}")


if __name__ == "__main__":
    main()
