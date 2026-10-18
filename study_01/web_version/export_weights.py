# Created: 2026-10-18 17:04 (KST)
"""
Export the desktop version's trained MLP weights for the web version.

Reads ../desktop_version/mnist_mlp.npz and writes model_weights.js, which
defines a global MODEL_WEIGHTS object with base64-encoded float32 arrays.
A .js file (instead of .json) is used so index.html also works when opened
directly from disk (file://), where fetch() of local files is blocked.

Usage:
    python export_weights.py
"""

import base64
import json
import os
from datetime import datetime

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_PATH = os.path.join(BASE_DIR, "..", "desktop_version", "mnist_mlp.npz")
OUTPUT_PATH = os.path.join(BASE_DIR, "model_weights.js")


def encode(array):
    """Return a JSON-friendly dict with the array's shape and float32 data."""
    data = np.ascontiguousarray(array, dtype="<f4")  # Little-endian float32
    return {"shape": list(data.shape), "data": base64.b64encode(data.tobytes()).decode("ascii")}


def main():
    npz = np.load(SOURCE_PATH)
    n_layers = len([k for k in npz.files if k.startswith("w")])
    layers = [{"w": encode(npz[f"w{i}"]), "b": encode(npz[f"b{i}"])} for i in range(n_layers)]

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(f"// Created: {timestamp} (KST) by export_weights.py - do not edit by hand\n")
        f.write("const MODEL_WEIGHTS = ")
        json.dump({"layers": layers}, f)
        f.write(";\n")
    print(f"Exported {n_layers} layers to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
