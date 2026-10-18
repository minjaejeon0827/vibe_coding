<!-- Created: 2026-10-18 16:59 (KST) -->
# CLAUDE.md — web_version

This file provides guidance to Claude Code (claude.ai/code) when working in `web_version/`. The repository-wide rules in `../CLAUDE.md` also apply.

## Overview

The browser version of the handwritten digit recognizer. The user draws a digit (0-9) on a canvas and the page predicts it live. It is plain HTML, CSS, and JavaScript, with no framework, build step, server, or npm dependencies. Open `index.html` directly (double-click it); the page works from `file://`.

Inference runs entirely in the browser using the model trained by `../desktop_version/`. The web version does no training of its own.

## Commands

```
python export_weights.py   # regenerate model_weights.js from ../desktop_version/mnist_mlp.npz
```

Run `export_weights.py` again whenever the desktop model is retrained. It needs Python 3.12 with NumPy (see `../desktop_version/CLAUDE.md`).

There is no test suite. To verify the recognition pipeline, upscale MNIST test images to 280x280 and run them through both Python (`preprocess()` and the model in `../desktop_version/digit_recognizer.py`) and JavaScript (`recognizer.js`, run in Node.js, which is installed). The two currently agree on 100% of 500 test images.

## Architecture

The three scripts are classic (non-module) scripts loaded in order. They share globals, because ES modules and `fetch()` are blocked on `file://`:

1. `model_weights.js`: **generated** by `export_weights.py`; do not edit it by hand. It defines `MODEL_WEIGHTS`, an object holding the MLP layers (784-256-128-10) as base64-encoded little-endian float32 arrays.
2. `recognizer.js`: DOM-free logic. It defines `MLP` (ReLU, softmax inference) and `preprocess()`. When `module` exists, it also exports them for Node.js so the logic can be tested without a browser.
3. `app.js`: the UI. It handles pointer drawing (mouse, touch, and pen), clears the canvas on right-click or the Clear button, and draws probability bars.

**Matching the desktop preprocessing is critical.** `preprocess()` must produce exactly the same input as `preprocess()` in the desktop version. That function crops to the bounding box, scales the longest side to 20 px, places the result in a 28x28 frame, and shifts it (with wrap-around) so its center of mass is at the center. The resize step in `resizeLanczos()` re-implements Pillow's `Image.LANCZOS`: two separable passes, a filter widened by the scale factor, and 8-bit rounding after each pass. An earlier area-average resize agreed with Python on only 98% of images and was 2 points less accurate, so do not swap in canvas `drawImage` scaling.

The canvas buffer is 280x280 regardless of its CSS display size. `toCanvasPoint()` maps pointer coordinates into that buffer, and the brush radius (10 px) matches the desktop version.
