<!-- Created: 2026-10-18 16:59 (KST) -->
# CLAUDE.md — desktop_version

This file provides guidance to Claude Code (claude.ai/code) when working in `desktop_version/`. The repository-wide rules in `../CLAUDE.md` also apply.

## Overview

A Tkinter desktop app: the user draws a digit (0-9) and a NumPy MLP trained on MNIST predicts it live. All logic is in `digit_recognizer.py`. `digit_recognizer.pyw` is the launcher that runs it when the file is double-clicked in Windows Explorer.

The Korean notes in the `digit_recognizer.py` module docstring were added by the user; leave them in place. The docstring is a raw string (`r"""`) so the Windows paths in those notes do not trigger invalid-escape warnings.

## Environment

- Windows, with both Python 3.13 and 3.12 installed. Only **3.12** has the dependencies (`numpy`, `Pillow`; `tkinter` comes with Python). No ML framework is installed, and none is used.
- `python` on PATH runs 3.12. The Windows launcher (`py`/`pyw`) defaults to 3.13, so `digit_recognizer.pyw` begins with a `#! python3.12` shebang. Removing that line breaks double-click launching.
- `pythonw` has no console, so the `.pyw` launcher shows exceptions in a Tk message box.

## Commands

Run these from `desktop_version/`. Paths are resolved relative to the script, so running from another folder also works.

```
python digit_recognizer.py               # open the GUI (trains first if mnist_mlp.npz is missing)
python digit_recognizer.py --retrain     # retrain, then open the GUI
python digit_recognizer.py --train-only  # retrain and exit (~1-2 min, prints test accuracy per epoch)
```

There is no test suite, linter, or build step. To check the recognition pipeline without the GUI, upscale MNIST test images to the 280x280 canvas size and run them through `preprocess()` and then the model. This check currently scores ~98.6%; the model scores ~98.4% on the raw test set.

## Architecture

`digit_recognizer.py` runs this pipeline:

1. **Data**: `load_mnist()` downloads the gzipped IDX files from the Google Cloud Storage MNIST mirror into `data/` the first time they are needed, then parses them with `load_idx()`.
2. **Model**: `MLP` (784-256-128-10, ReLU, softmax) implements `forward` and `backward` by hand. `train_model()` contains its own Adam optimizer loop, uses `random_shift` augmentation and a per-epoch learning-rate decay, and saves weights to `mnist_mlp.npz` as `w{i}` and `b{i}` arrays. `MLP.load` infers the layer count from those keys.
3. **Preprocessing**: `preprocess()` reproduces how MNIST digits were prepared. It crops to the bounding box, scales the longest side to 20 px, pastes the result into a 28x28 frame, and shifts it so the center of mass is at the center. Model accuracy on drawn input depends on this step matching MNIST.
4. **GUI**: `DigitApp` draws every stroke twice: on the visible `tk.Canvas` and on an off-screen PIL `Image`. Prediction reads the PIL image, not the canvas. Any change to drawing or clearing must update both.

`data/`, `mnist_mlp.npz`, and `__pycache__/` are generated files. Deleting `mnist_mlp.npz` triggers retraining on the next launch.
