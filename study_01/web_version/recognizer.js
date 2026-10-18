// Created: 2026-10-18 17:04 (KST)
// DOM-free recognition logic: weight decoding, MLP inference, and
// preprocessing that mirrors preprocess() in ../desktop_version/digit_recognizer.py.
// It is kept separate from app.js so it can be tested in Node.js.

/** Decode a base64 little-endian float32 array exported by export_weights.py. */
function decodeArray({ shape, data }) {
  const bytes = Uint8Array.from(atob(data), (c) => c.charCodeAt(0));
  return { shape, values: new Float32Array(bytes.buffer) };
}

/** Fully connected network with ReLU hidden layers and a softmax output. */
class MLP {
  constructor(weights) {
    this.layers = weights.layers.map((layer) => ({ w: decodeArray(layer.w), b: decodeArray(layer.b) }));
  }

  /** Return the 10 class probabilities for a 784-element input. */
  predict(input) {
    let x = input;
    this.layers.forEach(({ w, b }, index) => {
      const [nIn, nOut] = w.shape;
      const out = Float32Array.from(b.values);
      for (let i = 0; i < nIn; i++) {
        const xi = x[i];
        if (xi === 0) continue; // Inputs are mostly zero, so skip them
        const row = i * nOut;
        for (let j = 0; j < nOut; j++) out[j] += xi * w.values[row + j];
      }
      const isLast = index === this.layers.length - 1;
      if (!isLast) for (let j = 0; j < nOut; j++) out[j] = Math.max(0, out[j]);
      x = out;
    });
    return softmax(x);
  }
}

function softmax(logits) {
  const max = Math.max(...logits);
  const exp = Array.from(logits, (v) => Math.exp(v - max));
  const sum = exp.reduce((a, b) => a + b, 0);
  return exp.map((v) => v / sum);
}

function lanczos(x) {
  if (x === 0) return 1;
  if (x <= -3 || x >= 3) return 0;
  const px = Math.PI * x;
  return (3 * Math.sin(px) * Math.sin(px / 3)) / (px * px);
}

/**
 * Resample one axis with a Lanczos-3 filter the way Pillow does: when
 * downscaling, the filter is widened by the scale factor (antialiasing), and
 * results are rounded and clamped to 8 bits after each pass.
 */
function resamplePass(src, srcW, srcH, dstLen, horizontal) {
  const srcLen = horizontal ? srcW : srcH;
  const scale = srcLen / dstLen;
  const filterScale = Math.max(scale, 1);
  const support = 3 * filterScale;
  const dstW = horizontal ? dstLen : srcW;
  const dstH = horizontal ? srcH : dstLen;
  const dst = new Float32Array(dstW * dstH);

  for (let i = 0; i < dstLen; i++) {
    const center = (i + 0.5) * scale;
    const start = Math.max(0, Math.trunc(center - support + 0.5));
    const end = Math.min(srcLen, Math.trunc(center + support + 0.5));
    const weights = [];
    let total = 0;
    for (let j = start; j < end; j++) {
      const w = lanczos((j - center + 0.5) / filterScale);
      weights.push(w);
      total += w;
    }
    for (let k = 0; k < weights.length; k++) weights[k] /= total || 1;

    const lines = horizontal ? srcH : srcW;
    for (let line = 0; line < lines; line++) {
      let sum = 0;
      for (let k = 0; k < weights.length; k++) {
        const j = start + k;
        sum += weights[k] * (horizontal ? src[line * srcW + j] : src[j * srcW + line]);
      }
      const value = Math.min(255, Math.max(0, Math.round(sum)));
      if (horizontal) dst[line * dstW + i] = value;
      else dst[i * dstW + line] = value;
    }
  }
  return dst;
}

/** Resize a grayscale image (values 0-255) like Pillow's Image.LANCZOS. */
function resizeLanczos(src, srcW, srcH, dstW, dstH) {
  const tmp = resamplePass(src, srcW, srcH, dstW, true);
  return resamplePass(tmp, dstW, srcH, dstH, false);
}

/**
 * Convert a grayscale image (white digit on black, values 0-255) to a
 * 784-element input in [0, 1]: crop to the digit, fit it into a 20x20 box
 * keeping the aspect ratio, then place it in a 28x28 frame centered by its
 * center of mass. Returns null for an empty image.
 */
function preprocess(gray, width, height) {
  let minX = width, minY = height, maxX = -1, maxY = -1;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (gray[y * width + x] > 0) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }
  if (maxX < 0) return null;

  const cropW = maxX - minX + 1;
  const cropH = maxY - minY + 1;
  const crop = new Float32Array(cropW * cropH);
  for (let y = 0; y < cropH; y++) {
    for (let x = 0; x < cropW; x++) crop[y * cropW + x] = gray[(y + minY) * width + x + minX];
  }

  const scale = 20 / Math.max(cropW, cropH);
  const newW = Math.max(1, Math.round(cropW * scale));
  const newH = Math.max(1, Math.round(cropH * scale));
  const digit = resizeLanczos(crop, cropW, cropH, newW, newH);

  const frame = new Float32Array(28 * 28);
  const offX = Math.floor((28 - newW) / 2);
  const offY = Math.floor((28 - newH) / 2);
  for (let y = 0; y < newH; y++) {
    for (let x = 0; x < newW; x++) frame[(y + offY) * 28 + x + offX] = digit[y * newW + x] / 255;
  }

  // Shift (with wrap-around, like np.roll) so the center of mass sits at the center
  let total = 0, sumX = 0, sumY = 0;
  for (let y = 0; y < 28; y++) {
    for (let x = 0; x < 28; x++) {
      const v = frame[y * 28 + x];
      total += v;
      sumX += x * v;
      sumY += y * v;
    }
  }
  const shiftX = Math.round(13.5 - sumX / total);
  const shiftY = Math.round(13.5 - sumY / total);
  const result = new Float32Array(28 * 28);
  for (let y = 0; y < 28; y++) {
    for (let x = 0; x < 28; x++) {
      const ny = (((y + shiftY) % 28) + 28) % 28;
      const nx = (((x + shiftX) % 28) + 28) % 28;
      result[ny * 28 + nx] = frame[y * 28 + x];
    }
  }
  return result;
}

// Export for Node.js tests. Object.assign (rather than `module.exports =`) keeps this
// a classic script for type checkers, so app.js still sees MLP and preprocess as globals.
// @ts-ignore: `module` only exists in Node.js
if (typeof module !== "undefined") Object.assign(module.exports, { MLP, preprocess, resizeLanczos });
