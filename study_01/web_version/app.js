// Created: 2026-10-18 17:04 (KST)
// Browser UI: drawing on the canvas and showing predictions.
// Depends on the globals MODEL_WEIGHTS (model_weights.js) and MLP/preprocess (recognizer.js).

const BRUSH_RADIUS = 10; // Same as the desktop version, in canvas pixels

const canvas = /** @type {HTMLCanvasElement} */ (document.getElementById("pad"));
const ctx = canvas.getContext("2d", { willReadFrequently: true });
const digitEl = document.getElementById("digit");
const confidenceEl = document.getElementById("confidence");
const barsEl = document.getElementById("bars");

const model = new MLP(MODEL_WEIGHTS);
let lastPoint = null;
let predictQueued = false;

// One bar per digit showing its predicted probability
const bars = Array.from({ length: 10 }, (_, d) => {
  const li = document.createElement("li");
  li.innerHTML = `<span>${d}</span><span class="track"><span class="fill"></span></span><span class="pct"></span>`;
  barsEl.appendChild(li);
  return {
    li,
    fill: /** @type {HTMLElement} */ (li.querySelector(".fill")),
    pct: /** @type {HTMLElement} */ (li.querySelector(".pct")),
  };
});

function clearPad() {
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  digitEl.textContent = "?";
  confidenceEl.textContent = "Draw a digit";
  showProbabilities(null);
}

/** Map a pointer event to canvas pixel coordinates (the canvas may be scaled by CSS). */
function toCanvasPoint(event) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: ((event.clientX - rect.left) * canvas.width) / rect.width,
    y: ((event.clientY - rect.top) * canvas.height) / rect.height,
  };
}

function paint(from, to) {
  ctx.strokeStyle = "#fff";
  ctx.lineWidth = BRUSH_RADIUS * 2;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.beginPath();
  ctx.moveTo(from.x, from.y);
  ctx.lineTo(to.x, to.y);
  ctx.stroke();
}

/** Run at most one prediction per animation frame while drawing. */
function schedulePredict() {
  if (predictQueued) return;
  predictQueued = true;
  requestAnimationFrame(() => {
    predictQueued = false;
    predict();
  });
}

function predict() {
  const { data } = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const gray = new Uint8Array(canvas.width * canvas.height);
  for (let i = 0; i < gray.length; i++) gray[i] = data[i * 4]; // Red channel (drawing is grayscale)

  const input = preprocess(gray, canvas.width, canvas.height);
  if (!input) return;
  const probs = model.predict(input);
  const best = probs.indexOf(Math.max(...probs));
  digitEl.textContent = String(best);
  confidenceEl.textContent = `Confidence: ${(probs[best] * 100).toFixed(1)}%`;
  showProbabilities(probs, best);
}

function showProbabilities(probs, best = -1) {
  bars.forEach(({ li, fill, pct }, d) => {
    const p = probs ? probs[d] : 0;
    fill.style.width = `${p * 100}%`;
    pct.textContent = probs ? `${(p * 100).toFixed(1)}%` : "";
    li.classList.toggle("best", d === best);
  });
}

canvas.addEventListener("pointerdown", (event) => {
  if (event.button === 2) return; // Right click clears (see contextmenu below)
  canvas.setPointerCapture(event.pointerId);
  lastPoint = toCanvasPoint(event);
  paint(lastPoint, lastPoint);
  schedulePredict();
});

canvas.addEventListener("pointermove", (event) => {
  if (!lastPoint) return;
  const point = toCanvasPoint(event);
  paint(lastPoint, point);
  lastPoint = point;
  schedulePredict();
});

["pointerup", "pointercancel"].forEach((type) =>
  canvas.addEventListener(type, () => {
    lastPoint = null;
  })
);

canvas.addEventListener("contextmenu", (event) => {
  event.preventDefault();
  clearPad();
});

document.getElementById("clear").addEventListener("click", clearPad);

clearPad();
