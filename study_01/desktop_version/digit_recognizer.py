r"""
Handwritten digit recognizer.

Draw a digit (0-9) with the mouse on the canvas and the program predicts it.
A small neural network (multi-layer perceptron) written in pure NumPy is
trained on the MNIST dataset the first time the program runs; the trained
weights are cached in a file so later launches start instantly.

Usage:
    python digit_recognizer.py            # train if needed, then open the GUI
    python digit_recognizer.py --retrain  # force retraining before opening the GUI
    python digit_recognizer.py --train-only
    
참고:
손글씨 숫자 인식 프로그램을 만들어 실행했습니다. 지금 화면에 "Handwritten Digit Recognizer" 창이 열려 있을 거예요. 코드와 주석은 모두 영어로 작성했습니다.

사용법                                                                                                             
- 왼쪽 검은 캔버스에 마우스 왼쪽 버튼으로 0~9 숫자를 그리면, 그리는 동안 바로 인식합니다.
- 오른쪽에 인식한 숫자, 확신도(%), 숫자별 확률 막대가 표시됩니다.
- 마우스 오른쪽 클릭이나 Clear 버튼으로 캔버스를 지웁니다.

구성 (digit_recognizer.py 파일 하나)
- 학습: 처음 실행할 때 MNIST 손글씨 데이터 6만 장을 data/ 폴더에 내려받아 신경망을 학습시킵니다. 신경망은 NumPy만으로 직접 구현했고, 별도 머신러닝 라이브러리는 설치하지 않았습니다.
- 정확도: MNIST 테스트 데이터 1만 장에서 **98.4%**입니다.
- 저장: 학습한 가중치는 mnist_mlp.npz에 저장되므로 다음 실행부터는 바로 창이 뜹니다.
- 전처리: 그린 숫자를 MNIST와 같은 방식으로 다듬어서 인식합니다. 숫자 부분만 잘라 20×20 크기에 맞추고, 무게중심이 28×28 이미지의 가운데에 오도록 옮깁니다.
- 확인: MNIST 테스트 이미지 500장을 캔버스 크기로 키워 이 전처리 과정에 넣어 보니 98.6%를 맞혔습니다. 다만 실제로 마우스로 그려서 테스트해 보지는 않았습니다. 마우스로 그린 글씨는 MNIST와 모양이 달라서 정확도가 이보다 낮을 수 있습니다.

실행 명령
python digit_recognizer.py               # GUI 실행
python digit_recognizer.py --retrain     # 다시 학습한 후 GUI 실행
python digit_recognizer.py --train-only  # 학습만 실행

이제 윈도우 탐색기에서 **D:\minjae\vibe_coding\study_01\digit_recognizer.pyw**를 더블클릭하면 숫자 인식 프로그램이 실행됩니다. 더블클릭과 같은 방식으로 직접 열어 보니 "Handwritten Digit Recognizer" 창이 떴습니다. 지금 그 창이 열려 있을 거예요.

만든 파일: digit_recognizer.pyw
- .pyw 파일은 검은 콘솔 창 없이 프로그램 창만 엽니다.
- 기존 digit_recognizer.py를 불러와 실행하는 작은 실행 파일이라서, 프로그램 본체 코드는 그대로입니다.
- 콘솔이 없어서 오류 내용이 보이지 않으므로, 오류가 나면 메시지 창으로 보여 주게 했습니다.

고친 문제: 이 PC에는 Python 3.13과 3.12가 함께 설치되어 있습니다. 더블클릭하면 기본으로 3.13이 실행되는데, 3.13에는 numpy가 없어서 프로그램이 바로 꺼졌습니다. 그래서 파일 맨 윗줄에 #! python3.12를 넣어 항상 3.12로 실행되게 했습니다.

바탕화면에서 바로 실행하고 싶으시면, 이 파일을 마우스 오른쪽 버튼으로 클릭하고 **보내기 → 바탕 화면(바로 가기 만들기)**를 선택하세요.

I created D:\minjae\vibe_coding\study_01\CLAUDE.md. It covers four things:

- Project overview: what the program does, plus the rule that all code and comments are written in English.
- Environment notes: both Python 3.13 and 3.12 are installed, and only 3.12 has numpy and Pillow. If the #! python3.12 line at the top of the .pyw file is removed, double-clicking it stops working.
- Commands: the three ways to run the program. There is no test suite, linter or build step, so I described the check that runs MNIST test images through the preprocessing step instead.
- Architecture: the data → model → preprocessing → GUI flow, and two things that are easy to miss:
  - Strokes are drawn both on the visible canvas and on a hidden image, and prediction reads only the hidden image.
  - Recognition accuracy depends on the preprocessing matching how MNIST images were made.

Something I found: the module docstring of digit_recognizer.py (lines 14–43) now contains my earlier Korean explanation, which looks like you pasted it in. That doesn't follow the "all code and comments in English" rule, but I left it as is and noted in CLAUDE.md that these notes are yours and should be kept. If you'd like, I can move them into a separate README.md or translate them into English.

The folder has no README, Cursor rules or Copilot rules, and there are no Codex or Gemini settings to import.

앞으로 새로 만드는 모든 파일의 맨 위에 만든 날짜와 시간을 주석으로 넣도록 CLAUDE.md에 규칙을 추가했습니다. 이 프로젝트에서 작업하는 다음 세션에서도 이 규칙이 그대로 적용됩니다.

- 형식: 파일 종류에 맞는 주석으로 씁니다. 예를 들어 Python 파일에는 # Created: 2026-10-18 16:54 (KST)를, Markdown이나 HTML 파일에는 <!-- Created: ... -->를 넣습니다.
- 시간: 추측하지 않고 Get-Date로 실제 현재 시간을 확인해서 넣습니다.

이 규칙은 이 프로젝트(study_01)에서만 적용됩니다. 다른 프로젝트에서도 똑같이 하길 원하시면 말씀해 주세요. 그리고 이미 있는 파일(digit_recognizer.py, .pyw, CLAUDE.md)에는 주석을 넣지 않았습니다. 원하시면 이 파일들에도 추가해 드릴게요.

두 폴더를 만들고, 기존 데스크톱 프로그램을 desktop_version으로 옮겼습니다. 옮긴 위치에서도 학습된 모델이 정상적으로 불러와지는 것을 확인했습니다.

study_01/
├── CLAUDE.md          ← 전체 개요 + 공통 규칙
├── desktop_version/
│   ├── CLAUDE.md      ← 새로 만듦
│   ├── digit_recognizer.py / .pyw
│   ├── mnist_mlp.npz
│   └── data/
└── web_version/
    └── CLAUDE.md      ← 새로 만듦

- 최상위 CLAUDE.md: 두 버전의 개요와 공통 규칙만 남겼습니다. 공통 규칙은 영어로 코드 작성하기, 새 파일에 날짜와 시간 주석 넣기입니다.
- desktop_version/CLAUDE.md: 기존에 최상위에 있던 내용(실행 명령, 구조, Python 3.12 관련 주의사항)을 옮겼습니다.
- web_version/CLAUDE.md: 아직 코드도 없고 사용할 기술(HTML/JS, React 등)도 정하지 않았다고 적었습니다. 그래서 프로젝트를 시작하기 전에 기술을 먼저 확인하도록 했습니다. 웹 버전에서도 데스크톱 버전의 학습된 모델을 그대로 쓸 수 있고, 숫자 이미지를 다듬는 전처리를 데스크톱 버전과 똑같이 맞춰야 한다는 점도 적었습니다.
- 새로 만든 두 CLAUDE.md 파일의 맨 위에는 <!-- Created: 2026-10-18 16:59 (KST) --> 주석을 넣었습니다.

알아 두실 점
- 이제 실행 파일은 D:\minjae\vibe_coding\study_01\desktop_version\digit_recognizer.pyw에 있습니다. 전에 바탕화면 바로 가기를 만드셨다면 그 바로 가기는 더 이상 작동하지 않으니 다시 만들어 주세요.
- 코드 맨 위 설명에 붙여 넣으신 한국어 메모 속 D:\minjae... 경로의 \ 문자 때문에 실행할 때 SyntaxWarning 경고가 뜹니다. 실행에는 아무 문제가 없습니다. 원하시면 메모 내용은 그대로 두고 경고만 없애 드릴게요.

두 가지 모두 끝났습니다. 지금 기본 브라우저에 웹 버전이 열려 있고, 데스크톱 버전 창도 함께 떠 있을 거예요.

1. 데스크톱 버전 클릭 실행

D:\minjae\vibe_coding\study_01\desktop_version\digit_recognizer.pyw를 더블클릭하면 실행됩니다. 이 파일은 폴더를 옮긴 뒤에도 그대로 동작해서 따로 고칠 것이 없었습니다. 더블클릭과 같은 방식으로 직접 열어 보니 창이 정상적으로 떴습니다.

2. 웹 버전 개발

D:\minjae\vibe_coding\study_01\web_version\index.html을 더블클릭하면 브라우저에서 바로 실행됩니다.

사용할 기술은 아직 정하지 않았었기 때문에, 가장 간단한 순수 HTML/CSS/JavaScript로 만들었습니다. 설치할 것도, 서버도, 빌드 과정도 필요 없습니다.

- 모델: 새로 학습시키지 않고, 데스크톱 버전에서 학습한 모델을 그대로 가져와 브라우저 안에서 인식합니다.
- 사용법: 데스크톱 버전과 같습니다. 그리면서 바로 인식하고, 숫자별 확률 막대를 보여 주며, 마우스 오른쪽 클릭이나 Clear 버튼으로 지웁니다. 터치와 펜으로도 그릴 수 있습니다.
- 파일: 모든 새 파일의 맨 위에 만든 날짜와 시간 주석을 넣었습니다.
  - index.html, style.css, app.js: 화면
  - recognizer.js: 인식 로직
  - export_weights.py: 모델을 웹용 파일로 내보내는 스크립트
  - model_weights.js: 내보낸 모델 파일

검증 결과
- MNIST 테스트 이미지 500장으로 Python(데스크톱)과 JavaScript(웹)의 결과를 비교했습니다.
  - 처음에는 두 버전의 답이 98%만 같았습니다. 이미지 크기를 줄이는 방식이 서로 달랐기 때문입니다.
  - 웹 버전도 데스크톱 버전과 똑같은 방식으로 줄이도록 고쳤더니, 두 버전의 답이 100% 일치했습니다. 정확도는 98.6%입니다.
- 화면 없이 브라우저를 실행해 캔버스에 0, 1, 7을 그려 보니 모두 맞게 인식했습니다. 확신도는 99.8~99.9%였습니다.
- 다만 실제 사람이 마우스로 직접 그려서 테스트해 보지는 않았습니다.

알아 두실 점
- 데스크톱 버전에서 모델을 다시 학습시키면 python web_version\export_weights.py를 실행해야 웹 버전에도 새 모델이 적용됩니다.
- web_version/CLAUDE.md와 최상위 CLAUDE.md에 웹 버전의 구조와 사용법을 적어 두었습니다.

웹 버전을 로컬 서버로 실행하고 기본 브라우저에서 열었습니다. 지금 브라우저에 http://127.0.0.1:8000/ 페이지가 떠 있을 거예요.

- 서버 확인: 페이지에 필요한 파일 5개(HTML, CSS, JS, 모델 파일)가 모두 정상적으로 전달되는 것을 확인했습니다. 화면 동작은 지난번에 화면 없는 브라우저로 미리 확인해 두었는데, 0, 1, 7을 그려 보니 모두 맞게 인식했습니다.
- 사용법: 검은 상자에 숫자를 그리면 오른쪽에 인식한 숫자와 확률 막대가 나옵니다. 마우스 오른쪽 클릭이나 Clear 버튼으로 지웁니다.
- 주소: 브라우저 창을 닫았다면 주소창에 http://127.0.0.1:8000/을 다시 입력하면 됩니다. 서버는 이 컴퓨터에서만 접속할 수 있게 열어 두었습니다.
- 서버 종료: 서버는 백그라운드에서 계속 실행 중입니다. 끄고 싶으시면 말씀해 주세요.

서버 없이 web_version\index.html을 더블클릭해도 똑같이 동작합니다. 그리다가 인식이 잘 안 되는 숫자가 있으면 알려 주세요.

No type checker is installed and the project has no config for one, so I'll run them one-off with npx (nothing is added to the project).

  Ran 5 shell commands
Loaded CLAUDE.md
Loaded desktop_version\CLAUDE.md

The project has no type checker set up: no tsconfig, mypy or pyright config, and CLAUDE.md says there's no linter or build step. So I ran pyright on the Python and tsc --checkJs on the web JS one time through npx, without adding anything to the project. Both report errors, but none of them is a real bug. Each one comes from the checker missing information, not from the code doing the wrong thing.

Python (desktop_version/digit_recognizer.py): 3 errors, 22 warnings
- Lines 292–293, float * None: backward() fills its gradient lists with [None] * n (line 236) and then sets every slot. Pyright can't see that, so it treats g as possibly None. Not a bug.
- Line 324, Image.LANCZOS "not a known attribute": the attribute exists in your installed Pillow (12.3.0) and works at runtime. The stubs pyright bundles just don't list it. Image.Resampling.LANCZOS would satisfy the checker without changing behavior.
- 22 "unsupported escape sequence" warnings (lines 34–124): these come from the D:\minjae\... Windows paths in your Korean notes in the docstring. CLAUDE.md says to leave those notes as they are, and the warning doesn't affect running the app.

JS (web_version/): 15 errors
- getContext, width, height and style "not on HTMLElement/Element": getElementById and querySelector return a generic element type, and the code has no JSDoc saying which kind of element it is. The page works fine in the browser.
- MLP and preprocess "not found" in app.js: recognizer.js ends with a module.exports line so it can also be tested in Node. That makes tsc treat the file as a module, so it doesn't see these names as globals. In the browser they are globals, because index.html loads the files as plain <script> tags.
- Buffer "not found" (recognizer.js:10): this is the Node-only fallback branch, and Node type definitions aren't installed.

Nothing needs fixing for the code to work. If you want both checks to come back clean, I can add small annotations and switch to Image.Resampling.LANCZOS.
"""

import argparse
import gzip
import os
import sys
import tkinter as tk
import urllib.request

import numpy as np
from PIL import Image, ImageDraw

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_PATH = os.path.join(BASE_DIR, "mnist_mlp.npz")

MNIST_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def download_mnist():
    """Download the MNIST files into DATA_DIR if they are not present yet."""
    os.makedirs(DATA_DIR, exist_ok=True)
    for filename in MNIST_FILES.values():
        path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(path):
            print(f"Downloading {filename} ...")
            urllib.request.urlretrieve(MNIST_URL + filename, path)


def load_idx(filename):
    """Read a gzipped IDX file and return its contents as a NumPy array."""
    with gzip.open(os.path.join(DATA_DIR, filename), "rb") as f:
        data = f.read()
    magic = int.from_bytes(data[0:4], "big")
    if magic == 2051:  # Image file: count, rows, cols
        count, rows, cols = (int.from_bytes(data[i:i + 4], "big") for i in (4, 8, 12))
        return np.frombuffer(data, np.uint8, offset=16).reshape(count, rows * cols)
    if magic == 2049:  # Label file: count
        return np.frombuffer(data, np.uint8, offset=8)
    raise ValueError(f"Unknown IDX magic number {magic} in {filename}")


def load_mnist():
    """Return (x_train, y_train, x_test, y_test) with pixels scaled to [0, 1]."""
    download_mnist()
    x_train = load_idx(MNIST_FILES["train_images"]).astype(np.float32) / 255.0
    y_train = load_idx(MNIST_FILES["train_labels"]).astype(np.int64)
    x_test = load_idx(MNIST_FILES["test_images"]).astype(np.float32) / 255.0
    y_test = load_idx(MNIST_FILES["test_labels"]).astype(np.int64)
    return x_train, y_train, x_test, y_test


def random_shift(batch, max_shift=2):
    """Augment a batch by shifting each 28x28 image a few pixels at random."""
    images = batch.reshape(-1, 28, 28)
    shifted = np.empty_like(images)
    shifts = np.random.randint(-max_shift, max_shift + 1, size=(len(images), 2))
    for i, (dy, dx) in enumerate(shifts):
        shifted[i] = np.roll(images[i], (dy, dx), axis=(0, 1))
    return shifted.reshape(len(images), -1)


# ---------------------------------------------------------------------------
# Neural network (NumPy MLP)
# ---------------------------------------------------------------------------

class MLP:
    """Fully connected network with ReLU hidden layers and a softmax output."""

    def __init__(self, layer_sizes=(784, 256, 128, 10), seed=0):
        rng = np.random.default_rng(seed)
        self.weights = []
        self.biases = []
        for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            # He initialization works well with ReLU activations
            self.weights.append(rng.normal(0, np.sqrt(2.0 / n_in), (n_in, n_out)).astype(np.float32))
            self.biases.append(np.zeros(n_out, dtype=np.float32))

    @property
    def params(self):
        return self.weights + self.biases

    def forward(self, x):
        """Run the network and return (probabilities, cached activations)."""
        activations = [x]
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            x = np.maximum(0, x @ w + b)
            activations.append(x)
        logits = x @ self.weights[-1] + self.biases[-1]
        logits -= logits.max(axis=1, keepdims=True)  # Numerical stability
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        return probs, activations

    def backward(self, probs, activations, labels):
        """Return gradients of the cross-entropy loss w.r.t. all parameters."""
        n = len(labels)
        delta = probs.copy()
        delta[np.arange(n), labels] -= 1
        delta /= n

        grad_w = [np.empty(0)] * len(self.weights)
        grad_b = [np.empty(0)] * len(self.biases)
        for i in reversed(range(len(self.weights))):
            grad_w[i] = activations[i].T @ delta
            grad_b[i] = delta.sum(axis=0)
            if i > 0:
                # Propagate through the ReLU of the previous layer
                delta = (delta @ self.weights[i].T) * (activations[i] > 0)
        return grad_w + grad_b

    def predict_proba(self, x):
        return self.forward(x)[0]

    def save(self, path):
        arrays = {f"w{i}": w for i, w in enumerate(self.weights)}
        arrays.update({f"b{i}": b for i, b in enumerate(self.biases)})
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path):
        data = np.load(path)
        model = cls.__new__(cls)
        n_layers = len([k for k in data.files if k.startswith("w")])
        model.weights = [data[f"w{i}"] for i in range(n_layers)]
        model.biases = [data[f"b{i}"] for i in range(n_layers)]
        return model


def accuracy(model, x, y):
    return float((model.predict_proba(x).argmax(axis=1) == y).mean())


def train_model(epochs=12, batch_size=128, lr=1e-3):
    """Train an MLP on MNIST with the Adam optimizer and save it to MODEL_PATH."""
    x_train, y_train, x_test, y_test = load_mnist()
    model = MLP()

    # Adam optimizer state
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    m = [np.zeros_like(p) for p in model.params]
    v = [np.zeros_like(p) for p in model.params]
    step = 0

    n = len(x_train)
    for epoch in range(1, epochs + 1):
        order = np.random.permutation(n)
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            xb = random_shift(x_train[idx])
            yb = y_train[idx]

            probs, activations = model.forward(xb)
            grads = model.backward(probs, activations, yb)

            step += 1
            for i, (p, g) in enumerate(zip(model.params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * g
                v[i] = beta2 * v[i] + (1 - beta2) * g * g
                m_hat = m[i] / (1 - beta1 ** step)
                v_hat = v[i] / (1 - beta2 ** step)
                p -= lr * m_hat / (np.sqrt(v_hat) + eps)  # In-place update

        lr *= 0.85  # Simple learning-rate decay
        print(f"Epoch {epoch:2d}/{epochs}  test accuracy: {accuracy(model, x_test, y_test):.4f}")

    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    return model


# ---------------------------------------------------------------------------
# Preprocessing of the hand-drawn image (mimics how MNIST was created)
# ---------------------------------------------------------------------------

def preprocess(image):
    """
    Convert a grayscale PIL image (white digit on black) to a 784-vector:
    crop to the digit, fit it into a 20x20 box keeping the aspect ratio,
    then place it in a 28x28 frame centered by its center of mass.
    """
    bbox = image.getbbox()
    if bbox is None:
        return None
    digit = image.crop(bbox)

    w, h = digit.size
    scale = 20.0 / max(w, h)
    new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
    digit = digit.resize(new_size, Image.Resampling.LANCZOS)

    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(digit, ((28 - new_size[0]) // 2, (28 - new_size[1]) // 2))
    pixels = np.asarray(canvas, dtype=np.float32) / 255.0

    # Shift so that the center of mass sits at the image center
    ys, xs = np.indices(pixels.shape)
    total = pixels.sum()
    cy, cx = (ys * pixels).sum() / total, (xs * pixels).sum() / total
    pixels = np.roll(pixels, (int(round(13.5 - cy)), int(round(13.5 - cx))), axis=(0, 1))
    return pixels.reshape(1, -1)


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class DigitApp:
    CANVAS_SIZE = 280
    BRUSH_RADIUS = 10

    def __init__(self, root, model):
        self.model = model
        self.root = root
        root.title("Handwritten Digit Recognizer")
        root.resizable(False, False)

        # Visible canvas plus an off-screen PIL image that mirrors every stroke
        self.canvas = tk.Canvas(root, width=self.CANVAS_SIZE, height=self.CANVAS_SIZE,
                                bg="black", cursor="cross")
        self.canvas.grid(row=0, column=0, rowspan=3, padx=10, pady=10)
        self.image = Image.new("L", (self.CANVAS_SIZE, self.CANVAS_SIZE), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.last_point = None

        self.canvas.bind("<Button-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Button-3>", lambda _: self.clear())

        self.result_label = tk.Label(root, text="?", font=("Arial", 72, "bold"), width=2)
        self.result_label.grid(row=0, column=1, padx=10)
        self.confidence_label = tk.Label(root, text="Draw a digit", font=("Arial", 12))
        self.confidence_label.grid(row=1, column=1, padx=10)

        # One horizontal bar per digit showing its predicted probability
        self.bars = tk.Canvas(root, width=200, height=200)
        self.bars.grid(row=2, column=1, padx=10)
        self.draw_bars(np.zeros(10))

        tk.Button(root, text="Clear (right click)", command=self.clear).grid(
            row=3, column=0, columnspan=2, pady=(0, 10))

    def on_press(self, event):
        self.last_point = (event.x, event.y)
        self.paint(event.x, event.y, event.x, event.y)

    def on_drag(self, event):
        x0, y0 = self.last_point or (event.x, event.y)
        self.paint(x0, y0, event.x, event.y)
        self.last_point = (event.x, event.y)
        self.predict()

    def on_release(self, _event):
        self.last_point = None
        self.predict()

    def paint(self, x0, y0, x1, y1):
        r = self.BRUSH_RADIUS
        self.canvas.create_line(x0, y0, x1, y1, fill="white", width=2 * r,
                                capstyle=tk.ROUND, smooth=True)
        self.draw.line([x0, y0, x1, y1], fill=255, width=2 * r)
        self.draw.ellipse([x1 - r, y1 - r, x1 + r, y1 + r], fill=255)

    def predict(self):
        x = preprocess(self.image)
        if x is None:
            return
        probs = self.model.predict_proba(x)[0]
        digit = int(probs.argmax())
        self.result_label.config(text=str(digit))
        self.confidence_label.config(text=f"Confidence: {probs[digit] * 100:.1f}%")
        self.draw_bars(probs)

    def draw_bars(self, probs):
        self.bars.delete("all")
        best = int(probs.argmax()) if probs.any() else -1
        for d, p in enumerate(probs):
            y = d * 20 + 2
            self.bars.create_text(10, y + 8, text=str(d), font=("Arial", 10))
            self.bars.create_rectangle(25, y + 2, 25 + 170 * p, y + 15,
                                       fill="#2e8b57" if d == best else "#7a9cc6", outline="")

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, self.CANVAS_SIZE, self.CANVAS_SIZE], fill=0)
        self.result_label.config(text="?")
        self.confidence_label.config(text="Draw a digit")
        self.draw_bars(np.zeros(10))


def main():
    parser = argparse.ArgumentParser(description="Handwritten digit recognizer")
    parser.add_argument("--retrain", action="store_true", help="retrain the model even if one is saved")
    parser.add_argument("--train-only", action="store_true", help="train the model and exit")
    args = parser.parse_args()

    if args.retrain or args.train_only or not os.path.exists(MODEL_PATH):
        model = train_model()
    else:
        model = MLP.load(MODEL_PATH)
    if args.train_only:
        return

    root = tk.Tk()
    DigitApp(root, model)
    root.mainloop()


if __name__ == "__main__":
    sys.exit(main())
