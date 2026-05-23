# BreakCAPTCHA

A CNN-based CAPTCHA solver that generates synthetic training data, preprocesses images with OpenCV, trains a character classifier with TensorFlow/Keras, and deploys as a Chrome browser extension using TensorFlow.js for client-side inference.

---

## Overview

BreakCAPTCHA targets standard 4-character alphanumeric CAPTCHAs (A–Z, 0–9). It treats CAPTCHA solving as a sequence recognition problem: the full image is fed to a CRNN (convolutional layers followed by bidirectional LSTMs) and trained end-to-end with CTC loss, which aligns the model's per-column output to the 4-character label without any explicit segmentation. The trained model is exported to TensorFlow.js and bundled inside a Chrome extension that auto-detects and solves CAPTCHAs on any webpage.

---

## Architecture

```
Raw CAPTCHA image (200x80 px)
        │
        ▼
[ Preprocessing ]
  Resize to 32x200 → Grayscale → Normalize to [0,1]
  (no thresholding — the network sees the full grayscale image)
        │
        ▼ (1 image, 32x200x1)
[ CRNN + CTC ]
  Conv(64) → Conv(128) → Conv(256) → Conv(256)   (pool height, keep width)
  → Reshape to 50 timesteps × 512 features
  → BiLSTM(128) → BiLSTM(128)
  → Dense(37, softmax)   (36 classes + 1 CTC blank)
  → Greedy CTC decode
        │
        ▼
[ Predicted text ] → Auto-filled into CAPTCHA input field
```

---

## Project Structure

```
BreakCAPTCHA/
├── data/
│   ├── generate_captchas.py      # Phase 1: synthetic CAPTCHA generator
│   └── dataset/                  # Generated images (gitignored)
├── preprocessing/
│   └── preprocess.py             # Phase 2: resize / grayscale / normalize pipeline
├── model/
│   ├── model_architecture.py     # CRNN definition
│   ├── train.py                  # Phase 3: training script
│   └── evaluate.py               # Evaluation and metrics
├── export/
│   └── convert_to_tfjs.py        # Phase 4: Keras → TF.js conversion
├── extension/
│   ├── manifest.json             # Chrome Manifest V3
│   ├── content.js                # CAPTCHA detection + solving
│   ├── background.js             # Service worker
│   ├── popup.html / popup.js     # Toggle UI
│   ├── preprocessing.js          # Browser-side image preprocessing
│   ├── solver.js                 # TF.js model loading + inference
│   ├── lib/                      # Bundled TF.js (tf.min.js)
│   └── tfjs_model/               # Converted model files (gitignored)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup

**Requirements:** Python 3.11, pip, Google Chrome

```bash
# Clone the repository
git clone https://github.com/madhavseth512/BreakCAPTCHA.git
cd BreakCAPTCHA

# Create and activate virtual environment
py -3.11 -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS

# Install dependencies
pip install -r requirements.txt
```

> **Note:** TensorFlow 2.15 on native Windows runs on CPU only. GPU training requires WSL2. For this model size (~300K parameters, ~40K samples), CPU training completes in under 15 minutes.

---

## Usage

Run each phase in order from the project root with the virtual environment active.

### Phase 1 — Generate Training Data

```bash
python -m data.generate_captchas
```

Generates 10,000 synthetic CAPTCHA images into `data/dataset/`. Labels are embedded in filenames (`ABCD_00001.png`).

Options:
```
--count   Number of images to generate  (default: 10000)
--seed    Random seed                   (default: 42)
--output  Output directory              (default: data/dataset)
```

---

### Phase 2 — Preprocess Images

```bash
python -m preprocessing.preprocess
```

Resizes each image to 32×200, converts to grayscale, and normalizes to [0,1] — no thresholding or segmentation. Saves whole-image arrays (`X_*.npy`), integer label sequences (`y_*.npy`, shape `(N, 4)`), and `char_classes.json` to `data/processed/`. Also writes `sample_check.png` (5 preprocessed images) so you can confirm glyphs are legible.

Options:
```
--input   Raw CAPTCHA directory   (default: data/dataset)
--output  Processed data output   (default: data/processed)
```

---

### Phase 3 — Train the Model

```bash
python -m model.train
```

Trains the CRNN with CTC loss and saves the best inference model (image → softmax) to `model/saved_model/captcha_model.h5`. A custom callback greedy-decodes the validation set each epoch and logs real per-character and full-CAPTCHA accuracy.

> **Validation gate — run this first.** Before a full run, confirm the pipeline is wired correctly by overfitting a tiny subset:
> ```bash
> python -m model.train --overfit 100
> ```
> It must reach ~100% full-CAPTCHA accuracy on 100 samples. If it can't memorize 100 images, the loss/decode wiring is broken — fix that before training for real.

Options:
```
--data        Processed data directory   (default: data/processed)
--output      Model output directory     (default: model/saved_model)
--epochs      Max training epochs        (default: 50)
--batch-size  Batch size                 (default: 32)
--overfit N   Overfit N samples to verify wiring, then exit (gate)
```

---

### Phase 4 — Evaluate

```bash
python -m model.evaluate
```

Reports per-character accuracy, per-CAPTCHA accuracy, per-class breakdown, and 10 sample predictions vs ground truth.

---

### Phase 5 — Export to TF.js

```bash
# Install TF.js converter (separate from main requirements)
pip install tensorflowjs==4.10.0

python -m export.convert_to_tfjs
```

Converts `captcha_model.h5` to TensorFlow.js LayersModel format and copies the output + `char_classes.json` into `extension/tfjs_model/`.

---

### Phase 6 — Load the Chrome Extension

1. Download `tf.min.js` from the [TensorFlow.js releases](https://github.com/tensorflow/tfjs/releases) and place it at `extension/lib/tf.min.js`
2. Open Chrome and navigate to `chrome://extensions`
3. Enable **Developer mode** (top-right toggle)
4. Click **Load unpacked** and select the `extension/` directory
5. The BreakCAPTCHA icon will appear in the toolbar — click it to enable/disable

---

## How It Works

### Preprocessing

Preprocessing is deliberately minimal — the CRNN learns its own features, so the goal is to preserve information, not to clean the image. Each CAPTCHA is resized to 32×200, converted to grayscale, and normalized to [0, 1]. **No thresholding or segmentation.**

Earlier versions used Otsu binarization, but the `captcha` library renders colored glyphs over noise curves: a global threshold merges glyphs with the noise and destroys the stroke detail the convolutional layers rely on. Feeding full grayscale keeps that signal intact.

The same pipeline is re-implemented in JavaScript (`preprocessing.js`) for browser-side use, matching the Python output without requiring OpenCV.js.

### Model

A CRNN: convolutional feature extractor → sequence model → CTC. The convolutions pool height aggressively but keep the width as a 50-step time axis, so the bidirectional LSTMs read the image left-to-right and CTC aligns that 50-step sequence to the 4-character label.

| Layer | Output Shape | Notes |
|---|---|---|
| Conv2D(64, 3×3) + BN + ReLU + MaxPool(2,2) | 16×100×64 | Edges and strokes |
| Conv2D(128, 3×3) + BN + ReLU + MaxPool(2,2) | 8×50×128 | Letter parts |
| Conv2D(256, 3×3) + BN + ReLU + MaxPool(2,1) | 4×50×256 | Pool height only |
| Conv2D(256, 3×3) + BN + ReLU + MaxPool(2,1) | 2×50×256 | Preserve width = timesteps |
| Permute + Reshape | 50×512 | Width becomes the time axis |
| Dense(64, ReLU) | 50×64 | Feature bottleneck |
| Bidirectional LSTM(128) ×2 | 50×256 | Left-right sequence context |
| Dense(37, softmax) | 50×37 | 36 classes + 1 CTC blank |

Trained end-to-end with **CTC loss** (no per-character labels or segmentation), Adam (lr=0.002, gradient clipping), early stopping on val_loss (patience=8), and ReduceLROnPlateau. Training batches drop the final remainder so a tiny last batch can't corrupt BatchNorm's inference statistics. Inference uses greedy CTC decoding.

Why CRNN+CTC over the earlier per-character CNN: the previous design used global average pooling, which discarded all spatial layout — the per-position heads couldn't tell character 1 from character 4 and stalled near the 1/36 random baseline. The CRNN keeps the horizontal sequence intact, which is exactly what positional character recognition needs.

### Chrome Extension

The extension loads TF.js from a bundled `tf.min.js` (no CDN, works on sites with strict CSPs). When a CAPTCHA image is detected on a page, the model is lazy-loaded once and cached. The JS preprocessing pipeline resizes and normalizes the whole image, the CRNN produces a per-timestep softmax sequence, a greedy CTC decode (reimplemented in JS, since TF.js has no built-in `ctc_decode`) turns it into text, and the result is filled into the nearest CAPTCHA input field.

Character class ordering is loaded at runtime from `char_classes.json` (generated during training), not hardcoded — ensuring the extension's label mapping always matches the model exactly.

---

## Validation Gates

CTC models fail *silently*: a wiring bug (mis-encoded labels, wrong blank-token index, a transposed time axis, or a mismatched decode) doesn't crash — it just trains to garbage that looks like a "bad model." Earlier iterations of this project burned several full training runs on exactly this. To prevent that, the pipeline is instrumented with four cheap gates that must pass **before** any expensive training run. Each isolates a different failure class and runs in seconds to minutes.

| Gate | What it checks | How to run | Utility it serves |
|---|---|---|---|
| **1 — Visual** | Preprocessed glyphs are still legible (grayscale preserved, not destroyed) | inspect `data/processed/sample_check.png` after `python -m preprocessing.preprocess` | Catches preprocessing that silently destroys the signal (e.g. over-aggressive thresholding). No model can recover from ruined input. |
| **2 — Label round-trip** | `char → index → char` encoding is loss-less and aligned with filenames | automatic assert during `python -m preprocessing.preprocess` | Catches off-by-one / blank-token / class-ordering bugs in the label mapping — the most common cause of a model that "won't learn." |
| **3 — Build-time shape** | The CNN produces `TIME_STEPS` (50) ≥ label length (4), and the realized time axis matches | automatic assert in `build_model()` | Guarantees CTC always has room to align the sequence; fails loudly if a future architecture change over-pools the width. |
| **4 — Overfit 100** | The full pipeline can memorize 100 samples to ~100% accuracy | `python -m model.train --overfit 100` | The decisive end-to-end proof. If the model can't memorize 100 images, the loss/decode wiring is broken — fix that before spending a real run. This is the gate that catches what static checks can't. |

> **Workflow:** run gates 1–3 (free, part of preprocessing/build), then gate 4. Only launch the full training run after gate 4 prints `GATE 4 PASSED`.

### What the gates uncovered

Building this model, the gates surfaced two non-obvious, silent failures that would otherwise have wasted multi-hour runs:

1. **BatchNorm corruption from a tiny remainder batch.** A final mini-batch of just a few samples produced garbage BatchNorm statistics that poisoned the moving averages used at *inference* time. Training loss looked healthy (≈0.5) while decoded accuracy stayed at 0%. Fix: training batches now drop the remainder.
2. **Learning rate too high for mini-batch noise.** `lr=5e-3` made the loss bounce instead of descend under mini-batch gradient noise (it only worked full-batch). Fix: `lr=2e-3`, with gradient clipping and `ReduceLROnPlateau`.

With both fixes, gate 4 reaches **100% full-CAPTCHA accuracy** on 100 samples — confirming the pipeline end-to-end.

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| tensorflow | 2.15.0 | Model training |
| opencv-python | 4.8.1.78 | Image preprocessing |
| numpy | 1.26.4 | Array operations |
| scikit-learn | 1.4.2 | Label encoding, train/val split |
| matplotlib | 3.8.4 | Training history plots |
| captcha | 0.6.0 | Synthetic data generation |
| tensorflowjs | 4.10.0 | Model export (Phase 5 only) |

---

## License

MIT
