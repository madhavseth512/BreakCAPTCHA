# BreakCAPTCHA

A CNN-based CAPTCHA solver that generates synthetic training data, preprocesses images with OpenCV, trains a character classifier with TensorFlow/Keras, and deploys as a Chrome browser extension using TensorFlow.js for client-side inference.

---

## Overview

BreakCAPTCHA targets standard 4-character alphanumeric CAPTCHAs (A–Z, 0–9). It treats CAPTCHA solving as a per-character classification problem: the image is segmented into individual characters using contour detection, each character is classified by a CNN, and the results are concatenated to produce the full prediction. The trained model is exported to TensorFlow.js and bundled inside a Chrome extension that auto-detects and solves CAPTCHAs on any webpage.

---

## Architecture

```
Raw CAPTCHA image (200x80 px)
        │
        ▼
[ OpenCV Preprocessing ]
  Grayscale → Otsu threshold → Morphological close
  → Vertical projection → Find 3 valley columns
  → Split into 4 strips → Crop & resize to 28x28
        │
        ▼ (4 character images, 28x28x1)
[ CNN Classifier ]
  Conv2D(32) → BatchNorm → ReLU → MaxPool
  Conv2D(64) → BatchNorm → ReLU → MaxPool
  Flatten → Dense(128) → Dropout(0.4) → Dense(36, softmax)
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
│   ├── preprocess.py             # Phase 2: OpenCV preprocessing pipeline
│   └── helpers.py                # Contour utilities
├── model/
│   ├── model_architecture.py     # CNN definition
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

Runs the full OpenCV pipeline on every image — grayscale, Otsu threshold, morphological close, contour detection, wide-contour splitting — and saves character crops as NumPy arrays to `data/processed/`.

Options:
```
--input   Raw CAPTCHA directory   (default: data/dataset)
--output  Processed data output   (default: data/processed)
```

Segmentation uses vertical projection — skip rate should be ~0% on synthetic data.

---

### Phase 3 — Train the Model

```bash
python -m model.train
```

Trains the CNN character classifier and saves the best model to `model/saved_model/captcha_model.h5`. Prints per-character and estimated per-CAPTCHA validation accuracy after training.

Options:
```
--data     Processed data directory   (default: data/processed)
--output   Model output directory     (default: model/saved_model)
--epochs   Max training epochs        (default: 15)
--batch    Batch size                 (default: 64)
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

Each CAPTCHA image is converted to grayscale and binarized using Otsu's method with `THRESH_BINARY_INV`, producing white characters on a black background. A morphological close operation removes noise.

Character segmentation uses **vertical projection**: white pixel values are summed along every column to produce a 1D density profile. The 3 columns with the lowest pixel density — the natural gaps between characters — are found near the expected boundaries (25%, 50%, 75% of image width, ±20px search window). These become the split points, dividing the image into 4 strips. Each strip is resized to 28×28 and normalized to [0, 1].

This approach was chosen over contour-based segmentation because the `captcha` library renders noise curves that bridge characters together, merging all 4 letters into a single connected component. Projection segmentation is immune to this — it works directly on pixel density regardless of connectivity.

The same pipeline is re-implemented in JavaScript (`preprocessing.js`) for browser-side use, matching the Python output without requiring OpenCV.js.

### Model

A compact CNN with two convolutional blocks followed by a fully connected classifier:

| Layer | Output Shape | Notes |
|---|---|---|
| Conv2D(32, 3×3) + BatchNorm + ReLU | 28×28×32 | Learns edges and strokes |
| MaxPooling(2×2) | 14×14×32 | Translation invariance |
| Conv2D(64, 3×3) + BatchNorm + ReLU | 14×14×64 | Learns letter shapes |
| MaxPooling(2×2) | 7×7×64 | |
| Flatten | 3136 | |
| Dense(128) + Dropout(0.4) | 128 | Classification head |
| Dense(36, softmax) | 36 | One class per character |

Trained with Adam (lr=0.001), categorical cross-entropy loss, early stopping on val_loss (patience=4), and ReduceLROnPlateau.

### Chrome Extension

The extension loads TF.js from a bundled `tf.min.js` (no CDN, works on sites with strict CSPs). When a CAPTCHA image is detected on a page, the model is lazy-loaded once and cached. The JS preprocessing pipeline segments the CAPTCHA, the model classifies each character, and the predicted text is filled into the nearest CAPTCHA input field.

Character class ordering is loaded at runtime from `char_classes.json` (generated during training), not hardcoded — ensuring the extension's label mapping always matches the model exactly.

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
