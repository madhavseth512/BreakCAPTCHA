"""
CAPTCHA preprocessing pipeline — CRNN + CTC version.

No segmentation, no binarization. The full grayscale CAPTCHA is fed to the
network and CTC handles character alignment internally.

Pipeline (per image):
    load PNG -> resize to 32x200 (HxW) -> grayscale -> normalize to [0.0, 1.0]

Why no Otsu threshold: the `captcha` library renders colored glyphs over noise
curves; a global threshold merges glyphs with the noise and discards the stroke
detail the CNN learns from. Modern CRNNs prefer rich grayscale input.

Saves to output_dir:
    X_train.npy        shape (N_train, 32, 200, 1)  float32
    X_val.npy          shape (N_val,   32, 200, 1)  float32
    y_train.npy        shape (N_train, 4)           int32   (character indices)
    y_val.npy          shape (N_val,   4)           int32
    char_classes.json  ['0'..'9', 'A'..'Z']  — index ordering (blank = index 36, implicit)
    sample_check.png   5 preprocessed images for visual sanity-check (gate 1)

Usage:
    python -m preprocessing.preprocess
    python -m preprocessing.preprocess --input data/dataset --output data/processed
"""

import os
import json
import argparse
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

TARGET_H = 32
TARGET_W = 200
N_CHARS = 4
SEED = 42


def load_and_preprocess(path):
    """Load a CAPTCHA PNG, resize to (TARGET_H, TARGET_W), grayscale, normalize.

    No thresholding — the network sees the full grayscale glyphs.
    """
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.resize(img, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    norm = gray.astype("float32") / 255.0
    return norm.reshape(TARGET_H, TARGET_W, 1)


def save_sample_check(images, labels, classes, output_dir, n=5):
    """Gate 1: save a few preprocessed images so glyphs can be eyeballed."""
    n = min(n, len(images))
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 2))
    if n == 1:
        axes = [axes]
    for ax, img, lbl in zip(axes, images[:n], labels[:n]):
        ax.imshow(img.squeeze(), cmap="gray")
        ax.set_title("".join(classes[i] for i in lbl), fontsize=10)
        ax.axis("off")
    plt.suptitle("Preprocessed samples — glyphs should be clearly visible", fontsize=11)
    plt.tight_layout()
    path = os.path.join(output_dir, "sample_check.png")
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"Sample check image saved -> '{path}'  (gate 1: confirm glyphs are legible)")


def prepare_dataset(dataset_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    filenames = sorted(f for f in os.listdir(dataset_dir) if f.endswith(".png"))
    print(f"Found {len(filenames)} PNG files in '{dataset_dir}'")

    images = []
    label_texts = []
    skipped = 0

    for fn in filenames:
        label_text = fn.split("_")[0]
        if len(label_text) != N_CHARS:
            skipped += 1
            continue
        img = load_and_preprocess(os.path.join(dataset_dir, fn))
        if img is None:
            skipped += 1
            continue
        images.append(img)
        label_texts.append(label_text)

    print(f"Loaded: {len(images)} images  |  Skipped: {skipped}")
    if not images:
        raise RuntimeError(f"No usable images found in '{dataset_dir}'. Run Phase 1 first.")

    # Class ordering from all chars present. LabelEncoder sorts lexicographically
    # -> digits 0-9 (indices 0-9) then letters A-Z (indices 10-35).
    # CTC blank is index 36 (= num_classes), implicit and not stored here.
    all_chars = sorted(set(c for lbl in label_texts for c in lbl))
    encoder = LabelEncoder()
    encoder.fit(all_chars)
    num_classes = len(encoder.classes_)

    classes_path = os.path.join(output_dir, "char_classes.json")
    with open(classes_path, "w") as f:
        json.dump(encoder.classes_.tolist(), f)
    print(f"Classes ({num_classes}): {encoder.classes_.tolist()}")
    print(f"CTC blank token -> index {num_classes} (implicit)")
    print(f"Saved class ordering -> '{classes_path}'")

    X = np.array(images, dtype="float32")  # (N, 32, 200, 1)

    # Integer label sequences (N, 4) — CTC consumes index sequences, not one-hot.
    y = np.array(
        [[encoder.transform([c])[0] for c in lbl] for lbl in label_texts],
        dtype="int32",
    )

    # Gate 2: round-trip a few labels through encode->decode and assert match.
    classes = encoder.classes_.tolist()
    for k in range(min(5, len(y))):
        decoded = "".join(classes[i] for i in y[k])
        assert decoded == label_texts[k], (
            f"Label round-trip mismatch at {k}: {decoded!r} != {label_texts[k]!r}"
        )
    print("Gate 2 passed: label encode->decode round-trip matches.")

    idx_train, idx_val = train_test_split(
        np.arange(len(X)), test_size=0.2, random_state=SEED
    )

    X_train, X_val = X[idx_train], X[idx_val]
    y_train, y_val = y[idx_train], y[idx_val]

    np.save(os.path.join(output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(output_dir, "X_val.npy"),   X_val)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "y_val.npy"),   y_val)

    save_sample_check(X_train, y_train, classes, output_dir)

    print(f"\nTrain: {len(idx_train)} samples  |  Val: {len(idx_val)} samples")
    print(f"X_train: {X_train.shape}  |  X_val: {X_val.shape}")
    print(f"y_train: {y_train.shape} (int32)  |  y_val: {y_val.shape}")
    print(f"\nAll arrays saved to '{output_dir}/'")


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess CAPTCHA images for CRNN+CTC.")
    parser.add_argument("--input",  type=str, default="data/dataset",   help="Directory of raw CAPTCHA PNGs")
    parser.add_argument("--output", type=str, default="data/processed", help="Output directory for .npy files")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare_dataset(args.input, args.output)
