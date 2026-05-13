"""
CAPTCHA preprocessing pipeline — multi-output CNN version.

No segmentation. The full CAPTCHA image is fed directly to the network.

Pipeline:
    load PNG → resize to 80×200 → grayscale → normalize to [0.0, 1.0]

Saves to output_dir:
    X_train.npy          shape (N_train, 80, 200, 1)  float32
    X_val.npy            shape (N_val,   80, 200, 1)  float32
    y1_train.npy         shape (N_train, 36)           one-hot float32  (char position 1)
    y2_train.npy .. y4_train.npy                       (positions 2–4)
    y1_val.npy   .. y4_val.npy                         (val equivalents)
    char_classes.json    ['0'..'9', 'A'..'Z']  — LabelEncoder ordering

Usage:
    python preprocessing/preprocess.py
    python preprocessing/preprocess.py --input data/dataset --output data/processed
"""

import os
import json
import argparse
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

TARGET_H = 80
TARGET_W = 200
N_CHARS = 4


def load_and_preprocess(path):
    """Load a CAPTCHA PNG, resize to (TARGET_H, TARGET_W), grayscale, normalize."""
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.resize(img, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    norm = gray.astype("float32") / 255.0
    return norm.reshape(TARGET_H, TARGET_W, 1)


def prepare_dataset(dataset_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    filenames = sorted(f for f in os.listdir(dataset_dir) if f.endswith(".png"))
    print(f"Found {len(filenames)} PNG files in '{dataset_dir}'")

    images = []
    labels = []
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
        labels.append(label_text)

    print(f"Loaded: {len(images)} images  |  Skipped: {skipped}")

    # Build class ordering from all chars present in the dataset.
    # LabelEncoder sorts lexicographically → digits first, then letters.
    # Saved to char_classes.json — the single source of truth for label mapping.
    all_chars = sorted(set(c for lbl in labels for c in lbl))
    encoder = LabelEncoder()
    encoder.fit(all_chars)
    num_classes = len(encoder.classes_)

    classes_path = os.path.join(output_dir, "char_classes.json")
    with open(classes_path, "w") as f:
        json.dump(encoder.classes_.tolist(), f)
    print(f"Classes ({num_classes}): {encoder.classes_.tolist()}")
    print(f"Saved class ordering → '{classes_path}'")

    X = np.array(images, dtype="float32")  # (N, 80, 200, 1)

    # Build one one-hot label array per character position
    y_raw    = [np.array([lbl[i] for lbl in labels]) for i in range(N_CHARS)]
    y_enc    = [encoder.transform(yi) for yi in y_raw]
    y_onehot = [to_categorical(yc, num_classes=num_classes).astype("float32") for yc in y_enc]

    indices = np.arange(len(X))
    idx_train, idx_val = train_test_split(indices, test_size=0.2, random_state=42)

    X_train, X_val = X[idx_train], X[idx_val]

    np.save(os.path.join(output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(output_dir, "X_val.npy"),   X_val)
    for i in range(N_CHARS):
        np.save(os.path.join(output_dir, f"y{i+1}_train.npy"), y_onehot[i][idx_train])
        np.save(os.path.join(output_dir, f"y{i+1}_val.npy"),   y_onehot[i][idx_val])

    print(f"\nTrain: {len(idx_train)} samples  |  Val: {len(idx_val)} samples")
    print(f"X_train: {X_train.shape}  |  X_val: {X_val.shape}")
    print(f"Label shape per position (train): {y_onehot[0][idx_train].shape}")
    print(f"\nAll arrays saved to '{output_dir}/'")


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess CAPTCHA images for multi-output CNN.")
    parser.add_argument("--input",  type=str, default="data/dataset",   help="Directory of raw CAPTCHA PNGs")
    parser.add_argument("--output", type=str, default="data/processed", help="Output directory for .npy files")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare_dataset(args.input, args.output)
