"""
CAPTCHA image preprocessing pipeline.

Pipeline steps:
  1. Load image
  2. Grayscale conversion
  3. Otsu binary thresholding (THRESH_BINARY_INV) — letters become white on black
  4. Morphological close — fill small gaps, remove noise
  5. Contour detection (external only)
  6. Filter tiny blobs, split wide contours, sort left-to-right
  7. Crop each character, resize to 28x28, normalize to [0, 1]

Usage (CLI):
    python -m preprocessing.preprocess
    python -m preprocessing.preprocess --input data/dataset --output data/processed
"""

import os
import argparse
import cv2
import numpy as np

from preprocessing.helpers import get_bounding_rects, split_wide_rects, sort_left_to_right

TARGET_SIZE = (28, 28)
CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


class CaptchaPreprocessor:
    """Stateless preprocessor — all methods take and return image arrays."""

    def load_image(self, path):
        return cv2.imread(path)

    def to_grayscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def binarize(self, gray):
        # THRESH_BINARY_INV: dark letters → white (255), light background → black (0)
        # THRESH_OTSU: auto-select threshold from histogram — no hardcoded value
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary

    def denoise(self, binary):
        # MORPH_CLOSE fills small gaps inside characters without merging neighbours
        kernel = np.ones((2, 2), np.uint8)
        return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    def find_rects(self, binary):
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects = get_bounding_rects(contours)
        rects = split_wide_rects(rects)
        rects = sort_left_to_right(rects)
        return rects

    def extract_characters(self, binary, rects):
        """Crop each rect, resize to 28x28, normalize to float [0, 1]."""
        chars = []
        h_img, w_img = binary.shape
        for (x, y, w, h) in rects:
            pad = 2
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w_img, x + w + pad)
            y2 = min(h_img, y + h + pad)

            crop = binary[y1:y2, x1:x2]
            crop = cv2.resize(crop, TARGET_SIZE, interpolation=cv2.INTER_AREA)
            crop = crop.astype("float32") / 255.0
            crop = np.expand_dims(crop, axis=-1)   # (28, 28, 1)
            chars.append(crop)
        return chars

    def process(self, image_path):
        """Full pipeline for one image. Returns list of 4 character arrays, or None on failure."""
        image = self.load_image(image_path)
        if image is None:
            return None
        gray = self.to_grayscale(image)
        binary = self.binarize(gray)
        binary = self.denoise(binary)
        rects = self.find_rects(binary)
        if len(rects) != 4:
            return None
        return self.extract_characters(binary, rects)


def build_dataset(image_dir):
    """Process all PNGs and return (X, y) arrays.

    X — shape (N, 28, 28, 1), float32
    y — shape (N,), string character labels
    """
    preprocessor = CaptchaPreprocessor()
    all_chars = []
    all_labels = []
    skipped = 0

    filenames = sorted(f for f in os.listdir(image_dir) if f.endswith(".png"))
    for filename in filenames:
        label_text = filename.split("_")[0]
        if len(label_text) != 4:
            skipped += 1
            continue

        chars = preprocessor.process(os.path.join(image_dir, filename))
        if chars is None:
            skipped += 1
            continue

        for i, char_img in enumerate(chars):
            all_chars.append(char_img)
            all_labels.append(label_text[i])

    X = np.array(all_chars, dtype="float32")
    y = np.array(all_labels)
    return X, y, skipped


def prepare_dataset(image_dir, output_dir):
    """Process dataset and save numpy arrays to output_dir.

    Saves:
        X.npy  — (N, 28, 28, 1) character images
        y.npy  — (N,) string character labels
    """
    os.makedirs(output_dir, exist_ok=True)

    total_files = sum(1 for f in os.listdir(image_dir) if f.endswith(".png"))
    print(f"Processing {total_files} images from '{image_dir}'...")

    X, y, skipped = build_dataset(image_dir)

    np.save(os.path.join(output_dir, "X.npy"), X)
    np.save(os.path.join(output_dir, "y.npy"), y)

    print(f"\nResults:")
    print(f"  Total images      : {total_files}")
    print(f"  Skipped (bad seg) : {skipped}  ({skipped/total_files*100:.1f}%)")
    print(f"  Character samples : {len(X)}")
    print(f"  Unique classes    : {len(set(y))}")
    print(f"\nSaved to '{output_dir}/'  (X.npy, y.npy)")


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess CAPTCHA dataset into character arrays.")
    parser.add_argument("--input", type=str, default="data/dataset", help="Directory of raw CAPTCHA PNGs")
    parser.add_argument("--output", type=str, default="data/processed", help="Output directory for .npy files")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare_dataset(args.input, args.output)
