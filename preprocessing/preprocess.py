"""
CAPTCHA image preprocessing pipeline.

Pipeline steps:
  1. Load image
  2. Grayscale conversion
  3. Otsu binary thresholding (THRESH_BINARY_INV) — letters become white on black
  4. Morphological close — fill small gaps, remove noise
  5. Vertical projection segmentation — find 3 valley columns to split into 4 chars
  6. Crop each character strip, resize to 28x28, normalize to [0, 1]

Note on segmentation approach:
  Contour-based segmentation fails on captcha-library output because all 4
  characters merge into a single connected component (noise curves bridge them).
  Vertical projection finds the 3 columns with lowest white-pixel density —
  the natural gaps between characters — and uses those as split boundaries.
  This gives near-zero skip rate on uniformly generated synthetic CAPTCHAs.

Usage (CLI):
    python -m preprocessing.preprocess
    python -m preprocessing.preprocess --input data/dataset --output data/processed
"""

import os
import argparse
import cv2
import numpy as np

TARGET_SIZE = (28, 28)
CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
N_CHARS = 4


class CaptchaPreprocessor:
    """Stateless preprocessor — all methods take and return image arrays."""

    def load_image(self, path):
        return cv2.imread(path)

    def to_grayscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def binarize(self, gray):
        # THRESH_BINARY_INV: dark letters → white (255), light background → black (0)
        # THRESH_OTSU: auto-select threshold from histogram — no hardcoded value needed
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary

    def denoise(self, binary):
        # MORPH_CLOSE fills small gaps inside characters without merging neighbours
        kernel = np.ones((2, 2), np.uint8)
        return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    def segment_by_projection(self, binary):
        """Split the binary image into N_CHARS strips using vertical projection.

        Sums white pixel values along each column to build a 1D profile.
        The 3 natural gaps between the 4 characters are the lowest-density
        columns. We search for each split near the expected position (at 25%,
        50%, 75% of image width) within a ±20px window so small shifts in
        character placement are handled correctly.

        Returns a list of N_CHARS (x, y, w, h) bounding rects, one per character.
        """
        h, w = binary.shape
        col_sums = binary.sum(axis=0).astype(float)

        search_range = 20
        split_xs = []
        for i in range(1, N_CHARS):
            center = int(w * i / N_CHARS)
            lo = max(0, center - search_range)
            hi = min(w, center + search_range)
            # argmin gives the column with fewest white pixels — the inter-char gap
            split_x = lo + int(np.argmin(col_sums[lo:hi]))
            split_xs.append(split_x)

        boundaries = [0] + split_xs + [w]
        rects = []
        for i in range(N_CHARS):
            x1, x2 = boundaries[i], boundaries[i + 1]
            rects.append((x1, 0, x2 - x1, h))

        return rects

    def extract_characters(self, binary, rects):
        """Crop each rect, resize to 28x28, normalize to float [0, 1]."""
        chars = []
        img_h, img_w = binary.shape
        for (x, y, w, h) in rects:
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(img_w, x + w)
            y2 = min(img_h, y + h)

            crop = binary[y1:y2, x1:x2]
            if crop.size == 0:
                return None
            crop = cv2.resize(crop, TARGET_SIZE, interpolation=cv2.INTER_AREA)
            crop = crop.astype("float32") / 255.0
            crop = np.expand_dims(crop, axis=-1)   # (28, 28, 1)
            chars.append(crop)
        return chars

    def process(self, image_path):
        """Full pipeline for one image. Returns list of 4 character arrays, or None."""
        image = self.load_image(image_path)
        if image is None:
            return None
        gray = self.to_grayscale(image)
        binary = self.binarize(gray)
        binary = self.denoise(binary)
        rects = self.segment_by_projection(binary)
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
    print(f"  Skipped (bad seg) : {skipped}  ({skipped / total_files * 100:.1f}%)")
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
