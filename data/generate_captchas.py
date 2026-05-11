"""
Synthetic CAPTCHA data generator.

Generates labeled 4-character CAPTCHA images using the `captcha` library.
Labels are embedded in filenames: {LABEL}_{INDEX:05d}.png

Usage:
    python -m data.generate_captchas
    python -m data.generate_captchas --count 5000 --seed 7 --output data/dataset
"""

import os
import random
import string
import argparse
from captcha.image import ImageCaptcha

CHARACTER_SET = string.ascii_uppercase + string.digits  # A-Z + 0-9, 36 classes
CAPTCHA_LENGTH = 4
IMAGE_WIDTH = 200
IMAGE_HEIGHT = 80


def generate_dataset(count: int, seed: int, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)

    generator = ImageCaptcha(width=IMAGE_WIDTH, height=IMAGE_HEIGHT)

    for i in range(count):
        text = "".join(random.choices(CHARACTER_SET, k=CAPTCHA_LENGTH))
        filename = f"{text}_{i:05d}.png"
        filepath = os.path.join(output_dir, filename)
        generator.write(text, filepath)

        if (i + 1) % 500 == 0:
            print(f"Generated {i + 1}/{count}...")

    print(f"\nDone. {count} images saved to '{output_dir}'.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic CAPTCHA images.")
    parser.add_argument("--count", type=int, default=10_000, help="Number of images to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default="data/dataset", help="Output directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_dataset(count=args.count, seed=args.seed, output_dir=args.output)
