"""
Generate 20 test CAPTCHAs for the local test page.

Run once before serving the test page:
    python test_page/generate_test_captchas.py

Then serve with:
    python -m http.server 8080 --directory test_page
    Open http://localhost:8080 in Chrome
"""

import os
import json
import random
import string

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "captchas")
MANIFEST   = os.path.join(SCRIPT_DIR, "captchas.json")
N          = 20
SEED       = 123
CHARS      = string.digits + string.ascii_uppercase  # 0-9, A-Z (matches training data)


def main():
    try:
        from captcha.image import ImageCaptcha
    except ImportError:
        print("ERROR: captcha package not installed.  Run: pip install captcha==0.6.0")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rng       = random.Random(SEED)
    generator = ImageCaptcha(width=200, height=80)

    manifest = []
    for i in range(N):
        label    = "".join(rng.choices(CHARS, k=4))
        filename = f"{i:03d}_{label}.png"
        path     = os.path.join(OUTPUT_DIR, filename)
        generator.write(label, path)
        manifest.append({"file": f"captchas/{filename}", "label": label})
        print(f"  {i+1:>3}/{N}  {label}  ->  {filename}")

    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nGenerated {N} CAPTCHAs in '{OUTPUT_DIR}/'")
    print(f"Manifest: '{MANIFEST}'")
    print("\nServe the test page:")
    print("  python -m http.server 8080 --directory test_page")
    print("  Open http://localhost:8080 in Chrome")


if __name__ == "__main__":
    main()
