"""
Generate placeholder extension icons using OpenCV (already in requirements.txt).

Run once before loading the extension in Chrome:
    python extension/generate_icons.py
"""

import os
import cv2
import numpy as np

ICON_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icons")
os.makedirs(ICON_DIR, exist_ok=True)

# Deep blue background (#1e5fa8 in BGR), white text.
BG  = (168, 95, 30)    # BGR
FG  = (255, 255, 255)  # white

for size in [16, 48, 128]:
    img = np.full((size, size, 3), BG, dtype=np.uint8)

    if size >= 48:
        scale     = size / 80.0
        thickness = max(1, round(scale * 2))
        cv2.putText(
            img, "BC",
            (round(size * 0.08), round(size * 0.72)),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale * 0.7, FG, thickness, cv2.LINE_AA,
        )

    path = os.path.join(ICON_DIR, f"icon{size}.png")
    cv2.imwrite(path, img)
    print(f"  {path}")

print("Icons generated.")
