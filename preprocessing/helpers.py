"""
Contour utility functions for the CAPTCHA preprocessing pipeline.
"""

import cv2
import numpy as np


def get_bounding_rects(contours, top_n=8):
    """Extract bounding rectangles from the largest contours by actual pixel area.

    Uses cv2.contourArea() (actual filled area) rather than bounding rect
    dimensions. Noise curves from the captcha library can span the full image
    width but are thin lines — their actual area is tiny. Character blobs are
    large and filled, so they consistently rank in the top N.

    Args:
        contours: raw contours from cv2.findContours
        top_n:    how many largest candidates to keep before split logic (default 8,
                  since up to 4 merged pairs could exist before splitting)

    Returns:
        list of (x, y, w, h) tuples for the top_n largest contours by area
    """
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 50:           # discard specks too small to be any part of a character
            continue
        x, y, w, h = cv2.boundingRect(contour)
        candidates.append((area, x, y, w, h))

    # Keep only the largest N blobs — characters always dominate by area
    candidates.sort(key=lambda c: c[0], reverse=True)
    candidates = candidates[:top_n]

    return [(x, y, w, h) for (_, x, y, w, h) in candidates]


def split_wide_rects(rects):
    """Split bounding rects that are too wide (two merged characters).

    If a rect's width exceeds 1.5x the median width of all rects,
    it likely contains two touching characters. Split it vertically
    down the middle.

    Returns updated list, still unsorted.
    """
    if not rects:
        return rects

    median_w = np.median([w for (_, _, w, _) in rects])
    result = []
    for (x, y, w, h) in rects:
        if w > median_w * 1.5:
            half = w // 2
            result.append((x, y, half, h))
            result.append((x + half, y, w - half, h))
        else:
            result.append((x, y, w, h))
    return result


def sort_left_to_right(rects):
    """Sort bounding rects by x-coordinate (reading order)."""
    return sorted(rects, key=lambda r: r[0])
