"""
Contour utility functions for the CAPTCHA preprocessing pipeline.
"""

import numpy as np


def get_bounding_rects(contours):
    """Extract and filter bounding rectangles from contours.

    Filters out noise (very small blobs) that aren't character-sized.
    Returns list of (x, y, w, h) tuples.
    """
    import cv2
    rects = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w >= 5 and h >= 15:
            rects.append((x, y, w, h))
    return rects


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
