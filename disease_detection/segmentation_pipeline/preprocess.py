"""
preprocess.py

Automatic leaf extraction and prompt generation for SAM2.
"""

import cv2
import numpy as np

from config import *


# ==========================================================
# LEAF SEGMENTATION
# ==========================================================

def generate_leaf_mask(image):
    """
    Generate a binary mask of the leaf using Otsu thresholding.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    _, mask = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = np.ones((7, 7), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    return mask


# ==========================================================
# KEEP ONLY LARGEST OBJECT
# ==========================================================

def largest_component(mask):

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    if num_labels <= 1:
        return np.zeros_like(mask)

    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    result = np.zeros_like(mask)

    result[labels == largest] = 255

    return result


# ==========================================================
# GENERATE PROMPTS
# ==========================================================

def generate_prompt_points(mask):

    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        return np.empty((0, 2), dtype=np.int32)

    xmin = xs.min()
    xmax = xs.max()

    ymin = ys.min()
    ymax = ys.max()

    cx = (xmin + xmax) // 2
    cy = (ymin + ymax) // 2

    points = [

        (cx, cy),

        (xmin + (xmax - xmin) // 3, cy),

        (xmin + 2 * (xmax - xmin) // 3, cy),

    ]

    valid = []

    for x, y in points:

        x = np.clip(x, 0, mask.shape[1] - 1)
        y = np.clip(y, 0, mask.shape[0] - 1)

        if mask[y, x] > 0:
            valid.append((x, y))

    return np.array(valid, dtype=np.int32)


# ==========================================================
# COMPLETE PIPELINE
# ==========================================================

def automatic_prompt_generation(image):

    mask = generate_leaf_mask(image)

    mask = largest_component(mask)

    points = generate_prompt_points(mask)

    return mask, points

