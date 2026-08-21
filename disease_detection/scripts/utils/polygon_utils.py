"""
Polygon processing utilities for YOLO segmentation formats.

Provides functions for mask-to-polygon conversion, Douglas-Peucker simplification,
coordinate normalization, polygon validity checks, and formatting.
"""

from typing import List, Tuple, Optional
import cv2
import numpy as np


def mask_to_polygons(
    mask_np: np.ndarray,
    epsilon_ratio: float = 0.005,
    min_area_pixels: int = 10,
) -> List[np.ndarray]:
    """
    Converts a binary mask (uint8: 0 or 255) into a list of simplified polygon coordinates.

    Args:
        mask_np (np.ndarray): Binary mask array [H, W].
        epsilon_ratio (float): Douglas-Peucker simplification parameter relative to contour arc length.
        min_area_pixels (int): Minimum contour area threshold to retain polygon.

    Returns:
        List[np.ndarray]: List of polygon coordinate arrays [N, 2] in (x, y) pixel coordinates.
    """
    if mask_np is None or np.sum(mask_np > 0) == 0:
        return []

    binary_mask = (mask_np > 0).astype(np.uint8)
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    polygons: List[np.ndarray] = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area_pixels:
            continue

        # Douglas-Peucker Simplification
        perimeter = cv2.arcLength(cnt, True)
        epsilon = epsilon_ratio * perimeter
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # Reshape contour to [N, 2]
        pts = approx.reshape(-1, 2)

        # A valid polygon must have at least 3 points
        if len(pts) >= 3:
            polygons.append(pts)

    return polygons


def normalize_polygon(
    polygon_pts: np.ndarray, img_width: int, img_height: int
) -> Optional[List[float]]:
    """
    Normalizes polygon (x, y) coordinates to [0.0, 1.0] relative to image dimensions.

    Args:
        polygon_pts (np.ndarray): Array of shape [N, 2] with (x, y) coordinates.
        img_width (int): Image width in pixels.
        img_height (int): Image height in pixels.

    Returns:
        Optional[List[float]]: Flattened list [x1, y1, x2, y2, ...] or None if invalid.
    """
    if img_width <= 0 or img_height <= 0 or len(polygon_pts) < 3:
        return None

    normalized_coords: List[float] = []
    for x, y in polygon_pts:
        norm_x = max(0.0, min(1.0, float(x) / img_width))
        norm_y = max(0.0, min(1.0, float(y) / img_height))
        normalized_coords.extend([round(norm_x, 6), round(norm_y, 6)])

    return normalized_coords


def validate_yolo_polygon_line(
    class_id: int, normalized_coords: List[float]
) -> Tuple[bool, str]:
    """
    Validates a single YOLO segmentation polygon line.

    Args:
        class_id (int): Integer class ID (>= 0).
        normalized_coords (List[float]): Flattened list [x1, y1, x2, y2, ...].

    Returns:
        Tuple[bool, str]: (is_valid, error_reason)
    """
    if class_id < 0:
        return False, f"Negative class_id: {class_id}"

    if len(normalized_coords) < 6 or len(normalized_coords) % 2 != 0:
        return False, f"Invalid coordinate count ({len(normalized_coords)}). Must be even and >= 6."

    for i, val in enumerate(normalized_coords):
        if not (0.0 <= val <= 1.0):
            return False, f"Coordinate at index {i} out of bounds [0.0, 1.0]: {val}"

    return True, "OK"
