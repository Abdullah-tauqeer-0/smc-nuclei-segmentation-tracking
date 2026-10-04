"""Turn a binary nuclear mask into per-object shape descriptors."""
import math

import cv2
import numpy as np


def remove_small_objects(mask: np.ndarray, min_area: int = 80) -> np.ndarray:
    """Zero out 8-connected foreground components smaller than ``min_area`` pixels.

    ``mask`` is a uint8 array whose foreground is non-zero.
    """
    mask = (mask > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[labels].astype(np.uint8)


def _eccentricity(moments: dict) -> float:
    if moments["m00"] == 0:
        return 0.0
    mu20 = moments["mu20"] / moments["m00"]
    mu11 = moments["mu11"] / moments["m00"]
    mu02 = moments["mu02"] / moments["m00"]
    common = math.sqrt((mu20 - mu02) ** 2 + 4 * mu11**2)
    lam1 = (mu20 + mu02 + common) / 2.0
    lam2 = (mu20 + mu02 - common) / 2.0
    return math.sqrt(1 - lam2 / lam1) if lam1 > 0 else 0.0


def build_annotation(image_path: str, mask: np.ndarray) -> dict:
    """Describe every connected object in ``mask`` (uint8, foreground non-zero).

    Per object: centroid (x, y), area, bbox (min_row, min_col, max_row, max_col),
    eccentricity, equivalent diameter, perimeter, solidity, roundness (4*pi*A/P^2)
    and the external contour (``outline``).
    """
    binary = (mask > 0).astype(np.uint8)
    n, labeled, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    cells = []
    for label in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[label])
        region = (labeled[y : y + h, x : x + w] == label).astype(np.uint8)
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        outline, perimeter, ecc, solidity = [], 0.0, 0.0, 0.0
        if contours:
            cnt = contours[0].reshape(-1, 2)
            cnt[:, 0] += x
            cnt[:, 1] += y
            if len(cnt) > 1:
                cnt = np.vstack([cnt, cnt[0]])
            outline = cnt.tolist()
            perimeter = cv2.arcLength(contours[0], True)
            ecc = _eccentricity(cv2.moments(contours[0]))
            hull_area = cv2.contourArea(cv2.convexHull(contours[0]))
            solidity = area / hull_area if hull_area > 0 else 0.0

        cells.append(
            {
                "id": label,
                "centroid": [float(centroids[label][0]), float(centroids[label][1])],
                "area": float(area),
                "bbox": [y, x, y + h - 1, x + w - 1],
                "eccentricity": ecc,
                "equivalent_diameter": math.sqrt(4.0 * area / math.pi) if area > 0 else 0.0,
                "perimeter": perimeter,
                "solidity": solidity,
                "roundness": 4.0 * math.pi * area / perimeter**2 if perimeter > 0 else 0.0,
                "outline": outline,
            }
        )

    return {"image_path": image_path, "object_count": len(cells), "cells": cells}
