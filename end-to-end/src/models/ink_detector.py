"""Ink density detector to classify cells as blank or filled.

Following the layered architecture, this operates as a heuristic-based
classification model at the models layer.
"""

from __future__ import annotations
import cv2
import numpy as np


def ink_density(bgr: np.ndarray) -> float:
    """Fraction of pixels darker than 128 (proxy for ink density)."""
    if bgr is None or bgr.size == 0:
        return 0.0
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(np.sum(g < 128) / g.size)


def is_filled(bgr: np.ndarray, threshold: float = 0.02) -> bool:
    """Check if the cell is filled (ink density >= threshold)."""
    return ink_density(bgr) >= threshold
