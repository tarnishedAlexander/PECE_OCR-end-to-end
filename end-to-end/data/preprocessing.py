"""Preprocessing functions for exam answer sheet images."""

import cv2
import numpy as np
from pathlib import Path


def resize_with_aspect_ratio(
    image: np.ndarray,
    target_width: int = 1024,
    target_height: int = 1448,
) -> np.ndarray:
    """
    Resize an image while preserving aspect ratio by scaling and padding to target size.
    
    Args:
        image: Input BGR image.
        target_width: Target width in pixels.
        target_height: Target height in pixels.
    
    Returns:
        Resized and padded image with white padding.
    """
    scale = min(target_width / image.shape[1], target_height / image.shape[0])
    new_width = int(round(image.shape[1] * scale))
    new_height = int(round(image.shape[0] * scale))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    canvas = np.full((target_height, target_width, 3), 255, dtype=np.uint8)
    y_offset = (target_height - new_height) // 2
    x_offset = (target_width - new_width) // 2
    canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized
    
    return canvas


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image to [0, 1] range.
    
    Args:
        image: Input BGR image with dtype uint8.
    
    Returns:
        Normalized image with dtype float32 in range [0, 1].
    """
    return image.astype(np.float32) / 255.0


def preprocess_image(
    input_image_bgr: np.ndarray,
    target_width: int = 1024,
    target_height: int = 1448,
) -> dict:
    """
    Complete preprocessing pipeline: resize, normalize, and return results.
    
    Args:
        input_image_bgr: Input BGR image.
        target_width: Target width for resizing.
        target_height: Target height for resizing.
    
    Returns:
        Dictionary with keys:
        - 'resized': BGR resized image (uint8).
        - 'normalized': Normalized image (float32, [0, 1]).
    """
    resized = resize_with_aspect_ratio(input_image_bgr, target_width, target_height)
    normalized = normalize_image(resized)
    
    return {
        'resized': resized,
        'normalized': normalized,
    }
