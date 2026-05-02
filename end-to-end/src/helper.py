from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def show_image(image, title: str, cmap: str | None = None) -> None:
    """Render an image with matplotlib and hide axes for cleaner notebook output."""
    plt.figure()
    if image.ndim == 3 and image.shape[2] == 3 and cmap is None:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(image, cmap=cmap or 'gray')
    plt.title(title)
    plt.axis('off')
    plt.show()


def show_grid(images: list[np.ndarray], titles: list[str], cols: int = 2) -> None:
    """Display a small gallery of images in a fixed grid."""
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = np.array(axes).reshape(-1)
    for idx, ax in enumerate(axes):
        ax.axis('off')
        if idx < len(images):
            image = images[idx]
            if image.ndim == 3 and image.shape[2] == 3:
                ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                ax.imshow(image, cmap='gray')
            ax.set_title(titles[idx])
    plt.tight_layout()
    plt.show()


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    """Convert grayscale images to BGR so OpenCV drawing and OCR inputs stay consistent."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image.copy()


def find_test_image() -> Path:
    """Look for the first image in the datasets folder."""
    current_dir = Path.cwd()
    possible_paths = [
        current_dir / 'datasets',
        current_dir.parent / 'datasets',
        Path('/workspaces/ocr/datasets'),
    ]

    for datasets_path in possible_paths:
        if datasets_path.exists():
            for pattern in ('*.png', '*.jpg', '*.jpeg', '*.webp', '*.bmp', '*.tif', '*.tiff'):
                matches = sorted(datasets_path.rglob(pattern))
                if matches:
                    return matches[0]

    raise FileNotFoundError(f'No image files found in datasets. Searched: {possible_paths}')


def load_test_image() -> np.ndarray:
    """Load the first available image from the datasets folder."""
    image_path = find_test_image()
    loaded = cv2.imread(str(image_path))
    if loaded is not None:
        print(f'Loaded test image from: {image_path}')
        return loaded
    raise IOError(f'Failed to load image from {image_path}')
