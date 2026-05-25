"""Data augmentation functions for exam answer sheet images."""

import cv2
import numpy as np
import albumentations as A
from typing import List


def build_augmentation_pipeline() -> A.Compose:
    """
    Build an augmentation pipeline with light rotation and brightness shift.
    
    Returns:
        An albumentations.Compose pipeline ready to apply to images.
    """
    return A.Compose([
        A.Rotate(limit=5, fill=(255, 255, 255), border_mode=cv2.BORDER_CONSTANT, p=0.9),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.0, p=0.8),
    ])


def add_salt_and_pepper_noise(image: np.ndarray, amount: float = 0.01, salt_vs_pepper: float = 0.5) -> np.ndarray:
    """Add light salt-and-pepper noise to an image.

    This is useful for checking whether the downstream preprocessing is too
    sensitive to scanner or phone-camera artifacts.
    """
    noisy = image.copy()
    total_pixels = image.shape[0] * image.shape[1]
    num_salt = int(total_pixels * amount * salt_vs_pepper)
    num_pepper = int(total_pixels * amount * (1.0 - salt_vs_pepper))

    if image.ndim == 2:
        channels = 1
    else:
        channels = image.shape[2]

    salt_coords = [np.random.randint(0, dim, num_salt) for dim in image.shape[:2]]
    pepper_coords = [np.random.randint(0, dim, num_pepper) for dim in image.shape[:2]]

    if channels == 1:
        noisy[salt_coords[0], salt_coords[1]] = 255
        noisy[pepper_coords[0], pepper_coords[1]] = 0
    else:
        noisy[salt_coords[0], salt_coords[1], :] = 255
        noisy[pepper_coords[0], pepper_coords[1], :] = 0

    return noisy


def augment_image(image: np.ndarray, augmentation: A.Compose) -> np.ndarray:
    """
    Apply augmentation to a single image.
    
    Args:
        image: Input BGR image.
        augmentation: Albumentations.Compose pipeline.
    
    Returns:
        Augmented BGR image.
    """
    return augmentation(image=image)['image']


def generate_augmented_variants(
    image: np.ndarray,
    num_variants: int = 4,
) -> List[np.ndarray]:
    """
    Generate multiple augmented variants of the same image.
    
    Args:
        image: Input BGR image.
        num_variants: Number of augmented variants to generate.
    
    Returns:
        List of augmented BGR images.
    """
    augmentation = build_augmentation_pipeline()
    variants = []
    for _ in range(num_variants):
        augmented = augment_image(image, augmentation)
        variants.append(augmented)

    # Keep one explicit salt-and-pepper sample in the mix when possible.
    if num_variants > 0:
        variants[-1] = add_salt_and_pepper_noise(variants[-1], amount=0.008)
    
    return variants
