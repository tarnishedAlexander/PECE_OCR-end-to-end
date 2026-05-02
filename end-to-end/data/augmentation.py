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
    
    return variants
