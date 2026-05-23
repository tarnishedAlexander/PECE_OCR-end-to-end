"""Inference utilities for OCR + CNN ensemble.

The notebook uses a CNN prediction plus a small OCR voting pass over several
image variants. This module keeps that behavior in a reusable form while still
allowing the caller to inject model state from the notebook or a loader.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import pytesseract
import torch
from PIL import Image
from torchvision import transforms

from src.data.preprocessing import normalize_character_crop


DEFAULT_CHAR_CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
DEFAULT_WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_INFERENCE_CONTEXT: dict[str, Any] = {
    "model": None,
    "char_classes": DEFAULT_CHAR_CLASSES,
    "device": DEFAULT_DEVICE,
    "transform": None,
}


def _build_default_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def set_inference_context(
    model: Any | None = None,
    char_classes: Sequence[str] | None = None,
    device: torch.device | str | None = None,
    transform: transforms.Compose | None = None,
) -> None:
    """Set the default model and label mapping used by the helpers below."""
    _INFERENCE_CONTEXT["model"] = model
    if char_classes is not None:
        _INFERENCE_CONTEXT["char_classes"] = list(char_classes)
    if device is not None:
        _INFERENCE_CONTEXT["device"] = torch.device(device)
    if transform is not None:
        _INFERENCE_CONTEXT["transform"] = transform


def _resolve_context(model=None, char_classes=None, device=None, transform=None):
    resolved_model = model if model is not None else _INFERENCE_CONTEXT["model"]
    resolved_char_classes = list(char_classes) if char_classes is not None else _INFERENCE_CONTEXT["char_classes"]
    resolved_device = torch.device(device) if device is not None else _INFERENCE_CONTEXT["device"]
    resolved_transform = transform if transform is not None else _INFERENCE_CONTEXT["transform"]
    if resolved_transform is None:
        resolved_transform = _build_default_transform()
    return resolved_model, resolved_char_classes, resolved_device, resolved_transform


def _to_pil_rgb(image: np.ndarray) -> Image.Image:
    if image.ndim == 2:
        return Image.fromarray(image).convert("RGB")
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).convert("RGB")


def _clean_ocr_text(text: str, whitelist: str = DEFAULT_WHITELIST) -> str:
    allowed = set(whitelist)
    filtered = "".join(ch for ch in text.upper() if ch in allowed)
    return filtered.strip()


def _ocr_variant_text(image: np.ndarray, whitelist: str = DEFAULT_WHITELIST) -> tuple[str, float]:
    config = (
        "--psm 10 --oem 1 "
        f"-c tessedit_char_whitelist={whitelist}"
    )
    data = pytesseract.image_to_data(
        image,
        config=config,
        output_type=pytesseract.Output.DICT,
    )

    best_char = ""
    best_conf = -1.0

    texts = data.get("text", [])
    confs = data.get("conf", [])
    for idx, raw_text in enumerate(texts):
        cleaned = _clean_ocr_text(str(raw_text), whitelist=whitelist)
        if not cleaned:
            continue
        try:
            conf = float(confs[idx])
        except Exception:
            conf = -1.0
        if conf > best_conf:
            best_conf = conf
            best_char = cleaned[0]

    if not best_char:
        return "?", 0.0

    if best_conf < 0:
        best_conf = 0.0
    return best_char, round(float(best_conf), 2)


@torch.no_grad()
def predict_char_cnn(
    bgr: np.ndarray,
    debug: bool = False,
    model: Any | None = None,
    char_classes: Sequence[str] | None = None,
    device: torch.device | str | None = None,
    transform: transforms.Compose | None = None,
) -> tuple[str, float]:
    """Predict a character with the CNN branch.

    Returns a `(char, confidence)` pair. If the character crop cannot be
    normalized or no model context is configured, returns `('?', 0.0)`.
    """
    resolved_model, resolved_char_classes, resolved_device, resolved_transform = _resolve_context(
        model=model,
        char_classes=char_classes,
        device=device,
        transform=transform,
    )

    if resolved_model is None:
        return "?", 0.0

    resolved_model.eval()

    norm = normalize_character_crop(bgr, final_size=128, padding=30, debug=debug)
    if norm is None:
        return "?", 0.0

    pil_img = _to_pil_rgb(norm)
    x = resolved_transform(pil_img).unsqueeze(0).to(resolved_device)

    logits = resolved_model(x)[0]
    probs = torch.softmax(logits, dim=0)
    idx = int(torch.argmax(probs).item())
    char = resolved_char_classes[idx] if 0 <= idx < len(resolved_char_classes) else "?"
    return char, round(float(probs[idx].item()), 2)


def predict_char_ocr(
    bgr: np.ndarray,
    debug: bool = True,
    whitelist: str = DEFAULT_WHITELIST,
) -> tuple[str, float, str]:
    """Predict a character by voting over OCR-friendly variants.

    The variants mirror the notebook: base normalized crop, dilated, eroded,
    and an adaptive-threshold re-pass.
    """
    base = normalize_character_crop(bgr, final_size=128, padding=30, debug=False)
    if base is None:
        return "?", 0.0, ""

    variants: list[tuple[str, np.ndarray]] = [("base", base)]

    kernel = np.ones((2, 2), np.uint8)

    dilated = cv2.dilate(255 - base, kernel, iterations=1)
    variants.append(("dilated", 255 - dilated))

    eroded = cv2.erode(255 - base, kernel, iterations=1)
    variants.append(("eroded", 255 - eroded))

    adaptive = cv2.adaptiveThreshold(
        cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY),
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        8,
    )
    adaptive_bgr = cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)
    adaptive_norm = normalize_character_crop(adaptive_bgr, final_size=128, padding=30, debug=False)
    if adaptive_norm is not None:
        variants.append(("adaptive", adaptive_norm))

    best_char = "?"
    best_conf = 0.0
    best_variant = ""

    for variant_name, variant_img in variants:
        char, conf = _ocr_variant_text(variant_img, whitelist=whitelist)
        if conf >= best_conf:
            best_char = char
            best_conf = conf
            best_variant = variant_name

    return best_char, round(float(best_conf), 2), best_variant


def classify_char(
    bgr: np.ndarray,
    debug: bool = False,
    model: Any | None = None,
    char_classes: Sequence[str] | None = None,
    device: torch.device | str | None = None,
    transform: transforms.Compose | None = None,
    whitelist: str = DEFAULT_WHITELIST,
) -> tuple[str, str, float, float, str, str, str]:
    """Run both branches and return the notebook-compatible 7-tuple.

    The notebook currently returns the OCR prediction as the final label and
    keeps the CNN prediction for diagnostics, so that behavior is preserved.
    """
    cnn_char, cnn_conf = predict_char_cnn(
        bgr,
        debug=debug,
        model=model,
        char_classes=char_classes,
        device=device,
        transform=transform,
    )
    ocr_char, ocr_conf, ocr_variant = predict_char_ocr(
        bgr,
        debug=debug,
        whitelist=whitelist,
    )

    return ocr_char, "ocr", cnn_conf, ocr_conf, cnn_char, ocr_char, ocr_variant

