"""Inference utilities for OCR + CNN ensemble.

The notebook uses a CNN prediction plus a small OCR voting pass over several
image variants. This module keeps that behavior in a reusable form while still
allowing the caller to inject model state from the notebook or a loader.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import cv2
import numpy as np
try:
    import pytesseract
    _PYTESSERACT_AVAILABLE = True
except ImportError:
    pytesseract = None
    _PYTESSERACT_AVAILABLE = False
try:
    import torch
    from torchvision import transforms
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None
    transforms = None
    _TORCH_AVAILABLE = False
from PIL import Image

from src.data.preprocessing import normalize_character_crop


DEFAULT_CHAR_CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
DEFAULT_WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") if _TORCH_AVAILABLE else None

_INFERENCE_CONTEXT: dict[str, Any] = {
    "model": None,
    "char_classes": DEFAULT_CHAR_CLASSES,
    "device": DEFAULT_DEVICE,
    "transform": None,
}


def _build_default_transform() -> Any:
    if not _TORCH_AVAILABLE:
        return None
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
    device: Any | None = None,
    transform: Any | None = None,
) -> None:
    """Set the default model and label mapping used by the helpers below."""
    _INFERENCE_CONTEXT["model"] = model
    if char_classes is not None:
        _INFERENCE_CONTEXT["char_classes"] = list(char_classes)
    if device is not None:
        _INFERENCE_CONTEXT["device"] = torch.device(device) if _TORCH_AVAILABLE else None
    if transform is not None:
        _INFERENCE_CONTEXT["transform"] = transform


def _resolve_context(model=None, char_classes=None, device=None, transform=None):
    resolved_model = model if model is not None else _INFERENCE_CONTEXT["model"]
    resolved_char_classes = list(char_classes) if char_classes is not None else _INFERENCE_CONTEXT["char_classes"]
    resolved_device = torch.device(device) if (device is not None and _TORCH_AVAILABLE) else _INFERENCE_CONTEXT["device"]
    resolved_transform = transform if transform is not None else _INFERENCE_CONTEXT["transform"]
    if resolved_transform is None:
        resolved_transform = _build_default_transform()
    return resolved_model, resolved_char_classes, resolved_device, resolved_transform


def _to_pil_gray(image: np.ndarray) -> Image.Image:
    """Convert a grayscale or BGR image to a PIL grayscale image.

    normalize_character_crop returns a single-channel (grayscale) image.
    The notebook passes it directly to Image.fromarray, so we do the same.
    """
    if image.ndim == 2:
        return Image.fromarray(image)  # already grayscale
    # BGR -> gray for colour inputs
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))


def _clean_ocr_text(text: str, whitelist: str = DEFAULT_WHITELIST) -> str:
    allowed = set(whitelist)
    filtered = "".join(ch for ch in text.upper() if ch in allowed)
    return filtered.strip()


def _ocr_variant_text(image: np.ndarray, whitelist: str = DEFAULT_WHITELIST) -> tuple[str, float]:
    if not _PYTESSERACT_AVAILABLE:
        return "?", 0.0
    config = (
        "--psm 10 --oem 1 "
        f"-c tessedit_char_whitelist={whitelist}"
    )
    try:
        data = pytesseract.image_to_data(
            image,
            config=config,
            output_type=pytesseract.Output.DICT,
        )
    except pytesseract.TesseractNotFoundError:
        return "?", 0.0
    except Exception:
        return "?", 0.0

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


def no_grad_decorator(func):
    if _TORCH_AVAILABLE:
        return torch.no_grad()(func)
    return func


@no_grad_decorator
def predict_char_cnn(
    bgr: np.ndarray,
    debug: bool = False,
    model: Any | None = None,
    char_classes: Sequence[str] | None = None,
    device: Any | None = None,
    transform: Any | None = None,
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

    if not _TORCH_AVAILABLE or resolved_model is None:
        return "?", 0.0

    resolved_model.eval()

    norm = normalize_character_crop(bgr, final_size=128, padding=30, debug=debug)
    if norm is None:
        return "?", 0.0

    pil_img = _to_pil_gray(norm).convert("RGB")  # Notebook uses grayscale; convert to RGB for MobileNetV2
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

    # Pytesseract returns confidence in 0-100 range; normalise to 0-1 to match
    # CNN confidence (mirrors the notebook: round(best_conf / 100, 2)).
    return best_char, round(float(best_conf) / 100.0, 2), best_variant


def classify_char(
    bgr: np.ndarray,
    debug: bool = False,
    policy: str = "ocr",
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

    # Decide final label according to policy
    policy = (policy or "ocr").lower()
    final_char = ocr_char
    final_source = "ocr"

    if policy == "cnn":
        final_char = cnn_char
        final_source = "cnn"
    elif policy == "auto":
        # Prefer CNN when its confidence clearly exceeds OCR
        try:
            if float(cnn_conf) >= float(ocr_conf) + 0.10:
                final_char = cnn_char
                final_source = "cnn"
            else:
                final_char = ocr_char
                final_source = "ocr"
        except Exception:
            final_char = ocr_char
            final_source = "ocr"

    return final_char, final_source, cnn_conf, ocr_conf, cnn_char, ocr_char, ocr_variant


def load_checkpoint_model(
    ckpt_path: str | Path,
    device: Any | None = None,
) -> tuple[Any, list]:
    """Load a checkpoint saved by `train.py` and return a model + class list.

    If torch or the cnn builder is unavailable, raises ImportError.
    """
    try:
        import torch
    except Exception as e:
        raise ImportError("PyTorch is required to load checkpoints") from e

    from pathlib import Path as _Path
    ckpt_path = _Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location=(device or DEFAULT_DEVICE))

    try:
        from src.models.cnn import build_mobilenet_v2
    except Exception as e:
        raise ImportError("Could not import model builder to instantiate checkpoint model") from e

    classes = ckpt.get("classes", DEFAULT_CHAR_CLASSES)
    num_classes = ckpt.get("num_classes", len(classes))

    model = build_mobilenet_v2(num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt["state_dict"]) if "state_dict" in ckpt else model.load_state_dict(ckpt)

    model = model.to(device or DEFAULT_DEVICE)

    return model, list(classes)

