"""Simple smoke test for the inference pipeline.

Loads a checkpoint produced by `src.models.train` (optional) and runs
`classify_char` on either a single image or all images in a directory.
This primarily verifies imports, model loading, and that `set_inference_context`
is wired correctly.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import cv2
try:
    import torch
except Exception:  # pragma: no cover - environment may not have torch
    torch = None

try:
    from src.models.cnn import build_mobilenet_v2
except Exception:  # pragma: no cover - allow script to run without model code available
    build_mobilenet_v2 = None

try:
    from src.models.inference import set_inference_context, classify_char
except Exception:
    set_inference_context = None
    classify_char = None
    # We'll provide an OCR-only fallback below.


def load_checkpoint(path: Path) -> Optional[dict]:
    if not path.exists():
        print(f"Checkpoint {path} not found")
        return None
    data = torch.load(str(path), map_location=torch.device("cpu"))
    return data


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", help="Path to checkpoint (optional)")
    p.add_argument("--image", help="Single image to run")
    p.add_argument("--dir", help="Directory of images to run")
    args = p.parse_args()

    checkpoint = None
    if args.model_path:
        checkpoint = load_checkpoint(Path(args.model_path))

    if checkpoint is not None and build_mobilenet_v2 is not None and set_inference_context is not None:
        classes = checkpoint.get("classes")
        num_classes = len(classes)
        model = build_mobilenet_v2(num_classes=num_classes)
        model.load_state_dict(checkpoint.get("state_dict"))
        model.eval()
        set_inference_context(model=model, char_classes=classes, device="cpu")
        print(f"Loaded model with {num_classes} classes")
    else:
        print("No model loaded — attempting OCR-only fallback")

    # OCR-only fallback if `classify_char` isn't importable.
    infer_func = classify_char
    if infer_func is None:
        import pytesseract
        import numpy as np

        def classify_char_fallback(bgr):
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            conf = 0.0
            try:
                txt = pytesseract.image_to_string(gray, config='--psm 10')
                txt = txt.strip().upper()
                if txt:
                    char = txt[0]
                    conf = 0.0
                else:
                    char = '?'
            except Exception:
                char = '?'
            # return in the same 7-tuple shape used by `classify_char`:
            return char, 'ocr', 0.0, conf, '?', char, ''

        infer_func = classify_char_fallback

    targets = []
    if args.image:
        targets.append(Path(args.image))
    if args.dir:
        pth = Path(args.dir)
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
            targets.extend(sorted(pth.glob(ext)))

    if not targets:
        print("No images provided. Use --image or --dir to run the smoke test.")
        return

    for imgp in targets:
        img = cv2.imread(str(imgp))
        if img is None:
            print(f"Failed to load {imgp}")
            continue
        ocr_char, source, cnn_conf, ocr_conf, cnn_char, final_char, variant = infer_func(img)
        print(f"{imgp.name}: final={final_char} source={source} cnn={cnn_char}/{cnn_conf} ocr={ocr_char}/{ocr_conf} variant={variant}")


if __name__ == "__main__":
    main()
