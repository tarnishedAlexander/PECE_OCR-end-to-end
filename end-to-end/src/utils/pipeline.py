"""PECE Exam Processing Pipeline.

Orchestrates:
  1. Document flattening and shadow reduction.
  2. Grid cell / square contour detection and self-calibration.
  3. Blank-vs-filled classification using ink density filtering.
  4. Character classification (CNN + OCR) on filled cells.
"""

from __future__ import annotations
import cv2
import numpy as np
from typing import Any, Dict, List

from src.data.preprocessing import _flatten_document, reduce_shadow, crop_cell
from src.models.ink_detector import ink_density, is_filled


def process_exam(img: np.ndarray, img_name: str = "exam", debug: bool = False, policy: str = "ocr") -> Dict[str, Any]:
    """Process an exam sheet image: flatten, detect squares, filter blank/filled, and run OCR.

    Args:
        img: BGR numpy image.
        img_name: Name of the image.
        debug: If True, prints diagnostic information.

    Returns:
        Dict containing:
            "img_name": str
            "img_original": original image
            "img_deskewed": flattened/shadow-reduced image
            "vis": annotated visualization image
            "results": list of cell dictionaries
            "squares": list of detected square bounding boxes (x, y, w, h)
    """
    if img is None:
        raise ValueError("Input image cannot be None")

    # Step 1: Flatten/align the document
    img_flattened = _flatten_document(img)
    H, W = img_flattened.shape[:2]

    # Step 2: Shadow reduction (used ONLY for thresholding, to match the notebook)
    no_shadow = reduce_shadow(img_flattened)
    img_deskewed = img_flattened  # Keeps the raw flattened image (with shadows) for crop_cell

    # Step 3: Threshold and binarize
    gray = cv2.cvtColor(no_shadow, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 15, 4
    )

    # Step 4: Define ROI (Region of Interest for the character grids)
    ROI_TOP = 0.25
    ROI_BOTTOM = 0.70
    ROI_LEFT = 0.02
    ROI_RIGHT = 0.98

    y1, y2 = int(H * ROI_TOP), int(H * ROI_BOTTOM)
    x1, x2 = int(W * ROI_LEFT), int(W * ROI_RIGHT)

    roi_thresh = thresh[y1:y2, x1:x2]

    # Step 5: Morphological closing to clean up borders
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    roi_thresh = cv2.morphologyEx(roi_thresh, cv2.MORPH_CLOSE, kernel)

    # Step 6: Find contours in the ROI
    contours, _ = cv2.findContours(roi_thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    MIN_SIDE_LOOSE = int(H * 0.015)
    MAX_SIDE_LOOSE = int(H * 0.070)

    candidates = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if (MIN_SIDE_LOOSE < w < MAX_SIDE_LOOSE and
                MIN_SIDE_LOOSE < h < MAX_SIDE_LOOSE and
                0.5 < w/h < 2.0):
            candidates.append((x, y, w, h))

    if not candidates:
        if debug:
            print(f"No grid cells found in {img_name}")
        return {
            "img_name": img_name,
            "img_original": img,
            "img_deskewed": img_deskewed,
            "vis": img_deskewed.copy(),
            "results": [],
            "squares": [],
        }

    # Step 7: Self-calibration filter based on median width/height
    median_w = np.median([w for _, _, w, _ in candidates])
    median_h = np.median([h for _, _, _, h in candidates])

    squares_roi = [
        (x, y, w, h) for x, y, w, h in candidates
        if 0.6 < w/median_w < 1.4 and 0.6 < h/median_h < 1.4
    ]

    # Step 8: Deduplicate overlapping square boxes
    squares_roi = sorted(squares_roi, key=lambda b: b[2]*b[3], reverse=True)
    kept = []
    for b in squares_roi:
        x1b, y1b, w1, h1 = b
        skip = False
        for k2 in kept:
            x2b, y2b, w2, h2 = k2
            ix = max(0, min(x1b+w1, x2b+w2) - max(x1b, x2b))
            iy = max(0, min(y1b+h1, y2b+h2) - max(y1b, y2b))
            if w1*h1 > 0 and (ix*iy)/(w1*h1) > 0.5:
                skip = True
                break
        if not skip:
            kept.append(b)

    # Step 9: Shift coordinates back to full image coordinate system
    squares = sorted(
        [(x + x1, y + y1, w, h) for x, y, w, h in kept],
        key=lambda b: (b[1] // 20, b[0])
    )

    # Step 10: Blank vs Filled Cell Classification via Ink Density Filtering
    results = []
    for (x, y, w, h) in squares:
        c = crop_cell(img_deskewed, x, y, w, h, pad=10)
        if c.size == 0:
            continue

        # Call our newly extracted ink density helpers!
        filled = is_filled(c, threshold=0.02)
        density = ink_density(c)

        results.append({
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'filled': filled,
            'density': round(density, 3),
            'letter': ''
        })

    # Step 11: Attempt OCR character classification if libraries are available
    try:
        from src.models.inference import classify_char
        for r in results:
            if r['filled']:
                c = crop_cell(img_deskewed, r['x'], r['y'], r['w'], r['h'], pad=10)
                try:
                    letter, source, cnn2_conf, ocr_conf, _, _, _ = classify_char(c, debug=False, policy=policy)
                    r['letter'] = letter
                    r['source'] = source
                    r['cnn2_conf'] = cnn2_conf
                    r['ocr_conf'] = ocr_conf
                except Exception:
                    r['letter'] = '?'
                    r['source'] = ''
                    r['cnn2_conf'] = 0.0
                    r['ocr_conf'] = 0.0
    except Exception:
        # Fallback if torch/inference packages cannot be imported
        if debug:
            print("OCR/CNN classification skipped (inference module unavailable).")

    # Step 12: Generate visualization overlay
    vis = img_deskewed.copy()
    for r in results:
        x, y, w, h = r['x'], r['y'], r['w'], r['h']
        color = (0, 200, 0) if r['filled'] else (200, 200, 200)
        cv2.rectangle(vis, (x, y), (x+w, y+h), color, 2)
        if r['filled'] and r['letter']:
            cv2.putText(vis, r['letter'], (x+3, y+h-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    return {
        "img_name": img_name,
        "img_original": img,
        "img_deskewed": img_deskewed,
        "vis": vis,
        "results": results,
        "squares": squares,
    }
