#!/usr/bin/env python3
"""
Process all files in the datasets/ folder: load (pdf/image), run preprocessing + section-crop OCR,
and save annotated overlays and JSON results into `src/results/`.

Outputs per input file (basename):
- results/{basename}_overlay.jpg          : full-page overlay with detected section boxes
- results/{basename}_{section}.jpg       : annotated crop for each section (section_a, section_b, respuestas)
- results/{basename}_results.json        : structured JSON of all kept detections

This script mirrors the crop-first OCR approach used in the notebook.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import sys

import cv2
import numpy as np
from paddleocr import PaddleOCR

ROOT = Path(__file__).resolve().parent
DATASETS = ROOT.parent / 'datasets'
RESULTS = ROOT / 'results'
RESULTS.mkdir(parents=True, exist_ok=True)

import sys
# Ensure end-to-end folder is on path so `data` and `src` packages resolve
ROOT_PARENT = ROOT.parent
if str(ROOT_PARENT) not in sys.path:
    sys.path.insert(0, str(ROOT_PARENT))

from data.preprocessing import preprocess_image
import src.utils.helper as helper
load_image = helper.load_image


def _bbox_bounds(item: dict) -> tuple[int, int, int, int]:
    return helper._bbox_bounds(item)


def _parse_ocr_results(raw_results) -> list[dict]:
    detections: list[dict] = []
    if not isinstance(raw_results, list) or not raw_results:
        return detections

    first_item = raw_results[0]
    if isinstance(first_item, list) and first_item and isinstance(first_item[0], (list, tuple)):
        parsed = first_item
    elif isinstance(first_item, dict):
        rec_texts = first_item.get('rec_texts', [])
        rec_scores = first_item.get('rec_scores', [])
        dt_polys = first_item.get('dt_polys', [])
        count = min(len(rec_texts), len(rec_scores), len(dt_polys))
        parsed = [[dt_polys[idx], (rec_texts[idx], rec_scores[idx])] for idx in range(count)]
    else:
        parsed = []

    for detection in parsed:
        if not isinstance(detection, (list, tuple)) or len(detection) < 2:
            continue
        bbox, rec = detection[0], detection[1]
        if not isinstance(rec, (list, tuple)) or len(rec) < 2:
            continue
        bbox_array = np.array(bbox, dtype=np.int32)
        if bbox_array.ndim != 2 or bbox_array.shape[0] < 3:
            continue
        text, confidence = rec[0], rec[1]
        detections.append({
            'text': str(text),
            'bbox': [[int(point[0]), int(point[1])] for point in bbox_array.tolist()],
            'confidence': float(confidence),
        })
    return detections


def detect_anchor_regions(structured_results: list[dict], img_h: int, img_w: int) -> dict:
    def _norm_text(v: str) -> str:
        return re.sub(r'\s+', ' ', str(v).strip().lower())

    a_hits = [it for it in structured_results if re.search(r'(seccion|sección|section)\s*a\b', _norm_text(it['text']))]
    b_hits = [it for it in structured_results if re.search(r'(seccion|sección|section)\s*b\b', _norm_text(it['text']))]
    r_hits = [it for it in structured_results if re.search(r'\brespuestas?\b', _norm_text(it['text']))]

    def bottom_or_default(hits, default_y):
        return max((_bbox_bounds(h)[3] for h in hits), default=default_y)

    def top_or_default(hits, default_y):
        return min((_bbox_bounds(h)[1] for h in hits), default=default_y)

    a_bottom = bottom_or_default(a_hits, int(img_h * 0.08))
    b_top = top_or_default(b_hits, int(img_h * 0.45))
    b_bottom = bottom_or_default(b_hits, int(img_h * 0.56))
    r_top = top_or_default(r_hits, int(img_h * 0.70))
    r_bottom = bottom_or_default(r_hits, int(img_h * 0.75))

    def clip(x1, y1, x2, y2):
        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        x2 = max(0, min(x2, img_w - 1))
        y2 = max(0, min(y2, img_h - 1))
        if x2 <= x1:
            x2 = min(img_w - 1, x1 + 1)
        if y2 <= y1:
            y2 = min(img_h - 1, y1 + 1)
        return (x1, y1, x2, y2)

    return {
        'section_a': clip(0, a_bottom + 4, img_w - 1, max(a_bottom + 5, b_top - 4)),
        'section_b': clip(0, b_bottom + 4, img_w - 1, max(b_bottom + 5, r_top - 4)),
        'respuestas': clip(0, r_bottom + 4, img_w - 1, img_h - 1),
    }


def run_for_file(path: Path, ocr_engine: PaddleOCR) -> None:
    print(f'Processing {path.name}')
    img = load_image(path)
    pre = preprocess_image(img, target_width=1024, target_height=1448)
    ocr_ready_bgr = pre['resized']
    img_h, img_w = ocr_ready_bgr.shape[:2]

    # Run page-level OCR
    raw = ocr_engine.predict(ocr_ready_bgr)
    structured_results = _parse_ocr_results(raw)

    # Detect section regions and run crop-level OCR (shift back to page coords)
    section_regions = detect_anchor_regions(structured_results, img_h=img_h, img_w=img_w)

    section_ocr = {}
    for name, (x1, y1, x2, y2) in section_regions.items():
        crop = ocr_ready_bgr[y1:y2 + 1, x1:x2 + 1].copy()
        raw_crop = ocr_engine.predict(crop)
        parsed = _parse_ocr_results(raw_crop)
        # shift coordinates
        for it in parsed:
            it['bbox'] = [[x + x1, y + y1] for x, y in it['bbox']]
        section_ocr[name] = parsed

    # Post-process respuestas: keep numeric-ish left column (row numbers)
    # Use same dominant-band approach as notebook
    def keep_respuestas_numbers_only(items):
        numeric = [it for it in items if re.fullmatch(r'\s*\d{1,2}\.?\s*', it['text'])]
        if not numeric:
            return []
        x_centers = np.array([(_bbox_bounds(it)[0] + _bbox_bounds(it)[2]) / 2.0 for it in numeric])
        xs = np.sort(x_centers)
        bands = []
        cur = [float(xs[0])]
        for x in xs[1:]:
            if abs(x - cur[-1]) <= 40:
                cur.append(float(x))
            else:
                bands.append(cur)
                cur = [float(x)]
        bands.append(cur)
        best = max(bands, key=lambda b: (len(b), -np.mean(b)))
        center = float(np.mean(best))
        kept = [it for it in numeric if abs(((_bbox_bounds(it)[0] + _bbox_bounds(it)[2]) / 2.0) - center) <= 45]
        kept.sort(key=lambda it: _bbox_bounds(it)[1])
        return kept

    section_ocr['respuestas'] = keep_respuestas_numbers_only(section_ocr.get('respuestas', []))

    # Save outputs
    base = path.stem.replace(' ', '_')
    # Full overlay
    overlay = ocr_ready_bgr.copy()
    for name, (x1, y1, x2, y2) in section_regions.items():
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(overlay, name, (x1 + 4, max(14, y1 + 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imwrite(str(RESULTS / f'{base}_overlay.jpg'), overlay)

    # Section crop images annotated
    for name, (x1, y1, x2, y2) in section_regions.items():
        crop = ocr_ready_bgr[y1:y2 + 1, x1:x2 + 1].copy()
        for it in section_ocr.get(name, []):
            pts = np.array(it['bbox'], dtype=np.int32)
            pts[:, 0] -= x1
            pts[:, 1] -= y1
            if pts.size:
                cv2.polylines(crop, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
                cv2.putText(crop, it['text'], tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.imwrite(str(RESULTS / f'{base}_{name}.jpg'), crop)

    # JSON
    out = {
        'file': str(path.name),
        'section_regions': {k: v for k, v in section_regions.items()},
        'section_ocr': section_ocr,
    }
    with open(RESULTS / f'{base}_results.json', 'w', encoding='utf-8') as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)

    print(f'Wrote results for {path.name} -> {RESULTS}')


def main():
    ocr_engine = PaddleOCR(
        text_detection_model_name='PP-OCRv5_mobile_det',
        text_recognition_model_name='PP-OCRv5_mobile_rec',
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        enable_hpi=False,
        engine='paddle_dynamic',
    )

    patterns = ('*.pdf', '*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp', '*.heic', '*.heif')
    files = []
    for pat in patterns:
        files.extend(sorted(DATASETS.rglob(pat)))

    if not files:
        print('No files found in datasets/')
        return

    for f in files:
        try:
            run_for_file(f, ocr_engine)
        except Exception as e:
            print(f'Error processing {f}: {e}', file=sys.stderr)


if __name__ == '__main__':
    main()
