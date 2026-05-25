"""
form_extractor.py – Extract structured fields from a cropped PECE exam form.

Pipeline:
  1. Crop the form via doc_crop.crop_to_border()
  2. Run Tesseract OCR (Spanish) on the cropped image
  3. Parse known field labels from the OCR text
  4. Save all results to  results/output.csv  and  results/output.json

Public API:
    extract_fields(image_path: str) -> dict
    process_dataset(dataset_dir: str, output_dir: str) -> list[dict]
"""

import re
import sys
import csv
import json
import cv2
import numpy as np
import pytesseract
from pathlib import Path

# Local import – must run from the workspace root or have it on sys.path
from src.data.preprocessing import crop_to_border

# ── Field definitions ─────────────────────────────────────────────────────────
# Each entry: (output_key, list_of_label_variants_to_search_for_in_ocr_text)
FIELDS = [
    ("apellido_paterno",  ["apellido paterno", "ap. paterno", "paterno"]),
    ("apellido_materno",  ["apellido materno", "ap. materno", "materno"]),
    ("nombre",            ["nombre(s)", "nombres", "nombre"]),
    ("ci",                ["c.i.", "ci", "carnet"]),
    ("extension",         ["ext.", "extension", "extensión"]),
    ("celular",           ["celular", "cel.", "telefono", "teléfono"]),
    ("genero",            ["género", "genero", "sexo"]),
    ("colegio",           ["colegio", "unidad educativa"]),
    ("tipo_de_colegio",   ["tipo de colegio", "tipo colegio", "t. colegio"]),
    ("ciudad",            ["ciudad"]),
]

EMPTY_RECORD = {
    "id":               None,
    "apellido_paterno": None,
    "apellido_materno": None,
    "nombre":           None,
    "ci":               None,
    "extension":        None,
    "celular":          None,
    "genero":           None,
    "colegio":          None,
    "tipo_de_colegio":  None,
    "ciudad":           None,
}

CSV_COLUMNS = list(EMPTY_RECORD.keys())


# ── OCR helper ────────────────────────────────────────────────────────────────

def _ocr(image: np.ndarray) -> str:
    """Run Tesseract on a BGR image and return raw text (lowercase)."""
    # Upscale for better OCR accuracy on small text
    scale = 2
    big = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    # Light denoising + sharpening
    gray = cv2.GaussianBlur(gray, (1, 1), 0)
    config = "--oem 3 --psm 6 -l spa"
    return pytesseract.image_to_string(gray, config=config).lower()


# ── Field parser ──────────────────────────────────────────────────────────────

def _value_after_label(text: str, labels: list[str]) -> str | None:
    """
    Search for any of *labels* in *text* (case-insensitive) and return the
    text that appears immediately after the label on the same line.
    Returns None if no label is found or the value is blank.
    """
    for label in labels:
        # Match the label followed by optional colon/space, then capture the rest of the line
        pattern = re.compile(
            re.escape(label) + r"[:\s]*([^\n]+)",
            re.IGNORECASE,
        )
        m = pattern.search(text)
        if m:
            value = m.group(1).strip()
            # Filter out common OCR artifacts and empty captures
            value = re.sub(r"[|_\-]{2,}", "", value).strip()
            if value:
                return value
    return None


def _parse_fields(ocr_text: str) -> dict:
    """Parse FIELDS out of raw OCR text and return a flat dict."""
    record = {}
    for key, labels in FIELDS:
        record[key] = _value_after_label(ocr_text, labels)
    return record


# ── Public API ────────────────────────────────────────────────────────────────

def extract_fields(image_path: str) -> dict:
    """
    Full pipeline for one image:
      1. Crop to border.
      2. OCR the cropped form.
      3. Parse field values.

    Returns a dict with all FIELDS plus 'id' (filename stem) and
    'source_file' (original path).
    """
    path = Path(image_path)
    record = dict(EMPTY_RECORD)
    record["id"] = path.stem

    try:
        cropped   = crop_to_border(str(path))
        ocr_text  = _ocr(cropped)
        parsed    = _parse_fields(ocr_text)
        record.update(parsed)
        record["_raw_ocr"] = ocr_text          # keep for debugging
    except Exception as e:
        record["_error"] = str(e)

    return record


def process_dataset(dataset_dir: str, output_dir: str) -> list:
    """
    Process every supported image in *dataset_dir* (recursively).
    Saves:
        <output_dir>/output.csv
        <output_dir>/output.json
    Returns the list of all extracted records.
    """
    dataset_dir = Path(dataset_dir)
    output_dir  = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp",
                "*.tif", "*.tiff", "*.heic", "*.heif", "*.pdf")
    files = sorted({f.resolve() for p in patterns for f in dataset_dir.rglob(p)})

    print(f"Found {len(files)} file(s) in '{dataset_dir}'")
    records = []

    for i, fpath in enumerate(files, 1):
        print(f"  [{i:>3}/{len(files)}] {fpath.name} … ", end="", flush=True)
        rec = extract_fields(str(fpath))
        records.append(rec)
        status = "OK" if rec.get("_error") is None else f"ERROR: {rec['_error']}"
        print(status)

    # ── CSV ──────────────────────────────────────────────────────────────────
    csv_path = output_dir / "output.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    # ── JSON ─────────────────────────────────────────────────────────────────
    json_path = output_dir / "output.json"
    # Exclude internal debug keys from the JSON export
    clean = [{k: v for k, v in r.items() if not k.startswith("_")} for r in records]
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(clean, fh, ensure_ascii=False, indent=2)

    print(f"\nSaved → {csv_path}")
    print(f"Saved → {json_path}")
    return records


# ── __main__ ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Extract PECE form fields from a dataset.")
    ap.add_argument("dataset_dir",  help="Path to the datasets/ folder.")
    ap.add_argument("--output_dir", default="src/results",
                    help="Where to save output.csv and output.json (default: src/results).")
    args = ap.parse_args()

    records = process_dataset(args.dataset_dir, args.output_dir)
    print(f"\nDone. {len(records)} record(s) extracted.")
