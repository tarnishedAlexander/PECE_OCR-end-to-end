"""
PECE – local dataset pipeline  (no PaddleOCR, no process_all.py)

Replaces the Colab `files.upload()` from the original PECE.ipynb with an
automatic scan of the  datasets/  folder.  Every supported image / PDF is
preprocessed and run through the original OpenCV grid-detection algorithm.

Run from the  end-to-end/  directory:
    python PECE_local.py
"""

# ── 1. Paths & sys.path ───────────────────────────────────────────────────────
import sys, os, json
from pathlib import Path

NOTEBOOK_DIR = Path(__file__).resolve().parent   # …/end-to-end/
if str(NOTEBOOK_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOK_DIR))

DATASETS_DIR = NOTEBOOK_DIR / "datasets"
RESULTS_DIR  = NOTEBOOK_DIR / "src" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── 2. Imports ────────────────────────────────────────────────────────────────
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")           # change to "TkAgg" for interactive windows
import matplotlib.pyplot as plt

# project helpers (src/helper.py  +  src/data/preprocessing.py)
import src.helper as helper
from src.data.preprocessing import preprocess_image

load_image = helper.load_image

# ── 3. PARAMS (original PECE.ipynb values) ───────────────────────────────────
PARAMS = {
    "h_kernel_w":    80,
    "h_kernel_h":     1,
    "v_kernel_w":     1,
    "v_kernel_h":    80,
    "min_cell_area": 1_000,
    "max_cell_area": 500_000,
    # preprocessing target
    "target_width":  1024,
    "target_height": 1448,
}

# ── 4. Core functions (unchanged from original PECE.ipynb) ────────────────────
def robust_threshold(image):
    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 15, 3)


def detect_lines(thresh):
    h_k = cv2.getStructuringElement(cv2.MORPH_RECT,
                                    (PARAMS["h_kernel_w"], PARAMS["h_kernel_h"]))
    v_k = cv2.getStructuringElement(cv2.MORPH_RECT,
                                    (PARAMS["v_kernel_w"], PARAMS["v_kernel_h"]))
    h = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_k)
    v = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_k)
    return h, v, cv2.add(h, v)


def find_cells(grid):
    contours, _ = cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cells = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if PARAMS["min_cell_area"] < area < PARAMS["max_cell_area"]:
            cells.append((x, y, w, h))
    return cells


def visualize(img, thresh, grid, cells, title="", save_path=None):
    vis = img.copy()
    for (x, y, w, h) in cells:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes[0, 0].imshow(cv2.cvtColor(img,  cv2.COLOR_BGR2RGB)); axes[0, 0].set_title("Original");          axes[0, 0].axis("off")
    axes[0, 1].imshow(thresh, cmap="gray");                   axes[0, 1].set_title("Threshold");         axes[0, 1].axis("off")
    axes[1, 0].imshow(grid,   cmap="gray");                   axes[1, 0].set_title("Grid");              axes[1, 0].axis("off")
    axes[1, 1].imshow(cv2.cvtColor(vis,  cv2.COLOR_BGR2RGB)); axes[1, 1].set_title(f"Cells: {len(cells)}"); axes[1, 1].axis("off")
    if title:
        fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100)
    plt.show()
    plt.close(fig)

# ── 5. Full pipeline for one file ─────────────────────────────────────────────
def run_pipeline(image_path: Path) -> dict:
    print(f"\nProcessing: {image_path.name}")

    img     = load_image(image_path)                          # BGR ndarray
    pre     = preprocess_image(img,
                               target_width=PARAMS["target_width"],
                               target_height=PARAMS["target_height"])
    resized = pre["resized"]                                  # uint8 BGR 1024×1448

    thresh          = robust_threshold(resized)
    _, _, grid      = detect_lines(thresh)
    cells           = find_cells(grid)

    print(f"  → {len(cells)} cell(s) detected")

    stem      = image_path.stem.replace(" ", "_")
    overlay_p = RESULTS_DIR / f"{stem}_overlay.png"
    visualize(resized, thresh, grid, cells,
              title=image_path.name, save_path=overlay_p)

    result = {
        "file":    image_path.name,
        "n_cells": len(cells),
        "cells":   [{"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
                    for x, y, w, h in cells],
    }
    with open(RESULTS_DIR / f"{stem}_cells.json", "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)

    return result

# ── 6. Collect dataset files ───────────────────────────────────────────────────
PATTERNS = ("*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff","*.heic","*.heif","*.pdf")
seen, files = set(), []
for pat in PATTERNS:
    for f in sorted(DATASETS_DIR.rglob(pat)):
        key = f.resolve()
        if key not in seen:
            seen.add(key); files.append(f)

print(f"Found {len(files)} file(s) in {DATASETS_DIR.name}/\n")

# ── 7. Run on every file ───────────────────────────────────────────────────────
results = []
for fp in files:
    try:
        results.append(run_pipeline(fp))
    except Exception as e:
        print(f"  ERROR {fp.name}: {e}", file=sys.stderr)

# ── 8. Summary ────────────────────────────────────────────────────────────────
print(f"\n{'─'*60}")
print(f"{'File':<50} {'Cells':>6}")
print(f"{'─'*60}")
for r in results:
    print(f"  {r['file']:<48} {r['n_cells']:>6}")
print(f"\nDone – results in: {RESULTS_DIR}")
