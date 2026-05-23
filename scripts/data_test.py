"""Generate quick visual checks for preprocessing.

This samples a few images from `end-to-end/datasets`, runs the current
preprocessing pipeline, and writes comparison sheets to
`end-to-end/src/results/dataTest`.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT / "end-to-end"
if str(PROJECT_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import preprocess_image
from src.utils.helper import load_image


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".heic", ".heif", ".pdf"}


def collect_images(dataset_dir: Path) -> list[Path]:
    files = [p for p in dataset_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
    return sorted(files)


def save_comparison_sheet(image_path: Path, output_dir: Path) -> None:
    image = load_image(image_path)
    preprocessed = preprocess_image(image)

    # Keep a context-preserving view around the page for comparison.
    context_view = image.copy()
    if context_view.shape[1] > 1800:
        scale = 1800 / context_view.shape[1]
        context_view = cv2.resize(context_view, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    flattened_view = preprocessed["resized"].copy()
    if flattened_view.shape[0] > 0 and flattened_view.shape[1] > 0:
        flattened_view = cv2.copyMakeBorder(flattened_view, 24, 24, 24, 24, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    rows = 2
    cols = 2
    fig, axes = plt.subplots(rows, cols, figsize=(14, 10))

    axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(cv2.cvtColor(context_view, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("Context-preserving view")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(cv2.cvtColor(flattened_view, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title(f"Flattened document (corners={preprocessed['corners_found']})")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(preprocessed["normalized"], cmap="gray")
    axes[1, 1].set_title("Normalized page")
    axes[1, 1].axis("off")

    fig.suptitle(image_path.name, fontsize=14)
    fig.tight_layout()

    out_path = output_dir / f"{image_path.stem}_data_test.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run preprocessing and augmentation checks on sample images.")
    parser.add_argument("--dataset-dir", default=str(PROJECT_ROOT / "datasets"), help="Path to the datasets folder.")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "src" / "results" / "dataTest"), help="Where to write the comparison sheets.")
    parser.add_argument("--count", type=int, default=4, help="How many images to sample.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for sampling.")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(dataset_dir)
    if not images:
        raise FileNotFoundError(f"No supported images found in {dataset_dir}")

    random.seed(args.seed)
    sample_size = min(args.count, len(images))
    sampled = random.sample(images, sample_size)

    for image_path in sampled:
        save_comparison_sheet(image_path, output_dir)
        print(f"Saved comparison sheet for {image_path.name}")

    print(f"Done. Wrote {sample_size} sheet(s) to {output_dir}")


if __name__ == "__main__":
    main()