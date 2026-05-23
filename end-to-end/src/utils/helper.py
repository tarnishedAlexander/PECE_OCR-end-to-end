from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
try:
    import pypdfium2 as pdfium
except Exception:
    pdfium = None
try:
    import pillow_heif
    from pillow_heif import register_heif_opener
    _pillow_heif_available = True
    try:
        register_heif_opener()
    except Exception:
        # register may fail on some setups; we'll still try PIL.open for HEIC
        pass
except Exception:
    _pillow_heif_available = False


def show_image(image, title: str, cmap: str | None = None) -> None:
    """Render an image with matplotlib and hide axes for cleaner notebook output."""
    plt.figure()
    if image.ndim == 3 and image.shape[2] == 3 and cmap is None:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(image, cmap=cmap or 'gray')
    plt.title(title)
    plt.axis('off')
    plt.show()


def show_grid(images: list[np.ndarray], titles: list[str], cols: int = 2) -> None:
    """Display a small gallery of images in a fixed grid."""
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = np.array(axes).reshape(-1)
    for idx, ax in enumerate(axes):
        ax.axis('off')
        if idx < len(images):
            image = images[idx]
            if image.ndim == 3 and image.shape[2] == 3:
                ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                ax.imshow(image, cmap='gray')
            ax.set_title(titles[idx])
    plt.tight_layout()
    plt.show()


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    """Convert grayscale images to BGR so OpenCV drawing and OCR inputs stay consistent."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image.copy()


def find_test_image() -> Path:
    """Look for the first image in the datasets folder."""
    current_dir = Path.cwd()
    possible_paths = [
        current_dir / 'datasets',
        current_dir.parent / 'datasets',
        Path('/workspaces/ocr/datasets'),
    ]

    for datasets_path in possible_paths:
        if datasets_path.exists():
            for pattern in ('*.png', '*.jpg', '*.jpeg', '*.webp', '*.bmp', '*.tif', '*.tiff', '*.pdf', '*.heic', '*.heif'):
                matches = sorted(datasets_path.rglob(pattern))
                if matches:
                    return matches[0]

    raise FileNotFoundError(f'No image files found in datasets. Searched: {possible_paths}')


def load_test_image() -> np.ndarray:
    """Load the first available image from the datasets folder."""
    image_path = find_test_image()
    suffix = image_path.suffix.lower()
    if suffix == '.pdf':
        if pdfium is None:
            raise RuntimeError('pypdfium2 is required to load PDF files but is not installed')
        # Render the first page to a PIL image then convert to BGR numpy
        doc = pdfium.PdfDocument(str(image_path))
        try:
            page = doc.get_page(0)
            pil = page.render_topil()
            page.close()
        finally:
            doc.close()
        loaded = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        print(f'Loaded PDF first page from: {image_path}')
        return loaded

    loaded = cv2.imread(str(image_path))
    if loaded is not None:
        print(f'Loaded test image from: {image_path}')
        return loaded

    raise IOError(f'Failed to load image from {image_path}')


def load_image(path: Path | str) -> np.ndarray:
    """Load an image from a given path. Supports image files and PDFs (first page).

    Args:
        path: Path to image or PDF file.

    Returns:
        BGR numpy image.
    """
    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f'{image_path} does not exist')

    suffix = image_path.suffix.lower()
    if suffix == '.pdf':
        if pdfium is None:
            raise RuntimeError('pypdfium2 is required to load PDF files but is not installed')
        doc = pdfium.PdfDocument(str(image_path))
        try:
            page = doc.get_page(0)
            # Try several rendering APIs depending on pypdfium2 version
            pil = None
            try:
                pil = page.render_topil()
            except Exception:
                try:
                    # some versions expose a top-level helper
                    pil = pdfium.render_page_to_pil(page)
                except Exception:
                    try:
                        # fallback: render to bitmap then to PIL via numpy
                        bmp = page.render(scale=2)
                        pil = bmp.to_pil()
                        bmp.close()
                    except Exception:
                        pil = None
            finally:
                page.close()
        finally:
            doc.close()

        if pil is None:
            raise RuntimeError('Could not render PDF page: pypdfium2 API mismatch. Please upgrade pypdfium2.')

        loaded = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        return loaded

    # Handle HEIC/HEIF explicitly (Pillow may need pillow-heif plugin)
    if suffix in ('.heic', '.heif'):
        try:
            pil = Image.open(str(image_path))
            pil = _correct_exif_orientation(pil)
            pil = pil.convert('RGB')
            loaded = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        except Exception as e:
            raise RuntimeError('Failed to read HEIC/HEIF image. Install pillow-heif (pip install pillow-heif)') from e
        # Auto-rotate to portrait if image is horizontal
        if loaded.shape[1] > loaded.shape[0]:
            loaded = cv2.rotate(loaded, cv2.ROTATE_90_CLOCKWISE)
        return loaded

    loaded = cv2.imread(str(image_path))
    if loaded is None:
        # Try PIL as a fallback for some formats
        try:
            pil = Image.open(str(image_path))
            pil = _correct_exif_orientation(pil)
            pil = pil.convert('RGB')
            loaded = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        except Exception as e:
            raise IOError(f'Failed to load image from {image_path}: {e}')
    # Auto-rotate to portrait if image is horizontal
    if loaded.shape[1] > loaded.shape[0]:
        loaded = cv2.rotate(loaded, cv2.ROTATE_90_CLOCKWISE)
    return loaded


def _correct_exif_orientation(pil_img: Image.Image) -> Image.Image:
    """If EXIF orientation tag is present, transpose the PIL image accordingly."""
    try:
        exif = pil_img._getexif()
    except Exception:
        exif = None
    if not exif:
        return pil_img
    try:
        from PIL import ExifTags
        orientation_key = next(k for k, v in ExifTags.TAGS.items() if v == 'Orientation')
        orientation = exif.get(orientation_key)
    except Exception:
        return pil_img

    if orientation == 1:
        return pil_img
    if orientation == 2:
        return pil_img.transpose(Image.FLIP_LEFT_RIGHT)
    if orientation == 3:
        return pil_img.rotate(180, expand=True)
    if orientation == 4:
        return pil_img.transpose(Image.FLIP_TOP_BOTTOM)
    if orientation == 5:
        return pil_img.transpose(Image.FLIP_LEFT_RIGHT).rotate(90, expand=True)
    if orientation == 6:
        return pil_img.rotate(270, expand=True)
    if orientation == 7:
        return pil_img.transpose(Image.FLIP_LEFT_RIGHT).rotate(270, expand=True)
    if orientation == 8:
        return pil_img.rotate(90, expand=True)
    return pil_img


def bbox_bounds(item: dict) -> tuple[int, int, int, int]:
    """Return integer (x1,y1,x2,y2) bounds for an OCR `item` with `bbox` key."""
    xs = [int(p[0]) for p in item['bbox']]
    ys = [int(p[1]) for p in item['bbox']]
    return min(xs), min(ys), max(xs), max(ys)


# expose a short underscore name for backward compatibility with notebook helpers
_bbox_bounds = bbox_bounds
