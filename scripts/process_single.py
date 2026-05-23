"""CLI helper to process a single image using `src.utils.pipeline`.

Usage:
    python scripts/process_single.py /path/to/image.jpg
"""
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT / "end-to-end"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python scripts/process_single.py <image_path>')
        sys.exit(1)

    img_path = Path(sys.argv[1])
    if not img_path.exists():
        print('Image not found:', img_path)
        sys.exit(1)

    # Lazy import to keep CLI lightweight
    from src.utils.helper import load_image
    from src.utils.pipeline import process_exam

    img = load_image(img_path)
    res = process_exam(img, img_name=img_path.stem, debug=False)
    print('Processed:', res['img_name'])
