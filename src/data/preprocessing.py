"""Advanced Preprocessing: Precise marker-based anchoring."""

import cv2
import numpy as np
from pathlib import Path


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Return points ordered [top-left, top-right, bottom-right, bottom-left]."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _flatten_document(image: np.ndarray) -> np.ndarray:
    """Warp the largest page-like contour into a flat upright rectangle."""
    h, w = image.shape[:2]
    scale = 1000 / h
    small = cv2.resize(image, (int(w * scale), 1000))

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    doc_corners = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(c) > 0.15 * (small.shape[0] * small.shape[1]):
            doc_corners = (approx.reshape(4, 2) / scale).astype("float32")
            break

    if doc_corners is None:
        return image

    ordered = _order_points(doc_corners)
    (tl, tr, br, bl) = ordered
    wW = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    wH = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))

    dst = np.array([[0, 0], [wW - 1, 0], [wW - 1, wH - 1], [0, wH - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(ordered, dst)
    return cv2.warpPerspective(image, M, (wW, wH))


def _find_largest_dark_blob(region: np.ndarray):
    """Find the largest dark square-like blob within a region."""
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    best = None
    best_area = 0
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 30:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        aspect = bw / float(bh) if bh > 0 else 0
        if 0.5 <= aspect <= 2.0 and area > best_area:
            best = (x, y, bw, bh)
            best_area = area

    return best


def _detect_crop_box(warped: np.ndarray, padding: int = 4):
    """Detect the inner content box using the four registration marks."""
    H, W = warped.shape[:2]
    mH, mW = H // 2, W // 2

    quadrants = {
        "TL": (0, 0, mW, mH),
        "TR": (mW, 0, W - mW, mH),
        "BL": (0, mH, mW, H - mH),
        "BR": (mW, mH, W - mW, H - mH),
    }

    marks = {}
    for name, (xo, yo, qw, qh) in quadrants.items():
        region = warped[yo:yo + qh, xo:xo + qw]
        blob = _find_largest_dark_blob(region)
        if blob is not None:
            bx, by, bw, bh = blob
            marks[name] = {
                "x1": xo + bx,
                "y1": yo + by,
                "x2": xo + bx + bw,
                "y2": yo + by + bh,
            }

    if len(marks) < 4:
        return None

    top = max(marks["TL"]["y2"], marks["TR"]["y2"])
    bottom = min(marks["BL"]["y1"], marks["BR"]["y1"])
    left = max(marks["TL"]["x2"], marks["BL"]["x2"])
    right = min(marks["TR"]["x1"], marks["BR"]["x1"])

    if bottom <= top or right <= left:
        return None

    return (
        left + padding,
        top + padding,
        right - padding,
        bottom - padding,
    )


def crop_to_border(image_path: str, padding: int = 4) -> np.ndarray:
    """Load an image, flatten it, then crop to the inner black border."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    warped = _flatten_document(image)
    bbox = _detect_crop_box(warped, padding=padding)

    if bbox is None:
        return warped

    x1, y1, x2, y2 = bbox
    H, W = warped.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    return warped[y1:y2, x1:x2]

def order_points(pts):
    """Orders coordinates as [top-left, top-right, bottom-right, bottom-left]."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

def apply_scan_filter(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dilated_img = cv2.dilate(gray, np.ones((7, 7), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 21)
    diff_img = 255 - cv2.absdiff(gray, bg_img)
    norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    return cv2.cvtColor(norm_img, cv2.COLOR_GRAY2BGR)

def detect_document(image):
    height, width = image.shape[:2]
    work_height = 1000
    ratio = height / work_height
    work_img = cv2.resize(image, (int(width / ratio), work_height))
    
    gray = cv2.cvtColor(work_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Binary thresholding - adjusted to be more robust
    _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)
    
    cnts, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    for c in cnts:
        area = cv2.contourArea(c)
        # Wider area range for different photo distances
        if 50 < area < 10000:
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = w / float(h)
            # More lenient aspect ratio (0.5 to 1.5)
            if 0.5 <= aspect_ratio <= 1.5:
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    candidates.append((cX, cY))
    
    if len(candidates) >= 4:
        candidates = np.array(candidates)
        # Pick the 4 points closest to the image corners
        img_corners = np.array([[0,0], [work_img.shape[1], 0], [work_img.shape[1], work_height], [0, work_height]])
        final_points = []
        for ic in img_corners:
            dists = np.linalg.norm(candidates - ic, axis=1)
            final_points.append(candidates[np.argmin(dists)])
            
        return (np.array(final_points) * ratio).astype("float32")
            
    return None

def get_edge_map(image):
    height, width = image.shape[:2]
    work_height = 1000
    ratio = height / work_height
    work_img = cv2.resize(image, (int(width / ratio), work_height))
    gray = cv2.cvtColor(work_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)
    
    vis = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    
    # Draw ALL candidates in RED
    cnts, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        area = cv2.contourArea(c)
        if 50 < area < 10000:
            x, y, w, h = cv2.boundingRect(c)
            if 0.5 <= (w/h) <= 1.5:
                cv2.circle(vis, (x+w//2, y+h//2), 15, (0, 0, 255), 2)

    # Draw FINAL 4 in GREEN
    corners = detect_document(image)
    if corners is not None:
        for pt in (corners / ratio).astype(int):
            cv2.circle(vis, tuple(pt), 25, (0, 255, 0), 4)
            
    return vis

def preprocess_image(input_image_bgr, target_width=1024, target_height=1448):
    """Prepare a full document page for OCR without cutting out the page body.

    This prefers the document-outline warp from `_flatten_document` so we keep
    the printed instructions, headers, and handwritten regions intact.
    """
    processed = _flatten_document(input_image_bgr)

    enhanced = apply_scan_filter(processed)
    resized = cv2.resize(enhanced, (target_width, target_height), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    
    return {
        'resized': resized,
        'normalized': normalized,
        'corners_found': processed is not input_image_bgr
    }


# ----- Notebook-compatible helpers (reduce_shadow, deskew, crop_cell)
def reduce_shadow(img: np.ndarray) -> np.ndarray:
    """Reduce large-scale shadows per-channel (useful for phone photos).

    This implementation mirrors the notebook helper used by `PECE2.ipynb`.
    """
    planes = []
    for ch in cv2.split(img):
        bg = cv2.medianBlur(cv2.dilate(ch, np.ones((7, 7), np.uint8)), 21)
        diff = cv2.normalize(255 - cv2.absdiff(ch, bg), None, 0, 255, cv2.NORM_MINMAX)
        planes.append(diff)
    return cv2.merge(planes)


def deskew(gray: np.ndarray) -> tuple[np.ndarray, float]:
    """Correct small rotation using dominant Hough line angle."""
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
    if lines is None:
        return gray, 0.0
    angles = [np.degrees(t) - 90 for _, t in lines[:, 0] if abs(np.degrees(t) - 90) < 45]
    if not angles:
        return gray, 0.0
    angle = float(np.median(angles))
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated, angle


def apply_rotation_to_color(img_bgr: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 0.5:
        return img_bgr
    h, w = img_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img_bgr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def crop_cell(img: np.ndarray, x: int, y: int, w: int, h: int, pad: int = 10) -> np.ndarray:
    """Crop the inner region of a detected cell (used for OCR / classification)."""
    return img[max(0, y + pad): y + h - pad, max(0, x + pad): x + w - pad]



def normalize_character_crop(
    bgr: np.ndarray,
    final_size: int = 128,
    padding: int = 30,
    debug: bool = False,
) -> np.ndarray | None:
    """Normalize a character crop for CNN and OCR use.

    This mirrors the notebook helper: threshold, clean, find the best contour,
    center it in a square canvas, resize, invert to black-on-white, and add a
    small border.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((2, 2), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    h, w = th.shape
    cx_img, cy_img = w // 2, h // 2

    best = None
    best_score = float("inf")

    for c in contours:
        area = cv2.contourArea(c)
        if area < 10:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        cx = x + bw // 2
        cy = y + bh // 2
        dist = (cx - cx_img) ** 2 + (cy - cy_img) ** 2
        score = dist - area * 2
        if score < best_score:
            best_score = score
            best = (x, y, bw, bh)

    if best is None:
        return None

    x, y, bw, bh = best
    char = th[y:y + bh, x:x + bw]

    side = max(bw, bh) + padding
    square = np.zeros((side, side), dtype=np.uint8)
    x_offset = (side - bw) // 2
    y_offset = (side - bh) // 2
    square[y_offset:y_offset + bh, x_offset:x_offset + bw] = char

    final = cv2.resize(square, (final_size, final_size), interpolation=cv2.INTER_CUBIC)
    final = 255 - final
    final = cv2.copyMakeBorder(final, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)

    if debug:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(4, 4))
        plt.imshow(final, cmap="gray")
        plt.title("Normalized Character")
        plt.axis("off")
        plt.show()

    return final
