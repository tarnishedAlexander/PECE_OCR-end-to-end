import os, glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Find local datasets folder
cwd = os.getcwd()
LOCAL_DATASETS_DIR = os.path.join(cwd, 'end-to-end', 'datasets')
if os.path.exists('/workspace/end-to-end/datasets'):
    LOCAL_DATASETS_DIR = '/workspace/end-to-end/datasets'
elif os.path.exists(os.path.join(cwd, 'datasets')):
    LOCAL_DATASETS_DIR = os.path.join(cwd, 'datasets')

exts = ('*.jpg','*.jpeg','*.png','*.bmp','*.heic','*.heif')
imgs = []
for e in exts:
    imgs.extend(glob.glob(os.path.join(LOCAL_DATASETS_DIR, e)))
if not imgs:
    raise FileNotFoundError(f'No images found in {LOCAL_DATASETS_DIR}')
img_path = imgs[0]
print('Using image:', img_path)

# Functions (simplified from notebook)
def reduce_shadow(img):
    planes = []
    for ch in cv2.split(img):
        bg = cv2.medianBlur(cv2.dilate(ch, np.ones((7,7), np.uint8)), 21)
        diff = cv2.normalize(255 - cv2.absdiff(ch, bg), None, 0, 255, cv2.NORM_MINMAX)
        planes.append(diff)
    return cv2.merge(planes)

def ink_density(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return np.sum(g < 128) / g.size

PAD = 10
INK_TH = 0.02

# Load image
img = cv2.imread(img_path)
H, W = img.shape[:2]
print(f'Image size: {W}x{H}')

no_shadow = reduce_shadow(img)
gray = cv2.cvtColor(no_shadow, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0)
thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 4)

# ROI
ROI_TOP, ROI_BOTTOM, ROI_LEFT, ROI_RIGHT = 0.25, 0.70, 0.02, 0.98
y1 = int(H * ROI_TOP)
y2 = int(H * ROI_BOTTOM)
x1 = int(W * ROI_LEFT)
x2 = int(W * ROI_RIGHT)
roi_thresh = thresh[y1:y2, x1:x2]

# Morph close
k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
roi_thresh = cv2.morphologyEx(roi_thresh, cv2.MORPH_CLOSE, k)

contours, _ = cv2.findContours(roi_thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
MIN_SIDE_LOOSE = int(H * 0.015)
MAX_SIDE_LOOSE = int(H * 0.070)

candidates = []
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    if (MIN_SIDE_LOOSE < w < MAX_SIDE_LOOSE and MIN_SIDE_LOOSE < h < MAX_SIDE_LOOSE and 0.5 < w/h < 2.0):
        candidates.append((x, y, w, h))

if len(candidates) == 0:
    print('No candidates found')
    exit(0)

median_w = np.median([w for x,y,w,h in candidates])
median_h = np.median([h for x,y,w,h in candidates])
print(f'Median cell size: {median_w:.0f}x{median_h:.0f} px')

squares_roi = [(x,y,w,h) for x,y,w,h in candidates if 0.6 < w/median_w < 1.4 and 0.6 < h/median_h < 1.4]

# Deduplicate
squares_roi = sorted(squares_roi, key=lambda b: b[2]*b[3], reverse=True)
kept = []
for b in squares_roi:
    x1b,y1b,w1,h1 = b
    skip = False
    for k2 in kept:
        x2b,y2b,w2,h2 = k2
        ix = max(0, min(x1b+w1, x2b+w2) - max(x1b, x2b))
        iy = max(0, min(y1b+h1, y2b+h2) - max(y1b, y2b))
        if w1*h1 > 0 and ix*iy/(w1*h1) > 0.5:
            skip = True
            break
    if not skip:
        kept.append(b)

squares = sorted([(x+x1, y+y1, w, h) for x,y,w,h in kept], key=lambda b: (b[1]//20, b[0]))
print('Detected squares:', len(squares))

# Check filled
results = []
for (x,y,w,h) in squares:
    # crop with padding
    cy0 = max(0, y+PAD)
    cy1 = y+h-PAD
    cx0 = max(0, x+PAD)
    cx1 = x+w-PAD
    c = img[cy0:cy1, cx0:cx1]
    if c.size == 0:
        continue
    filled = ink_density(c) >= INK_TH
    results.append({'x':x,'y':y,'w':w,'h':h,'filled':filled,'density':round(ink_density(c),3)})

filled_cells = [r for r in results if r['filled']]
blank_cells = [r for r in results if not r['filled']]
print('Total :', len(results))
print('Filled:', len(filled_cells))
print('Blank :', len(blank_cells))

# show overlay
vis = img.copy()
for r in results:
    x,y,w,h = r['x'],r['y'],r['w'],r['h']
    col = (0,200,0) if r['filled'] else (200,200,200)
    cv2.rectangle(vis, (x,y), (x+w,y+h), col, 2)

# Save overlay to workspace
out_path = os.path.join(os.getcwd(), 'detection_overlay.jpg')
cv2.imwrite(out_path, vis)
print('Overlay saved to', out_path)
