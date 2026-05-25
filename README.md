# Exam Answer Sheet OCR

A compact end-to-end pipeline for preprocessing and OCR of exam answer sheets (letter-by-letter name grid + OMR bubble zone).

## What this repository contains
- Notebook: `end-to-end/exam_answer_sheet_ocr.ipynb` — interactive pipeline that loads images from `datasets/`, preprocesses them, applies light augmentation, runs OCR, and extracts grid cells.
- Docker setup: `docker/docker-compose.yml` + `docker/Dockerfile` for running the environment in a container.
- Python requirements: `requirements.txt` (listed below).

## Quick start (Docker)
Build and run the container, then open a shell in the `ocr` service:

```bash
docker compose -f docker/docker-compose.yml up --build && \
  docker compose -f docker/docker-compose.yml exec -it ocr /bin/bash
```

Inside the container you can start Jupyter Lab and open the notebook:

```bash
jupyter lab --ip=0.0.0.0 --no-browser --allow-root --port=8888
# then open http://localhost:8889 in your browser
```

Alternative (local virtualenv)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# open the notebook with jupyter or run scripts
jupyter lab
```

## requirements.txt
The project dependencies are provided in `requirements.txt` (top-level). Install with `pip install -r requirements.txt`.

## Details of this project
- Removed synthetic-sheet generator and switched the notebook to load images from `datasets/`.
- Simplified preprocessing to a light pipeline for now:
  - Resize to 1024×1448 (aspect ratio preserved) and pad
  - Normalize pixels to the 0–1 range
  - Keep `ocr_ready_bgr` for downstream steps
- Data augmentation (light) implemented using `albumentations`:
  - Random rotation (±5°)
  - Brightness shift (up to ~15%)
- OCR:
  - Integrated PaddleOCR with explicit model choices: DBNet-style detector + Transformer-style recognizer (PP-OCRv5 mobile models)
  - Forced `paddle_dynamic` inference engine to avoid a runtime backend crash present for `paddle_static` in this environment
  - Used `ocr_engine.predict(...)` and added robust parsing to support multiple PaddleOCR output shapes
  - Visualized detections and produced a `structured_results` list of dictionaries with `text`, `bbox`, and `confidence`
- Grid handling kept: HoughLinesP-based detection of grid lines and cell cropping. The notebook still includes a fallback crop when detection is weak.

## Current script flow
- `scripts/process_single.py`: process one image end-to-end using the modular pipeline.
- `scripts/run_qualitative_test.py`: run the pipeline on a sample image and save the overlay output.
- `scripts/match_names.py`: match extracted text against the reference CSV and write the report CSV/TXT.

## Notebook status
- `end-to-end/PECE2.ipynb` is now a thin demo shell only.
- The core OCR/CNN logic lives in `end-to-end/src/`.
- The text-matching stage was extracted from the notebook into `scripts/match_names.py`.
