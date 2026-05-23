"""Single-file exam-processing pipeline placeholder.

The notebook logic will be moved here incrementally, but the package surface
stays simple for now.
"""

from __future__ import annotations

from typing import Any, Dict


def process_exam(img: Any, img_name: str = "exam", debug: bool = False) -> Dict[str, Any]:
    return {
        "img_name": img_name,
        "img_original": img,
        "img_deskewed": img,
        "vis": img,
        "results": [],
        "squares": [],
    }
