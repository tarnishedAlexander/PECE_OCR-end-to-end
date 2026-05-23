"""Matching helpers for extracted text.

This keeps the naming utilities in the same lightweight package as the
pipeline entrypoint.
"""

from __future__ import annotations

from typing import List, Tuple


def levenshtein_match(query: str, choices: List[str], topk: int = 3) -> List[Tuple[str, int]]:
    return []


def knn_match(query: str, topk: int = 3) -> List[Tuple[str, float]]:
    return []
