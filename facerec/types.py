from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

BBox = Tuple[float, float, float, float]


@dataclass
class DetectedFace:
    bbox: BBox
    det_score: float
    embedding: np.ndarray


@dataclass
class KnownEmbedding:
    identity_id: int
    name: str
    embedding: np.ndarray
    source_path: str


@dataclass
class MatchResult:
    name: str
    confidence: float
    is_unknown: bool
    identity_id: Optional[int] = None
