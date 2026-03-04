from __future__ import annotations

from typing import Any, List

import numpy as np
from insightface.app import FaceAnalysis

from .config import EngineConfig
from .types import DetectedFace

# ?? ripped STRAIGHT from the docs u not getting ur code back 
def _get_face_attr(face: Any, key: str, default: Any = None) -> Any:
    if hasattr(face, key):
        return getattr(face, key)
    if isinstance(face, dict):
        return face.get(key, default)
    try:
        return face[key]
    except Exception:
        return default

# didnt grab all of this from a tutorial but watching one really helped me design this 
class FaceEngine:
    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.app = FaceAnalysis(name=config.model_name, providers=config.provider_chain())
        self.app.prepare(ctx_id=config.ctx_id(), det_size=(config.det_size, config.det_size))

    def detect_faces(self, image_bgr: np.ndarray) -> List[DetectedFace]:
        raw_faces = self.app.get(image_bgr)
        detected: List[DetectedFace] = []

        for raw_face in raw_faces:
            det_score = float(_get_face_attr(raw_face, "det_score", 1.0))
            if det_score < self.config.min_face_score:
                continue

            bbox = _get_face_attr(raw_face, "bbox")
            embedding = _get_face_attr(raw_face, "normed_embedding")
            if embedding is None:
                embedding = _get_face_attr(raw_face, "embedding")

            if bbox is None or embedding is None:
                continue

            bbox_array = np.asarray(bbox, dtype=np.float32).flatten()
            if bbox_array.size < 4:
                continue

            embedding_array = np.asarray(embedding, dtype=np.float32).flatten()
            norm = np.linalg.norm(embedding_array)
            if norm <= 0:
                continue

            normalized_embedding = (embedding_array / norm).astype(np.float32)

            detected.append(
                DetectedFace(
                    bbox=(
                        float(bbox_array[0]),
                        float(bbox_array[1]),
                        float(bbox_array[2]),
                        float(bbox_array[3]),
                    ),
                    det_score=det_score,
                    embedding=normalized_embedding,
                )
            )

        return detected

    @staticmethod
    def largest_face(faces: List[DetectedFace]) -> DetectedFace | None:
        if not faces:
            return None

        return max(
            faces,
            key=lambda face: max(0.0, face.bbox[2] - face.bbox[0]) * max(0.0, face.bbox[3] - face.bbox[1]),
        )
