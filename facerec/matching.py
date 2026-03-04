from __future__ import annotations

from typing import Sequence

import numpy as np

from .types import KnownEmbedding, MatchResult


class EmbeddingMatcher:
    def __init__(self, known_embeddings: Sequence[KnownEmbedding]) -> None:
        self._known_embeddings = list(known_embeddings)
        self._matrix: np.ndarray
        self._names: np.ndarray
        self._identity_ids: np.ndarray

        if not self._known_embeddings:
            self._matrix = np.empty((0, 0), dtype=np.float32)
            self._names = np.array([], dtype=object)
            self._identity_ids = np.array([], dtype=np.int64)
            return

        reference_dim = int(self._known_embeddings[0].embedding.size)
        filtered = [item for item in self._known_embeddings if int(item.embedding.size) == reference_dim]
        if not filtered:
            self._matrix = np.empty((0, 0), dtype=np.float32)
            self._names = np.array([], dtype=object)
            self._identity_ids = np.array([], dtype=np.int64)
            return

        self._known_embeddings = filtered
        matrix = np.vstack([item.embedding for item in self._known_embeddings]).astype(np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0

        self._matrix = matrix / norms
        self._names = np.array([item.name for item in self._known_embeddings], dtype=object)
        self._identity_ids = np.array([item.identity_id for item in self._known_embeddings], dtype=np.int64)

    @property
    def empty(self) -> bool:
        return self._matrix.size == 0

    def match(self, query_embedding: np.ndarray, threshold: float) -> MatchResult:
        if self.empty:
            return MatchResult(name="unknown", confidence=0.0, is_unknown=True, identity_id=None)

        query = np.asarray(query_embedding, dtype=np.float32).flatten()
        if query.size != self._matrix.shape[1]:
            return MatchResult(name="unknown", confidence=0.0, is_unknown=True, identity_id=None)

        norm = np.linalg.norm(query)
        if norm <= 0:
            return MatchResult(name="unknown", confidence=0.0, is_unknown=True, identity_id=None)

        query = query / norm
        similarities = self._matrix @ query
        best_idx = int(np.argmax(similarities))
        confidence = float(similarities[best_idx])
        name = str(self._names[best_idx])
        identity_id = int(self._identity_ids[best_idx])

        if confidence < threshold:
            return MatchResult(name="unknown", confidence=confidence, is_unknown=True, identity_id=None)

        return MatchResult(name=name, confidence=confidence, is_unknown=False, identity_id=identity_id)
