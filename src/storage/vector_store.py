"""
Vector Store (Prototype Version)

Purpose:
Stores embeddings of tracked objects when they are LOST,
so we can later search for them in another camera.

This is an in-memory implementation for prototyping.

In Production This Would Be:
- FAISS (fast local ANN search)
- Milvus / Pinecone (distributed)
- pgvector (Postgres)
"""

import numpy as np
from typing import Dict, Tuple, Iterable


class VectorStore:
    def __init__(self):
        """
        Internal Structure:
        {
            object_id: {
                "embedding": np.ndarray,
                "camera_id": str,
                "timestamp": float
            }
        }
        """
        self._store: Dict[int, Dict] = {}

    # --------------------------------------------------
    # ADD EMBEDDING
    # --------------------------------------------------
    def add(self, object_id: int, embedding: np.ndarray,
            camera_id: str = None, timestamp: float = None):
        """
        Save embedding of a LOST object.

        Args:
            object_id: tracker ID
            embedding: feature vector
            camera_id: where it was lost
            timestamp: when it was lost
        """

        if not isinstance(embedding, np.ndarray):
            raise TypeError("Embedding must be numpy array")

        self._store[object_id] = {
            "embedding": embedding,
            "camera_id": camera_id,
            "timestamp": timestamp
        }

    # --------------------------------------------------
    # FETCH ALL EMBEDDINGS
    # --------------------------------------------------
    def get_all(self) -> Iterable[Tuple[int, np.ndarray]]:
        """
        Returns iterator of (object_id, embedding)

        Used by SimilaritySearch.
        """
        for obj_id, data in self._store.items():
            yield obj_id, data["embedding"]

    # --------------------------------------------------
    # GET FULL METADATA (optional use)
    # --------------------------------------------------
    def get_metadata(self, object_id: int):
        return self._store.get(object_id, None)

    # --------------------------------------------------
    # CHECK EMPTY
    # --------------------------------------------------
    def is_empty(self) -> bool:
        return len(self._store) == 0

    # --------------------------------------------------
    # REMOVE ENTRY (after successful ReID)
    # --------------------------------------------------
    def remove(self, object_id: int):
        if object_id in self._store:
            del self._store[object_id]

    # --------------------------------------------------
    # CLEAR STORE
    # --------------------------------------------------
    def clear(self):
        self._store.clear()

    # --------------------------------------------------
    # DEBUG INFO
    # --------------------------------------------------
    def __len__(self):
        return len(self._store)

    def __repr__(self):
        return f"VectorStore(size={len(self._store)})"
