"""
Embedding Engine

Converts a detected / tracked object into a feature vector
used for cross-camera Re-Identification.

This is a LIGHTWEIGHT prototype encoder.
Later you can replace with:
- OSNet
- FastReID
- CLIP embeddings
"""

import numpy as np
import cv2


class EmbeddingEngine:
    def __init__(self, size=(64, 128)):
        """
        Args:
            size: normalized crop size for embedding extraction
        """
        self.size = size

    # --------------------------------------------------
    # PUBLIC API USED BY PIPELINE
    # --------------------------------------------------
    def from_track(self, track):
        """
        Generate embedding from a tracker object.

        Expected Track Interface:
            track.bbox  -> (x1, y1, x2, y2)
            track.frame -> original frame reference
        """

        if track.frame is None:
            raise ValueError("Track has no frame attached for embedding extraction")

        x1, y1, x2, y2 = map(int, track.bbox)

        crop = track.frame[y1:y2, x1:x2]

        if crop.size == 0:
            return self._empty_embedding()

        return self._encode(crop)

    # --------------------------------------------------
    # CORE ENCODER (Prototype)
    # --------------------------------------------------
    def _encode(self, image):
        """
        Convert image crop → embedding vector
        """

        image = cv2.resize(image, self.size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Normalize
        image = image.astype("float32") / 255.0

        # Flatten → simple appearance descriptor
        embedding = image.flatten()

        # L2 normalize (important for cosine similarity)
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding

        return embedding / norm

    def _empty_embedding(self):
        return np.zeros(self.size[0] * self.size[1], dtype="float32")
