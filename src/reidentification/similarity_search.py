import numpy as np


class SimilaritySearch:
    def __init__(self, vector_store, threshold=0.7):
        self.vector_store = vector_store
        self.threshold = threshold

    def _cosine(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def query(self, embedding):
        """
        Returns (object_id, similarity_score) if match found.
        Else (None, 0).
        """
        if self.vector_store.is_empty():
            return None, 0.0

        best_id = None
        best_score = 0.0

        for object_id, stored_emb in self.vector_store.get_all():
            score = self._cosine(embedding, stored_emb)

            if score > best_score:
                best_score = score
                best_id = object_id

        if best_score >= self.threshold:
            return best_id, best_score

        return None, best_score
