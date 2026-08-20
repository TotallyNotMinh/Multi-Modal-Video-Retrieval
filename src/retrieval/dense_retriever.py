import time
import numpy as np
from typing import List, Dict, Tuple

class DenseRetriever:
    """
    High-performance vector search engine over pre-normalized keyframe matrix.
    """
    def __init__(self, matrix: np.ndarray, records: List[Dict]):
        self.matrix = matrix  # (N, dim) float32 normalized
        self.records = records
        self.total_keyframes = matrix.shape[0]

    def search(self, query_vec: np.ndarray, top_k: int = 100) -> List[Tuple[Dict, float]]:
        """
        Computes cosine similarities via dot product and returns top_k (record, score).
        Handles both 1D and 2D query vectors safely.
        """
        q = np.squeeze(np.asarray(query_vec, dtype=np.float32))
        if q.ndim != 1:
            q = q.reshape(-1)

        norm = np.linalg.norm(q)
        if norm > 1e-12:
            q = q / norm

        if self.total_keyframes == 0:
            return []

        scores = np.dot(self.matrix, q)  # Shape (N,)
        
        k = min(top_k, self.total_keyframes)
        if k >= self.total_keyframes:
            top_indices = np.argsort(scores)[::-1]
        else:
            top_indices = np.argpartition(scores, -k)[-k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        results = [(self.records[idx], float(scores[idx])) for idx in top_indices]
        return results

    def get_all_scores(self, query_vec: np.ndarray) -> np.ndarray:
        q = np.squeeze(np.asarray(query_vec, dtype=np.float32))
        if q.ndim != 1:
            q = q.reshape(-1)
        norm = np.linalg.norm(q)
        if norm > 1e-12:
            q = q / norm
        return np.dot(self.matrix, q)
