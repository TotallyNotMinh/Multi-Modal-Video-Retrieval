import os
import gc
import pickle
import numpy as np
import faiss
from typing import List, Dict, Tuple, Optional

class FAISSIndex:
    """
    High-Performance FAISS Indexing for Large-Scale Video Keyframe Retrieval.
    Supports incremental chunk-based building to prevent RAM spikes on millions of vectors.
    """
    def __init__(
        self,
        dim: int = 1152,
        index_type: str = "FlatIP",
        nlist: int = 1024
    ):
        self.dim = dim
        self.index_type = index_type
        self.nlist = nlist
        self.index: Optional[faiss.Index] = None
        self.records: List[Dict] = []

    def init_index(self, dim: int, total_estimated: int = 100000):
        self.dim = dim
        if self.index_type == "FlatIP" or total_estimated < 2048:
            self.index = faiss.IndexFlatIP(dim)
        elif self.index_type == "IVFFlat":
            quantizer = faiss.IndexFlatIP(dim)
            actual_nlist = max(1, min(self.nlist, max(1, total_estimated // 10)))
            self.index = faiss.IndexIVFFlat(quantizer, dim, actual_nlist, faiss.METRIC_INNER_PRODUCT)
        self.records = []

    def add_batch(self, matrix_batch: np.ndarray, records_batch: List[Dict]):
        """
        Incrementally adds a chunk of vectors to the FAISS index to keep RAM constant.
        """
        if self.index is None:
            self.init_index(matrix_batch.shape[1])

        batch_f32 = np.ascontiguousarray(matrix_batch.astype(np.float32))
        
        # If IVFFlat requires training on the first batch
        if hasattr(self.index, "is_trained") and not self.index.is_trained:
            print(f"[FAISSIndex] Training index on initial batch of {len(batch_f32)} vectors...")
            self.index.train(batch_f32)

        self.index.add(batch_f32)
        self.records.extend(records_batch)
        del batch_f32

    def build(
        self,
        matrix: np.ndarray,
        records: List[Dict],
        save_path_prefix: Optional[str] = None
    ):
        """
        Builds the FAISS index from an in-memory normalized float32 matrix (N, dim).
        """
        N, d = matrix.shape
        self.init_index(d, total_estimated=N)
        self.add_batch(matrix, records)

        print(f"[FAISSIndex] Index built successfully. Total indexed: {self.index.ntotal}")

        if save_path_prefix:
            self.save(save_path_prefix)

    def save(self, path_prefix: str):
        dirname = os.path.dirname(path_prefix)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
            
        idx_path = f"{path_prefix}.index"
        meta_path = f"{path_prefix}_meta.pkl"

        faiss.write_index(self.index, idx_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.records, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"[FAISSIndex] Saved FAISS index to {idx_path} and metadata to {meta_path}")

    def load(self, path_prefix: str):
        idx_path = f"{path_prefix}.index"
        meta_path = f"{path_prefix}_meta.pkl"

        if not os.path.exists(idx_path) or not os.path.exists(meta_path):
            raise FileNotFoundError(f"Index files not found at {path_prefix}")

        self.index = faiss.read_index(idx_path)
        with open(meta_path, "rb") as f:
            self.records = pickle.load(f)

        self.dim = self.index.d
        print(f"[FAISSIndex] Loaded index with {self.index.ntotal} vectors (dim={self.dim}).")
        return self

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = 100
    ) -> List[Tuple[Dict, float]]:
        """
        Searches the index and returns list of (record, score).
        """
        if self.index is None:
            raise ValueError("Index is not loaded or built.")

        q = np.squeeze(np.asarray(query_vec, dtype=np.float32)).reshape(1, -1)
        norm = np.linalg.norm(q)
        if norm > 1e-12:
            q = q / norm

        scores, indices = self.index.search(q, min(top_k, self.index.ntotal))
        
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if 0 <= idx < len(self.records):
                results.append((self.records[idx], float(score)))

        return results
