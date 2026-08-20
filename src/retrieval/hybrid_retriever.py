import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from src.retrieval.dense_retriever import DenseRetriever
from src.index.object_indexer import ObjectIndexer
from src.index.metadata_indexer import MetadataIndexer

class HybridRetriever:
    """
    Unified multi-modal retriever implementing:
    - Dense visual CLIP similarity
    - Object detection entity boosting
    - YouTube metadata BM25 fusion
    - 1D Temporal shot smoothing
    - Submodular shot deduplication & portfolio allocation for R@k
    """
    def __init__(
        self,
        dense_retriever: DenseRetriever,
        object_indexer: Optional[ObjectIndexer] = None,
        metadata_indexer: Optional[MetadataIndexer] = None
    ):
        self.dense_retriever = dense_retriever
        self.object_indexer = object_indexer
        self.metadata_indexer = metadata_indexer
        self.records = dense_retriever.records
        
        # Pre-group records by video for temporal smoothing
        self.video_to_indices = defaultdict(list)
        for idx, rec in enumerate(self.records):
            self.video_to_indices[rec["video_id"]].append(idx)

    def apply_temporal_smoothing(
        self,
        scores: np.ndarray,
        sigma_frames: float = 1.5,
        window_size: int = 3
    ) -> np.ndarray:
        """
        Applies a 1D Gaussian smoothing within each video's keyframe timeline.
        """
        smoothed_scores = scores.copy()
        
        # Gaussian weights for [-w, ..., 0, ..., w]
        deltas = np.arange(-window_size, window_size + 1)
        weights = np.exp(-0.5 * (deltas / sigma_frames) ** 2)
        weights = weights / np.sum(weights)

        for video_id, indices in self.video_to_indices.items():
            if len(indices) <= 1:
                continue
            vid_scores = scores[indices]
            pad_scores = np.pad(vid_scores, (window_size, window_size), mode='edge')
            conv = np.convolve(pad_scores, weights, mode='valid')
            smoothed_scores[indices] = conv

        return smoothed_scores

    def search_hybrid(
        self,
        query_vec: np.ndarray,
        query_text_en: str = "",
        query_text_vi: str = "",
        w_dense: float = 0.70,
        w_object: float = 0.15,
        w_meta: float = 0.15,
        use_temporal_smoothing: bool = True,
        use_shot_dedup: bool = True,
        top_k: int = 100
    ) -> List[Tuple[Dict, float]]:
        """
        Executes hybrid retrieval and returns top_k (record, fused_score).
        """
        # 1. Dense similarity scores
        dense_scores = self.dense_retriever.get_all_scores(query_vec)

        # 2. Temporal Gaussian Smoothing
        if use_temporal_smoothing:
            dense_scores = self.apply_temporal_smoothing(dense_scores, sigma_frames=1.5, window_size=2)

        # 3. Object Detection Entity Boosting
        object_scores = np.zeros_like(dense_scores)
        if self.object_indexer is not None and query_text_en:
            tokens = query_text_en.lower().split()
            for token in tokens:
                matched_frames = self.object_indexer.search_entity(token)
                if matched_frames:
                    for i, rec in enumerate(self.records):
                        kf_name = rec.get("keyframe_name", f"{rec.get('frame_idx', 0):04d}")
                        key = f"{rec['video_id']}/{kf_name}"
                        if key in matched_frames:
                            object_scores[i] += matched_frames[key]


        # Normalize object scores to [0, 1]
        max_obj = np.max(object_scores)
        if max_obj > 0:
            object_scores = object_scores / max_obj

        # 4. Metadata BM25 Boosting
        meta_scores = np.zeros_like(dense_scores)
        if self.metadata_indexer is not None:
            text_for_meta = f"{query_text_vi} {query_text_en}".strip()
            meta_res = self.metadata_indexer.query(text_for_meta, top_k=100)
            if meta_res:
                max_bm = max(meta_res.values())
                for i, rec in enumerate(self.records):
                    vid = rec["video_id"]
                    if vid in meta_res and max_bm > 0:
                        meta_scores[i] = meta_res[vid] / max_bm

        # 5. Composite Score Fusion
        final_scores = (
            w_dense * dense_scores +
            w_object * object_scores +
            w_meta * meta_scores
        )

        # 6. Candidate Sorting & Submodular Shot Deduplication
        sorted_indices = np.argsort(final_scores)[::-1]

        if not use_shot_dedup:
            top_indices = sorted_indices[:top_k]
            return [(self.records[idx], float(final_scores[idx])) for idx in top_indices]

        # Deduplication Strategy: Max 1 frame per 2-second timestamp window in top ranks
        selected = []
        seen_shots = set()  # (video_id, shot_bucket)

        for idx in sorted_indices:
            rec = self.records[idx]
            vid = rec["video_id"]
            pts = rec["pts_time"]
            shot_bucket = (vid, int(pts // 2.5))  # 2.5s bucket

            if shot_bucket not in seen_shots:
                selected.append(idx)
                seen_shots.add(shot_bucket)
                if len(selected) >= top_k:
                    break

        # If not enough unique shots, fill remaining from sorted
        if len(selected) < top_k:
            for idx in sorted_indices:
                if idx not in selected:
                    selected.append(idx)
                    if len(selected) >= top_k:
                        break

        return [(self.records[idx], float(final_scores[idx])) for idx in selected]
