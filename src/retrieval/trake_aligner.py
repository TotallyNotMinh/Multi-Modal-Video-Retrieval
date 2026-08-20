import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict
from src.retrieval.dense_retriever import DenseRetriever

class TRAKEAligner:
    """
    Monotonic Dynamic Programming Alignment Engine for TRAKE Queries.

    TRAKE pipeline is split into two explicit phases:

    Phase 1 — Video Retrieval (this class, IN SCOPE):
        Given N sequential sub-event queries [e_1, ..., e_N], ranks candidate videos
        by computing an optimal strictly monotonic keyframe sequence (f_1 < f_2 < ... < f_N)
        over the scene-adaptive FAISS index. Returns the top-K videos by alignment score.
        The `aligned_frames` in the output are best-effort estimates from the scene index
        (nearest sampled scene boundary frame to each event's approximate location).

    Phase 2 — Exact Semantic Keyframe Localization (FUTURE WORK / VLM):
        Once the correct video is identified by Phase 1, a VLM (e.g. Gemini, InternVL)
        is called on the raw `.mp4` to precisely extract the N semantic keyframe indices
        within ground-truth windows of ≤ 10 frames. This phase is NOT implemented here.

    Note on current TRAKE submission quality:
        With Phase 1 only, TRAKE R-Score is limited by the scene index resolution
        (~1–3 frames per shot). Exact sub-10-frame window accuracy requires Phase 2 VLM.
    """

    def __init__(self, dense_retriever: DenseRetriever):
        self.dense_retriever = dense_retriever
        self.records = dense_retriever.records
        self.matrix = dense_retriever.matrix
        
        # Pre-group records by video
        self.video_to_indices = defaultdict(list)
        for idx, rec in enumerate(self.records):
            self.video_to_indices[rec["video_id"]].append(idx)

    def align_sequence(
        self,
        event_vectors: List[np.ndarray],  # List of N normalized vectors
        top_k_videos: int = 10
    ) -> List[Dict]:
        """
        Returns top candidate videos with their aligned keyframe sequences and total alignment score.
        """
        N_events = len(event_vectors)
        if N_events == 0:
            return []

        # 1. Video Candidate Ranking via Mean Event Vector
        mean_event_vec = np.mean(event_vectors, axis=0)
        norm = np.linalg.norm(mean_event_vec)
        if norm > 1e-12:
            mean_event_vec = mean_event_vec / norm
            
        all_dense_scores = self.dense_retriever.get_all_scores(mean_event_vec)

        video_max_scores = {}
        for vid, indices in self.video_to_indices.items():
            if indices:
                video_max_scores[vid] = float(np.max(all_dense_scores[indices]))

        top_vids = sorted(video_max_scores.keys(), key=lambda v: video_max_scores[v], reverse=True)[:top_k_videos]

        results = []

        # 2. Monotonic Dynamic Programming Alignment per candidate video
        for vid in top_vids:
            indices = self.video_to_indices[vid]
            T_frames = len(indices)
            if T_frames < N_events:
                continue

            vid_matrix = self.matrix[indices]  # (T_frames, D)
            
            # S[i, j] = cos_sim(event_i, frame_j)
            S = np.dot(event_vectors, vid_matrix.T)  # (N, T)

            dp = np.full((N_events, T_frames), -1e9, dtype=np.float32)
            parent = np.full((N_events, T_frames), -1, dtype=np.int32)

            dp[0, :] = S[0, :]

            for i in range(1, N_events):
                # Vectorized / running maximum over past frames k < j
                running_max_val = -1e9
                best_k = -1
                for j in range(i, T_frames):
                    prev_val = dp[i - 1, j - 1]
                    if prev_val > running_max_val:
                        running_max_val = prev_val
                        best_k = j - 1
                    if best_k != -1:
                        dp[i, j] = S[i, j] + running_max_val
                        parent[i, j] = best_k

            best_last_j = int(np.argmax(dp[N_events - 1, :]))
            best_total_score = float(dp[N_events - 1, best_last_j])

            if best_total_score <= -1e8:
                continue

            aligned_local_indices = [best_last_j]
            curr_j = best_last_j
            for i in range(N_events - 1, 0, -1):
                curr_j = parent[i, curr_j]
                aligned_local_indices.append(curr_j)

            aligned_local_indices = aligned_local_indices[::-1]  # Monotonic order: [j_1 < j_2 < ... < j_N]

            aligned_records = [self.records[indices[j]] for j in aligned_local_indices]
            aligned_frames = [rec["frame_idx"] for rec in aligned_records]

            results.append({
                "video_id": vid,
                "score": best_total_score / N_events,
                "aligned_frames": aligned_frames,
                "aligned_records": aligned_records
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results
