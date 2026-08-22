from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Any
import numpy as np

from src.index.transcript_semantic_index import TranscriptSemanticIndex
from src.retrieval.reranker import BGEReranker


class HybridTranscriptEngine:
    """
    Unified Hybrid Speech Transcript Search Engine.
    Combines:
    1. Lexical BM25 (Exact keywords, proper nouns, numbers, acronyms)
    2. Dense Semantic Vector Retrieval (Paraphrase, semantic intent, topic similarity)
    3. Neural Cross-Encoder Reranking (BGE-Reranker-v2-m3)
    4. Temporal timestamp mapping to video keyframes.
    """

    def __init__(
        self,
        bm25_engine: Any,
        semantic_index: Optional[TranscriptSemanticIndex] = None,
        reranker: Optional[BGEReranker] = None
    ):
        self.bm25_engine = bm25_engine
        self.semantic_index = semantic_index
        self.reranker = reranker

    def search_segments(
        self,
        query: str,
        w_bm25: float = 0.50,
        w_semantic: float = 0.50,
        top_k: int = 200,
        fusion_method: str = "weighted_norm",  # "weighted_norm" or "rrf"
        rerank_top_n: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Executes hybrid lexical + semantic search over speech transcript segments.
        Returns sorted list of segment match records with composite score.
        """
        if not query or not query.strip():
            return []

        query = query.strip()
        bm25_scores = {}
        bm25_docs = {}

        # 1. Lexical BM25 Search
        if self.bm25_engine is not None and w_bm25 > 0:
            bm25_hits = self.bm25_engine.search(query, top_k=top_k * 2)
            if bm25_hits:
                max_bm = max(s for _, s in bm25_hits)
                for doc_idx, b_score in bm25_hits:
                    doc = self.bm25_engine.docs[doc_idx]
                    key = (doc["video_id"], round(doc["start_sec"], 2), round(doc["end_sec"], 2))
                    norm_score = b_score / max(1e-6, max_bm)
                    bm25_scores[key] = norm_score
                    bm25_docs[key] = doc

        # 2. Dense Semantic Search
        sem_scores = {}
        sem_docs = {}
        if self.semantic_index is not None and w_semantic > 0:
            sem_hits = self.semantic_index.query(query, top_k=top_k * 2)
            if sem_hits:
                max_sem = max(s for _, s in sem_hits)
                min_sem = min(s for _, s in sem_hits)
                denom = max(1e-6, max_sem - min_sem)
                for seg_meta, s_score in sem_hits:
                    key = (seg_meta["video_id"], round(seg_meta["start_sec"], 2), round(seg_meta["end_sec"], 2))
                    norm_score = (s_score - min_sem) / denom if denom > 0 else s_score
                    sem_scores[key] = norm_score
                    sem_docs[key] = seg_meta

        # 3. Fusion
        all_keys = set(bm25_scores.keys()) | set(sem_scores.keys())
        if not all_keys:
            return []

        fused_results = []

        if fusion_method == "rrf":
            # Reciprocal Rank Fusion (k=60)
            rrf_k = 60.0
            bm25_rank = {k: r for r, (k, _) in enumerate(sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True), 1)}
            sem_rank = {k: r for r, (k, _) in enumerate(sorted(sem_scores.items(), key=lambda x: x[1], reverse=True), 1)}

            for k in all_keys:
                r_bm = bm25_rank.get(k, 1000)
                r_sem = sem_rank.get(k, 1000)
                score = (w_bm25 / (rrf_k + r_bm)) + (w_semantic / (rrf_k + r_sem))
                doc = bm25_docs.get(k) or sem_docs.get(k)
                fused_results.append({
                    "video_id": doc["video_id"],
                    "start_sec": doc["start_sec"],
                    "end_sec": doc["end_sec"],
                    "text": doc["text"],
                    "bm25_score": bm25_scores.get(k, 0.0),
                    "semantic_score": sem_scores.get(k, 0.0),
                    "fused_score": score
                })
        else:
            # Weighted Normalized Score Combination
            for k in all_keys:
                s_bm = bm25_scores.get(k, 0.0)
                s_sem = sem_scores.get(k, 0.0)
                fused_score = (w_bm25 * s_bm) + (w_semantic * s_sem)
                doc = bm25_docs.get(k) or sem_docs.get(k)
                fused_results.append({
                    "video_id": doc["video_id"],
                    "start_sec": doc["start_sec"],
                    "end_sec": doc["end_sec"],
                    "text": doc["text"],
                    "bm25_score": s_bm,
                    "semantic_score": s_sem,
                    "fused_score": fused_score
                })

        fused_results.sort(key=lambda x: x["fused_score"], reverse=True)

        # 4. Neural Cross-Encoder Reranking (BGE-Reranker-v2-m3)
        if self.reranker is not None and fused_results:
            top_candidates = fused_results[:rerank_top_n]
            reranked = self.reranker.rerank(query, top_candidates, top_k=rerank_top_n)
            # Replace top candidate scores with reranker scores
            for item in reranked:
                item["fused_score"] = item.get("rerank_score", item["fused_score"])
            
            # Combine reranked top candidates with the remainder
            fused_results = reranked + fused_results[rerank_top_n:]

        return fused_results[:top_k]

    def compute_keyframe_speech_scores(
        self,
        query: str,
        records: List[Dict],
        video_to_records: Dict[str, List[int]],
        w_bm25: float = 0.50,
        w_semantic: float = 0.50,
        temporal_window_sec: float = 3.0,
        top_k: int = 200
    ) -> Tuple[np.ndarray, Dict[int, str]]:
        """
        Maps hybrid transcript segment scores to video keyframe array.
        Returns (keyframe_speech_scores_array, {keyframe_idx: matched_speech_text}).
        """
        scores = np.zeros(len(records), dtype=np.float32)
        matched_texts = {}

        if not query or not query.strip():
            return scores, matched_texts

        segment_hits = self.search_segments(query, w_bm25=w_bm25, w_semantic=w_semantic, top_k=top_k)
        if not segment_hits:
            return scores, matched_texts

        max_fused = max(hit["fused_score"] for hit in segment_hits) if segment_hits else 1.0

        for hit in segment_hits:
            vid = hit["video_id"]
            st = hit["start_sec"]
            et = hit["end_sec"]
            txt = hit["text"]
            norm_score = hit["fused_score"] / max(1e-6, max_fused)

            kf_indices = video_to_records.get(vid, [])
            for k_idx in kf_indices:
                pts = records[k_idx]["pts_time"]
                # Temporal overlap window
                if (st - temporal_window_sec) <= pts <= (et + temporal_window_sec):
                    if norm_score > scores[k_idx]:
                        scores[k_idx] = norm_score
                        matched_texts[k_idx] = txt

        return scores, matched_texts
