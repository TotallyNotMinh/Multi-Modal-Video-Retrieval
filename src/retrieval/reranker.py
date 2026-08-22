"""
BGE Cross-Encoder Reranker for Vietnamese Video Speech Transcripts.
Provides high-precision neural reranking using BAAI/bge-reranker-v2-m3.
"""

import os
import sys
import time
from typing import List, Dict, Tuple, Optional, Any, Union
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Ensure workspace root is in sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


class BGEReranker:
    """
    Local Cross-Encoder Reranker using BAAI/bge-reranker-v2-m3.
    Re-scores and re-ranks top candidate speech transcript passages given a user query.
    """

    def __init__(
        self,
        model_name_or_path: str = "BAAI/bge-reranker-v2-m3",
        device: Optional[str] = None,
        use_fp16: bool = True,
        batch_size: int = 32,
        max_length: int = 256
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model_name = model_name_or_path
        self.use_fp16 = use_fp16 and (self.device == "cuda" or "cuda" in str(self.device))
        self.batch_size = max(1, int(batch_size))
        self.max_length = max_length

        print(f"[BGEReranker] Loading '{model_name_or_path}' on device '{self.device}' (fp16={self.use_fp16}, batch_size={self.batch_size}, max_length={self.max_length})...")
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        torch_dtype = torch.float16 if self.use_fp16 else torch.float32
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"[BGEReranker] Loaded reranker model in {time.time() - t0:.2f}s.")

    @torch.inference_mode()
    def compute_scores(self, pairs: List[Tuple[str, str]], batch_size: Optional[int] = None) -> np.ndarray:
        """
        Computes relevance scores for a list of (query, passage) text pairs.
        Returns float32 numpy array of sigmoid-normalized relevance scores in [0, 1].
        """
        if not pairs:
            return np.empty((0,), dtype=np.float32)

        bs = max(1, int(batch_size or self.batch_size))
        all_scores = []

        for i in range(0, len(pairs), bs):
            batch_pairs = pairs[i : i + bs]
            queries = [p[0] for p in batch_pairs]
            passages = [p[1] for p in batch_pairs]

            encoded = self.tokenizer(
                queries,
                passages,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)

            logits = self.model(**encoded).logits.view(-1)
            # Apply Sigmoid to convert logits into calibrated probabilities [0, 1]
            probs = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
            all_scores.extend(probs)

        return np.array(all_scores, dtype=np.float32)

    def rerank(
        self,
        query: str,
        candidates: List[Union[Dict[str, Any], Tuple[Dict[str, Any], float]]],
        top_k: int = 5,
        batch_size: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Reranks a list of candidate segment dictionaries or (segment_dict, dense_score) tuples.
        
        Preserves all metadata:
          - video_id
          - segment_id
          - start_sec, end_sec
          - start_frame, end_frame
          - text (refined transcript)
          - dense_score (original E5 score)
          - rerank_score (BGE cross-encoder score)
        
        Returns top_k candidate records sorted descending by rerank_score.
        """
        if not candidates or not query.strip():
            return []

        query = query.strip()
        normalized_candidates: List[Dict[str, Any]] = []

        for item in candidates:
            if isinstance(item, tuple):
                seg_dict, d_score = item
                c_copy = dict(seg_dict)
                c_copy["dense_score"] = float(d_score)
            elif isinstance(item, dict):
                c_copy = dict(item)
                if "dense_score" not in c_copy:
                    c_copy["dense_score"] = float(c_copy.get("score", 0.0))
            else:
                continue
            normalized_candidates.append(c_copy)

        if not normalized_candidates:
            return []

        # Construct pairs: (query, candidate_text)
        pairs = [(query, c.get("text", c.get("refined_text", "")).strip()) for c in normalized_candidates]

        try:
            rerank_scores = self.compute_scores(pairs, batch_size=batch_size)
            for c, r_score in zip(normalized_candidates, rerank_scores):
                c["rerank_score"] = float(r_score)

            # Sort descending by BGE rerank score
            sorted_candidates = sorted(normalized_candidates, key=lambda x: x["rerank_score"], reverse=True)
            return sorted_candidates[:top_k]

        except Exception as e:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[BGEReranker] Warning: Reranking failed ({e}), falling back to dense ranking.", file=sys.stderr)
            # Fallback to original dense ranking
            sorted_candidates = sorted(normalized_candidates, key=lambda x: x.get("dense_score", 0.0), reverse=True)
            for c in sorted_candidates:
                c["rerank_score"] = c.get("dense_score", 0.0)
            return sorted_candidates[:top_k]
