"""
Transcript Semantic Indexer & FAISS Dense Vector Database for Vietnamese Transcripts.
Extracts refined_text segments, generates embeddings, and executes fast semantic similarity search.
"""

import os
import glob
import json
import time
import pickle
from typing import List, Dict, Tuple, Optional, Any
import numpy as np

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

from src.encoding.transcript_encoder import TranscriptEncoder


class TranscriptSemanticIndex:
    """
    Dense Semantic Index for speech transcript segments.
    Uses FAISS IndexFlatIP (exact cosine similarity over normalized embeddings) or NumPy matrix fallback.
    """

    def __init__(
        self,
        refined_asr_dir: str = "cache/asr_transcripts_refined",
        raw_asr_dir: str = "cache/asr_transcripts",
        cache_dir: str = "cache",
        encoder: Optional[TranscriptEncoder] = None
    ):
        self.refined_asr_dir = refined_asr_dir
        self.raw_asr_dir = raw_asr_dir
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        self.matrix_path = os.path.join(cache_dir, "transcript_embeddings.npy")
        self.meta_path = os.path.join(cache_dir, "transcript_semantic_meta.pkl")
        self.faiss_path = os.path.join(cache_dir, "transcript_semantic.index")

        self.encoder = encoder
        self.segments_meta: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.faiss_index: Optional[Any] = None

    def _get_encoder(self) -> TranscriptEncoder:
        if self.encoder is None:
            self.encoder = TranscriptEncoder()
        return self.encoder

    def load_or_build(self, force_rebuild: bool = False) -> "TranscriptSemanticIndex":
        """
        Loads existing precomputed index from disk if available, otherwise builds from transcripts.
        """
        if not force_rebuild and os.path.exists(self.matrix_path) and os.path.exists(self.meta_path):
            print(f"[TranscriptSemanticIndex] Loading cached index from {self.meta_path}...")
            t0 = time.time()
            with open(self.meta_path, "rb") as f:
                self.segments_meta = pickle.load(f)

            self.embeddings = np.load(self.matrix_path)

            if HAS_FAISS and os.path.exists(self.faiss_path):
                self.faiss_index = faiss.read_index(self.faiss_path)
            elif HAS_FAISS:
                dim = self.embeddings.shape[1]
                self.faiss_index = faiss.IndexFlatIP(dim)
                self.faiss_index.add(self.embeddings)

            print(f"[TranscriptSemanticIndex] Loaded {len(self.segments_meta)} segment vectors ({self.embeddings.shape}) in {time.time() - t0:.2f}s.")
            return self

        return self.build()

    def build(self) -> "TranscriptSemanticIndex":
        """
        Gathers all transcript segments (preferring refined_text from refined_asr_dir,
        falling back to raw_asr_dir), encodes them, and saves to FAISS & NumPy matrix.
        """
        print("[TranscriptSemanticIndex] Building dense transcript index from speech files...")
        t0 = time.time()

        # Discover all available video IDs from both refined and raw directories
        raw_files = sorted(glob.glob(os.path.join(self.raw_asr_dir, "*.json")))
        refined_files = sorted(glob.glob(os.path.join(self.refined_asr_dir, "*.json")))

        all_video_ids = set()
        for f in raw_files + refined_files:
            all_video_ids.add(os.path.basename(f).replace(".json", ""))

        all_video_ids = sorted(list(all_video_ids))
        print(f"[TranscriptSemanticIndex] Found {len(all_video_ids)} total videos to index ({len(refined_files)} refined).")

        collected_segments = []
        passages_to_encode = []

        for vid in all_video_ids:
            refined_path = os.path.join(self.refined_asr_dir, f"{vid}.json")
            raw_path = os.path.join(self.raw_asr_dir, f"{vid}.json")

            data = None
            is_refined = False

            if os.path.exists(refined_path):
                try:
                    with open(refined_path, "r", encoding="utf-8") as fp:
                        data = json.load(fp)
                        is_refined = True
                except Exception:
                    data = None

            if data is None and os.path.exists(raw_path):
                try:
                    with open(raw_path, "r", encoding="utf-8") as fp:
                        data = json.load(fp)
                        is_refined = False
                except Exception:
                    data = None

            if not data:
                continue

            if isinstance(data, dict):
                data = data.get("segments", [])

            for s in data:
                if is_refined:
                    txt = s.get("refined_text", s.get("text", "")).strip()
                else:
                    txt = s.get("cleaned_text", s.get("text", "")).strip()

                if not txt:
                    continue

                st = float(s.get("start_sec", s.get("start", 0.0)))
                et = float(s.get("end_sec", s.get("end", st + 1.0)))
                seg_id = int(s.get("segment_id", len(collected_segments)))

                record = {
                    "video_id": vid,
                    "segment_id": seg_id,
                    "start_sec": st,
                    "end_sec": et,
                    "start_frame": s.get("start_frame", 0),
                    "end_frame": s.get("end_frame", 0),
                    "text": txt,
                    "is_refined": is_refined
                }

                collected_segments.append(record)
                passages_to_encode.append(txt)

        print(f"[TranscriptSemanticIndex] Total speech segments extracted: {len(collected_segments):,}")

        if not passages_to_encode:
            print("[TranscriptSemanticIndex] Warning: No segments found to encode.")
            self.segments_meta = []
            encoder = self._get_encoder()
            self.embeddings = np.empty((0, encoder.dim), dtype=np.float32)
            return self

        # Dense encoding
        encoder = self._get_encoder()
        print(f"[TranscriptSemanticIndex] Encoding {len(passages_to_encode):,} passages with batch_size={encoder.batch_size}...")
        enc_t0 = time.time()
        self.embeddings = encoder.encode_passages(passages_to_encode)
        print(f"[TranscriptSemanticIndex] Encoded embeddings matrix {self.embeddings.shape} in {time.time() - enc_t0:.2f}s.")

        self.segments_meta = collected_segments

        # Save numpy matrix and metadata
        np.save(self.matrix_path, self.embeddings)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.segments_meta, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Build and save FAISS index
        if HAS_FAISS:
            dim = self.embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dim)
            self.faiss_index.add(self.embeddings)
            faiss.write_index(self.faiss_index, self.faiss_path)

        print(f"[TranscriptSemanticIndex] Saved index files to '{self.cache_dir}' in {time.time() - t0:.2f}s total.")
        return self

    def query(self, query_text: str, top_k: int = 200) -> List[Tuple[Dict[str, Any], float]]:
        """
        Executes semantic search for `query_text` and returns top_k (segment_dict, cosine_similarity).
        """
        if self.embeddings is None or len(self.segments_meta) == 0:
            return []

        encoder = self._get_encoder()
        q_vec = encoder.encode_query(query_text)  # (dim,)

        if HAS_FAISS and self.faiss_index is not None:
            q_mat = np.expand_dims(q_vec, axis=0).astype(np.float32)
            scores, indices = self.faiss_index.search(q_mat, min(top_k, len(self.segments_meta)))
            results = []
            for idx, score in zip(indices[0], scores[0]):
                if idx >= 0:
                    results.append((self.segments_meta[idx], float(score)))
            return results
        else:
            # Fallback exact dot product
            scores = np.dot(self.embeddings, q_vec)
            k = min(top_k, len(scores))
            top_indices = np.argpartition(scores, -k)[-k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
            return [(self.segments_meta[i], float(scores[i])) for i in top_indices]
