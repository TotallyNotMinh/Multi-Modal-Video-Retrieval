import os
import glob
import json
import pickle
import re
from typing import Dict, List, Tuple, Optional
from rank_bm25 import BM25Okapi

class MetadataIndexer:
    """
    Unified Multi-Modal Lexical Indexer.
    Ingests:
    1. YouTube metadata (title, description, tags, keywords)
    2. Whisper ASR speech transcripts (Vietnamese)
    3. On-screen OCR text banners
    """
    def __init__(
        self,
        media_info_dir: str = "data/media-info-aic25-b1/media-info",
        asr_dir: str = "cache/asr_transcripts",
        ocr_dir: str = "cache/ocr_text",
        cache_dir: str = "cache"
    ):
        self.media_info_dir = media_info_dir
        if not os.path.exists(media_info_dir) and os.path.exists("cache/media-info-aic25-b1/media-info"):
            self.media_info_dir = "cache/media-info-aic25-b1/media-info"
        self.asr_dir = asr_dir
        self.ocr_dir = ocr_dir
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.video_ids: List[str] = []
        self.corpus: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None
        self.metadata_dict: Dict[str, Dict] = {}

    def _tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        text = text.lower()
        tokens = re.findall(r"\w+", text, flags=re.UNICODE)
        return tokens

    def build_and_cache(self, force: bool = False):
        cache_path = os.path.join(self.cache_dir, "unified_metadata_bm25.pkl")
        if not force and os.path.exists(cache_path):
            print(f"[MetadataIndexer] Loading cached BM25 index from {cache_path}...")
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
                self.video_ids = data["video_ids"]
                self.bm25 = data["bm25"]
                self.metadata_dict = data["metadata_dict"]
            print(f"[MetadataIndexer] Loaded lexical index for {len(self.video_ids)} videos.")
            return self

        print("[MetadataIndexer] Building unified BM25 index from YouTube info, ASR, and OCR...")
        json_files = glob.glob(os.path.join(self.media_info_dir, "*.json"))
        print(f"Found {len(json_files)} media-info json files.")

        self.video_ids = []
        self.corpus = []
        self.metadata_dict = {}

        for jf in json_files:
            video_id = os.path.splitext(os.path.basename(jf))[0]
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                continue

            title = meta.get("title", "")
            description = meta.get("description", "")
            keywords = " ".join(meta.get("keywords", [])) if isinstance(meta.get("keywords"), list) else str(meta.get("keywords", ""))
            tags = " ".join(meta.get("tags", [])) if isinstance(meta.get("tags"), list) else str(meta.get("tags", ""))

            # Ingest ASR if available
            asr_text = ""
            asr_file = os.path.join(self.asr_dir, f"{video_id}.json")
            if os.path.exists(asr_file):
                try:
                    with open(asr_file, "r", encoding="utf-8") as af:
                        asr_segs = json.load(af)
                        asr_text = " ".join([s.get("text", "") for s in asr_segs])
                except Exception:
                    pass

            # Ingest OCR if available
            ocr_text = ""
            ocr_file = os.path.join(self.ocr_dir, f"{video_id}.json")
            if os.path.exists(ocr_file):
                try:
                    with open(ocr_file, "r", encoding="utf-8") as of:
                        ocr_data = json.load(of)
                        ocr_text = " ".join(ocr_data.values()) if isinstance(ocr_data, dict) else ""
                except Exception:
                    pass

            full_text = f"{title} {keywords} {tags} {asr_text} {ocr_text} {description}"
            tokens = self._tokenize(full_text)

            self.video_ids.append(video_id)
            self.corpus.append(tokens)
            self.metadata_dict[video_id] = {
                "title": title,
                "description": description[:300],
                "keywords": keywords,
                "has_asr": bool(asr_text),
                "has_ocr": bool(ocr_text)
            }

        self.bm25 = BM25Okapi(self.corpus)
        
        with open(cache_path, "wb") as f:
            pickle.dump({
                "video_ids": self.video_ids,
                "bm25": self.bm25,
                "metadata_dict": self.metadata_dict
            }, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"[MetadataIndexer] Saved unified BM25 index to {cache_path}")
        return self

    def query(self, text: str, top_k: int = 50) -> Dict[str, float]:
        """
        Returns {video_id: bm25_score} for top matching videos.
        """
        if self.bm25 is None or not self.video_ids:
            return {}
        tokens = self._tokenize(text)
        if not tokens:
            return {}
        scores = self.bm25.get_scores(tokens)
        
        top_indices = scores.argsort()[::-1][:top_k]
        result = {}
        for idx in top_indices:
            if scores[idx] > 0:
                result[self.video_ids[idx]] = float(scores[idx])
        return result
