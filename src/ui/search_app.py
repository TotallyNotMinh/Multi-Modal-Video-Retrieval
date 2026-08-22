import os
import sys

# Silence non-critical FFmpeg/H.264 decoder seeking warnings (e.g., mmco: unref short failure)
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

import glob
import json
import time
import pickle
import base64
import urllib.parse
import urllib.request
import re
import math
from collections import Counter, defaultdict
import numpy as np
import torch
import cv2
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from typing import List, Dict, Tuple, Optional, Any

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.query.text_encoder import CLIPTextEncoder
from src.query.translator import QueryTranslator
from src.index.transcript_semantic_index import TranscriptSemanticIndex
from src.retrieval.reranker import BGEReranker
from src.retrieval.context_expander import TranscriptContextExpander


def sanitize_video_id(video_id: Any) -> str:
    """Sanitizes video_id against path traversal and special characters."""
    if not video_id:
        return ""
    cleaned = os.path.basename(str(video_id).strip())
    return re.sub(r"[^a-zA-Z0-9_\-]", "", cleaned)


# Vietnamese Stopwords & Common Noise Words to Filter
VI_STOPWORDS = {
    "là", "của", "và", "có", "trong", "được", "cho", "với", "các", "ở", "này",
    "đã", "để", "những", "khi", "ra", "đến", "về", "như", "tại", "từ", "vào",
    "bị", "bởi", "lại", "thì", "mà", "cũng", "đang", "sẽ", "phải", "qua",
    "sau", "trước", "theo", "gì", "ai", "nào", "đâu", "đây", "đó", "hãy",
    "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín", "mười",
    "nhiều", "rất", "quá", "lắm", "hơn", "nhất", "như_thế_nào", "tại_sao"
}


class BM25Engine:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs = []  # List[Dict] with 'video_id', 'start_sec', 'end_sec', 'text'
        self.doc_len = []
        self.avgdl = 0.0
        self.inverted_index = defaultdict(list)  # word -> List[(doc_idx, tf)]
        self.df = defaultdict(int)  # word -> doc_freq
        self.N = 0
        self.video_asr_timeline = defaultdict(list)  # video_id -> List[(st, et, txt)]

    def _tokenize(self, text: str) -> List[str]:
        # Normalize and split into lowercased alphanumeric tokens
        clean_text = re.sub(r"[^\w\s]", " ", text.lower())
        tokens = [t for t in clean_text.split() if len(t) > 1 or t.isalnum()]
        return tokens

    def index_transcripts(self, refined_dir: str = "cache/asr_transcripts_refined", raw_dir: str = "cache/asr_transcripts"):
        print("   • Indexing Vietnamese ASR transcripts for BM25 & Temporal Subtitles...")
        t0 = time.time()
        
        # 1. Discover all transcript JSON files (Prioritize refined)
        refined_files = {
            os.path.splitext(os.path.basename(p))[0]: p
            for p in glob.glob(f"{refined_dir}/*.json")
        }
        raw_files = {
            os.path.splitext(os.path.basename(p))[0]: p
            for p in glob.glob(f"{raw_dir}/*.json")
        }
        
        all_vids = sorted(list(set(refined_files.keys()) | set(raw_files.keys())))
        if not all_vids:
            print("   • Warning: No ASR transcripts found to index.")
            return

        for vid in all_vids:
            f_path = refined_files.get(vid) or raw_files.get(vid)
            is_refined = vid in refined_files
            try:
                with open(f_path, "r", encoding="utf-8") as fp:
                    data = json.load(fp)

                segments = []
                if is_refined and isinstance(data, list):
                    segments = data
                elif isinstance(data, dict):
                    segments = data.get("segments", [])
                elif isinstance(data, list):
                    segments = data

                for s in segments:
                    st = float(s.get("start_sec", s.get("start", 0.0)))
                    et = float(s.get("end_sec", s.get("end", st + 1.0)))
                    txt = s.get("refined_text", s.get("cleaned_text", s.get("text", ""))).strip()
                    if not txt:
                        continue

                    doc_idx = len(self.docs)
                    self.docs.append({
                        "video_id": vid,
                        "start_sec": st,
                        "end_sec": et,
                        "text": txt
                    })
                    self.video_asr_timeline[vid].append((st, et, txt))

                    tokens = self._tokenize(txt)
                    self.doc_len.append(len(tokens))
                    
                    tf_counter = Counter(tokens)
                    for term, tf in tf_counter.items():
                        self.inverted_index[term].append((doc_idx, tf))
                        self.df[term] += 1

            except Exception:
                continue

        # Sort timeline segments for every video by timestamp
        for vid in self.video_asr_timeline:
            self.video_asr_timeline[vid].sort(key=lambda x: x[0])

        self.N = len(self.docs)
        self.avgdl = sum(self.doc_len) / self.N if self.N > 0 else 0.0
        print(f"   • BM25 indexed {self.N} speech segments across {len(all_vids)} videos in {time.time() - t0:.2f}s.")

    def index_ocr(self, ocr_dir: str = "cache/ocr_text", fps: float = 25.0):
        if not os.path.exists(ocr_dir):
            return
        t0 = time.time()
        ocr_files = sorted(glob.glob(f"{ocr_dir}/*.json"))
        if not ocr_files:
            return

        ocr_added = 0
        for ocr_f in ocr_files:
            vid = os.path.splitext(os.path.basename(ocr_f))[0]
            try:
                with open(ocr_f, "r", encoding="utf-8") as fp:
                    ocr_data = json.load(fp)
            except Exception:
                continue
            if not isinstance(ocr_data, dict):
                continue

            entries = []
            for f_key, raw_text in ocr_data.items():
                cleaned = str(raw_text).strip()
                if not cleaned:
                    continue
                try:
                    f_idx = int(str(f_key).replace("f_", ""))
                    t_sec = f_idx / fps
                except ValueError:
                    t_sec = 0.0
                entries.append((t_sec, cleaned))

            entries.sort(key=lambda x: x[0])
            if not entries:
                continue

            cur_text = entries[0][1]
            cur_st = entries[0][0]
            cur_et = entries[0][0] + 3.0
            for t_sec, text in entries[1:]:
                if text == cur_text and t_sec <= (cur_et + 3.0):
                    cur_et = max(cur_et, t_sec + 3.0)
                else:
                    doc_idx = len(self.docs)
                    self.docs.append({"video_id": vid, "start_sec": cur_st, "end_sec": cur_et, "text": cur_text, "source": "ocr"})
                    tokens = self._tokenize(cur_text)
                    self.doc_len.append(len(tokens))
                    for term, tf in Counter(tokens).items():
                        self.inverted_index[term].append((doc_idx, tf))
                        self.df[term] += 1
                    ocr_added += 1
                    cur_text = text
                    cur_st = t_sec
                    cur_et = t_sec + 3.0

            doc_idx = len(self.docs)
            self.docs.append({"video_id": vid, "start_sec": cur_st, "end_sec": cur_et, "text": cur_text, "source": "ocr"})
            tokens = self._tokenize(cur_text)
            self.doc_len.append(len(tokens))
            for term, tf in Counter(tokens).items():
                self.inverted_index[term].append((doc_idx, tf))
                self.df[term] += 1
            ocr_added += 1

        self.N = len(self.docs)
        self.avgdl = sum(self.doc_len) / self.N if self.N > 0 else 0.0
        print(f"   • BM25 indexed {ocr_added} merged OCR banner segments across {len(ocr_files)} videos in {time.time() - t0:.2f}s (Total Docs: {self.N}).")

    def search(self, query: str, top_k: int = 300) -> List[Tuple[int, float]]:
        if not query or self.N == 0:
            return []

        raw_tokens = self._tokenize(query)
        if not raw_tokens:
            return []

        # Separate meaningful tokens vs stopwords
        content_tokens = [t for t in raw_tokens if t not in VI_STOPWORDS]
        q_tokens = content_tokens if content_tokens else raw_tokens

        if not q_tokens or self.N == 0:
            return []

        scores = defaultdict(float)
        avgdl = max(1e-6, self.avgdl)
        
        for qt in q_tokens:
            postings = self.inverted_index.get(qt)
            if not postings:
                continue
            n_q = self.df[qt]
            # IDF with Robertson-Spärck Jones formulation
            idf = math.log((self.N - n_q + 0.5) / (n_q + 0.5) + 1.0)
            
            for doc_idx, tf in postings:
                dl = self.doc_len[doc_idx]
                tf_norm = (tf * (self.k1 + 1.0)) / (tf + self.k1 * (1.0 - self.b + self.b * (dl / avgdl)))
                scores[doc_idx] += idf * tf_norm

        if not scores:
            return []

        top_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return top_docs


class SearchEngine:
    def __init__(self):
        print("🚀 Initializing AIC 2026 Multi-Modal Retrieval Studio...")
        t0 = time.time()
        
        # 1. Load Matrix & Metadata
        matrix_path = "cache/features_matrix.npy"
        meta_path = "cache/faiss_siglip_meta.pkl"
        
        if not os.path.exists(matrix_path) or not os.path.exists(meta_path):
            raise FileNotFoundError(f"Missing {matrix_path} or {meta_path}. Please check cache directory.")
            
        self.matrix = np.load(matrix_path, mmap_mode="r")
        with open(meta_path, "rb") as f:
            self.records = pickle.load(f)
            
        print(f"   • Loaded Feature Matrix: {self.matrix.shape} ({len(self.records)} keyframes)")

        # 2. Build Video-to-Keyframe & Global Index Mapping for Instant Lookup
        self.video_to_records = defaultdict(list)
        self.frame_lookup = {}
        for idx, rec in enumerate(self.records):
            self.video_to_records[rec["video_id"]].append(idx)
            self.frame_lookup[(rec["video_id"], rec["frame_idx"])] = idx

        # 3. Build Video Path Index
        self.video_paths = {}
        for p in glob.glob("data/Videos_*/video/*.mp4"):
            v_name = os.path.splitext(os.path.basename(p))[0]
            self.video_paths[v_name] = p
        print(f"   • Indexed {len(self.video_paths)} local MP4 video files.")

        # 4. Build BM25 Inverted Index over Refined Speech Transcripts
        self.bm25 = BM25Engine()
        self.bm25.index_transcripts("cache/asr_transcripts_refined", "cache/asr_transcripts")

        # 4a. Load Dense Semantic Transcript Index (E5 / FAISS)
        self.semantic_index = None
        try:
            self.semantic_index = TranscriptSemanticIndex().load_or_build()
            self.semantic_index._get_encoder()
            print(f"   • Loaded Dense Transcript Semantic Index ({len(self.semantic_index.segments_meta)} segments).")
        except Exception as e:
            print(f"   • Warning: Could not load TranscriptSemanticIndex: {e}")

        # 4b. Load Neural Cross-Encoder Reranker (BGE-Reranker-v2-m3)
        self.reranker = None
        try:
            self.reranker = BGEReranker(
                model_name_or_path="BAAI/bge-reranker-v2-m3",
                batch_size=32,
                max_length=256
            )
            print("   • Loaded BGE-Reranker-v2-m3 Neural Cross-Encoder.")
        except Exception as e:
            print(f"   • Warning: Could not load BGEReranker: {e}")

        # 4c. Load Transcript Context Expander & Deduplicator
        self.context_expander = None
        if self.semantic_index is not None and getattr(self.semantic_index, "segments_meta", None):
            self.context_expander = TranscriptContextExpander(self.semantic_index.segments_meta)
            print(f"   • Loaded Transcript Context Expander & Deduplicator ({len(self.semantic_index.segments_meta)} segments).")

        # 4d. Load OCR Data Index
        self.ocr_dir = "cache/ocr_text"
        self.ocr_cache = {}
        if os.path.exists(self.ocr_dir):
            for ocr_f in glob.glob(f"{self.ocr_dir}/*.json"):
                vid = os.path.splitext(os.path.basename(ocr_f))[0]
                try:
                    with open(ocr_f, "r", encoding="utf-8") as fp:
                        self.ocr_cache[vid] = json.load(fp)
                except Exception:
                    pass
        print(f"   • Loaded OCR annotations for {len(self.ocr_cache)} videos.")
        self.bm25.index_ocr(self.ocr_dir)

        # 5. Load Models
        self.encoder = CLIPTextEncoder(device="cuda" if torch.cuda.is_available() else "cpu")
        self.translator = QueryTranslator(use_online=True)

        # 6. Thumbnail Memory + Disk Cache
        self.thumb_dir = "cache/thumbnails"
        os.makedirs(self.thumb_dir, exist_ok=True)
        self.thumb_mem_cache: Dict[str, bytes] = {}
        self.max_thumb_mem_cache = 4096
        
        print(f"✅ Search Engine ready in {time.time() - t0:.2f}s!")

    def get_asr_at_pts(self, video_id: str, pts_time: float, max_dist_sec: float = 15.0) -> str:
        """
        Retrieves the exact active speech transcript at pts_time, or nearest segment within max_dist_sec.
        """
        video_id = sanitize_video_id(video_id)
        timeline = self.bm25.video_asr_timeline.get(video_id, [])
        if not timeline:
            return ""

        # 1. Exact active segment (with 1.5s tolerance around boundaries)
        for st, et, txt in timeline:
            if (st - 1.5) <= pts_time <= (et + 1.5):
                return txt

        # 2. Nearest segment within max_dist_sec
        best_txt = ""
        min_dist = float("inf")
        best_st = 0.0
        for st, et, txt in timeline:
            mid = (st + et) / 2.0
            dist = abs(mid - pts_time)
            if dist < min_dist:
                min_dist = dist
                best_txt = txt
                best_st = st

        if min_dist <= max_dist_sec and best_txt:
            return f"[{best_st:.1f}s] {best_txt}"
        return ""

    def get_video_context(self, video_id: str, pts_time: float = 0.0, window_sec: float = 45.0) -> Dict:
        """
        Extracts temporal speech transcripts (ASR) and on-screen OCR text around pts_time for video_id.
        """
        video_id = sanitize_video_id(video_id)
        if not video_id:
            return {"video_id": "", "pts_time": pts_time, "asr_segments": [], "ocr_lines": []}

        # 1. Fast In-Memory ASR Timeline Lookup
        timeline = self.bm25.video_asr_timeline.get(video_id, [])
        segments = []
        for st, et, txt in timeline:
            is_near = (abs(st - pts_time) <= window_sec) or (abs(et - pts_time) <= window_sec) or (st <= pts_time <= et)
            segments.append({
                "start_sec": round(st, 2),
                "end_sec": round(et, 2),
                "text": txt,
                "is_near": is_near
            })

        # 2. OCR Data
        ocr_data = self.ocr_cache.get(video_id, {})
        if not ocr_data and os.path.exists(os.path.join("cache/ocr_text", f"{video_id}.json")):
            try:
                with open(os.path.join("cache/ocr_text", f"{video_id}.json"), "r", encoding="utf-8") as fp:
                    ocr_data = json.load(fp)
            except Exception:
                ocr_data = {}

        ocr_lines = []
        fps = 25.0
        if isinstance(ocr_data, dict):
            for f_key, raw_text in ocr_data.items():
                try:
                    f_idx = int(str(f_key).replace("f_", ""))
                    t_sec = f_idx / fps
                except ValueError:
                    t_sec = 0.0

                cleaned = str(raw_text).strip()
                if cleaned:
                    ocr_lines.append({
                        "frame_idx": f_key,
                        "time_sec": round(t_sec, 2),
                        "text": cleaned,
                        "is_near": abs(t_sec - pts_time) <= window_sec
                    })
        ocr_lines.sort(key=lambda x: x["time_sec"])

        near_ocr = [o for o in ocr_lines if o.get("is_near")]
        filtered_ocr = near_ocr if near_ocr else ocr_lines[:30]

        return {
            "video_id": video_id,
            "pts_time": pts_time,
            "asr_segments": segments,
            "ocr_lines": filtered_ocr,
            "total_asr_count": len(segments),
            "total_ocr_count": len(ocr_lines)
        }

    def extract_clip_frames(self, video_id: str, start_sec: float, end_sec: float, num_frames: int = 6) -> List[str]:
        """
        Uniformly samples `num_frames` from `video_id` between `start_sec` and `end_sec`,
        resizes to 512px width for efficient VLM transmission, and returns base64 data URLs.
        """
        vid_path = self.video_paths.get(video_id)
        if not vid_path or not os.path.exists(vid_path):
            return []

        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            return []

        b64_frames = []
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            st = max(0.0, float(start_sec))
            et = max(st + 0.1, float(end_sec))
            
            timestamps = np.linspace(st, et, num_frames)
            for t in timestamps:
                target_frame = min(int(round(t * fps)), max(0, total_frames - 1))
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                ret, frame = cap.read()
                if not ret:
                    continue
                
                h, w = frame.shape[:2]
                target_w = 512
                target_h = int(h * (target_w / w))
                resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
                
                success, enc_jpg = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if success:
                    b64_str = base64.b64encode(enc_jpg.tobytes()).decode("utf-8")
                    b64_frames.append(f"data:image/jpeg;base64,{b64_str}")
        finally:
            cap.release()

        return b64_frames

    def chat_completion(
        self,
        messages: List[Dict],
        video_id: str,
        frame_idx: int = 0,
        pts_time: float = 0.0,
        start_sec: Optional[float] = None,
        end_sec: Optional[float] = None,
        query: str = "",
        api_key: str = "",
        provider: str = "openrouter",
        model: str = "",
        custom_url: str = ""
    ) -> Dict:
        """
        Context-grounded multimodal VLM chat completion using short clip frames, ASR transcripts, and OCR context.
        """
        # Validate Start and End Time
        if start_sec is None or end_sec is None:
            return {
                "error": "Vui lòng đánh dấu Thời gian Bắt đầu (Start) và Kết thúc (End) của đoạn clip trước khi hỏi VLM."
            }

        try:
            start_sec = float(start_sec)
            end_sec = float(end_sec)
        except (ValueError, TypeError):
            return {
                "error": "Thời gian Start hoặc End không đúng định dạng số giây."
            }

        if not (math.isfinite(start_sec) and math.isfinite(end_sec)):
            return {
                "error": "Thời gian Start hoặc End không hợp lệ (NaN hoặc Infinity)."
            }

        if start_sec < 0 or end_sec <= start_sec:
            return {
                "error": f"Khoảng thời gian không hợp lệ: Start ({start_sec:.2f}s) phải nhỏ hơn End ({end_sec:.2f}s)."
            }

        clip_duration = end_sec - start_sec

        # Determine API Key
        final_key = api_key.strip() if api_key else ""
        if not final_key:
            if provider == "openai":
                final_key = os.environ.get("OPENAI_API_KEY", "")
            elif provider == "openrouter":
                final_key = os.environ.get("OPENROUTER_API_KEY", "")
            else: # omniroute / default localhost proxy
                final_key = os.environ.get("OMNIROUTE_API_KEY", os.environ.get("ANTHROPIC_AUTH_TOKEN", "omniroute"))

        if not final_key and provider not in ("custom", "omniroute"):
            return {
                "error": "API Key chưa được cấu hình. Vui lòng mở Cài đặt (⚙️) trên Chatbot để nhập API Key, hoặc cấu hình biến môi trường OPENROUTER_API_KEY / OPENAI_API_KEY trên server."
            }

        # Determine Model & Endpoint
        if provider == "openai":
            api_url = "https://api.openai.com/v1/chat/completions"
            final_model = model.strip() if model.strip() else "gpt-4o-mini"
        elif provider == "openrouter":
            api_url = "https://openrouter.ai/api/v1/chat/completions"
            final_model = model.strip() if model.strip() else os.environ.get("OPENROUTER_MODEL", "minimax/minimax-m3")
        elif provider == "custom" and custom_url.strip():
            api_url = custom_url.strip()
            final_model = model.strip() if model.strip() else "antigravity/gemini-3.6-flash-medium"
        else: # omniroute (default localhost proxy)
            api_url = custom_url.strip() if (provider == "custom" and custom_url.strip()) else "http://localhost:20128/v1/chat/completions"
            final_model = model.strip() if model.strip() else "antigravity/gemini-3.6-flash-medium"

        # 1. Extract Visual Frames for the Short Clip
        clip_frames = self.extract_clip_frames(video_id, start_sec, end_sec, num_frames=6)

        # 2. Gather Speech ASR & On-Screen OCR Context strictly within or around the clip [start_sec, end_sec]
        context_data = self.get_video_context(video_id, pts_time=(start_sec + end_sec) / 2.0, window_sec=max(30.0, clip_duration))
        
        # Filter ASR segments overlapping with [start_sec - 5s, end_sec + 5s]
        clip_asr = [
            s for s in context_data["asr_segments"]
            if (s["start_sec"] <= end_sec + 3.0 and s["end_sec"] >= start_sec - 3.0)
        ]
        used_asr = clip_asr if clip_asr else context_data["asr_segments"][:20]
        asr_str_list = [f"[{s['start_sec']}s - {s['end_sec']}s]: {s['text']}" for s in used_asr]
        asr_context = "\n".join(asr_str_list) if asr_str_list else "(Không có lời thoại ASR trong đoạn clip này)"

        # Filter OCR detections overlapping with [start_sec - 3s, end_sec + 3s]
        clip_ocr = [
            o for o in context_data["ocr_lines"]
            if (start_sec - 3.0) <= o["time_sec"] <= (end_sec + 3.0)
        ]
        used_ocr = clip_ocr if clip_ocr else context_data["ocr_lines"][:20]
        ocr_str_list = [f"[{o['time_sec']}s]: {o['text']}" for o in used_ocr]
        ocr_context = "\n".join(ocr_str_list) if ocr_str_list else "(Không có chữ OCR trên màn hình trong đoạn clip này)"

        # 3. Retrieve Top BGE Reranked Global Key Transcript Segments -> Expand with Neighbors & Deduplicate
        rerank_context_block = ""
        if query and self.semantic_index is not None and self.reranker is not None:
            try:
                raw_sem = self.semantic_index.query(query, top_k=30)
                reranked_top = self.reranker.rerank(query, raw_sem, top_k=10)
                if self.context_expander is not None:
                    expanded_windows = self.context_expander.expand_and_deduplicate(
                        reranked_top,
                        neighbor_window=1,
                        max_windows=3
                    )
                    formatted_ctx = self.context_expander.format_context_for_prompt(expanded_windows)
                    if formatted_ctx:
                        rerank_context_block = formatted_ctx + "\n\n"
                elif reranked_top:
                    rerank_lines = []
                    for r_item in reranked_top[:5]:
                        v = r_item.get("video_id", "")
                        st_s = r_item.get("start_sec", 0.0)
                        et_s = r_item.get("end_sec", 0.0)
                        t = r_item.get("text", r_item.get("refined_text", ""))
                        score = r_item.get("rerank_score", 0.0)
                        rerank_lines.append(f"- [{v} | {st_s:.1f}s - {et_s:.1f}s] (Độ khớp BGE: {score:.3f}): \"{t}\"")
                    rerank_context_block = f"[CÁC ĐOẠN LỜI THOẠI TRỌNG TÂM KHỚP NHẤT TỪ BGE-RERANKER]:\n" + "\n".join(rerank_lines) + "\n\n"
            except Exception:
                pass

        system_prompt = (
            "Bạn là Trợ lý VLM (Vision-Language Model) chuyên gia phân tích video cho Hệ thống Truy vấn AIC 2026.\n"
            "Người dùng đã cắt và đánh dấu một ĐOẠN CLIP VIDEO NGẮN cụ thể để đặt câu hỏi.\n\n"
            f"[THÔNG TIN ĐOẠN CLIP ĐƯỢC CHỌN]\n"
            f"- Video ID: {video_id}\n"
            f"- Mốc thời gian clip: từ {start_sec:.2f}s đến {end_sec:.2f}s (Thời lượng: {clip_duration:.2f} giây)\n"
            f"- Khung hình đại diện: Frame {frame_idx}\n"
            f"- Câu truy vấn tìm kiếm gốc: \"{query}\"\n"
            f"- Số khung hình hình ảnh VLM nhận được từ clip: {len(clip_frames)} frames\n\n"
            f"[LỜI THOẠI BĂNG GHI ÂM (ASR) TRONG ĐOẠN CLIP]:\n{asr_context}\n\n"
            f"[CHỮ HIỂN THỊ TRÊN MÀN HÌNH / BANNER (OCR) TRONG ĐOẠN CLIP]:\n{ocr_context}\n\n"
            f"{rerank_context_block}"
            "HƯỚNG DẪN TRẢ LỜI CHO VLM:\n"
            "1. Kết hợp chặt chẽ giữa HÌNH ẢNH TRỰC QUAN (các frames được cung cấp trong clip) và LỜI THOẠI ASR + CHỮ OCR để trả lời chính xác, súc tích.\n"
            "2. Tập trung phân tích hành động, diễn biến, nhân vật, đồ vật, chữ hiển thị, hoặc sự kiện diễn ra cụ thể trong đoạn clip [start, end].\n"
            "3. Trả lời bằng tiếng Việt trực tiếp, rõ ràng (hoặc tiếng Anh nếu người dùng hỏi bằng tiếng Anh)."
        )

        formatted_messages = [{"role": "system", "content": system_prompt}]
        
        # Build conversation messages: for the latest user message, attach the visual clip frames as multimodal content
        for idx, m in enumerate(messages):
            r = m.get("role", "user")
            c = m.get("content", "")
            if not c:
                continue

            if r == "user" and idx == len(messages) - 1 and clip_frames:
                # Multimodal message with images + text
                content_parts = []
                for b64_url in clip_frames:
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": b64_url}
                    })
                content_parts.append({
                    "type": "text",
                    "text": c
                })
                formatted_messages.append({"role": "user", "content": content_parts})
            elif r in ("user", "assistant"):
                formatted_messages.append({"role": r, "content": c})

        headers = {
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/TotallyNotMinh/aic2026",
            "X-Title": "AIC 2026 Video Assistant"
        }
        if final_key:
            headers["Authorization"] = f"Bearer {final_key}"

        body = {
            "model": final_model,
            "messages": formatted_messages,
            "temperature": 0.3,
            "stream": False
        }

        try:
            req = urllib.request.Request(api_url, data=json.dumps(body).encode("utf-8"), headers=headers)
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw_text = resp.read().decode("utf-8")
                reply_text = ""

                # 1. Try parsing standard OpenAI JSON response
                if not raw_text.strip().startswith("data:"):
                    try:
                        resp_data = json.loads(raw_text)
                        choices = resp_data.get("choices", [])
                        if choices:
                            reply_text = choices[0].get("message", {}).get("content", "")
                    except Exception:
                        pass

                # 2. Fallback to parsing Server-Sent Events (SSE) streaming format if stream was returned
                if not reply_text and "data:" in raw_text:
                    stream_chunks = []
                    for line in raw_text.splitlines():
                        line = line.strip()
                        if line.startswith("data:") and not line.endswith("[DONE]"):
                            chunk_json = line[5:].strip()
                            try:
                                chunk_obj = json.loads(chunk_json)
                                delta = chunk_obj.get("choices", [{}])[0].get("delta", {})
                                content_piece = delta.get("content", "")
                                if content_piece:
                                    stream_chunks.append(content_piece)
                            except Exception:
                                pass
                    reply_text = "".join(stream_chunks).strip()

                if reply_text:
                    return {
                        "reply": reply_text,
                        "video_id": video_id,
                        "frame_idx": frame_idx,
                        "pts_time": pts_time,
                        "start_sec": start_sec,
                        "end_sec": end_sec,
                        "clip_duration": round(clip_duration, 2),
                        "frames_sent": len(clip_frames),
                        "asr_count": len(used_asr),
                        "ocr_count": len(used_ocr)
                    }
                else:
                    return {"error": "API did not return any choices or content."}
        except urllib.error.HTTPError as e:
            try:
                err_detail = e.read().decode("utf-8")
                err_json = json.loads(err_detail)
                err_msg = err_json.get("error", {}).get("message", err_detail)
            except Exception:
                err_msg = str(e)
            return {"error": f"VLM API Error ({e.code}): {err_msg}"}
        except Exception as e:
            return {"error": f"Failed to connect to VLM API: {str(e)}"}

    def extract_thumbnail(self, video_id: str, frame_idx: int) -> Optional[bytes]:
        video_id = sanitize_video_id(video_id)
        if not video_id:
            return None

        cache_key = f"{video_id}_{frame_idx}"
        if cache_key in self.thumb_mem_cache:
            return self.thumb_mem_cache[cache_key]

        thumb_path = os.path.join(self.thumb_dir, f"{cache_key}.jpg")
        if os.path.exists(thumb_path):
            try:
                with open(thumb_path, "rb") as f:
                    data = f.read()
                    if len(self.thumb_mem_cache) < self.max_thumb_mem_cache:
                        self.thumb_mem_cache[cache_key] = data
                    return data
            except Exception:
                pass

        vid_path = self.video_paths.get(video_id)
        if not vid_path or not os.path.exists(vid_path):
            return None

        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            cap.release()
            return None

        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                return None
            
            h, w = frame.shape[:2]
            target_w = 480
            target_h = int(h * (target_w / w))
            resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
            
            success, enc_jpg = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if success:
                jpg_bytes = enc_jpg.tobytes()
                # Store in memory cache
                if len(self.thumb_mem_cache) < self.max_thumb_mem_cache:
                    self.thumb_mem_cache[cache_key] = jpg_bytes
                
                # Atomic file write to avoid multi-thread collision
                tmp_path = f"{thumb_path}.{os.getpid()}.{time.time_ns()}.tmp"
                try:
                    with open(tmp_path, "wb") as f:
                        f.write(jpg_bytes)
                    os.replace(tmp_path, thumb_path)
                except Exception:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                return jpg_bytes
        finally:
            cap.release()
        return None

    def _compute_keyword_scores(
        self,
        keywords: Optional[List[Dict]],
        w_dense: float,
        w_asr: float,
        matched_asr_dict: Optional[Dict[int, str]] = None
    ) -> np.ndarray:
        total_kw_score = np.zeros(len(self.records), dtype=np.float32)
        if not keywords:
            return total_kw_score

        for kw in keywords:
            kw_text = kw.get("text", "").strip()
            kw_weight = float(kw.get("weight", 1.0))
            is_exact = kw.get("exact", False) or kw.get("mode") == "exact"
            if not kw_text or kw_weight <= 0:
                continue

            if is_exact:
                # --- EXACT KEYWORD MATCH (No CLIP encoding, pure substring & phrase matching on transcripts) ---
                phrase_lower = kw_text.lower()
                kw_exact_scores = np.zeros(len(self.records), dtype=np.float32)

                for doc in self.bm25.docs:
                    txt_lower = doc["text"].lower()
                    if phrase_lower in txt_lower:
                        vid = doc["video_id"]
                        st = doc["start_sec"]
                        et = doc["end_sec"]
                        txt = doc["text"]
                        
                        for k_idx in self.video_to_records.get(vid, []):
                            pts = self.records[k_idx]["pts_time"]
                            if (st - 3.0) <= pts <= (et + 3.0):
                                kw_exact_scores[k_idx] = max(kw_exact_scores[k_idx], 1.5)
                                if matched_asr_dict is not None:
                                    matched_asr_dict[k_idx] = f"🔤 [EXACT] \"{txt}\""

                total_kw_score += kw_weight * kw_exact_scores

            else:
                # --- SEMANTIC VISUAL + SPEECH MATCH (CLIP + BM25) ---
                kw_en = self.translator.translate(kw_text)
                kw_prompts = self.translator.generate_prompts(kw_en)
                kw_vec = self.encoder.encode_text(kw_prompts, ensemble=True)
                kw_dense = np.dot(self.matrix, kw_vec)
                kw_d_min, kw_d_max = float(np.min(kw_dense)), float(np.max(kw_dense))
                kw_d_denom = max(1e-6, kw_d_max - kw_d_min)
                norm_kw_dense = (kw_dense - kw_d_min) / kw_d_denom

                kw_asr_scores = np.zeros(len(self.records), dtype=np.float32)
                kw_bm25_hits = self.bm25.search(kw_text, top_k=200)
                if kw_bm25_hits:
                    max_kw_b = max(s for _, s in kw_bm25_hits)
                    for doc_idx, b_score in kw_bm25_hits:
                        doc = self.bm25.docs[doc_idx]
                        vid = doc["video_id"]
                        st = doc["start_sec"]
                        et = doc["end_sec"]
                        norm_b = b_score / max(1e-6, max_kw_b)
                        for k_idx in self.video_to_records.get(vid, []):
                            pts = self.records[k_idx]["pts_time"]
                            if (st - 3.0) <= pts <= (et + 3.0):
                                if norm_b > kw_asr_scores[k_idx]:
                                    kw_asr_scores[k_idx] = norm_b

                kw_fused = (w_dense * norm_kw_dense) + (w_asr * kw_asr_scores)
                total_kw_score += kw_weight * kw_fused

        return total_kw_score

    def _apply_temporal_nms(
        self,
        top_indices: np.ndarray,
        top_k: int,
        initial_candidates: Optional[List[int]] = None,
        window_sec: float = 2.0
    ) -> List[int]:
        """
        Deduplicates keyframe candidates belonging to the same video within +/- window_sec.
        """
        seen_timestamps = defaultdict(list)
        final_candidates: List[int] = []

        if initial_candidates:
            for idx in initial_candidates:
                if idx not in final_candidates:
                    rec = self.records[idx]
                    vid = rec["video_id"]
                    pts = rec["pts_time"]
                    seen_timestamps[vid].append(pts)
                    final_candidates.append(idx)

        for idx in top_indices:
            if idx in final_candidates:
                continue
            rec = self.records[idx]
            vid = rec["video_id"]
            pts = rec["pts_time"]
            
            is_dup = any(abs(pts - prev_pts) < window_sec for prev_pts in seen_timestamps[vid])
            if is_dup and len(final_candidates) < top_k:
                continue

            seen_timestamps[vid].append(pts)
            final_candidates.append(idx)
            if len(final_candidates) >= top_k:
                break

        return final_candidates

    def _decompose_query_subscenes(self, query: str) -> List[str]:
        """
        Extracts individual chronological scene descriptions from a complex multi-event prompt.
        """
        split_pattern = r"[.\n;]|(?:\b(?:sau đó|tiếp theo|tiếp đến|kế tiếp|bắt đầu bằng|rồi đến|kết thúc bằng|sau cùng|then|after that|followed by|next)\b)"
        raw_parts = [p.strip() for p in re.split(split_pattern, query, flags=re.IGNORECASE)]
        clean_parts = []
        stop_phrases = {"đoạn phim", "video", "cảnh quay", "clip", "hình ảnh", "bộ phim", "đoạn video", "đoạn clip"}
        for p in raw_parts:
            p_clean = re.sub(r"^[,:;\-\s]+|[,:;\-\s]+$", "", p).strip()
            if len(p_clean) >= 12 and p_clean.lower() not in stop_phrases:
                clean_parts.append(p_clean)
        return clean_parts

    def search(
        self,
        query: str,
        keywords: Optional[List[Dict]] = None,
        w_dense: float = 0.50,
        w_asr: float = 0.50,
        top_k: int = 100,
        asr_window_sec: float = 5.0
    ) -> Dict:
        t0 = time.time()
        query = query.strip()
        if not query:
            return {"translated_query": "", "search_time_ms": 0, "total_results": 0, "results": []}

        # Validate and clamp numeric parameters
        try:
            top_k = max(1, min(1000, int(top_k)))
        except (ValueError, TypeError):
            top_k = 100

        try:
            w_dense = float(w_dense)
            if not math.isfinite(w_dense):
                w_dense = 0.50
        except (ValueError, TypeError):
            w_dense = 0.50

        try:
            w_asr = float(w_asr)
            if not math.isfinite(w_asr):
                w_asr = 0.50
        except (ValueError, TypeError):
            w_asr = 0.50

        try:
            asr_window_sec = max(0.5, min(30.0, float(asr_window_sec)))
        except (ValueError, TypeError):
            asr_window_sec = 5.0

        # --- A. Dense Visual Search (CLIP Multi-Scene + Holistic) ---
        en_query = self.translator.translate(query)
        prompts = self.translator.generate_prompts(en_query)
        q_vec = self.encoder.encode_text(prompts, ensemble=True)
        dense_scores_whole = np.dot(self.matrix, q_vec)  # (177321,)

        sub_clauses = self._decompose_query_subscenes(query)
        best_sub_idx = None

        if len(sub_clauses) > 1:
            sub_vectors = []
            for sc in sub_clauses:
                en_sc = self.translator.translate(sc)
                prompts_sc = self.translator.generate_prompts(en_sc)
                sub_vectors.append(self.encoder.encode_text(prompts_sc, ensemble=True))

            sub_matrix = np.column_stack(sub_vectors)
            sub_scores_raw = np.dot(self.matrix, sub_matrix)

            # 1. Absolute-Weighted Min-Max Calibration
            sub_scores_norm = np.zeros_like(sub_scores_raw)
            for c in range(sub_scores_raw.shape[1]):
                col = sub_scores_raw[:, c]
                c_min, c_max = float(np.min(col)), float(np.max(col))
                col_norm = (col - c_min) / max(1e-6, c_max - c_min)
                # Scale by absolute peak confidence to prevent irrelevant sub-scenes from becoming 1.0
                col_conf = float(np.clip((c_max - 0.20) / 0.13, 0.15, 1.0))
                sub_scores_norm[:, c] = col_norm * col_conf

            w_min, w_max = float(np.min(dense_scores_whole)), float(np.max(dense_scores_whole))
            norm_whole = (dense_scores_whole - w_min) / max(1e-6, w_max - w_min)
            whole_conf = float(np.clip((w_max - 0.20) / 0.13, 0.15, 1.0))
            norm_whole = norm_whole * whole_conf

            # 2. Sub-scene Pooling: 0.70 * max + 0.30 * top-2 mean
            if sub_scores_norm.shape[1] >= 2:
                top2_vals = np.partition(sub_scores_norm, -2, axis=1)[:, -2:]
                top2_mean = np.mean(top2_vals, axis=1)
                max_sub_pooled = (0.70 * np.max(sub_scores_norm, axis=1)) + (0.30 * top2_mean)
            else:
                max_sub_pooled = np.max(sub_scores_norm, axis=1)

            best_sub_idx = np.argmax(sub_scores_norm, axis=1)
            norm_dense_scores = (0.35 * norm_whole) + (0.65 * max_sub_pooled)
            dense_scores = (0.35 * dense_scores_whole) + (0.65 * np.max(sub_scores_raw, axis=1))
        else:
            d_min, d_max = float(np.min(dense_scores_whole)), float(np.max(dense_scores_whole))
            d_denom = max(1e-6, d_max - d_min)
            col_norm = (dense_scores_whole - d_min) / d_denom
            # Absolute confidence weighting on visual CLIP
            d_conf = float(np.clip((d_max - 0.20) / 0.13, 0.15, 1.0))
            norm_dense_scores = col_norm * d_conf
            dense_scores = dense_scores_whole

        # --- B. Speech Transcript Search (BM25 Lexical + E5 Semantic) ---
        bm25_hits = self.bm25.search(query, top_k=300)
        sem_hits = []
        if self.semantic_index is not None:
            try:
                sem_hits = self.semantic_index.query(query, top_k=300)
            except Exception:
                sem_hits = []
        
        # Aggregate BM25 & Semantic scores to keyframes by video and timestamp
        keyframe_asr_scores = np.zeros(len(self.records), dtype=np.float32)
        keyframe_asr_texts = {}

        if bm25_hits:
            for doc_idx, b_score in bm25_hits:
                doc = self.bm25.docs[doc_idx]
                vid = doc["video_id"]
                st = doc["start_sec"]
                et = doc["end_sec"]
                txt = doc["text"]
                # Sigmoidal absolute BM25 calibration: BM25 / (25.0 + BM25)
                norm_b = float(b_score) / (25.0 + float(b_score))

                # Assign boost to all keyframes belonging to this video within [st - asr_window_sec, et + asr_window_sec]
                kf_indices = self.video_to_records.get(vid, [])
                for k_idx in kf_indices:
                    pts = self.records[k_idx]["pts_time"]
                    if (st - asr_window_sec) <= pts <= (et + asr_window_sec):
                        if norm_b > keyframe_asr_scores[k_idx]:
                            keyframe_asr_scores[k_idx] = norm_b
                            keyframe_asr_texts[k_idx] = txt

        if sem_hits:
            if self.reranker is not None:
                # 1. Rerank Top 30 dense semantic candidates with BGE-Reranker-v2-m3
                top_dense_candidates = sem_hits[:30]
                reranked_candidate_dicts = self.reranker.rerank(
                    query=query,
                    candidates=top_dense_candidates,
                    top_k=min(30, len(top_dense_candidates))
                )
                
                # 2. Expand neighboring segment context for temporal completeness
                expanded_candidates = self.context_expander.expand_and_deduplicate(
                    ranked_candidates=reranked_candidate_dicts,
                    neighbor_window=1,
                    max_windows=15
                )
                
                if expanded_candidates:
                    for cand in expanded_candidates:
                        vid = cand["video_id"]
                        st = cand["start_sec"]
                        et = cand["end_sec"]
                        txt = cand.get("text", "")
                        raw_s = cand.get("rerank_score", cand.get("score", 0.0))
                        # Absolute E5 / BGE rerank score calibration
                        if "rerank_score" in cand:
                            # BGE logits passed through sigmoid
                            norm_s = 1.0 / (1.0 + math.exp(-float(raw_s)))
                        else:
                            # E5 cosine similarity mapped from [0.70, 0.90] -> [0.0, 1.0]
                            norm_s = float(np.clip((raw_s - 0.70) / 0.20, 0.0, 1.0))
                        
                        kf_indices = self.video_to_records.get(vid, [])
                        for k_idx in kf_indices:
                            pts = self.records[k_idx]["pts_time"]
                            if (st - asr_window_sec) <= pts <= (et + asr_window_sec):
                                if norm_s > keyframe_asr_scores[k_idx]:
                                    keyframe_asr_scores[k_idx] = norm_s
                                    if not keyframe_asr_texts.get(k_idx):
                                        keyframe_asr_texts[k_idx] = txt
            else:
                for cand_dict, s_score in sem_hits:
                    vid = cand_dict["video_id"]
                    st = cand_dict["start_sec"]
                    et = cand_dict["end_sec"]
                    txt = cand_dict.get("text", "")
                    norm_s = float(np.clip((s_score - 0.70) / 0.20, 0.0, 1.0))

                    kf_indices = self.video_to_records.get(vid, [])
                    for k_idx in kf_indices:
                        pts = self.records[k_idx]["pts_time"]
                        if (st - asr_window_sec) <= pts <= (et + asr_window_sec):
                            if norm_s > keyframe_asr_scores[k_idx]:
                                keyframe_asr_scores[k_idx] = norm_s
                                if not keyframe_asr_texts.get(k_idx):
                                    keyframe_asr_texts[k_idx] = txt

        # --- C. Multi-Modal Fusion ---
        fused_scores = (w_dense * norm_dense_scores) + (w_asr * keyframe_asr_scores)
        if keywords:
            fused_scores += self._compute_keyword_scores(keywords, w_dense, w_asr, keyframe_asr_texts)

        # Get top-K candidate indices with safe bounds
        k_pool = min(max(top_k * 4, 1), len(self.records))
        top_indices = np.argpartition(fused_scores, -k_pool)[-k_pool:]
        top_indices = top_indices[np.argsort(fused_scores[top_indices])[::-1]]

        # Diversity / Non-Maximum Suppression (deduplicate within ±2 seconds in same video)
        final_candidates = self._apply_temporal_nms(top_indices, top_k=top_k, window_sec=2.0)

        # Build output response list
        results = []
        for rank, idx in enumerate(final_candidates, start=1):
            rec = self.records[idx]
            vid = rec["video_id"]
            pts = rec["pts_time"]
            f_idx = rec["frame_idx"]
            fps = rec.get("fps", 25.0)

            asr_text = keyframe_asr_texts.get(idx, "")
            if not asr_text:
                asr_text = self.get_asr_at_pts(vid, pts)

            matched_scene = ""
            if best_sub_idx is not None and len(sub_clauses) > 1:
                matched_scene = sub_clauses[best_sub_idx[idx]]

            results.append({
                "rank": rank,
                "video_id": vid,
                "frame_idx": f_idx,
                "fps": fps,
                "pts_time": round(pts, 4),
                "raw_dense_score": round(float(dense_scores_whole[idx]), 4),
                "norm_dense_score": round(float(norm_dense_scores[idx]), 4),
                "raw_asr_score": round(float(keyframe_asr_scores[idx]), 4),
                "retrieval_score": round(float(fused_scores[idx]), 4),
                "score": round(float(fused_scores[idx]), 4),
                "matched_asr": asr_text,
                "matched_scene": matched_scene,
                "thumb_url": f"/api/frame?video_id={vid}&frame_idx={f_idx}",
                "video_url": f"/api/video?video_id={vid}"
            })

        search_time_ms = round((time.time() - t0) * 1000, 1)
        return {
            "query_vi": query,
            "translated_query": en_query,
            "sub_scenes": sub_clauses if len(sub_clauses) > 1 else [],
            "asr_window_sec": asr_window_sec,
            "search_time_ms": search_time_ms,
            "total_results": len(results),
            "results": results
        }

    def refine(
        self,
        query: str,
        marked_items: List[Dict],
        keywords: Optional[List[Dict]] = None,
        w_dense: float = 0.50,
        w_asr: float = 0.50,
        top_k: int = 100,
        alpha: float = 0.65,
        beta: float = 0.35,
        asr_window_sec: float = 5.0
    ) -> Dict:
        t0 = time.time()
        query = query.strip()
        if not query and not marked_items:
            return {"query_vi": "", "translated_query": "", "search_time_ms": 0, "total_results": 0, "results": []}

        # Validate and clamp numeric parameters
        try:
            top_k = max(1, min(1000, int(top_k)))
        except (ValueError, TypeError):
            top_k = 100

        try:
            w_dense = float(w_dense)
            if not math.isfinite(w_dense):
                w_dense = 0.50
        except (ValueError, TypeError):
            w_dense = 0.50

        try:
            w_asr = float(w_asr)
            if not math.isfinite(w_asr):
                w_asr = 0.50
        except (ValueError, TypeError):
            w_asr = 0.50

        try:
            asr_window_sec = max(0.5, min(30.0, float(asr_window_sec)))
        except (ValueError, TypeError):
            asr_window_sec = 5.0

        # 1. Base Text Embedding
        en_query = self.translator.translate(query) if query else ""
        if en_query:
            prompts = self.translator.generate_prompts(en_query)
            q_orig = self.encoder.encode_text(prompts, ensemble=True)
        else:
            q_orig = None

        # 2. Positive Feedback Indices
        pos_indices = []
        for m in marked_items:
            vid = sanitize_video_id(m.get("video_id"))
            try:
                fid = int(m.get("frame_idx", 0))
            except (ValueError, TypeError):
                fid = 0
            try:
                target_pts = float(m.get("pts_time", 0.0))
            except (ValueError, TypeError):
                target_pts = 0.0

            key = (vid, fid)
            if key in self.frame_lookup:
                pos_indices.append(self.frame_lookup[key])
            elif vid in self.video_to_records and self.video_to_records[vid]:
                k_indices = self.video_to_records[vid]
                best_k = min(k_indices, key=lambda i: abs(self.records[i]["pts_time"] - target_pts))
                pos_indices.append(best_k)

        # 3. Vector Blending
        if pos_indices:
            pos_vecs = np.array([self.matrix[idx] for idx in pos_indices], dtype=np.float32)
            q_pos = np.mean(pos_vecs, axis=0)
            q_pos_norm = np.linalg.norm(q_pos)
            if q_pos_norm > 1e-12:
                q_pos = q_pos / q_pos_norm

            if q_orig is not None:
                q_blend = alpha * q_orig + beta * q_pos
                q_blend_norm = np.linalg.norm(q_blend)
                if q_blend_norm > 1e-12:
                    q_blend = q_blend / q_blend_norm
            else:
                q_blend = q_pos
        else:
            q_blend = q_orig if q_orig is not None else np.zeros(self.matrix.shape[1], dtype=np.float32)

        dense_scores = np.dot(self.matrix, q_blend)
        d_min, d_max = float(np.min(dense_scores)), float(np.max(dense_scores))
        d_denom = max(1e-6, d_max - d_min)
        norm_dense_scores = (dense_scores - d_min) / d_denom

        # 4. Temporal Proximity Bonus for Confirmed Scenes
        scene_mod = np.zeros(len(self.records), dtype=np.float32)
        for m in marked_items:
            vid = sanitize_video_id(m.get("video_id"))
            try:
                target_pts = float(m.get("pts_time", 0.0))
            except (ValueError, TypeError):
                target_pts = 0.0

            if vid in self.video_to_records:
                for k_idx in self.video_to_records[vid]:
                    rec = self.records[k_idx]
                    dist = abs(rec["pts_time"] - target_pts)
                    if dist <= 45.0:
                        bonus = 0.25 * (1.0 - (dist / 45.0))
                        scene_mod[k_idx] = max(scene_mod[k_idx], bonus)
                    else:
                        scene_mod[k_idx] = max(scene_mod[k_idx], 0.08)

        norm_dense_scores = np.clip(norm_dense_scores + scene_mod, 0.0, 1.0)

        # 5. BM25 Speech Retrieval
        keyframe_asr_scores = np.zeros(len(self.records), dtype=np.float32)
        keyframe_asr_texts = {}
        if query:
            bm25_hits = self.bm25.search(query, top_k=300)
            if bm25_hits:
                max_bm25 = max(score for _, score in bm25_hits)
                for doc_idx, b_score in bm25_hits:
                    doc = self.bm25.docs[doc_idx]
                    vid = doc["video_id"]
                    st = doc["start_sec"]
                    et = doc["end_sec"]
                    txt = doc["text"]
                    norm_b = b_score / max(1e-6, max_bm25)
                    kf_indices = self.video_to_records.get(vid, [])
                    for k_idx in kf_indices:
                        pts = self.records[k_idx]["pts_time"]
                        if (st - asr_window_sec) <= pts <= (et + asr_window_sec):
                            if norm_b > keyframe_asr_scores[k_idx]:
                                keyframe_asr_scores[k_idx] = norm_b
                                keyframe_asr_texts[k_idx] = txt

        # 6. Fusion & NMS
        fused_scores = (w_dense * norm_dense_scores) + (w_asr * keyframe_asr_scores)
        if keywords:
            fused_scores += self._compute_keyword_scores(keywords, w_dense, w_asr, keyframe_asr_texts)

        k_pool = min(max(top_k * 4, 1), len(self.records))
        top_indices = np.argpartition(fused_scores, -k_pool)[-k_pool:]
        top_indices = top_indices[np.argsort(fused_scores[top_indices])[::-1]]

        final_candidates = self._apply_temporal_nms(
            top_indices,
            top_k=top_k,
            initial_candidates=pos_indices,
            window_sec=2.0
        )

        sub_clauses = self._decompose_query_subscenes(query) if query else []
        pos_set = set(pos_indices)
        results = []
        for rank, idx in enumerate(final_candidates, start=1):
            rec = self.records[idx]
            vid = rec["video_id"]
            pts = rec["pts_time"]
            f_idx = rec["frame_idx"]
            fps = rec.get("fps", 25.0)
            score_val = 1.0 if idx in pos_set else round(float(fused_scores[idx]), 4)
            asr_text = keyframe_asr_texts.get(idx, "")
            if not asr_text:
                asr_text = self.get_asr_at_pts(vid, pts)

            matched_scene = ""

            results.append({
                "rank": rank,
                "video_id": vid,
                "frame_idx": f_idx,
                "fps": fps,
                "pts_time": round(pts, 4),
                "raw_dense_score": round(float(dense_scores[idx]), 4),
                "norm_dense_score": round(float(norm_dense_scores[idx]), 4),
                "raw_asr_score": round(float(keyframe_asr_scores[idx]), 4),
                "retrieval_score": round(float(fused_scores[idx]), 4),
                "score": score_val,
                "matched_asr": asr_text,
                "matched_scene": matched_scene,
                "thumb_url": f"/api/frame?video_id={vid}&frame_idx={f_idx}",
                "video_url": f"/api/video?video_id={vid}"
            })

        # Explicitly ensure pinned items are placed at the absolute top (Rank 1)
        pinned_items = [m for m in marked_items if m.get("isPinned")]
        if pinned_items:
            pinned_results = []
            for p in pinned_items:
                p_vid = p.get("video_id")
                p_fid = int(p.get("frame_idx", 0))
                p_pts = float(p.get("pts_time", 0.0))
                p_fps = float(p.get("fps", 25.0))
                # Remove any existing match to avoid duplicate
                results = [r for r in results if not (r["video_id"] == p_vid and r["frame_idx"] == p_fid)]
                p_asr = self.get_asr_at_pts(p_vid, p_pts)
                pinned_results.append({
                    "rank": 1,
                    "video_id": p_vid,
                    "frame_idx": p_fid,
                    "fps": p_fps,
                    "pts_time": round(p_pts, 2),
                    "dense_score": 1.0,
                    "score": 1.0,
                    "matched_asr": f"🎯 Pinned Match · {p_asr}" if p_asr else "🎯 Pinned Match",
                    "thumb_url": f"/api/frame?video_id={p_vid}&frame_idx={p_fid}",
                    "video_url": f"/api/video?video_id={p_vid}",
                    "is_pinned": True
                })
            results = pinned_results + results
            results = results[:top_k]
            for r_idx, r in enumerate(results, start=1):
                r["rank"] = r_idx

        search_time_ms = round((time.time() - t0) * 1000, 1)
        return {
            "query_vi": query,
            "translated_query": en_query,
            "search_time_ms": search_time_ms,
            "total_results": len(results),
            "results": results
        }


ENGINE = None

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AIC 2026 Multi-Modal Retrieval Studio</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script>
    tailwind.config = {
      darkMode: 'class',
      theme: {
        extend: {
          colors: {
            brand: { 50: '#ecfdf5', 500: '#10b981', 600: '#059669', 700: '#047857' }
          }
        }
      }
    };
  </script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js"></script>
  <script>if (typeof JSZip === 'undefined') { document.write('<script src="https://cdn.jsdelivr.net/npm/jszip@3.10.1/dist/jszip.min.js"><\/script>'); }</script>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    body { font-family: 'Inter', sans-serif; }
    .glass { background: rgba(15, 23, 42, 0.85); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.08); }
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0f172a; }
    ::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
  </style>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen flex flex-col antialiased selection:bg-brand-500 selection:text-white">

  <!-- Header -->
  <header class="sticky top-0 z-40 glass border-b border-slate-800/80 px-6 py-3.5 flex items-center justify-between shadow-2xl">
    <div class="flex items-center gap-3">
      <div class="w-9 h-9 rounded-xl bg-gradient-to-br from-emerald-400 to-teal-600 flex items-center justify-center font-bold text-slate-950 shadow-lg shadow-emerald-500/20">
        AI
      </div>
      <div>
        <h1 class="font-bold text-base tracking-tight text-white">
          AIC 2026 Multi-Modal Retrieval Studio
        </h1>
      </div>
    </div>

    <div class="flex items-center gap-2.5">
      <!-- Hidden File Input for Sidebar & Modal Uploads -->
      <input type="file" id="queryFileInput" accept=".zip,.txt,.json" multiple class="hidden" onchange="handleQueryFileUpload(event)" />

      <!-- Batch Export ZIP Button -->
      <button id="exportAllZipBtn" onclick="exportAllQueriesZip()" class="text-xs font-semibold px-3 py-1.5 rounded-lg bg-gradient-to-r from-amber-500/20 to-orange-500/20 hover:from-amber-500/30 hover:to-orange-500/30 text-amber-300 border border-amber-500/40 transition flex items-center gap-1.5 shadow-sm" title="Đóng gói toàn bộ đáp án của tất cả các câu truy vấn thành submission.zip">
        <span>📦</span>
        <span id="exportAllZipLabel">Export All (.zip)</span>
        <span id="batchProgressBadge" class="hidden px-1.5 py-0.2 rounded-full bg-amber-400 text-slate-950 font-mono text-[10px] font-bold">0/0</span>
      </button>

      <!-- VLM Model & Assistant Launch Button -->
      <button onclick="toggleChatbot()" class="text-xs font-semibold px-3 py-1.5 rounded-lg bg-gradient-to-r from-emerald-500/20 to-teal-500/20 hover:from-emerald-500/30 hover:to-teal-500/30 text-emerald-300 border border-emerald-500/40 transition flex items-center gap-1.5 shadow-sm" title="Mở Trợ lý VLM phân tích video & clip">
        <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
        <span class="font-bold">🤖 VLM</span>
        <span id="headerModelBadge" class="font-mono text-[10px] px-1.5 py-0.5 rounded bg-slate-900 text-emerald-400 border border-slate-700">minimax/minimax-m3</span>
      </button>

      <div class="flex items-center gap-1.5 bg-slate-900/80 px-2 py-1 rounded-lg border border-slate-700/60">
        <span class="text-[11px] text-slate-400 font-semibold">Query ID:</span>
        <input id="queryIdInput" type="text" value="query-1" placeholder="query-1" class="w-20 bg-slate-950 text-emerald-300 font-mono text-xs rounded px-1.5 py-1 border border-slate-700 focus:outline-none" oninput="updateExportButtonLabel()" />
        <button id="exportCsvBtn" onclick="exportCSV()" class="text-xs font-semibold px-2.5 py-1 rounded bg-emerald-500 hover:bg-emerald-400 text-slate-950 transition flex items-center gap-1 shadow-sm font-bold" title="Xuất file CSV riêng cho câu query hiện tại">
          <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path></svg>
          <span id="exportBtnLabel">Export KIS</span>
        </button>
      </div>
    </div>
  </header>

  <!-- Two-Column App Layout with Persistent Left Query Sidebar -->
  <div class="flex-1 w-full max-w-[1720px] mx-auto p-4 md:p-6 flex flex-col md:flex-row gap-6 items-start">

    <!-- Persistent Left Queries Sidebar (Always Visible) -->
    <aside id="querySidebar" class="w-full md:w-80 lg:w-[370px] flex-shrink-0 bg-[#080b14]/95 backdrop-blur-xl rounded-2xl p-4 shadow-2xl border border-slate-800/80 flex flex-col gap-3.5 md:sticky md:top-20 max-h-[calc(100vh-6rem)] overflow-hidden z-50 relative">
      
      <!-- Sidebar Header -->
      <div class="flex items-center justify-between border-b border-slate-800/70 pb-3">
        <div class="flex items-center gap-3 min-w-0">
          <div class="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center text-white shadow-lg shadow-indigo-500/25 flex-shrink-0">
            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
            </svg>
          </div>
          <div class="min-w-0 flex-1">
            <h2 class="font-extrabold text-sm tracking-tight text-white uppercase font-sans">
              AIC 2026 STUDIO
            </h2>
          </div>
        </div>

        <div class="flex items-center gap-1">
          <button onclick="document.getElementById('queryFileInput').click()" class="p-1.5 bg-slate-900/80 hover:bg-slate-800 text-sky-400 border border-slate-800 rounded-lg text-xs transition" title="Upload Queries (.zip / .txt)">
            ➕
          </button>
          <button onclick="resetSessionStorage()" class="p-1.5 bg-slate-900/80 hover:bg-amber-950/50 text-amber-400 border border-slate-800 rounded-lg text-xs transition" title="Reset Session">
            🧹
          </button>
        </div>
      </div>

      <!-- Filter Controls: Search Bar + Mode Tabs -->
      <div class="flex flex-col gap-2.5">
        <!-- Search Input -->
        <div class="relative flex items-center">
          <div class="absolute left-3 text-slate-500 pointer-events-none">
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
          </div>
          <input 
            id="sidebarFilterInput" 
            type="text" 
            placeholder="Filter queries..." 
            class="w-full bg-[#0d1222] text-slate-200 placeholder-slate-500 text-xs rounded-xl pl-9 pr-8 py-2.5 border border-slate-800/80 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500/40 transition font-medium"
            oninput="filterSidebarQueries(this.value)"
          />
          <button id="sidebarFilterClearBtn" onclick="clearSidebarFilter()" class="hidden absolute right-2.5 text-slate-500 hover:text-slate-300 text-xs font-bold px-1">
            ✕
          </button>
        </div>

        <!-- Filter Tabs Row -->
        <div class="flex items-center justify-between gap-1 text-xs">
          <button id="filterTab_all" onclick="setSidebarModeFilter('all')" class="flex-1 py-1.5 px-1.5 rounded-lg text-center transition text-[11px] font-bold bg-slate-800 text-white border border-slate-700 shadow-sm">
            All (<span id="count_all">0</span>)
          </button>
          <button id="filterTab_kis" onclick="setSidebarModeFilter('kis')" class="flex-1 py-1.5 px-1.5 rounded-lg text-center transition text-[11px] font-bold text-sky-400 hover:bg-slate-900/60">
            KIS (<span id="count_kis">0</span>)
          </button>
          <button id="filterTab_qa" onclick="setSidebarModeFilter('qa')" class="flex-1 py-1.5 px-1.5 rounded-lg text-center transition text-[11px] font-bold text-purple-400 hover:bg-slate-900/60">
            QA (<span id="count_qa">0</span>)
          </button>
          <button id="filterTab_trake" onclick="setSidebarModeFilter('trake')" class="flex-1 py-1.5 px-1.5 rounded-lg text-center transition text-[11px] font-bold text-amber-400 hover:bg-slate-900/60">
            TRAKE (<span id="count_trake">0</span>)
          </button>
        </div>
      </div>

      <!-- Scrollable Query Cards List (Always Visible on Side) -->
      <div id="queryCardsContainer" class="flex-1 overflow-y-auto flex flex-col gap-2.5 pr-1 min-h-[140px]">
        <!-- Empty State -->
        <div id="querySidebarEmptyState" class="py-12 text-center text-slate-500 text-xs flex flex-col items-center gap-2">
          <span class="text-3xl">📂</span>
          <span>Chưa có gói câu hỏi nào.</span>
          <button onclick="document.getElementById('queryFileInput').click()" class="mt-1 px-3 py-1.5 bg-sky-500/20 hover:bg-sky-500/30 text-sky-300 border border-sky-500/40 rounded-lg text-xs font-semibold transition">
            Upload Queries (.zip)
          </button>
        </div>
      </div>

      <!-- Sidebar Bottom Action Buttons -->
      <div class="border-t border-slate-800/80 pt-3 flex items-center justify-between gap-2">
        <button onclick="saveActiveQueryResult()" class="flex-1 py-2 bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-bold text-xs rounded-xl transition flex items-center justify-center gap-1 shadow-md shadow-emerald-500/20">
          💾 Save Current
        </button>
        <button onclick="exportAllQueriesZip()" class="flex-1 py-2 bg-gradient-to-r from-amber-500 to-orange-500 hover:from-amber-400 hover:to-orange-400 text-slate-950 font-bold text-xs rounded-xl transition flex items-center justify-center gap-1 shadow-md shadow-amber-500/20">
          📦 Export ZIP
        </button>
      </div>
    </aside>

    <!-- Main Workspace Content -->
    <main class="flex-1 min-w-0 w-full flex flex-col gap-5">

      <!-- Task Mode Selector (plan.txt) -->
      <div class="glass rounded-2xl p-2.5 shadow-xl flex flex-wrap items-center justify-between gap-3 border border-slate-800">
        <div class="flex items-center gap-1.5 bg-slate-950/90 p-1 rounded-xl border border-slate-800">
          <button id="modeBtn_kis" onclick="setTaskMode('kis')" class="px-4 py-2 rounded-lg font-bold text-xs transition flex items-center gap-2 bg-emerald-500 text-slate-950 shadow-md shadow-emerald-500/20">
            <span>🔍</span>
            <span>1. Retrieval (KIS)</span>
          </button>
          <button id="modeBtn_qa" onclick="setTaskMode('qa')" class="px-4 py-2 rounded-lg font-bold text-xs transition flex items-center gap-2 text-slate-400 hover:text-slate-200">
            <span>💬</span>
            <span>2. Q&A Mode</span>
          </button>
          <button id="modeBtn_trake" onclick="setTaskMode('trake')" class="px-4 py-2 rounded-lg font-bold text-xs transition flex items-center gap-2 text-slate-400 hover:text-slate-200">
            <span>⏱️</span>
            <span>3. TRAKE Mode</span>
          </button>
        </div>

        <div id="modeDescriptionText" class="text-xs text-slate-400 flex items-center gap-1.5">
          <span class="text-emerald-400 font-semibold">Mode KIS:</span> Tìm kiếm & xuất file <code class="bg-slate-900 px-1.5 py-0.5 rounded text-emerald-300 font-mono text-[11px]">&lt;query_id&gt;-kis.csv</code> (video_id,frame_idx)
        </div>
      </div>

    <!-- Search Input Card -->
    <div class="glass rounded-2xl p-5 shadow-xl flex flex-col gap-4">

      <!-- Quick Query Navigator Bar -->
      <div id="queryActiveBar" class="flex flex-wrap items-center justify-between gap-2 bg-slate-950/80 p-2.5 rounded-xl border border-slate-800 text-xs">
        <div class="flex items-center gap-2">
          <span class="text-sky-400 font-bold flex items-center gap-1">
            🎯 Active Query:
          </span>
          <select id="quickQuerySelector" onchange="selectQuery(this.value, true)" class="bg-slate-900 border border-slate-700 text-slate-200 font-mono text-xs rounded-lg px-2.5 py-1 focus:outline-none focus:border-sky-400 font-semibold">
            <option value="query-1">query-1 (Default)</option>
          </select>
          <span id="quickQueryStatusBadge" class="text-[10px] px-2 py-0.5 rounded-full bg-slate-800 text-slate-400 font-semibold">
            ○ Unanswered
          </span>
        </div>

        <div class="flex items-center gap-1.5">
          <button onclick="cycleQuery(-1)" class="px-2.5 py-1 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg text-xs font-semibold transition" title="Chuyển đến câu query trước">
            ⏮️ Prev
          </button>
          <button onclick="cycleQuery(1)" class="px-2.5 py-1 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg text-xs font-semibold transition" title="Chuyển đến câu query tiếp theo">
            Next ⏭️
          </button>
        </div>
      </div>

      <div class="relative flex items-center">
        <div class="absolute left-4 text-slate-400">
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path></svg>
        </div>
        <input 
          id="queryInput" 
          type="text" 
          placeholder="Nhập mô tả sự kiện (VD: 'Mẩu tin giới thiệu về đàn hổ tại một địa phương ở miền Nam vừa có thêm 3-6 con hổ con', 'ức gà trộn cải ngồng')..." 
          class="w-full bg-slate-900/90 text-slate-100 placeholder-slate-500 text-sm rounded-xl pl-12 pr-28 py-3.5 border border-slate-700/60 focus:outline-none focus:ring-2 focus:ring-emerald-500/40 focus:border-emerald-500 transition font-medium"
          onkeydown="if(event.key==='Enter') executeSearch()"
        />
        <button 
          onclick="executeSearch()" 
          class="absolute right-2 px-5 py-2 rounded-lg bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-semibold text-xs transition shadow-md shadow-emerald-500/20 active:scale-95"
        >
          Search
        </button>
      </div>

      <!-- Keyword Emphasis & Exact Match Section -->
      <div class="flex flex-col gap-2 pt-2 border-t border-slate-800/60 text-xs">
        <div class="flex flex-wrap items-center justify-between gap-2">
          <div class="flex items-center gap-2">
            <span class="text-slate-400 font-semibold flex items-center gap-1">
              ⚡ Keywords:
            </span>
            <div class="flex items-center gap-1.5">
              <input 
                id="newKeywordInput" 
                type="text" 
                placeholder="Thêm từ khóa (VD: 'hổ con', 'Lộc Trời', '60 Giây')..." 
                class="bg-slate-900 text-slate-200 placeholder-slate-500 text-xs rounded-lg px-3 py-1.5 border border-slate-700/60 focus:outline-none focus:ring-1 focus:ring-emerald-500/50 w-64"
                onkeydown="if(event.key==='Enter') addKeyword(true)"
              />
              <button 
                onclick="addKeyword(true)" 
                class="px-2.5 py-1.5 rounded-lg bg-amber-500/20 hover:bg-amber-500/30 text-amber-300 font-semibold text-xs border border-amber-500/40 transition flex items-center gap-1"
                title="Exact keyword/phrase substring match on transcript without neural CLIP encoding"
              >
                🔤 + Exact Match
              </button>
              <button 
                onclick="addKeyword(false)" 
                class="px-2.5 py-1.5 rounded-lg bg-emerald-500/20 hover:bg-emerald-500/30 text-emerald-300 font-semibold text-xs border border-emerald-500/30 transition flex items-center gap-1"
                title="Semantic visual CLIP + speech BM25 match"
              >
                🧠 + Semantic
              </button>
            </div>
          </div>
          <span class="text-[11px] text-slate-500"><b>Exact Match</b> searches direct text/speech transcripts with 0% semantic drift.</span>
        </div>
        <div id="keywordChipsContainer" class="flex flex-wrap gap-2 pt-0.5"></div>
      </div>

      <!-- Live Controls & Query Translation Info -->
      <div class="flex flex-wrap items-center justify-between gap-4 pt-1 border-t border-slate-800/60 text-xs">
        <div class="flex items-center gap-2 text-slate-400">
          <span>Translated English Query:</span>
          <span id="transQuery" class="text-emerald-300 font-mono bg-emerald-950/40 px-2.5 py-1 rounded border border-emerald-800/40 italic">None</span>
        </div>

        <div class="flex items-center gap-6">
          <div class="flex items-center gap-2">
            <span class="text-slate-400">Visual (CLIP):</span>
            <input id="wDense" type="range" min="0" max="1" step="0.05" value="0.50" class="w-20 accent-emerald-500" oninput="updateWeights()">
            <span id="wDenseVal" class="text-slate-300 font-mono w-7">0.50</span>
          </div>
          <div class="flex items-center gap-2">
            <span class="text-slate-400">Speech (BM25):</span>
            <input id="wASR" type="range" min="0" max="1" step="0.05" value="0.50" class="w-20 accent-emerald-500" oninput="updateWeights()">
            <span id="wASRVal" class="text-slate-300 font-mono w-7">0.50</span>
          </div>
          <div class="flex items-center gap-2">
            <span class="text-slate-400">Top-K:</span>
            <select id="topKSelect" class="bg-slate-900 border border-slate-700/60 rounded px-2 py-1 text-slate-200 focus:outline-none">
              <option value="25">25</option>
              <option value="50">50</option>
              <option value="100" selected>100</option>
            </select>
          </div>
        </div>
      </div>
    </div>

    <!-- Results Status Bar & Feedback Controls -->
    <div class="flex flex-wrap items-center justify-between gap-3 text-xs text-slate-400 px-1 bg-slate-900/60 p-3 rounded-xl border border-slate-800/80">
      <div class="flex items-center gap-3">
        <div id="statusText"></div>
        <div id="timingBadge" class="hidden font-mono bg-slate-900 px-2.5 py-1 rounded border border-slate-800 text-emerald-400"></div>
      </div>
      <div class="flex items-center gap-2">
        <button id="saveQueryResultBtn" onclick="saveActiveQueryResult()" class="text-xs font-bold px-3.5 py-1.5 rounded-lg bg-gradient-to-r from-emerald-500 to-teal-500 hover:from-emerald-400 hover:to-teal-400 text-slate-950 shadow-md shadow-emerald-500/20 transition flex items-center gap-1.5" title="Lưu kết quả của câu query hiện tại vào danh sách gói bài thi">
          <span>💾</span>
          <span id="saveQueryResultLabel">Save KIS (Top 100)</span>
        </button>
        <button id="rerankBtn" onclick="executeRerank()" class="hidden text-xs font-bold px-3.5 py-1.5 rounded-lg bg-emerald-500 hover:bg-emerald-400 text-slate-950 shadow-md shadow-emerald-500/20 transition flex items-center gap-1.5 animate-pulse">
          🎯 Re-rank (0 marked)
        </button>
        <button id="clearMarksBtn" onclick="clearMarks()" class="hidden text-xs font-semibold px-2.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-rose-300 border border-slate-700/60 transition">
          ✕ Clear Marks
        </button>
      </div>
    </div>

    <!-- Video Filter & Display Toolbar -->
    <div id="videoFilterBar" class="glass rounded-xl p-2.5 px-3.5 shadow-md flex flex-wrap items-center justify-between gap-3 border border-slate-800 text-xs">
      <div class="flex items-center gap-2 flex-wrap">
        <span class="text-slate-400 font-semibold flex items-center gap-1">
          <span>🎬</span> Filter Video:
        </span>
        <select id="videoFilterSelect" onchange="onVideoFilterSelectChanged(this.value)" class="bg-slate-900 border border-slate-700 text-emerald-300 font-mono text-xs rounded-lg px-2.5 py-1 focus:outline-none focus:border-emerald-400">
          <option value="">All Videos (No filter)</option>
        </select>
        <input id="videoFilterInput" type="text" placeholder="Search Video ID..." oninput="onVideoFilterInputChanged(this.value)" class="w-40 bg-slate-950 text-slate-200 placeholder-slate-600 rounded-lg px-2.5 py-1 border border-slate-800 focus:outline-none focus:border-emerald-500 font-mono text-xs" />
        <button id="clearVideoFilterBtn" onclick="clearVideoFilter()" class="hidden px-2 py-1 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded text-[11px] transition">
          ✕ Clear Filter
        </button>
      </div>

      <div class="flex items-center gap-2">
        <span id="filteredCountBadge" class="text-[11px] text-slate-400 font-mono">Showing 0 / 0 keyframes</span>
      </div>
    </div>

    <!-- Gallery Grid -->
    <div id="resultsGrid" class="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4"></div>

  </main>
</div>

  <!-- Video Inspection Modal (Non-blocking: Queries Sidebar stays visible on the left) -->
  <div id="detailModal" onclick="if(event.target === this) closeModal()" class="fixed inset-0 z-40 bg-black/40 hidden flex items-center justify-center md:pl-80 lg:pl-96 p-2 md:p-6 overflow-y-auto">
    <div class="glass max-w-4xl w-full rounded-2xl overflow-hidden shadow-2xl border border-slate-700 flex flex-col max-h-[92vh]">
      
      <!-- Modal Header -->
      <div class="p-4 border-b border-slate-800 flex items-center justify-between bg-slate-900/40">
        <div class="flex items-center gap-3">
          <div class="w-3 h-3 rounded-full bg-emerald-400 animate-pulse"></div>
          <h3 id="modalTitle" class="font-bold text-sm text-slate-200 font-mono">Video Player</h3>
        </div>
        <button onclick="closeModal()" class="text-slate-400 hover:text-white text-xl p-1">✕</button>
      </div>

      <!-- Modal Body -->
      <div class="p-3.5 flex flex-col gap-2.5 overflow-y-auto">
        
        <!-- Video Player Element (Compact 36vh Fit - Keeps TRAKE and Controls Fully Visible) -->
        <div class="w-full max-w-[680px] mx-auto flex items-center justify-center bg-slate-950/90 rounded-xl border border-slate-800/80 p-1">
          <video id="mainVideoPlayer" controls autoplay playsinline class="w-full max-h-[36vh] object-contain rounded-lg block shadow-lg"></video>
        </div>

        <!-- Video Player Fast Seeking & Controls -->
        <div class="flex flex-wrap items-center justify-between gap-2 bg-slate-900/80 p-2 rounded-xl border border-slate-800 text-xs">
          <div class="flex items-center gap-2">
            <button onclick="seekRel(-5)" class="px-2.5 py-1.5 bg-slate-800 hover:bg-slate-700 rounded text-slate-200 font-medium">⏪ -5s</button>
            <button onclick="seekRel(-1)" class="px-2.5 py-1.5 bg-slate-800 hover:bg-slate-700 rounded text-slate-200 font-medium">⏮ -1s</button>
            <button onclick="seekRel(1)" class="px-2.5 py-1.5 bg-slate-800 hover:bg-slate-700 rounded text-slate-200 font-medium">+1s ⏭</button>
            <button onclick="seekRel(5)" class="px-2.5 py-1.5 bg-slate-800 hover:bg-slate-700 rounded text-slate-200 font-medium">+5s ⏩</button>
            <button onclick="jumpToCandidate()" class="px-3 py-1.5 bg-emerald-500/20 hover:bg-emerald-500/30 text-emerald-300 font-semibold rounded border border-emerald-500/40">
              🎯 Jump to Candidate Frame
            </button>
          </div>

          <!-- Clip Range Marking Controls -->
          <div class="flex items-center gap-1.5 bg-slate-950/80 px-2.5 py-1.5 rounded-lg border border-slate-700/60">
            <button onclick="setModalClipStart()" class="px-2 py-1 bg-emerald-500/20 hover:bg-emerald-500/30 text-emerald-300 font-semibold rounded text-[11px] border border-emerald-500/40 transition flex items-center gap-1" title="Đánh dấu mốc Start từ thời điểm phát video hiện tại">
              ⏱️ Set Start
            </button>
            <span id="modalClipStartBadge" class="font-mono text-emerald-400 text-xs font-bold px-1">--:--</span>
            <span class="text-slate-500 text-xs">──</span>
            <span id="modalClipEndBadge" class="font-mono text-amber-300 text-xs font-bold px-1">--:--</span>
            <button onclick="setModalClipEnd()" class="px-2 py-1 bg-amber-500/20 hover:bg-amber-500/30 text-amber-300 font-semibold rounded text-[11px] border border-amber-500/40 transition flex items-center gap-1" title="Đánh dấu mốc End từ thời điểm phát video hiện tại">
              ⏱️ Set End
            </button>
          </div>

          <div class="flex items-center gap-3">
            <span class="text-slate-400">Speed:</span>
            <button onclick="setSpeed(1.0)" class="speed-btn px-2 py-1 bg-slate-800 rounded text-slate-300" data-spd="1.0">1.0x</button>
            <button onclick="setSpeed(1.5)" class="speed-btn px-2 py-1 bg-slate-800 rounded text-slate-300" data-spd="1.5">1.5x</button>
            <button onclick="setSpeed(2.0)" class="speed-btn px-2 py-1 bg-slate-800 rounded text-slate-300" data-spd="2.0">2.0x</button>
          </div>
        </div>

        <!-- Metadata & Live Frame Tracker -->
        <div class="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
          <div class="bg-slate-900 p-2 rounded-lg border border-slate-800">
            <span class="text-slate-500 font-semibold block mb-0.5 text-[10px]">VIDEO ID</span>
            <span id="modalVideo" class="text-emerald-400 font-mono text-xs font-bold truncate block"></span>
          </div>
          <div class="bg-slate-900 p-2 rounded-lg border border-slate-800">
            <span class="text-slate-500 font-semibold block mb-0.5 text-[10px]">CANDIDATE FRAME</span>
            <span id="modalCandidateFrame" class="text-slate-200 font-mono text-xs truncate block"></span>
          </div>
          <div class="bg-slate-900 p-2 rounded-lg border border-slate-800">
            <span class="text-slate-500 font-semibold block mb-0.5 text-[10px]">PLAYING FRAME</span>
            <span id="modalCurrentFrame" class="text-amber-300 font-mono text-xs font-bold block">0 (00:00)</span>
          </div>
          <div class="bg-slate-900 p-2 rounded-lg border border-slate-800">
            <span class="text-slate-500 font-semibold block mb-0.5 text-[10px]">MATCH STATUS</span>
            <span id="modalPinnedStatus" class="text-slate-400 font-mono text-xs block">Not Pinned</span>
          </div>
        </div>

        <!-- Refined Speech Dialogue -->
        <div class="bg-slate-900 p-2.5 rounded-lg border border-slate-800 flex flex-col gap-1">
          <span class="text-slate-400 text-xs font-semibold">🎙️ REFINED SPEECH DIALOGUE (BM25 MATCH)</span>
          <p id="modalASR" class="text-xs text-slate-200 italic leading-relaxed"></p>
        </div>

        <!-- Mode-Specific UI in Modal -->
        <!-- 1. Q&A Answer Section (Active in Q&A Mode) -->
        <div id="modalQABox" class="bg-slate-900 p-2.5 rounded-xl border border-amber-500/30 flex flex-col gap-2">
          <div class="flex items-center justify-between text-xs">
            <span class="text-amber-400 font-bold flex items-center gap-1.5">
              💬 Q&A Answer (Câu trả lời cho Video này):
            </span>
            <span id="modalQACharCount" class="text-[10px] text-slate-500 font-mono">0/100 chars</span>
          </div>
          <div class="flex items-center gap-2 text-xs">
            <input id="modalQAInput" type="text" maxlength="100" placeholder="Nhập câu trả lời (VD: 5, Năm người, Màu đỏ, rất đẹp)..." class="flex-1 bg-slate-950 text-amber-200 placeholder-slate-600 rounded-lg px-3 py-2 border border-slate-700 focus:outline-none focus:border-amber-400 font-medium" oninput="onModalQAInputChanged()" />
            <button onclick="saveModalQAAnswer()" class="px-3.5 py-2 bg-amber-500 hover:bg-amber-400 text-slate-950 font-bold rounded-lg transition whitespace-nowrap">
              💾 Lưu Answer
            </button>
          </div>
        </div>

        <!-- 2. TRAKE Event Sequence Section (Active in TRAKE Mode) -->
        <div id="modalTrakeBox" class="bg-slate-900 p-2.5 rounded-xl border border-indigo-500/40 flex flex-col gap-2">
          <div class="flex items-center justify-between text-xs">
            <div class="flex items-center gap-1.5">
              <span class="text-indigo-400 font-bold flex items-center gap-1">⏱️ TRAKE Event Sequence:</span>
              <span id="modalTrakeCountBadge" class="font-mono text-[10px] px-2 py-0.5 rounded bg-slate-950 text-indigo-300 border border-indigo-500/30">0 events</span>
            </div>
            <div class="flex items-center gap-2">
              <button onclick="addCurrentFrameToTrake()" class="px-2.5 py-1 bg-indigo-600 hover:bg-indigo-500 text-white font-bold rounded text-[11px] transition flex items-center gap-1 shadow-sm">
                ➕ Mark Current Frame
              </button>
              <button onclick="saveModalTrakeEvents(false)" class="px-2.5 py-1 bg-indigo-500 hover:bg-indigo-400 text-slate-950 font-bold rounded text-[11px] transition flex items-center gap-1 shadow-sm" title="Lưu chuỗi sự kiện TRAKE cho video này">
                💾 Save TRAKE
              </button>
              <button onclick="clearAllTrakeEvents()" class="px-2 py-1 bg-slate-800 hover:bg-rose-900/50 text-rose-400 rounded text-[11px] transition">
                🗑️ Clear All
              </button>
            </div>
          </div>
          <div id="modalTrakeEventsList" class="flex flex-wrap gap-2 text-xs min-h-[32px] items-center">
            <span class="text-slate-500 text-[11px] italic">Chưa có mốc sự kiện nào. Hãy tua video và bấm "+ Mark Current Frame".</span>
          </div>
        </div>
      </div>

      <!-- Modal Footer -->
      <div class="p-4 border-t border-slate-800 bg-slate-900/60 flex flex-wrap items-center justify-between gap-3">
        <div class="flex flex-wrap items-center gap-2">
          <!-- Standout Pin Current Frame Button -->
          <button onclick="pinModalPlayingFrame()" class="px-3.5 py-2 bg-gradient-to-r from-amber-500 to-emerald-500 hover:from-amber-400 hover:to-emerald-400 text-slate-950 font-extrabold text-xs rounded-lg shadow-lg shadow-emerald-500/20 transition flex items-center gap-1.5" title="Chọn chính xác khung hình đang dừng/phát làm kết quả chính">
            🎯 Pin Current Frame as Match
          </button>
          <!-- Neighbor Flood Button -->
          <button onclick="floodNeighborsModal()" class="px-3 py-2 bg-gradient-to-r from-sky-600 to-teal-600 hover:from-sky-500 hover:to-teal-500 text-white font-bold text-xs rounded-lg shadow-md shadow-sky-500/20 transition flex items-center gap-1.5" title="Tự động điền 100 khung hình liền kề xung quanh frame này để tối đa hóa điểm R@k">
            ⚡ Flood Top 100 Neighbors
          </button>
          <button id="modalSaveTrakeFooterBtn" onclick="saveModalTrakeEvents(true)" class="px-3.5 py-2 bg-gradient-to-r from-indigo-600 to-indigo-500 hover:from-indigo-500 hover:to-indigo-400 text-white font-bold text-xs rounded-lg shadow-md shadow-indigo-500/30 transition flex items-center gap-1.5 hidden">
            💾 Save TRAKE & Close
          </button>
          <button id="modalSaveQAFooterBtn" onclick="saveModalQAAnswer(true)" class="px-3.5 py-2 bg-gradient-to-r from-amber-500 to-amber-400 hover:from-amber-400 hover:to-amber-300 text-slate-950 font-bold text-xs rounded-lg shadow-md shadow-amber-500/20 transition flex items-center gap-1.5 hidden">
            💾 Save Q&A & Close
          </button>
          <button id="modalCopyBtn" class="px-3 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 font-semibold text-xs rounded-lg transition">
            📋 Copy Candidate
          </button>
          <button id="modalCopyCurrentBtn" onclick="copyCurrentPlayingFrame(this)" class="px-3 py-2 bg-slate-800 hover:bg-slate-700 text-amber-300 border border-amber-500/30 font-semibold text-xs rounded-lg transition">
            📸 Copy Paused Frame
          </button>
          <button onclick="askVlmFromModalClip()" class="px-3 py-2 bg-slate-800 hover:bg-slate-700 text-emerald-400 border border-emerald-500/30 font-semibold text-xs rounded-lg transition flex items-center gap-1.5" title="Hỏi VLM về đoạn clip ngắn đã đánh dấu [Start, End]">
            🤖 Ask VLM About Clip
          </button>
          <button onclick="markCurrentModalFrameAndRerank()" class="px-3 py-2 bg-emerald-500/20 hover:bg-emerald-500/30 text-emerald-300 border border-emerald-500/40 font-bold text-xs rounded-lg transition flex items-center gap-1.5">
            🎯 Mark & Re-rank
          </button>
        </div>
        <button onclick="closeModal()" class="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs rounded-lg transition">
          Close
        </button>
      </div>
    </div>
  </div>

  <!-- Floating AI Assistant Open Button (when drawer is minimized or closed) -->
  <div id="aiChatFab" class="fixed bottom-6 right-6 z-50">
    <button onclick="toggleChatbot()" class="glass px-4 py-3 rounded-2xl bg-gradient-to-r from-emerald-500 to-teal-600 hover:from-emerald-400 hover:to-teal-500 text-slate-950 font-bold text-xs shadow-2xl shadow-emerald-500/40 border border-emerald-300/50 transition-all transform hover:scale-105 active:scale-95 flex items-center gap-2.5">
      <span class="text-xl">🤖</span>
      <div class="flex flex-col text-left leading-tight">
        <span class="font-extrabold text-[12px] text-slate-950">VLM Assistant</span>
        <span id="fabModelText" class="text-[9px] text-slate-900/80 font-mono font-semibold">antigravity/gemini-3.6-flash-medium</span>
      </div>
      <span id="aiFabBadge" class="hidden px-1.5 py-0.5 bg-slate-950 text-emerald-400 text-[10px] font-mono rounded-full border border-emerald-500/30">0</span>
    </button>
  </div>

  <!-- AI Chatbot Drawer / Window -->
  <div id="aiChatDrawer" class="fixed bottom-6 right-6 z-50 w-[460px] max-w-[calc(100vw-2rem)] h-[640px] max-h-[calc(100vh-5rem)] glass rounded-2xl shadow-2xl border border-emerald-500/30 flex flex-col overflow-hidden transition-all duration-300 hidden">
    
    <!-- Chatbot Header -->
    <div class="px-4 py-3 bg-slate-900/90 border-b border-slate-800 flex items-center justify-between">
      <div class="flex items-center gap-2.5 min-w-0">
        <div class="w-7 h-7 rounded-lg bg-emerald-500/20 text-emerald-400 flex items-center justify-center font-bold text-sm border border-emerald-500/30">
          🤖
        </div>
        <div class="min-w-0">
          <div class="flex items-center gap-1.5">
            <h3 class="font-bold text-xs text-slate-200 truncate">VLM Video Assistant</h3>
            <span id="drawerModelPill" class="text-[9px] px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-mono">antigravity/gemini-3.6-flash-medium</span>
          </div>
          <p id="aiChatActiveVideoLabel" class="text-[10px] text-slate-400 truncate font-mono">No video selected</p>
        </div>
      </div>

      <div class="flex items-center gap-1">
        <button onclick="toggleChatSettings()" class="p-1.5 text-slate-400 hover:text-emerald-400 hover:bg-slate-800 rounded-lg transition" title="Chatbot Settings (API Key & Provider)">
          ⚙️
        </button>
        <button onclick="clearChatHistory()" class="p-1.5 text-slate-400 hover:text-rose-400 hover:bg-slate-800 rounded-lg transition" title="Clear Conversation">
          🗑️
        </button>
        <button onclick="toggleChatbot()" class="p-1.5 text-slate-400 hover:text-white hover:bg-slate-800 rounded-lg transition" title="Close AI Assistant">
          ✕
        </button>
      </div>
    </div>

    <!-- Active Video Context Bar / Video Selector -->
    <div class="px-4 py-2 bg-slate-950/60 border-b border-slate-800/80 flex items-center justify-between text-[11px] text-slate-400">
      <div class="flex items-center gap-1.5 min-w-0">
        <span class="text-slate-500 font-semibold">Video:</span>
        <select id="aiVideoSelect" onchange="onActiveVideoChange(this.value)" class="bg-slate-900 text-emerald-300 font-mono text-[11px] rounded px-1.5 py-0.5 border border-slate-700/60 focus:outline-none max-w-[170px] truncate">
          <option value="">(Chưa chọn video)</option>
        </select>
      </div>
      <div id="aiContextStats" class="font-mono text-[10px] text-emerald-400/80 truncate">
        0 ASR • 0 OCR
      </div>
    </div>

    <!-- VLM Clip Interval & Range Marking Bar -->
    <div class="px-4 py-2.5 bg-slate-900/90 border-b border-slate-800 flex flex-col gap-1.5">
      <div class="flex items-center justify-between text-[11px]">
        <span class="text-slate-300 font-semibold flex items-center gap-1">
          🎞️ <span class="text-emerald-400">VLM Clip Range:</span>
        </span>
        <span id="aiClipDurationBadge" class="font-mono text-[10px] px-2 py-0.5 rounded bg-slate-950 text-amber-400 border border-slate-800 font-medium">
          Chưa đánh dấu
        </span>
      </div>
      <div class="flex items-center gap-2 text-xs">
        <div class="flex items-center gap-1 flex-1 min-w-0">
          <span class="text-slate-500 text-[10px] font-semibold">Start:</span>
          <input id="aiClipStartInput" type="number" step="0.1" min="0" placeholder="0.0" class="w-full bg-slate-950 text-emerald-300 font-mono text-xs rounded px-2 py-1 border border-slate-700 focus:outline-none focus:border-emerald-500" oninput="onClipRangeChanged()" />
          <button onclick="setClipStartFromPlayer()" class="px-1.5 py-1 bg-slate-800 hover:bg-slate-700 text-emerald-400 rounded text-[10px] border border-slate-700" title="Lấy mốc thời gian hiện tại từ Video Player">⏱️</button>
        </div>
        <span class="text-slate-600 font-bold">→</span>
        <div class="flex items-center gap-1 flex-1 min-w-0">
          <span class="text-slate-500 text-[10px] font-semibold">End:</span>
          <input id="aiClipEndInput" type="number" step="0.1" min="0" placeholder="0.0" class="w-full bg-slate-950 text-amber-300 font-mono text-xs rounded px-2 py-1 border border-slate-700 focus:outline-none focus:border-amber-500" oninput="onClipRangeChanged()" />
          <button onclick="setClipEndFromPlayer()" class="px-1.5 py-1 bg-slate-800 hover:bg-slate-700 text-amber-300 rounded text-[10px] border border-slate-700" title="Lấy mốc thời gian hiện tại từ Video Player">⏱️</button>
        </div>
        <button onclick="autoFillClipWindow()" class="px-2 py-1 bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 rounded text-[10px] whitespace-nowrap font-medium transition" title="Tự động lấy khoảng ±5s quanh Keyframe">
          ⚡ Auto ±5s
        </button>
      </div>
    </div>

    <!-- Q&A Final Answer Bar inside Chatbot -->
    <div id="aiQAAnswerBox" class="px-4 py-2.5 bg-slate-900/95 border-b border-slate-800 flex flex-col gap-1.5">
      <div class="flex items-center justify-between text-[11px]">
        <span class="text-amber-400 font-bold flex items-center gap-1">
          ✍️ <span id="aiQAAnswerLabel">Q&A Answer Box:</span>
        </span>
        <span id="aiQACharCount" class="text-[10px] text-slate-500 font-mono">0/100 chars</span>
      </div>
      <div class="flex items-center gap-2 text-xs">
        <input id="aiQAAnswerInput" type="text" maxlength="100" placeholder="Nhập câu trả lời cho video này (VD: 5, Năm người, Màu đỏ)..." class="flex-1 bg-slate-950 text-amber-200 placeholder-slate-600 rounded-lg px-2.5 py-1.5 border border-amber-500/30 focus:outline-none focus:border-amber-400 font-medium text-xs" oninput="onDrawerQAInputChanged()" />
        <button onclick="saveDrawerQAAnswer()" class="px-3 py-1.5 bg-amber-500 hover:bg-amber-400 text-slate-950 font-bold rounded-lg transition whitespace-nowrap text-xs">
          💾 Lưu
        </button>
      </div>
    </div>

    <!-- Settings Overlay (Hidden by default) -->
    <div id="aiSettingsOverlay" class="p-4 bg-slate-900/95 border-b border-slate-800 flex flex-col gap-3 text-xs hidden">
      <div class="flex items-center justify-between">
        <span class="font-bold text-slate-200">⚙️ AI Model & API Configuration</span>
        <button onclick="toggleChatSettings()" class="text-slate-400 hover:text-white">✕</button>
      </div>
      
      <div class="flex flex-col gap-1">
        <label class="text-[11px] text-slate-400 font-medium">Provider:</label>
        <select id="aiProviderSelect" onchange="onProviderChange()" class="bg-slate-950 text-slate-200 rounded px-2 py-1.5 border border-slate-700 focus:outline-none">
          <option value="omniroute" selected>OmniRoute (Localhost 20128 - Gemini 3.6 Flash Medium)</option>
          <option value="openrouter">OpenRouter (DeepSeek, Claude, Llama...)</option>
          <option value="openai">OpenAI (GPT-4o, GPT-4o-mini)</option>
          <option value="custom">Custom OpenAI-compatible URL</option>
        </select>
      </div>

      <div id="aiCustomUrlGroup" class="flex flex-col gap-1 hidden">
        <label class="text-[11px] text-slate-400 font-medium">Custom API URL:</label>
        <input id="aiCustomUrlInput" type="text" placeholder="http://localhost:20128/v1/chat/completions" class="bg-slate-950 text-slate-200 rounded px-2.5 py-1.5 border border-slate-700 focus:outline-none" />
      </div>

      <div class="flex flex-col gap-1">
        <label class="text-[11px] text-slate-400 font-medium">Model Identifier:</label>
        <input id="aiModelInput" type="text" value="antigravity/gemini-3.6-flash-medium" placeholder="antigravity/gemini-3.6-flash-medium" class="bg-slate-950 text-slate-200 rounded px-2.5 py-1.5 border border-slate-700 focus:outline-none" />
      </div>

      <div class="flex flex-col gap-1">
        <label class="text-[11px] text-slate-400 font-medium">API Key (Optional override):</label>
        <input id="aiApiKeyInput" type="password" value="omniroute" placeholder="omniroute" class="bg-slate-950 text-slate-200 rounded px-2.5 py-1.5 border border-slate-700 focus:outline-none" />
        <span class="text-[10px] text-slate-500">Mặc định: 'omniroute' (OmniRoute localhost proxy).</span>
      </div>

      <div class="flex items-center justify-between pt-1 border-t border-slate-800">
        <label class="flex items-center gap-2 text-[11px] text-slate-300 cursor-pointer">
          <input id="aiAutoOpenCheckbox" type="checkbox" checked class="accent-emerald-500 rounded" />
          Tự động mở khi bấm Mark Correct
        </label>
        <button onclick="saveChatSettings()" class="px-3 py-1 bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-bold rounded text-xs transition">
          Lưu
        </button>
      </div>
    </div>

    <!-- Quick Action Prompt Chips -->
    <div class="p-2.5 bg-slate-900/50 border-b border-slate-800/60 flex flex-wrap gap-1.5 text-[11px]">
      <button onclick="sendQuickPrompt('Tóm tắt nội dung và lời thoại chính của đoạn video này.')" class="px-2 py-1 bg-slate-800/80 hover:bg-slate-700 text-emerald-300 rounded-lg border border-emerald-500/20 transition truncate max-w-[195px]">
        🎙️ Tóm tắt đoạn này
      </button>
      <button onclick="sendQuickPrompt('Gợi ý các từ khóa truy vấn tiếp theo dựa trên nội dung video này.')" class="px-2 py-1 bg-slate-800/80 hover:bg-slate-700 text-amber-300 rounded-lg border border-amber-500/20 transition truncate max-w-[195px]">
        🔍 Gợi ý từ khóa tiếp
      </button>
      <button onclick="sendQuickPrompt('Các địa điểm, tổ chức, nhân vật và sự kiện xuất hiện trong cảnh này là gì?')" class="px-2 py-1 bg-slate-800/80 hover:bg-slate-700 text-teal-300 rounded-lg border border-teal-500/20 transition truncate max-w-[195px]">
        🏢 Thực thể & Địa điểm
      </button>
      <button onclick="sendQuickPrompt('Liệt kê mốc thời gian chi tiết các sự kiện diễn ra trong video.')" class="px-2 py-1 bg-slate-800/80 hover:bg-slate-700 text-indigo-300 rounded-lg border border-indigo-500/20 transition truncate max-w-[195px]">
        ⏱️ Mốc thời gian
    </div>

    <!-- Chat Messages Scroll Area -->
    <div id="aiChatMessages" class="flex-1 p-4 overflow-y-auto flex flex-col gap-3 text-xs">
      <div class="self-start max-w-[90%] p-3 rounded-2xl bg-slate-900 border border-slate-800 text-slate-300 leading-relaxed">
        👋 Xin chào! Tôi là Trợ lý AI phân tích video. Khi bạn đánh dấu một video chính xác, tôi sẽ tự động trích xuất lời thoại ASR và chữ OCR của video đó để hỗ trợ bạn tóm tắt, trả lời câu hỏi và gợi ý truy vấn tiếp theo!
      </div>
    </div>

    <!-- Chat Input Area -->
    <div class="p-3 bg-slate-900/90 border-t border-slate-800 flex items-end gap-2">
      <textarea id="aiChatInput" rows="1" placeholder="Hỏi về video này... (Enter gửi, Shift+Enter xuống dòng)" class="flex-1 bg-slate-950 text-slate-100 placeholder-slate-500 text-xs rounded-xl px-3 py-2 border border-slate-700/60 focus:outline-none focus:ring-1 focus:ring-emerald-500/50 resize-none max-h-24" onkeydown="handleChatInputKey(event)"></textarea>
      <button id="aiSendBtn" onclick="sendChatMessage()" class="p-2 bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-bold rounded-xl transition shadow-md shadow-emerald-500/20 flex items-center justify-center">
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"></path></svg>
      </button>
    </div>

  </div>

  <script>
    function escapeHTML(str) {
      if (str === null || str === undefined) return '';
      return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
    }

    let currentResults = [];
    let currentModalItem = null;
    let markedMap = new Map(); // Positive references: key -> { video_id, frame_idx, pts_time, isPinned }
    let keywordsList = [];
    let nextKwId = 1;
    const videoElem = document.getElementById('mainVideoPlayer');

    // Task Modes (plan.txt): 'kis' | 'qa' | 'trake'
    let currentTaskMode = 'kis';
    let qaAnswersMap = {}; // video_id -> answer_string (max 100 chars)
    let trakeEventsMap = {}; // video_id -> [ { frame_idx: int, pts_time: float } ]

    // Query Package Session State (plan.txt & rules.txt)
    let queryPackage = []; // Array of { id, filename, prompt, mode, status: 'unanswered'|'saved', savedData: {...} }
    let activeQueryId = 'query-1';

    // Video Filter State
    let activeVideoFilter = '';

    // Video Clip Range Tracking for VLM
    let videoClipRanges = {}; // vid -> { start_sec: float, end_sec: float }
    let modalClipStartSec = null;
    let modalClipEndSec = null;

    // AI Chatbot State
    let currentActiveChatVideo = null; // { video_id, frame_idx, pts_time }
    let videoChatHistories = {}; // video_id -> [ { role, content, time } ]
    let isChatSending = false;
    let aiSettings = {
      provider: 'omniroute',
      model: 'antigravity/gemini-3.6-flash-medium',
      apiKey: 'omniroute',
      customUrl: 'http://localhost:20128/v1/chat/completions',
      autoOpen: false // Fix: Disabled auto popup
    };

    // --- Session Persistence (localStorage) ---
    const SESSION_STORAGE_KEY = 'aic_studio_session_v3';

    function saveSessionToLocalStorage() {
      try {
        const payload = {
          queryPackage: queryPackage,
          activeQueryId: activeQueryId,
          markedMap: Array.from(markedMap.entries()),
          qaAnswersMap: qaAnswersMap,
          trakeEventsMap: trakeEventsMap,
          videoClipRanges: videoClipRanges,
          currentTaskMode: currentTaskMode,
          currentResults: currentResults.slice(0, 100)
        };
        localStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify(payload));
      } catch (e) {
        console.warn("Unable to save session to localStorage:", e);
      }
    }

    function loadSessionFromLocalStorage() {
      try {
        const raw = localStorage.getItem(SESSION_STORAGE_KEY);
        if (!raw) return false;
        const data = JSON.parse(raw);
        if (!data || !data.queryPackage || data.queryPackage.length === 0) return false;

        queryPackage = data.queryPackage || [];
        activeQueryId = data.activeQueryId || (queryPackage[0] ? queryPackage[0].id : 'query-1');
        markedMap = new Map(data.markedMap || []);
        qaAnswersMap = data.qaAnswersMap || {};
        trakeEventsMap = data.trakeEventsMap || {};
        videoClipRanges = data.videoClipRanges || {};
        currentTaskMode = data.currentTaskMode || 'kis';
        currentResults = data.currentResults || [];

        renderQueryNavigator();
        selectQuery(activeQueryId, false);
        if (currentResults.length > 0) {
          populateVideoFilterDropdown(currentResults);
          renderGrid(currentResults);
          updateMarkedUI();
        }
        return true;
      } catch (e) {
        console.warn("Failed to restore session from localStorage:", e);
        return false;
      }
    }

    function resetSessionStorage() {
      if (confirm("Bạn có chắc chắn muốn reset toàn bộ phiên làm việc (xóa toàn bộ câu hỏi và đáp án đã lưu)?")) {
        try {
          localStorage.removeItem(SESSION_STORAGE_KEY);
        } catch (e) {}
        queryPackage = [];
        markedMap.clear();
        qaAnswersMap = {};
        trakeEventsMap = {};
        currentResults = [];
        activeVideoFilter = '';
        renderQueryNavigator();
        showQuickToast("🧹 Đã làm sạch toàn bộ phiên làm việc!");
      }
    }

    function showQuickToast(msg) {
      let toast = document.getElementById('quickToast');
      if (!toast) {
        toast = document.createElement('div');
        toast.id = 'quickToast';
        toast.className = 'fixed top-20 right-6 z-50 glass px-4 py-2.5 rounded-xl border border-emerald-500/40 text-emerald-300 text-xs font-bold shadow-2xl transition-all duration-300 transform translate-y-0 opacity-100 flex items-center gap-2';
        document.body.appendChild(toast);
      }
      toast.innerHTML = msg;
      toast.classList.remove('hidden');
      toast.style.opacity = '1';
      setTimeout(() => {
        toast.style.opacity = '0';
        setTimeout(() => toast.classList.add('hidden'), 300);
      }, 2500);
    }

    // --- Video-Level Filtering ---
    function populateVideoFilterDropdown(results) {
      const select = document.getElementById('videoFilterSelect');
      if (!select) return;
      
      const counts = {};
      results.forEach(r => {
        counts[r.video_id] = (counts[r.video_id] || 0) + 1;
      });

      const sortedVids = Object.keys(counts).sort((a, b) => counts[b] - counts[a]);

      let html = `<option value="">All Videos (${results.length} frames)</option>`;
      sortedVids.forEach(v => {
        html += `<option value="${v}">${v} (${counts[v]} frames)</option>`;
      });
      select.innerHTML = html;
      select.value = activeVideoFilter;
    }

    function onVideoFilterSelectChanged(val) {
      activeVideoFilter = (val || '').trim();
      const input = document.getElementById('videoFilterInput');
      if (input) input.value = activeVideoFilter;
      applyVideoFilter();
    }

    function onVideoFilterInputChanged(val) {
      activeVideoFilter = (val || '').trim();
      const select = document.getElementById('videoFilterSelect');
      if (select) select.value = select.querySelector(`option[value="${activeVideoFilter}"]`) ? activeVideoFilter : '';
      applyVideoFilter();
    }

    function clearVideoFilter() {
      activeVideoFilter = '';
      const input = document.getElementById('videoFilterInput');
      if (input) input.value = '';
      const select = document.getElementById('videoFilterSelect');
      if (select) select.value = '';
      applyVideoFilter();
    }

    function isolateVideo(vid) {
      activeVideoFilter = vid;
      const input = document.getElementById('videoFilterInput');
      if (input) input.value = vid;
      const select = document.getElementById('videoFilterSelect');
      if (select) select.value = vid;
      applyVideoFilter();
    }

    function applyVideoFilter() {
      const clearBtn = document.getElementById('clearVideoFilterBtn');
      if (clearBtn) {
        if (activeVideoFilter) clearBtn.classList.remove('hidden');
        else clearBtn.classList.add('hidden');
      }
      renderGrid(currentResults);
    }

    // --- Neighbor Flooding Helper ---
    function generateNeighborFrames(vid, centerFrame, total = 100, step = 1) {
      const rows = [];
      rows.push({ video_id: vid, frame_idx: centerFrame });
      
      let offset = 1;
      while (rows.length < total) {
        const fBefore = centerFrame - (offset * step);
        const fAfter = centerFrame + (offset * step);
        
        if (fBefore >= 0 && rows.length < total) {
          rows.push({ video_id: vid, frame_idx: fBefore });
        }
        if (rows.length < total) {
          rows.push({ video_id: vid, frame_idx: fAfter });
        }
        offset++;
      }
      return rows;
    }

    // --- Shared CSV Output Builder ---
    function buildCSVLinesForQuery(q) {
      if (!q) return [];
      const mode = q.mode || 'kis';
      const saved = q.savedData || {};
      const lines = [];

      if (mode === 'kis') {
        // Priority 1: Flooded neighbor frames
        if (saved.floodFrames && saved.floodFrames.length > 0) {
          saved.floodFrames.slice(0, 100).forEach(f => {
            lines.push(`${f.video_id},${f.frame_idx}`);
          });
          return lines;
        }

        // Priority 2: Pinned frame as #1
        if (saved.pinnedFrame) {
          lines.push(`${saved.pinnedFrame.video_id},${saved.pinnedFrame.frame_idx}`);
        }

        // Priority 3: Marked positive items
        if (saved.markedItems && saved.markedItems.length > 0) {
          saved.markedItems.forEach(item => {
            const row = `${item.video_id},${item.frame_idx}`;
            if (!lines.includes(row) && lines.length < 100) {
              lines.push(row);
            }
          });
        }

        // Priority 4: Fill remaining from search results
        const resList = saved.results || (q.id === activeQueryId ? currentResults : []);
        resList.forEach(r => {
          const row = `${r.video_id},${r.frame_idx}`;
          if (!lines.includes(row) && lines.length < 100) {
            lines.push(row);
          }
        });
      } else if (mode === 'qa') {
        const ans = saved.qaAnswer !== undefined ? saved.qaAnswer : (Object.values(qaAnswersMap)[0] || '');
        const ansField = formatCSVField(ans);

        // Priority 1: Flooded neighbor frames
        if (saved.floodFrames && saved.floodFrames.length > 0) {
          saved.floodFrames.slice(0, 100).forEach(f => {
            lines.push(`${f.video_id},${f.frame_idx},${ansField}`);
          });
          return lines;
        }

        // Priority 2: Pinned frame
        if (saved.pinnedFrame) {
          lines.push(`${saved.pinnedFrame.video_id},${saved.pinnedFrame.frame_idx},${ansField}`);
        }

        // Priority 3: Marked items
        if (saved.markedItems && saved.markedItems.length > 0) {
          saved.markedItems.forEach(item => {
            const row = `${item.video_id},${item.frame_idx},${ansField}`;
            if (!lines.some(l => l.startsWith(`${item.video_id},${item.frame_idx}`)) && lines.length < 100) {
              lines.push(row);
            }
          });
        }

        // Priority 4: Search results
        const resList = saved.results || (q.id === activeQueryId ? currentResults : []);
        resList.forEach(r => {
          const row = `${r.video_id},${r.frame_idx},${ansField}`;
          if (!lines.some(l => l.startsWith(`${r.video_id},${r.frame_idx}`)) && lines.length < 100) {
            lines.push(row);
          }
        });
      } else if (mode === 'trake') {
        if (saved.trakeEvents && saved.trakeEvents.length > 0) {
          const vid = saved.trakeVideoId || (saved.markedItems && saved.markedItems[0] ? saved.markedItems[0].video_id : 'L21_V001');
          const sortedFrames = saved.trakeEvents.map(e => e.frame_idx).sort((a, b) => a - b);
          lines.push(`${vid},${sortedFrames.join(',')}`);
        } else {
          const resList = saved.results || (q.id === activeQueryId ? currentResults : []);
          const targetVids = [];
          if (saved.markedItems) {
            saved.markedItems.forEach(v => { if (!targetVids.includes(v.video_id)) targetVids.push(v.video_id); });
          }
          resList.forEach(r => { if (!targetVids.includes(r.video_id)) targetVids.push(r.video_id); });

          targetVids.forEach(vid => {
            if (lines.length >= 100) return;
            const events = (trakeEventsMap && trakeEventsMap[vid]) || [];
            if (events.length > 0) {
              const sortedFrames = events.map(e => e.frame_idx).sort((a, b) => a - b);
              lines.push(`${vid},${sortedFrames.join(',')}`);
            } else {
              const kfs = resList.filter(r => r.video_id === vid).map(r => r.frame_idx).slice(0, 4);
              if (kfs.length > 0) {
                lines.push(`${vid},${kfs.join(',')}`);
              }
            }
          });
        }
      }

      // Fallback if completely empty
      if (lines.length === 0) {
        if (mode === 'kis') lines.push("L21_V001,0");
        else if (mode === 'qa') lines.push("L21_V001,0,");
        else if (mode === 'trake') lines.push("L21_V001,0,30,60,90");
      }

      return lines;
    }

    // --- Query Package Upload & Robust Multi-Format Parser ---
    function detectQueryMode(filename, promptText) {
      const fn = (filename || '').toLowerCase();
      const pt = (promptText || '').toLowerCase();
      
      if (fn.includes('-qa') || fn.includes('_qa') || fn.includes('/qa/') || fn.endsWith('qa.txt') || fn.endsWith('qa.csv') || fn.endsWith('qa.json')) return 'qa';
      if (fn.includes('-trake') || fn.includes('_trake') || fn.includes('/trake/') || fn.endsWith('trake.txt') || fn.endsWith('trake.csv') || fn.endsWith('trake.json')) return 'trake';
      if (fn.includes('-kis') || fn.includes('_kis') || fn.includes('/kis/') || fn.endsWith('kis.txt') || fn.endsWith('kis.csv') || fn.endsWith('kis.json')) return 'kis';
      
      if (pt.includes('hỏi đáp') || pt.includes('câu hỏi') || pt.includes('bao nhiêu') || pt.includes('ở đâu') || pt.includes('ai là') || pt.includes('màu gì') || pt.includes('biển số')) {
        return 'qa';
      }
      if (pt.includes('chuỗi sự kiện') || pt.includes('mốc thời gian') || pt.includes('các phân cảnh từ lúc') || pt.includes('trake')) {
        return 'trake';
      }
      return 'kis';
    }

    function cleanQueryId(filename) {
      const parts = filename.replace(/\\/g, '/').split('/').filter(p => p && p !== '.' && !p.startsWith('__MACOSX'));
      const base = parts.pop().replace(/\.[^/.]+$/, "");
      
      if (parts.length > 0) {
        const parent = parts[parts.length - 1].toLowerCase();
        if (['kis', 'qa', 'trake'].includes(parent) && !base.toLowerCase().includes(parent)) {
          return `${parent}-${base}`.replace(/\s+/g, '_');
        }
      }
      return base.replace(/\s+/g, '_');
    }

    function parseAndAddQueries(filePath, rawContent, targetList) {
      if (!rawContent || !rawContent.trim()) return;
      const trimmed = rawContent.trim();
      const fn = filePath.toLowerCase();

      // Case 1: JSON file containing query definitions
      if (fn.endsWith('.json')) {
        try {
          const parsed = JSON.parse(trimmed);
          if (Array.isArray(parsed)) {
            parsed.forEach((item, idx) => {
              if (typeof item === 'string') {
                const qid = `${cleanQueryId(filePath)}_${idx + 1}`;
                targetList.push({
                  id: qid,
                  filename: `${qid}.txt`,
                  prompt: item.trim(),
                  mode: detectQueryMode(qid, item),
                  status: 'unanswered',
                  savedData: null
                });
              } else if (item && typeof item === 'object') {
                const qid = (item.id || item.query_id || item.name || `${cleanQueryId(filePath)}_${idx + 1}`).toString().replace(/\s+/g, '_');
                const prompt = (item.prompt || item.query || item.text || item.description || JSON.stringify(item)).trim();
                const mode = item.mode || detectQueryMode(qid + ' ' + (item.type || ''), prompt);
                targetList.push({
                  id: qid,
                  filename: `${qid}.txt`,
                  prompt: prompt,
                  mode: mode,
                  status: 'unanswered',
                  savedData: null
                });
              }
            });
            return;
          } else if (typeof parsed === 'object' && parsed !== null) {
            const arr = parsed.queries || parsed.data || parsed.items;
            if (Array.isArray(arr)) {
              arr.forEach((item, idx) => {
                const qid = (item.id || item.query_id || item.name || `${cleanQueryId(filePath)}_${idx + 1}`).toString().replace(/\s+/g, '_');
                const prompt = (item.prompt || item.query || item.text || item.description || JSON.stringify(item)).trim();
                const mode = item.mode || detectQueryMode(qid, prompt);
                targetList.push({
                  id: qid,
                  filename: `${qid}.txt`,
                  prompt: prompt,
                  mode: mode,
                  status: 'unanswered',
                  savedData: null
                });
              });
              return;
            } else {
              const keys = Object.keys(parsed);
              if (keys.length > 0) {
                keys.forEach(k => {
                  const val = parsed[k];
                  const prompt = typeof val === 'string' ? val.trim() : (val.prompt || val.text || JSON.stringify(val));
                  const mode = (typeof val === 'object' && val.mode) ? val.mode : detectQueryMode(k, prompt);
                  targetList.push({
                    id: k.replace(/\s+/g, '_'),
                    filename: `${k}.txt`,
                    prompt: prompt,
                    mode: mode,
                    status: 'unanswered',
                    savedData: null
                  });
                });
                return;
              }
            }
          }
        } catch (e) {}
      }

      // Case 2: Plain text / CSV files
      const qid = cleanQueryId(filePath);
      const mode = detectQueryMode(filePath, trimmed);

      const existing = targetList.find(q => q.id === qid);
      if (existing) {
        existing.prompt = trimmed;
        existing.mode = mode;
      } else {
        targetList.push({
          id: qid,
          filename: filePath.split('/').pop(),
          prompt: trimmed,
          mode: mode,
          status: 'unanswered',
          savedData: null
        });
      }
    }

    async function handleQueryFileUpload(event) {
      const files = event.target.files;
      if (!files || files.length === 0) return;

      showQuickToast("⏳ Đang giải nén và nạp gói câu hỏi...");
      let loadedQueries = [];

      try {
        for (let i = 0; i < files.length; i++) {
          const file = files[i];
          const name = file.name.toLowerCase();

          if (name.endsWith('.zip')) {
            if (typeof JSZip === 'undefined') {
              throw new Error("Thư viện JSZip chưa sẵn sàng. Vui lòng kiểm tra kết nối mạng.");
            }
            const zip = await JSZip.loadAsync(file);
            const entries = [];
            
            zip.forEach((relPath, zipEntry) => {
              const basename = relPath.split('/').pop();
              if (!zipEntry.dir && !relPath.includes('__MACOSX') && !basename.startsWith('.') && !relPath.endsWith('.DS_Store')) {
                entries.push({ relPath, zipEntry });
              }
            });

            // Natural sort entries by query path/name
            entries.sort((a, b) => a.relPath.localeCompare(b.relPath, undefined, { numeric: true, sensitivity: 'base' }));

            for (const item of entries) {
              const text = await item.zipEntry.async('string');
              parseAndAddQueries(item.relPath, text, loadedQueries);
            }
          } else if (name.endsWith('.txt') || name.endsWith('.json') || name.endsWith('.csv')) {
            const text = await file.text();
            parseAndAddQueries(file.name, text, loadedQueries);
          }
        }

        if (loadedQueries.length > 0) {
          // Replace current query package with loaded package
          queryPackage = loadedQueries;
          activeQueryId = queryPackage[0].id;
          
          // Reset sidebar filters to 'all' so all loaded queries are visible
          sidebarModeFilter = 'all';
          sidebarSearchText = '';
          const filterInput = document.getElementById('sidebarFilterInput');
          if (filterInput) filterInput.value = '';
          const clearBtn = document.getElementById('sidebarFilterClearBtn');
          if (clearBtn) clearBtn.classList.add('hidden');
          
          // Update mode filter tabs UI
          ['all', 'kis', 'qa', 'trake'].forEach(t => {
            const btn = document.getElementById(`filterTab_${t}`);
            if (!btn) return;
            if (t === 'all') {
              btn.className = 'flex-1 py-1.5 px-1.5 rounded-lg text-center transition text-[11px] font-bold bg-slate-800 text-white border border-slate-700 shadow-sm';
            } else if (t === 'kis') {
              btn.className = 'flex-1 py-1.5 px-1.5 rounded-lg text-center transition text-[11px] font-bold text-sky-400 hover:bg-slate-900/60';
            } else if (t === 'qa') {
              btn.className = 'flex-1 py-1.5 px-1.5 rounded-lg text-center transition text-[11px] font-bold text-purple-400 hover:bg-slate-900/60';
            } else if (t === 'trake') {
              btn.className = 'flex-1 py-1.5 px-1.5 rounded-lg text-center transition text-[11px] font-bold text-amber-400 hover:bg-slate-900/60';
            }
          });

          renderQueryNavigator();
          saveSessionToLocalStorage();
          selectQuery(activeQueryId, false); // Don't trigger heavy search blocking
          showQuickToast(`✅ Đã nạp thành công ${loadedQueries.length} câu truy vấn!`);
        } else {
          alert("Không tìm thấy file câu hỏi hợp lệ trong tệp đã chọn.");
        }
      } catch (err) {
        console.error("Lỗi khi tải gói câu hỏi:", err);
        alert("Lỗi tải gói câu hỏi: " + err.message);
      } finally {
        event.target.value = '';
      }
    }

    function addQueryToSession(filename, rawText) {
      parseAndAddQueries(filename, rawText, queryPackage);
      saveSessionToLocalStorage();
    }

    let sidebarModeFilter = 'all';
    let sidebarSearchText = '';

    function setSidebarModeFilter(mode) {
      sidebarModeFilter = mode;
      
      const tabs = ['all', 'kis', 'qa', 'trake'];
      tabs.forEach(t => {
        const btn = document.getElementById(`filterTab_${t}`);
        if (!btn) return;
        if (t === mode) {
          if (t === 'all') {
            btn.className = 'flex-1 py-1.5 px-1.5 rounded-lg text-center transition text-[11px] font-bold bg-slate-800 text-white border border-slate-700 shadow-sm';
          } else if (t === 'kis') {
            btn.className = 'flex-1 py-1.5 px-1.5 rounded-lg text-center transition text-[11px] font-bold bg-sky-500/20 text-sky-300 border border-sky-500/40 shadow-sm';
          } else if (t === 'qa') {
            btn.className = 'flex-1 py-1.5 px-1.5 rounded-lg text-center transition text-[11px] font-bold bg-purple-500/20 text-purple-300 border border-purple-500/40 shadow-sm';
          } else if (t === 'trake') {
            btn.className = 'flex-1 py-1.5 px-1.5 rounded-lg text-center transition text-[11px] font-bold bg-amber-500/20 text-amber-300 border border-amber-500/40 shadow-sm';
          }
        } else {
          if (t === 'all') {
            btn.className = 'flex-1 py-1.5 px-1.5 rounded-lg text-center transition text-[11px] font-bold text-slate-400 hover:bg-slate-900/60';
          } else if (t === 'kis') {
            btn.className = 'flex-1 py-1.5 px-1.5 rounded-lg text-center transition text-[11px] font-bold text-sky-400 hover:bg-slate-900/60';
          } else if (t === 'qa') {
            btn.className = 'flex-1 py-1.5 px-1.5 rounded-lg text-center transition text-[11px] font-bold text-purple-400 hover:bg-slate-900/60';
          } else if (t === 'trake') {
            btn.className = 'flex-1 py-1.5 px-1.5 rounded-lg text-center transition text-[11px] font-bold text-amber-400 hover:bg-slate-900/60';
          }
        }
      });

      renderQueryNavigator();
    }

    function filterSidebarQueries(val) {
      sidebarSearchText = (val || '').toLowerCase().trim();
      const clearBtn = document.getElementById('sidebarFilterClearBtn');
      if (clearBtn) {
        if (sidebarSearchText) {
          clearBtn.classList.remove('hidden');
        } else {
          clearBtn.classList.add('hidden');
        }
      }
      renderQueryNavigator();
    }

    function clearSidebarFilter() {
      const input = document.getElementById('sidebarFilterInput');
      if (input) input.value = '';
      filterSidebarQueries('');
    }

    function renderQueryNavigator() {
      const container = document.getElementById('queryCardsContainer');
      if (!container) return;

      const countAllEl = document.getElementById('count_all');
      const countKisEl = document.getElementById('count_kis');
      const countQaEl = document.getElementById('count_qa');
      const countTrakeEl = document.getElementById('count_trake');

      const totalAll = queryPackage.length;
      const countKis = queryPackage.filter(q => q.mode === 'kis').length;
      const countQa = queryPackage.filter(q => q.mode === 'qa').length;
      const countTrake = queryPackage.filter(q => q.mode === 'trake').length;

      if (countAllEl) countAllEl.innerText = totalAll;
      if (countKisEl) countKisEl.innerText = countKis;
      if (countQaEl) countQaEl.innerText = countQa;
      if (countTrakeEl) countTrakeEl.innerText = countTrake;

      if (!queryPackage || queryPackage.length === 0) {
        container.innerHTML = `
          <div id="querySidebarEmptyState" class="py-12 text-center text-slate-500 text-xs flex flex-col items-center gap-2">
            <span class="text-3xl">📂</span>
            <span>Chưa có gói câu hỏi nào.</span>
            <button onclick="document.getElementById('queryFileInput').click()" class="mt-1 px-3 py-1.5 bg-sky-500/20 hover:bg-sky-500/30 text-sky-300 border border-sky-500/40 rounded-lg text-xs font-semibold transition">
              Upload Queries (.zip)
            </button>
          </div>
        `;
        updateBatchProgress();
        return;
      }

      // Filter items by mode & search text
      let filtered = queryPackage;
      if (sidebarModeFilter !== 'all') {
        filtered = filtered.filter(q => q.mode === sidebarModeFilter);
      }
      if (sidebarSearchText) {
        filtered = filtered.filter(q => {
          const idMatch = (q.id || '').toLowerCase().includes(sidebarSearchText);
          const promptMatch = (q.prompt || '').toLowerCase().includes(sidebarSearchText);
          return idMatch || promptMatch;
        });
      }

      if (filtered.length === 0) {
        container.innerHTML = `
          <div class="py-8 text-center text-slate-500 text-xs flex flex-col items-center gap-1.5">
            <span>🔍 Không tìm thấy query phù hợp</span>
            <button onclick="clearSidebarFilter(); setSidebarModeFilter('all');" class="text-indigo-400 hover:underline text-[11px]">
              Xóa bộ lọc
            </button>
          </div>
        `;
        updateBatchProgress();
        return;
      }

      container.innerHTML = '';

      filtered.forEach((q) => {
        try {
          const isSelected = q.id === activeQueryId;
          const isSaved = q.status === 'saved' || (q.savedData && (
            (q.savedData.markedItems && q.savedData.markedItems.length > 0) || 
            q.savedData.qaAnswer || 
            (q.savedData.trakeEvents && q.savedData.trakeEvents.length > 0) || 
            q.savedData.pinnedFrame || 
            q.savedData.floodFrames
          ));

          let modeBadge = 'KIS';
          let modeColor = 'bg-sky-950/90 text-sky-400 border border-sky-800/40';
          let rowCountText = '100 rows';

          if (q.mode === 'qa') {
            modeBadge = 'QA';
            modeColor = 'bg-purple-950/90 text-purple-300 border border-purple-800/40';
            rowCountText = '100 rows';
          } else if (q.mode === 'trake') {
            modeBadge = 'TRAKE';
            modeColor = 'bg-amber-950/90 text-amber-300 border border-amber-800/40';
            rowCountText = '20 rows';
          }

          let savedSummary = '';
          if (isSaved && q.savedData) {
            if (q.savedData.floodFrames) {
              rowCountText = '100 rows';
              savedSummary = `⚡ 100 Neighbors (${q.savedData.floodVideoId || ''} F${q.savedData.floodCenter || 0})`;
            } else if (q.savedData.pinnedFrame) {
              rowCountText = '1 row';
              savedSummary = `🎯 Pinned: ${q.savedData.pinnedFrame.video_id} F${q.savedData.pinnedFrame.frame_idx}`;
            } else if (q.mode === 'kis') {
              const mCount = q.savedData.markedItems ? q.savedData.markedItems.length : 0;
              savedSummary = `✓ ${mCount > 0 ? `${mCount} marked + ` : ''}Top 100 frames`;
            } else if (q.mode === 'qa') {
              savedSummary = `💬 "${escapeHTML(q.savedData.qaAnswer || '')}"`;
            } else if (q.mode === 'trake') {
              const evCount = q.savedData.trakeEvents ? q.savedData.trakeEvents.length : 0;
              rowCountText = `${evCount > 0 ? evCount : 20} rows`;
              savedSummary = `⏱️ ${evCount} events`;
            }
          }

          const card = document.createElement('div');
          card.className = `p-3 rounded-xl border transition-all cursor-pointer flex flex-col gap-1.5 ${
            isSelected 
              ? 'bg-[#13162b] border-indigo-500 ring-2 ring-indigo-500/40 shadow-lg shadow-indigo-500/10' 
              : 'bg-[#0b0f19]/90 border-slate-800/80 hover:border-slate-700 hover:bg-slate-900/60'
          }`;
          card.onclick = () => selectQuery(q.id);

          const safeId = escapeHTML(q.id || 'query');
          const safePrompt = escapeHTML(q.prompt || '');

          card.innerHTML = `
            <div class="flex items-center justify-between">
              <div class="flex items-center gap-1.5 min-w-0">
                <span class="text-[10px] px-1.5 py-0.5 rounded font-mono font-bold ${modeColor}">${modeBadge}</span>
                <span class="font-mono font-bold text-xs truncate ${isSelected ? 'text-white' : 'text-slate-200'}">${safeId}</span>
              </div>
              <div class="flex items-center gap-1.5 flex-shrink-0">
                ${isSaved ? `
                  <span class="text-[10px] px-1.5 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 font-semibold flex items-center gap-0.5">
                    ✓ Saved
                  </span>
                ` : `
                  <span class="text-[10px] px-1.5 py-0.5 rounded-full bg-slate-800 text-slate-400 font-medium">
                    ○ Pending
                  </span>
                `}
                <span class="text-[10px] font-mono text-emerald-400/90 font-medium">${rowCountText}</span>
              </div>
            </div>

            <p class="text-[11px] text-slate-400 line-clamp-2 leading-relaxed font-sans mt-0.5">
              ${safePrompt || '<i class="text-slate-600">(Trống - click để tìm kiếm)</i>'}
            </p>

            ${savedSummary ? `
              <div class="text-[10px] text-emerald-400/90 font-mono bg-emerald-950/40 px-2 py-0.5 rounded border border-emerald-800/40 truncate mt-0.5">
                ${savedSummary}
              </div>
            ` : ''}
          `;
          container.appendChild(card);
        } catch (e) {
          console.error("Lỗi khi render query card:", q, e);
        }
      });

      // Sync Quick Query Dropdown Selector
      const quickSelect = document.getElementById('quickQuerySelector');
      if (quickSelect) {
        quickSelect.innerHTML = '';
        queryPackage.forEach(q => {
          const opt = document.createElement('option');
          opt.value = q.id;
          const statusIcon = q.status === 'saved' ? '✓' : '○';
          opt.innerText = `${statusIcon} [${q.mode.toUpperCase()}] ${q.id}`;
          if (q.id === activeQueryId) opt.selected = true;
          quickSelect.appendChild(opt);
        });
      }

      // Sync Quick Query Status Badge
      const quickBadge = document.getElementById('quickQueryStatusBadge');
      if (quickBadge) {
        const activeQ = queryPackage.find(q => q.id === activeQueryId);
        if (activeQ && activeQ.status === 'saved') {
          quickBadge.className = 'text-[10px] px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 font-semibold';
          quickBadge.innerText = '✓ Saved';
        } else {
          quickBadge.className = 'text-[10px] px-2 py-0.5 rounded-full bg-slate-800 text-slate-400 font-semibold';
          quickBadge.innerText = '○ Unanswered';
        }
      }

      updateBatchProgress();
    }

    function updateBatchProgress() {
      const answeredCount = queryPackage.filter(q => q.status === 'saved').length;
      const total = queryPackage.length;
      const badge = document.getElementById('batchProgressBadge');
      const label = document.getElementById('exportAllZipLabel');

      if (total > 0) {
        if (badge) {
          badge.innerText = `${answeredCount}/${total}`;
          badge.classList.remove('hidden');
        }
        if (label) label.innerText = `Export All (${answeredCount}/${total})`;
      } else {
        if (badge) badge.classList.add('hidden');
        if (label) label.innerText = 'Export All (.zip)';
      }
    }

    function selectQuery(qid, triggerSearch = true) {
      const query = queryPackage.find(q => q.id === qid);
      if (!query) return;

      activeQueryId = qid;
      const qInput = document.getElementById('queryIdInput');
      if (qInput) qInput.value = qid;
      const mainInput = document.getElementById('queryInput');
      if (mainInput) mainInput.value = query.prompt;

      const quickSelect = document.getElementById('quickQuerySelector');
      if (quickSelect) quickSelect.value = qid;

      setTaskMode(query.mode);

      // Restore saved data if exists
      if (query.savedData) {
        if (query.savedData.markedItems) {
          markedMap = new Map(query.savedData.markedItems.map(item => [`${item.video_id}_${item.frame_idx}`, item]));
        } else {
          markedMap.clear();
        }

        if (query.savedData.qaAnswer !== undefined) {
          qaAnswersMap = {};
          if (query.savedData.qaVideoId) {
            qaAnswersMap[query.savedData.qaVideoId] = query.savedData.qaAnswer;
          }
        }

        if (query.savedData.trakeEvents !== undefined) {
          trakeEventsMap = {};
          if (query.savedData.trakeVideoId) {
            trakeEventsMap[query.savedData.trakeVideoId] = query.savedData.trakeEvents;
          }
        }

        currentResults = query.savedData.results || [];
        populateVideoFilterDropdown(currentResults);
        renderGrid(currentResults);
        updateMarkedUI();
      } else {
        // Auto-search for newly selected query prompt
        if (triggerSearch && query.prompt) {
          executeSearch();
        }
      }

      renderQueryNavigator();
      saveSessionToLocalStorage();
    }

    function cycleQuery(direction) {
      if (!queryPackage || queryPackage.length === 0) return;
      const currentIdx = queryPackage.findIndex(q => q.id === activeQueryId);
      const nextIdx = (currentIdx + direction + queryPackage.length) % queryPackage.length;
      selectQuery(queryPackage[nextIdx].id, false);
    }

    function clearQueryPackageSession() {
      if (confirm("Bạn có chắc chắn muốn xóa toàn bộ gói câu hỏi đã tải lên không?")) {
        queryPackage = [];
        renderQueryNavigator();
        saveSessionToLocalStorage();
      }
    }

    function saveActiveQueryResult() {
      const qid = (document.getElementById('queryIdInput').value || activeQueryId || 'query-1').trim();
      let query = queryPackage.find(q => q.id === qid);

      if (!query) {
        query = {
          id: qid,
          filename: `${qid}.txt`,
          prompt: document.getElementById('queryInput').value.trim(),
          mode: currentTaskMode,
          status: 'unanswered',
          savedData: null
        };
        queryPackage.push(query);
      }

      const marked_items = Array.from(markedMap.values());
      const firstVid = marked_items.length > 0 ? marked_items[0].video_id : (currentResults[0] ? currentResults[0].video_id : '');
      const qaAns = firstVid ? (qaAnswersMap[firstVid] || '') : (Object.values(qaAnswersMap)[0] || '');
      const trakeEvs = firstVid ? (trakeEventsMap[firstVid] || []) : (Object.values(trakeEventsMap)[0] || []);

      if (!query.savedData) query.savedData = {};
      query.savedData.mode = currentTaskMode;
      query.savedData.markedItems = marked_items;
      query.savedData.qaAnswer = qaAns;
      query.savedData.qaVideoId = firstVid;
      query.savedData.trakeEvents = trakeEvs;
      query.savedData.trakeVideoId = firstVid;
      query.savedData.results = currentResults;
      query.status = 'saved';

      renderQueryNavigator();
      saveSessionToLocalStorage();

      let desc = '';
      if (query.savedData.floodFrames) {
        desc = `100 neighbor frames around Frame ${query.savedData.floodCenter}`;
      } else if (query.savedData.pinnedFrame) {
        desc = `pinned Frame ${query.savedData.pinnedFrame.frame_idx} (${query.savedData.pinnedFrame.video_id})`;
      } else if (currentTaskMode === 'kis') {
        desc = `${marked_items.length} marked keyframes + top 100 results`;
      } else if (currentTaskMode === 'qa') {
        desc = `answer "${qaAns}"`;
      } else if (currentTaskMode === 'trake') {
        desc = `${trakeEvs.length} event marks`;
      }
      showQuickToast(`✅ Đã lưu kết quả cho [${qid}] (${desc})!`);
    }

    async function exportAllQueriesZip() {
      if (!queryPackage || queryPackage.length === 0) {
        const qid = (document.getElementById('queryIdInput').value || 'query-1').trim();
        saveActiveQueryResult();
      }

      const zip = new JSZip();
      const subFolder = zip.folder("submission");

      queryPackage.forEach(q => {
        const lines = buildCSVLinesForQuery(q);
        const csvContent = lines.join(String.fromCharCode(10)) + String.fromCharCode(10);
        subFolder.file(`${q.id}.csv`, csvContent);
      });

      const blob = await zip.generateAsync({ type: "blob" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `submission_aic2026_${Date.now()}.zip`;
      a.click();
      URL.revokeObjectURL(url);
    }

    function setTaskMode(mode) {
      currentTaskMode = mode;
      const btnKis = document.getElementById('modeBtn_kis');
      const btnQa = document.getElementById('modeBtn_qa');
      const btnTrake = document.getElementById('modeBtn_trake');
      const desc = document.getElementById('modeDescriptionText');
      const saveLabel = document.getElementById('saveQueryResultLabel');

      [btnKis, btnQa, btnTrake].forEach(b => {
        if (b) b.className = 'px-4 py-2 rounded-lg font-bold text-xs transition flex items-center gap-2 text-slate-400 hover:text-slate-200';
      });

      if (mode === 'kis') {
        if (btnKis) btnKis.className = 'px-4 py-2 rounded-lg font-bold text-xs transition flex items-center gap-2 bg-emerald-500 text-slate-950 shadow-md shadow-emerald-500/20';
        if (desc) desc.innerHTML = `<span class="text-emerald-400 font-semibold">Mode KIS:</span> Tìm kiếm & xuất file <code class="bg-slate-900 px-1.5 py-0.5 rounded text-emerald-300 font-mono text-[11px]">&lt;query_id&gt;-kis.csv</code> (video_id,frame_idx)`;
        if (saveLabel) saveLabel.innerText = 'Save KIS (Top 100)';
      } else if (mode === 'qa') {
        if (btnQa) btnQa.className = 'px-4 py-2 rounded-lg font-bold text-xs transition flex items-center gap-2 bg-amber-400 text-slate-950 shadow-md shadow-amber-400/20';
        if (desc) desc.innerHTML = `<span class="text-amber-400 font-semibold">Mode Q&A:</span> Chọn video, mở VLM hỏi đáp & nhập câu trả lời, xuất file <code class="bg-slate-900 px-1.5 py-0.5 rounded text-amber-300 font-mono text-[11px]">&lt;query_id&gt;-qa.csv</code> (video_id,frame_idx,answer)`;
        if (saveLabel) saveLabel.innerText = 'Save Q&A Result';
      } else if (mode === 'trake') {
        if (btnTrake) btnTrake.className = 'px-4 py-2 rounded-lg font-bold text-xs transition flex items-center gap-2 bg-indigo-500 text-white shadow-md shadow-indigo-500/20';
        if (desc) desc.innerHTML = `<span class="text-indigo-400 font-semibold">Mode TRAKE:</span> Đánh dấu chuỗi mốc sự kiện theo thời gian, xuất file <code class="bg-slate-900 px-1.5 py-0.5 rounded text-indigo-300 font-mono text-[11px]">&lt;query_id&gt;-trake.csv</code> (video_id,f1,f2,f3,f4...)`;
        if (saveLabel) saveLabel.innerText = 'Save TRAKE Events';
      }

      updateExportButtonLabel();
      renderGrid(currentResults);
      syncModalModeUI();
    }

    function updateExportButtonLabel() {
      const qid = (document.getElementById('queryIdInput').value || 'query-1').trim();
      const label = document.getElementById('exportBtnLabel');
      if (!label) return;
      if (currentTaskMode === 'kis') {
        label.innerText = `Export ${qid}-kis.csv`;
      } else if (currentTaskMode === 'qa') {
        label.innerText = `Export ${qid}-qa.csv`;
      } else if (currentTaskMode === 'trake') {
        label.innerText = `Export ${qid}-trake.csv`;
      }
    }

    function syncModalModeUI() {
      const qaBox = document.getElementById('modalQABox');
      const trakeBox = document.getElementById('modalTrakeBox');
      const saveTrakeFooter = document.getElementById('modalSaveTrakeFooterBtn');
      const saveQAFooter = document.getElementById('modalSaveQAFooterBtn');

      if (qaBox && trakeBox) {
        if (currentTaskMode === 'qa') {
          qaBox.classList.remove('hidden');
          trakeBox.classList.add('hidden');
        } else if (currentTaskMode === 'trake') {
          qaBox.classList.add('hidden');
          trakeBox.classList.remove('hidden');
        } else {
          qaBox.classList.remove('hidden');
          trakeBox.classList.remove('hidden');
        }
      }

      if (saveTrakeFooter) {
        if (currentTaskMode === 'trake') saveTrakeFooter.classList.remove('hidden');
        else saveTrakeFooter.classList.add('hidden');
      }

      if (saveQAFooter) {
        if (currentTaskMode === 'qa') saveQAFooter.classList.remove('hidden');
        else saveQAFooter.classList.add('hidden');
      }
    }

    function onModalQAInputChanged() {
      const input = document.getElementById('modalQAInput');
      const val = input ? input.value : '';
      const counter = document.getElementById('modalQACharCount');
      if (counter) counter.innerText = `${val.length}/100 chars`;
    }

    function saveModalQAAnswer(closeAfter = false) {
      if (!currentModalItem) return;
      const vid = currentModalItem.video_id;
      const input = document.getElementById('modalQAInput');
      const val = (input ? input.value : '').trim().slice(0, 100);
      qaAnswersMap[vid] = val;

      const fps = currentModalItem.fps || 25.0;
      const currentSec = videoElem ? videoElem.currentTime : currentModalItem.pts_time;
      const currentFrame = Math.round(currentSec * fps);

      markedMap.set(`${vid}_${currentFrame}`, {
        video_id: vid,
        frame_idx: currentFrame,
        pts_time: currentSec
      });
      updateMarkedUI();

      if (currentActiveChatVideo && currentActiveChatVideo.video_id === vid) {
        const dInput = document.getElementById('aiQAAnswerInput');
        if (dInput) dInput.value = val;
        const dCount = document.getElementById('aiQACharCount');
        if (dCount) dCount.innerText = `${val.length}/100 chars`;
      }

      saveActiveQueryResult();

      if (closeAfter) {
        closeModal();
      } else {
        renderGrid(currentResults);
        alert(`✅ Đã lưu câu trả lời cho ${vid}: "${val}"`);
      }
    }

    function saveModalTrakeEvents(closeAfter = false) {
      if (!currentModalItem) return;
      const vid = currentModalItem.video_id;
      const fps = currentModalItem.fps || 25.0;
      const currentSec = videoElem ? videoElem.currentTime : currentModalItem.pts_time;
      const currentFrame = Math.round(currentSec * fps);

      if (!trakeEventsMap[vid] || trakeEventsMap[vid].length === 0) {
        trakeEventsMap[vid] = [{ frame_idx: currentFrame, pts_time: currentSec }];
      }

      markedMap.set(`${vid}_${currentFrame}`, {
        video_id: vid,
        frame_idx: currentFrame,
        pts_time: currentSec
      });
      updateMarkedUI();

      saveActiveQueryResult();

      if (closeAfter) {
        closeModal();
      } else {
        renderTrakeEventsModal(vid);
        renderGrid(currentResults);
        alert(`✅ Đã lưu chuỗi ${trakeEventsMap[vid].length} sự kiện TRAKE cho ${vid}!`);
      }
    }

    function saveTrakeForCard(vid, fid, pts, event) {
      if (event) event.stopPropagation();
      if (!trakeEventsMap[vid] || trakeEventsMap[vid].length === 0) {
        trakeEventsMap[vid] = [{ frame_idx: fid, pts_time: pts }];
      }
      markedMap.set(`${vid}_${fid}`, { video_id: vid, frame_idx: fid, pts_time: pts });
      updateMarkedUI();
      saveActiveQueryResult();
      renderGrid(currentResults);
      alert(`✅ Đã lưu video ${vid} (F${fid}) làm đáp án TRAKE!`);
    }

    function saveQAPromptForCard(vid, fid, pts, event) {
      if (event) event.stopPropagation();
      const currentAns = qaAnswersMap[vid] || '';
      const promptAns = prompt(`Nhập câu trả lời Q&A cho video ${vid}:`, currentAns);
      if (promptAns !== null) {
        qaAnswersMap[vid] = promptAns.trim().slice(0, 100);
        markedMap.set(`${vid}_${fid}`, { video_id: vid, frame_idx: fid, pts_time: pts });
        updateMarkedUI();
        saveActiveQueryResult();
        renderGrid(currentResults);
        alert(`✅ Đã lưu câu trả lời cho ${vid}: "${qaAnswersMap[vid]}"`);
      }
    }

    function onDrawerQAInputChanged() {
      const input = document.getElementById('aiQAAnswerInput');
      const val = input ? input.value : '';
      const counter = document.getElementById('aiQACharCount');
      if (counter) counter.innerText = `${val.length}/100 chars`;
    }

    function saveDrawerQAAnswer() {
      if (!currentActiveChatVideo) {
        alert("Chưa chọn video để gán câu trả lời.");
        return;
      }
      const vid = currentActiveChatVideo.video_id;
      const input = document.getElementById('aiQAAnswerInput');
      const val = (input ? input.value : '').trim().slice(0, 100);
      qaAnswersMap[vid] = val;

      markedMap.set(`${vid}_${currentActiveChatVideo.frame_idx}`, {
        video_id: vid,
        frame_idx: currentActiveChatVideo.frame_idx,
        pts_time: currentActiveChatVideo.pts_time
      });
      updateMarkedUI();

      if (currentModalItem && currentModalItem.video_id === vid) {
        const mInput = document.getElementById('modalQAInput');
        if (mInput) mInput.value = val;
        const mCount = document.getElementById('modalQACharCount');
        if (mCount) mCount.innerText = `${val.length}/100 chars`;
      }

      saveActiveQueryResult();
      renderGrid(currentResults);
      alert(`✅ Đã lưu câu trả lời cho ${vid}: "${val}"`);
    }

    function useAIResponseAsQA(text) {
      if (!currentActiveChatVideo) return;
      const vid = currentActiveChatVideo.video_id;
      const clean = text.replace(/^["'\s]+|["'\s]+$/g, '').slice(0, 100);
      qaAnswersMap[vid] = clean;

      markedMap.set(`${vid}_${currentActiveChatVideo.frame_idx}`, {
        video_id: vid,
        frame_idx: currentActiveChatVideo.frame_idx,
        pts_time: currentActiveChatVideo.pts_time
      });
      updateMarkedUI();

      const dInput = document.getElementById('aiQAAnswerInput');
      if (dInput) dInput.value = clean;
      const dCount = document.getElementById('aiQACharCount');
      if (dCount) dCount.innerText = `${clean.length}/100 chars`;

      if (currentModalItem && currentModalItem.video_id === vid) {
        const mInput = document.getElementById('modalQAInput');
        if (mInput) mInput.value = clean;
        const mCount = document.getElementById('modalQACharCount');
        if (mCount) mCount.innerText = `${clean.length}/100 chars`;
      }

      saveActiveQueryResult();
      renderGrid(currentResults);
      alert(`✅ Đã chọn câu trả lời từ AI cho ${vid}: "${clean}"`);
    }

    function addCurrentFrameToTrake() {
      if (!currentModalItem || !videoElem) return;
      const vid = currentModalItem.video_id;
      const fps = currentModalItem.fps || 25.0;
      const pts = Math.max(0, videoElem.currentTime);
      const fid = Math.round(pts * fps);

      if (!trakeEventsMap[vid]) {
        trakeEventsMap[vid] = [];
      }

      if (trakeEventsMap[vid].some(e => e.frame_idx === fid)) {
        alert(`Frame ${fid} đã có trong chuỗi sự kiện của video ${vid}.`);
        return;
      }

      trakeEventsMap[vid].push({ frame_idx: fid, pts_time: pts });
      trakeEventsMap[vid].sort((a, b) => a.frame_idx - b.frame_idx);

      markedMap.set(`${vid}_${fid}`, {
        video_id: vid,
        frame_idx: fid,
        pts_time: pts
      });
      updateMarkedUI();
      saveActiveQueryResult();

      renderTrakeEventsModal(vid);
      renderGrid(currentResults);
    }

    function removeTrakeEvent(vid, fid) {
      if (!trakeEventsMap[vid]) return;
      trakeEventsMap[vid] = trakeEventsMap[vid].filter(e => e.frame_idx !== fid);
      saveActiveQueryResult();
      renderTrakeEventsModal(vid);
      renderGrid(currentResults);
    }

    function clearAllTrakeEvents() {
      if (!currentModalItem) return;
      const vid = currentModalItem.video_id;
      trakeEventsMap[vid] = [];
      renderTrakeEventsModal(vid);
      renderGrid(currentResults);
    }

    function renderTrakeEventsModal(vid) {
      const container = document.getElementById('modalTrakeEventsList');
      const badge = document.getElementById('modalTrakeCountBadge');
      if (!container || !badge) return;
      const events = trakeEventsMap[vid] || [];

      badge.innerText = `${events.length} events`;

      if (events.length === 0) {
        container.innerHTML = `<span class="text-slate-500 text-[11px] italic">Chưa có mốc sự kiện nào. Hãy tua video và bấm "+ Mark Current Frame".</span>`;
        return;
      }

      container.innerHTML = '';
      events.forEach((ev, idx) => {
        const chip = document.createElement('div');
        chip.className = 'flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-indigo-950/80 border border-indigo-500/50 text-indigo-200 text-xs shadow-sm';
        chip.innerHTML = `
          <span class="font-bold text-indigo-400">#${idx + 1}</span>
          <span class="font-mono font-bold">F${ev.frame_idx}</span>
          <span class="text-[10px] text-slate-400 font-mono">(${formatTime(ev.pts_time)})</span>
          <button onclick="removeTrakeEvent('${vid}', ${ev.frame_idx})" class="ml-1 text-slate-400 hover:text-rose-400 font-bold" title="Xóa mốc sự kiện này">✕</button>
        `;
        container.appendChild(chip);
      });
    }

    function syncModelBadges() {
      const elH = document.getElementById('headerModelBadge');
      const elF = document.getElementById('fabModelText');
      const elD = document.getElementById('drawerModelPill');
      if (elH) elH.innerText = aiSettings.model;
      if (elF) elF.innerText = aiSettings.model;
      if (elD) elD.innerText = aiSettings.model;
    }

    // Load AI Settings on Startup
    function loadChatSettings() {
      try {
        const p = localStorage.getItem('ai_provider');
        const m = localStorage.getItem('ai_model');
        const k = localStorage.getItem('ai_api_key');
        const u = localStorage.getItem('ai_custom_url');
        const a = localStorage.getItem('ai_auto_open');

        if (p) aiSettings.provider = p;
        if (m) aiSettings.model = m;
        if (k) aiSettings.apiKey = k;
        if (u) aiSettings.customUrl = u;
        if (a !== null) aiSettings.autoOpen = (a === 'true');

        document.getElementById('aiProviderSelect').value = aiSettings.provider;
        document.getElementById('aiModelInput').value = aiSettings.model;
        document.getElementById('aiApiKeyInput').value = aiSettings.apiKey;
        document.getElementById('aiCustomUrlInput').value = aiSettings.customUrl;
        document.getElementById('aiAutoOpenCheckbox').checked = aiSettings.autoOpen;
        onProviderChange();
        syncModelBadges();
      } catch (e) {}
    }

    function saveChatSettings() {
      aiSettings.provider = document.getElementById('aiProviderSelect').value;
      const defaultModel = aiSettings.provider === 'openai' ? 'gpt-4o-mini' : (aiSettings.provider === 'openrouter' ? 'minimax/minimax-m3' : 'antigravity/gemini-3.6-flash-medium');
      aiSettings.model = document.getElementById('aiModelInput').value.trim() || defaultModel;
      aiSettings.apiKey = document.getElementById('aiApiKeyInput').value.trim() || (aiSettings.provider === 'omniroute' ? 'omniroute' : '');
      aiSettings.customUrl = document.getElementById('aiCustomUrlInput').value.trim();
      aiSettings.autoOpen = document.getElementById('aiAutoOpenCheckbox').checked;

      try {
        localStorage.setItem('ai_provider', aiSettings.provider);
        localStorage.setItem('ai_model', aiSettings.model);
        localStorage.setItem('ai_api_key', aiSettings.apiKey);
        localStorage.setItem('ai_custom_url', aiSettings.customUrl);
        localStorage.setItem('ai_auto_open', aiSettings.autoOpen.toString());
      } catch (e) {}

      syncModelBadges();
      document.getElementById('aiSettingsOverlay').classList.add('hidden');
    }

    function toggleChatSettings() {
      const overlay = document.getElementById('aiSettingsOverlay');
      overlay.classList.toggle('hidden');
    }

    function onProviderChange() {
      const provider = document.getElementById('aiProviderSelect').value;
      const customGroup = document.getElementById('aiCustomUrlGroup');
      const modelInput = document.getElementById('aiModelInput');
      const keyInput = document.getElementById('aiApiKeyInput');
      
      if (provider === 'custom') {
        customGroup.classList.remove('hidden');
      } else {
        customGroup.classList.add('hidden');
      }

      if (provider === 'omniroute') {
        if (!modelInput.value || modelInput.value.includes('minimax') || modelInput.value.includes('gpt')) {
          modelInput.value = 'antigravity/gemini-3.6-flash-medium';
        }
        if (!keyInput.value) keyInput.value = 'omniroute';
      } else if (provider === 'openai') {
        if (!modelInput.value.includes('gpt')) modelInput.value = 'gpt-4o-mini';
      } else if (provider === 'openrouter') {
        if (!modelInput.value.includes('minimax')) modelInput.value = 'minimax/minimax-m3';
      }
    }

    function setModalClipStart() {
      if (!videoElem) return;
      modalClipStartSec = Math.max(0, parseFloat(videoElem.currentTime.toFixed(2)));
      document.getElementById('modalClipStartBadge').innerText = formatTime(modalClipStartSec);
      if (currentModalItem) {
        const vid = currentModalItem.video_id;
        if (!videoClipRanges[vid]) videoClipRanges[vid] = {};
        videoClipRanges[vid].start_sec = modalClipStartSec;
        if (currentActiveChatVideo && currentActiveChatVideo.video_id === vid) {
          syncClipRangeToUI(vid);
        }
      }
    }

    function setModalClipEnd() {
      if (!videoElem) return;
      modalClipEndSec = Math.max(0, parseFloat(videoElem.currentTime.toFixed(2)));
      document.getElementById('modalClipEndBadge').innerText = formatTime(modalClipEndSec);
      if (currentModalItem) {
        const vid = currentModalItem.video_id;
        if (!videoClipRanges[vid]) videoClipRanges[vid] = {};
        videoClipRanges[vid].end_sec = modalClipEndSec;
        if (currentActiveChatVideo && currentActiveChatVideo.video_id === vid) {
          syncClipRangeToUI(vid);
        }
      }
    }

    function askVlmFromModalClip() {
      if (!currentModalItem) return;
      const vid = currentModalItem.video_id;
      const pts = videoElem.currentTime;
      const fps = currentModalItem.fps || 25.0;
      const fid = Math.round(pts * fps);

      if (!videoClipRanges[vid]) videoClipRanges[vid] = {};
      if (modalClipStartSec !== null) videoClipRanges[vid].start_sec = modalClipStartSec;
      if (modalClipEndSec !== null) videoClipRanges[vid].end_sec = modalClipEndSec;

      if (videoClipRanges[vid].start_sec === undefined || videoClipRanges[vid].end_sec === undefined) {
        videoClipRanges[vid].start_sec = Math.max(0, parseFloat((pts - 5.0).toFixed(2)));
        videoClipRanges[vid].end_sec = parseFloat((pts + 5.0).toFixed(2));
      }

      openChatbotForVideo(vid, fid, pts, true);
    }

    function setClipStartFromPlayer() {
      const cur = videoElem && !isNaN(videoElem.currentTime) ? videoElem.currentTime : 0;
      document.getElementById('aiClipStartInput').value = cur.toFixed(2);
      onClipRangeChanged();
    }

    function setClipEndFromPlayer() {
      const cur = videoElem && !isNaN(videoElem.currentTime) ? videoElem.currentTime : 0;
      document.getElementById('aiClipEndInput').value = cur.toFixed(2);
      onClipRangeChanged();
    }

    function autoFillClipWindow() {
      if (!currentActiveChatVideo) return;
      const pts = currentActiveChatVideo.pts_time || 0;
      const st = Math.max(0, parseFloat((pts - 5.0).toFixed(2)));
      const et = parseFloat((pts + 5.0).toFixed(2));
      document.getElementById('aiClipStartInput').value = st;
      document.getElementById('aiClipEndInput').value = et;
      onClipRangeChanged();
    }

    function onClipRangeChanged() {
      if (!currentActiveChatVideo) return;
      const vid = currentActiveChatVideo.video_id;
      const stVal = document.getElementById('aiClipStartInput').value.trim();
      const etVal = document.getElementById('aiClipEndInput').value.trim();
      const badge = document.getElementById('aiClipDurationBadge');

      if (stVal === '' || etVal === '') {
        badge.className = 'font-mono text-[10px] px-2 py-0.5 rounded bg-slate-950 text-amber-400 border border-slate-800 font-medium';
        badge.innerText = 'Chưa đánh dấu';
        return;
      }

      const st = parseFloat(stVal);
      const et = parseFloat(etVal);

      if (isNaN(st) || isNaN(et)) {
        badge.className = 'font-mono text-[10px] px-2 py-0.5 rounded bg-slate-950 text-amber-400 border border-slate-800 font-medium';
        badge.innerText = 'Chưa hoàn tất';
        return;
      }

      if (st < 0 || st >= et) {
        badge.className = 'font-mono text-[10px] px-2 py-0.5 rounded bg-rose-950/80 text-rose-300 border border-rose-800 font-bold';
        badge.innerText = `Lỗi: Start ≥ End`;
        return;
      }

      const dur = et - st;
      badge.className = 'font-mono text-[10px] px-2 py-0.5 rounded bg-emerald-950/80 text-emerald-300 border border-emerald-700 font-bold';
      badge.innerText = `⏱️ ${dur.toFixed(1)}s (${st.toFixed(1)}s → ${et.toFixed(1)}s)`;

      if (!videoClipRanges[vid]) videoClipRanges[vid] = {};
      videoClipRanges[vid].start_sec = st;
      videoClipRanges[vid].end_sec = et;
    }

    function syncClipRangeToUI(vid) {
      const range = videoClipRanges[vid];
      const stInput = document.getElementById('aiClipStartInput');
      const etInput = document.getElementById('aiClipEndInput');
      if (range && range.start_sec !== undefined && range.end_sec !== undefined) {
        stInput.value = range.start_sec;
        etInput.value = range.end_sec;
      } else {
        stInput.value = '';
        etInput.value = '';
      }
      onClipRangeChanged();
    }

    function toggleChatbot() {
      const drawer = document.getElementById('aiChatDrawer');
      const isHidden = drawer.classList.contains('hidden');
      if (isHidden) {
        drawer.classList.remove('hidden');
        if (!currentActiveChatVideo && currentResults.length > 0) {
          const first = currentResults[0];
          openChatbotForVideo(first.video_id, first.frame_idx, first.pts_time, true);
        }
      } else {
        drawer.classList.add('hidden');
      }
    }

    function openChatbotForVideo(vid, fid, pts, forceOpen = false) {
      if (!vid) return;

      currentActiveChatVideo = {
        video_id: vid,
        frame_idx: fid,
        pts_time: pts
      };

      // Sync clip range
      if (!videoClipRanges[vid]) {
        videoClipRanges[vid] = {
          start_sec: Math.max(0, parseFloat((pts - 5.0).toFixed(2))),
          end_sec: parseFloat((pts + 5.0).toFixed(2))
        };
      }
      syncClipRangeToUI(vid);

      // Update header
      document.getElementById('aiChatActiveVideoLabel').innerText = `${vid} • Frame ${fid} (${formatTime(pts)})`;

      // Update video selector
      updateVideoSelectorDropdown();

      // Fetch Video Context Stats
      fetch(`/api/video_context?video_id=${vid}&pts_time=${pts}`)
        .then(res => res.json())
        .then(data => {
          const asrCount = data.asr_segments ? data.asr_segments.length : 0;
          const ocrCount = data.ocr_lines ? data.ocr_lines.length : 0;
          document.getElementById('aiContextStats').innerText = `🎙️ ${asrCount} ASR • 🔤 ${ocrCount} OCR`;
        })
        .catch(() => {
          document.getElementById('aiContextStats').innerText = `Context Ready`;
        });

      // Sync Q&A Answer Input
      const curAns = qaAnswersMap[vid] || '';
      const dInput = document.getElementById('aiQAAnswerInput');
      if (dInput) dInput.value = curAns;
      const dCount = document.getElementById('aiQACharCount');
      if (dCount) dCount.innerText = `${curAns.length}/100 chars`;

      // Render Messages
      renderChatMessages();

      // Auto-open or force open
      if (forceOpen || aiSettings.autoOpen) {
        const drawer = document.getElementById('aiChatDrawer');
        drawer.classList.remove('hidden');
      }
    }

    function openChatbotForModalCurrentFrame() {
      if (!currentModalItem) return;
      const fps = currentModalItem.fps || 25.0;
      const currentSec = videoElem.currentTime;
      const currentFrame = Math.round(currentSec * fps);
      openChatbotForVideo(currentModalItem.video_id, currentFrame, currentSec, true);
    }

    function updateVideoSelectorDropdown() {
      const sel = document.getElementById('aiVideoSelect');
      sel.innerHTML = '';

      const optionsMap = new Map();
      if (currentActiveChatVideo) {
        optionsMap.set(currentActiveChatVideo.video_id, `${currentActiveChatVideo.video_id} (F${currentActiveChatVideo.frame_idx})`);
      }
      markedMap.forEach((v, k) => {
        optionsMap.set(v.video_id, `${v.video_id} (F${v.frame_idx}) ✓`);
      });

      if (optionsMap.size === 0) {
        sel.innerHTML = '<option value="">(Chưa chọn video)</option>';
        return;
      }

      optionsMap.forEach((label, vid) => {
        const opt = document.createElement('option');
        opt.value = vid;
        opt.innerText = label;
        if (currentActiveChatVideo && currentActiveChatVideo.video_id === vid) {
          opt.selected = true;
        }
        sel.appendChild(opt);
      });
    }

    function onActiveVideoChange(vid) {
      if (!vid) return;
      let targetItem = null;
      markedMap.forEach((v) => {
        if (v.video_id === vid) targetItem = v;
      });
      if (!targetItem) {
        targetItem = currentResults.find(r => r.video_id === vid);
      }
      if (targetItem) {
        openChatbotForVideo(targetItem.video_id, targetItem.frame_idx, targetItem.pts_time, true);
      }
    }

    function renderChatMessages() {
      const container = document.getElementById('aiChatMessages');
      container.innerHTML = '';

      if (!currentActiveChatVideo) {
        container.innerHTML = `
          <div class="self-start max-w-[90%] p-3 rounded-2xl bg-slate-900 border border-slate-800 text-slate-300 leading-relaxed">
            👋 Hãy chọn hoặc đánh dấu một video chính xác để bắt đầu phân tích!
          </div>
        `;
        return;
      }

      const vid = currentActiveChatVideo.video_id;
      const history = videoChatHistories[vid] || [];

      // Initial context banner bubble
      const banner = document.createElement('div');
      banner.className = 'self-start max-w-[95%] p-3 rounded-2xl bg-slate-900/90 border border-emerald-500/20 text-slate-300 leading-relaxed shadow-sm';
      banner.innerHTML = `
        <div class="flex items-center gap-1.5 text-emerald-400 font-semibold mb-1">
          <span>🎬 Đang phân tích: <b>${vid}</b></span>
          <span class="text-[10px] text-slate-400 font-mono">(${currentActiveChatVideo.frame_idx}f • ${currentActiveChatVideo.pts_time}s)</span>
        </div>
        <p class="text-[11px] text-slate-400">VLM sẽ nhận chuỗi video frames của đoạn clip được đánh dấu cùng lời thoại ASR & OCR để trả lời chính xác nhất!</p>
      `;
      container.appendChild(banner);

      // Render conversation
      history.forEach((msg, idx) => {
        const isUser = msg.role === 'user';
        const bubble = document.createElement('div');
        bubble.className = isUser
          ? 'self-end max-w-[85%] p-3 rounded-2xl bg-emerald-600 text-white leading-relaxed shadow-md'
          : 'self-start max-w-[92%] p-3 rounded-2xl bg-slate-900 border border-slate-800 text-slate-200 leading-relaxed shadow-md';

        if (isUser) {
          bubble.innerText = msg.content;
        } else {
          bubble.innerHTML = `
            <div class="flex items-center justify-between gap-2 mb-1.5 pb-1 border-b border-slate-800/80">
              <span class="font-bold text-[10px] text-emerald-400 flex items-center gap-1">🤖 VLM Video Analyst</span>
              <div class="flex items-center gap-2">
                <button onclick="useAIResponseAsQA(${JSON.stringify(msg.content)})" class="text-[10px] px-2 py-0.5 rounded bg-amber-500/20 text-amber-300 hover:bg-amber-500/30 border border-amber-500/40 font-semibold transition" title="Đặt câu trả lời này làm đáp án Q&A">✍️ Chọn làm Answer</button>
                <button onclick="copyText(this, ${JSON.stringify(msg.content)})" class="text-[10px] text-slate-400 hover:text-white transition">📋 Copy</button>
              </div>
            </div>
            <div class="prose-content leading-relaxed">${formatMarkdown(msg.content)}</div>
          `;
        }
        container.appendChild(bubble);
      });

      container.scrollTop = container.scrollHeight;
    }

    function formatMarkdown(text) {
      if (!text) return '';
      let safe = text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');

      // Code blocks
      safe = safe.replace(/```([\s\S]*?)```/g, '<pre class="bg-slate-950 p-2.5 rounded-lg my-1.5 font-mono text-[11px] text-emerald-300 overflow-x-auto border border-slate-800"><code>$1</code></pre>');
      // Inline code
      safe = safe.replace(/`([^`]+)`/g, '<code class="bg-slate-950 text-emerald-300 px-1 py-0.5 rounded font-mono text-[11px] border border-slate-800">$1</code>');
      // Blockquotes
      safe = safe.replace(/^>\s+(.*)$/gm, '<blockquote class="border-l-2 border-emerald-500/60 pl-2 text-slate-400 my-1 text-[11px]">$1</blockquote>');
      // Bold
      safe = safe.replace(/\*\*([^*]+)\*\*/g, '<b class="text-white font-semibold">$1</b>');
      // Bullet lists
      safe = safe.replace(/^[•\-\*]\s+(.*)$/gm, '<li class="ml-4 list-disc text-slate-200 my-0.5">$1</li>');
      // Numbered lists
      safe = safe.replace(/^\d+\.\s+(.*)$/gm, '<li class="ml-4 list-decimal text-slate-200 my-0.5">$1</li>');
      // Linebreaks
      safe = safe.replace(/\n/g, '<br/>');
      return safe;
    }

    function copyText(btn, text) {
      navigator.clipboard.writeText(text);
      const orig = btn.innerText;
      btn.innerText = '✅ Copied!';
      setTimeout(() => btn.innerText = orig, 1500);
    }

    function clearChatHistory() {
      if (currentActiveChatVideo) {
        videoChatHistories[currentActiveChatVideo.video_id] = [];
        renderChatMessages();
      }
    }

    function handleChatInputKey(e) {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendChatMessage();
      }
    }

    function sendQuickPrompt(promptText) {
      sendChatMessage(promptText);
    }

    async function sendChatMessage(customText = null) {
      if (isChatSending) return;
      if (!currentActiveChatVideo) {
        alert("Vui lòng chọn hoặc đánh dấu một video trước khi đặt câu hỏi.");
        return;
      }

      const stVal = document.getElementById('aiClipStartInput').value.trim();
      const etVal = document.getElementById('aiClipEndInput').value.trim();

      if (stVal === '' || etVal === '') {
        alert("⚠️ Vui lòng đánh dấu Thời gian Bắt đầu (Start) và Kết thúc (End) của đoạn clip cần phân tích trước khi gửi câu hỏi cho VLM.");
        document.getElementById('aiClipStartInput').focus();
        return;
      }

      const startSec = parseFloat(stVal);
      const endSec = parseFloat(etVal);

      if (isNaN(startSec) || isNaN(endSec) || startSec < 0 || endSec <= startSec) {
        alert(`⚠️ Khoảng thời gian clip không hợp lệ: Start (${stVal}s) phải nhỏ hơn End (${etVal}s).`);
        return;
      }

      const input = document.getElementById('aiChatInput');
      const text = (customText !== null ? customText : input.value).trim();
      if (!text) return;

      if (customText === null) {
        input.value = '';
      }

      const vid = currentActiveChatVideo.video_id;
      if (!videoChatHistories[vid]) {
        videoChatHistories[vid] = [];
      }

      // Add user message
      videoChatHistories[vid].push({ role: 'user', content: text });
      renderChatMessages();

      // Show loading placeholder in chat
      const container = document.getElementById('aiChatMessages');
      const loadingBubble = document.createElement('div');
      loadingBubble.id = 'aiChatLoadingBubble';
      loadingBubble.className = 'self-start max-w-[85%] p-3 rounded-2xl bg-slate-900 border border-slate-800 text-slate-400 text-xs flex items-center gap-2 animate-pulse';
      loadingBubble.innerHTML = `
        <span class="w-2 h-2 rounded-full bg-emerald-400 animate-ping"></span>
        <span>🎞️ Đang trích xuất frames từ clip [${startSec.toFixed(1)}s - ${endSec.toFixed(1)}s] & gửi VLM...</span>
      `;
      container.appendChild(loadingBubble);
      container.scrollTop = container.scrollHeight;

      isChatSending = true;
      document.getElementById('aiSendBtn').disabled = true;

      const currentQuery = document.getElementById('queryInput').value.trim();

      try {
        const payload = {
          video_id: vid,
          frame_idx: currentActiveChatVideo.frame_idx,
          pts_time: currentActiveChatVideo.pts_time,
          start_sec: startSec,
          end_sec: endSec,
          query: currentQuery,
          messages: videoChatHistories[vid],
          provider: aiSettings.provider,
          model: aiSettings.model,
          api_key: aiSettings.apiKey,
          custom_url: aiSettings.customUrl
        };

        const res = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });

        const data = await res.json();
        const placeholder = document.getElementById('aiChatLoadingBubble');
        if (placeholder) placeholder.remove();

        if (data.error) {
          videoChatHistories[vid].push({
            role: 'assistant',
            content: `⚠️ **Lỗi:** ${data.error}\n\n*Gợi ý: Bấm vào biểu tượng ⚙️ (Cài đặt) ở góc trên Chatbot để cấu hình API Key hoặc Provider.*`
          });
        } else {
          const metaBadge = `\n\n> 🎞️ *Clip [${data.start_sec}s - ${data.end_sec}s] (${data.clip_duration}s • ${data.frames_sent} frames • ${data.asr_count} ASR • ${data.ocr_count} OCR)*`;
          videoChatHistories[vid].push({
            role: 'assistant',
            content: (data.reply || '(Không có nội dung trả lời từ mô hình)') + metaBadge
          });
        }
      } catch (err) {
        const placeholder = document.getElementById('aiChatLoadingBubble');
        if (placeholder) placeholder.remove();
        videoChatHistories[vid].push({
          role: 'assistant',
          content: `❌ **Không thể kết nối đến API:** ${err.message}`
        });
      } finally {
        isChatSending = false;
        document.getElementById('aiSendBtn').disabled = false;
        renderChatMessages();
      }
    }

    videoElem.ontimeupdate = () => {
      if (!currentModalItem) return;
      const fps = currentModalItem.fps || 25.0;
      const currentSec = videoElem.currentTime;
      const currentFrame = Math.round(currentSec * fps);
      document.getElementById('modalCurrentFrame').innerText = `${currentFrame} (${formatTime(currentSec)}.${Math.floor((currentSec%1)*100)})`;
    };

    function updateWeights() {
      document.getElementById('wDenseVal').innerText = parseFloat(document.getElementById('wDense').value).toFixed(2);
      document.getElementById('wASRVal').innerText = parseFloat(document.getElementById('wASR').value).toFixed(2);
    }

    // --- Keyword Emphasis & Exact Matching ---
    function addKeyword(isExact = true) {
      const input = document.getElementById('newKeywordInput');
      const text = input.value.trim();
      if (!text) return;

      keywordsList.push({ id: nextKwId++, text: text, weight: 1.0, exact: isExact });
      input.value = '';
      renderKeywords();

      if (document.getElementById('queryInput').value.trim()) {
        if (markedMap.size > 0) {
          executeRerank();
        } else {
          executeSearch();
        }
      }
    }

    function toggleKeywordMode(id) {
      const kw = keywordsList.find(k => k.id === id);
      if (kw) {
        kw.exact = !kw.exact;
        renderKeywords();
        if (document.getElementById('queryInput').value.trim()) {
          if (markedMap.size > 0) {
            executeRerank();
          } else {
            executeSearch();
          }
        }
      }
    }

    function removeKeyword(id) {
      keywordsList = keywordsList.filter(k => k.id !== id);
      renderKeywords();
      if (document.getElementById('queryInput').value.trim()) {
        if (markedMap.size > 0) {
          executeRerank();
        } else {
          executeSearch();
        }
      }
    }

    function updateKeywordWeight(id, weightVal) {
      const kw = keywordsList.find(k => k.id === id);
      if (kw) {
        kw.weight = parseFloat(weightVal);
        const label = document.getElementById(`kwVal_${id}`);
        if (label) label.innerText = `${kw.weight.toFixed(1)}x`;
      }
    }

    function onKeywordWeightChange() {
      if (document.getElementById('queryInput').value.trim()) {
        if (markedMap.size > 0) {
          executeRerank();
        } else {
          executeSearch();
        }
      }
    }

    function renderKeywords() {
      const container = document.getElementById('keywordChipsContainer');
      container.innerHTML = '';
      keywordsList.forEach(kw => {
        const chip = document.createElement('div');
        const isExact = kw.exact;
        const chipBorder = isExact ? 'border-amber-500/50 bg-amber-950/20' : 'border-emerald-500/40 bg-slate-900';
        const tagColor = isExact ? 'text-amber-300' : 'text-emerald-300';
        const modeLabel = isExact ? '🔤 EXACT' : '🧠 SEMANTIC';

        chip.className = `flex items-center gap-2 ${chipBorder} border px-2.5 py-1.5 rounded-lg text-xs shadow-sm`;
        chip.innerHTML = `
          <button onclick="toggleKeywordMode(${kw.id})" class="text-[10px] font-bold px-1.5 py-0.5 rounded ${isExact ? 'bg-amber-500/30 text-amber-300' : 'bg-emerald-500/20 text-emerald-300'} hover:opacity-80 transition" title="Click to toggle between Exact Match and Semantic Mode">
            ${modeLabel}
          </button>
          <span class="font-medium ${tagColor}">"${kw.text}"</span>
          <div class="flex items-center gap-1 border-l border-slate-700/60 pl-2">
            <input type="range" min="0.1" max="2.0" step="0.1" value="${kw.weight}" class="w-14 accent-emerald-500" oninput="updateKeywordWeight(${kw.id}, this.value)" onchange="onKeywordWeightChange()">
            <span id="kwVal_${kw.id}" class="text-slate-400 font-mono text-[10px] w-6">${kw.weight.toFixed(1)}x</span>
          </div>
          <button onclick="removeKeyword(${kw.id})" class="text-slate-400 hover:text-rose-400 text-xs ml-1 font-bold">✕</button>
        `;
        container.appendChild(chip);
      });
    }

    // --- Correct Marks Management ---
    // --- Correct Marks Management ---
    function toggleMark(vid, fid, pts, event) {
      if (event) event.stopPropagation();
      const key = `${vid}_${fid}`;
      if (markedMap.has(key)) {
        markedMap.delete(key);
      } else {
        markedMap.set(key, { video_id: vid, frame_idx: fid, pts_time: pts, isPinned: false });
      }
      updateMarkedUI();
      renderGrid(currentResults);
      saveSessionToLocalStorage();
    }

    function clearMarks() {
      markedMap.clear();
      updateMarkedUI();
      renderGrid(currentResults);
      saveSessionToLocalStorage();
    }

    function updateMarkedUI() {
      const count = markedMap.size;
      const btn = document.getElementById('rerankBtn');
      const clearBtn = document.getElementById('clearMarksBtn');
      const fabBadge = document.getElementById('aiFabBadge');

      if (fabBadge) {
        if (count > 0) {
          fabBadge.innerText = count;
          fabBadge.classList.remove('hidden');
        } else {
          fabBadge.classList.add('hidden');
        }
      }

      if (count > 0) {
        btn.classList.remove('hidden');
        btn.innerHTML = `🎯 Re-rank (${count} marked)`;
        clearBtn.classList.remove('hidden');
      } else {
        btn.classList.add('hidden');
        clearBtn.classList.add('hidden');
      }
      updateVideoSelectorDropdown();
    }

    // --- Search Execution ---
    async function executeSearch() {
      const q = document.getElementById('queryInput').value.trim();
      if (!q) return;

      markedMap.clear();
      updateMarkedUI();

      const wDense = parseFloat(document.getElementById('wDense').value);
      const wASR = parseFloat(document.getElementById('wASR').value);
      const topK = parseInt(document.getElementById('topKSelect').value);

      document.getElementById('statusText').innerHTML = `<span class="animate-pulse text-emerald-400">Searching multi-modal index across 177k frames + BM25 speech...</span>`;

      try {
        const res = await fetch('/api/search', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query: q, keywords: keywordsList, w_dense: wDense, w_asr: wASR, top_k: topK })
        });
        const data = await res.json();
        currentResults = data.results || [];

        document.getElementById('transQuery').innerText = data.translated_query || 'N/A';
        document.getElementById('statusText').innerText = '';
        
        const badge = document.getElementById('timingBadge');
        badge.classList.remove('hidden');
        badge.innerText = `⚡ ${data.search_time_ms} ms`;

        populateVideoFilterDropdown(currentResults);
        renderGrid(currentResults);
        updateMarkedUI();
        saveSessionToLocalStorage();
      } catch (err) {
        document.getElementById('statusText').innerText = `❌ Error: ${err.message}`;
      }
    }

    async function executeRerank() {
      const q = document.getElementById('queryInput').value.trim();
      const marked_items = Array.from(markedMap.values());
      if (marked_items.length === 0) return;

      const wDense = parseFloat(document.getElementById('wDense').value);
      const wASR = parseFloat(document.getElementById('wASR').value);
      const topK = parseInt(document.getElementById('topKSelect').value);

      document.getElementById('statusText').innerHTML = `<span class="animate-pulse text-emerald-400">Re-ranking with ${marked_items.length} confirmed keyframe embeddings & scene proximity...</span>`;

      try {
        const res = await fetch('/api/refine', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            query: q,
            marked_items: marked_items,
            keywords: keywordsList,
            w_dense: wDense,
            w_asr: wASR,
            top_k: topK
          })
        });
        const data = await res.json();
        currentResults = data.results || [];

        document.getElementById('transQuery').innerText = data.translated_query || 'N/A';
        document.getElementById('statusText').innerText = '';
        
        const badge = document.getElementById('timingBadge');
        badge.classList.remove('hidden');
        badge.innerText = `⚡ ${data.search_time_ms} ms`;

        populateVideoFilterDropdown(currentResults);
        renderGrid(currentResults);
        updateMarkedUI();
        saveSessionToLocalStorage();
      } catch (err) {
        document.getElementById('statusText').innerText = `❌ Error: ${err.message}`;
      }
    }

    function renderGrid(items) {
      const grid = document.getElementById('resultsGrid');
      grid.innerHTML = '';

      let displayItems = items || [];
      if (activeVideoFilter) {
        displayItems = displayItems.filter(item => item.video_id.toLowerCase().includes(activeVideoFilter.toLowerCase()));
      }

      const countBadge = document.getElementById('filteredCountBadge');
      if (countBadge) {
        countBadge.innerText = `Showing ${displayItems.length} / ${(items || []).length} keyframes`;
      }

      if (!displayItems || displayItems.length === 0) {
        grid.innerHTML = `<div class="col-span-full py-16 text-center text-slate-500">No matching keyframes found${activeVideoFilter ? ` for video filter "${activeVideoFilter}"` : ''}.</div>`;
        return;
      }

      const activeQ = queryPackage.find(q => q.id === activeQueryId);
      const pinnedInfo = activeQ && activeQ.savedData && activeQ.savedData.pinnedFrame;

      displayItems.forEach((item) => {
        const origIdx = items.indexOf(item);
        const key = `${item.video_id}_${item.frame_idx}`;
        const isMarked = markedMap.has(key);
        const isPinned = (pinnedInfo && pinnedInfo.video_id === item.video_id && pinnedInfo.frame_idx === item.frame_idx) || (markedMap.get(key) && markedMap.get(key).isPinned);
        const vid = item.video_id;

        const card = document.createElement('div');
        const borderClass = isPinned
          ? 'border-2 border-amber-400 ring-2 ring-amber-500/40 shadow-amber-500/20'
          : (isMarked 
              ? 'border-2 border-emerald-400 ring-2 ring-emerald-500/30 shadow-emerald-500/20' 
              : 'hover:border-emerald-500/50 hover:shadow-emerald-500/10');

        card.className = `glass rounded-xl overflow-hidden shadow-lg ${borderClass} transition-all duration-200 flex flex-col group`;

        const matchPct = (Math.min(1.0, Math.max(0.0, item.score)) * 100).toFixed(1);
        const rankColor = item.rank === 1 ? 'bg-amber-400 text-slate-950 font-bold' : (item.rank <= 3 ? 'bg-slate-200 text-slate-900 font-bold' : 'bg-slate-800 text-slate-300');

        const qaAns = qaAnswersMap[vid];
        const trakeEvents = trakeEventsMap[vid] || [];

        card.innerHTML = `
          <div class="relative aspect-video bg-slate-900 overflow-hidden cursor-pointer" onclick="openModal(${origIdx >= 0 ? origIdx : 0})">
            <img src="${item.thumb_url}" loading="lazy" class="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300" onerror="this.src='https://via.placeholder.com/480x270/0f172a/64748b?text=Frame+Preview'" />
            
            <div class="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
              <div class="w-12 h-12 rounded-full bg-emerald-500 text-slate-950 flex items-center justify-center font-bold pl-1 shadow-lg transform group-hover:scale-110 transition-transform">
                ▶
              </div>
            </div>

            <div class="absolute top-2 left-2 ${rankColor} px-2 py-0.5 rounded text-[11px] shadow font-bold">
              #${item.rank}
            </div>
            ${isPinned ? `
              <div class="absolute top-2 left-12 bg-amber-400 text-slate-950 px-2 py-0.5 rounded text-[11px] font-black shadow flex items-center gap-1 animate-pulse">
                🎯 PINNED
              </div>
            ` : (isMarked ? `
              <div class="absolute top-2 left-12 bg-emerald-500 text-slate-950 px-2 py-0.5 rounded text-[11px] font-bold shadow flex items-center gap-1">
                ✓ MARKED
              </div>
            ` : '')}
            <div class="absolute top-2 right-2 bg-slate-950/80 backdrop-blur-md px-2 py-0.5 rounded text-[11px] font-mono text-emerald-400 border border-emerald-500/20 font-bold">
              ${matchPct}%
            </div>
            <div class="absolute bottom-2 right-2 bg-slate-950/80 px-1.5 py-0.5 rounded text-[10px] font-mono text-slate-300">
              ⏱ ${formatTime(item.pts_time)}
            </div>
          </div>
          
          <div class="p-3.5 flex flex-col justify-between flex-1 gap-2.5">
            <div>
              <div class="flex items-center justify-between text-xs">
                <span class="font-bold text-slate-200 tracking-wide font-mono">${item.video_id}</span>
                <span class="text-slate-400 font-mono">Frame ${item.frame_idx}</span>
              </div>

              ${item.matched_scene ? `
                <div class="mt-1.5 text-[10px] text-sky-300 font-medium bg-sky-950/50 px-2 py-1 rounded-md border border-sky-800/50 line-clamp-2 leading-tight flex items-start gap-1">
                  <span class="text-sky-400 font-bold whitespace-nowrap">🎯 Matched:</span>
                  <span class="italic truncate">${item.matched_scene}</span>
                </div>
              ` : ''}

              <!-- Task Mode Badges on Card -->
              ${currentTaskMode === 'qa' ? `
                <div class="mt-2 text-[11px] ${qaAns ? 'bg-amber-950/40 border-amber-500/40 text-amber-200' : 'bg-slate-900 border-slate-800 text-slate-500 italic'} p-2 rounded-lg border leading-relaxed flex items-start gap-1">
                  <span class="font-bold text-amber-400 whitespace-nowrap">💬 Answer:</span>
                  <span class="truncate font-medium">${qaAns ? `"${qaAns}"` : '(Chưa có đáp án - Click Play hoặc AI để nhập)'}</span>
                </div>
              ` : ''}

              ${currentTaskMode === 'trake' ? `
                <div class="mt-2 text-[11px] ${trakeEvents.length > 0 ? 'bg-indigo-950/40 border-indigo-500/40 text-indigo-200' : 'bg-slate-900 border-slate-800 text-slate-500 italic'} p-2 rounded-lg border leading-relaxed">
                  <span class="font-bold text-indigo-400">⏱️ TRAKE:</span> ${trakeEvents.length > 0 ? `<b>${trakeEvents.length} events</b> [${trakeEvents.map(e => 'F' + e.frame_idx).join(', ')}]` : '(Chưa có events - Click Play để đánh dấu chuỗi)'}
                </div>
              ` : ''}
              
              ${item.matched_asr ? `
                <div class="mt-2 text-[11px] text-emerald-200 italic bg-emerald-950/40 p-2.5 rounded-lg border border-emerald-800/40 line-clamp-3 leading-relaxed">
                  🎙️ "${item.matched_asr}"
                </div>` : ''}
            </div>

            <div class="flex items-center gap-1.5 pt-2 border-t border-slate-800/60 flex-wrap">
              <button onclick="toggleMark('${item.video_id}', ${item.frame_idx}, ${item.pts_time}, event)" class="flex-1 py-1.5 ${isMarked ? 'bg-emerald-500 text-slate-950 font-bold shadow-md shadow-emerald-500/20' : 'bg-slate-800 hover:bg-slate-700 text-emerald-400'} text-[11px] font-semibold rounded transition flex items-center justify-center gap-1">
                ${isMarked ? '✓ Correct' : '+ Mark'}
              </button>
              <button onclick="isolateVideo('${item.video_id}'); event.stopPropagation();" class="px-2 py-1.5 bg-slate-800 hover:bg-slate-700 text-sky-300 border border-sky-500/20 text-[11px] font-semibold rounded transition flex items-center gap-1" title="Chỉ hiển thị các khung hình thuộc video ${item.video_id}">
                🎬 Isolate
              </button>
              ${currentTaskMode === 'trake' ? `
                <button onclick="saveTrakeForCard('${item.video_id}', ${item.frame_idx}, ${item.pts_time}, event)" class="px-2.5 py-1.5 bg-indigo-600 hover:bg-indigo-500 text-white text-[11px] font-bold rounded transition flex items-center gap-1 shadow-sm" title="Lưu video này làm đáp án TRAKE cho câu query">
                  💾 Save
                </button>
              ` : ''}
              ${currentTaskMode === 'qa' ? `
                <button onclick="saveQAPromptForCard('${item.video_id}', ${item.frame_idx}, ${item.pts_time}, event)" class="px-2.5 py-1.5 bg-amber-500 hover:bg-amber-400 text-slate-950 text-[11px] font-bold rounded transition flex items-center gap-1 shadow-sm" title="Nhập và lưu đáp án Q&A">
                  💾 Save
                </button>
              ` : ''}
              <button onclick="openChatbotForVideo('${item.video_id}', ${item.frame_idx}, ${item.pts_time}, true); event.stopPropagation();" class="px-2 py-1.5 bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 text-[11px] font-semibold rounded transition flex items-center gap-1" title="Hỏi Trợ lý AI về video/clip này">
                🤖 AI
              </button>
              <button onclick="copySubmission('${item.video_id}', ${item.frame_idx}, this)" class="px-2 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 text-[11px] font-semibold rounded transition flex items-center justify-center gap-1">
                📋 Copy
              </button>
              <button onclick="openModal(${origIdx >= 0 ? origIdx : 0})" class="px-2 py-1.5 bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 text-[11px] font-semibold rounded transition flex items-center gap-1">
                ▶ Play
              </button>
            </div>
          </div>
        `;
        grid.appendChild(card);
      });
    }

    function formatTime(sec) {
      const m = Math.floor(sec / 60);
      const s = Math.floor(sec % 60);
      return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
    }

    function copySubmission(vid, fid, btn) {
      navigator.clipboard.writeText(`${vid},${fid}`);
      const orig = btn.innerText;
      btn.innerText = '✅ Copied!';
      btn.classList.add('text-emerald-400');
      setTimeout(() => {
        btn.innerText = orig;
        btn.classList.remove('text-emerald-400');
      }, 1500);
    }

    function openModal(idx) {
      const item = currentResults[idx];
      if (!item) return;

      currentModalItem = item;
      const vid = item.video_id;

      document.getElementById('modalTitle').innerText = `${vid} (Playing at ${item.pts_time}s)`;
      document.getElementById('modalVideo').innerText = vid;
      document.getElementById('modalCandidateFrame').innerText = `Frame ${item.frame_idx} (${item.pts_time}s)`;
      document.getElementById('modalASR').innerText = item.matched_asr || '(No direct speech transcript around timestamp)';
      let currentModalSegments = [];
      fetch(`/api/video_context?video_id=${encodeURIComponent(vid)}&pts_time=${item.pts_time}`)
        .then(r => r.json())
        .then(data => {
          if (data && data.asr_segments && data.asr_segments.length > 0) {
            currentModalSegments = data.asr_segments;
            updateModalSubtitle(videoElem.currentTime);
          }
        }).catch(() => {});

      const activeQ = queryPackage.find(q => q.id === activeQueryId);
      const isPinned = activeQ && activeQ.savedData && activeQ.savedData.pinnedFrame && activeQ.savedData.pinnedFrame.video_id === vid;
      const pinStatus = document.getElementById('modalPinnedStatus');
      if (pinStatus) {
        if (isPinned) {
          pinStatus.innerHTML = `<span class="text-amber-300 font-bold font-mono">🎯 Frame ${activeQ.savedData.pinnedFrame.frame_idx}</span>`;
        } else {
          pinStatus.innerHTML = `<span class="text-slate-500 font-mono">Not Pinned</span>`;
        }
      }

      // Sync modal clip marks
      const range = videoClipRanges[vid];
      if (range && range.start_sec !== undefined) {
        modalClipStartSec = range.start_sec;
        document.getElementById('modalClipStartBadge').innerText = formatTime(modalClipStartSec);
      } else {
        modalClipStartSec = null;
        document.getElementById('modalClipStartBadge').innerText = '--:--';
      }

      if (range && range.end_sec !== undefined) {
        modalClipEndSec = range.end_sec;
        document.getElementById('modalClipEndBadge').innerText = formatTime(modalClipEndSec);
      } else {
        modalClipEndSec = null;
        document.getElementById('modalClipEndBadge').innerText = '--:--';
      }

      // Sync Q&A Answer Input in Modal
      const curAns = qaAnswersMap[vid] || '';
      const mInput = document.getElementById('modalQAInput');
      if (mInput) mInput.value = curAns;
      const mCount = document.getElementById('modalQACharCount');
      if (mCount) mCount.innerText = `${curAns.length}/100 chars`;

      // Render TRAKE Events in Modal
      renderTrakeEventsModal(vid);

      // Sync Mode UI
      syncModalModeUI();

      videoElem.src = item.video_url;
      const initialFps = item.fps || 25.0;
      videoElem.currentTime = Math.max(0, (item.frame_idx / initialFps) - 1.0);
      videoElem.play();

      const updateModalSubtitle = (curSec) => {
        if (!currentModalSegments || currentModalSegments.length === 0) return;
        let found = null;
        for (const seg of currentModalSegments) {
          if ((seg.start_sec - 0.8) <= curSec && curSec <= (seg.end_sec + 0.8)) {
            found = seg;
            break;
          }
        }
        const asrEl = document.getElementById('modalASR');
        if (asrEl) {
          if (found) {
            asrEl.innerText = `[${found.start_sec.toFixed(1)}s - ${found.end_sec.toFixed(1)}s] ${found.text}`;
          } else if (item.matched_asr && Math.abs(curSec - item.pts_time) < 4.0) {
            asrEl.innerText = item.matched_asr;
          }
        }
      };

      const updateFrameDisplay = () => {
        const fps = (currentModalItem && currentModalItem.fps) || 25.0;
        const curSec = videoElem.currentTime;
        const curFr = Math.max(0, Math.round(curSec * fps));
        const frLabel = document.getElementById('modalCurrentFrame');
        if (frLabel) frLabel.innerText = `${curFr} (${formatTime(curSec)})`;
        updateModalSubtitle(curSec);
        if ('requestVideoFrameCallback' in HTMLVideoElement.prototype && !videoElem.paused) {
          videoElem._rvfcId = videoElem.requestVideoFrameCallback(updateFrameDisplay);
        }
      };

      videoElem.onplay = () => {
        if ('requestVideoFrameCallback' in HTMLVideoElement.prototype) {
          videoElem._rvfcId = videoElem.requestVideoFrameCallback(updateFrameDisplay);
        }
      };
      videoElem.ontimeupdate = updateFrameDisplay;
      videoElem.onseeked = updateFrameDisplay;

      const copyBtn = document.getElementById('modalCopyBtn');
      copyBtn.onclick = () => {
        navigator.clipboard.writeText(`${item.video_id},${item.frame_idx}`);
        copyBtn.innerText = '✅ Copied!';
        setTimeout(() => copyBtn.innerText = '📋 Copy Candidate', 1500);
      };

      document.getElementById('detailModal').classList.remove('hidden');
    }

    function pinModalPlayingFrame() {
      if (!currentModalItem) return;
      const fps = currentModalItem.fps || 25.0;
      const currentSec = videoElem.currentTime;
      const currentFrame = Math.max(0, Math.round(currentSec * fps));
      const vid = currentModalItem.video_id;

      currentModalItem.pinned_frame = currentFrame;

      const key = `${vid}_${currentFrame}`;
      markedMap.set(key, {
        video_id: vid,
        frame_idx: currentFrame,
        pts_time: currentSec,
        isPinned: true
      });

      const qid = (document.getElementById('queryIdInput').value || activeQueryId || 'query-1').trim();
      let query = queryPackage.find(q => q.id === qid);
      if (query) {
        if (!query.savedData) query.savedData = {};
        query.savedData.pinnedFrame = { video_id: vid, frame_idx: currentFrame, pts_time: currentSec };
        query.savedData.pinnedVideoId = vid;
        query.status = 'saved';
      }

      const pinStatus = document.getElementById('modalPinnedStatus');
      if (pinStatus) {
        pinStatus.innerHTML = `<span class="text-amber-300 font-bold font-mono">🎯 Frame ${currentFrame} (${formatTime(currentSec)})</span>`;
      }

      updateMarkedUI();
      renderGrid(currentResults);
      saveSessionToLocalStorage();
      renderQueryNavigator();

      showQuickToast(`🎯 Đã chọn Frame ${currentFrame} (${vid}) làm mốc đáp án chính xác!`);
    }

    function floodNeighborsModal() {
      if (!currentModalItem) return;
      const fps = currentModalItem.fps || 25.0;
      const currentSec = videoElem.currentTime;
      const currentFrame = currentModalItem.pinned_frame !== undefined 
        ? currentModalItem.pinned_frame 
        : Math.max(0, Math.round(currentSec * fps));
      const vid = currentModalItem.video_id;

      const neighborRows = generateNeighborFrames(vid, currentFrame, 100, 1);
      
      const qid = (document.getElementById('queryIdInput').value || activeQueryId || 'query-1').trim();
      let query = queryPackage.find(q => q.id === qid);
      if (query) {
        if (!query.savedData) query.savedData = {};
        query.savedData.floodFrames = neighborRows;
        query.savedData.floodCenter = currentFrame;
        query.savedData.floodVideoId = vid;
        query.status = 'saved';
      }

      markedMap.set(`${vid}_${currentFrame}`, {
        video_id: vid,
        frame_idx: currentFrame,
        pts_time: currentSec,
        isPinned: true
      });

      updateMarkedUI();
      saveSessionToLocalStorage();
      renderQueryNavigator();

      showQuickToast(`⚡ Đã điền 100 frame liền kề quanh Frame ${currentFrame} (${vid}) vào kết quả xuất bài!`);
    }

    function markCurrentModalFrameAndRerank() {
      if (!currentModalItem) return;
      const fps = currentModalItem.fps || 25.0;
      const currentSec = videoElem.currentTime;
      const currentFrame = Math.round(currentSec * fps);
      const vid = currentModalItem.video_id;

      markedMap.set(`${vid}_${currentFrame}`, {
        video_id: vid,
        frame_idx: currentFrame,
        pts_time: currentSec,
        isPinned: false
      });

      closeModal();
      executeRerank();
    }

    function jumpToCandidate() {
      if (currentModalItem) {
        const fps = currentModalItem.fps || 25.0;
        videoElem.currentTime = currentModalItem.frame_idx / fps;
        videoElem.pause();
        const curFr = currentModalItem.frame_idx;
        const frLabel = document.getElementById('modalCurrentFrame');
        if (frLabel) frLabel.innerText = `${curFr} (${formatTime(videoElem.currentTime)})`;
      }
    }

    function seekRel(delta) {
      videoElem.currentTime = Math.max(0, videoElem.currentTime + delta);
    }

    function setSpeed(spd) {
      videoElem.playbackRate = spd;
      document.querySelectorAll('.speed-btn').forEach(btn => {
        if (parseFloat(btn.dataset.spd) === spd) {
          btn.className = 'speed-btn px-2 py-1 bg-emerald-500 text-slate-950 font-bold rounded';
        } else {
          btn.className = 'speed-btn px-2 py-1 bg-slate-800 text-slate-300 rounded';
        }
      });
    }

    function copyCurrentPlayingFrame(btn) {
      if (!currentModalItem) return;
      const fps = currentModalItem.fps || 25.0;
      const currentFrame = Math.round(videoElem.currentTime * fps);
      navigator.clipboard.writeText(`${currentModalItem.video_id},${currentFrame}`);
      const orig = btn.innerText;
      btn.innerText = `✅ Copied (${currentModalItem.video_id}, ${currentFrame})!`;
      setTimeout(() => btn.innerText = orig, 1500);
    }

    function closeModal() {
      videoElem.pause();
      videoElem.src = '';
      videoElem.ontimeupdate = null;
      videoElem.onseeked = null;
      videoElem.onplay = null;
      document.getElementById('detailModal').classList.add('hidden');
    }

    // RFC-4180 Compliant CSV escaping (as specified in plan.txt)
    function formatCSVField(val) {
      if (val === null || val === undefined) return '';
      const str = String(val).replace(/\r/g, ' ').replace(/\n/g, ' ').trim().slice(0, 100);
      if (str.includes(',') || str.includes('"')) {
        return `"${str.replace(/"/g, '""')}"`;
      }
      return str;
    }

    function exportCSV() {
      const qid = (document.getElementById('queryIdInput').value || 'query-1').trim();
      let query = queryPackage.find(q => q.id === qid);
      if (!query) {
        saveActiveQueryResult();
        query = queryPackage.find(q => q.id === qid);
      }

      const filename = `${qid}-${currentTaskMode}.csv`;
      const lines = buildCSVLinesForQuery(query);

      if (lines.length === 0) {
        alert("Không có kết quả nào để xuất CSV. Vui lòng thực hiện tìm kiếm trước.");
        return;
      }

      const csvContent = lines.join(String.fromCharCode(10)) + String.fromCharCode(10);
      const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      a.click();
      URL.revokeObjectURL(url);
    }

    // Initialize Settings & Task Mode on DOM ready
    window.addEventListener('DOMContentLoaded', () => {
      loadChatSettings();
      const restored = loadSessionFromLocalStorage();
      if (!restored) {
        renderQueryNavigator();
      }
    });
  </script>
</body>
</html>
"""


class RequestHandler(BaseHTTPRequestHandler):
    def _send_json_error(self, status_code: int, error_message: str):
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(json.dumps({"error": error_message}, ensure_ascii=False).encode("utf-8"))

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        try:
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path
            query = urllib.parse.parse_qs(parsed.query)

            if path == "/":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(HTML_PAGE.encode("utf-8"))

            elif path == "/api/frame":
                vid = sanitize_video_id(query.get("video_id", [""])[0])
                try:
                    fid = int(query.get("frame_idx", [0])[0])
                except (ValueError, TypeError):
                    fid = 0

                jpg_data = ENGINE.extract_thumbnail(vid, fid)
                if jpg_data:
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Cache-Control", "public, max-age=86400")
                    self.end_headers()
                    self.wfile.write(jpg_data)
                else:
                    self.send_response(404)
                    self.end_headers()

            elif path == "/api/video":
                vid = sanitize_video_id(query.get("video_id", [""])[0])
                vid_path = ENGINE.video_paths.get(vid)
                if not vid_path or not os.path.exists(vid_path):
                    self.send_response(404)
                    self.end_headers()
                    return

                file_size = os.path.getsize(vid_path)
                range_header = self.headers.get("Range")

                if not range_header:
                    self.send_response(200)
                    self.send_header("Content-Type", "video/mp4")
                    self.send_header("Content-Length", str(file_size))
                    self.send_header("Accept-Ranges", "bytes")
                    self.end_headers()
                    try:
                        with open(vid_path, "rb") as f:
                            while chunk := f.read(65536):
                                self.wfile.write(chunk)
                    except Exception:
                        pass
                    return

                range_match = re.match(r"bytes=(\d+)-(\d*)", range_header)
                if not range_match:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{file_size}")
                    self.end_headers()
                    return

                start = int(range_match.group(1))
                end = int(range_match.group(2)) if range_match.group(2) else file_size - 1
                end = min(end, file_size - 1)

                if start < 0 or start >= file_size or start > end:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{file_size}")
                    self.end_headers()
                    return

                chunk_size = end - start + 1

                self.send_response(206)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                self.send_header("Content-Length", str(chunk_size))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()

                try:
                    with open(vid_path, "rb") as f:
                        f.seek(start)
                        bytes_left = chunk_size
                        while bytes_left > 0:
                            read_len = min(bytes_left, 65536)
                            data = f.read(read_len)
                            if not data:
                                break
                            self.wfile.write(data)
                            bytes_left -= len(data)
                except Exception:
                    pass

            elif path == "/api/video_context":
                vid = sanitize_video_id(query.get("video_id", [""])[0])
                try:
                    pts = float(query.get("pts_time", [0.0])[0])
                    if not math.isfinite(pts):
                        pts = 0.0
                except (ValueError, TypeError):
                    pts = 0.0
                ctx = ENGINE.get_video_context(vid, pts)
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps(ctx, ensure_ascii=False).encode("utf-8"))

            else:
                self.send_response(404)
                self.end_headers()
        except Exception as exc:
            self._send_json_error(500, f"Server Error: {str(exc)}")

    def do_POST(self):
        try:
            parsed = urllib.parse.urlparse(self.path)
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len) if content_len > 0 else b"{}"
            try:
                payload = json.loads(body.decode("utf-8")) if body else {}
            except Exception:
                self._send_json_error(400, "Invalid JSON payload format.")
                return

            if parsed.path == "/api/search":
                q = str(payload.get("query", ""))
                keywords = payload.get("keywords") if isinstance(payload.get("keywords"), list) else []
                w_dense = payload.get("w_dense", 0.50)
                w_asr = payload.get("w_asr", 0.50)
                top_k = payload.get("top_k", 100)

                res = ENGINE.search(q, keywords=keywords, w_dense=w_dense, w_asr=w_asr, top_k=top_k)

                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps(res, ensure_ascii=False).encode("utf-8"))

            elif parsed.path == "/api/refine":
                q = str(payload.get("query", ""))
                marked_items = payload.get("marked_items") if isinstance(payload.get("marked_items"), list) else []
                keywords = payload.get("keywords") if isinstance(payload.get("keywords"), list) else []
                w_dense = payload.get("w_dense", 0.50)
                w_asr = payload.get("w_asr", 0.50)
                top_k = payload.get("top_k", 100)

                res = ENGINE.refine(q, marked_items=marked_items, keywords=keywords, w_dense=w_dense, w_asr=w_asr, top_k=top_k)

                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps(res, ensure_ascii=False).encode("utf-8"))

            elif parsed.path == "/api/chat":
                video_id = sanitize_video_id(payload.get("video_id", ""))
                try:
                    frame_idx = int(payload.get("frame_idx", 0))
                except (ValueError, TypeError):
                    frame_idx = 0
                try:
                    pts_time = float(payload.get("pts_time", 0.0))
                except (ValueError, TypeError):
                    pts_time = 0.0

                start_sec = payload.get("start_sec")
                end_sec = payload.get("end_sec")
                user_query = str(payload.get("query", ""))
                messages = payload.get("messages") if isinstance(payload.get("messages"), list) else []
                api_key = str(payload.get("api_key", ""))
                provider = str(payload.get("provider", "omniroute"))
                model = str(payload.get("model", ""))
                custom_url = str(payload.get("custom_url", ""))

                res = ENGINE.chat_completion(
                    messages=messages,
                    video_id=video_id,
                    frame_idx=frame_idx,
                    pts_time=pts_time,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    query=user_query,
                    api_key=api_key,
                    provider=provider,
                    model=model,
                    custom_url=custom_url
                )

                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps(res, ensure_ascii=False).encode("utf-8"))
            else:
                self.send_response(404)
                self.end_headers()
        except Exception as exc:
            self._send_json_error(500, f"Internal Server Error: {str(exc)}")

    def log_message(self, format, *args):
        pass


def launch_server(port: int = 8080):
    global ENGINE
    ENGINE = SearchEngine()

    server = ThreadingHTTPServer(("0.0.0.0", port), RequestHandler)
    print(f"\n=======================================================")
    print(f"🌟 AIC 2026 Multi-Modal Retrieval Studio Live at:")
    print(f"👉 http://localhost:{port}")
    print(f"=======================================================\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped.")


if __name__ == "__main__":
    launch_server(port=8080)
