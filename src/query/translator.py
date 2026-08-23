import os
import re
import json
import urllib.request
from typing import List, Optional

try:
    from deep_translator import GoogleTranslator
    _HAS_TRANSLATOR = True
except ImportError:
    _HAS_TRANSLATOR = False


class QueryTranslator:
    """
    Translates Vietnamese natural language queries to English for CLIP alignment
    using local OmniRoute LLM with GoogleTranslator & Dictionary fallbacks,
    backed by a persistent disk and memory cache.
    """
    def __init__(
        self,
        use_online: bool = True,
        omniroute_url: str = "http://localhost:20128/v1/chat/completions",
        cache_file: str = "cache/translation_cache.json"
    ):
        self.use_online = use_online
        self.omniroute_url = omniroute_url
        self.model_name = "antigravity/gemini-3.6-flash-medium"
        self.cache_file = cache_file
        self.google_translator = GoogleTranslator(source='vi', target='en') if (_HAS_TRANSLATOR and use_online) else None
        self.cache = {}
        self._fail_count = 0
        self._omniroute_available = True
        
        # Load persistent disk cache
        self._load_disk_cache()

        # Bilingual fallback dictionary for key visual entities and actions
        self.dict_vi_en = {
            "người": "person", "đàn ông": "man", "phụ nữ": "woman", "trẻ em": "child",
            "xe hơi": "car", "ô tô": "car", "xe máy": "motorcycle", "xe đạp": "bicycle", "xe buýt": "bus",
            "áo đỏ": "red shirt", "áo xanh": "blue shirt", "áo trắng": "white shirt", "áo đen": "black shirt",
            "váy": "dress", "mũ": "hat", "kính": "glasses", "hoa": "flower", "cây": "tree",
            "chó": "dog", "mèo": "cat", "chim": "bird", "bò": "cow", "ngựa": "horse",
            "máy tính": "laptop", "điện thoại": "phone", "ly": "cup", "bàn": "table", "ghế": "chair",
            "phát biểu": "speaking speech", "chạy": "running", "nhảy": "jumping", "bơi": "swimming",
            "nấu ăn": "cooking", "ăn": "eating", "uống": "drinking", "hát": "singing",
            "bữa tiệc": "party", "sân khấu": "stage", "ngoài trời": "outdoors", "trong nhà": "indoors",
            "họp báo": "press conference", "thời sự": "news broadcast", "trao giải": "award ceremony",
            "bóng đá": "football soccer", "nhảy cao": "high jump", "tiếp đất": "landing",
            "thủy lợi": "irrigation water dam", "đập": "dam", "kênh": "canal waterway",
            "ngăn mặn": "saltwater prevention gate", "chữa cháy": "firefighting fire rescue"
        }

    def _load_disk_cache(self):
        if self.cache_file and os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
            except Exception:
                self.cache = {}

    def _save_disk_cache(self):
        if not self.cache_file:
            return
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def clean_vietnamese_query(self, query: str) -> str:
        """
        Strips query boilerplate without removing legitimate prefixes of compound words.
        """
        prefixes = [
            r"^(tìm\s+(đoạn\s+)?(video|clip|phim|cảnh)\s+(về|quay\s+cảnh)?\s*)",
            r"^(video|clip|đoạn\s+video|đoạn\s+clip|đoạn\s+phim)\s+(về|quay\s+cảnh)?\s*",
            r"^(hình\s+ảnh|cảnh\s+quay|cận\s+cảnh)\s+(về|quay)?\s*",
            r"^(trong\s+video(\s+quay\s+cảnh|\s+về)?\s*)",
            r"^(cho\s+tôi\s+xem\s+)",
            r"^(hãy\s+tìm\s+)",
            r"^(tìm\s+kiếm\s+)",
            r"^(tìm\s+)",
            # Strip 'cảnh' only if not part of 'cảnh sát', 'cảnh báo', 'cảnh giác', 'cảnh quan'
            r"^(cảnh\s+(?!(sát|báo|giác|quan|vật|sắc)\b))",
        ]
        cleaned = query.strip()
        for p in prefixes:
            cleaned = re.sub(p, "", cleaned, flags=re.IGNORECASE).strip()
        self._omniroute_available = True
        return cleaned

    def _translate_via_omniroute(self, text: str) -> Optional[str]:
        if not getattr(self, "_omniroute_available", True):
            return None
        body = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are a professional translator for visual search. Translate the given Vietnamese video search query to clear, concise, natural English visual search terms. Preserve specific named entities, brands, and landmarks. Output ONLY the translated English text with no quotes, explanations, or trailing punctuation."},
                {"role": "user", "content": text}
            ],
            "temperature": 0.0,
            "stream": False
        }
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(self.omniroute_url, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                raw_bytes = resp.read()
                raw_str = raw_bytes.decode("utf-8").strip()
                translated = ""
                if raw_str.startswith("data:"):
                    for line in raw_str.splitlines():
                        line = line.strip()
                        if line.startswith("data:") and line != "data: [DONE]":
                            try:
                                chunk = json.loads(line[5:].strip())
                                delta = chunk["choices"][0].get("delta", {})
                                translated += delta.get("content", "")
                            except Exception:
                                pass
                else:
                    res = json.loads(raw_str)
                    translated = res["choices"][0]["message"]["content"].strip()

                translated = re.sub(r"^[\"']|[\"']$", "", translated).strip()
                if translated:
                    self._fail_count = 0
                    return translated
        except Exception:
            self._fail_count = getattr(self, "_fail_count", 0) + 1
            if self._fail_count >= 5:
                self._omniroute_available = False
            return None
        return None

    def translate(self, query: str) -> str:
        cleaned = self.clean_vietnamese_query(query)
        if not cleaned:
            return ""

        if cleaned in self.cache:
            return self.cache[cleaned]

        res = None
        # 1. High-fidelity translation via OmniRoute LLM (fast fail)
        if self.use_online and getattr(self, "_omniroute_available", True):
            res = self._translate_via_omniroute(cleaned)

        # 2. GoogleTranslator fallback
        if not res and self.use_online and self.google_translator is not None:
            try:
                en_text = self.google_translator.translate(cleaned)
                if en_text:
                    res = en_text.strip()
            except Exception:
                pass

        is_authoritative = bool(res)
        # 3. Fallback keyword replacement dictionary
        if not res:
            en_words = []
            lower_q = cleaned.lower()
            for vi_k, en_v in self.dict_vi_en.items():
                if vi_k in lower_q:
                    en_words.append(en_v)
            if en_words:
                res = " ".join(en_words)

        if not res:
            res = cleaned

        if is_authoritative:
            self.cache[cleaned] = res
            self._save_disk_cache()
        return res

    def generate_prompts(self, query_en: str) -> List[str]:
        """
        Generates prompt ensemble templates for CLIP text encoder.
        """
        if not query_en:
            return [""]
        return [
            query_en,
            f"a photo of {query_en}",
            f"a video frame showing {query_en}",
            f"a scene of {query_en}",
            f"{query_en} in high quality"
        ]
