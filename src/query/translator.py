import re
from typing import List, Optional
try:
    from deep_translator import GoogleTranslator
    _HAS_TRANSLATOR = True
except ImportError:
    _HAS_TRANSLATOR = False

class QueryTranslator:
    """
    Translates Vietnamese natural language queries to English for CLIP alignment
    and generates visual prompt variations.
    """
    def __init__(self, use_online: bool = True):
        self.use_online = use_online and _HAS_TRANSLATOR
        self.translator = GoogleTranslator(source='vi', target='en') if self.use_online else None
        
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
            "bóng đá": "football soccer", "nhảy cao": "high jump", "tiếp đất": "landing"
        }

    def clean_vietnamese_query(self, query: str) -> str:
        """
        Strips query boilerplate such as 'Tìm video về...', 'Tìm cảnh...', 'Trong video...'.
        """
        prefixes = [
            r"^tìm\s+video\s+về\s+",
            r"^tìm\s+cảnh\s+",
            r"^tìm\s+đoạn\s+video\s+",
            r"^tìm\s+",
            r"^trong\s+video\s+quay\s+cảnh\s+",
            r"^trong\s+video\s+về\s+",
            r"^trong\s+video\s+",
        ]
        cleaned = query.strip()
        for p in prefixes:
            cleaned = re.sub(p, "", cleaned, flags=re.IGNORECASE).strip()
        return cleaned

    def translate(self, query: str) -> str:
        cleaned = self.clean_vietnamese_query(query)
        if self.use_online and self.translator is not None:
            try:
                en_text = self.translator.translate(cleaned)
                if en_text:
                    return en_text
            except Exception as e:
                print(f"[QueryTranslator] Online translation error: {e}, using dictionary fallback.")

        # Fallback keyword replacement
        en_words = []
        lower_q = cleaned.lower()
        for vi_k, en_v in self.dict_vi_en.items():
            if vi_k in lower_q:
                en_words.append(en_v)
        if en_words:
            return " ".join(en_words)
        return cleaned

    def generate_prompts(self, query_en: str) -> List[str]:
        """
        Generates prompt ensemble templates for CLIP text encoder.
        """
        return [
            query_en,
            f"a photo of {query_en}",
            f"a video frame showing {query_en}",
            f"a scene of {query_en}",
            f"{query_en} in high quality"
        ]
