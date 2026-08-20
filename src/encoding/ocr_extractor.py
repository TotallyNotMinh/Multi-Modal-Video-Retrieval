import os
import glob
from typing import List, Dict, Optional

class OCRExtractor:
    """
    On-screen Optical Character Recognition for Vietnamese video keyframes and banners.
    """
    def __init__(self, device: str = "cuda:0", languages: Optional[List[str]] = None):
        self.device = device
        self.languages = languages if languages is not None else ["vi", "en"]
        self.reader = None

    def _lazy_load_reader(self):
        if self.reader is None:
            try:
                import easyocr
                import torch
                use_gpu = ("cuda" in str(self.device)) and torch.cuda.is_available()
                self.reader = easyocr.Reader(self.languages, gpu=use_gpu)
                print("[OCRExtractor] EasyOCR loaded successfully.")
            except Exception as e:
                print(f"[OCRExtractor] Note: OCR running in fallback dummy mode ({e}).")
                self.reader = "dummy"

    def get_reader(self):
        self._lazy_load_reader()
        return self.reader

    def extract_text_from_frame(self, frame_rgb) -> str:
        """
        Extracts concatenated text directly from an in-memory RGB numpy array.
        """
        self._lazy_load_reader()
        if self.reader is None or self.reader == "dummy":
            return ""

        try:
            results = self.reader.readtext(frame_rgb)
            texts = [r[1] for r in results if len(r) > 2 and float(r[2]) > 0.3]
            return " ".join(texts)
        except Exception:
            return ""

    def extract_text_from_image(self, image_path: str) -> str:
        """
        Extracts concatenated on-screen text from an image file.
        """
        self._lazy_load_reader()
        if not os.path.exists(image_path) or self.reader == "dummy":
            return ""

        try:
            results = self.reader.readtext(image_path)
            # results: list of [bbox, text, conf]
            texts = [r[1] for r in results if len(r) > 2 and float(r[2]) > 0.3]
            return " ".join(texts)
        except Exception:
            return ""

    def batch_extract_video_keyframes(self, keyframe_paths: List[str]) -> Dict[str, str]:
        """
        Extracts OCR text for all keyframes in a video.
        """
        ocr_results = {}
        for kf in keyframe_paths:
            kf_name = os.path.splitext(os.path.basename(kf))[0]
            text = self.extract_text_from_image(kf)
            if text:
                ocr_results[kf_name] = text
        return ocr_results

