import os
import glob
from typing import List, Dict, Optional
import numpy as np

# Ensure numpy 1.x compatibility shim for scipy/paddle
if not hasattr(np, 'long'):
    np.long = int
if not hasattr(np, 'ulong'):
    np.ulong = int

class OCRExtractor:
    """
    On-screen Optical Character Recognition for Vietnamese video keyframes and banners
    using a GPU-accelerated Hybrid Architecture:
      - PaddleOCR DBNet for fast, robust bounding box text detection
      - VietOCR Seq2Seq/Transformer for native, accurate Vietnamese diacritics
    """
    def __init__(self, device: str = "cuda:0", rec_model: str = "vgg_seq2seq"):
        self.device = device
        self.rec_model = rec_model
        self.paddle_det = None
        self.viet_rec = None

    def _lazy_load_reader(self):
        if self.paddle_det is None or self.viet_rec is None:
            try:
                import torch
                from PIL import Image
                from paddleocr import PaddleOCR
                from vietocr.tool.predictor import Predictor
                from vietocr.tool.config import Cfg

                use_gpu = ("cuda" in str(self.device)) and torch.cuda.is_available()
                
                # 1. Fast GPU Text Detector
                self.paddle_det = PaddleOCR(
                    use_angle_cls=False,
                    rec=False,
                    use_gpu=use_gpu,
                    show_log=False
                )
                
                # 2. Native Vietnamese Text Recognizer
                v_cfg = Cfg.load_config_from_name(self.rec_model)
                v_cfg['device'] = self.device if use_gpu else 'cpu'
                v_cfg['predictor']['beamsearch'] = False
                self.viet_rec = Predictor(v_cfg)
                
                print(f"[OCRExtractor] Hybrid PaddleDet + VietOCR ({self.rec_model}) loaded successfully (GPU={use_gpu}).")
            except Exception as e:
                print(f"[OCRExtractor] Note: OCR running in fallback dummy mode ({e}).")
                self.paddle_det = "dummy"
                self.viet_rec = "dummy"

    def get_reader(self):
        self._lazy_load_reader()
        return self.viet_rec

    def extract_text_from_frame(self, frame_rgb) -> str:
        """
        Extracts concatenated text directly from an in-memory RGB numpy array.
        """
        self._lazy_load_reader()
        if self.paddle_det is None or self.paddle_det == "dummy":
            return ""

        try:
            from PIL import Image
            boxes_res = self.paddle_det.ocr(frame_rgb, det=True, rec=False, cls=False)
            if not boxes_res or not boxes_res[0]:
                return ""

            boxes = boxes_res[0]
            crops = []
            h, w = frame_rgb.shape[:2]

            for b in boxes:
                xs = [pt[0] for pt in b]
                ys = [pt[1] for pt in b]
                x1, x2 = max(0, int(min(xs))), min(w, int(max(xs)))
                y1, y2 = max(0, int(min(ys))), min(h, int(max(ys)))
                if (x2 - x1) < 8 or (y2 - y1) < 8:
                    continue
                crop_arr = frame_rgb[y1:y2, x1:x2]
                crops.append(Image.fromarray(crop_arr))

            if not crops:
                return ""

            texts = self.viet_rec.predict_batch(crops)
            cleaned = [t.strip() for t in texts if t and len(t.strip()) > 1]
            return " ".join(cleaned)
        except Exception:
            return ""

    def extract_text_from_image(self, image_path: str) -> str:
        """
        Extracts concatenated on-screen text from an image file.
        """
        if not os.path.exists(image_path):
            return ""

        try:
            import cv2
            img_bgr = cv2.imread(image_path)
            if img_bgr is None:
                return ""
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            return self.extract_text_from_frame(img_rgb)
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



