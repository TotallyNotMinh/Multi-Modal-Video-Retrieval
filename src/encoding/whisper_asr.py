import os
import subprocess
import json
import torch
from typing import List, Dict, Optional
from src.index.frame_mapper import FrameMapper

class WhisperASR:
    """
    Automatic Speech Recognition for Vietnamese video audio using Whisper large-v3 / medium / base.
    Maps transcribed speech segments to exact video timestamps and frame indices.
    """
    def __init__(
        self,
        model_size: str = "large-v3",
        device: Optional[str] = None,
        language: str = "vi"
    ):
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model_size = model_size
        self.language = language
        self.mapper = FrameMapper()
        self.model = None

    def _lazy_load_model(self):
        if self.model is None:
            print(f"[WhisperASR] Loading Whisper '{self.model_size}' on {self.device}...")
            try:
                import whisper
                self.model = whisper.load_model(self.model_size, device=self.device)
            except Exception as e:
                print(f"[WhisperASR] Note: Whisper running in fallback mode ({e}).")
                self.model = "dummy"

    def transcribe_video(
        self,
        video_path: str,
        fps: float = 30.0
    ) -> List[Dict]:
        """
        Transcribes audio from an MP4 video file and returns aligned segments.
        """
        self._lazy_load_model()
        if not os.path.exists(video_path) or self.model == "dummy":
            return []

        try:
            result = self.model.transcribe(
                video_path,
                language=self.language,
                task="transcribe",
                fp16=("cuda" in str(self.device))
            )
        except Exception as e:
            print(f"[WhisperASR] Error transcribing {video_path}: {e}")
            return []

        segments = []
        raw_segments = result.get("segments", [])
        video_id = os.path.splitext(os.path.basename(video_path))[0]

        for seg in raw_segments:
            start_t = float(seg.get("start", 0.0))
            end_t = float(seg.get("end", 0.0))
            text = seg.get("text", "").strip()
            
            if not text:
                continue

            start_frame = int(start_t * fps)
            end_frame = int(end_t * fps)

            segments.append({
                "video_id": video_id,
                "start_sec": start_t,
                "end_sec": end_t,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "text": text
            })

        return segments
