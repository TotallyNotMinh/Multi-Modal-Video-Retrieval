import os
import gc
import torch
from typing import List, Dict, Optional
from src.index.frame_mapper import FrameMapper


class WhisperASR:
    """
    High-throughput Automatic Speech Recognition for Vietnamese video audio using
    faster-whisper (CTranslate2) with large-v3-turbo / large-v3.

    VRAM Maximization & Auto-Batch OOM Recovery:
      - Starts with maximum batch size (e.g. 64 / 32) to saturate GPU memory and maximize throughput.
      - Uses faster_whisper.BatchedInferencePipeline for high-concurrency transformer decoding.
      - Silero VAD pre-filter eliminates non-speech audio (silence/music).
      - Automatically catches CUDA Out-Of-Memory and dynamic memory pressure, halving the
        batch size: 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1, then persisting the stable batch size.
    """

    def __init__(
        self,
        model_size: str = "large-v3-turbo",
        device: Optional[str] = None,
        language: str = "vi",
        initial_batch_size: int = 64,
        beam_size: int = 1,
        best_of: int = 1,
    ):
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model_size = model_size
        self.language = language
        self.beam_size = beam_size
        self.best_of = best_of
        self._batch_size = initial_batch_size
        self.mapper = FrameMapper()
        self.model = None
        self.batched_pipeline = None

    def _lazy_load_model(self):
        if self.model is not None:
            return

        device_type = "cuda" if "cuda" in str(self.device) else "cpu"
        compute_type = "float16" if device_type == "cuda" else "int8"
        device_index = 0
        if "cuda:" in str(self.device):
            try:
                device_index = int(str(self.device).split(":")[-1])
            except ValueError:
                device_index = 0

        print(f"[WhisperASR] Loading faster-whisper '{self.model_size}' on {device_type}:{device_index} ({compute_type})...")
        try:
            from faster_whisper import WhisperModel
            self.model = WhisperModel(
                self.model_size,
                device=device_type,
                device_index=device_index,
                compute_type=compute_type,
                num_workers=2,
                cpu_threads=4,
            )
            print("[WhisperASR] faster-whisper loaded successfully.")
        except Exception as e:
            print(f"[WhisperASR] Note: faster-whisper load failed ({e}), checking vanilla whisper fallback.")
            try:
                import whisper
                self.model = whisper.load_model("base", device=self.device)
                print("[WhisperASR] Fallback to vanilla whisper base.")
            except Exception as e2:
                print(f"[WhisperASR] Running in dummy fallback mode ({e2}).")
                self.model = "dummy"

    def transcribe_video(
        self,
        video_path: str,
        fps: float = 30.0,
    ) -> List[Dict]:
        """
        Transcribes audio from an MP4 video file and returns aligned segments.
        Auto-halves batch size dynamically upon CUDA out-of-memory errors until it succeeds.
        """
        self._lazy_load_model()
        if not os.path.exists(video_path) or self.model == "dummy":
            return []

        # Check if fallback model
        if not hasattr(self.model, "transcribe"):
            return []

        # Handle vanilla whisper fallback if loaded
        if self.model.__class__.__module__.startswith("whisper"):
            try:
                result = self.model.transcribe(
                    video_path,
                    language=self.language,
                    task="transcribe",
                    fp16=("cuda" in str(self.device)),
                )
                segments = []
                video_id = os.path.splitext(os.path.basename(video_path))[0]
                for seg in result.get("segments", []):
                    start_t = float(seg.get("start", 0.0))
                    end_t = float(seg.get("end", 0.0))
                    text = seg.get("text", "").strip()
                    if text:
                        segments.append({
                            "video_id": video_id,
                            "start_sec": start_t,
                            "end_sec": end_t,
                            "start_frame": int(round(start_t * fps)),
                            "end_frame": int(round(end_t * fps)),
                            "text": text,
                        })
                return segments
            except Exception as e:
                print(f"[WhisperASR] Fallback whisper error: {e}")
                return []

        # faster-whisper with adaptive VRAM saturation and OOM auto-halving
        return self._transcribe_with_auto_batch(video_path, fps)

    def _transcribe_with_auto_batch(self, video_path: str, fps: float) -> List[Dict]:
        """
        Executes faster-whisper transcription with dynamic batch halving on OOM.
        Starts at self._batch_size and auto-reduces: 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1.
        """
        batch_size = self._batch_size
        video_id = os.path.splitext(os.path.basename(video_path))[0]

        vad_params = {
            "min_silence_duration_ms": 500,
            "speech_pad_ms": 200,
            "threshold": 0.35,
        }

        while batch_size >= 1:
            try:
                if batch_size > 1:
                    # Use faster-whisper BatchedInferencePipeline for high VRAM parallelism
                    try:
                        from faster_whisper import BatchedInferencePipeline
                        if self.batched_pipeline is None or getattr(self, "_pipeline_model", None) != self.model:
                            self.batched_pipeline = BatchedInferencePipeline(model=self.model)
                            self._pipeline_model = self.model

                        segments_gen, _ = self.batched_pipeline.transcribe(
                            video_path,
                            language=self.language,
                            task="transcribe",
                            beam_size=self.beam_size,
                            best_of=self.best_of,
                            vad_filter=True,
                            vad_parameters=vad_params,
                            batch_size=batch_size,
                            word_timestamps=False,
                            condition_on_previous_text=False,
                        )
                    except (ImportError, AttributeError):
                        # If BatchedInferencePipeline not available, fall back to standard transcribe
                        segments_gen, _ = self.model.transcribe(
                            video_path,
                            language=self.language,
                            task="transcribe",
                            beam_size=self.beam_size,
                            best_of=self.best_of,
                            vad_filter=True,
                            vad_parameters=vad_params,
                            word_timestamps=False,
                            condition_on_previous_text=False,
                        )
                else:
                    # Single-stream execution (minimal VRAM mode)
                    segments_gen, _ = self.model.transcribe(
                        video_path,
                        language=self.language,
                        task="transcribe",
                        beam_size=self.beam_size,
                        best_of=self.best_of,
                        vad_filter=True,
                        vad_parameters=vad_params,
                        word_timestamps=False,
                        condition_on_previous_text=False,
                    )

                segments = []
                for seg in segments_gen:
                    text = seg.text.strip()
                    if not text:
                        continue
                    start_t = float(seg.start)
                    end_t = float(seg.end)
                    segments.append({
                        "video_id": video_id,
                        "start_sec": start_t,
                        "end_sec": end_t,
                        "start_frame": int(round(start_t * fps)),
                        "end_frame": int(round(end_t * fps)),
                        "text": text,
                    })

                # Persist the highest working batch size for subsequent videos
                self._batch_size = batch_size
                return segments

            except (torch.cuda.OutOfMemoryError, RuntimeError, Exception) as e:
                err_str = str(e).lower()
                is_oom = "out of memory" in err_str or "cuda" in err_str or isinstance(e, torch.cuda.OutOfMemoryError)
                if is_oom and batch_size > 1:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    new_bs = max(1, batch_size // 2)
                    print(f"[WhisperASR] ⚠️ GPU OutOfMemory at batch_size={batch_size} -> auto-reducing to batch_size={new_bs} and retrying...")
                    batch_size = new_bs
                else:
                    if is_oom and batch_size <= 1:
                        print(f"[WhisperASR] ❌ OOM even at batch_size=1 on {video_path}")
                        return []
                    print(f"[WhisperASR] Transcription error on {video_path}: {e}")
                    return []

        return []
