import os
os.environ["HF_HUB_DISABLE_FILE_LOCKING"] = "1"

import gc
import torch
from typing import List, Dict, Optional
from src.index.frame_mapper import FrameMapper


import numpy as np

def extract_audio_from_video(video_path: str, target_sr: int = 16000) -> Optional[np.ndarray]:
    """
    Extracts 16kHz mono float32 audio array from an MP4 video file in ~0.2s using PyAV.
    Falls back to ffmpeg subprocess if needed.
    """
    try:
        import av
        container = av.open(video_path)
        if not container.streams.audio:
            container.close()
            return None
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=target_sr)
        chunks = []
        for frame in container.decode(stream):
            resampled = resampler.resample(frame)
            if resampled:
                for r in resampled:
                    chunks.append(r.to_ndarray())
        container.close()
        if chunks:
            arr = np.concatenate(chunks, axis=1)[0].astype(np.float32)
            # Normalize to [-1.0, 1.0] if needed
            max_val = np.max(np.abs(arr))
            if max_val > 1.0:
                arr = arr / max_val
            return arr
        return None
    except Exception:
        pass

    # Fallback: ffmpeg subprocess
    try:
        import subprocess, tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            cmd = ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", str(target_sr), "-ac", "1", tmp.name]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            import soundfile as sf
            audio, sr = sf.read(tmp.name)
            return audio.astype(np.float32)
    except Exception:
        return None


class WhisperASR:
    """
    High-throughput Automatic Speech Recognition for Vietnamese video audio using
    PhoWhisper (HuggingFace) or faster-whisper (CTranslate2).
    """

    def __init__(
        self,
        model_size: str = "vinai/PhoWhisper-small",
        device: Optional[str] = None,
        language: str = "vi",
        initial_batch_size: int = 32,
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
        self.is_hf_pipeline = False

    def _lazy_load_model(self):
        if self.model is not None:
            return

        is_phowhisper = "phowhisper" in self.model_size.lower() or "vinai/" in self.model_size.lower()

        if is_phowhisper:
            hf_model_id = "vinai/PhoWhisper-small" if "small" in self.model_size.lower() else self.model_size
            print(f"[WhisperASR] Loading PhoWhisper '{hf_model_id}' via HuggingFace ASR pipeline on {self.device} (float16)...")
            try:
                from transformers import pipeline
                dtype = torch.float16 if ("cuda" in str(self.device) and torch.cuda.is_available()) else torch.float32
                try:
                    self.model = pipeline(
                        "automatic-speech-recognition",
                        model=hf_model_id,
                        torch_dtype=dtype,
                        device=self.device,
                        model_kwargs={"local_files_only": True},
                    )
                except Exception:
                    self.model = pipeline(
                        "automatic-speech-recognition",
                        model=hf_model_id,
                        torch_dtype=dtype,
                        device=self.device,
                    )
                self.is_hf_pipeline = True
                print("[WhisperASR] PhoWhisper pipeline loaded successfully.")
                return
            except Exception as e:
                print(f"[WhisperASR] HuggingFace pipeline load failed ({e}), checking faster-whisper/ctranslate2...")

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
        """
        self._lazy_load_model()
        if not os.path.exists(video_path) or self.model == "dummy":
            return []

        if self.is_hf_pipeline:
            return self._transcribe_with_hf_pipeline(video_path, fps)

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

    def _transcribe_with_hf_pipeline(self, video_path: str, fps: float) -> List[Dict]:
        """
        Executes PhoWhisper inference via Hugging Face ASR pipeline with batched chunking.
        Extracts 16kHz audio in-memory to prevent soundfile MP4 decoding failures.
        """
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        try:
            audio_array = extract_audio_from_video(video_path, target_sr=16000)
            if audio_array is None or len(audio_array) == 0:
                return []

            pipe_out = self.model(
                {"raw": audio_array, "sampling_rate": 16000},
                chunk_length_s=30,
                stride_length_s=4,
                batch_size=max(8, self._batch_size),
                return_timestamps=True,
                generate_kwargs={"language": "vi", "task": "transcribe"}
            )
            chunks = pipe_out.get("chunks", []) if isinstance(pipe_out, dict) else []
            if not chunks and isinstance(pipe_out, dict) and pipe_out.get("text"):
                chunks = [{"timestamp": (0.0, 0.0), "text": pipe_out.get("text", "")}]

            segments = []
            for ch in chunks:
                ts = ch.get("timestamp", (0.0, 0.0))
                start_t = float(ts[0]) if ts[0] is not None else 0.0
                end_t = float(ts[1]) if (len(ts) > 1 and ts[1] is not None) else start_t + 3.0
                text = ch.get("text", "").strip()
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
            print(f"[WhisperASR] PhoWhisper pipeline error on {video_path}: {e}")
            return []

    def _transcribe_with_auto_batch(self, video_path: str, fps: float) -> List[Dict]:
        """
        Executes faster-whisper transcription with dynamic batch halving on OOM.
        Starts at self._batch_size and auto-reduces: 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1.
        """
        batch_size = self._batch_size
        video_id = os.path.splitext(os.path.basename(video_path))[0]

        vad_params = {
            "min_silence_duration_ms": 500,
            "speech_pad_ms": 100,
            "threshold": 0.50,
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
