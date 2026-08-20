import os
import gc
import glob
import json
import time
import argparse
import torch
from tqdm import tqdm
from src.encoding.whisper_asr import WhisperASR


def extract_all_whisper_asr(
    videos_root: str = "data",
    output_dir: str = "cache/asr_transcripts",
    device: str = "cuda:0",
    model_size: str = "large-v3-turbo",
    batch_size: int = 32,
    video_sample: str = None,
):
    """
    Transcribes video audio tracks into Vietnamese timestamped segments using
    faster-whisper large-v3-turbo + FP16 + Silero VAD + Auto-Batch OOM recovery.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Safe device fallback
    if "cuda" in device and not torch.cuda.is_available():
        device = "cpu"

    asr = WhisperASR(
        model_size=model_size,
        device=device,
        language="vi",
        initial_batch_size=batch_size,
    )

    if video_sample and os.path.exists(video_sample):
        video_files = [video_sample]
    else:
        video_files = sorted(glob.glob(os.path.join(videos_root, "Videos_L*", "video", "*.mp4")))

    print(f"[Whisper ASR] Found {len(video_files)} video files on device {device} using '{model_size}' (initial batch={batch_size}).")

    t0 = time.time()
    for vid_path in tqdm(video_files, desc="Running faster-whisper ASR"):
        vid_name = os.path.splitext(os.path.basename(vid_path))[0]
        out_json = os.path.join(output_dir, f"{vid_name}.json")

        if os.path.exists(out_json):
            continue

        segments = asr.transcribe_video(vid_path)

        # Atomic write
        tmp_json = f"{out_json}.tmp.{os.getpid()}"
        with open(tmp_json, "w", encoding="utf-8") as f:
            json.dump(segments, f, indent=2, ensure_ascii=False)
        os.replace(tmp_json, out_json)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"[Whisper ASR] Transcription complete in {elapsed/60:.2f} minutes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Vietnamese ASR with faster-whisper large-v3-turbo + VAD.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model-size", type=str, default="large-v3-turbo")
    parser.add_argument("--batch-size", type=int, default=32, help="Initial batch size (halves on OOM).")
    parser.add_argument("--video-sample", type=str, default=None, help="Run on a single video file.")
    args = parser.parse_args()

    extract_all_whisper_asr(
        device=args.device,
        model_size=args.model_size,
        batch_size=args.batch_size,
        video_sample=args.video_sample,
    )
