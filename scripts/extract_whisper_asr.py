import os
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
    model_size: str = "large-v3"
):
    """
    Transcribes all video audio tracks using Whisper in Vietnamese with atomic writes.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Safe device fallback
    if "cuda" in device and not torch.cuda.is_available():
        device = "cpu"

    asr = WhisperASR(model_size=model_size, device=device, language="vi")
    video_files = sorted(glob.glob(os.path.join(videos_root, "Videos_L*", "video", "*.mp4")))
    print(f"[Whisper ASR] Found {len(video_files)} video files on device {device}.")

    t0 = time.time()
    for vid_path in tqdm(video_files, desc="Running Whisper ASR"):
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

    elapsed = time.time() - t0
    print(f"[Whisper ASR] Transcription complete in {elapsed/3600:.2f} hours.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model", type=str, default="large-v3")
    args = parser.parse_args()

    extract_all_whisper_asr(device=args.device, model_size=args.model)
