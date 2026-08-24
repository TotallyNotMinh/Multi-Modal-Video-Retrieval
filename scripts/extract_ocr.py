import os
# Strict single-thread clamping to keep laptop CPU cool
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["PADDLE_DISABLE_STATIC_MEMORY"] = "1"

import sys
import glob
import json
import time
import argparse
import cv2
import torch
from tqdm import tqdm

# Cap thread pools for OpenCV and PyTorch to single-thread
cv2.setNumThreads(1)
try:
    torch.set_num_threads(1)
except Exception:
    pass

# Ensure repo root is always in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.encoding.ocr_extractor import OCRExtractor


def extract_all_ocr(
    videos_root: str = "data",
    output_dir: str = "cache/ocr_text",
    device: str = "cuda:0",
    sample_interval_sec: float = 3.0,
    num_shards: int = 1,
    shard_id: int = 0,
    overwrite: bool = False,
):
    """
    Extracts on-screen text banners directly from MP4 videos every N seconds (e.g., 3.0s).
    Completely eliminates the need for 30GB of keyframe JPG files.
    Supports multi-GPU sharding across isolated workers.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if "cuda" in device and not torch.cuda.is_available():
        device = "cpu"

    ocr = OCRExtractor(device=device)
    video_files = sorted(glob.glob(os.path.join(videos_root, "Videos_L*", "video", "*.mp4")))
    total_all = len(video_files)
    if num_shards > 1:
        video_files = [f for idx, f in enumerate(video_files) if idx % num_shards == shard_id]

    print(f"[OCR Extraction] Found {len(video_files)}/{total_all} video files (Shard {shard_id}/{num_shards}) on device {device}.")

    # Epoch timestamp for start of VietOCR run (Aug 24, 2026 02:00:00)
    VIETOCR_CUTOFF_TIMESTAMP = 1787511600

    t0 = time.time()
    for vid_path in tqdm(video_files, desc=f"Running OCR Shard {shard_id}"):
        vid_name = os.path.splitext(os.path.basename(vid_path))[0]
        out_json = os.path.join(output_dir, f"{vid_name}.json")

        # Skip only if file was successfully extracted with new VietOCR today
        is_new_vietocr = (
            os.path.exists(out_json)
            and os.path.getsize(out_json) > 10
            and os.path.getmtime(out_json) >= VIETOCR_CUTOFF_TIMESTAMP
        )
        if is_new_vietocr and not overwrite:
            continue

        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            continue

        ocr_results = {}
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1000
            frame_stride = max(1, int(round(fps * sample_interval_sec)))
            target_frame_indices = list(range(0, total_frames, frame_stride))

            for f_idx in target_frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                text = ocr.extract_text_from_frame(rgb)
                if text:
                    ocr_results[f"f_{f_idx}"] = text
                del frame
                del rgb
                # Pacing sleep to allow CPU to downclock and stay cool
                time.sleep(0.015)
        finally:
            cap.release()

        # Atomic write
        tmp_json = f"{out_json}.tmp.{os.getpid()}"
        with open(tmp_json, "w", encoding="utf-8") as f:
            json.dump(ocr_results, f, indent=2, ensure_ascii=False)
        os.replace(tmp_json, out_json)

        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed = time.time() - t0

    print(f"[OCR Extraction Shard {shard_id}] Finished in {elapsed/60:.2f} minutes.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--interval", type=float, default=3.0)
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of GPU shards.")
    parser.add_argument("--shard-id", type=int, default=0, help="Current shard ID (0-indexed).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing extracted JSONs.")
    args = parser.parse_args()

    extract_all_ocr(
        device=args.device,
        sample_interval_sec=args.interval,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        overwrite=args.overwrite,
    )

