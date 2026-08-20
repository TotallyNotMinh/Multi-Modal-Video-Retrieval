import os
import glob
import json
import time
import argparse
import cv2
import torch
from tqdm import tqdm
from src.encoding.ocr_extractor import OCRExtractor

def extract_all_ocr(
    videos_root: str = "data",
    output_dir: str = "cache/ocr_text",
    device: str = "cuda:0",
    sample_interval_sec: float = 3.0
):
    """
    Extracts on-screen text banners directly from MP4 videos every N seconds (e.g., 3.0s).
    Completely eliminates the need for 30GB of keyframe JPG files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if "cuda" in device and not torch.cuda.is_available():
        device = "cpu"

    ocr = OCRExtractor(device=device)
    video_files = sorted(glob.glob(os.path.join(videos_root, "Videos_L*", "video", "*.mp4")))
    print(f"[OCR Extraction] Found {len(video_files)} video files on device {device}.")

    t0 = time.time()
    for vid_path in tqdm(video_files, desc="Running Video OCR"):
        vid_name = os.path.splitext(os.path.basename(vid_path))[0]
        out_json = os.path.join(output_dir, f"{vid_name}.json")

        if os.path.exists(out_json):
            continue

        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            continue

        ocr_results = {}
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_stride = max(1, int(round(fps * sample_interval_sec)))
            curr_frame = 0

            while True:
                if curr_frame % frame_stride != 0:
                    if not cap.grab():
                        break
                else:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Pass in-memory RGB array directly to OCR
                    text = ocr.extract_text_from_frame(rgb)
                    if text:
                        ocr_results[f"f_{curr_frame}"] = text

                curr_frame += 1
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

    print(f"[OCR Extraction] Finished in {elapsed/60:.2f} minutes.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--interval", type=float, default=3.0)
    args = parser.parse_args()

    extract_all_ocr(device=args.device, sample_interval_sec=args.interval)
