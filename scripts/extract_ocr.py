import os
import glob
import json
import time
import argparse
import torch
from tqdm import tqdm
from src.encoding.ocr_extractor import OCRExtractor

def extract_all_ocr(
    keyframes_root: str = "data",
    output_dir: str = "cache/ocr_text",
    device: str = "cuda:0"
):
    """
    Extracts on-screen text from all keyframe images with atomic writes.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if "cuda" in device and not torch.cuda.is_available():
        device = "cpu"

    ocr = OCRExtractor(device=device)
    video_dirs = sorted(glob.glob(os.path.join(keyframes_root, "Keyframes_L*", "keyframes", "*")))
    print(f"[OCR Extraction] Found {len(video_dirs)} video keyframe folders on device {device}.")

    t0 = time.time()
    for vdir in tqdm(video_dirs, desc="Running Keyframe OCR"):
        vid_name = os.path.basename(vdir)
        out_json = os.path.join(output_dir, f"{vid_name}.json")

        if os.path.exists(out_json):
            continue

        kf_paths = sorted(glob.glob(os.path.join(vdir, "*.jpg")))
        ocr_res = ocr.batch_extract_video_keyframes(kf_paths)

        # Atomic write
        tmp_json = f"{out_json}.tmp.{os.getpid()}"
        with open(tmp_json, "w", encoding="utf-8") as f:
            json.dump(ocr_res, f, indent=2, ensure_ascii=False)
        os.replace(tmp_json, out_json)

    elapsed = time.time() - t0
    print(f"[OCR Extraction] Finished in {elapsed/60:.2f} minutes.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    extract_all_ocr(device=args.device)
