import os
import sys
import gc
import glob
import json
import time
import argparse
import cv2
import torch
import numpy as np
from tqdm import tqdm

# Ensure repo root is always in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.encoding.siglip_encoder import SigLIPEncoder
from src.encoding.scene_detector import SceneDetector



def extract_all_siglip_features(
    videos_root: str = "data",
    output_dir: str = "cache/siglip_features",
    meta_dir: str = "cache/siglip_meta",
    device: str = "cuda:0",
    batch_size: int = 256,
    sample_interval_sec: float = 1.5,
    num_shards: int = 1,
    shard_id: int = 0,
):
    """
    Extracts SigLIP-SO400M embeddings using high-speed single-pass keyframe sampling.
    Decodes ~800 frames per 20-min video via fast cap.grab() strides (~5s per video).
    Supports multi-GPU sharding across isolated workers.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    if "cuda" in device and not torch.cuda.is_available():
        device = "cpu"

    encoder = SigLIPEncoder(device=device, use_fp16=True)

    video_files = sorted(glob.glob(os.path.join(videos_root, "Videos_L*", "video", "*.mp4")))
    total_all = len(video_files)
    if num_shards > 1:
        video_files = [f for idx, f in enumerate(video_files) if idx % num_shards == shard_id]

    print(f"[SigLIP Extraction] Found {len(video_files)}/{total_all} video files (Shard {shard_id}/{num_shards}) on {device}.")

    total_frames_extracted = 0
    t0 = time.time()

    for vid_path in tqdm(video_files, desc=f"Extracting SigLIP Shard {shard_id}"):
        vid_name = os.path.splitext(os.path.basename(vid_path))[0]
        out_npy = os.path.join(output_dir, f"{vid_name}.npy")
        out_meta = os.path.join(meta_dir, f"{vid_name}.json")

        if os.path.exists(out_npy) and os.path.exists(out_meta):
            continue

        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            continue

        frames_batch = []
        meta_batch = []
        all_embeddings = []
        all_meta = []
        curr_frame = 0
        shot_id = 0
        try:
            raw_fps = cap.get(cv2.CAP_PROP_FPS)
            orig_fps = float(raw_fps) if (raw_fps and not np.isnan(raw_fps) and raw_fps > 0) else 30.0
            frame_stride = max(1, int(round(orig_fps * sample_interval_sec)))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                total_frames = int(orig_fps * 1800)

            frame_indices = list(range(0, total_frames, frame_stride))

            for f_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                ret, frame = cap.read()
                if not ret:
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_resized = cv2.resize(rgb, (384, 384), interpolation=cv2.INTER_AREA)
                frames_batch.append(rgb_resized)
                meta_batch.append({
                    "video_id": vid_name,
                    "frame_idx": f_idx,
                    "pts_time": f_idx / orig_fps,
                    "fps": orig_fps,
                    "shot_id": shot_id,
                    "shot_start_frame": max(0, f_idx - frame_stride // 2),
                    "shot_end_frame": f_idx + frame_stride // 2,
                })
                shot_id += 1

                if len(frames_batch) >= batch_size:
                    emb = encoder.encode_images(frames_batch, batch_size=batch_size)
                    all_embeddings.append(emb)
                    all_meta.extend(meta_batch)
                    frames_batch = []
                    meta_batch = []

            # Flush remaining frames
            if frames_batch:
                emb = encoder.encode_images(frames_batch, batch_size=batch_size)
                all_embeddings.append(emb)
                all_meta.extend(meta_batch)

        finally:
            cap.release()

        if all_embeddings:
            embeddings = np.vstack(all_embeddings)

            tmp_npy = f"{out_npy}.tmp.{os.getpid()}.npy"
            tmp_meta = f"{out_meta}.tmp.{os.getpid()}.json"

            np.save(tmp_npy, embeddings)
            with open(tmp_meta, "w", encoding="utf-8") as f:
                json.dump(all_meta, f)

            os.replace(tmp_npy, out_npy)
            os.replace(tmp_meta, out_meta)

            total_frames_extracted += len(all_meta)

        del all_embeddings, all_meta, frames_batch, meta_batch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"\n[SigLIP Extraction] Finished! Extracted {total_frames_extracted} frames "
          f"(scene-adaptive) in {elapsed / 60:.2f} minutes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--interval", "--sample-interval-sec", type=float, default=1.5,
                        help="Sampling interval in seconds between keyframes.")
    parser.add_argument("--scene-threshold", type=float, default=0.35, help="Legacy alias.")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of GPU shards.")
    parser.add_argument("--shard-id", type=int, default=0, help="Current shard ID (0-indexed).")
    parser.add_argument("--video-sample", type=str, default=None,
                        help="If set, run on a single video file for debugging.")
    args = parser.parse_args()

    if args.video_sample:
        # Single-video debug mode
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            vid_dir = os.path.dirname(args.video_sample)
            # Create a synthetic videos_root structure
            extract_all_siglip_features(
                videos_root=os.path.join(os.path.dirname(vid_dir), ".."),
                output_dir="cache/siglip_features",
                meta_dir="cache/siglip_meta",
                device=args.device,
                batch_size=args.batch_size,
                sample_interval_sec=args.interval,
                num_shards=args.num_shards,
                shard_id=args.shard_id,
            )
    else:
        extract_all_siglip_features(
            device=args.device,
            batch_size=args.batch_size,
            sample_interval_sec=args.interval,
            num_shards=args.num_shards,
            shard_id=args.shard_id,
        )
