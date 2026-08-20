import os
import gc
import glob
import json
import time
import argparse
import cv2
import torch
import numpy as np
from tqdm import tqdm
from src.encoding.siglip_encoder import SigLIPEncoder
from src.encoding.scene_detector import SceneDetector


def extract_all_siglip_features(
    videos_root: str = "data",
    output_dir: str = "cache/siglip_features",
    meta_dir: str = "cache/siglip_meta",
    device: str = "cuda:0",
    batch_size: int = 256,
    scene_threshold: float = 0.35,

):
    """
    Extracts SigLIP-SO400M embeddings using scene-adaptive shot sampling.

    Replaces fixed 5 FPS stride with content-adaptive shot boundary detection:
      - SceneDetector runs a sequential CPU pass (~1700 fps) to find shot boundaries.
      - Per-shot adaptive policy selects 1–N keyframe indices (no redundant static frames).
      - Frames are decoded sequentially with cap.grab() skipping — no cap.set() seeks,
        which avoids I-frame snap errors in H.264 streams.

    Metadata schema (backward-compatible with all downstream consumers):
      Required: video_id, frame_idx, pts_time, fps
      Optional: shot_id, shot_start_frame, shot_end_frame  (additive, never removed)
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    if "cuda" in device and not torch.cuda.is_available():
        device = "cpu"

    encoder = SigLIPEncoder(device=device, use_fp16=True)
    detector = SceneDetector(threshold=scene_threshold)

    video_files = sorted(glob.glob(os.path.join(videos_root, "Videos_L*", "video", "*.mp4")))
    print(f"[SigLIP Extraction] Found {len(video_files)} video files. Scene-adaptive sampling on {device}.")

    total_frames_extracted = 0
    t0 = time.time()

    for vid_path in tqdm(video_files, desc="Extracting SigLIP (scene-adaptive)"):
        vid_name = os.path.splitext(os.path.basename(vid_path))[0]
        out_npy = os.path.join(output_dir, f"{vid_name}.npy")
        out_meta = os.path.join(meta_dir, f"{vid_name}.json")

        if os.path.exists(out_npy) and os.path.exists(out_meta):
            continue

        # --- Pass 1: Scene detection (sequential, CPU, ~1700 fps) ---
        shots = detector.detect_shots(vid_path)
        if not shots:
            continue

        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            continue

        frames_batch = []
        meta_batch = []
        all_embeddings = []
        all_meta = []

        try:
            raw_fps = cap.get(cv2.CAP_PROP_FPS)
            orig_fps = float(raw_fps) if (raw_fps and not np.isnan(raw_fps) and raw_fps > 0) else 30.0

            # Build a sorted set of (frame_idx -> shot_dict) for O(1) lookup
            target_map: dict[int, dict] = {}
            for shot in shots:
                for f in detector.get_sample_frames(shot, orig_fps):
                    target_map[f] = shot

            if not target_map:
                continue

            sorted_targets = sorted(target_map.keys())
            target_iter = iter(sorted_targets)
            next_target = next(target_iter, None)

            curr_frame = 0
            # --- Pass 2: Sequential decode, grab()-skip non-target frames ---
            while next_target is not None:
                if curr_frame < next_target:
                    if not cap.grab():
                        break
                    curr_frame += 1
                    continue

                ret, frame = cap.read()
                if not ret:
                    break

                shot = target_map[curr_frame]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_resized = cv2.resize(rgb, (384, 384), interpolation=cv2.INTER_AREA)
                frames_batch.append(rgb_resized)
                meta_batch.append({
                    # Required fields (backward-compatible schema)
                    "video_id": vid_name,
                    "frame_idx": curr_frame,
                    "pts_time": curr_frame / orig_fps,
                    "fps": orig_fps,
                    # Additive shot fields (never replace existing keys)
                    "shot_id": shot["shot_id"],
                    "shot_start_frame": shot["start_frame"],
                    "shot_end_frame": shot["end_frame"],
                })

                next_target = next(target_iter, None)
                curr_frame += 1

                # Flush to GPU when batch is full
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

            tmp_npy = f"{out_npy}.tmp.{os.getpid()}"
            tmp_meta = f"{out_meta}.tmp.{os.getpid()}"

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
    parser.add_argument("--scene-threshold", type=float, default=0.35,

                        help="Histogram correlation drop threshold for scene cut detection (0-1).")
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
                scene_threshold=args.scene_threshold,
            )
    else:
        extract_all_siglip_features(
            device=args.device,
            batch_size=args.batch_size,
            scene_threshold=args.scene_threshold,
        )
