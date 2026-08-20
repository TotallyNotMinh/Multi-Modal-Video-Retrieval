import os
import cv2
import numpy as np
import torch
from typing import Tuple, List, Dict, Optional

class VideoDecoder:
    """
    Stage 2 Exact Frame Localizer.
    Decodes raw .mp4 video around Stage 1 candidate timestamps at full frame rate (30 fps),
    evaluates visual similarity on every continuous frame, and pinpoints the exact peak [s, e] frame index.
    """
    def __init__(self, encoder, device: str = "cuda:0"):
        self.encoder = encoder
        self.device = device

    def localize_exact_frame(
        self,
        video_path: str,
        candidate_pts_time: float,
        query_vec: np.ndarray,
        window_seconds: float = 8.0,
        sample_stride: int = 1
    ) -> Tuple[int, float, float]:
        """
        Extracts continuous frames in [pts - window, pts + window] and finds argmax frame_idx.
        Returns (best_frame_idx, best_score, best_pts_time).
        """
        if not os.path.exists(video_path):
            return 0, 0.0, candidate_pts_time

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0, 0.0, candidate_pts_time

        frames_rgb = []
        frame_indices = []
        frame_pts = []
        start_frame = 0

        try:
            raw_fps = cap.get(cv2.CAP_PROP_FPS)
            fps = float(raw_fps) if (raw_fps and not np.isnan(raw_fps) and raw_fps > 0) else 30.0
            total_frames = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))

            start_sec = max(0.0, candidate_pts_time - window_seconds)
            end_sec = min(total_frames / fps, candidate_pts_time + window_seconds)

            start_frame = max(0, min(int(start_sec * fps), total_frames - 1))
            end_frame = max(start_frame, min(int(end_sec * fps), total_frames - 1))

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            current_frame = start_frame
            while current_frame <= end_frame:
                ret, frame = cap.read()
                if not ret:
                    break

                if (current_frame - start_frame) % max(1, sample_stride) == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Resize to 384x384 to save RAM & match encoder resolution
                    rgb_resized = cv2.resize(rgb, (384, 384), interpolation=cv2.INTER_AREA)
                    frames_rgb.append(rgb_resized)
                    frame_indices.append(current_frame)
                    frame_pts.append(current_frame / fps)

                current_frame += 1
        finally:
            cap.release()

        if not frames_rgb:
            return start_frame, 0.0, candidate_pts_time

        # Batch encode frames
        frame_embeddings = self.encoder.encode_images(frames_rgb, batch_size=64)
        
        # Calculate cosine similarity with query_vec
        q = np.squeeze(query_vec)
        scores = np.dot(frame_embeddings.astype(np.float32), q.astype(np.float32))

        best_idx_local = int(np.argmax(scores))
        best_frame_idx = frame_indices[best_idx_local]
        best_score = float(scores[best_idx_local])
        best_pts = frame_pts[best_idx_local]

        return best_frame_idx, best_score, best_pts
