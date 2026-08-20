import itertools
import cv2
import numpy as np
from typing import List, Dict


class SceneDetector:
    """
    High-throughput shot boundary detector using HSV color histogram correlation.

    Processes video sequentially (no cap.set() seeks) at ~1700+ fps on CPU.
    Detects hard cuts by comparing consecutive frame histograms.

    After detection, get_sample_frames() applies an adaptive sampling policy
    to select 1–N representative keyframe indices per shot:
      - Short shots  (< 1.5s): 1 frame at 50% of shot
      - Medium shots (1.5–5s): 2 frames at 25% and 75%
      - Long shots   (> 5.0s): 3 anchors (10%, 50%, 90%) + 1 frame every 3s
    """

    def __init__(self, threshold: float = 0.35, min_shot_frames: int = 3):
        """
        Args:
            threshold:        Histogram correlation drop to trigger a scene cut.
                              correlation_drop = 1 - corr_score, cut fires when drop > threshold.
                              Range 0–1. Lower = more sensitive. Default 0.35 works well for news.
            min_shot_frames:  Minimum frames a shot must contain to be registered.
                              Prevents spurious single-frame flash cuts from fragmenting shots.
        """
        self.threshold = threshold
        self.min_shot_frames = min_shot_frames

    def detect_shots(self, video_path: str) -> List[Dict]:
        """
        Runs a full sequential pass over the video and returns a list of shots.

        Each shot dict contains:
            shot_id         (int)   — 0-based index of the shot
            start_frame     (int)   — inclusive first frame of the shot
            end_frame       (int)   — inclusive last frame of the shot
            duration_sec    (float) — wall-clock duration of the shot

        Returns an empty list if the video cannot be opened.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        raw_fps = cap.get(cv2.CAP_PROP_FPS)
        fps = float(raw_fps) if (raw_fps and not np.isnan(raw_fps) and raw_fps > 0) else 30.0

        shots: List[Dict] = []
        shot_start = 0
        prev_hist = None
        curr_frame = 0

        try:
            for curr_frame in itertools.count():
                ret, frame = cap.read()
                if not ret:
                    # Flush last shot
                    if curr_frame > shot_start and (curr_frame - shot_start) >= self.min_shot_frames:
                        shots.append(self._make_shot(len(shots), shot_start, curr_frame - 1, fps))
                    break

                # Compute HSV histogram on a 128×72 thumbnail (fast)
                small = cv2.resize(frame, (128, 72), interpolation=cv2.INTER_NEAREST)
                hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
                cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

                if prev_hist is not None:
                    corr = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                    drop = 1.0 - corr
                    shot_len = curr_frame - shot_start
                    if drop > self.threshold and shot_len >= self.min_shot_frames:
                        shots.append(self._make_shot(len(shots), shot_start, curr_frame - 1, fps))
                        shot_start = curr_frame

                prev_hist = hist

            # Fallback: If video has frames but no shots registered (e.g. extremely short video < min_shot_frames)
            if not shots and curr_frame > 0:
                shots.append(self._make_shot(0, 0, max(0, curr_frame - 1), fps))
        finally:
            cap.release()

        return shots


    def get_sample_frames(self, shot: Dict, fps: float) -> List[int]:
        """
        Adaptive frame-index sampling policy for a given shot.

        Returns a sorted list of frame indices to extract from this shot.
        All returned indices are guaranteed to be within [start_frame, end_frame].
        """
        s = shot["start_frame"]
        e = shot["end_frame"]
        T = shot["duration_sec"]

        if e <= s:
            return [s]

        span = e - s

        if T < 1.5:
            # 1 frame: midpoint
            return [s + span // 2]

        elif T <= 5.0:
            # 2 frames: 25% and 75%
            return [s + span // 4, s + 3 * span // 4]

        else:
            # 3 anchors at 10%, 50%, 90%
            anchors = [
                s + max(0, int(0.10 * span)),
                s + span // 2,
                s + min(span, int(0.90 * span)),
            ]
            # 1 intermediate frame every 3 seconds between anchors
            step = max(1, int(3.0 * fps))
            intermediate = list(range(anchors[0] + step, anchors[2], step))
            return sorted(set(anchors + intermediate))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_shot(shot_id: int, start: int, end: int, fps: float) -> Dict:
        return {
            "shot_id": shot_id,
            "start_frame": start,
            "end_frame": end,
            "duration_sec": max(0.0, (end - start) / fps),
        }
