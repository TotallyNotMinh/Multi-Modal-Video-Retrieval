import os
import glob
import pandas as pd
from typing import Dict, Optional, Tuple

class FrameMapper:
    """
    Handles mapping between keyframe numbers (n or '001'), pts_time, fps, and absolute frame_idx.
    """
    def __init__(self, map_dir: str = "data/map-keyframes-aic25-b1/map-keyframes"):
        if not os.path.exists(map_dir) and os.path.exists("cache/map-keyframes-aic25-b1/map-keyframes"):
            map_dir = "cache/map-keyframes-aic25-b1/map-keyframes"
        self.map_dir = map_dir
        self._cache: Dict[str, pd.DataFrame] = {}

    def _load_video_map(self, video_id: str) -> Optional[pd.DataFrame]:
        if video_id in self._cache:
            return self._cache[video_id]
        csv_path = os.path.join(self.map_dir, f"{video_id}.csv")
        if not os.path.exists(csv_path):
            return None
        df = pd.read_csv(csv_path)
        self._cache[video_id] = df
        return df

    def get_frame_info(self, video_id: str, keyframe_idx_0based: int) -> Dict:
        """
        keyframe_idx_0based: index in the .npy / sorted keyframe list (0, 1, 2...)
        """
        df = self._load_video_map(video_id)
        if df is None or keyframe_idx_0based >= len(df):
            return {
                "n": keyframe_idx_0based + 1,
                "pts_time": float(keyframe_idx_0based),
                "fps": 30.0,
                "frame_idx": keyframe_idx_0based * 30
            }
        row = df.iloc[keyframe_idx_0based]
        return {
            "n": int(row.get("n", keyframe_idx_0based + 1)),
            "pts_time": float(row.get("pts_time", 0.0)),
            "fps": float(row.get("fps", 30.0)),
            "frame_idx": int(row.get("frame_idx", 0))
        }

    def keyframe_name_to_frame_idx(self, video_id: str, kf_name: str) -> int:
        """
        kf_name: e.g. "001" or "001.jpg"
        """
        clean_name = os.path.splitext(kf_name)[0]
        try:
            n_val = int(clean_name)
        except ValueError:
            n_val = 1
        df = self._load_video_map(video_id)
        if df is not None:
            match = df[df["n"] == n_val]
            if len(match) > 0:
                return int(match.iloc[0]["frame_idx"])
            idx_0based = min(n_val - 1, len(df) - 1)
            if idx_0based >= 0:
                return int(df.iloc[idx_0based]["frame_idx"])
        return n_val * 30
