import os
import glob
import pickle
import numpy as np
from typing import List, Dict, Tuple
from src.index.frame_mapper import FrameMapper

class FeatureMatrixBuilder:
    """
    Consolidates individual video .npy CLIP feature files into a unified,
    L2-normalized feature matrix and builds a global keyframe lookup table.
    """
    def __init__(
        self,
        clip_dir: str = "data/clip-features-32",
        keyframes_root: str = "data",
        cache_dir: str = "cache"
    ):
        self.clip_dir = clip_dir
        self.keyframes_root = keyframes_root
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.mapper = FrameMapper()

    def find_keyframe_paths(self, video_id: str) -> List[str]:
        """
        Locates keyframe jpgs for a given video across all Keyframes_L* folders.
        """
        pattern = os.path.join(self.keyframes_root, "Keyframes_L*", "keyframes", video_id, "*.jpg")
        files = sorted(glob.glob(pattern))
        return files

    def build_and_cache(self, force: bool = False) -> Tuple[np.ndarray, List[Dict]]:
        matrix_path = os.path.join(self.cache_dir, "features_matrix.npy")
        meta_path = os.path.join(self.cache_dir, "metadata.pkl")

        if not force and os.path.exists(matrix_path) and os.path.exists(meta_path):
            print(f"[FeatureMatrixBuilder] Loading cached matrix from {matrix_path}...")
            matrix = np.load(matrix_path)
            with open(meta_path, "rb") as f:
                records = pickle.load(f)
            print(f"[FeatureMatrixBuilder] Loaded {matrix.shape[0]} keyframes, dim={matrix.shape[1]}")
            return matrix, records

        print("[FeatureMatrixBuilder] Building unified matrix from scratch...")
        npy_files = sorted(glob.glob(os.path.join(self.clip_dir, "*.npy")))
        print(f"Found {len(npy_files)} video feature files.")

        all_vectors = []
        all_records = []
        global_idx = 0

        for npy_file in npy_files:
            video_id = os.path.splitext(os.path.basename(npy_file))[0]
            feats = np.load(npy_file).astype(np.float32)  # (N_kf, 512)
            
            # L2 normalize each feature vector
            norms = np.linalg.norm(feats, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            feats_norm = feats / norms

            kf_paths = self.find_keyframe_paths(video_id)
            n_frames = feats.shape[0]

            for i in range(n_frames):
                kf_path = kf_paths[i] if i < len(kf_paths) else ""
                kf_name = os.path.splitext(os.path.basename(kf_path))[0] if kf_path else f"{i+1:04d}"
                
                frame_info = self.mapper.get_frame_info(video_id, i)

                record = {
                    "global_idx": global_idx,
                    "video_id": video_id,
                    "local_kf_idx": i,
                    "keyframe_name": kf_name,
                    "keyframe_path": kf_path,
                    "n": frame_info["n"],
                    "pts_time": frame_info["pts_time"],
                    "fps": frame_info["fps"],
                    "frame_idx": frame_info["frame_idx"]
                }

                all_vectors.append(feats_norm[i])
                all_records.append(record)
                global_idx += 1

        matrix = np.vstack(all_vectors).astype(np.float32)
        print(f"[FeatureMatrixBuilder] Built matrix shape: {matrix.shape}")

        np.save(matrix_path, matrix)
        with open(meta_path, "wb") as f:
            pickle.dump(all_records, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"[FeatureMatrixBuilder] Saved cache to {matrix_path} and {meta_path}")
        return matrix, all_records
