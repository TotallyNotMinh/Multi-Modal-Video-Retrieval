import os
import glob
import json
import pickle
from typing import Dict, List, Set
from collections import defaultdict

class ObjectIndexer:
    """
    Parses Faster R-CNN OpenImages V4 object detections and builds:
    1. Direct lookup: (video_id, kf_name) -> list of detected entities with scores & boxes
    2. Inverted index: entity_name -> dict of {(video_id, kf_name): max_score}
    """
    def __init__(
        self,
        objects_dir: str = "data/objects-aic25-b1/objects",
        cache_dir: str = "cache",
        min_conf: float = 0.3
    ):
        self.objects_dir = objects_dir
        self.cache_dir = cache_dir
        self.min_conf = min_conf
        os.makedirs(cache_dir, exist_ok=True)
        self.direct_index: Dict[str, Dict[str, List[Dict]]] = defaultdict(dict)
        self.inverted_index: Dict[str, Dict[str, float]] = defaultdict(dict)

    def build_and_cache(self, force: bool = False):
        cache_path = os.path.join(self.cache_dir, "objects_index.pkl")
        if not force and os.path.exists(cache_path):
            print(f"[ObjectIndexer] Loading cached object index from {cache_path}...")
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
                self.direct_index = data["direct"]
                self.inverted_index = data["inverted"]
            print(f"[ObjectIndexer] Loaded {len(self.inverted_index)} distinct object classes.")
            return self

        print("[ObjectIndexer] Building object index from json files...")
        json_files = glob.glob(os.path.join(self.objects_dir, "*", "*.json"))
        print(f"Found {len(json_files)} object json files.")

        for jf in json_files:
            parts = jf.replace("\\", "/").split("/")
            video_id = parts[-2]
            kf_name = os.path.splitext(parts[-1])[0]  # e.g. "0001" or "001"

            try:
                with open(jf, "r", encoding="utf-8") as f:
                    det_data = json.load(f)
            except Exception:
                continue

            classes = det_data.get("detection_class_entities", [])
            boxes = det_data.get("detection_boxes", [])
            scores = det_data.get("detection_scores", [])

            dets = []
            key = f"{video_id}/{kf_name}"

            for cls, box, score in zip(classes, boxes, scores):
                try:
                    s = float(score)
                    if s >= self.min_conf:
                        cls_norm = cls.strip().lower()
                        dets.append({
                            "class": cls.strip(),
                            "class_lower": cls_norm,
                            "box": [float(b) for b in box],
                            "score": s
                        })
                        if key not in self.inverted_index[cls_norm] or s > self.inverted_index[cls_norm][key]:
                            self.inverted_index[cls_norm][key] = s
                except (ValueError, TypeError):
                    continue

            self.direct_index[video_id][kf_name] = dets

        with open(cache_path, "wb") as f:
            pickle.dump({"direct": self.direct_index, "inverted": self.inverted_index}, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"[ObjectIndexer] Saved cache to {cache_path}")
        return self

    def get_objects(self, video_id: str, kf_name: str) -> List[Dict]:
        return self.direct_index.get(video_id, {}).get(kf_name, [])

    def search_entity(self, entity_name: str) -> Dict[str, float]:
        entity_norm = entity_name.strip().lower()
        return self.inverted_index.get(entity_norm, {})
