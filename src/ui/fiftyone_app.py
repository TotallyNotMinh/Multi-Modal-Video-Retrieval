import os
import glob
import json
import numpy as np
import fiftyone as fo
import fiftyone.brain as fob
from src.index.matrix_builder import FeatureMatrixBuilder
from src.index.object_indexer import ObjectIndexer

def build_fiftyone_dataset(dataset_name: str = "aic_2026_dataset", max_samples: int = 5000):
    """
    Builds or loads a FiftyOne dataset containing keyframe samples, Faster R-CNN detections,
    and precomputed ViT-B/32 CLIP embeddings.
    """
    if fo.dataset_exists(dataset_name):
        print(f"[FiftyOne] Loading existing dataset '{dataset_name}'...")
        dataset = fo.load_dataset(dataset_name)
        return dataset

    print(f"[FiftyOne] Creating new dataset '{dataset_name}'...")
    dataset = fo.Dataset(dataset_name)
    dataset.persistent = True

    matrix_builder = FeatureMatrixBuilder()
    matrix, records = matrix_builder.build_and_cache()
    object_indexer = ObjectIndexer().build_and_cache()

    samples_to_add = []
    embeddings_to_add = []

    # Limit to max_samples if specified for fast launch
    selected_records = records[:max_samples] if max_samples > 0 else records

    print(f"[FiftyOne] Ingesting {len(selected_records)} samples...")
    for i, rec in enumerate(selected_records):
        kf_path = rec["keyframe_path"]
        if not kf_path or not os.path.exists(kf_path):
            continue

        vid = rec["video_id"]
        kf_name = rec["keyframe_name"]

        # Parse object detections
        objs = object_indexer.get_objects(vid, kf_name)
        detections = []
        for o in objs:
            # Box format: [ymin, xmin, ymax, xmax] -> FiftyOne: [xmin, ymin, width, height]
            b = o["box"]
            ymin, xmin, ymax, xmax = b[0], b[1], b[2], b[3]
            w = max(0.0, xmax - xmin)
            h = max(0.0, ymax - ymin)
            detections.append(
                fo.Detection(
                    label=o["class"],
                    bounding_box=[xmin, ymin, w, h],
                    confidence=o["score"]
                )
            )

        sample = fo.Sample(
            filepath=os.path.abspath(kf_path),
            video_id=vid,
            keyframe_name=kf_name,
            frame_idx=rec["frame_idx"],
            pts_time=rec["pts_time"],
            fps=rec["fps"],
            faster_rcnn=fo.Detections(detections=detections)
        )

        samples_to_add.append(sample)
        embeddings_to_add.append(matrix[rec["global_idx"]])

    dataset.add_samples(samples_to_add)

    # Compute similarity index with precomputed embeddings
    print("[FiftyOne] Registering precomputed CLIP embeddings Brain similarity index...")
    fob.compute_similarity(
        dataset,
        embeddings=np.array(embeddings_to_add),
        brain_key="clip_sim",
        model="clip-vit-base32-torch"
    )

    print(f"[FiftyOne] Dataset '{dataset_name}' ready with {len(dataset)} samples.")
    return dataset

def launch_dashboard(dataset_name: str = "aic_2026_dataset", port: int = 5151):
    dataset = build_fiftyone_dataset(dataset_name=dataset_name)
    session = fo.launch_app(dataset, port=port, auto=False)
    print(f"\n[FiftyOne] App is running at http://localhost:{port}")
    return session

if __name__ == "__main__":
    launch_dashboard()
