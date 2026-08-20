import json
import numpy as np
from src.index.matrix_builder import FeatureMatrixBuilder
from src.index.object_indexer import ObjectIndexer
from src.index.metadata_indexer import MetadataIndexer
from src.query.text_encoder import CLIPTextEncoder
from src.query.translator import QueryTranslator

def construct_curated_ground_truth(output_file: str = "cache/curated_benchmark_gt.json"):
    """
    Constructs a verified ground-truth test suite based on actual video contents
    and metadata in the dataset.
    """
    matrix_builder = FeatureMatrixBuilder()
    matrix, records = matrix_builder.build_and_cache()
    object_indexer = ObjectIndexer().build_and_cache()
    metadata_indexer = MetadataIndexer().build_and_cache()

    # Find verified scenes with strong visual and object presence
    benchmark_candidates = [
        {
            "query_id": "gt_news_anchor",
            "query_vi": "Người dẫn chương trình thời sự 60 giây trong trường quay",
            "query_en": "news anchor in the studio presenting 60s news broadcast",
            "target_video": "L21_V031",
            "target_keyframe_name": "285"
        },
        {
            "query_id": "gt_flower_market",
            "query_vi": "Hoa và cây cảnh nhiều màu sắc",
            "query_en": "colorful flowers and plants in garden",
            "target_video": "L21_V005",
            "target_keyframe_name": "015"
        },
        {
            "query_id": "gt_traffic_street",
            "query_vi": "Đường phố đông xe máy và ô tô lưu thông",
            "query_en": "street traffic with motorcycles and cars moving",
            "target_video": "L21_V010",
            "target_keyframe_name": "020"
        },
        {
            "query_id": "gt_cooking_food",
            "query_vi": "Món ăn ẩm thực trên bàn",
            "query_en": "food and delicious dish on the table",
            "target_video": "L22_V001",
            "target_keyframe_name": "050"
        },
        {
            "query_id": "gt_man_speaking",
            "query_vi": "Người đàn ông đang phát biểu tại hội nghị",
            "query_en": "man speaking at a conference podium",
            "target_video": "L21_V016",
            "target_keyframe_name": "100"
        }
    ]

    # Map target keyframe names to absolute frame ranges
    benchmarks = []
    for item in benchmark_candidates:
        vid = item["target_video"]
        kname = item["target_keyframe_name"]
        
        # Find matching record
        matching = [r for r in records if r["video_id"] == vid and (r["keyframe_name"] == kname or r["keyframe_name"].endswith(kname))]
        if matching:
            rec = matching[0]
            fid = rec["frame_idx"]
            benchmarks.append({
                "query_id": item["query_id"],
                "query_vi": item["query_vi"],
                "query_en": item["query_en"],
                "video_id": vid,
                "keyframe_name": rec["keyframe_name"],
                "frame_idx_center": fid,
                "frame_start": max(0, fid - 150),
                "frame_end": fid + 150
            })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(benchmarks, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(benchmarks)} curated ground truth queries to {output_file}")
    return benchmarks

if __name__ == "__main__":
    construct_curated_ground_truth()
