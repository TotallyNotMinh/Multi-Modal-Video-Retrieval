#!/usr/bin/env python3
"""
Diagnostic Spot-Check for Pure Visual CLIP Pipeline:
1. Inspect 8 VISUAL_HYBRID queries.
2. Check translation, prompts, dot product distribution, and retrieved keyframe timestamps vs Ground Truth.
"""

import os
import sys
import json
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.eval_retrieval import SearchEngine, is_ground_truth_hit

def main():
    benchmark_file = "eval/vietnamese_retrieval_benchmark_stage2_rigorous.jsonl"
    with open(benchmark_file, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    vh_queries = [q for q in dataset if q.get("category") == "VISUAL_HYBRID"][:8]
    engine = SearchEngine()

    print("\n" + "="*90, flush=True)
    print("  VISUAL CLIP SPOT-CHECK: INSPECTING 8 VISUAL_HYBRID QUERIES", flush=True)
    print("="*90, flush=True)

    for idx, item in enumerate(vh_queries, start=1):
        q_vi = item["query"]
        gt_list = item.get("relevant_segments", [])
        gt_vid = gt_list[0]["video_id"] if gt_list else "N/A"
        gt_st = gt_list[0]["start_sec"] if gt_list else 0.0
        gt_et = gt_list[0]["end_sec"] if gt_list else 0.0

        # Translate query & encode prompts
        q_en = engine.translator.translate(q_vi)
        prompts = engine.translator.generate_prompts(q_en)
        q_vec = engine.encoder.encode_text(prompts, ensemble=True)
        scores = np.dot(engine.matrix, q_vec)

        # Top 5 pure visual keyframe indices
        top_k_idx = np.argsort(scores)[::-1][:5]

        print(f"\n[{idx}] Query (VI): \"{q_vi}\"", flush=True)
        print(f"    Translated: \"{q_en}\"", flush=True)
        print(f"    Target GT : Video {gt_vid} [{gt_st:.1f}s - {gt_et:.1f}s]", flush=True)
        print(f"    Raw Score Stats: min={scores.min():.3f}, max={scores.max():.3f}, mean={scores.mean():.3f}", flush=True)
        print(f"    Top 5 Visual Hits:", flush=True)
        for rank, k_i in enumerate(top_k_idx, start=1):
            rec = engine.records[k_i]
            vid = rec["video_id"]
            pts = rec["pts_time"]
            sc = scores[k_i]
            is_hit = any(is_ground_truth_hit({"video_id": vid, "pts_time": pts}, gt) for gt in gt_list)
            status = " [✓ HIT]" if is_hit else ""
            print(f"      #{rank}: Video {vid} @ {pts:6.1f}s (score: {sc:.4f}){status}", flush=True)

if __name__ == "__main__":
    main()
