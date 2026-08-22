#!/usr/bin/env python3
"""
Step 1: Compute exact NO_ANSWER metrics across multiple rejection thresholds.
Step 2: Error-analyze 20 worst LOW_OVERLAP failures (inspect translated query, raw CLIP, raw BM25, raw E5).
Step 3: Run Pure Dense (E5-Alone + CLIP) vs Lexical BM25-Alone vs Hybrid Fusion on the same benchmark.
"""

import os
import sys
import json
import numpy as np
from collections import defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.eval_retrieval import SearchEngine, is_ground_truth_hit

def main():
    benchmark_file = "eval/vietnamese_retrieval_benchmark_stage2_rigorous.jsonl"
    with open(benchmark_file, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    engine = SearchEngine()

    print("\n" + "="*80)
    print("  STEP 1: EXACT NO_ANSWER / UNANSWERABLE METRICS ANALYSIS (68 QUERIES)")
    print("="*80)

    unans_queries = [q for q in dataset if q.get("category") == "UNANSWERABLE" or q.get("answerability") == "unanswerable"]
    unans_scores = []
    hard_neg_hits = []

    for item in unans_queries:
        res = engine.search(query=item["query"], w_dense=0.70, w_asr=0.30, top_k=50, asr_window_sec=5.0)
        cands = res.get("results", [])
        top_score = cands[0]["score"] if cands else 0.0
        unans_scores.append(top_score)

        # Check if top candidate hit one of the hard negatives
        hit_hn = False
        if cands and item.get("hard_negative_segments"):
            top_cand = cands[0]
            for hn in item["hard_negative_segments"]:
                if is_ground_truth_hit(top_cand, hn):
                    hit_hn = True
                    break
        hard_neg_hits.append(hit_hn)

    unans_scores = np.array(unans_scores)
    print(f"Total Unanswerable Queries: {len(unans_scores)}")
    print(f"Score Distribution on Unanswerable Queries:")
    print(f"  • Min:    {np.min(unans_scores):.4f}")
    print(f"  • Mean:   {np.mean(unans_scores):.4f}")
    print(f"  • Median: {np.median(unans_scores):.4f}")
    print(f"  • Max:    {np.max(unans_scores):.4f}")
    print(f"  • Hard Negative Trap Rate: {np.mean(hard_neg_hits)*100:.1f}%\n")

    print("| Confidence Threshold (tau) | Abstention Accuracy (True Neg) | False Positive Rate |")
    print("|-----------------------------|--------------------------------|---------------------|")
    for thresh in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]:
        abstained = unans_scores < thresh
        acc = np.mean(abstained) * 100
        fp = (1.0 - np.mean(abstained)) * 100
        print(f"| tau = {thresh:<20.2f} | {acc:28.1f}% | {fp:17.1f}% |")

    print("\n" + "="*80)
    print("  STEP 2: ERROR ANALYSIS ON 20 LOW_OVERLAP MISSES")
    print("="*80)

    low_overlap_queries = [q for q in dataset if q.get("category") == "LOW_OVERLAP"]
    misses = []

    for item in low_overlap_queries:
        query_vi = item["query"]
        gt_list = item.get("relevant_segments", [])
        res = engine.search(query=query_vi, w_dense=0.70, w_asr=0.30, top_k=50, asr_window_sec=5.0)
        cands = res.get("results", [])

        first_hit_rank = None
        for cand in cands:
            for gt in gt_list:
                if is_ground_truth_hit(cand, gt):
                    first_hit_rank = cand["rank"]
                    break
            if first_hit_rank is not None:
                break

        if first_hit_rank is None or first_hit_rank > 10:
            en_trans = res.get("translated_query", "")
            bm25_hits = engine.bm25.search(query_vi, top_k=5)
            max_b_score = bm25_hits[0][1] if bm25_hits else 0.0
            sem_hits = engine.semantic_index.query(query_vi, top_k=5) if engine.semantic_index else []
            max_sem_score = sem_hits[0][1] if sem_hits else 0.0
            
            misses.append({
                "query_vi": query_vi,
                "translated_query": en_trans,
                "first_hit_rank": first_hit_rank if first_hit_rank else "MISS (>50)",
                "gt_video": gt_list[0]["video_id"] if gt_list else "N/A",
                "gt_seg": gt_list[0]["segment_id"] if gt_list else "N/A",
                "top_bm25_score": max_b_score,
                "top_e5_score": max_sem_score,
                "top1_retrieved_vid": cands[0]["video_id"] if cands else "N/A"
            })

    print(f"Total LOW_OVERLAP queries: {len(low_overlap_queries)} | Misses (>10): {len(misses)}")
    for idx, m in enumerate(misses[:15], start=1):
        print(f"\n[{idx:>2}] Query (VI): \"{m['query_vi']}\"")
        print(f"     Translated: \"{m['translated_query']}\"")
        print(f"     Target GT : Video {m['gt_video']} (Seg {m['gt_seg']}) -> Result: {m['first_hit_rank']}")
        print(f"     Top-1 Hit : Video {m['top1_retrieved_vid']} (BM25: {m['top_bm25_score']:.2f}, E5: {m['top_e5_score']:.4f})")

    print("\n" + "="*80)
    print("  STEP 3: PURE DENSE (E5+CLIP) vs BM25-ALONE vs HYBRID FUSION ON LOW_OVERLAP")
    print("="*80)

    configs = [
        ("Pure Dense (w_dense=1.0, w_asr=0.0)", 1.0, 0.0),
        ("Pure ASR/BM25 (w_dense=0.0, w_asr=1.0)", 0.0, 1.0),
        ("Hybrid Fusion (w_dense=0.7, w_asr=0.3)", 0.7, 0.3),
        ("Dense-Heavy Hybrid (w_dense=0.85, w_asr=0.15)", 0.85, 0.15),
    ]

    for name, wd, wa in configs:
        r1_list, r5_list, r10_list, mrr_list = [], [], [], []
        for item in low_overlap_queries:
            res = engine.search(query=item["query"], w_dense=wd, w_asr=wa, top_k=50, asr_window_sec=5.0)
            cands = res.get("results", [])
            gt_list = item.get("relevant_segments", [])
            first_hit = None
            for cand in cands:
                for gt in gt_list:
                    if is_ground_truth_hit(cand, gt):
                        first_hit = cand["rank"]
                        break
                if first_hit is not None:
                    break

            r1_list.append(1.0 if (first_hit is not None and first_hit <= 1) else 0.0)
            r5_list.append(1.0 if (first_hit is not None and first_hit <= 5) else 0.0)
            r10_list.append(1.0 if (first_hit is not None and first_hit <= 10) else 0.0)
            mrr_list.append(1.0 / first_hit if first_hit is not None else 0.0)

        print(f"| {name:<46} | R@1: {np.mean(r1_list)*100:4.1f}% | R@5: {np.mean(r5_list)*100:4.1f}% | R@10: {np.mean(r10_list)*100:4.1f}% | MRR: {np.mean(mrr_list):.4f} |")

if __name__ == "__main__":
    main()
