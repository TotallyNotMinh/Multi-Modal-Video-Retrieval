#!/usr/bin/env python3
"""
Comprehensive Retrieval Evaluation & Benchmarking Suite for Vietnamese Video Retrieval.
- Evaluates full benchmark across all categories.
- Outputs breakdown by category (Recall@1, Recall@5, Recall@10, Recall@25, MRR, nDCG@5).
- Evaluates NO_ANSWER queries with No-answer Accuracy, False-Positive Rate, and Abstention Precision.
- Supports sweeps and frozen evaluation runs.
"""

import os
import sys
import json
import time
import math
import argparse
from collections import defaultdict
from typing import List, Dict, Any, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.ui.search_app import SearchEngine


def is_ground_truth_hit(candidate: Dict[str, Any], gt_item: Dict[str, Any]) -> bool:
    """
    Checks if candidate matches the ground truth criteria:
    - Same video_id
    - Either specific keyframe_index matches OR pts_time falls within [start_sec, end_sec].
    """
    if candidate["video_id"] != gt_item["video_id"]:
        return False

    kf_list = gt_item.get("keyframe_indices", [])
    if candidate.get("frame_idx") in kf_list:
        return True

    st = float(gt_item.get("start_sec", -1.0))
    et = float(gt_item.get("end_sec", -1.0))
    pts = float(candidate.get("pts_time", -1.0))

    if st >= 0 and et >= 0 and pts >= 0:
        if st <= pts <= et:
            return True

    return False


def evaluate_query(
    engine: SearchEngine,
    query_item: Dict[str, Any],
    w_dense: float = 0.50,
    w_asr: float = 0.50,
    asr_window_sec: float = 5.0,
    top_k: int = 50,
    confidence_thresh: float = 0.35
) -> Dict[str, Any]:
    query_text = query_item["query"]
    category = query_item.get("category", "DIRECT_FACTUAL")
    gt_list = query_item.get("relevant_segments", query_item.get("ground_truth", []))
    is_unanswerable = (query_item.get("answerability") == "unanswerable") or (category == "NO_ANSWER")
    hard_negs = query_item.get("hard_negative_segments", [])

    t0 = time.time()
    res = engine.search(
        query=query_text,
        w_dense=w_dense,
        w_asr=w_asr,
        top_k=top_k,
        asr_window_sec=asr_window_sec
    )
    latency_ms = (time.time() - t0) * 1000.0

    candidates = res.get("results", [])
    top_score = candidates[0].get("score", 0.0) if candidates else 0.0

    if is_unanswerable:
        # Check if top prediction has false positive confidence or matches hard negatives
        # System abstains / rejects if top_score < confidence_thresh
        abstained = (top_score < confidence_thresh)
        # Check if top candidate is one of the hard negatives (false positive trap)
        hit_hard_neg = False
        if candidates and hard_negs:
            top_cand = candidates[0]
            for hn in hard_negs:
                if is_ground_truth_hit(top_cand, hn):
                    hit_hard_neg = True
                    break

        return {
            "query_id": query_item.get("query_id", ""),
            "query": query_text,
            "category": category,
            "is_unanswerable": True,
            "top_score": top_score,
            "abstained": abstained,
            "hit_hard_neg": hit_hard_neg,
            "latency_ms": latency_ms
        }

    # Answerable queries evaluation
    first_hit_rank = None
    matched_gt = None
    for cand in candidates:
        for gt in gt_list:
            if is_ground_truth_hit(cand, gt):
                first_hit_rank = cand["rank"]
                matched_gt = gt
                break
        if first_hit_rank is not None:
            break

    r1 = 1.0 if (first_hit_rank is not None and first_hit_rank <= 1) else 0.0
    r5 = 1.0 if (first_hit_rank is not None and first_hit_rank <= 5) else 0.0
    r10 = 1.0 if (first_hit_rank is not None and first_hit_rank <= 10) else 0.0
    r25 = 1.0 if (first_hit_rank is not None and first_hit_rank <= 25) else 0.0
    r50 = 1.0 if (first_hit_rank is not None and first_hit_rank <= 50) else 0.0
    mrr = (1.0 / first_hit_rank) if first_hit_rank is not None else 0.0
    ndcg5 = (1.0 / math.log2(first_hit_rank + 1)) if (first_hit_rank is not None and first_hit_rank <= 5) else 0.0

    return {
        "query_id": query_item.get("query_id", ""),
        "query": query_text,
        "category": category,
        "is_unanswerable": False,
        "first_hit_rank": first_hit_rank,
        "matched_gt": matched_gt,
        "r@1": r1,
        "r@5": r5,
        "r@10": r10,
        "r@25": r25,
        "r@50": r50,
        "mrr": mrr,
        "ndcg@5": ndcg5,
        "top_score": top_score,
        "latency_ms": latency_ms
    }


def run_full_benchmark(
    engine: SearchEngine,
    dataset: List[Dict[str, Any]],
    w_dense: float = 0.70,
    w_asr: float = 0.30,
    asr_window_sec: float = 5.0,
    top_k: int = 50,
    confidence_thresh: float = 0.35,
    verbose: bool = True
) -> Dict[str, Any]:
    query_results = []
    cat_results = defaultdict(list)
    unanswerable_results = []

    for i, item in enumerate(dataset, start=1):
        q_res = evaluate_query(
            engine=engine,
            query_item=item,
            w_dense=w_dense,
            w_asr=w_asr,
            asr_window_sec=asr_window_sec,
            top_k=top_k,
            confidence_thresh=confidence_thresh
        )
        query_results.append(q_res)
        cat = q_res["category"]

        if q_res["is_unanswerable"]:
            unanswerable_results.append(q_res)
        else:
            cat_results[cat].append(q_res)

        if verbose and (i % 25 == 0 or i == len(dataset) or i <= 10):
            if q_res["is_unanswerable"]:
                status = f"Abstained: {q_res['abstained']} (Score: {q_res['top_score']:.3f})"
            else:
                status = f"Rank #{q_res['first_hit_rank']}" if q_res['first_hit_rank'] else "MISS"
            print(f"  [{i:>3}/{len(dataset)}] {cat:<20} -> {status:<15} ({q_res['latency_ms']:.1f}ms) -> {q_res['query'][:45]}...")

    # Overall Answerable Metrics
    ans_queries = [q for q in query_results if not q["is_unanswerable"]]
    n_ans = max(1, len(ans_queries))

    overall_metrics = {
        "total_queries": len(dataset),
        "answerable_count": len(ans_queries),
        "unanswerable_count": len(unanswerable_results),
        "recall@1": sum(q["r@1"] for q in ans_queries) / n_ans,
        "recall@5": sum(q["r@5"] for q in ans_queries) / n_ans,
        "recall@10": sum(q["r@10"] for q in ans_queries) / n_ans,
        "recall@25": sum(q["r@25"] for q in ans_queries) / n_ans,
        "recall@50": sum(q["r@50"] for q in ans_queries) / n_ans,
        "mrr": sum(q["mrr"] for q in ans_queries) / n_ans,
        "ndcg@5": sum(q["ndcg@5"] for q in ans_queries) / n_ans,
        "avg_latency_ms": sum(q["latency_ms"] for q in query_results) / max(1, len(query_results))
    }

    # Per-Category Breakdown
    category_metrics = {}
    for cat, list_q in cat_results.items():
        cn = max(1, len(list_q))
        category_metrics[cat] = {
            "count": len(list_q),
            "recall@1": sum(q["r@1"] for q in list_q) / cn,
            "recall@5": sum(q["r@5"] for q in list_q) / cn,
            "recall@10": sum(q["r@10"] for q in list_q) / cn,
            "recall@25": sum(q["r@25"] for q in list_q) / cn,
            "mrr": sum(q["mrr"] for q in list_q) / cn,
            "ndcg@5": sum(q["ndcg@5"] for q in list_q) / cn
        }

    # NO_ANSWER Metrics
    if unanswerable_results:
        n_unans = len(unanswerable_results)
        no_ans_acc = sum(1 for q in unanswerable_results if q["abstained"]) / n_unans
        fp_rate = sum(1 for q in unanswerable_results if not q["abstained"]) / n_unans
        hard_neg_trap_rate = sum(1 for q in unanswerable_results if q["hit_hard_neg"]) / n_unans
        no_answer_metrics = {
            "count": n_unans,
            "no_answer_accuracy": no_ans_acc,
            "false_positive_rate": fp_rate,
            "hard_negative_trap_rate": hard_neg_trap_rate,
            "avg_unanswerable_score": sum(q["top_score"] for q in unanswerable_results) / n_unans
        }
    else:
        no_answer_metrics = {"count": 0}

    return {
        "overall": overall_metrics,
        "by_category": category_metrics,
        "no_answer": no_answer_metrics
    }


def print_formatted_report(eval_output: Dict[str, Any]):
    ov = eval_output["overall"]
    by_cat = eval_output["by_category"]
    no_ans = eval_output["no_answer"]

    print("\n" + "="*88)
    print("                       RETRIEVAL BENCHMARK REPORT (498 QUERIES)")
    print("="*88)
    print(f" Total Benchmark Queries : {ov['total_queries']}")
    print(f" Answerable Queries      : {ov['answerable_count']}")
    print(f" Unanswerable Queries    : {ov['unanswerable_count']}")
    print(f" Overall Recall@1        : {ov['recall@1']*100:.1f}%")
    print(f" Overall Recall@5        : {ov['recall@5']*100:.1f}%")
    print(f" Overall Recall@10       : {ov['recall@10']*100:.1f}%")
    print(f" Overall Recall@25       : {ov['recall@25']*100:.1f}%")
    print(f" Overall MRR             : {ov['mrr']:.4f}")
    print(f" Overall nDCG@5          : {ov['ndcg@5']:.4f}")
    print(f" Avg Query Latency       : {ov['avg_latency_ms']:.1f} ms")
    print("="*88)

    print("\n----------------------------------------------------------------------------------------")
    print("  PER-CATEGORY PERFORMANCE BREAKDOWN")
    print("----------------------------------------------------------------------------------------")
    print(f"| {'Category':<22} | {'Count':<6} | {'R@1':<7} | {'R@5':<7} | {'R@10':<7} | {'R@25':<7} | {'MRR':<7} | {'nDCG@5':<7} |")
    print(f"|{'-'*24}|{'-'*8}|{'-'*9}|{'-'*9}|{'-'*9}|{'-'*9}|{'-'*9}|{'-'*9}|")
    for cat, m in sorted(by_cat.items(), key=lambda x: x[1]['count'], reverse=True):
        print(f"| {cat:<22} | {m['count']:<6} | {m['recall@1']*100:<6.1f}% | {m['recall@5']*100:<6.1f}% | {m['recall@10']*100:<6.1f}% | {m['recall@25']*100:<6.1f}% | {m['mrr']:<7.4f} | {m['ndcg@5']:<7.4f} |")
    
    if no_ans["count"] > 0:
        print(f"| {'NO_ANSWER':<22} | {no_ans['count']:<6} | {'N/A':<7} | {'N/A':<7} | {'N/A':<7} | {'N/A':<7} | {'N/A':<7} | {'N/A':<7} |")
    print("----------------------------------------------------------------------------------------")

    if no_ans["count"] > 0:
        print("\n----------------------------------------------------------------------------------------")
        print("  NO-ANSWER & UNANSWERABLE RETRIEVAL METRICS")
        print("----------------------------------------------------------------------------------------")
        print(f"  • Unanswerable Queries Count : {no_ans['count']}")
        print(f"  • No-Answer Accuracy (Abstain): {no_ans['no_answer_accuracy']*100:.1f}%")
        print(f"  • False-Positive Rate        : {no_ans['false_positive_rate']*100:.1f}%")
        print(f"  • Hard Negative Trap Rate    : {no_ans['hard_negative_trap_rate']*100:.1f}%")
        print(f"  • Avg Peak Distractor Score  : {no_ans['avg_unanswerable_score']:.4f}")
        print("----------------------------------------------------------------------------------------\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate AIC 2026 Vietnamese Retrieval Benchmark")
    parser.add_argument("--gt-file", type=str, default="eval/vietnamese_retrieval_benchmark_500.jsonl", help="Path to ground truth JSON/JSONL file")
    parser.add_argument("--w-dense", type=float, default=0.70, help="Visual weight")
    parser.add_argument("--w-asr", type=float, default=0.30, help="Speech weight")
    parser.add_argument("--asr-window", type=float, default=5.0, help="ASR time window in seconds")
    parser.add_argument("--top-k", type=int, default=50, help="Top K candidates")
    parser.add_argument("--confidence-thresh", type=float, default=0.35, help="Threshold for abstaining on NO_ANSWER queries")
    parser.add_argument("--out-json", type=str, default="eval/benchmark_results_baseline.json", help="Path to save output JSON metrics")
    args = parser.parse_args()

    if not os.path.exists(args.gt_file):
        print(f"Error: Ground truth file not found: {args.gt_file}")
        sys.exit(1)

    if args.gt_file.endswith(".jsonl"):
        with open(args.gt_file, "r", encoding="utf-8") as f:
            dataset = [json.loads(line) for line in f if line.strip()]
    else:
        with open(args.gt_file, "r", encoding="utf-8") as f:
            dataset = json.load(f)

    print(f"\n=======================================================")
    print(f"  AIC 2026 Vietnamese Multimodal Retrieval Benchmark")
    print(f"  Frozen Benchmark File : {args.gt_file}")
    print(f"  Total Queries Loaded  : {len(dataset)}")
    print(f"=======================================================\n")

    engine = SearchEngine()

    eval_output = run_full_benchmark(
        engine=engine,
        dataset=dataset,
        w_dense=args.w_dense,
        w_asr=args.w_asr,
        asr_window_sec=args.asr_window,
        top_k=args.top_k,
        confidence_thresh=args.confidence_thresh,
        verbose=True
    )

    print_formatted_report(eval_output)

    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(eval_output, f, indent=2, ensure_ascii=False)
        print(f"[✓] Complete benchmark results saved to {args.out_json}\n")


if __name__ == "__main__":
    main()
