#!/usr/bin/env python3
"""
Full Benchmark Ablation with OCR-injected Candidate Generation across 5 Configurations.
Measures Recall@1, Recall@5, Recall@10, and MRR overall and per category.
"""

import os
import sys
import json
import time
import numpy as np
from collections import defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.eval_retrieval import SearchEngine, is_ground_truth_hit

def run_ablation():
    benchmark_file = "eval/vietnamese_retrieval_benchmark_stage2_rigorous.jsonl"
    with open(benchmark_file, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    ans_queries = [q for q in dataset if q.get("category") != "UNANSWERABLE" and q.get("answerability") != "unanswerable"]
    print(f"\nEvaluating {len(ans_queries)} answerable queries across 5 configurations with OCR candidate generation...\n")

    engine = SearchEngine()

    configs = [
        ("Pure Visual Dense (1.0 Dense / 0.0 ASR)", 1.0, 0.0),
        ("Pure Speech/OCR (0.0 Dense / 1.0 ASR)", 0.0, 1.0),
        ("Hybrid Baseline (0.70 Dense / 0.30 ASR)", 0.70, 0.30),
        ("ASR-Heavy Hybrid (0.30 Dense / 0.70 ASR)", 0.30, 0.70),
        ("Dense-Heavy Hybrid (0.85 Dense / 0.15 ASR)", 0.85, 0.15),
    ]

    all_results = {name: {"all_r1": [], "all_r5": [], "all_r10": [], "all_mrr": [], "by_cat": defaultdict(lambda: defaultdict(list))} for name, _, _ in configs}

    t0 = time.time()
    for idx, item in enumerate(ans_queries):
        q_text = item["query"]
        cat = item["category"]
        gt_list = item.get("relevant_segments", [])

        # For each configuration, perform search
        for name, wd, wa in configs:
            res = engine.search(query=q_text, w_dense=wd, w_asr=wa, top_k=50, asr_window_sec=5.0)
            cands = res.get("results", [])

            first_hit = None
            for cand in cands:
                for gt in gt_list:
                    if is_ground_truth_hit(cand, gt):
                        first_hit = cand["rank"]
                        break
                if first_hit is not None:
                    break

            r1 = 1.0 if (first_hit is not None and first_hit <= 1) else 0.0
            r5 = 1.0 if (first_hit is not None and first_hit <= 5) else 0.0
            r10 = 1.0 if (first_hit is not None and first_hit <= 10) else 0.0
            mrr = (1.0 / first_hit) if first_hit is not None else 0.0

            all_results[name]["all_r1"].append(r1)
            all_results[name]["all_r5"].append(r5)
            all_results[name]["all_r10"].append(r10)
            all_results[name]["all_mrr"].append(mrr)

            all_results[name]["by_cat"][cat]["r1"].append(r1)
            all_results[name]["by_cat"][cat]["r5"].append(r5)
            all_results[name]["by_cat"][cat]["r10"].append(r10)
            all_results[name]["by_cat"][cat]["mrr"].append(mrr)

    print(f"\nEvaluation finished in {time.time() - t0:.2f}s.\n")

    # Format Overall Table
    print("="*96)
    print("  POST-OCR CANDIDATE GENERATION ABLATION: OVERALL (627 ANSWERABLE QUERIES)")
    print("="*96)
    print(f"| {'Configuration':<44} | {'Recall@1':<10} | {'Recall@5':<10} | {'Recall@10':<10} | {'MRR':<8} |")
    print(f"|{'-'*46}|{'-'*12}|{'-'*12}|{'-'*12}|{'-'*10}|")
    for name, _, _ in configs:
        r1 = np.mean(all_results[name]["all_r1"]) * 100
        r5 = np.mean(all_results[name]["all_r5"]) * 100
        r10 = np.mean(all_results[name]["all_r10"]) * 100
        mrr = np.mean(all_results[name]["all_mrr"])
        print(f"| {name:<44} | {r1:8.1f}%  | {r5:8.1f}%  | {r10:8.1f}%  | {mrr:8.4f} |")

    # Format Category Breakdown
    cats = sorted(all_results[configs[0][0]]["by_cat"].keys())
    print("\n" + "="*96)
    print("  CATEGORY-BY-CATEGORY BREAKDOWN")
    print("="*96)
    for cat in cats:
        n_cat = len(all_results[configs[0][0]]["by_cat"][cat]["r1"])
        print(f"\n--- Category: {cat} (n = {n_cat}) ---")
        print(f"| {'Configuration':<44} | {'Recall@1':<10} | {'Recall@5':<10} | {'Recall@10':<10} | {'MRR':<8} |")
        print(f"|{'-'*46}|{'-'*12}|{'-'*12}|{'-'*12}|{'-'*10}|")
        for name, _, _ in configs:
            r1 = np.mean(all_results[name]["by_cat"][cat]["r1"]) * 100
            r5 = np.mean(all_results[name]["by_cat"][cat]["r5"]) * 100
            r10 = np.mean(all_results[name]["by_cat"][cat]["r10"]) * 100
            mrr = np.mean(all_results[name]["by_cat"][cat]["mrr"])
            print(f"| {name:<44} | {r1:8.1f}%  | {r5:8.1f}%  | {r10:8.1f}%  | {mrr:8.4f} |")

    # Save results JSON
    summary_data = {}
    for name, _, _ in configs:
        summary_data[name] = {
            "overall": {
                "R@1": float(np.mean(all_results[name]["all_r1"])),
                "R@5": float(np.mean(all_results[name]["all_r5"])),
                "R@10": float(np.mean(all_results[name]["all_r10"])),
                "MRR": float(np.mean(all_results[name]["all_mrr"]))
            },
            "by_category": {
                c: {
                    "n": len(all_results[name]["by_cat"][c]["r1"]),
                    "R@1": float(np.mean(all_results[name]["by_cat"][c]["r1"])),
                    "R@5": float(np.mean(all_results[name]["by_cat"][c]["r5"])),
                    "R@10": float(np.mean(all_results[name]["by_cat"][c]["r10"])),
                    "MRR": float(np.mean(all_results[name]["by_cat"][c]["mrr"]))
                }
                for c in cats
            }
        }
    with open("eval/ocr_candidate_generation_ablation_results.json", "w", encoding="utf-8") as fp:
        json.dump(summary_data, fp, indent=2, ensure_ascii=False)
    print("\nSaved full ablation results to eval/ocr_candidate_generation_ablation_results.json")

if __name__ == "__main__":
    run_ablation()
