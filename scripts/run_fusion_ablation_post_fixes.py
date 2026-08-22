#!/usr/bin/env python3
"""
Post-Fixes Ablation Suite across all 4 Fusion Configurations:
- Absolute Calibration + OmniRoute LLM Translation
- Evaluates full 695-query benchmark (627 answerable queries across all 7 categories)
- Measures R@1, R@5, R@10, and MRR per category and overall.
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


def evaluate_config(engine: SearchEngine, dataset: list, w_dense: float, w_asr: float, name: str):
    ans_queries = [q for q in dataset if q.get("category") != "UNANSWERABLE" and q.get("answerability") != "unanswerable"]
    cat_r1 = defaultdict(list)
    cat_r5 = defaultdict(list)
    cat_r10 = defaultdict(list)
    cat_mrr = defaultdict(list)

    all_r1, all_r5, all_r10, all_mrr = [], [], [], []

    for item in ans_queries:
        q_text = item["query"]
        cat = item["category"]
        gt_list = item.get("relevant_segments", [])

        res = engine.search(query=q_text, w_dense=w_dense, w_asr=w_asr, top_k=50, asr_window_sec=5.0)
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

        cat_r1[cat].append(r1)
        cat_r5[cat].append(r5)
        cat_r10[cat].append(r10)
        cat_mrr[cat].append(mrr)

        all_r1.append(r1)
        all_r5.append(r5)
        all_r10.append(r10)
        all_mrr.append(mrr)

    return {
        "name": name,
        "overall": {
            "R@1": np.mean(all_r1),
            "R@5": np.mean(all_r5),
            "R@10": np.mean(all_r10),
            "MRR": np.mean(all_mrr)
        },
        "by_category": {
            cat: {
                "n": len(cat_r1[cat]),
                "R@1": np.mean(cat_r1[cat]),
                "R@5": np.mean(cat_r5[cat]),
                "R@10": np.mean(cat_r10[cat]),
                "MRR": np.mean(cat_mrr[cat])
            }
            for cat in cat_r1
        }
    }


def main():
    benchmark_file = "eval/vietnamese_retrieval_benchmark_stage2_rigorous.jsonl"
    with open(benchmark_file, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    engine = SearchEngine()

    configs = [
        ("Pure Visual Dense (1.0 Dense / 0.0 ASR)", 1.0, 0.0),
        ("Pure Speech/OCR (0.0 Dense / 1.0 ASR)", 0.0, 1.0),
        ("Hybrid Baseline (0.70 Dense / 0.30 ASR)", 0.70, 0.30),
        ("ASR-Heavy Hybrid (0.30 Dense / 0.70 ASR)", 0.30, 0.70),
        ("Dense-Heavy Hybrid (0.85 Dense / 0.15 ASR)", 0.85, 0.15),
    ]

    all_results = []
    print(f"\nEvaluating {len(dataset)} queries across 4 fusion configurations post-calibration & translation fixes...\n")
    for name, wd, wa in configs:
        res = evaluate_config(engine, dataset, wd, wa, name)
        all_results.append(res)

    print("\n" + "="*96)
    print("  POST-FIXES FUSION ABLATION: OVERALL PERFORMANCE ON 627 ANSWERABLE BENCHMARK QUERIES")
    print("="*96)
    print(f"| {'Configuration':<44} | {'R@1':<8} | {'R@5':<8} | {'R@10':<8} | {'MRR':<8} |")
    print(f"|{'-'*46}|{'-'*10}|{'-'*10}|{'-'*10}|{'-'*10}|")
    for r in all_results:
        ov = r["overall"]
        print(f"| {r['name']:<44} | {ov['R@1']*100:6.1f}% | {ov['R@5']*100:6.1f}% | {ov['R@10']*100:6.1f}% | {ov['MRR']:8.4f} |")

    # Per-category comparison
    cats = list(all_results[0]["by_category"].keys())
    for cat in sorted(cats):
        n_cat = all_results[0]["by_category"][cat]["n"]
        print("\n" + "-"*96)
        print(f"  Category: {cat} (n = {n_cat})")
        print("-"*96)
        print(f"| {'Configuration':<44} | {'R@1':<8} | {'R@5':<8} | {'R@10':<8} | {'MRR':<8} |")
        print(f"|{'-'*46}|{'-'*10}|{'-'*10}|{'-'*10}|{'-'*10}|")
        for r in all_results:
            cm = r["by_category"][cat]
            print(f"| {r['name']:<44} | {cm['R@1']*100:6.1f}% | {cm['R@5']*100:6.1f}% | {cm['R@10']*100:6.1f}% | {cm['MRR']:8.4f} |")


if __name__ == "__main__":
    main()
