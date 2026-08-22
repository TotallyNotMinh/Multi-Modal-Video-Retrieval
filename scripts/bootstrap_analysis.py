#!/usr/bin/env python3
"""
Bootstrap 95% Confidence Interval Analysis for Retrieval Benchmark Categories.
Runs 2,000 bootstrap iterations per category on R@1 and MRR.
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


def compute_bootstrap_ci(
    benchmark_file: str = "eval/vietnamese_retrieval_benchmark_stage2_rigorous.jsonl",
    w_dense: float = 0.70,
    w_asr: float = 0.30,
    asr_window: float = 5.0,
    n_bootstraps: int = 2000,
    seed: int = 42
):
    np.random.seed(seed)
    with open(benchmark_file, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    engine = SearchEngine()

    cat_r1 = defaultdict(list)
    cat_mrr = defaultdict(list)

    print(f"Evaluating {len(dataset)} queries across categories...")
    for item in dataset:
        if item.get("category") == "UNANSWERABLE" or item.get("answerability") == "unanswerable":
            continue

        q_text = item["query"]
        cat = item["category"]
        gt_list = item.get("relevant_segments", [])

        res = engine.search(
            query=q_text,
            w_dense=w_dense,
            w_asr=w_asr,
            top_k=50,
            asr_window_sec=asr_window
        )
        candidates = res.get("results", [])

        first_hit_rank = None
        for cand in candidates:
            for gt in gt_list:
                if is_ground_truth_hit(cand, gt):
                    first_hit_rank = cand["rank"]
                    break
            if first_hit_rank is not None:
                break

        r1 = 1.0 if (first_hit_rank is not None and first_hit_rank <= 1) else 0.0
        mrr = (1.0 / first_hit_rank) if first_hit_rank is not None else 0.0

        cat_r1[cat].append(r1)
        cat_mrr[cat].append(mrr)

    print("\n" + "="*96)
    print("      PER-CATEGORY SAMPLE SIZES (n) & BOOTSTRAP 95% CONFIDENCE INTERVALS (2,000 RESAMPLES)")
    print("="*96)
    print(f"| {'Category':<22} | {'n':<5} | {'R@1 Point':<10} | {'R@1 95% CI':<18} | {'MRR Point':<10} | {'MRR 95% CI':<18} |")
    print(f"|{'-'*24}|{'-'*7}|{'-'*12}|{'-'*20}|{'-'*12}|{'-'*20}|")

    for cat in sorted(cat_r1.keys(), key=lambda c: len(cat_r1[c]), reverse=True):
        r1_arr = np.array(cat_r1[cat])
        mrr_arr = np.array(cat_mrr[cat])
        n = len(r1_arr)

        # Empirical point estimates
        r1_pt = np.mean(r1_arr)
        mrr_pt = np.mean(mrr_arr)

        # Bootstrap sampling
        boot_r1 = []
        boot_mrr = []
        for _ in range(n_bootstraps):
            indices = np.random.choice(n, size=n, replace=True)
            boot_r1.append(np.mean(r1_arr[indices]))
            boot_mrr.append(np.mean(mrr_arr[indices]))

        r1_ci_low, r1_ci_high = np.percentile(boot_r1, [2.5, 97.5])
        mrr_ci_low, mrr_ci_high = np.percentile(boot_mrr, [2.5, 97.5])

        r1_ci_str = f"[{r1_ci_low*100:4.1f}% - {r1_ci_high*100:4.1f}%]"
        mrr_ci_str = f"[{mrr_ci_low:5.4f} - {mrr_ci_high:5.4f}]"

        print(f"| {cat:<22} | {n:<5} | {r1_pt*100:8.1f}% | {r1_ci_str:<18} | {mrr_pt:10.4f} | {mrr_ci_str:<18} |")

    print("="*96 + "\n")


if __name__ == "__main__":
    compute_bootstrap_ci()
