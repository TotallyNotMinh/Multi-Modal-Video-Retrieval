import os, sys, json
import numpy as np
from collections import defaultdict

REPO_ROOT = os.getcwd()
sys.path.insert(0, REPO_ROOT)

from scripts.eval_retrieval import SearchEngine, is_ground_truth_hit

benchmark_file = "eval/vietnamese_retrieval_benchmark_stage2_rigorous.jsonl"
with open(benchmark_file, "r", encoding="utf-8") as f:
    dataset = [json.loads(line) for line in f if line.strip()]

ans_queries = [q for q in dataset if q.get("category") != "UNANSWERABLE" and q.get("answerability") != "unanswerable"]
unans_queries = [q for q in dataset if q.get("category") == "UNANSWERABLE" or q.get("answerability") == "unanswerable"]

print(f"Loaded {len(ans_queries)} answerable queries and {len(unans_queries)} unanswerable queries.")

engine = SearchEngine()

ans_scores = []
ans_hit_scores = []
for q in ans_queries:
    res = engine.search(query=q["query"], w_dense=0.30, w_asr=0.70, top_k=10)
    cands = res.get("results", [])
    top_sc = cands[0]["score"] if cands else 0.0
    ans_scores.append(top_sc)
    
    gt_list = q.get("relevant_segments", [])
    if cands and any(is_ground_truth_hit(cands[0], gt) for gt in gt_list):
        ans_hit_scores.append(top_sc)

unans_scores = []
for q in unans_queries:
    res = engine.search(query=q["query"], w_dense=0.30, w_asr=0.70, top_k=10)
    cands = res.get("results", [])
    top_sc = cands[0]["score"] if cands else 0.0
    unans_scores.append(top_sc)

print("\n--- SCORE DISTRIBUTIONS ---")
print(f"Answerable Top-1 (All n={len(ans_scores)}): Mean={np.mean(ans_scores):.4f}, Median={np.median(ans_scores):.4f}, Min={np.min(ans_scores):.4f}, Max={np.max(ans_scores):.4f}")
print(f"Answerable Top-1 (Correct Hits n={len(ans_hit_scores)}): Mean={np.mean(ans_hit_scores):.4f}, Median={np.median(ans_hit_scores):.4f}, Min={np.min(ans_hit_scores):.4f}, Max={np.max(ans_hit_scores):.4f}")
print(f"Unanswerable Top-1 (n={len(unans_scores)}): Mean={np.mean(unans_scores):.4f}, Median={np.median(unans_scores):.4f}, Min={np.min(unans_scores):.4f}, Max={np.max(unans_scores):.4f}")

# Threshold Sweep
print("\n--- THRESHOLD (tau) SWEEP ---")
print(f"| {'Threshold (tau)':<16} | {'Abstention Acc':<16} | {'False Positive Rate':<22} | {'Ans Retention (R@1)':<20} |")
print(f"|{'-'*18}|{'-'*18}|{'-'*24}|{'-'*22}|")

for tau in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
    abstain_acc = np.mean([s < tau for s in unans_scores]) * 100
    fpr = 100.0 - abstain_acc
    ans_kept = np.mean([s >= tau for s in ans_hit_scores]) * 100
    print(f"| tau = {tau:<10.2f} | {abstain_acc:12.1f}%    | {fpr:18.1f}%    | {ans_kept:16.1f}%   |")
