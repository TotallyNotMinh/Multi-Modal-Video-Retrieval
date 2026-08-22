import os, sys, json
import numpy as np

REPO_ROOT = os.getcwd()
sys.path.insert(0, REPO_ROOT)

from scripts.eval_retrieval import SearchEngine, is_ground_truth_hit

benchmark_file = "eval/vietnamese_retrieval_benchmark_stage2_rigorous.jsonl"
with open(benchmark_file, "r", encoding="utf-8") as f:
    dataset = [json.loads(line) for line in f if line.strip()]

ans_queries = [q for q in dataset if q.get("category") != "UNANSWERABLE" and q.get("answerability") != "unanswerable"]
unans_queries = [q for q in dataset if q.get("category") == "UNANSWERABLE" or q.get("answerability") == "unanswerable"]

engine = SearchEngine()

print(f"Retrieving top candidates for {len(unans_queries)} unanswerable and {len(ans_queries)} answerable queries...")

unans_eval_data = []
for q in unans_queries:
    res = engine.search(query=q["query"], w_dense=0.30, w_asr=0.70, top_k=5)
    cands = res.get("results", [])
    top_cand = cands[0] if cands else None
    
    # Extract context for top candidate
    context = {}
    if top_cand:
        vid = top_cand["video_id"]
        pts = top_cand["pts_time"]
        context = engine.get_video_context(vid, pts_time=pts, window_sec=30.0)
        
    unans_eval_data.append({
        "query_id": q.get("query_id", ""),
        "query": q["query"],
        "is_unanswerable": True,
        "ground_truth_reason": q.get("ground_truth_reason", ""),
        "top_candidate": top_cand,
        "context": context
    })

ans_eval_pool = []
for q in ans_queries:
    res = engine.search(query=q["query"], w_dense=0.30, w_asr=0.70, top_k=5)
    cands = res.get("results", [])
    top_cand = cands[0] if cands else None
    
    gt_list = q.get("relevant_segments", [])
    is_hit = False
    if top_cand and any(is_ground_truth_hit(top_cand, gt) for gt in gt_list):
        is_hit = True
        
    if top_cand and is_hit:
        vid = top_cand["video_id"]
        pts = top_cand["pts_time"]
        context = engine.get_video_context(vid, pts_time=pts, window_sec=30.0)
        ans_eval_pool.append({
            "query_id": q.get("query_id", ""),
            "query": q["query"],
            "is_unanswerable": False,
            "ground_truth_reason": q.get("ground_truth_reason", ""),
            "top_candidate": top_cand,
            "context": context
        })

# Match 68 positive queries by nearest score to the 68 unanswerable queries
unans_scores = [d["top_candidate"]["score"] if d["top_candidate"] else 0.0 for d in unans_eval_data]
ans_eval_pool = sorted(ans_eval_pool, key=lambda x: x["top_candidate"]["score"])

# Pick 68 matched positives
matched_positives = []
pool_scores = [x["top_candidate"]["score"] for x in ans_eval_pool]
used_indices = set()

for u_sc in unans_scores:
    # Find closest unused in pool
    best_idx = None
    best_diff = 999.0
    for idx, p_sc in enumerate(pool_scores):
        if idx not in used_indices:
            diff = abs(p_sc - u_sc)
            if diff < best_diff:
                best_diff = diff
                best_idx = idx
    if best_idx is not None:
        used_indices.add(best_idx)
        matched_positives.append(ans_eval_pool[best_idx])

print(f"\nPrepared 68 Negatives (Score Mean={np.mean(unans_scores):.4f}, Range=[{np.min(unans_scores):.4f}, {np.max(unans_scores):.4f}])")
matched_pos_scores = [m["top_candidate"]["score"] for m in matched_positives]
print(f"Prepared {len(matched_positives)} Matched Positives (Score Mean={np.mean(matched_pos_scores):.4f}, Range=[{np.min(matched_pos_scores):.4f}, {np.max(matched_pos_scores):.4f}])")

# Save dataset for Stage 2 isolated eval
with open("eval/stage2_paired_eval_set.json", "w", encoding="utf-8") as f:
    json.dump({
        "negatives": unans_eval_data,
        "matched_positives": matched_positives
    }, f, indent=2, ensure_ascii=False)

print("\nSaved paired dataset to eval/stage2_paired_eval_set.json")
