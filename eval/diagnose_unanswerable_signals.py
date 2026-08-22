import os, sys, json
import numpy as np

REPO_ROOT = os.getcwd()
sys.path.insert(0, REPO_ROOT)

from scripts.eval_retrieval import SearchEngine, is_ground_truth_hit

benchmark_file = "eval/vietnamese_retrieval_benchmark_stage2_rigorous.jsonl"
with open(benchmark_file, "r", encoding="utf-8") as f:
    dataset = [json.loads(line) for line in f if line.strip()]

ans_queries = [q for q in dataset if q.get("category") != "UNANSWERABLE" and q.get("answerability") != "unanswerable"][:30]
unans_queries = [q for q in dataset if q.get("category") == "UNANSWERABLE" or q.get("answerability") == "unanswerable"][:30]

engine = SearchEngine()

print("\n--- DETAILED SIGNAL BREAKDOWN FOR UNANSWERABLE VS ANSWERABLE ---\n")

def analyze_signals(queries, label):
    print(f"=== {label} (Sample n={len(queries)}) ===")
    bm25_maxes = []
    e5_maxes = []
    bge_logits = []
    clip_maxes = []
    
    for q in queries:
        q_text = q["query"]
        # 1. BM25
        b_hits = engine.bm25.search(q_text, top_k=5)
        top_b = b_hits[0][1] if b_hits else 0.0
        bm25_maxes.append(top_b)
        
        # 2. E5
        s_hits = engine.semantic_index.query(q_text, top_k=5) if engine.semantic_index else []
        top_e5 = s_hits[0][1] if s_hits else 0.0
        e5_maxes.append(top_e5)
        
        # 3. BGE
        top_bge = -999.0
        if s_hits and engine.reranker:
            cands = engine.reranker.rerank(query=q_text, candidates=s_hits[:10], top_k=1)
            if cands:
                top_bge = cands[0].get("rerank_score", -999.0)
        bge_logits.append(top_bge)
        
        # 4. CLIP
        en_q = engine.translator.translate(q_text)
        prompts = engine.translator.generate_prompts(en_q)
        q_vec = engine.encoder.encode_text(prompts, ensemble=True)
        clip_sc = np.dot(engine.matrix, q_vec)
        top_clip = float(np.max(clip_sc))
        clip_maxes.append(top_clip)

    print(f"  BM25 Max Score : Mean = {np.mean(bm25_maxes):.2f}, Median = {np.median(bm25_maxes):.2f}, Zero-count = {sum(1 for x in bm25_maxes if x == 0)}/{len(queries)}")
    print(f"  E5 Max Cosine  : Mean = {np.mean(e5_maxes):.4f}, Median = {np.median(e5_maxes):.4f}, Min = {np.min(e5_maxes):.4f}, Max = {np.max(e5_maxes):.4f}")
    print(f"  BGE Top Logit  : Mean = {np.mean(bge_logits):.4f}, Median = {np.median(bge_logits):.4f}, Min = {np.min(bge_logits):.4f}, Max = {np.max(bge_logits):.4f}")
    print(f"  CLIP Max Dot   : Mean = {np.mean(clip_maxes):.4f}, Median = {np.median(clip_maxes):.4f}, Min = {np.min(clip_maxes):.4f}, Max = {np.max(clip_maxes):.4f}\n")

analyze_signals(ans_queries, "ANSWERABLE QUERIES")
analyze_signals(unans_queries, "UNANSWERABLE QUERIES")
