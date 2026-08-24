#!/usr/bin/env python3
"""
High-Speed Vectorized Dual Benchmark Runner:
Faithfully computes exact SearchEngine candidate representations (CLIP subscene decomposition,
BM25 lexical calibration, E5/BGE semantic rerank + context expansion) ONCE per query,
then evaluates all 5 fusion configurations with exact Temporal NMS deduplication.
Executes both benchmarks (1,427 total queries) in ~1-2 minutes.
"""

import os
import sys
import json
import time
import math
import numpy as np

if not hasattr(np, "long"):
    np.long = int
if not hasattr(np, "ulong"):
    np.ulong = int

from collections import defaultdict
from typing import List, Dict, Any, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.eval_retrieval import SearchEngine, is_ground_truth_hit

CONFIGS = [
    ("Pure Visual Dense (1.0 Dense / 0.0 ASR)", 1.0, 0.0),
    ("Pure Speech/OCR (0.0 Dense / 1.0 ASR)", 0.0, 1.0),
    ("Hybrid Baseline (0.70 Dense / 0.30 ASR)", 0.70, 0.30),
    ("ASR-Heavy Hybrid (0.30 Dense / 0.70 ASR)", 0.30, 0.70),
    ("Dense-Heavy Hybrid (0.85 Dense / 0.15 ASR)", 0.85, 0.15),
    ("Adaptive Category Routing (Dynamic Intent)", None, None),
]


def extract_query_representations(engine: SearchEngine, query: str, asr_window_sec: float = 5.0) -> Tuple[np.ndarray, np.ndarray]:
    """Computes exact SearchEngine dense visual scores and speech/OCR scores for a query."""
    # --- A. Dense Visual Search (CLIP Multi-Scene + Holistic) ---
    en_query = engine.translator.translate(query)
    prompts = engine.translator.generate_prompts(en_query)
    q_vec = engine.encoder.encode_text(prompts, ensemble=True)
    dense_scores_whole = np.dot(engine.matrix, q_vec)

    sub_clauses = engine._decompose_query_subscenes(query)
    if len(sub_clauses) > 1:
        sub_vectors = []
        for sc in sub_clauses:
            en_sc = engine.translator.translate(sc)
            prompts_sc = engine.translator.generate_prompts(en_sc)
            sub_vectors.append(engine.encoder.encode_text(prompts_sc, ensemble=True))

        sub_matrix = np.column_stack(sub_vectors)
        sub_scores_raw = np.dot(engine.matrix, sub_matrix)

        sub_scores_norm = np.zeros_like(sub_scores_raw)
        for c in range(sub_scores_raw.shape[1]):
            col = sub_scores_raw[:, c]
            c_min, c_max = float(np.min(col)), float(np.max(col))
            col_norm = (col - c_min) / max(1e-6, c_max - c_min)
            col_conf = float(np.clip((c_max - 0.20) / 0.13, 0.15, 1.0))
            sub_scores_norm[:, c] = col_norm * col_conf

        w_min, w_max = float(np.min(dense_scores_whole)), float(np.max(dense_scores_whole))
        norm_whole = (dense_scores_whole - w_min) / max(1e-6, w_max - w_min)
        whole_conf = float(np.clip((w_max - 0.20) / 0.13, 0.15, 1.0))
        norm_whole = norm_whole * whole_conf

        if sub_scores_norm.shape[1] >= 2:
            top2_vals = np.partition(sub_scores_norm, -2, axis=1)[:, -2:]
            top2_mean = np.mean(top2_vals, axis=1)
            max_sub_pooled = (0.70 * np.max(sub_scores_norm, axis=1)) + (0.30 * top2_mean)
        else:
            max_sub_pooled = np.max(sub_scores_norm, axis=1)

        norm_dense_scores = (0.35 * norm_whole) + (0.65 * max_sub_pooled)
    else:
        d_min, d_max = float(np.min(dense_scores_whole)), float(np.max(dense_scores_whole))
        d_denom = max(1e-6, d_max - d_min)
        col_norm = (dense_scores_whole - d_min) / d_denom
        d_conf = float(np.clip((d_max - 0.20) / 0.13, 0.15, 1.0))
        norm_dense_scores = col_norm * d_conf

    # --- B. Speech Transcript Search (BM25 Lexical + E5 Semantic) ---
    bm25_hits = engine.bm25.search(query, top_k=300)
    sem_hits = []
    if engine.semantic_index is not None:
        try:
            sem_hits = engine.semantic_index.query(query, top_k=300)
        except Exception:
            sem_hits = []

    keyframe_asr_scores = np.zeros(len(engine.records), dtype=np.float32)

    if bm25_hits:
        for doc_idx, b_score in bm25_hits:
            doc = engine.bm25.docs[doc_idx]
            vid = doc["video_id"]
            st = doc["start_sec"]
            et = doc["end_sec"]
            norm_b = float(b_score) / (25.0 + float(b_score))

            kf_indices = engine.video_to_records.get(vid, [])
            for k_idx in kf_indices:
                pts = engine.records[k_idx]["pts_time"]
                if (st - asr_window_sec) <= pts <= (et + asr_window_sec):
                    if norm_b > keyframe_asr_scores[k_idx]:
                        keyframe_asr_scores[k_idx] = norm_b

    if sem_hits:
        if engine.reranker is not None:
            top_dense_candidates = sem_hits[:30]
            reranked_candidate_dicts = engine.reranker.rerank(
                query=query,
                candidates=top_dense_candidates,
                top_k=min(30, len(top_dense_candidates))
            )
            expanded_candidates = engine.context_expander.expand_and_deduplicate(
                ranked_candidates=reranked_candidate_dicts,
                neighbor_window=1,
                max_windows=15
            )
            if expanded_candidates:
                for cand in expanded_candidates:
                    vid = cand["video_id"]
                    st = cand["start_sec"]
                    et = cand["end_sec"]
                    raw_s = cand.get("rerank_score", cand.get("score", 0.0))
                    # Single-sigmoid BGE normalization (BGEReranker output is already in [0, 1])
                    if "rerank_score" in cand:
                        norm_s = float(np.clip(raw_s, 0.0, 1.0))
                    else:
                        norm_s = float(np.clip((raw_s - 0.70) / 0.20, 0.0, 1.0))

                    kf_indices = engine.video_to_records.get(vid, [])
                    for k_idx in kf_indices:
                        pts = engine.records[k_idx]["pts_time"]
                        if (st - asr_window_sec) <= pts <= (et + asr_window_sec):
                            if norm_s > keyframe_asr_scores[k_idx]:
                                keyframe_asr_scores[k_idx] = norm_s
        else:
            for cand_dict, s_score in sem_hits:
                vid = cand_dict["video_id"]
                st = cand_dict["start_sec"]
                et = cand_dict["end_sec"]
                norm_s = float(np.clip((s_score - 0.70) / 0.20, 0.0, 1.0))
                kf_indices = engine.video_to_records.get(vid, [])
                for k_idx in kf_indices:
                    pts = engine.records[k_idx]["pts_time"]
                    if (st - asr_window_sec) <= pts <= (et + asr_window_sec):
                        if norm_s > keyframe_asr_scores[k_idx]:
                            keyframe_asr_scores[k_idx] = norm_s

    return norm_dense_scores, keyframe_asr_scores


def evaluate_dataset(engine: SearchEngine, dataset_path: str, dataset_name: str, dataset_key: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    print(f"\n================================================================================")
    print(f"  RUNNING BENCHMARK ON: {dataset_name}")
    print(f"  Source file: {dataset_path}")
    print(f"================================================================================")

    with open(dataset_path, "r", encoding="utf-8") as f:
        raw_items = [json.loads(line) for line in f if line.strip()]

    queries = [
        q for q in raw_items
        if q.get("category") != "UNANSWERABLE" and q.get("answerability") != "unanswerable"
    ]

    print(f"[*] Total Answerable Queries to Evaluate: {len(queries)}")

    results = {
        name: {
            "all_r1": [],
            "all_r5": [],
            "all_r10": [],
            "all_r25": [],
            "all_mrr": [],
            "by_cat": defaultdict(lambda: defaultdict(list))
        }
        for name, _, _ in CONFIGS
    }

    per_query_records = []
    t0 = time.time()
    for idx, item in enumerate(queries, 1):
        q_id = item.get("query_id", f"q_{idx:06d}")
        q_text = item["query"]
        cat = item.get("category", "UNKNOWN")
        gt_list = item.get("relevant_segments", [])

        # 1. Compute exact multimodal scores once
        norm_dense, norm_asr = extract_query_representations(engine, q_text, asr_window_sec=5.0)

        # Route query intent with Evidence-Gated Prior
        intent = engine.classify_query_intent(q_text)
        max_asr = float(np.max(norm_asr)) if len(norm_asr) > 0 else 0.0
        
        if intent.get("category") == "visual_entity_text_grounded":
            adaptive_wd, adaptive_wa = 0.25, 0.75
        elif intent.get("category") == "speech_dialogue_grounded":
            adaptive_wd, adaptive_wa = 0.20, 0.80
        elif max_asr >= 0.50:
            # Strong transcript/speech match detected
            adaptive_wd, adaptive_wa = 0.30, 0.70
        else:
            # Transcript match is absent/weak -> visual prior
            adaptive_wd, adaptive_wa = 0.70, 0.30

        query_eval_entry = {
            "query_id": q_id,
            "dataset": dataset_key,
            "query": q_text,
            "gt_category": cat,
            "predicted_category": intent.get("category", "unknown"),
            "predicted_confidence": intent.get("confidence", 0.5),
            "max_asr_score": max_asr,
            "is_default_fallback": (intent.get("category") == "balanced_multimodal"),
            "adaptive_weights": {"w_dense": adaptive_wd, "w_asr": adaptive_wa},
            "metrics": {}
        }

        # 2. Evaluate all configurations instantly
        for name, wd, wa in CONFIGS:
            if wd is None and wa is None:
                eff_wd, eff_wa = adaptive_wd, adaptive_wa
            else:
                eff_wd, eff_wa = wd, wa

            fused = (eff_wd * norm_dense) + (eff_wa * norm_asr)

            k_pool = min(200, len(engine.records))
            top_indices = np.argpartition(fused, -k_pool)[-k_pool:]
            top_indices = top_indices[np.argsort(fused[top_indices])[::-1]]

            final_candidates_idx = engine._apply_temporal_nms(top_indices, top_k=50, window_sec=1.5)

            candidates = []
            for rank_c, c_idx in enumerate(final_candidates_idx, start=1):
                rec = engine.records[c_idx]
                candidates.append({
                    "rank": rank_c,
                    "video_id": rec["video_id"],
                    "frame_idx": rec.get("frame_idx"),
                    "pts_time": rec["pts_time"],
                    "score": float(fused[c_idx])
                })

            first_hit = None
            for cand in candidates:
                for gt in gt_list:
                    if is_ground_truth_hit(cand, gt):
                        first_hit = cand["rank"]
                        break
                if first_hit is not None:
                    break

            r1 = 1.0 if (first_hit is not None and first_hit <= 1) else 0.0
            r5 = 1.0 if (first_hit is not None and first_hit <= 5) else 0.0
            r10 = 1.0 if (first_hit is not None and first_hit <= 10) else 0.0
            r25 = 1.0 if (first_hit is not None and first_hit <= 25) else 0.0
            mrr = (1.0 / first_hit) if first_hit is not None else 0.0

            results[name]["all_r1"].append(r1)
            results[name]["all_r5"].append(r5)
            results[name]["all_r10"].append(r10)
            results[name]["all_r25"].append(r25)
            results[name]["all_mrr"].append(mrr)

            results[name]["by_cat"][cat]["r1"].append(r1)
            results[name]["by_cat"][cat]["r5"].append(r5)
            results[name]["by_cat"][cat]["r10"].append(r10)
            results[name]["by_cat"][cat]["r25"].append(r25)
            results[name]["by_cat"][cat]["mrr"].append(mrr)

            query_eval_entry["metrics"][name] = {
                "first_hit_rank": first_hit,
                "r1": int(r1),
                "r5": int(r5),
                "r10": int(r10),
                "r25": int(r25),
                "mrr": float(mrr)
            }

        per_query_records.append(query_eval_entry)

        if idx % 50 == 0 or idx == len(queries):
            print(f"  -> Evaluated {idx}/{len(queries)} queries ({time.time() - t0:.2f}s)...", flush=True)

    elapsed = time.time() - t0
    print(f"[✓] Finished {dataset_name} in {elapsed:.2f}s ({elapsed/len(queries)*1000:.1f}ms/query)\n")

    summary = {
        "dataset_name": dataset_name,
        "dataset_path": dataset_path,
        "total_queries": len(queries),
        "elapsed_sec": round(elapsed, 2),
        "overall": {},
        "by_category": {}
    }

    for name, _, _ in CONFIGS:
        summary["overall"][name] = {
            "R@1": float(np.mean(results[name]["all_r1"]) * 100),
            "R@5": float(np.mean(results[name]["all_r5"]) * 100),
            "R@10": float(np.mean(results[name]["all_r10"]) * 100),
            "R@25": float(np.mean(results[name]["all_r25"]) * 100),
            "MRR": float(np.mean(results[name]["all_mrr"])),
        }

    cats = sorted(results[CONFIGS[0][0]]["by_cat"].keys())
    for cat in cats:
        n_cat = len(results[CONFIGS[0][0]]["by_cat"][cat]["r1"])
        summary["by_category"][cat] = {
            "n": n_cat,
            "pct_of_dataset": round(n_cat / len(queries) * 100, 2),
            "configs": {}
        }
        for name, _, _ in CONFIGS:
            summary["by_category"][cat]["configs"][name] = {
                "R@1": float(np.mean(results[name]["by_cat"][cat]["r1"]) * 100),
                "R@5": float(np.mean(results[name]["by_cat"][cat]["r5"]) * 100),
                "R@10": float(np.mean(results[name]["by_cat"][cat]["r10"]) * 100),
                "R@25": float(np.mean(results[name]["by_cat"][cat]["r25"]) * 100),
                "MRR": float(np.mean(results[name]["by_cat"][cat]["mrr"])),
            }

    return summary, per_query_records


def main():
    asr_benchmark_path = "eval/vietnamese_retrieval_benchmark_stage2_rigorous.jsonl"
    vis_benchmark_path = "eval/visual_benchmark_from_raw_frames_1024x576.jsonl"
    out_json_path = "eval/dual_benchmark_results_summary.json"
    out_breakdown_path = "eval/query_routing_breakdown.jsonl"

    print("[*] Initializing SearchEngine...")
    engine = SearchEngine()

    asr_summary, asr_query_records = evaluate_dataset(engine, asr_benchmark_path, "ASR-Focused Benchmark (Transcript-Derived)", "asr_focused")
    vis_summary, vis_query_records = evaluate_dataset(engine, vis_benchmark_path, "Visual-Focused Benchmark (Raw Frame-Derived 1024x576)", "visual_focused")

    final_report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "asr_focused_benchmark": asr_summary,
        "visual_focused_benchmark": vis_summary
    }

    os.makedirs(os.path.dirname(os.path.abspath(out_json_path)), exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2)

    all_records = asr_query_records + vis_query_records
    with open(out_breakdown_path, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n[✓] Dual benchmark evaluation complete! Saved structured results to {out_json_path}")
    print(f"[✓] Saved {len(all_records)} query breakdown logs to {out_breakdown_path}\n")

    for report_name, b_data in [("ASR-FOCUSED BENCHMARK (627 QUERIES)", asr_summary), ("VISUAL-FOCUSED BENCHMARK (800 QUERIES)", vis_summary)]:
        print(f"{'='*85}")
        print(f"  {report_name}")
        print(f"{'='*85}")
        print(f"{'Configuration':<45} | {'R@1 (%)':<9} | {'R@5 (%)':<9} | {'R@10 (%)':<9} | {'R@25 (%)':<9} | {'MRR':<8}")
        print(f"{'-'*45}-|-{'-'*9}-|-{'-'*9}-|-{'-'*9}-|-{'-'*9}-|-{'-'*8}")
        for name, _, _ in CONFIGS:
            ov = b_data["overall"][name]
            print(f"{name:<45} | {ov['R@1']:<9.2f} | {ov['R@5']:<9.2f} | {ov['R@10']:<9.2f} | {ov['R@25']:<9.2f} | {ov['MRR']:<8.4f}")
        print(f"{'='*85}\n")


if __name__ == "__main__":
    main()
