import os
import json
import time
import numpy as np
from tabulate import tabulate
from src.index.matrix_builder import FeatureMatrixBuilder
from src.index.object_indexer import ObjectIndexer
from src.index.metadata_indexer import MetadataIndexer
from src.query.text_encoder import CLIPTextEncoder
from src.query.translator import QueryTranslator
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.evaluation.metrics import AICMetrics
from experiments.find_precise_ground_truth import construct_curated_ground_truth

def run_retrieval_suite():
    print("=" * 70)
    print("      AIC 2026 - TASK 1 (TEXTUAL KIS) RETRIEVAL ABLATION EXPERIMENTS")
    print("=" * 70)

    # 1. Load Ground Truth Benchmark
    benchmark_path = "cache/curated_benchmark_gt.json"
    benchmarks = construct_curated_ground_truth(benchmark_path)
    print(f"Loaded {len(benchmarks)} Curated Ground Truth Benchmark Queries.\n")

    # 2. Build / Load Indexes
    t0 = time.time()
    matrix_builder = FeatureMatrixBuilder()
    matrix, records = matrix_builder.build_and_cache()
    print(f"Loaded {matrix.shape[0]} keyframe embeddings in {time.time() - t0:.2f}s.")

    dense_retriever = DenseRetriever(matrix, records)
    object_indexer = ObjectIndexer().build_and_cache()
    metadata_indexer = MetadataIndexer().build_and_cache()
    hybrid_retriever = HybridRetriever(dense_retriever, object_indexer, metadata_indexer)

    text_encoder = CLIPTextEncoder(device="cpu")
    translator = QueryTranslator(use_online=True)

    # Pre-translate all queries and encode prompts
    print("\nPreparing queries...")
    prepared_queries = []
    for bm in benchmarks:
        vi_q = bm["query_vi"]
        en_q = bm.get("query_en") or translator.translate(vi_q)
        prompts = translator.generate_prompts(en_q)
        
        raw_vec = text_encoder.encode_text(vi_q, ensemble=False)
        trans_vec = text_encoder.encode_text(en_q, ensemble=False)
        ensemble_vec = text_encoder.encode_text(prompts, ensemble=True)

        prepared_queries.append({
            "gt": bm,
            "vi_q": vi_q,
            "en_q": en_q,
            "prompts": prompts,
            "raw_vec": raw_vec,
            "trans_vec": trans_vec,
            "ensemble_vec": ensemble_vec
        })

    # Define Ablation Strategies
    strategies = [
        {
            "name": "1. Raw Vietnamese (No Translation)",
            "type": "dense",
            "vec_key": "raw_vec"
        },
        {
            "name": "2. Translated English (Vi -> En)",
            "type": "dense",
            "vec_key": "trans_vec"
        },
        {
            "name": "3. Translated + Prompt Ensemble",
            "type": "dense",
            "vec_key": "ensemble_vec"
        },
        {
            "name": "4. Ensemble + 1D Shot Temporal Smoothing",
            "type": "hybrid",
            "vec_key": "ensemble_vec",
            "w_dense": 1.0, "w_obj": 0.0, "w_meta": 0.0,
            "use_temporal": True, "use_dedup": False
        },
        {
            "name": "5. Hybrid Tri-Modal (Dense + Obj + Meta)",
            "type": "hybrid",
            "vec_key": "ensemble_vec",
            "w_dense": 0.60, "w_obj": 0.15, "w_meta": 0.25,
            "use_temporal": True, "use_dedup": False
        },
        {
            "name": "6. Full SOTA (Hybrid + Submodular Shot Dedup)",
            "type": "hybrid",
            "vec_key": "ensemble_vec",
            "w_dense": 0.60, "w_obj": 0.15, "w_meta": 0.25,
            "use_temporal": True, "use_dedup": True
        }
    ]

    results_table = []
    strategy_details = {}

    for strat in strategies:
        s_name = strat["name"]
        print(f"\nEvaluating Strategy: {s_name}...")
        all_preds = []
        all_gts = []
        latencies = []

        for q in prepared_queries:
            gt = q["gt"]
            vec = q[strat["vec_key"]]

            t_start = time.time()
            if strat["type"] == "dense":
                res = dense_retriever.search(vec, top_k=100)
            else:
                res = hybrid_retriever.search_hybrid(
                    query_vec=vec,
                    query_text_en=q["en_q"],
                    query_text_vi=q["vi_q"],
                    w_dense=strat["w_dense"],
                    w_object=strat["w_obj"],
                    w_meta=strat["w_meta"],
                    use_temporal_smoothing=strat["use_temporal"],
                    use_shot_dedup=strat["use_dedup"],
                    top_k=100
                )
            latency_ms = (time.time() - t_start) * 1000
            latencies.append(latency_ms)

            preds = [{"video_id": r[0]["video_id"], "frame_idx": r[0]["frame_idx"]} for r in res]
            all_preds.append(preds)
            all_gts.append(gt)

        metrics = AICMetrics.evaluate_benchmark(all_preds, all_gts)
        avg_lat = np.mean(latencies)

        row = [
            s_name,
            f"{metrics['R@1']*100:.1f}%",
            f"{metrics['R@5']*100:.1f}%",
            f"{metrics['R@20']*100:.1f}%",
            f"{metrics['R@50']*100:.1f}%",
            f"{metrics['R@100']*100:.1f}%",
            f"{metrics['Final_Score']*100:.2f}",
            f"{avg_lat:.1f} ms"
        ]
        results_table.append(row)
        strategy_details[s_name] = metrics

    headers = ["Strategy", "R@1", "R@5", "R@20", "R@50", "R@100", "Final Score", "Latency"]
    print("\n" + "=" * 85)
    print("                     FINAL EXPERIMENTAL COMPARISON TABLE")
    print("=" * 85)
    print(tabulate(results_table, headers=headers, tablefmt="fancy_grid"))

    # Save results to cache
    output_res_file = "cache/task1_experiment_results.json"
    with open(output_res_file, "w", encoding="utf-8") as f:
        json.dump({
            "table": results_table,
            "details": strategy_details
        }, f, indent=2)
    print(f"\nResults saved to {output_res_file}")

if __name__ == "__main__":
    run_retrieval_suite()
