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

def optimize_hyperparameters():
    print("=" * 70)
    print("      AIC 2026 - TASK 1 RETRIEVAL HYPERPARAMETER OPTIMIZATION")
    print("=" * 70)

    # 1. Load Data & Indexes
    matrix, records = FeatureMatrixBuilder().build_and_cache()
    dense_retriever = DenseRetriever(matrix, records)
    object_indexer = ObjectIndexer().build_and_cache()
    metadata_indexer = MetadataIndexer().build_and_cache()
    hybrid_retriever = HybridRetriever(dense_retriever, object_indexer, metadata_indexer)

    text_encoder = CLIPTextEncoder(device="cpu")
    translator = QueryTranslator(use_online=True)

    with open("cache/curated_benchmark_gt.json", "r", encoding="utf-8") as f:
        benchmarks = json.load(f)

    # Pre-encode queries
    encoded_queries = []
    for bm in benchmarks:
        vi_q = bm["query_vi"]
        en_q = bm.get("query_en") or translator.translate(vi_q)
        prompts = translator.generate_prompts(en_q)
        vec = text_encoder.encode_text(prompts, ensemble=True)
        encoded_queries.append({
            "gt": bm,
            "vi_q": vi_q,
            "en_q": en_q,
            "vec": vec
        })

    # Grid search over weights and smoothing parameters
    configs = [
        {"w_d": 0.8, "w_o": 0.1, "w_m": 0.1, "sigma": 1.0, "w_size": 2, "dedup_stride": 2.0},
        {"w_d": 0.7, "w_o": 0.15, "w_m": 0.15, "sigma": 1.5, "w_size": 3, "dedup_stride": 2.5},
        {"w_d": 0.5, "w_o": 0.2, "w_m": 0.3, "sigma": 2.0, "w_size": 4, "dedup_stride": 3.0},
        {"w_d": 0.4, "w_o": 0.2, "w_m": 0.4, "sigma": 1.5, "w_size": 3, "dedup_stride": 2.5},
        {"w_d": 0.6, "w_o": 0.1, "w_m": 0.3, "sigma": 1.5, "w_size": 3, "dedup_stride": 2.5},
    ]

    best_score = -1.0
    best_config = None
    table_rows = []

    for i, cfg in enumerate(configs, 1):
        all_preds = []
        all_gts = []
        t0 = time.time()

        for q in encoded_queries:
            gt = q["gt"]
            # Search
            res = hybrid_retriever.search_hybrid(
                query_vec=q["vec"],
                query_text_en=q["en_q"],
                query_text_vi=q["vi_q"],
                w_dense=cfg["w_d"],
                w_object=cfg["w_o"],
                w_meta=cfg["w_m"],
                use_temporal_smoothing=True,
                use_shot_dedup=True,
                top_k=100
            )
            preds = [{"video_id": r[0]["video_id"], "frame_idx": r[0]["frame_idx"]} for r in res]
            all_preds.append(preds)
            all_gts.append(gt)

        metrics = AICMetrics.evaluate_benchmark(all_preds, all_gts)
        elapsed = (time.time() - t0) * 1000 / len(encoded_queries)

        score = metrics["Final_Score"]
        if score > best_score:
            best_score = score
            best_config = cfg

        row = [
            f"Config {i} (D:{cfg['w_d']}, O:{cfg['w_o']}, M:{cfg['w_m']})",
            f"{metrics['R@1']*100:.1f}%",
            f"{metrics['R@5']*100:.1f}%",
            f"{metrics['R@20']*100:.1f}%",
            f"{metrics['R@50']*100:.1f}%",
            f"{metrics['R@100']*100:.1f}%",
            f"{score*100:.2f}",
            f"{elapsed:.1f} ms"
        ]
        table_rows.append(row)

    headers = ["Configuration", "R@1", "R@5", "R@20", "R@50", "R@100", "Final Score", "Avg Latency"]
    print("\n" + tabulate(table_rows, headers=headers, tablefmt="fancy_grid"))
    print(f"\n[Best Solution] {best_config} with Final Score = {best_score*100:.2f}")

if __name__ == "__main__":
    optimize_hyperparameters()
