#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BGE Reranker Regression Benchmark & Evaluation for Vietnamese Video Speech Retrieval.
Compares:
  - Dense E5-Large + FAISS (Top 30)
  - Dense E5-Large (Top 30) -> BGE-Reranker-v2-m3 (Top 5)

Evaluates accuracy on target failure cases, latency breakdowns, and GPU VRAM footprint.
"""

import os
import sys
import time
import argparse
from typing import List, Dict, Tuple, Any
import torch

# Ensure repo root is in sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.index.transcript_semantic_index import TranscriptSemanticIndex
from src.retrieval.reranker import BGEReranker


BENCHMARK_QUERIES = [
    {
        "id": "case_1_europe_wildfire",
        "category": "Climate / Wildfire Disasters",
        "query": "Cháy rừng và nắng nóng kỷ lục ở châu Âu",
        "target_keywords": ["cháy rừng", "nắng nóng", "địa trung hải", "hy lạp", "croatia", "tây ban nha"],
        "expected_behavior": "Promote explicit wildfire / extreme heat segments over generic heavy rain segments."
    },
    {
        "id": "case_2_heart_transplant",
        "category": "Medical Emergency / Organ Transport",
        "query": "Vận chuyển cấp tốc trái tim từ Hà Nội về Huế ghép cho bệnh nhân",
        "target_keywords": ["trái tim", "hà nội", "huế", "vận chuyển", "ghép"],
        "expected_behavior": "Promote the 4h52m heart transportation segment (L21_V001 @ 287.4s) to Rank #1."
    },
    {
        "id": "case_3_subsidence_salinity",
        "category": "Semantic Paraphrase (No direct match)",
        "query": "Hiện tượng đất bị lún sụt và nước biển xâm nhập mặn ở các tỉnh miền Tây",
        "target_keywords": ["sụt lún", "sạt lở", "xâm nhập mặn", "đồng bằng sông cửu long", "miền tây"],
        "expected_behavior": "Match Mekong Delta subsidence and salinity intrusion proposal with high confidence."
    },
    {
        "id": "case_4_dyke_erosion",
        "category": "Policy & Disaster Prevention",
        "query": "Họp bàn giải pháp phòng chống thiên tai và sạt lở đê điều",
        "target_keywords": ["đê", "sạt lở", "ban bố tình trạng khẩn cấp", "thiên tai"],
        "expected_behavior": "Rank East Sea dyke erosion meeting (L21_V007 @ 143.4s) in Top 1-2."
    }
]


def get_gpu_memory_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def main():
    parser = argparse.ArgumentParser(description="Benchmark BGE Reranker on Vietnamese Transcripts")
    parser.add_argument("--top-k-dense", type=int, default=30, help="Dense FAISS candidate pool size (default: 30)")
    parser.add_argument("--top-k-final", type=int, default=5, help="Final reranked results count (default: 5)")
    parser.add_argument("--batch-size", type=int, default=16, help="Reranker batch size (default: 16)")
    parser.add_argument("--max-length", type=int, default=384, help="Max token sequence length (default: 384)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print("=" * 85)
    print("🔬 BGE-RERANKER-V2-M3 BENCHMARK & REGRESSION EVALUATION")
    print("=" * 85)
    print(f"Device          : {args.device}")
    print(f"Dense Candidates: Top {args.top_k_dense}")
    print(f"Final Reranked  : Top {args.top_k_final}")
    print(f"Batch Size      : {args.batch_size}")
    print(f"Max Seq Length  : {args.max_length}")
    print("=" * 85)

    # 1. Initialize Dense Semantic Index
    print("\n[1/2] Loading Dense Semantic Index (multilingual-e5-large + FAISS)...")
    dense_index = TranscriptSemanticIndex()
    dense_index.load_or_build()
    dense_index._get_encoder()

    # 2. Initialize BGE Reranker
    print("\n[2/2] Loading BAAI/bge-reranker-v2-m3...")
    reranker = BGEReranker(
        model_name_or_path="BAAI/bge-reranker-v2-m3",
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print("\n" + "=" * 85)
    print("🎯 RUNNING BENCHMARK QUERIES & COMPARISON")
    print("=" * 85)

    dense_latencies = []
    rerank_latencies = []
    total_latencies = []

    for q_idx, item in enumerate(BENCHMARK_QUERIES, 1):
        q_text = item["query"]
        cat = item["category"]
        exp = item["expected_behavior"]

        print("\n" + "-" * 85)
        print(f"📌 [Test {q_idx}/4] {cat}")
        print(f"   Query: \"{q_text}\"")
        print(f"   Goal : {exp}")
        print("-" * 85)

        # Step A: Dense Retrieval (Top 30)
        t_d0 = time.time()
        dense_hits = dense_index.query(q_text, top_k=args.top_k_dense)
        dense_ms = (time.time() - t_d0) * 1000.0
        dense_latencies.append(dense_ms)

        # Step B: BGE Reranking (Top 30 -> Top 5)
        t_r0 = time.time()
        reranked_hits = reranker.rerank(q_text, dense_hits, top_k=args.top_k_final)
        rerank_ms = (time.time() - t_r0) * 1000.0
        rerank_latencies.append(rerank_ms)

        total_ms = dense_ms + rerank_ms
        total_latencies.append(total_ms)

        # Map dense ranks
        dense_rank_map = {}
        for rank_0, (seg, score) in enumerate(dense_hits, 1):
            key = (seg["video_id"], seg["segment_id"])
            dense_rank_map[key] = (rank_0, score)

        print(f"   ⏱️ Latency: E5 Retrieval: {dense_ms:.1f}ms | BGE Rerank ({len(dense_hits)} pairs): {rerank_ms:.1f}ms | Total: {total_ms:.1f}ms")
        print(f"\n   🏆 Top {args.top_k_final} Reranked Results (vs Dense Original Rank):")

        for r_rank, r_item in enumerate(reranked_hits, 1):
            key = (r_item["video_id"], r_item["segment_id"])
            d_rank, d_score = dense_rank_map.get(key, (-1, r_item.get("dense_score", 0.0)))
            r_score = r_item["rerank_score"]

            rank_shift = "=" if d_rank == r_rank else (f"▲ +{d_rank - r_rank}" if d_rank > r_rank else f"▼ -{r_rank - d_rank}")
            vid = r_item["video_id"]
            st = r_item["start_sec"]
            et = r_item["end_sec"]
            txt = r_item["text"].replace("\n", " ")
            if len(txt) > 115:
                txt = txt[:115] + "..."

            print(f"   [{r_rank}] BGE: {r_score:.4f} (Dense Rank #{d_rank} | E5: {d_score:.4f}) [{rank_shift}]")
            print(f"       {vid} [{st:.1f}s - {et:.1f}s]: \"{txt}\"")

    # Summary Statistics
    peak_vram_mb = get_gpu_memory_mb()
    print("\n" + "=" * 85)
    print("📊 BENCHMARK SUMMARY & TELEMETRY")
    print("=" * 85)
    print(f"  • Avg Dense Retrieval Latency : {sum(dense_latencies)/len(dense_latencies):.2f} ms")
    print(f"  • Avg BGE Reranking Latency   : {sum(rerank_latencies)/len(rerank_latencies):.2f} ms")
    print(f"  • Avg Total Local Latency     : {sum(total_latencies)/len(total_latencies):.2f} ms (Target < 150 ms)")
    print(f"  • Peak GPU VRAM Allocated     : {peak_vram_mb:.1f} MB (~{peak_vram_mb/1024:.2f} GB / 6.0 GB)")
    print("=" * 85)


if __name__ == "__main__":
    main()
