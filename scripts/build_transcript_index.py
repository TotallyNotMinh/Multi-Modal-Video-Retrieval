#!/usr/bin/env python3
"""
Build Dense FAISS Semantic Index for Vietnamese Video Speech Transcripts.
Generates embeddings from refined transcripts (falling back to raw transcripts) and caches to disk.
"""

import os
import sys
import argparse
import time

# Ensure repo root is in sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.encoding.transcript_encoder import TranscriptEncoder
from src.index.transcript_semantic_index import TranscriptSemanticIndex


def main():
    parser = argparse.ArgumentParser(description="Build Vietnamese Transcript Dense Vector Index")
    parser.add_argument("--refined-dir", default="cache/asr_transcripts_refined", help="Path to refined transcripts directory")
    parser.add_argument("--raw-dir", default="cache/asr_transcripts", help="Path to raw ASR transcripts directory")
    parser.add_argument("--cache-dir", default="cache", help="Path to output cache directory")
    parser.add_argument("--model", default="intfloat/multilingual-e5-large", help="HuggingFace embedding model name")
    parser.add_argument("--batch-size", type=int, default=64, help="Encoding batch size")
    parser.add_argument("--device", default=None, help="Device (cuda or cpu)")
    parser.add_argument("--force", action="store_true", help="Force recomputation of embeddings")

    args = parser.parse_args()

    print("=" * 60)
    print("🚀 BUILDING DENSE TRANSCRIPT SEMANTIC VECTOR INDEX")
    print("=" * 60)

    encoder = TranscriptEncoder(
        model_name_or_path=args.model,
        device=args.device,
        batch_size=args.batch_size
    )

    indexer = TranscriptSemanticIndex(
        refined_asr_dir=args.refined_dir,
        raw_asr_dir=args.raw_dir,
        cache_dir=args.cache_dir,
        encoder=encoder
    )

    t0 = time.time()
    indexer.build()
    print(f"\n✅ Finished building transcript semantic index in {time.time() - t0:.2f}s!")
    print(f"   • Embeddings Matrix: {indexer.matrix_path}")
    print(f"   • Metadata File:     {indexer.meta_path}")
    if os.path.exists(indexer.faiss_path):
        print(f"   • FAISS Index:       {indexer.faiss_path}")


if __name__ == "__main__":
    main()
