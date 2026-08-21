#!/usr/bin/env python3
"""
Dual-GPU Parallel Runner for ASR Transcript Refinement on Kaggle (2x Tesla T4) or multi-GPU nodes.

Spawns 2 concurrent workers (GPU 0 & GPU 1), partitions files with strided sharding,
and merges shard manifests into a unified ledger upon completion.
"""

import os
import sys
import time
import subprocess
import argparse
import json

# Ensure repo root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def main():
    parser = argparse.ArgumentParser(description="Run dual-GPU transcript refinement in parallel.")
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen3-1.7B",
        help="HuggingFace model ID (default: Qwen/Qwen3-1.7B)",
    )
    parser.add_argument(
        "--transcripts-dir",
        type=str,
        default="asr_transcripts/cache/asr_transcripts",
        help="Input transcripts directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="asr_transcripts/cache/asr_transcripts",
        help="Output directory",
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default="cache/refinement_manifest.json",
        help="Final merged manifest path",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=12,
        help="Segments per prompt chunk (default: 12)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=4096,
        help="Max generation tokens (default: 4096)",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=2,
        help="Number of GPUs to use (default: 2)",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        choices=[None, "4bit"],
        help="Quantization mode (e.g. 4bit)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-refinement even if marked completed in manifest",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.manifest_path)), exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 65)
    print(f"🚀 Launching Dual-GPU Refinement on {args.num_gpus} GPUs")
    print(f"Model          : {args.model_id}")
    print(f"Quantization   : {args.quantization or 'FP16/BF16'}")
    print(f"Transcripts Dir: {args.transcripts_dir}")
    print(f"Batch Size     : {args.batch_size}")
    print("=" * 65)

    processes = []
    shard_manifests = []
    start_time = time.time()

    for gpu_idx in range(args.num_gpus):
        shard_manifest = f"cache/manifest_gpu{gpu_idx}.json"
        shard_manifests.append(shard_manifest)

        cmd = [
            sys.executable,
            "scripts/refine_transcripts_qwen.py",
            "--model-id", args.model_id,
            "--transcripts-dir", args.transcripts_dir,
            "--output-dir", args.output_dir,
            "--manifest-path", shard_manifest,
            "--batch-size", str(args.batch_size),
            "--max-new-tokens", str(args.max_new_tokens),
            "--num-shards", str(args.num_gpus),
            "--shard-id", str(gpu_idx),
            "--device", f"cuda:{gpu_idx}",
        ]
        if args.quantization:
            cmd.extend(["--quantization", args.quantization])
        if args.force:
            cmd.append("--force")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)

        print(f"[GPU {gpu_idx}] Starting worker (shard {gpu_idx}/{args.num_gpus})...")
        p = subprocess.Popen(cmd, env=env)
        processes.append((gpu_idx, p))

    # Wait for all processes to complete
    failed = False
    for gpu_idx, p in processes:
        ret = p.wait()
        if ret != 0:
            print(f"[!] Worker GPU {gpu_idx} exited with error code {ret}", file=sys.stderr)
            failed = True
        else:
            print(f"[✓] Worker GPU {gpu_idx} completed successfully.")

    elapsed = time.time() - start_time
    print(f"\nAll GPU workers finished in {elapsed / 60:.2f} minutes.")

    # Merge manifests
    print("\nMerging shard manifests into unified ledger...")
    merge_cmd = [
        sys.executable,
        "scripts/refine_transcripts_qwen.py",
        "--manifest-path", args.manifest_path,
        "--merge-manifests", *shard_manifests,
    ]
    subprocess.run(merge_cmd, check=True)
    print("🎉 Dual-GPU refinement workflow finished successfully!")


if __name__ == "__main__":
    main()
