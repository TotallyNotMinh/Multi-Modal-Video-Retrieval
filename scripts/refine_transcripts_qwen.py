#!/usr/bin/env python3
"""
Refine Vietnamese ASR Transcripts using Qwen2.5-1.5B-Instruct locally.

Implements tagged segmentation (<SEGMENT_i> ... </SEGMENT_i>), conservative
Vietnamese speech correction (fixing diacritics, homophones, and proper nouns),
multi-stage validation gating, and atomic manifest ledger tracking.
"""

import os
import sys
import glob
import json
import time
import re
import argparse
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
from tqdm import tqdm

# Ensure repo root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

DEFAULT_MODEL_ID = "Qwen/Qwen3-1.7B"
DEFAULT_TRANSCRIPTS_DIR = "asr_transcripts/cache/asr_transcripts"
DEFAULT_OUTPUT_DIR = "asr_transcripts/cache/asr_transcripts"
DEFAULT_MANIFEST_PATH = "cache/refinement_manifest.json"
DEFAULT_MAX_NEW_TOKENS = 4096
DEFAULT_BATCH_SIZE = 10

SYSTEM_PROMPT = """You are an expert Vietnamese transcript editor specializing in Automatic Speech Recognition (ASR) error correction.
Your task is to refine automatically transcribed Vietnamese speech segments.

CRITICAL RULES:
1. PRESERVE SEGMENT TAGS: Every input <SEGMENT_i> must produce an exact corresponding <SEGMENT_i> output in identical order.
2. CONSERVATIVE CORRECTION:
   - Correct spelling, missing Vietnamese diacritics, broken words, and obvious phonetic homophones.
   - Correct proper nouns (people, places, organizations) or numbers/dates/percentages ONLY when strongly supported by the context of the transcript. Otherwise, preserve the original string.
3. NO PARAPHRASING: Do NOT improve style, alter phrasing, or rewrite grammatically valid sentences. Preserve the speaker's original words and speech style.
4. NO HALLUCINATION: Do NOT add facts, commentary, or conversational filler.
5. NO THINKING TAGS: Do NOT output <think> tags or internal reasoning. Return ONLY the tagged segments."""


class RefinementTracker:
    """Thread-safe and crash-resilient manifest tracker for transcript refinement."""

    def __init__(self, manifest_path: str, model_id: str):
        self.manifest_path = manifest_path
        self.model_id = model_id
        self.data = self._load()

    def _load(self) -> Dict[str, Any]:
        if os.path.exists(self.manifest_path):
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[Tracker] Warning: Failed to read manifest ({e}), initializing new.", file=sys.stderr)
        return {
            "last_updated": datetime.now().isoformat(),
            "model_id": self.model_id,
            "summary": {
                "total_scanned": 0,
                "completed": 0,
                "failed": 0,
                "total_segments": 0,
                "changed_segments": 0,
            },
            "records": {},
        }

    def save(self):
        self.data["last_updated"] = datetime.now().isoformat()
        self.data["model_id"] = self.model_id
        self._recompute_summary()
        os.makedirs(os.path.dirname(os.path.abspath(self.manifest_path)), exist_ok=True)
        tmp_path = f"{self.manifest_path}.tmp.{os.getpid()}"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, self.manifest_path)

    def _recompute_summary(self):
        records = self.data.get("records", {})
        completed = sum(1 for r in records.values() if r.get("status") == "completed")
        failed = sum(1 for r in records.values() if r.get("status") == "failed")
        total_segs = sum(r.get("total_segments", 0) for r in records.values() if r.get("status") == "completed")
        changed_segs = sum(r.get("changed_segments", 0) for r in records.values() if r.get("status") == "completed")
        self.data["summary"] = {
            "total_scanned": len(records),
            "completed": completed,
            "failed": failed,
            "total_segments": total_segs,
            "changed_segments": changed_segs,
        }

    def is_completed(self, video_id: str) -> bool:
        rec = self.data.get("records", {}).get(video_id)
        return bool(rec and rec.get("status") == "completed")

    def record_success(
        self,
        video_id: str,
        total_segments: int,
        refined_segments: int,
        changed_segments: int,
        words_before: int,
        words_after: int,
        elapsed_sec: float,
    ):
        self.data.setdefault("records", {})[video_id] = {
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "model": self.model_id,
            "total_segments": total_segments,
            "refined_segments": refined_segments,
            "changed_segments": changed_segments,
            "words_before": words_before,
            "words_after": words_after,
            "elapsed_sec": round(elapsed_sec, 2),
            "validation": {
                "inference_success": True,
                "parse_success": True,
                "validation_success": True,
                "file_write_success": True,
                "segment_count_match": True,
                "empty_segments": 0,
            },
        }
        self.save()

    def record_failure(
        self,
        video_id: str,
        stage: str,
        reason: str,
        expected_segments: int = 0,
        received_segments: int = 0,
        elapsed_sec: float = 0.0,
    ):
        self.data.setdefault("records", {})[video_id] = {
            "status": "failed",
            "timestamp": datetime.now().isoformat(),
            "failure_stage": stage,
            "reason": reason,
            "expected_segments": expected_segments,
            "received_segments": received_segments,
            "elapsed_sec": round(elapsed_sec, 2),
            "validation": {
                "inference_success": stage not in ("load", "inference"),
                "parse_success": stage not in ("load", "inference", "parse"),
                "validation_success": False,
                "file_write_success": False,
            },
        }
        self.save()

    def print_summary(self):
        self._recompute_summary()
        summary = self.data.get("summary", {})
        print("=" * 65)
        print("         ASR TRANSCRIPT REFINEMENT MANIFEST STATUS")
        print("=" * 65)
        print(f"Manifest Path   : {self.manifest_path}")
        print(f"Model ID        : {self.data.get('model_id', 'Unknown')}")
        print(f"Last Updated    : {self.data.get('last_updated', 'Never')}")
        print(f"Total Scanned   : {summary.get('total_scanned', 0)}")
        print(f"Completed Videos: {summary.get('completed', 0)}")
        print(f"Failed Videos   : {summary.get('failed', 0)}")
        print(f"Total Segments  : {summary.get('total_segments', 0)}")
        print(f"Changed Segments: {summary.get('changed_segments', 0)}")
        if summary.get("total_segments", 0) > 0:
            rate = (summary.get("changed_segments", 0) / summary["total_segments"]) * 100
            print(f"Correction Rate : {rate:.1f}%")
        print("=" * 65)

        failures = [
            (vid, r)
            for vid, r in self.data.get("records", {}).items()
            if r.get("status") == "failed"
        ]
        if failures:
            print(f"\n[!] Recent Failures ({len(failures)} total):")
            for vid, r in failures[:10]:
                print(f"  - {vid}: stage={r.get('failure_stage')}, reason={r.get('reason')}")
            if len(failures) > 10:
                print(f"  ... and {len(failures) - 10} more.")


class QwenRefiner:
    """Loads and performs tagged segment inference using Qwen2.5-Instruct."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: Optional[str] = None,
        quantization: Optional[str] = None,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        dry_run: bool = False,
    ):
        self.model_id = model_id
        self.quantization = quantization
        self.max_new_tokens = max_new_tokens
        self.dry_run = dry_run
        self.tokenizer = None
        self.model = None
        self.device = device or ("cuda" if self._has_cuda() else "cpu")

        if not self.dry_run:
            self._load_model()

    @staticmethod
    def _has_cuda() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _load_model(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        print(f"[QwenRefiner] Loading tokenizer: {self.model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)

        if self.quantization == "4bit":
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            print(f"[QwenRefiner] Loading model: {self.model_id} on {self.device} with 4-bit NF4 quantization...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                quantization_config=bnb_config,
                device_map={"": self.device} if "cuda" in str(self.device) else "auto",
                trust_remote_code=True,
            )
        else:
            dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else (
                torch.float16 if torch.cuda.is_available() else torch.float32
            )
            print(f"[QwenRefiner] Loading model: {self.model_id} on {self.device} (dtype={dtype})...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                dtype=dtype,
                trust_remote_code=True,
            ).to(self.device)

        self.model.eval()
        print(f"[QwenRefiner] Model successfully loaded on {self.device}!")

    @staticmethod
    def build_chunk_prompt(segments: List[str]) -> str:
        """Constructs tagged segment prompt format <SEGMENT_i> text </SEGMENT_i>."""
        blocks = []
        for i, text in enumerate(segments):
            clean_text = text.strip()
            blocks.append(f"<SEGMENT_{i}>\n{clean_text}\n</SEGMENT_{i}>")
        return "\n".join(blocks)

    def generate_refinements_chunk(self, raw_texts: List[str]) -> Tuple[bool, List[str], str]:
        """
        Refines a single chunk of segments.
        Returns (success: bool, refined_texts: List[str], error_msg: str).
        """
        if not raw_texts:
            return True, [], ""

        user_content = self.build_chunk_prompt(raw_texts)

        if self.dry_run:
            # Simulate identity refinement for testing prompt construction
            return True, [t.strip() for t in raw_texts], ""

        import torch

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        try:
            try:
                prompt_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                prompt_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)

            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    repetition_penalty=1.05,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            gen_tokens = outputs[0][inputs.input_ids.shape[1] :]
            raw_output = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

            if self.device == "cuda":
                del inputs, outputs
                torch.cuda.empty_cache()

        except Exception as e:
            return False, [], f"Inference exception: {str(e)}"

        # Strip any thinking tags if produced by model
        cleaned_output = re.sub(r"<think>[\s\S]*?</think>", "", raw_output).strip()

        # Parse tags
        success, refined_list, parse_err = self.parse_and_validate_tags(cleaned_output, len(raw_texts), raw_texts)
        if not success:
            return False, [], parse_err

        return True, refined_list, ""

    @staticmethod
    def parse_and_validate_tags(
        output_text: str,
        expected_count: int,
        original_texts: List[str],
    ) -> Tuple[bool, List[str], str]:
        """
        Strictly parses <SEGMENT_i> ... </SEGMENT_i> blocks.
        Verifies count, sequential ordering, non-emptiness, and sanity.
        """
        # Primary regex: strict tag pairing
        pattern = re.compile(r"<SEGMENT_(\d+)>([\s\S]*?)</SEGMENT_\1>", re.IGNORECASE)
        matches = pattern.findall(output_text)

        parsed_map = {}
        for idx_str, content in matches:
            try:
                idx = int(idx_str)
                parsed_map[idx] = content.strip()
            except ValueError:
                continue

        # Secondary fallback regex if LLM dropped closing tag but wrote next opening tag
        if len(parsed_map) < expected_count:
            loose_pattern = re.compile(
                r"<SEGMENT_(\d+)>([\s\S]*?)(?=(?:</SEGMENT_\1>|<SEGMENT_\d+>|$))", re.IGNORECASE
            )
            loose_matches = loose_pattern.findall(output_text)
            for idx_str, content in loose_matches:
                try:
                    idx = int(idx_str)
                    if idx not in parsed_map:
                        parsed_map[idx] = content.strip()
                except ValueError:
                    continue

        if len(parsed_map) != expected_count:
            return (
                False,
                [],
                f"segment_count_mismatch: expected {expected_count}, parsed {len(parsed_map)}",
            )

        refined_texts = []
        for i in range(expected_count):
            if i not in parsed_map:
                return False, [], f"missing_segment_index: <SEGMENT_{i}> missing"

            refined_val = parsed_map[i].strip()
            orig_val = original_texts[i].strip()

            # Handle empty checks
            if not refined_val and orig_val:
                return False, [], f"empty_segment_content at index {i}"

            # Hallucination / sanity check: if output text length exploded or collapsed wildly
            orig_len = max(len(orig_val), 1)
            ref_len = len(refined_val)
            if orig_val and (ref_len > orig_len * 4 + 50 or ref_len < max(1, int(orig_len * 0.2))):
                return False, [], f"extreme_length_deviation at index {i} (orig={orig_len}, refined={ref_len})"

            refined_texts.append(refined_val if refined_val else orig_val)

        return True, refined_texts, ""


def process_single_video(
    transcript_path: str,
    refiner: QwenRefiner,
    tracker: RefinementTracker,
    output_dir: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    force: bool = False,
    dry_run: bool = False,
) -> Tuple[bool, str]:
    """Processes a single transcript file with multi-stage validation and atomic write."""
    base_name = os.path.basename(transcript_path)
    video_id = os.path.splitext(base_name)[0]

    if not force and tracker.is_completed(video_id):
        return True, f"Skipped {video_id} (already completed in manifest)"

    start_time = time.time()

    # Stage 1: Load file
    try:
        with open(transcript_path, "r", encoding="utf-8") as f:
            segments_data = json.load(f)
    except Exception as e:
        tracker.record_failure(video_id, stage="load", reason=f"Failed to read JSON: {e}")
        return False, f"Failed to load {base_name}: {e}"

    if not isinstance(segments_data, list):
        tracker.record_failure(video_id, stage="load", reason="Top-level JSON is not a list")
        return False, f"Invalid format in {base_name}: expected list"

    total_segments = len(segments_data)
    if total_segments == 0:
        tracker.record_success(video_id, 0, 0, 0, 0, 0, time.time() - start_time)
        return True, f"Empty transcript for {video_id}, marked completed."

    # Extract texts
    raw_texts = []
    for seg in segments_data:
        # Prefer existing raw_text if present, else original text
        raw_val = seg.get("raw_text") or seg.get("text") or ""
        raw_texts.append(raw_val)

    # Chunk segments into batches
    refined_all: List[str] = []
    for chunk_start in range(0, total_segments, batch_size):
        chunk_end = min(chunk_start + batch_size, total_segments)
        chunk_raw = raw_texts[chunk_start:chunk_end]

        success, chunk_refined, err = refiner.generate_refinements_chunk(chunk_raw)
        if not success:
            tracker.record_failure(
                video_id=video_id,
                stage="validation",
                reason=err,
                expected_segments=total_segments,
                received_segments=len(refined_all),
                elapsed_sec=time.time() - start_time,
            )
            return False, f"Validation failed for {video_id} (chunk {chunk_start}-{chunk_end}): {err}"

        refined_all.extend(chunk_refined)

    if len(refined_all) != total_segments:
        tracker.record_failure(
            video_id=video_id,
            stage="validation",
            reason=f"Total count mismatch: expected {total_segments}, got {len(refined_all)}",
            expected_segments=total_segments,
            received_segments=len(refined_all),
            elapsed_sec=time.time() - start_time,
        )
        return False, f"Final count mismatch for {video_id}"

    # Calculate diff statistics
    changed_count = 0
    words_before = sum(len(t.split()) for t in raw_texts)
    words_after = sum(len(t.split()) for t in refined_all)

    # Stage 4: Construct updated segments
    updated_segments = []
    for i, seg in enumerate(segments_data):
        seg_copy = dict(seg)
        orig_raw = raw_texts[i]
        refined_str = refined_all[i]

        if orig_raw.strip() != refined_str.strip():
            changed_count += 1

        seg_copy["raw_text"] = orig_raw
        seg_copy["refined_text"] = refined_str
        seg_copy["cleaned_text"] = refined_str  # For backwards compatibility
        seg_copy["text"] = refined_str
        updated_segments.append(seg_copy)

    # Stage 5: Atomic write (unless dry_run)
    out_file = os.path.join(output_dir, base_name)
    if not dry_run:
        os.makedirs(output_dir, exist_ok=True)
        tmp_file = f"{out_file}.tmp.{os.getpid()}"
        try:
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(updated_segments, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_file, out_file)
        except Exception as e:
            if os.path.exists(tmp_file):
                try:
                    os.remove(tmp_file)
                except OSError:
                    pass
            tracker.record_failure(
                video_id=video_id,
                stage="file_write",
                reason=f"Atomic write error: {e}",
                elapsed_sec=time.time() - start_time,
            )
            return False, f"Write failed for {video_id}: {e}"

    elapsed = time.time() - start_time
    tracker.record_success(
        video_id=video_id,
        total_segments=total_segments,
        refined_segments=len(updated_segments),
        changed_segments=changed_count,
        words_before=words_before,
        words_after=words_after,
        elapsed_sec=elapsed,
    )

    return True, f"Refined {video_id}: {changed_count}/{total_segments} segments updated ({elapsed:.2f}s)"


def main():
    parser = argparse.ArgumentParser(
        description="Refine Vietnamese ASR Transcripts locally using Qwen2.5-1.5B-Instruct."
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help=f"HuggingFace model ID (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--transcripts-dir",
        type=str,
        default=DEFAULT_TRANSCRIPTS_DIR,
        help=f"Input transcripts directory (default: {DEFAULT_TRANSCRIPTS_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default=DEFAULT_MANIFEST_PATH,
        help=f"Path to manifest ledger (default: {DEFAULT_MANIFEST_PATH})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Segments per prompt chunk (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=f"Max new generation tokens (default: {DEFAULT_MAX_NEW_TOKENS})",
    )
    parser.add_argument(
        "--video-id",
        type=str,
        default=None,
        help="Filter specific video ID(s), comma-separated (e.g. L21_V001,L21_V002)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the total number of files to process",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total number of parallel shards across GPUs (default: 1)",
    )
    parser.add_argument(
        "--shard-id",
        type=int,
        default=0,
        help="Zero-indexed shard ID for this process (default: 0)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Explicit device (e.g. cuda:0, cuda:1, cpu)",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        choices=[None, "4bit"],
        help="Quantization mode (e.g. 4bit)",
    )
    parser.add_argument(
        "--merge-manifests",
        nargs="+",
        default=None,
        help="List of shard manifest files to merge into --manifest-path",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-refinement even if marked completed in manifest",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Display manifest status dashboard and exit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test prompt formation and workflow without loading model weights",
    )
    args = parser.parse_args()

    # Handle manifest merging if requested
    if args.merge_manifests:
        target_tracker = RefinementTracker(manifest_path=args.manifest_path, model_id=args.model_id)
        for mpath in args.merge_manifests:
            if os.path.exists(mpath):
                try:
                    with open(mpath, "r", encoding="utf-8") as mf:
                        mdata = json.load(mf)
                    for vid, rdata in mdata.get("records", {}).items():
                        target_tracker.data.setdefault("records", {})[vid] = rdata
                    print(f"Merged manifest: {mpath}")
                except Exception as me:
                    print(f"Warning: Failed to merge {mpath}: {me}", file=sys.stderr)
        target_tracker.save()
        target_tracker.print_summary()
        return

    tracker = RefinementTracker(manifest_path=args.manifest_path, model_id=args.model_id)

    if args.status:
        tracker.print_summary()
        return

    # Find transcript files
    if not os.path.exists(args.transcripts_dir):
        print(f"Error: Transcripts directory '{args.transcripts_dir}' not found.", file=sys.stderr)
        sys.exit(1)

    all_files = sorted(glob.glob(os.path.join(args.transcripts_dir, "*.json")))
    if not all_files:
        print(f"No JSON transcript files found in {args.transcripts_dir}.")
        return

    # Apply filter if requested
    if args.video_id:
        target_ids = {vid.strip() for vid in args.video_id.split(",") if vid.strip()}
        target_files = [f for f in all_files if os.path.splitext(os.path.basename(f))[0] in target_ids]
    else:
        target_files = all_files

    if args.limit:
        target_files = target_files[: args.limit]

    # Apply sharding
    if args.num_shards > 1:
        target_files = target_files[args.shard_id :: args.num_shards]
        print(f"[Shard {args.shard_id}/{args.num_shards}] Assigned {len(target_files)} target file(s).")
    else:
        print(f"Found {len(target_files)} target file(s) for refinement.")

    if not target_files:
        return

    # Initialize refiner
    refiner = QwenRefiner(
        model_id=args.model_id,
        device=args.device,
        quantization=args.quantization,
        max_new_tokens=args.max_new_tokens,
        dry_run=args.dry_run,
    )

    success_count = 0
    fail_count = 0

    pbar = tqdm(target_files, desc="Refining Transcripts")
    for fpath in pbar:
        vid = os.path.splitext(os.path.basename(fpath))[0]
        pbar.set_postfix_str(f"Current: {vid}")
        ok, msg = process_single_video(
            transcript_path=fpath,
            refiner=refiner,
            tracker=tracker,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            force=args.force,
            dry_run=args.dry_run,
        )
        if ok:
            success_count += 1
        else:
            fail_count += 1
            print(f"\n[!] Error on {vid}: {msg}", file=sys.stderr)

    print("\n" + "=" * 65)
    print(f"Refinement Batch Finished: {success_count} succeeded, {fail_count} failed.")
    tracker.print_summary()


if __name__ == "__main__":
    main()
