#!/usr/bin/env python3
"""
MiMo-V2.5 Free Vietnamese ASR Transcript Refinement Pipeline.

Sends complete video transcripts (preserving temporal segment boundaries via [SEGMENT i] markers)
to an OpenAI-compatible OmniRoute endpoint with strict alignment validation, exponential backoff,
atomic checkpointing, and comprehensive reporting.
"""

import os
import sys
import glob
import json
import time
import re
import math
import random
import argparse
import urllib.request
import urllib.error
from datetime import datetime, timezone
from typing import List, Dict, Tuple, Optional, Any, Set

# Ensure repo root is in sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

DEFAULT_INPUT_DIR = "cache/asr_transcripts"
DEFAULT_OUTPUT_DIR = "cache/asr_transcripts_refined"
DEFAULT_MANIFEST_PATH = "cache/refinement_mimo_manifest.json"
DEFAULT_API_BASE = "http://localhost:20128/v1"
DEFAULT_MODEL = "mimo-v2.5-free"
DEFAULT_TIMEOUT = 60
DEFAULT_MAX_RETRIES = 5

SYSTEM_PROMPT = """You are an expert Vietnamese transcript editor specializing in Automatic Speech Recognition (ASR) error correction.
Your task is to refine automatically transcribed Vietnamese speech segments from video audio.

CRITICAL REFINEMENT RULES:
1. Conservative Error Correction:
   - Correct obvious Vietnamese ASR recognition errors, homophones, spelling mistakes, missing or misplaced tone diacritics, punctuation, and capitalization.
   - Correct obvious word-boundary errors (e.g. erroneously split compound words) and clearly identifiable Vietnamese proper names/geographical entities when supported by surrounding context.
   - Preserve the exact original meaning and factual information (dates, numbers, statistics, names) unless the correction is highly certain.
2. Zero Paraphrasing / Zero Hallucination:
   - DO NOT summarize, paraphrase, or rephrase fluent speech.
   - DO NOT add new information or remove existing information.
   - DO NOT invent words or rewrite sentences to make them sound "better" or more poetic.
   - If a word or phrase is unclear or uncertain, leave the original text completely unchanged.
3. Strict Output Structure & Segment Integrity:
   - The input consists of numbered segments marked by `[SEGMENT i]`.
   - You MUST output EVERY segment in exact numerical order, starting from `[SEGMENT 0]` up to `[SEGMENT N-1]`.
   - Precede each segment's refined text with its exact marker `[SEGMENT i]`.
   - DO NOT skip, merge, reorder, or duplicate any segment markers.
   - DO NOT include conversational filler, preamble, commentary, explanations, or Markdown code fences (e.g. do NOT write "Here is the refined transcript" or "```").
   - Output ONLY the formatted segment markers and their refined text."""

STRICT_RETRY_REMINDER = """CRITICAL: Your previous response failed structural validation (missing or disordered segment tags).
You MUST output all segments strictly starting from [SEGMENT 0] up to [SEGMENT {max_id}] with no missing, duplicated, or reordered markers.
Output ONLY the segment markers and refined Vietnamese text."""


class TranscriptDataset:
    """Discovers, loads, and manages ASR transcript files."""

    def __init__(self, input_dir: str):
        self.input_dir = input_dir
        self.files = sorted(glob.glob(os.path.join(input_dir, "*.json")))

    def __len__(self) -> int:
        return len(self.files)

    def load_video(self, file_path: str) -> Tuple[str, List[Dict[str, Any]]]:
        vid_id = os.path.basename(file_path).replace(".json", "")
        with open(file_path, "r", encoding="utf-8") as f:
            segments = json.load(f)
        return vid_id, segments

    def get_all_video_ids(self) -> List[str]:
        return [os.path.basename(f).replace(".json", "") for f in self.files]

    def get_validation_subset_ids(self) -> List[str]:
        """Curates exactly 30 validation videos: 10 short/median, 10 average, 10 long (including longest)."""
        video_lengths = []
        for f in self.files:
            vid, segs = self.load_video(f)
            text = " ".join([s.get("text", "").strip() for s in segs if s.get("text")])
            words = len(text.split())
            if words > 0:
                video_lengths.append((vid, words, len(segs)))

        if not video_lengths:
            return []

        video_lengths.sort(key=lambda x: x[1])
        n = len(video_lengths)

        # 10 Short/Median (p10 to p50)
        short_indices = [int(n * p) for p in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.48, 0.50]]
        short_median = [video_lengths[i][0] for i in short_indices]

        # 10 Average (closest to 1565 words)
        avg_target = 1565
        by_avg = sorted(video_lengths, key=lambda x: abs(x[1] - avg_target))
        average_samples = [x[0] for x in sorted(by_avg[:10], key=lambda x: x[1])]

        # 10 Long (top percentiles, ensuring longest is included)
        long_samples = [video_lengths[-(i * 5 + 1)][0] for i in range(9, -1, -1)]
        if video_lengths[-1][0] not in long_samples:
            long_samples[-1] = video_lengths[-1][0]

        # Combine and preserve unique list of 30
        selected = []
        for v in short_median + average_samples + long_samples:
            if v not in selected:
                selected.append(v)

        return selected[:30]


class OmniRouteClient:
    """HTTP client communicating with OmniRoute SSE streaming API."""

    def __init__(self, api_base: str, model: str, api_key: Optional[str] = None, timeout: int = 180):
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.api_key = api_key or os.environ.get("OMNIROUTE_API_KEY", "")
        self.timeout = timeout

    def call_stream(self, messages: List[Dict[str, str]], temperature: float = 0.0) -> Tuple[str, Dict[str, Any]]:
        """Makes a streaming chat completion request and parses pure SSE content chunks."""
        url = f"{self.api_base}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "temperature": temperature,
        }
        data_bytes = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(url, data=data_bytes, headers=headers, method="POST")

        t0 = time.time()
        collected_chunks = []
        usage_info = {}

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line or line.startswith(":"):
                        continue
                    if line == "data: [DONE]":
                        break
                    if line.startswith("data: "):
                        chunk_str = line[6:].strip()
                        try:
                            chunk = json.loads(chunk_str)
                            # Check for upstream errors formatted inside JSON
                            if "error" in chunk:
                                err = chunk["error"]
                                err_msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
                                err_code = err.get("code", "") if isinstance(err, dict) else ""
                                raise RuntimeError(f"OmniRoute stream error ({err_code}): {err_msg}")

                            if "usage" in chunk and chunk["usage"]:
                                usage_info = chunk["usage"]

                            choices = chunk.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    collected_chunks.append(content)
                        except json.JSONDecodeError:
                            continue
        except urllib.error.HTTPError as e:
            err_body = ""
            try:
                err_body = e.read().decode("utf-8", errors="replace")[:300]
            except Exception:
                pass
            raise RuntimeError(f"HTTP {e.code}: {e.reason} | {err_body}")
        except urllib.error.URLError as e:
            raise RuntimeError(f"Connection error: {e.reason}")

        latency = time.time() - t0
        full_text = "".join(collected_chunks).strip()
        metadata = {
            "latency": latency,
            "usage": usage_info,
            "model": self.model,
        }
        return full_text, metadata


class SegmentValidator:
    """Validates structural and alignment integrity of refined transcripts."""

    SEGMENT_PATTERN = re.compile(r"\[SEGMENT\s+(\d+)\]\s*([\s\S]*?)(?=(?:\[SEGMENT\s+\d+\]|\Z))")

    @classmethod
    def parse_segments(cls, text: str) -> Dict[int, str]:
        """Extracts {segment_id: text} mappings from response."""
        matches = cls.SEGMENT_PATTERN.findall(text)
        result = {}
        for seg_id_str, seg_content in matches:
            seg_id = int(seg_id_str)
            # If duplicated, record last or track
            result[seg_id] = seg_content.strip()
        return result

    @classmethod
    def validate(
        cls,
        raw_output: str,
        input_segments: List[Dict[str, Any]]
    ) -> Tuple[bool, Optional[str], Optional[Dict[int, str]], Dict[str, Any]]:
        """
        Validates:
        - Output non-empty
        - No model conversational preamble / commentary
        - Every input segment ID [0 .. N-1] present exactly once in sequential order
        - No extra IDs
        - Ratio flags
        """
        metrics = {
            "char_ratio": 1.0,
            "word_ratio": 1.0,
            "suspicious_length": False,
            "changed_count": 0,
        }

        if not raw_output or not raw_output.strip():
            return False, "Empty model response", None, metrics

        # Check for forbidden conversational boilerplate at start
        lower_head = raw_output[:120].lower()
        forbidden_phrases = [
            "here is", "dưới đây là", "bản ghi", "corrected transcript",
            "chắc chắn rồi", "tất nhiên", "refined transcript"
        ]
        for phrase in forbidden_phrases:
            if phrase in lower_head and not lower_head.startswith("[segment 0]"):
                return False, f"Conversational preamble detected: {raw_output[:80]!r}", None, metrics

        matches = cls.SEGMENT_PATTERN.findall(raw_output)
        if not matches:
            return False, "No [SEGMENT i] markers found in output", None, metrics

        matched_ids = [int(m[0]) for m in matches]
        expected_n = len(input_segments)
        expected_ids = list(range(expected_n))

        if matched_ids != expected_ids:
            # Analyze discrepancy
            missing = set(expected_ids) - set(matched_ids)
            extras = set(matched_ids) - set(expected_ids)
            duplicates = [x for x in matched_ids if matched_ids.count(x) > 1]
            discrepancy = []
            if missing:
                discrepancy.append(f"Missing IDs: {sorted(missing)[:5]}")
            if extras:
                discrepancy.append(f"Extra IDs: {sorted(extras)[:5]}")
            if duplicates:
                discrepancy.append(f"Duplicate IDs: {sorted(set(duplicates))[:5]}")
            if not missing and not extras and not duplicates and matched_ids != expected_ids:
                discrepancy.append("IDs out of order")

            return False, f"Segment mismatch ({len(matched_ids)} vs {expected_n} expected): {', '.join(discrepancy)}", None, metrics

        parsed = {}
        for m in matches:
            seg_id = int(m[0])
            parsed[seg_id] = m[1].strip()

        # Compute lengths & differences
        orig_chars = sum(len(s.get("text", "").strip()) for s in input_segments)
        ref_chars = sum(len(t) for t in parsed.values())
        orig_words = sum(len(s.get("text", "").strip().split()) for s in input_segments)
        ref_words = sum(len(t.split()) for t in parsed.values())

        metrics["char_ratio"] = round(ref_chars / max(orig_chars, 1), 4)
        metrics["word_ratio"] = round(ref_words / max(orig_words, 1), 4)

        changed = 0
        for i, s in enumerate(input_segments):
            orig_t = s.get("text", "").strip()
            ref_t = parsed.get(i, "")
            if orig_t != ref_t:
                changed += 1
        metrics["changed_count"] = changed

        # Flag suspicious length deviation (warning only, not failure)
        if metrics["char_ratio"] < 0.65 or metrics["char_ratio"] > 1.35:
            metrics["suspicious_length"] = True

        return True, None, parsed, metrics


class RefinementPipeline:
    """Orchestrates the transcript refinement workflow with atomic checkpointing."""

    def __init__(
        self,
        input_dir: str = DEFAULT_INPUT_DIR,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        manifest_path: str = DEFAULT_MANIFEST_PATH,
        api_base: str = DEFAULT_API_BASE,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        timeout: int = DEFAULT_TIMEOUT,
        verbose: bool = False,
    ):
        self.dataset = TranscriptDataset(input_dir)
        self.output_dir = output_dir
        self.manifest_path = manifest_path
        self.client = OmniRouteClient(api_base, model, api_key, timeout)
        self.max_retries = max_retries
        self.verbose = verbose
        self.manifest = self._load_manifest()

        os.makedirs(self.output_dir, exist_ok=True)
        manifest_dir = os.path.dirname(self.manifest_path)
        if manifest_dir:
            os.makedirs(manifest_dir, exist_ok=True)

    def _load_manifest(self) -> Dict[str, Any]:
        if os.path.exists(self.manifest_path):
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"version": 1, "model": self.client.model, "videos": {}}

    def _save_manifest(self):
        temp_path = f"{self.manifest_path}.tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, ensure_ascii=False, indent=2)
        os.replace(temp_path, self.manifest_path)

    def format_input_prompt(self, segments: List[Dict[str, Any]]) -> str:
        """Formats segments into [SEGMENT i] blocks."""
        blocks = []
        for i, seg in enumerate(segments):
            text = seg.get("text", "").strip()
            blocks.append(f"[SEGMENT {i}]\n{text}")
        return "\n\n".join(blocks)

    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation (approx 1.3 tokens per whitespace word for Vietnamese)."""
        words = len(text.split())
        return int(words * 1.35) + 10

    def refine_single_video(self, video_id: str, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Refines one video transcript with exponential backoff and validation checks."""
        # Handle empty/silent transcripts
        non_empty = [s for s in segments if s.get("text", "").strip()]
        if not non_empty:
            refined_data = []
            for i, s in enumerate(segments):
                refined_data.append({
                    "video_id": video_id,
                    "segment_id": i,
                    "start_sec": s.get("start_sec", 0.0),
                    "end_sec": s.get("end_sec", 0.0),
                    "start_frame": s.get("start_frame", 0),
                    "end_frame": s.get("end_frame", 0),
                    "original_text": s.get("text", "").strip(),
                    "refined_text": s.get("text", "").strip(),
                })
            out_file = os.path.join(self.output_dir, f"{video_id}.json")
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(refined_data, f, ensure_ascii=False, indent=2)

            return {
                "video_id": video_id,
                "status": "completed_empty",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model": self.client.model,
                "input_tokens_est": 0,
                "output_tokens_est": 0,
                "total_tokens_est": 0,
                "latency_sec": 0.0,
                "segment_count": len(segments),
                "retries": 0,
                "char_ratio": 1.0,
                "word_ratio": 1.0,
                "changed_segments": 0,
                "error": None,
            }

        input_text = self.format_input_prompt(segments)
        input_tokens_est = self.estimate_tokens(SYSTEM_PROMPT) + self.estimate_tokens(input_text)
        max_id = len(segments) - 1

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": input_text},
        ]

        attempt = 0
        last_error = None
        backoff_delay = 2.0

        while attempt <= self.max_retries:
            attempt += 1
            try:
                raw_output, meta = self.client.call_stream(messages)
                valid, err_msg, parsed_dict, metrics = SegmentValidator.validate(raw_output, segments)

                if valid and parsed_dict is not None:
                    # Construct clean output
                    refined_data = []
                    for i, s in enumerate(segments):
                        refined_data.append({
                            "video_id": video_id,
                            "segment_id": i,
                            "start_sec": s.get("start_sec", 0.0),
                            "end_sec": s.get("end_sec", 0.0),
                            "start_frame": s.get("start_frame", 0),
                            "end_frame": s.get("end_frame", 0),
                            "original_text": s.get("text", "").strip(),
                            "refined_text": parsed_dict.get(i, s.get("text", "").strip()),
                        })

                    out_file = os.path.join(self.output_dir, f"{video_id}.json")
                    with open(out_file, "w", encoding="utf-8") as f:
                        json.dump(refined_data, f, ensure_ascii=False, indent=2)

                    output_tokens_est = self.estimate_tokens(raw_output)
                    usage = meta.get("usage", {})
                    in_tok = usage.get("prompt_tokens", input_tokens_est)
                    out_tok = usage.get("completion_tokens", output_tokens_est)

                    return {
                        "video_id": video_id,
                        "status": "completed",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "model": self.client.model,
                        "input_tokens_est": in_tok,
                        "output_tokens_est": out_tok,
                        "total_tokens_est": in_tok + out_tok,
                        "latency_sec": round(meta.get("latency", 0.0), 2),
                        "segment_count": len(segments),
                        "retries": attempt - 1,
                        "char_ratio": metrics["char_ratio"],
                        "word_ratio": metrics["word_ratio"],
                        "changed_segments": metrics["changed_count"],
                        "suspicious_length": metrics["suspicious_length"],
                        "error": None,
                    }
                else:
                    last_error = f"Validation failed: {err_msg}"
                    # Prepare strict retry prompt
                    if attempt <= self.max_retries:
                        messages = [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": input_text},
                            {"role": "assistant", "content": raw_output[:300]},
                            {"role": "user", "content": STRICT_RETRY_REMINDER.format(max_id=max_id)},
                        ]

            except Exception as e:
                last_error = str(e)
                # Check for rate limit / lockout
                is_rate_limit = "429" in last_error or "lockout" in last_error.lower() or "quota" in last_error.lower() or "403" in last_error or "circuit breaker" in last_error.lower()

                if is_rate_limit:
                    sleep_time = round(backoff_delay + random.uniform(0.5, 1.5), 1)
                    if self.verbose or attempt > 1:
                        print(f"    ↳ [Attempt {attempt}/{self.max_retries}] Rate limit / upstream lockout ({last_error[:60]}). Backing off {sleep_time}s...", flush=True)
                    time.sleep(sleep_time)
                    backoff_delay = min(backoff_delay * 2.0, 30.0)
                else:
                    if self.verbose:
                        print(f"    ↳ [Attempt {attempt}/{self.max_retries}] Request error ({last_error[:60]}). Retrying in 1.5s...", flush=True)
                    time.sleep(1.5)

        return {
            "video_id": video_id,
            "status": "failed",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": self.client.model,
            "input_tokens_est": input_tokens_est,
            "output_tokens_est": 0,
            "total_tokens_est": input_tokens_est,
            "latency_sec": 0.0,
            "segment_count": len(segments),
            "retries": attempt - 1,
            "char_ratio": 0.0,
            "word_ratio": 0.0,
            "changed_segments": 0,
            "suspicious_length": False,
            "error": last_error,
        }

    def run_dry_run(self) -> Dict[str, Any]:
        """Performs a thorough dry-run over all discovered videos without API calls."""
        total_videos = len(self.dataset)
        total_segments = 0
        total_words = 0
        total_estimated_tokens = 0
        empty_videos = []
        long_transcripts = []
        video_stats = []

        print("\n" + "=" * 60)
        print("🔍 RUNNING DRY-RUN ACROSS DATASET")
        print("=" * 60)

        for fpath in self.dataset.files:
            vid_id, segments = self.dataset.load_video(fpath)
            full_text = " ".join([s.get("text", "").strip() for s in segments if s.get("text")])
            words = len(full_text.split())
            seg_count = len(segments)
            tokens_est = self.estimate_tokens(SYSTEM_PROMPT) + self.estimate_tokens(self.format_input_prompt(segments))

            total_segments += seg_count
            total_words += words
            total_estimated_tokens += tokens_est

            if words == 0:
                empty_videos.append(vid_id)
            if tokens_est > 6000:
                long_transcripts.append((vid_id, words, seg_count, tokens_est))

            video_stats.append({
                "video_id": vid_id,
                "segments": seg_count,
                "words": words,
                "tokens_est": tokens_est,
            })

        avg_words = round(total_words / max(total_videos, 1), 2)
        avg_tokens = round(total_estimated_tokens / max(total_videos, 1), 2)
        avg_segs = round(total_segments / max(total_videos, 1), 2)

        print(f"\n📊 Dataset Summary:")
        print(f"  • Total Videos Discovered: {total_videos:,}")
        print(f"  • Total Segments:          {total_segments:,} (avg {avg_segs:.1f}/video)")
        print(f"  • Total Words (Syllables): {total_words:,} (avg {avg_words:.1f}/video)")
        print(f"  • Est. Input Tokens Total: {total_estimated_tokens:,} (avg {avg_tokens:.1f}/video)")
        print(f"  • Empty / Silent Videos:   {len(empty_videos)} {empty_videos[:5]}...")
        print(f"  • Long Transcripts (>6k tokens): {len(long_transcripts)} videos")

        if long_transcripts:
            print("\n📌 Top Longest Transcripts:")
            long_transcripts.sort(key=lambda x: x[3], reverse=True)
            for vid, w, s, tok in long_transcripts[:8]:
                print(f"    - {vid}: {w:,} words | {s} segments | ~{tok:,} est. tokens")

        print("\n✅ Dry-run complete. All transcripts parsed successfully.")
        return {
            "total_videos": total_videos,
            "total_segments": total_segments,
            "total_words": total_words,
            "total_estimated_tokens": total_estimated_tokens,
            "empty_videos": empty_videos,
            "long_transcripts": long_transcripts,
        }

    def process_videos(
        self,
        video_ids: List[str],
        resume: bool = True,
        is_validation_run: bool = False
    ) -> List[Dict[str, Any]]:
        """Processes a list of video IDs sequentially with atomic checkpointing."""
        results = []
        total = len(video_ids)
        successful = 0
        failed = 0
        skipped = 0

        print("\n" + "=" * 60)
        mode_str = "VALIDATION SUBSET (30 VIDEOS)" if is_validation_run else "BATCH PROCESSING"
        print(f"🚀 STARTING {mode_str} | Model: {self.client.model}")
        print("=" * 60 + "\n")

        for idx, vid_id in enumerate(video_ids, 1):
            # Checkpoint check
            out_file = os.path.join(self.output_dir, f"{vid_id}.json")
            if resume:
                if vid_id in self.manifest.get("videos", {}):
                    prev_record = self.manifest["videos"][vid_id]
                    if prev_record.get("status") in ["completed", "completed_empty"]:
                        skipped += 1
                        results.append(prev_record)
                        print(f"[{idx:3d}/{total}] ⏩ {vid_id}: Skipped (already completed in manifest)", flush=True)
                        continue
                elif os.path.exists(out_file):
                    skipped += 1
                    print(f"[{idx:3d}/{total}] ⏩ {vid_id}: Skipped (already exists on disk)", flush=True)
                    continue

            fpath = os.path.join(self.dataset.input_dir, f"{vid_id}.json")
            if not os.path.exists(fpath):
                print(f"[{idx:3d}/{total}] ⚠️ {vid_id}: File not found ({fpath})", flush=True)
                continue

            _, segments = self.dataset.load_video(fpath)
            res = self.refine_single_video(vid_id, segments)
            results.append(res)

            # Atomic save to manifest
            self.manifest["videos"][vid_id] = res
            self._save_manifest()

            if res["status"] in ["completed", "completed_empty"]:
                successful += 1
                chg_str = f"changed {res['changed_segments']}/{res['segment_count']} segs"
                ratio_str = f"char_ratio={res['char_ratio']:.3f}"
                flag_str = " ⚠️ [LENGTH_FLAG]" if res.get("suspicious_length") else ""
                print(f"[{idx:3d}/{total}] ✅ {vid_id}: Success ({res['latency_sec']}s, {res['total_tokens_est']} tok, {chg_str}, {ratio_str}){flag_str}", flush=True)
            else:
                failed += 1
                print(f"[{idx:3d}/{total}] ❌ {vid_id}: Failed after {res['retries']} retries ({res['error']})", flush=True)

        return results


def print_validation_report(results: List[Dict[str, Any]], output_dir: str):
    """Prints a detailed markdown table and summary for the 30-video validation run."""
    print("\n" + "=" * 80)
    print("📋 30-VIDEO VALIDATION REPORT")
    print("=" * 80 + "\n")

    print("| Video ID | Status | Segments | Est. In Tok | Out Tok | Total Tok | Latency | Retries | Char Ratio | Changed Segs | Notes |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |")

    total_in = 0
    total_out = 0
    total_lat = 0.0
    success_count = 0
    fail_count = 0

    for r in results:
        vid = r["video_id"]
        stat = "✅ OK" if r["status"] in ["completed", "completed_empty"] else "❌ FAIL"
        segs = r.get("segment_count", 0)
        in_t = r.get("input_tokens_est", 0)
        out_t = r.get("output_tokens_est", 0)
        tot_t = r.get("total_tokens_est", 0)
        lat = r.get("latency_sec", 0.0)
        retries = r.get("retries", 0)
        c_ratio = r.get("char_ratio", 1.0)
        chg = r.get("changed_segments", 0)
        notes = []
        if r.get("suspicious_length"):
            notes.append("LengthFlag")
        if r.get("error"):
            notes.append(r["error"][:30])
        notes_str = "; ".join(notes) if notes else "Clean"

        if "completed" in r["status"]:
            success_count += 1
            total_in += in_t
            total_out += out_t
            total_lat += lat
        else:
            fail_count += 1

        print(f"| `{vid}` | {stat} | {segs} | {in_t:,} | {out_t:,} | {tot_t:,} | {lat:.1f}s | {retries} | {c_ratio:.3f} | {chg}/{segs} | {notes_str} |")

    avg_lat = round(total_lat / max(success_count, 1), 2)
    avg_tok = round((total_in + total_out) / max(success_count, 1), 1)

    print("\n📊 Summary Statistics:")
    print(f"  • Evaluated Videos:       {len(results)}")
    print(f"  • Successful:             {success_count} / {len(results)}")
    print(f"  • Failed:                 {fail_count}")
    print(f"  • Avg Latency / Video:    {avg_lat}s")
    print(f"  • Avg Tokens / Video:     {avg_tok}")
    print(f"  • Total Tokens Consumed:  {(total_in + total_out):,}")
    print(f"  • Refined Output Dir:     `{output_dir}`")
    print("\n🛑 Pipeline paused as requested. Manual inspection required before batch run.")


def main():
    parser = argparse.ArgumentParser(description="MiMo-V2.5 Free Vietnamese ASR Transcript Refinement")
    parser.add_argument("--input", default=DEFAULT_INPUT_DIR, help="Input directory of ASR transcripts")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR, help="Output directory for refined transcripts")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST_PATH, help="Path to checkpoint manifest JSON")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE, help="OmniRoute API base URL")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model identifier")
    parser.add_argument("--api-key", default=None, help="OmniRoute API key")
    parser.add_argument("--workers", type=int, default=1, help="Worker concurrency (default: 1)")
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES, help="Max retry attempts")
    parser.add_argument("--request-timeout", type=int, default=DEFAULT_TIMEOUT, help="Request timeout in seconds")
    parser.add_argument("--dry-run", action="store_true", help="Perform dry-run across dataset without API requests")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N videos")
    parser.add_argument("--validation-sample", action="store_true", help="Run the curated 30-video validation subset")
    parser.add_argument("--video-ids", default=None, help="Comma-separated video IDs or path to file")
    parser.add_argument("--no-resume", action="store_true", help="Do not resume/skip completed videos")
    parser.add_argument("--verbose", action="store_true", help="Verbose output mode")

    args = parser.parse_args()

    pipeline = RefinementPipeline(
        input_dir=args.input,
        output_dir=args.output,
        manifest_path=args.manifest,
        api_base=args.api_base,
        model=args.model,
        api_key=args.api_key,
        max_retries=args.max_retries,
        timeout=args.request_timeout,
        verbose=args.verbose,
    )

    if args.dry_run:
        pipeline.run_dry_run()
        return

    # Determine videos to process
    if args.validation_sample:
        video_ids = pipeline.dataset.get_validation_subset_ids()
        print(f"Selected {len(video_ids)} curated validation videos.")
        results = pipeline.process_videos(video_ids, resume=(not args.no_resume), is_validation_run=True)
        print_validation_report(results, args.output)
        return

    if args.video_ids:
        if os.path.exists(args.video_ids):
            with open(args.video_ids, "r", encoding="utf-8") as f:
                video_ids = [line.strip() for line in f if line.strip()]
        else:
            video_ids = [v.strip() for v in args.video_ids.split(",") if v.strip()]
    else:
        video_ids = pipeline.dataset.get_all_video_ids()

    if args.limit:
        video_ids = video_ids[: args.limit]

    results = pipeline.process_videos(video_ids, resume=(not args.no_resume), is_validation_run=(bool(args.validation_sample) or len(video_ids) <= 30))
    print_validation_report(results, args.output)


if __name__ == "__main__":
    main()
