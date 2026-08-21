#!/usr/bin/env python3
"""
Refine Vietnamese ASR Transcripts using Scene Context & On-Screen OCR via OpenRouter (DeepSeek).

Cross-references noisy speech-to-text (ASR) segments with on-screen OCR text (news banners,
lower-thirds, headlines, speaker names, locations, statistics) to accurately correct:
  - Misheard proper nouns (people, places, government organizations, laws)
  - Missing or incorrect Vietnamese diacritics and word segmentation
  - ASR phonetic homophone substitutions and number formatting

Ensures JSON schema preservation:
  - 'raw_text': Original unedited speech-to-text output.
  - 'cleaned_text': Refined, error-corrected text from DeepSeek.
  - 'text': Points to 'cleaned_text' for downstream BM25, CLIP, and retrieval search indexing.
"""

import os
import sys
import glob
import json
import time
import re
import argparse
import urllib.request
import urllib.error
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional, Set

# Ensure repo root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ==============================================================================
# Default Configuration
# ==============================================================================
DEFAULT_OPENROUTER_MODEL = "minimax/minimax-m3"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MAX_SEGMENTS_PER_BATCH = 30
DEFAULT_REQUEST_DELAY = 0.5
DEFAULT_WORKERS = 4
# ==============================================================================

SYSTEM_PROMPT = """You are an expert Vietnamese news transcript and audio-visual editor.
Your task is to refine and correct an automatically generated Vietnamese ASR (speech-to-text) transcript for a video scene.

You are provided with:
1. Scene On-Screen OCR Text: Extracted text from on-screen graphics, lower-thirds, name tags, titles, headlines, locations, and statistics displayed in this video scene. This serves as ground-truth anchor for proper nouns, names, places, organizations, numbers, and technical terms.
2. Scene ASR Transcript Segments: A list of timestamped speech segments that may contain phonetic errors, missing diacritics, broken words, or mistranscribed entities.

IMPORTANT EDITING RULES:
1. Cross-reference ASR speech segments with OCR text:
   - When ASR misrecognized names, locations, titles, organizations, or numbers that match the scene's OCR banners, correct them using the authoritative spelling from OCR.
   - Example: ASR "Cường Thơ" -> OCR "CẦN THƠ" => correct to "Cần Thơ".
   - Example: ASR "ôm mồn, thúc nốt" -> OCR "Ô MÔN, THỐT NỐT" => correct to "Ô Môn, Thốt Nốt".
2. Fix Vietnamese diacritics, grammar, spelling, punctuation, and capitalization naturally.
3. Preserve the speaker's original voice, style, and complete meaning.
4. Do NOT paraphrase, summarize, or hallucinate facts that were not in the speech.
5. Preserve the exact segment count and keep every segment "id" unchanged.
6. If a phrase is ambiguous and unsupported by OCR or audio context, keep it unchanged rather than guessing.

OUTPUT FORMAT:
Return ONLY a valid JSON array where each object contains:
  - "id": integer (matching the input segment id)
  - "cleaned_text": string (the refined Vietnamese text)

Example:
[
  {"id": 0, "cleaned_text": "Chào mừng quý vị đến với chương trình 60 Giây của Đài Truyền hình Thành phố Hồ Chí Minh."},
  {"id": 1, "cleaned_text": "Tại thành phố Cần Thơ, trong 7 tháng đầu năm 2024 đã xảy ra 24 vụ sạt lở bờ sông."}
]
Output ONLY valid JSON. No conversational preamble, explanation, or markdown backticks outside the JSON array."""


def clean_ocr_text_line(text: str) -> str:
    """Cleans up common OCR noise artifacts (timestamps, UI icons, non-text symbols)."""
    if not text:
        return ""
    t = text.strip()
    # Remove clock/timestamp patterns (e.g. 06:30:11, 06830817, 06.32,11)
    t = re.sub(r"\b\d{1,2}[:.,]\d{2}(?:[:.,]\d{2})?\b", "", t)
    t = re.sub(r"\b06\d{4,}\b", "", t)
    t = re.sub(r"\b\d{4,}\b", "", t)
    # Remove UI artifacts
    t = re.sub(r"\b(SUBSCRIBED|HD|giây|già)\b", "", t, flags=re.IGNORECASE)
    # Keep standard characters, letters, punctuation, and numbers
    t = re.sub(r"[^\w\s\-,:./'%\"()]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()

    # Must contain at least 3 alphabetic letters to filter out lone digits/symbols
    letters = re.findall(r"[a-zA-Zà-ỹÀ-Ỹ]", t)
    if len(letters) < 3:
        return ""
    return t


def extract_scene_ocr_context(
    ocr_data: Dict[str, str],
    min_frame: int,
    max_frame: int,
    min_sec: float,
    max_sec: float,
    fps: float = 30.0,
    pad_sec: float = 2.0
) -> List[str]:
    """
    Extracts, deduplicates, and orders OCR text detected within the temporal bounds of a scene.
    """
    if not ocr_data:
        return []

    matched_texts: List[Tuple[int, str]] = []
    seen_texts: Set[str] = set()
    pad_frames = int(round(pad_sec * fps))

    for f_key, raw_text in ocr_data.items():
        try:
            f_idx = int(f_key.replace("f_", ""))
        except ValueError:
            continue

        in_frame_range = (min_frame - pad_frames) <= f_idx <= (max_frame + pad_frames)
        in_time_range = (min_sec - pad_sec) <= (f_idx / fps) <= (max_sec + pad_sec)

        if in_frame_range or in_time_range:
            cleaned = clean_ocr_text_line(raw_text)
            if cleaned and cleaned.lower() not in seen_texts:
                seen_texts.add(cleaned.lower())
                matched_texts.append((f_idx, cleaned))

    matched_texts.sort(key=lambda x: x[0])
    return [txt for _, txt in matched_texts]


def call_openrouter(
    payload_messages: List[Dict],
    api_key: str,
    model: str = DEFAULT_OPENROUTER_MODEL,
    temperature: float = 0.1,
    max_retries: int = 5,
    timeout: int = 60
) -> Optional[str]:
    """
    Calls OpenRouter chat completion API with robust exponential backoff retry.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/TotallyNotMinh/aic2026",
        "X-Title": "AIC 2026 Vietnamese ASR Refinement"
    }

    body = {
        "model": model,
        "temperature": temperature,
        "messages": payload_messages
    }

    data_bytes = json.dumps(body).encode("utf-8")

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(OPENROUTER_API_URL, data=data_bytes, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as response:
                res_json = json.loads(response.read().decode("utf-8"))
                choices = res_json.get("choices", [])
                if choices:
                    return choices[0].get("message", {}).get("content", "")
        except urllib.error.HTTPError as e:
            err_msg = ""
            try:
                err_msg = e.read().decode("utf-8")
            except Exception:
                pass
            
            # Retry on rate limits or server errors
            if e.code in (429, 500, 502, 503, 504):
                sleep_time = (2 ** attempt) * 2 + 1
                print(f"   ⚠️ HTTP {e.code} (Rate limit/Server). Retrying in {sleep_time}s... (Attempt {attempt+1}/{max_retries})")
                time.sleep(sleep_time)
            else:
                print(f"   ❌ OpenRouter HTTP {e.code} Error: {err_msg or e.reason}")
                return None
        except Exception as e:
            sleep_time = 2 + attempt
            print(f"   ⚠️ Connection error ({e}). Retrying in {sleep_time}s...")
            time.sleep(sleep_time)

    return None


def parse_llm_refined_json(raw_response: str) -> Optional[List[Dict]]:
    """
    Parses JSON array returned by the LLM, handling markdown codeblocks or slight syntax deviations.
    """
    if not raw_response:
        return None

    text = raw_response.strip()

    # Remove markdown codeblock fences if present
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    # 1. Direct JSON parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        elif isinstance(parsed, dict):
            for key in ("segments", "results", "output", "data"):
                if key in parsed and isinstance(parsed[key], list):
                    return parsed[key]
    except json.JSONDecodeError:
        pass

    # 2. Extract JSON array substring via regex
    match = re.search(r"\[\s*\{.*\}\s*\]", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

    # 3. Regex fallback to recover {"id": X, "cleaned_text": "..."} objects
    fallback_items = []
    pattern = re.compile(r'\{\s*"id"\s*:\s*(\d+)\s*,\s*"cleaned_text"\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}')
    for m in pattern.finditer(text):
        try:
            s_id = int(m.group(1))
            c_text = bytes(m.group(2), "utf-8").decode("unicode_escape")
            fallback_items.append({"id": s_id, "cleaned_text": c_text})
        except Exception:
            continue

    return fallback_items if fallback_items else None


def build_scene_prompt_payload(
    ocr_lines: List[str],
    segment_batch: List[Tuple[int, Dict]]
) -> List[Dict]:
    """
    Builds the messages payload containing OCR context and transcript segments for the LLM.
    """
    if ocr_lines:
        ocr_context_str = "\n".join([f"• {line}" for line in ocr_lines])
    else:
        ocr_context_str = "(No on-screen OCR text detected for this scene)"

    batch_input_data = []
    for s_idx, seg in segment_batch:
        raw_text = seg.get("raw_text") or seg.get("text", "")
        batch_input_data.append({
            "id": s_idx,
            "text": str(raw_text).strip()
        })

    user_content = (
        f"[SCENE ON-SCREEN OCR TEXT / BANNERS]\n{ocr_context_str}\n\n"
        f"[SCENE ASR TRANSCRIPT SEGMENTS TO REFINE]\n"
        f"{json.dumps(batch_input_data, ensure_ascii=False, indent=2)}\n\n"
        "Cross-reference ASR text with OCR text to correct mistranscribed names, locations, numbers, "
        "and entities. Output ONLY valid JSON array with 'id' and 'cleaned_text'."
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]


def refine_single_video(
    transcript_file: str,
    ocr_dir: str,
    api_key: str,
    model: str = DEFAULT_OPENROUTER_MODEL,
    max_segments_per_batch: int = DEFAULT_MAX_SEGMENTS_PER_BATCH,
    request_delay: float = DEFAULT_REQUEST_DELAY,
    force: bool = False,
    dry_run: bool = False
) -> Tuple[bool, str, int, int]:
    """
    Refines all transcript segments of a single video scene-by-scene with matching OCR text.
    Returns (success_flag, video_id, total_segments, refined_segments_count).
    """
    video_id = os.path.splitext(os.path.basename(transcript_file))[0]

    # 1. Load ASR Transcript
    try:
        with open(transcript_file, "r", encoding="utf-8") as f:
            segments = json.load(f)
    except Exception as e:
        print(f"❌ [{video_id}] Failed to load transcript JSON: {e}")
        return False, video_id, 0, 0

    if not segments or not isinstance(segments, list):
        return True, video_id, 0, 0

    # Check if already refined (when not forcing)
    # Refined means: at least one segment has cleaned_text != raw_text and cleaned_text is non-empty
    if not force and not dry_run:
        has_distinct_clean = any(
            s.get("cleaned_text") and s.get("raw_text") and s.get("cleaned_text") != s.get("raw_text")
            for s in segments
        )
        if has_distinct_clean:
            return True, video_id, len(segments), 0

    # 2. Load OCR text if available
    ocr_file = os.path.join(ocr_dir, f"{video_id}.json")
    ocr_data: Dict[str, str] = {}
    if os.path.exists(ocr_file):
        try:
            with open(ocr_file, "r", encoding="utf-8") as f:
                ocr_data = json.load(f)
                if not isinstance(ocr_data, dict):
                    ocr_data = {}
        except Exception:
            ocr_data = {}

    # 3. Group segments by scene_id (or chronological chunks)
    scene_groups: Dict[str, List[Tuple[int, Dict]]] = defaultdict(list)
    for idx, seg in enumerate(segments):
        sc_id = seg.get("scene_id") or f"{video_id}_scene_{idx // 20:03d}"
        scene_groups[sc_id].append((idx, seg))

    refined_map: Dict[int, str] = {}

    # 4. Process each scene
    for scene_id, seg_list in scene_groups.items():
        min_frame = min(s.get("start_frame", 0) for _, s in seg_list)
        max_frame = max(s.get("end_frame", 0) for _, s in seg_list)
        min_sec = min(s.get("start_sec", 0.0) for _, s in seg_list)
        max_sec = max(s.get("end_sec", 0.0) for _, s in seg_list)

        # Extract matching OCR lines for this scene
        ocr_lines = extract_scene_ocr_context(
            ocr_data=ocr_data,
            min_frame=min_frame,
            max_frame=max_frame,
            min_sec=min_sec,
            max_sec=max_sec
        )

        # Chunk scene segments if larger than max_segments_per_batch
        for i in range(0, len(seg_list), max_segments_per_batch):
            batch_slice = seg_list[i : i + max_segments_per_batch]
            messages = build_scene_prompt_payload(ocr_lines, batch_slice)

            if dry_run:
                print(f"\n--- [DRY RUN] {video_id} | {scene_id} | Segments: {len(batch_slice)} | OCR lines: {len(ocr_lines)} ---")
                print(messages[1]["content"][:350] + "...\n")
                for s_idx, s in batch_slice:
                    raw = s.get("raw_text") or s.get("text", "")
                    refined_map[s_idx] = raw
                continue

            # API Call
            response_text = call_openrouter(
                payload_messages=messages,
                api_key=api_key,
                model=model
            )

            if not response_text:
                print(f"   ⚠️ [{video_id}] Scene {scene_id} LLM call failed. Keeping raw text.")
                for s_idx, s in batch_slice:
                    raw = s.get("raw_text") or s.get("text", "")
                    refined_map[s_idx] = raw
                continue

            parsed_list = parse_llm_refined_json(response_text)
            if parsed_list:
                for item in parsed_list:
                    s_id = item.get("id")
                    c_text = str(item.get("cleaned_text", "")).strip()
                    if s_id is not None and c_text:
                        refined_map[int(s_id)] = c_text
            else:
                print(f"   ⚠️ [{video_id}] Could not parse JSON for scene {scene_id}. Keeping raw text.")
                for s_idx, s in batch_slice:
                    raw = s.get("raw_text") or s.get("text", "")
                    refined_map[s_idx] = raw

            if request_delay > 0:
                time.sleep(request_delay)

    if dry_run:
        return True, video_id, len(segments), len(refined_map)

    # 5. Reconstruct updated segments: preserve raw_text, set cleaned_text and text
    updated_segments = []
    changes_count = 0

    for idx, seg in enumerate(segments):
        # Guarantee raw_text preserves original unedited transcript
        raw_text = seg.get("raw_text") or seg.get("text", "")
        raw_text = str(raw_text).strip()
        
        cleaned_text = refined_map.get(idx, raw_text).strip()
        if not cleaned_text:
            cleaned_text = raw_text

        if cleaned_text != raw_text:
            changes_count += 1

        updated_seg = {
            "video_id": seg.get("video_id", video_id),
            "scene_id": seg.get("scene_id", f"{video_id}_scene_{idx // 20:03d}"),
            "start_sec": seg.get("start_sec", 0.0),
            "end_sec": seg.get("end_sec", 0.0),
            "start_frame": seg.get("start_frame", 0),
            "end_frame": seg.get("end_frame", 0),
            "raw_text": raw_text,
            "cleaned_text": cleaned_text,
            "text": cleaned_text  # Keep 'text' aligned with cleaned_text for downstream search
        }
        updated_segments.append(updated_seg)

    # 6. Atomic save
    tmp_path = f"{transcript_file}.tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(updated_segments, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, transcript_file)

    return True, video_id, len(segments), changes_count


def refine_all_transcripts_pipeline(
    transcripts_dir: str = "cache/asr_transcripts",
    ocr_dir: str = "cache/ocr_text",
    api_key: Optional[str] = None,
    model: str = DEFAULT_OPENROUTER_MODEL,
    workers: int = DEFAULT_WORKERS,
    max_segments_per_batch: int = DEFAULT_MAX_SEGMENTS_PER_BATCH,
    request_delay: float = DEFAULT_REQUEST_DELAY,
    force: bool = False,
    limit: Optional[int] = None,
    video_filter: Optional[List[str]] = None,
    dry_run: bool = False
):
    """
    Main orchestration pipeline for batch refining all transcripts using OpenRouter DeepSeek.
    """
    api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
    if not dry_run and (not api_key or api_key == "YOUR_OPENROUTER_API_KEY_HERE"):
        print("❌ Error: OPENROUTER_API_KEY is not set.")
        print("   Set it via: export OPENROUTER_API_KEY='sk-or-v1-...'")
        print("   Or pass it via CLI: python scripts/refine_transcripts_openrouter.py --api-key '...'")
        return

    if not os.path.exists(transcripts_dir):
        print(f"❌ Transcripts directory not found: {transcripts_dir}")
        return

    all_files = sorted(glob.glob(os.path.join(transcripts_dir, "*.json")))
    if not all_files:
        print(f"❌ No JSON transcript files found in {transcripts_dir}")
        return

    # Filter specific videos if requested
    if video_filter:
        filter_set = set(video_filter)
        all_files = [f for f in all_files if os.path.splitext(os.path.basename(f))[0] in filter_set]

    if limit and limit > 0:
        all_files = all_files[:limit]

    print("=" * 75)
    print(f"🚀 Starting Vietnamese ASR Scene+OCR Refinement Pipeline")
    print(f"   • Model               : {model}")
    print(f"   • Transcripts Dir     : {transcripts_dir} ({len(all_files)} files to process)")
    print(f"   • OCR Dir             : {ocr_dir}")
    print(f"   • Max Segments/Batch  : {max_segments_per_batch}")
    print(f"   • Concurrency Workers : {workers}")
    print(f"   • Mode                : {'DRY RUN' if dry_run else 'LIVE RUN'}")
    print(f"   • Force Re-refinement : {force}")
    print("=" * 75)

    t0 = time.time()
    processed_count = 0
    total_segments_processed = 0
    total_segments_modified = 0

    if workers <= 1:
        for idx, f_path in enumerate(all_files, 1):
            vid = os.path.splitext(os.path.basename(f_path))[0]
            ok, v_id, n_segs, n_mod = refine_single_video(
                transcript_file=f_path,
                ocr_dir=ocr_dir,
                api_key=api_key,
                model=model,
                max_segments_per_batch=max_segments_per_batch,
                request_delay=request_delay,
                force=force,
                dry_run=dry_run
            )
            if ok:
                processed_count += 1
                total_segments_processed += n_segs
                total_segments_modified += n_mod
                print(f"[{idx}/{len(all_files)}] ✅ {v_id} — {n_segs} segments ({n_mod} edited)")
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    refine_single_video,
                    f_path,
                    ocr_dir,
                    api_key,
                    model,
                    max_segments_per_batch,
                    request_delay,
                    force,
                    dry_run
                ): f_path
                for f_path in all_files
            }

            for idx, fut in enumerate(as_completed(futures), 1):
                f_path = futures[fut]
                vid = os.path.splitext(os.path.basename(f_path))[0]
                try:
                    ok, v_id, n_segs, n_mod = fut.result()
                    if ok:
                        processed_count += 1
                        total_segments_processed += n_segs
                        total_segments_modified += n_mod
                        print(f"[{idx}/{len(all_files)}] ✅ {v_id} — {n_segs} segments ({n_mod} edited)")
                    else:
                        print(f"[{idx}/{len(all_files)}] ⚠️ {v_id} completed with warnings.")
                except Exception as e:
                    print(f"[{idx}/{len(all_files)}] ❌ {vid} failed with error: {e}")

    elapsed = (time.time() - t0) / 60
    print("=" * 75)
    print(f"🎉 Refinement Pipeline Completed in {elapsed:.2f} minutes!")
    print(f"   • Videos Processed  : {processed_count}/{len(all_files)}")
    print(f"   • Total Segments    : {total_segments_processed}")
    print(f"   • Segments Corrected: {total_segments_modified}")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(
        description="Refine Vietnamese ASR transcripts with on-screen OCR text using OpenRouter DeepSeek."
    )
    parser.add_argument(
        "--transcripts-dir",
        type=str,
        default="cache/asr_transcripts",
        help="Path to directory containing ASR transcript JSON files."
    )
    parser.add_argument(
        "--ocr-dir",
        type=str,
        default="cache/ocr_text",
        help="Path to directory containing OCR text JSON files."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL),
        help=f"OpenRouter model identifier (default: '{DEFAULT_OPENROUTER_MODEL}')."
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("OPENROUTER_API_KEY", ""),
        help="OpenRouter API key (or set OPENROUTER_API_KEY env var)."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Number of concurrent worker threads (default: 4)."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_MAX_SEGMENTS_PER_BATCH,
        help="Max segments per scene LLM batch (default: 30)."
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_REQUEST_DELAY,
        help="Delay in seconds between requests per worker (default: 0.5s)."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-refinement even if segments have previously cleaned text."
    )
    parser.add_argument(
        "--video-id",
        type=str,
        default=None,
        help="Comma-separated list of specific video IDs to process (e.g. 'L21_V001,L21_V002')."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit processing to first N video files."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect prompts, scene batches, and OCR associations without calling OpenRouter API."
    )

    args = parser.parse_args()

    v_filter = [v.strip() for v in args.video_id.split(",") if v.strip()] if args.video_id else None

    refine_all_transcripts_pipeline(
        transcripts_dir=args.transcripts_dir,
        ocr_dir=args.ocr_dir,
        api_key=args.api_key,
        model=args.model,
        workers=args.workers,
        max_segments_per_batch=args.batch_size,
        request_delay=args.delay,
        force=args.force,
        limit=args.limit,
        video_filter=v_filter,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
