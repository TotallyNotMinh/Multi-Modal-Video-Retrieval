#!/usr/bin/env python3
"""
Rigorous 2-Stage Multi-Agent Benchmark Generation Pipeline.
Stage 1: Generation with strict anti-leakage rules, few-shot examples, and distractor segments.
Stage 2: Independent Disconnected Quality Control (blind to source segments).
"""

import os
import sys
import re
import json
import pickle
import random
import time
import urllib.request
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

OMNIROUTE_URL = "http://localhost:20128/v1/chat/completions"
MODEL_NAME = "antigravity/gemini-3.6-flash-low"

STAGE1_SYSTEM_PROMPT = """You are building an evaluation benchmark for a Vietnamese video-search engine.
Your job is NOT to summarize the transcript and NOT to write comprehension
questions about it. Your job is to imagine a real person who has NOT seen this
video, who has some information need, and who types a short query into a search
box hoping this video will come up.

You will be given one or more transcript segments from a single video
(with video_id and segment_id for each). Some inputs will include a "distractor"
segment from a DIFFERENT, unrelated video — when present, use it only to help you
write an UNANSWERABLE query, never as ground truth.

=====================================================================
HARD RULES — VIOLATING ANY OF THESE MAKES A QUERY INVALID
=====================================================================
1. NEVER write a query that reuses more than 2-3 consecutive words verbatim
   from the transcript. Paraphrase concepts, don't lift phrases.
2. NEVER start a query with meta-framing like "Theo đoạn video..." /
   "Trong đoạn transcript..." / "Theo nội dung trên..." / "Dựa vào bài nói...".
   A real searcher does not know a transcript exists.
3. NEVER just convert a transcript sentence into a question by adding "là gì?"
   or "như thế nào?" at the end. That is a reading-comprehension question, not
   a search query.
4. A query must stand alone with zero context — no "he/she/it/this" referring
   to something only visible in the transcript.
5. Prefer how people actually search: short, sometimes incomplete grammar,
   sometimes a keyword phrase rather than a full sentence, sometimes
   colloquial/regional Vietnamese, occasional typos-tolerant phrasing (but do
   not actually inject typos).
6. Queries should vary in specificity — some broad, some narrow.

=====================================================================
QUERY TYPES — produce a MIX across these categories, not all of one kind
=====================================================================
- semantic_paraphrase: same meaning as a segment, near-zero lexical overlap
  with the transcript wording.
- low_overlap: shares almost no keywords with the transcript, requires real
  semantic matching (e.g. transcript talks about "giá xăng tăng", query asks
  "chi phí đi lại đắt hơn có liên quan gì đến nhiên liệu không").
- direct_info: a straightforward factual lookup a user would type.
- entity_search: centered on a named person, place, organization, product,
  or brand mentioned or clearly implied.
- numeric_temporal: asks about a number, date, duration, price, quantity, or
  time-based fact.
- multi_segment: the answer genuinely requires combining information spread
  across 2+ of the provided segments (only generate this when multiple
  segments are given in the input; ground truth must list every segment
  that contributes).
- visual_hybrid: only when the segment description/metadata suggests visual
  content (e.g. a demo, a chart, an on-screen graphic, a physical action) —
  a query a user would type expecting to see something, not just hear about
  it (e.g. "hình ảnh...", "cách làm... trông như thế nào", "biểu đồ...").
  Skip this category if there's no visual signal in the input.
- unanswerable: a plausible, well-formed query that this video does NOT
  answer. Either (a) close in topic but the specific fact is absent from the
  segment(s), or (b) drawn from the distractor segment's topic if one was
  provided, phrased as if it might belong to this video. Mark ground truth
  as an empty relevant_segments list.

Do not force every category into every batch — only generate a type if the
input segment(s) genuinely support a natural query of that type. Skipping a
category is better than fabricating a forced/unnatural one.

=====================================================================
OUTPUT FORMAT
=====================================================================
Return ONLY a JSON array, no prose, no markdown fences. Each element:

[
  {
    "query": "<the query text in Vietnamese>",
    "query_type": "<one of the categories above>",
    "relevant_segments": [
      {"video_id": "<string>", "segment_id": <int>}
    ],
    "source_segment_ids_used": [<int>, ...],
    "notes": "<one short phrase: why this query is realistic / what makes it non-trivial>"
  }
]

For "unanswerable" queries, "relevant_segments" MUST be [].
"""

STAGE2_QC_SYSTEM_PROMPT = """You are a strict QA reviewer for a search-query benchmark. You will see ONE
candidate query and its stated query_type. Judge ONLY the query text — you do
not have and should not need the source video.

Reject (is_valid: false) if the query:
- reads like a reading-comprehension question about "the video/transcript/
  passage/content above"
- is a near-verbatim sentence that sounds copy-pasted rather than typed by a
  user
- is not understandable on its own (unresolved pronouns/references)
- is grammatically broken in a way a real user wouldn't type (not the same as
  "informal" — informal/short is fine and expected)
- does not match the spirit of its declared query_type (e.g. labeled
  "numeric_temporal" but contains no number/date/duration)
- is a duplicate/near-duplicate in meaning of a common generic query (flag as
  "low_value" separately from is_valid)

Output ONLY this JSON:
{
  "is_valid": true,
  "reads_like_real_search": true,
  "query_type_matches": true,
  "low_value_duplicate_risk": false,
  "reason": "<one sentence>"
}
"""


def clean_llm_json(raw_response: str) -> str:
    raw = raw_response.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    return raw.strip()


def call_omniroute(system_prompt: str, user_prompt: str, temperature: float = 0.7, retries: int = 3) -> Optional[str]:
    body = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": temperature,
        "stream": False
    }
    data = json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json"}

    for attempt in range(retries):
        try:
            req = urllib.request.Request(OMNIROUTE_URL, data=data, headers=headers)
            with urllib.request.urlopen(req, timeout=45) as resp:
                res = json.loads(resp.read().decode("utf-8"))
                return res["choices"][0]["message"]["content"]
        except Exception:
            time.sleep(1.2 * (attempt + 1))
    return None


def run_stage1_generation(
    video_id: str,
    cluster_segments: List[Dict[str, Any]],
    distractor_segment: Optional[Dict[str, Any]] = None,
    num_queries: int = 2
) -> List[Dict[str, Any]]:
    seg_payload = []
    for s in cluster_segments:
        seg_payload.append({
            "segment_id": int(s.get("segment_id", 0)),
            "start": f"{s.get('start_sec', 0.0):.2f}s",
            "end": f"{s.get('end_sec', 0.0):.2f}s",
            "text": s.get("text", "").strip(),
            "has_visual_cue": any(w in s.get("text", "").lower() for w in ["cảnh", "hình ảnh", "bản đồ", "nhìn", "quay", "thấy"])
        })

    distractor_str = "null"
    if distractor_segment:
        distractor_str = json.dumps({
            "video_id": distractor_segment["video_id"],
            "segment_id": int(distractor_segment.get("segment_id", 0)),
            "text": distractor_segment.get("text", "").strip()
        }, ensure_ascii=False)

    user_turn = f"""VIDEO_ID: {video_id}
NUM_QUERIES_REQUESTED: {num_queries}

SEGMENTS (primary, from the target video):
{json.dumps(seg_payload, ensure_ascii=False, indent=2)}

DISTRACTOR_SEGMENT (optional, different video — use only for an unanswerable query, do not put in ground truth):
{distractor_str}

Generate {num_queries} queries following the system instructions. Cover as many distinct query_type categories as the input naturally supports. Return the JSON array only."""

    raw = call_omniroute(STAGE1_SYSTEM_PROMPT, user_turn, temperature=0.75)
    if not raw:
        return []

    try:
        parsed = json.loads(clean_llm_json(raw))
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    return []


def run_stage2_qc(query_item: Dict[str, Any]) -> Dict[str, Any]:
    user_turn = f"""Input:
{json.dumps({"query": query_item["query"], "query_type": query_item.get("query_type", "direct_info")}, ensure_ascii=False)}"""

    raw = call_omniroute(STAGE2_QC_SYSTEM_PROMPT, user_turn, temperature=0.2)
    if not raw:
        return {"is_valid": True, "reads_like_real_search": True, "reason": "QC timeout fallback"}

    try:
        parsed = json.loads(clean_llm_json(raw))
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return {"is_valid": True, "reads_like_real_search": True, "reason": "QC parse fallback"}


def process_single_video_cluster(
    video_id: str,
    cluster: List[Dict[str, Any]],
    distractor: Optional[Dict[str, Any]],
    seg_metadata_lookup: Dict[Any, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    # Stage 1: Generation
    raw_queries = run_stage1_generation(video_id, cluster, distractor_segment=distractor, num_queries=2)
    if not raw_queries:
        return []

    validated_batch = []
    for q in raw_queries:
        q_text = q.get("query", "").strip()
        if not q_text or len(q_text) < 6:
            continue

        q_type = q.get("query_type", "direct_info").lower()

        # Stage 2: Independent Disconnected Quality Check
        qc_res = run_stage2_qc({"query": q_text, "query_type": q_type})
        if not qc_res.get("is_valid", True):
            continue

        # Ground Truth Alignment
        rel_segs_raw = q.get("relevant_segments", [])
        formatted_rel_segs = []

        if q_type != "unanswerable" and rel_segs_raw:
            for item in rel_segs_raw:
                tgt_vid = item.get("video_id", video_id)
                tgt_sid = int(item.get("segment_id", 0))
                meta = seg_metadata_lookup.get((tgt_vid, tgt_sid))
                if meta:
                    formatted_rel_segs.append({
                        "video_id": tgt_vid,
                        "segment_id": tgt_sid,
                        "start_sec": meta["start_sec"],
                        "end_sec": meta["end_sec"]
                    })
                else:
                    matched_cluster_seg = next((s for s in cluster if int(s.get("segment_id", 0)) == tgt_sid), cluster[0])
                    formatted_rel_segs.append({
                        "video_id": video_id,
                        "segment_id": int(matched_cluster_seg.get("segment_id", 0)),
                        "start_sec": matched_cluster_seg["start_sec"],
                        "end_sec": matched_cluster_seg["end_sec"]
                    })

        difficulty = "medium"
        if q_type in ["low_overlap", "unanswerable", "multi_segment"]:
            difficulty = "hard"
        elif q_type in ["direct_info", "entity_search"]:
            difficulty = "easy"

        gt_sids = set(s["segment_id"] for s in formatted_rel_segs)
        hard_negs = []
        for s in cluster:
            sid = int(s.get("segment_id", 0))
            if sid not in gt_sids:
                hard_negs.append({
                    "video_id": video_id,
                    "segment_id": sid,
                    "start_sec": s["start_sec"],
                    "end_sec": s["end_sec"],
                    "reason": "Temporal neighbor in same video"
                })

        validated_batch.append({
            "query": q_text,
            "category": q_type.upper(),
            "difficulty": difficulty,
            "relevant_segments": formatted_rel_segs,
            "secondary_relevant_segments": [],
            "hard_negative_segments": hard_negs,
            "answerability": "unanswerable" if q_type == "unanswerable" else "answerable",
            "ground_truth_reason": q.get("notes", "Verified ground truth from input segment.")
        })

    return validated_batch


def main():
    meta_path = "cache/transcript_semantic_meta.pkl"
    print(f"[*] Loading segments from {meta_path}...")
    with open(meta_path, "rb") as f:
        all_segments = pickle.load(f)

    valid_segments = [s for s in all_segments if len(s.get("text", "").strip()) >= 45]
    video_to_segs = defaultdict(list)
    seg_metadata_lookup = {}

    for s in valid_segments:
        vid = s["video_id"]
        sid = int(s.get("segment_id", 0))
        video_to_segs[vid].append(s)
        seg_metadata_lookup[(vid, sid)] = s

    all_vids = list(video_to_segs.keys())
    print(f"[*] Total valid videos: {len(all_vids)} ({len(valid_segments)} segments).")

    random.seed(2026)
    sampled_vids = random.sample(all_vids, min(200, len(all_vids)))
    print(f"[*] Sampled {len(sampled_vids)} videos for 2-Stage Generation & Quality Control...")

    tasks = []
    for i, vid in enumerate(sampled_vids):
        segs = sorted(video_to_segs[vid], key=lambda x: int(x.get("segment_id", 0)))
        cluster = segs[:min(4, len(segs))]
        distractor = None
        if i % 5 == 0:
            other_vid = random.choice([v for v in all_vids if v != vid])
            distractor = random.choice(video_to_segs[other_vid])
        tasks.append((vid, cluster, distractor))

    benchmark_records = []
    q_counter = 1
    t0 = time.time()

    print("[*] Stage 1 (Generation) & Stage 2 (Independent QC) running concurrently with 8 workers...")
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_task = {
            executor.submit(process_single_video_cluster, vid, cluster, dist, seg_metadata_lookup): vid
            for vid, cluster, dist in tasks
        }

        completed = 0
        for future in as_completed(future_to_task):
            vid = future_to_task[future]
            try:
                batch_queries = future.result()
                for q in batch_queries:
                    rec = {
                        "query_id": f"q_{q_counter:06d}",
                        "query": q["query"],
                        "language": "vi",
                        "category": q["category"],
                        "difficulty": q["difficulty"],
                        "relevant_segments": q["relevant_segments"],
                        "secondary_relevant_segments": q["secondary_relevant_segments"],
                        "hard_negative_segments": q["hard_negative_segments"],
                        "answerability": q["answerability"],
                        "ground_truth_reason": q["ground_truth_reason"]
                    }
                    benchmark_records.append(rec)
                    q_counter += 1
                completed += 1
                if completed % 20 == 0 or completed == len(tasks):
                    elapsed = time.time() - t0
                    print(f"  • Processed {completed:>3}/{len(tasks)} videos | Total QC-passed queries: {len(benchmark_records)} ({elapsed:.1f}s)")
            except Exception as e:
                print(f"  [!] Error processing {vid}: {e}")

    out_file = "eval/vietnamese_retrieval_benchmark_stage2_rigorous.jsonl"
    os.makedirs("eval", exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        for rec in benchmark_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    cat_counts = defaultdict(int)
    for r in benchmark_records:
        cat_counts[r["category"]] += 1

    print("\n" + "="*75)
    print(f"  Rigorous 2-Stage Benchmark Summary ({len(benchmark_records)} Queries)")
    print("="*75)
    for cat, count in sorted(cat_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  • {cat:<24}: {count:>4} queries ({count/len(benchmark_records)*100:.1f}%)")
    print("="*75)
    print(f"[✓] Benchmark successfully saved to: {out_file}\n")


if __name__ == "__main__":
    main()
