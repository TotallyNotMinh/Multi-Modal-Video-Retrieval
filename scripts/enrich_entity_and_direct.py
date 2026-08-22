#!/usr/bin/env python3
"""
Targeted Generator to expand ENTITY_SEARCH and DIRECT_INFO sample sizes to n=100 each.
Uses 2-Stage Generation & Independent QC with gemini-3.6-flash-low.
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
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

OMNIROUTE_URL = "http://localhost:20128/v1/chat/completions"
MODEL_NAME = "antigravity/gemini-3.6-flash-low"


def clean_llm_json(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    return raw.strip()


def call_omniroute(sys_p: str, user_p: str, temp: float = 0.7) -> Optional[str]:
    body = {
        "model": MODEL_NAME,
        "messages": [{"role": "system", "content": sys_p}, {"role": "user", "content": user_p}],
        "temperature": temp,
        "stream": False
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(OMNIROUTE_URL, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=40) as resp:
            res = json.loads(resp.read().decode("utf-8"))
            return res["choices"][0]["message"]["content"]
    except Exception:
        return None


def process_entity_direct_cluster(vid: str, segs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    trans_lines = []
    seg_lookup = {}
    for s in segs[:4]:
        sid = int(s.get("segment_id", 0))
        seg_lookup[sid] = s
        st = s.get("start_sec", 0.0)
        et = s.get("end_sec", 0.0)
        txt = s.get("text", "").strip()
        trans_lines.append(f"- Segment {sid} ({st:.1f}s-{et:.1f}s): {txt}")

    user_text = f"""You are generating search queries for a benchmark.
Generate EXACTLY 2 natural search queries for Video '{vid}' belonging to:
1. `entity_search`: Centered on a named person, place, hospital, university, company, brand, channel, or specific program.
2. `direct_info`: Straightforward factual query a user types into a search bar.

INPUT SEGMENTS:
{"\n".join(trans_lines)}

Return ONLY a JSON array:
[
  {{
    "query": "<Vietnamese search query 1>",
    "query_type": "entity_search",
    "relevant_segments": [{{"video_id": "{vid}", "segment_id": {int(segs[0].get('segment_id', 0))}}}],
    "notes": "<explanation>"
  }},
  {{
    "query": "<Vietnamese search query 2>",
    "query_type": "direct_info",
    "relevant_segments": [{{"video_id": "{vid}", "segment_id": {int(segs[-1].get('segment_id', 0))}}}],
    "notes": "<explanation>"
  }}
]"""

    raw_gen = call_omniroute("You generate search queries. Output ONLY valid JSON array.", user_text, temp=0.75)
    if not raw_gen:
        return []

    try:
        parsed = json.loads(clean_llm_json(raw_gen))
        if not isinstance(parsed, list):
            return []
    except Exception:
        return []

    out = []
    for item in parsed:
        q_text = item.get("query", "").strip()
        q_type = item.get("query_type", "direct_info").lower()
        if q_type not in ["entity_search", "direct_info"] or len(q_text) < 8:
            continue

        qc_user = f"Query: \"{q_text}\"\nCategory: \"{q_type}\""
        qc_sys = "You are a QA reviewer for search queries. Output ONLY JSON: {\"is_valid\": true|false, \"reason\": \"<sentence>\"}"
        qc_raw = call_omniroute(qc_sys, qc_user, temp=0.1)
        if qc_raw:
            try:
                qc_obj = json.loads(clean_llm_json(qc_raw))
                if not qc_obj.get("is_valid", True):
                    continue
            except Exception:
                pass

        rel_sids = [r.get("segment_id") for r in item.get("relevant_segments", []) if isinstance(r, dict)]
        if not rel_sids:
            rel_sids = [int(segs[0].get("segment_id", 0))]

        rel_segs = []
        for sid in rel_sids:
            if sid in seg_lookup:
                s = seg_lookup[sid]
                rel_segs.append({
                    "video_id": vid,
                    "segment_id": sid,
                    "start_sec": s["start_sec"],
                    "end_sec": s["end_sec"]
                })

        if not rel_segs:
            continue

        out.append({
            "query": q_text,
            "category": q_type.upper(),
            "difficulty": "easy" if q_type in ["direct_info", "entity_search"] else "medium",
            "relevant_segments": rel_segs,
            "secondary_relevant_segments": [],
            "hard_negative_segments": [],
            "answerability": "answerable",
            "ground_truth_reason": item.get("notes", "Targeted entity/direct sample.")
        })
    return out


def main():
    meta_path = "cache/transcript_semantic_meta.pkl"
    with open(meta_path, "rb") as f:
        all_segments = pickle.load(f)

    video_to_segs = defaultdict(list)
    for s in all_segments:
        if len(s.get("text", "").strip()) >= 50:
            video_to_segs[s["video_id"]].append(s)

    all_vids = list(video_to_segs.keys())
    random.seed(4242)
    selected_vids = random.sample(all_vids, min(80, len(all_vids)))
    print(f"[*] Mining 80 videos to expand ENTITY_SEARCH and DIRECT_INFO to n >= 100...")

    new_records = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_entity_direct_cluster, vid, video_to_segs[vid]): vid for vid in selected_vids}
        for f in as_completed(futures):
            res = f.result()
            new_records.extend(res)

    print(f"[+] Successfully generated {len(new_records)} targeted queries.")

    current_benchmark_path = "eval/vietnamese_retrieval_benchmark_stage2_rigorous.jsonl"
    with open(current_benchmark_path, "r", encoding="utf-8") as f:
        existing_records = [json.loads(line) for line in f if line.strip()]

    combined = existing_records + new_records
    for idx, rec in enumerate(combined, start=1):
        rec["query_id"] = f"q_{idx:06d}"

    out_merged = "eval/vietnamese_retrieval_benchmark_stage2_rigorous.jsonl"
    with open(out_merged, "w", encoding="utf-8") as f:
        for rec in combined:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    total_counts = defaultdict(int)
    for r in combined:
        total_counts[r["category"]] += 1

    print("\n" + "="*70)
    print(f"  FULLY EXPANDED BENCHMARK SUMMARY ({len(combined)} Total Queries)")
    print("="*70)
    for cat, count in sorted(total_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  • {cat:<24}: {count:>4} queries ({count/len(combined)*100:.1f}%)")
    print("="*70)
    print(f"[✓] Saved updated benchmark to {out_merged}\n")


if __name__ == "__main__":
    main()
