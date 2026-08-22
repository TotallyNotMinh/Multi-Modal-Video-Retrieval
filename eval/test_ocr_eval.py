import os, sys, glob, json, re
from collections import defaultdict, Counter
import numpy as np

REPO_ROOT = os.getcwd()
sys.path.insert(0, REPO_ROOT)

from scripts.eval_retrieval import SearchEngine, is_ground_truth_hit

engine = SearchEngine()

# Add OCR into BM25
fps = 25.0
ocr_files = sorted(glob.glob("cache/ocr_text/*.json"))
for ocr_f in ocr_files:
    vid = os.path.splitext(os.path.basename(ocr_f))[0]
    try:
        with open(ocr_f, "r", encoding="utf-8") as fp:
            ocr_data = json.load(fp)
    except Exception:
        continue
    if not isinstance(ocr_data, dict):
        continue
    entries = []
    for f_key, raw_text in ocr_data.items():
        cleaned = str(raw_text).strip()
        if not cleaned:
            continue
        try:
            f_idx = int(str(f_key).replace("f_", ""))
            t_sec = f_idx / fps
        except ValueError:
            t_sec = 0.0
        entries.append((t_sec, cleaned))
    entries.sort(key=lambda x: x[0])
    if not entries:
        continue
    cur_text = entries[0][1]
    cur_st = entries[0][0]
    cur_et = entries[0][0] + 3.0
    for t_sec, text in entries[1:]:
        if text == cur_text and t_sec <= (cur_et + 3.0):
            cur_et = max(cur_et, t_sec + 3.0)
        else:
            doc_idx = len(engine.bm25.docs)
            engine.bm25.docs.append({"video_id": vid, "start_sec": cur_st, "end_sec": cur_et, "text": cur_text, "source": "ocr"})
            tokens = engine.bm25._tokenize(cur_text)
            engine.bm25.doc_len.append(len(tokens))
            for term, tf in Counter(tokens).items():
                engine.bm25.inverted_index[term].append((doc_idx, tf))
                engine.bm25.df[term] += 1
            cur_text = text
            cur_st = t_sec
            cur_et = t_sec + 3.0
    doc_idx = len(engine.bm25.docs)
    engine.bm25.docs.append({"video_id": vid, "start_sec": cur_st, "end_sec": cur_et, "text": cur_text, "source": "ocr"})
    tokens = engine.bm25._tokenize(cur_text)
    engine.bm25.doc_len.append(len(tokens))
    for term, tf in Counter(tokens).items():
        engine.bm25.inverted_index[term].append((doc_idx, tf))
        engine.bm25.df[term] += 1

engine.bm25.N = len(engine.bm25.docs)
engine.bm25.avgdl = sum(engine.bm25.doc_len) / engine.bm25.N

q = "slide bài giảng cô Võ Hậu trường THPT Marie Curie"
gt = [{"video_id": "L25_V009", "start_sec": 225.0, "end_sec": 240.0, "keyframe_indices": [102]}]

hits = engine.bm25.search(q, top_k=10)
print("BM25 Search Hits for:", q)
for idx, s in hits:
    doc = engine.bm25.docs[idx]
    v_id = doc["video_id"]
    st = doc["start_sec"]
    et = doc["end_sec"]
    txt = doc["text"]
    src = doc.get("source")
    print(f"  Score {s:.2f} | Vid: {v_id} [{st:.1f}s - {et:.1f}s] source: {src} | Text: {txt[:70]}")

res = engine.search(q, w_dense=0.30, w_asr=0.70, top_k=10)
print("\nEngine Final Top 5 Results:")
for r in res["results"][:5]:
    hit = any(is_ground_truth_hit(r, g) for g in gt)
    r_rank = r["rank"]
    r_vid = r["video_id"]
    r_fidx = r["frame_idx"]
    r_pts = r["pts_time"]
    r_sc = r["score"]
    print(f"  Rank {r_rank}: {r_vid} frame {r_fidx} pts {r_pts} score {r_sc} (HIT: {hit})")
