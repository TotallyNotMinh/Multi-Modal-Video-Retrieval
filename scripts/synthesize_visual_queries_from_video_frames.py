#!/usr/bin/env python3
"""
Synthesize Grounded Visual Benchmark Queries using Gemini 2.5 Flash.
- Samples raw video files directly (not using pregenerated keyframes).
- Extracts frames at random timestamps, center-crops/resizes to 1024x576.
- Enforces SHOT-SNAPPED ground truth bounded by +-1.5s tolerance:
    start_sec = max(shot_prev_pts, sampled_pts - 1.5)
    end_sec = min(shot_next_pts, sampled_pts + 1.5)
- Feeds 1 frame per query to Gemini 2.5 Flash (vision) to generate natural Vietnamese search queries.
- Includes rate-limiting pacing delay and exponential backoff retry on HTTP 503/429.
- Supports incremental appending to existing benchmark files.
"""

import os
import sys
import re
import csv
import glob
import json
import base64
import random
import time
import argparse
import urllib.request
import urllib.error
import cv2
import numpy as np
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

OMNIROUTE_URL = os.environ.get("OMNIROUTE_URL", "http://localhost:20128/v1/chat/completions")

TARGET_WIDTH = 1024
TARGET_HEIGHT = 576
TIME_TOLERANCE_SEC = 1.5  # +- 1.5s

SYSTEM_PROMPT = """Bạn là chuyên gia thẩm định & tạo benchmark tìm kiếm video trực quan (Multi-Modal Video Retrieval Benchmark Creator).
Nhiệm vụ của bạn là nhìn vào hình ảnh khung hình video được cung cấp và tạo ra CÂU TRUY VẤN TÌM KIẾM TỰ NHIÊN bằng tiếng Việt mà một người dùng thực tế sẽ gõ vào công cụ tìm kiếm khi muốn tìm cảnh quay này.

QUY TẮC BẮT BUỘC:
1. Hoàn toàn dựa trên thị giác: Chỉ mô tả những gì nhìn thấy rõ trên bức ảnh (hành động nhân vật, phương tiện giao thông, động vật, phong cảnh, màu sắc trang phục, đồ vật, hoặc chữ/biển hiệu/chyron trên màn hình).
2. Phong cách tìm kiếm tự nhiên: Câu văn ngắn gọn (từ 5 đến 18 từ), súc tích. TUYỆT ĐỐI KHÔNG mở đầu bằng "Trong video", "Hình ảnh cho thấy", "Cảnh quay...", "Bức ảnh chụp...".
3. Tính cụ thể & phân biệt: Nêu rõ các chi tiết nhận diện đặc trưng của khung hình (ví dụ: "người phụ nữ áo vàng đang cắt hoa", "xe cứu thương bật đèn nháy trước cổng bệnh viện", "khung cảnh flycam bãi biển lúc hoàng hôn").

PHÂN LOẠI THỂ LOẠI (CATEGORY):
- `visual_action_scene_grounded`: Hành động, cử chỉ, sự kiện thể thao, cảnh chuyển động, phong cảnh thiên nhiên/đô thị.
- `visual_entity_text_grounded`: Nếu trên khung hình có chữ rõ nét (biển hiệu, tên người, banner thời sự, biển báo đường phố, logo).
- `visual_compositional_objects`: Bố cục cụ thể của đồ vật, màu sắc kết hợp, trang phục hoặc vật thể nổi bật.

ĐỊNH DẠNG ĐẦU RA BẮT BUỘC (Trả về DUY NHẤT một JSON Object hợp lệ):
{
  "query": "<Câu truy vấn tiếng Việt tự nhiên>",
  "category": "visual_action_scene_grounded | visual_entity_text_grounded | visual_compositional_objects",
  "visual_description": "<Mô tả ngắn gọn tiếng Anh về nội dung thị giác chính>",
  "difficulty": "easy | medium | hard"
}
"""


class ShotMapManager:
    """Loads and caches organizer keyframe/shot CSV maps for fast snapping."""
    def __init__(self, map_dir: str = "data/map-keyframes-aic25-b1/map-keyframes"):
        self.map_dir = map_dir
        self.cache: Dict[str, List[Dict[str, float]]] = {}

    def get_shot_bounds(self, video_id: str, pts_sec: float, tolerance: float = TIME_TOLERANCE_SEC) -> Tuple[float, float, List[int]]:
        if video_id not in self.cache:
            csv_path = os.path.join(self.map_dir, f"{video_id}.csv")
            if os.path.exists(csv_path):
                entries = []
                try:
                    with open(csv_path, "r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            entries.append({
                                "frame_idx": int(float(row.get("frame_idx", 0))),
                                "pts_time": float(row.get("pts_time", 0.0))
                            })
                    self.cache[video_id] = sorted(entries, key=lambda x: x["pts_time"])
                except Exception:
                    self.cache[video_id] = []
            else:
                self.cache[video_id] = []

        shots = self.cache.get(video_id, [])
        if not shots:
            return max(0.0, round(pts_sec - tolerance, 2)), round(pts_sec + tolerance, 2), []

        prev_pts = 0.0
        next_pts = shots[-1]["pts_time"]
        matched_kf_indices = []

        for i, s in enumerate(shots):
            s_pts = s["pts_time"]
            if s_pts <= pts_sec:
                prev_pts = s_pts
            if s_pts >= pts_sec and s_pts < next_pts:
                next_pts = s_pts
                break

        gt_start = max(round(prev_pts, 2), round(pts_sec - tolerance, 2))
        gt_end = min(round(next_pts, 2), round(pts_sec + tolerance, 2))
        if gt_end <= gt_start:
            gt_end = round(gt_start + tolerance, 2)

        for s in shots:
            if gt_start <= s["pts_time"] <= gt_end:
                matched_kf_indices.append(s["frame_idx"])

        return gt_start, gt_end, matched_kf_indices


def resize_and_crop_1024x576(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    target_aspect = TARGET_WIDTH / TARGET_HEIGHT
    current_aspect = w / h

    if abs(current_aspect - target_aspect) < 0.02:
        return cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)

    if current_aspect > target_aspect:
        new_w = int(h * target_aspect)
        start_x = (w - new_w) // 2
        cropped = frame[:, start_x : start_x + new_w]
    else:
        new_h = int(w / target_aspect)
        start_y = (h - new_h) // 2
        cropped = frame[start_y : start_y + new_h, :]

    return cv2.resize(cropped, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)


def extract_random_frame_from_video(video_path: str, excluded_windows: List[Tuple[float, float]] = None) -> Optional[Tuple[np.ndarray, float, float]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if total_frames < 30 or fps <= 0:
            return None

        duration = total_frames / fps
        if duration < 3.0:
            return None

        min_sec = max(1.0, duration * 0.10)
        max_sec = min(duration - 1.0, duration * 0.90)

        for _ in range(8):
            rand_sec = random.uniform(min_sec, max_sec)
            # Avoid overlapping with existing generated windows in same video
            if excluded_windows:
                if any(st - 5.0 <= rand_sec <= et + 5.0 for st, et in excluded_windows):
                    continue

            cap.set(cv2.CAP_PROP_POS_MSEC, rand_sec * 1000.0)
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                if np.mean(frame) > 20.0:
                    processed_frame = resize_and_crop_1024x576(frame)
                    return processed_frame, rand_sec, duration
    finally:
        cap.release()

    return None


def encode_frame_to_base64_jpeg(frame: np.ndarray, quality: int = 85) -> str:
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, buffer = cv2.imencode(".jpg", frame, encode_param)
    if not success:
        raise ValueError("Failed to encode frame to JPEG")
    b64_str = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"


def clean_llm_json(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    return raw.strip()


def parse_llm_response(raw_bytes: bytes) -> str:
    text = raw_bytes.decode("utf-8", errors="replace").strip()
    if text.startswith("{"):
        try:
            data = json.loads(text)
            return data["choices"][0]["message"]["content"]
        except Exception:
            pass

    chunks = []
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("data: "):
            payload = line[6:].strip()
            if payload == "[DONE]":
                continue
            try:
                chunk = json.loads(payload)
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                if "content" in delta:
                    chunks.append(delta["content"])
            except Exception:
                pass
    return "".join(chunks)


def query_gemini_vision_with_retry(
    model_name: str,
    base64_image_uri: str,
    max_retries: int = 3,
    base_delay: float = 1.5
) -> Optional[Dict[str, Any]]:
    body = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Hãy quan sát bức ảnh khung hình này và tạo 1 câu truy vấn tìm kiếm tiếng Việt tự nhiên kèm phân loại theo định dạng JSON yêu cầu."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": base64_image_uri
                        }
                    }
                ]
            }
        ],
        "temperature": 0.7
    }

    data = json.dumps(body).encode("utf-8")

    for attempt in range(max_retries):
        req = urllib.request.Request(OMNIROUTE_URL, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                content = parse_llm_response(resp.read())
                parsed = json.loads(clean_llm_json(content))
                if isinstance(parsed, dict) and "query" in parsed:
                    return parsed
        except urllib.error.HTTPError as e:
            if e.code in [503, 429, 502]:
                sleep_time = base_delay * (2 ** attempt) + random.uniform(0.5, 1.5)
                time.sleep(sleep_time)
            else:
                break
        except Exception:
            time.sleep(base_delay)

    return None


def process_single_video_query(
    video_path: str,
    model_name: str,
    shot_manager: ShotMapManager,
    excluded_windows: List[Tuple[float, float]],
    pacing_delay: float
) -> Optional[Dict[str, Any]]:
    # Optional worker pacing delay
    if pacing_delay > 0:
        time.sleep(random.uniform(pacing_delay * 0.7, pacing_delay * 1.3))

    res = extract_random_frame_from_video(video_path, excluded_windows=excluded_windows)
    if not res:
        return None

    frame, pts_sec, duration = res
    video_id = os.path.splitext(os.path.basename(video_path))[0]

    b64_uri = encode_frame_to_base64_jpeg(frame)
    llm_out = query_gemini_vision_with_retry(model_name, b64_uri)
    if not llm_out:
        return None

    query_text = llm_out.get("query", "").strip()
    if len(query_text) < 6:
        return None

    for bad_prefix in ["trong video", "hình ảnh", "đoạn phim", "cảnh quay"]:
        if query_text.lower().startswith(bad_prefix):
            query_text = query_text[len(bad_prefix):].strip(" :,.-")

    # Shot-snapped bounds clamped to +-1.5s
    gt_start, gt_end, kf_indices = shot_manager.get_shot_bounds(video_id, pts_sec, tolerance=TIME_TOLERANCE_SEC)

    return {
        "query": query_text,
        "category": llm_out.get("category", "visual_action_scene_grounded"),
        "visual_description": llm_out.get("visual_description", ""),
        "difficulty": llm_out.get("difficulty", "medium"),
        "frame_specs": {
            "width": TARGET_WIDTH,
            "height": TARGET_HEIGHT,
            "sampled_pts_sec": round(pts_sec, 2),
            "video_duration_sec": round(duration, 2),
            "tolerance_window_sec": TIME_TOLERANCE_SEC,
            "is_shot_snapped": True
        },
        "relevant_segments": [
            {
                "video_id": video_id,
                "start_sec": gt_start,
                "end_sec": gt_end,
                "keyframe_indices": kf_indices
            }
        ]
    }


def main():
    parser = argparse.ArgumentParser(description="Synthesize additional visual benchmark queries with rate-limit pacing")
    parser.add_argument("--num_queries", type=int, default=200, help="Number of NEW queries to synthesize")
    parser.add_argument("--videos_dir", type=str, default="data", help="Directory containing raw MP4 video files")
    parser.add_argument("--model", type=str, default="antigravity/gemini-2.5-flash", help="Model name")
    parser.add_argument("--map_dir", type=str, default="data/map-keyframes-aic25-b1/map-keyframes", help="Keyframe map CSV directory")
    parser.add_argument("--output", type=str, default="eval/visual_benchmark_from_raw_frames_1024x576.jsonl", help="Output JSONL benchmark file")
    parser.add_argument("--workers", type=int, default=3, help="Parallel worker threads (reduced to avoid rate limits)")
    parser.add_argument("--pacing_delay", type=float, default=1.0, help="Pacing delay in seconds between requests")
    args = parser.parse_args()

    all_videos = sorted(glob.glob(f"{args.videos_dir}/**/*.mp4", recursive=True))
    if not all_videos:
        print(f"[!] Error: No MP4 video files found in {args.videos_dir}.", file=sys.stderr)
        sys.exit(1)

    # Load existing queries if appending
    existing_records = []
    seen_video_windows = defaultdict(list)
    start_id_num = 1

    if os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        existing_records.append(rec)
                        vid = rec["relevant_segments"][0]["video_id"]
                        st = rec["relevant_segments"][0]["start_sec"]
                        et = rec["relevant_segments"][0]["end_sec"]
                        seen_video_windows[vid].append((st, et))
                    except Exception:
                        pass

        start_id_num = len(existing_records) + 1
        print(f"[*] Found {len(existing_records)} existing queries in {args.output}. Appending {args.num_queries} new queries (Starting ID: vis_raw_{start_id_num:04d})...")
    else:
        print(f"[*] Creating new dataset at {args.output} with {args.num_queries} queries...")

    print(f"[*] Discovered {len(all_videos)} raw MP4 video files.")
    print(f"[*] Pacing Configuration: Workers={args.workers}, Pacing Delay={args.pacing_delay:.1f}s, Exponential Backoff on 503 enabled")
    print(f"[*] Model: {args.model}")

    shot_manager = ShotMapManager(map_dir=args.map_dir)
    sampled_videos = random.sample(all_videos, min(len(all_videos), args.num_queries * 4))

    new_results = []
    query_counter = start_id_num

    print(f"\n[*] Starting paced extraction & vision synthesis...")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_single_video_query,
                v_path,
                args.model,
                shot_manager,
                seen_video_windows[os.path.splitext(os.path.basename(v_path))[0]],
                args.pacing_delay
            ): v_path
            for v_path in sampled_videos
        }

        for fut in as_completed(futures):
            try:
                record = fut.result()
                if record:
                    record["query_id"] = f"vis_raw_{query_counter:04d}"
                    query_counter += 1
                    new_results.append(record)
                    vid = record["relevant_segments"][0]["video_id"]
                    st = record["relevant_segments"][0]["start_sec"]
                    et = record["relevant_segments"][0]["end_sec"]
                    kfs = record["relevant_segments"][0]["keyframe_indices"]
                    cat = record["category"]
                    print(f"  [+] [{len(new_results)}/{args.num_queries}] ({cat}) \"{record['query']}\" -> {vid} [{st}s - {et}s] (kfs: {kfs})", flush=True)

                    if len(new_results) >= args.num_queries:
                        for f in futures:
                            f.cancel()
                        break
            except Exception as e:
                print(f"  [!] Error processing future: {e}", file=sys.stderr)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    all_final_records = existing_records + new_results[:args.num_queries]

    with open(args.output, "w", encoding="utf-8") as f:
        for item in all_final_records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n[✓] Finished synthesis: Exported total {len(all_final_records)} queries ({len(new_results[:args.num_queries])} new) to {args.output}")


if __name__ == "__main__":
    main()
