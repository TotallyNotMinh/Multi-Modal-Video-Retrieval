"""
Neighboring-Segment Expansion & Temporal Deduplication for Video Q&A Grounding.
Transforms raw ranked candidate segments into coherent, non-overlapping narrative windows.
"""

from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Any, Set


class TranscriptContextExpander:
    """
    Manages temporal expansion (k-1, k, k+1) and temporal deduplication for speech transcripts.
    Ensures LLM receives complete, continuous event context without redundant overlapping snippets.
    """

    def __init__(self, segments_meta: List[Dict[str, Any]]):
        self.segments_meta = segments_meta
        # Lookup: (video_id, segment_id) -> segment_dict
        self.segment_map: Dict[Tuple[str, int], Dict[str, Any]] = {}
        # Lookup: video_id -> list of segment_dicts sorted by segment_id
        self.video_segments: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for seg in segments_meta:
            vid = seg["video_id"]
            sid = int(seg.get("segment_id", 0))
            self.segment_map[(vid, sid)] = seg
            self.video_segments[vid].append(seg)

        for vid in self.video_segments:
            self.video_segments[vid].sort(key=lambda s: (int(s.get("segment_id", 0)), float(s.get("start_sec", 0.0))))

    def expand_and_deduplicate(
        self,
        ranked_candidates: List[Dict[str, Any]],
        neighbor_window: int = 1,
        max_windows: int = 4,
        max_gap_sec: float = 8.0,
        overlap_threshold: float = 0.35
    ) -> List[Dict[str, Any]]:
        """
        Takes ranked candidates (e.g. from BGE reranker), applies:
        1. Temporal deduplication to suppress identical story windows.
        2. Neighboring-segment expansion (k - neighbor_window .. k + neighbor_window).
        3. Contiguous time/frame window stitching and narrative concatenation.

        Returns a list of unified, non-redundant event context windows.
        """
        if not ranked_candidates:
            return []

        expanded_windows: List[Dict[str, Any]] = []
        # video_id -> list of covered (start_sec, end_sec) intervals
        covered_intervals: Dict[str, List[Tuple[float, float]]] = defaultdict(list)

        def is_subsumed(vid: str, st: float, et: float) -> bool:
            for c_st, c_et in covered_intervals[vid]:
                # Overlap duration
                o_st = max(st, c_st)
                o_et = min(et, c_et)
                if o_et > o_st:
                    overlap_len = o_et - o_st
                    target_len = max(1.0, et - st)
                    if (overlap_len / target_len) >= overlap_threshold:
                        return True
            return False

        for cand in ranked_candidates:
            if len(expanded_windows) >= max_windows:
                break

            vid = cand.get("video_id")
            sid = int(cand.get("segment_id", 0))
            st = float(cand.get("start_sec", 0.0))
            et = float(cand.get("end_sec", 0.0))

            if not vid or vid not in self.video_segments:
                continue

            # Check if this candidate is already covered by a previously accepted window
            if is_subsumed(vid, st, et):
                continue

            # Retrieve neighboring segments by chronological list index
            v_segs = self.video_segments.get(vid, [])
            anchor_idx = next((i for i, s in enumerate(v_segs) if int(s.get("segment_id", -1)) == sid), -1)

            if anchor_idx >= 0:
                start_idx = max(0, anchor_idx - neighbor_window)
                end_idx = min(len(v_segs), anchor_idx + neighbor_window + 1)
                gathered_segs = v_segs[start_idx:end_idx]
            else:
                gathered_segs = [cand]

            # Sort gathered segments by start_sec
            gathered_segs.sort(key=lambda s: float(s.get("start_sec", 0.0)))

            # Stitch contiguous segments together (checking max gap)
            stitched_texts: List[str] = []
            min_st = float(gathered_segs[0]["start_sec"])
            max_et = float(gathered_segs[-1]["end_sec"])
            min_frame = int(gathered_segs[0].get("start_frame", 0))
            max_frame = int(gathered_segs[-1].get("end_frame", 0))
            
            last_end = min_st
            for seg in gathered_segs:
                seg_st = float(seg.get("start_sec", 0.0))
                seg_et = float(seg.get("end_sec", 0.0))
                seg_text = seg.get("text", seg.get("refined_text", "")).strip()

                # If large time gap, add transition marker
                if (seg_st - last_end) > max_gap_sec:
                    stitched_texts.append(f"... [{seg_st:.1f}s] ...")
                
                if seg_text:
                    stitched_texts.append(seg_text)
                
                last_end = seg_et

            combined_text = " ".join(stitched_texts)
            # Clean up whitespace
            combined_text = " ".join(combined_text.split())

            rerank_score = float(cand.get("rerank_score", cand.get("score", 0.0)))
            dense_score = float(cand.get("dense_score", 0.0))

            window_obj = {
                "video_id": vid,
                "anchor_segment_id": sid,
                "num_segments": len(gathered_segs),
                "start_sec": round(min_st, 2),
                "end_sec": round(max_et, 2),
                "duration_sec": round(max_et - min_st, 2),
                "start_frame": min_frame,
                "end_frame": max_frame,
                "text": combined_text,
                "rerank_score": rerank_score,
                "dense_score": dense_score
            }

            expanded_windows.append(window_obj)
            covered_intervals[vid].append((min_st, max_et))

        return expanded_windows

    def format_context_for_prompt(self, expanded_windows: List[Dict[str, Any]]) -> str:
        """
        Formats expanded windows into clean, Markdown-formatted grounding context for VLM / LLM prompts.
        """
        if not expanded_windows:
            return ""

        lines = ["[NGỮ CẢNH LỜI THOẠI TOÀN CẢNH (ĐÃ MỞ RỘNG & KHỬ TRÙNG LẶP)]"]
        for idx, win in enumerate(expanded_windows, 1):
            vid = win["video_id"]
            st = win["start_sec"]
            et = win["end_sec"]
            dur = win["duration_sec"]
            num_segs = win["num_segments"]
            score = win["rerank_score"]
            txt = win["text"]

            lines.append(
                f"• Đoạn {idx} | Video [{vid}] từ {st}s - {et}s ({dur}s • {num_segs} segments • Độ khớp: {score:.3f}):\n"
                f"  \"{txt}\""
            )

        return "\n".join(lines)
